"""Minimal production writer for fixed-budget real GEDLIB interval labels."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, TimeoutError
from dataclasses import dataclass
import importlib
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

from src.data.tastemolnet_neurosed_production import (
    LABEL_CONTRACT,
    FixedBudgetPairInventory,
    NeuroSEDProductionDataError,
    atomic_json,
    load_fixed_budget_pair_inventory,
    load_json,
    load_jsonl,
    read_compact_npz,
    sha256_file,
    stable_sha256,
    write_compact_npz,
)
from src.eval.tastemolnet_neurosed_fixed_budget import (
    OFFICIAL_SED_EDIT_COSTS,
    build_official_pair_labels_manifest,
    directional_ged_cache_key,
    official_ged_interval_label,
    select_successful_reserve_pairs,
)
from src.eval.tastemolnet_neurosed_non_mip import (
    validate_non_mip_selection_manifest,
)
from src.utils.tastemolnet_neurosed_gedlib_build import (
    BUILD_SCHEMA,
    GED_LABEL_BACKEND_VARIANT,
    NON_MIP_METHOD_CONFIGS,
    PINNED_GREED_COMMIT,
)


GED_LABEL_MANIFEST_SCHEMA = "tastemolnet_neurosed_fixed_budget_ged_labels_v1"
GED_LABEL_PASS_MARKER = "[TASTE_GED_LABELS_FIXED_BUDGET_PASS]"
MINIMUM_PERSISTENT_FREE_BYTES = 100 * 1024**3


@dataclass(frozen=True, slots=True)
class GEDLabelRuntimeAuthority:
    pyged_module_path: Path
    pyged_module_sha256: str
    gedlib_commit: str
    gedlib_build_manifest_sha256: str
    gedlib_config_sha256: str
    feature_schema_sha256: str
    method: str
    method_args: str
    train_pair_budget: int
    validation_pair_budget: int
    selection_sha256: str
    selection_manifest_file_sha256: str
    selection_verifier_receipt_sha256: str


def _validate_build(build_manifest_path: Path) -> tuple[dict[str, Any], Path]:
    build = load_json(build_manifest_path)
    smoke = build.get("smoke")
    source = build.get("source_authority")
    dependencies = build.get("dependencies")
    if (
        build.get("schema_version") != BUILD_SCHEMA
        or build.get("status") != "PASS"
        or build.get("marker") != "[TASTE_NEUROSED_GEDLIB_BUILD_PASS]"
        or build.get("network_install_performed") is not False
        or build.get("GED_LABEL_BACKEND_VARIANT") != GED_LABEL_BACKEND_VARIANT
        or build.get("F2_BLP_USED") is not False
        or build.get("GUROBI_USED") is not False
        or build.get("ged_label_backend_variant") != GED_LABEL_BACKEND_VARIANT
        or build.get("f2_blp_used") is not False
        or build.get("gurobi_used") is not False
        or build.get("selected_ged_backend") is not None
        or type(smoke) is not dict
        or type(source) is not dict
        or type(dependencies) is not dict
        or source.get("official_greed_commit") != PINNED_GREED_COMMIT
    ):
        raise NeuroSEDProductionDataError("GEDLIB build authority is not PASS")
    module = Path(str(smoke.get("module_path") or "")).resolve()
    if (
        not module.is_file()
        or module.name.split(".", 1)[0] != "pyged"
        or sha256_file(module) != smoke.get("module_sha256")
    ):
        raise NeuroSEDProductionDataError("isolated pyged module changed")
    return build, module


def _validate_selection_receipt(
    receipt_path: Path,
    *,
    selection: Mapping[str, Any],
    selection_manifest_file_sha256: str,
) -> str:
    receipt = load_json(receipt_path)
    claimed = str(receipt.get("receipt_sha256") or "")
    if claimed != stable_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    ):
        raise NeuroSEDProductionDataError("non-MIP verifier receipt hash changed")
    if (
        receipt.get("schema_version")
        != "tastemolnet_neurosed_non_mip_gedlib_verifier_v1"
        or receipt.get("status") != "PASS"
        or receipt.get("marker") != "[TASTE_NON_MIP_GEDLIB_BACKEND_VERIFIED]"
        or receipt.get("independent_process_reopened_all_candidate_artifacts")
        is not True
        or receipt.get("selection_manifest_sha256")
        != selection_manifest_file_sha256
        or receipt.get("selection_sha256") != selection.get("selection_sha256")
        or receipt.get("selected_ged_backend")
        != selection.get("selected_ged_backend")
        or receipt.get("selected_ged_backend_config")
        != selection.get("backend_config")
        or receipt.get("selected_neurosed_train_pair_budget")
        != selection.get("selected_neurosed_train_pair_budget")
        or receipt.get("selected_neurosed_validation_pair_budget")
        != selection.get("selected_neurosed_validation_pair_budget")
    ):
        raise NeuroSEDProductionDataError("non-MIP verifier receipt changed")
    return claimed


def validate_runtime_authority(
    *,
    build_manifest_path: str | Path,
    selection_manifest_path: str | Path,
    selection_verifier_receipt_path: str | Path,
    train_pair_root: str | Path,
    validation_pair_root: str | Path,
) -> tuple[GEDLabelRuntimeAuthority, FixedBudgetPairInventory, FixedBudgetPairInventory]:
    build_path = Path(build_manifest_path).absolute()
    selection_path = Path(selection_manifest_path).absolute()
    receipt_path = Path(selection_verifier_receipt_path).absolute()
    build, module = _validate_build(build_path)
    selection = validate_non_mip_selection_manifest(
        load_json(selection_path), reopen_artifacts=True
    )
    selection_file_sha = sha256_file(selection_path)
    receipt_sha = _validate_selection_receipt(
        receipt_path,
        selection=selection,
        selection_manifest_file_sha256=selection_file_sha,
    )
    method = str(selection["selected_ged_backend"])
    method_args = str(selection["backend_config"])
    if method != "branch" or method_args != NON_MIP_METHOD_CONFIGS["branch"]:
        raise NeuroSEDProductionDataError("frozen branch backend changed")
    train_budget = int(selection["selected_neurosed_train_pair_budget"])
    validation_budget = int(selection["selected_neurosed_validation_pair_budget"])
    if (train_budget, validation_budget) != (5000, 1000):
        raise NeuroSEDProductionDataError("frozen 5000/1000 label budget changed")
    train_inventory = load_fixed_budget_pair_inventory(
        train_pair_root, split="train", requested_pair_count=train_budget
    )
    validation_inventory = load_fixed_budget_pair_inventory(
        validation_pair_root,
        split="validation",
        requested_pair_count=validation_budget,
    )
    feature_sha = str(train_inventory.manifest.get("feature_schema_sha256") or "")
    if feature_sha != validation_inventory.manifest.get("feature_schema_sha256"):
        raise NeuroSEDProductionDataError("train/validation feature schema differs")
    dependencies = build["dependencies"]
    if (
        selection.get("pyged_module_sha256") != sha256_file(module)
        or selection.get("gedlib_commit") != dependencies.get("gedlib_commit")
    ):
        raise NeuroSEDProductionDataError("selected backend binary changed")
    config_sha = stable_sha256(
        {
            "method": method,
            "method_args": [method_args],
            "edit_costs": OFFICIAL_SED_EDIT_COSTS,
            "build_overlay": build["build_overlay"],
        }
    )
    return (
        GEDLabelRuntimeAuthority(
            pyged_module_path=module,
            pyged_module_sha256=sha256_file(module),
            gedlib_commit=str(dependencies["gedlib_commit"]),
            gedlib_build_manifest_sha256=sha256_file(build_path),
            gedlib_config_sha256=config_sha,
            feature_schema_sha256=feature_sha,
            method=method,
            method_args=method_args,
            train_pair_budget=train_budget,
            validation_pair_budget=validation_budget,
            selection_sha256=str(selection["selection_sha256"]),
            selection_manifest_file_sha256=selection_file_sha,
            selection_verifier_receipt_sha256=receipt_sha,
        ),
        train_inventory,
        validation_inventory,
    )


def _solve_one(
    module_dir: str,
    query_data: tuple[list[int], list[tuple[int, int]]],
    target_data: tuple[list[int], list[tuple[int, int]]],
    method: str,
    method_args: str,
) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, module_dir)
    started = time.perf_counter()
    try:
        pyged = importlib.import_module("pyged")
        lower, upper = pyged.sed(
            query_data, target_data, [method], [method_args]
        )
        lower_value = float(lower)
        upper_value = float(upper)
        if (
            not math.isfinite(lower_value)
            or not math.isfinite(upper_value)
            or lower_value < 0
            or lower_value > upper_value
        ):
            raise RuntimeError("pyged returned invalid bounds")
        return {
            "status": "SUCCESS",
            "elapsed_seconds": time.perf_counter() - started,
            "lower_bound": lower_value,
            "upper_bound": upper_value,
            "exact_bound": lower_value == upper_value,
            "error": "",
        }
    except BaseException as exc:
        return {
            "status": "GEDLIB_ERROR",
            "elapsed_seconds": time.perf_counter() - started,
            "lower_bound": None,
            "upper_bound": None,
            "exact_bound": None,
            "error": f"{type(exc).__name__}: {exc}"[:512],
        }
    finally:
        sys.path.pop(0)


def _terminate_executor(executor: ProcessPoolExecutor) -> None:
    processes = list(getattr(executor, "_processes", {}).values())
    for process in processes:
        process.terminate()
    for process in processes:
        process.join(timeout=5)
    executor.shutdown(wait=False, cancel_futures=True)
    if any(process.is_alive() for process in processes):
        raise NeuroSEDProductionDataError("owned GEDLIB worker did not terminate")


def load_success_cache(cache_roots: Sequence[str | Path]) -> dict[str, dict[str, Any]]:
    """Adopt only successful rows from exact compact attempt tables."""

    cache: dict[str, dict[str, Any]] = {}
    for root_value in cache_roots:
        root = Path(root_value).absolute()
        for split in ("train", "validation"):
            path = root / f"{split}_attempts.npz"
            if not path.is_file():
                continue
            for row in read_compact_npz(path):
                if row["status"] != "SUCCESS":
                    continue
                key = str(row["cache_key"])
                prior = cache.get(key)
                if prior is not None and (
                    prior["lower_bound"] != row["lower_bound"]
                    or prior["upper_bound"] != row["upper_bound"]
                ):
                    raise NeuroSEDProductionDataError("successful GED cache conflicts")
                cache[key] = row
    return cache


def load_verified_selection_cache(
    selection_manifest_path: str | Path,
    *,
    authority: GEDLabelRuntimeAuthority,
    train_inventory: FixedBudgetPairInventory,
) -> dict[str, dict[str, Any]]:
    """Adopt deterministic successful canary rows already verified twice."""

    selection = validate_non_mip_selection_manifest(
        load_json(selection_manifest_path), reopen_artifacts=True
    )
    method = str(selection["selected_ged_backend"])
    report = selection["candidate_reports"][method]
    replays = report["replays"]
    if len(replays) != 2 or replays[0]["outcome_sha256"] != replays[1]["outcome_sha256"]:
        raise NeuroSEDProductionDataError("selected canary replays are not identical")
    canary_pairs = load_jsonl(selection["pairs_jsonl_path"])
    canary_ids = [str(row.get("pair_id") or "") for row in canary_pairs]
    frozen_prefix = [pair.pair_id for pair in train_inventory.pairs[: len(canary_ids)]]
    if len(canary_ids) != 100 or canary_ids != frozen_prefix:
        raise NeuroSEDProductionDataError(
            "verified 100-pair canary is not the frozen train-inventory prefix"
        )
    replay_rows = [load_jsonl(replay["observations_path"]) for replay in replays]
    if len(replay_rows[0]) != 100 or len(replay_rows[1]) != 100:
        raise NeuroSEDProductionDataError("selected canary observation count changed")
    cache: dict[str, dict[str, Any]] = {}
    for pair, left, right in zip(train_inventory.pairs, *replay_rows):
        replay_fields = (
            "pair_id",
            "query_graph_id",
            "target_graph_id",
            "status",
            "lower_bound",
            "upper_bound",
            "exact_bound",
            "error",
            "query_canonical_graph_sha256",
            "target_canonical_graph_sha256",
        )
        normalized_left = tuple(left.get(field) for field in replay_fields)
        normalized_right = tuple(right.get(field) for field in replay_fields)
        if normalized_left != normalized_right or left.get("pair_id") != pair.pair_id:
            raise NeuroSEDProductionDataError("selected canary outcomes changed")
        if left.get("status") != "SUCCESS":
            continue
        if (
            left.get("query_canonical_graph_sha256")
            != pair.query.canonical_graph_sha256
            or left.get("target_canonical_graph_sha256")
            != pair.target.canonical_graph_sha256
            or left.get("query_graph_id") != pair.metadata["query_graph_id"]
            or left.get("target_graph_id") != pair.metadata["target_graph_id"]
        ):
            raise NeuroSEDProductionDataError("selected canary graph identity changed")
        key = directional_ged_cache_key(
            query_canonical_graph_sha256=pair.query.canonical_graph_sha256,
            target_canonical_graph_sha256=pair.target.canonical_graph_sha256,
            gedlib_config_sha256=authority.gedlib_config_sha256,
            feature_schema_sha256=authority.feature_schema_sha256,
        )
        cache[key] = {
            "pair_id": pair.pair_id,
            "query_hash": pair.query.canonical_graph_sha256,
            "target_hash": pair.target.canonical_graph_sha256,
            "lower_bound": float(left["lower_bound"]),
            "upper_bound": float(left["upper_bound"]),
            "exact_bound": bool(left.get("exact_bound", False)),
            "backend": authority.method,
            "backend_config_hash": authority.gedlib_config_sha256,
            "verified_selection_cache": True,
        }
    return cache


def _observation(
    pair: Any,
    *,
    authority: GEDLabelRuntimeAuthority,
    result: Mapping[str, Any],
    cache_key: str,
    cache_hit: bool,
) -> dict[str, Any]:
    row = pair.metadata
    return {
        "pair_id": pair.pair_id,
        "query_graph_id": row["query_graph_id"],
        "target_graph_id": row["target_graph_id"],
        "query_split": row["query_split"],
        "target_split": row["target_split"],
        "query_canonical_graph_sha256": row["query_canonical_graph_sha256"],
        "target_canonical_graph_sha256": row["target_canonical_graph_sha256"],
        "status": result["status"],
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "lower_bound": result["lower_bound"],
        "upper_bound": result["upper_bound"],
        "exact_bound": result["exact_bound"],
        "error": str(result.get("error") or ""),
        "cache_key": cache_key,
        "cache_hit": bool(cache_hit),
        "backend": authority.method,
        "backend_config_hash": authority.gedlib_config_sha256,
        "label_contract": LABEL_CONTRACT,
    }


def solve_inventory(
    inventory: FixedBudgetPairInventory,
    *,
    authority: GEDLabelRuntimeAuthority,
    workers: int,
    pair_timeout_seconds: float,
    success_cache: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if workers not in (1, 2):
        raise NeuroSEDProductionDataError("GED label workers must be one or two")
    if not math.isfinite(pair_timeout_seconds) or not 0 < pair_timeout_seconds <= 600:
        raise NeuroSEDProductionDataError("per-pair timeout must be in (0,600]")
    observations: list[dict[str, Any]] = []
    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        for pair in inventory.pairs:
            cache_key = directional_ged_cache_key(
                query_canonical_graph_sha256=pair.query.canonical_graph_sha256,
                target_canonical_graph_sha256=pair.target.canonical_graph_sha256,
                gedlib_config_sha256=authority.gedlib_config_sha256,
                feature_schema_sha256=authority.feature_schema_sha256,
            )
            cached = success_cache.get(cache_key)
            if cached is not None:
                if (
                    cached.get("backend") != authority.method
                    or cached.get("backend_config_hash")
                    != authority.gedlib_config_sha256
                    or cached.get("query_hash")
                    != pair.query.canonical_graph_sha256
                    or cached.get("target_hash")
                    != pair.target.canonical_graph_sha256
                ):
                    raise NeuroSEDProductionDataError("cached GED authority changed")
                result = {
                    "status": "SUCCESS",
                    "elapsed_seconds": 0.0,
                    "lower_bound": cached["lower_bound"],
                    "upper_bound": cached["upper_bound"],
                    "exact_bound": cached["exact_bound"],
                    "error": "",
                }
                observations.append(
                    _observation(
                        pair,
                        authority=authority,
                        result=result,
                        cache_key=cache_key,
                        cache_hit=True,
                    )
                )
                continue
            future = executor.submit(
                _solve_one,
                str(authority.pyged_module_path.parent),
                pair.query.pyged_data(),
                pair.target.pyged_data(),
                authority.method,
                authority.method_args,
            )
            try:
                result = future.result(timeout=pair_timeout_seconds)
            except TimeoutError:
                _terminate_executor(executor)
                executor = ProcessPoolExecutor(max_workers=workers)
                result = {
                    "status": "TIMEOUT",
                    "elapsed_seconds": float(pair_timeout_seconds),
                    "lower_bound": None,
                    "upper_bound": None,
                    "exact_bound": None,
                    "error": "owned GEDLIB worker exceeded per-pair hard wall",
                }
            observations.append(
                _observation(
                    pair,
                    authority=authority,
                    result=result,
                    cache_key=cache_key,
                    cache_hit=False,
                )
            )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return observations


def _compact_row(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": observation["pair_id"],
        "query_graph_id": observation["query_graph_id"],
        "target_graph_id": observation["target_graph_id"],
        "query_hash": observation["query_canonical_graph_sha256"],
        "target_hash": observation["target_canonical_graph_sha256"],
        "split": observation["query_split"],
        "lower_bound": observation["lower_bound"],
        "upper_bound": observation["upper_bound"],
        "exact_bound": bool(observation.get("exact_bound", False)),
        "label_contract": LABEL_CONTRACT,
        "backend": observation["backend"],
        "backend_config_hash": observation["backend_config_hash"],
        "status": observation["status"],
        "elapsed_seconds": observation["elapsed_seconds"],
        "cache_key": observation["cache_key"],
        "cache_hit": observation["cache_hit"],
        "error": observation.get("error", ""),
    }


def select_inventory_labels(
    inventory: FixedBudgetPairInventory,
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select frozen labels, admitting exact-budget inventories without reserve."""

    if len(observations) != len(inventory.pairs):
        raise NeuroSEDProductionDataError("GED observations do not cover inventory")
    if inventory.reserve_available_count:
        return select_successful_reserve_pairs(
            observations, requested_pair_count=inventory.requested_pair_count
        )
    successful_ids = [
        str(row["pair_id"])
        for row in observations
        if row.get("status") == "SUCCESS"
    ]
    timeout_count = sum(row.get("status") == "TIMEOUT" for row in observations)
    error_count = sum(row.get("status") == "GEDLIB_ERROR" for row in observations)
    return {
        "schema_version": "tastemolnet_neurosed_reserve_selection_v1",
        "status": (
            "PASS"
            if len(successful_ids) == inventory.requested_pair_count
            else "BLOCKED_GEDLIB_LABEL_YIELD"
        ),
        "requested_pair_count": inventory.requested_pair_count,
        "reserve_candidate_count": inventory.requested_pair_count,
        "attempted_pair_count": len(observations),
        "successful_pair_count": len(successful_ids),
        "timeout_count": timeout_count,
        "error_count": error_count,
        "reserve_used": 0,
        "selected_pair_ids": successful_ids,
        "selected_in_sampler_order": True,
        "ged_value_based_selection_used": False,
        "approximate_label_fallback_used": False,
    }


def _write_split(
    output_root: Path,
    inventory: FixedBudgetPairInventory,
    observations: Sequence[Mapping[str, Any]],
    *,
    authority: GEDLabelRuntimeAuthority,
) -> tuple[dict[str, Any], dict[str, Any]]:
    split = inventory.split
    attempts_path = output_root / f"{split}_attempts.npz"
    attempts_sha = write_compact_npz(
        attempts_path, [_compact_row(row) for row in observations]
    )
    reserve = select_inventory_labels(inventory, observations)
    if reserve["status"] != "PASS":
        raise NeuroSEDProductionDataError(
            "BLOCKED_GEDLIB_LABEL_YIELD: "
            f"{split} has {reserve['successful_pair_count']}/"
            f"{inventory.requested_pair_count} successful labels and "
            f"reserve_available_count={inventory.reserve_available_count}"
        )
    by_pair_id = {str(row["pair_id"]): row for row in observations}
    selected_observations = [
        by_pair_id[pair_id] for pair_id in reserve["selected_pair_ids"]
    ]
    official_rows = [
        official_ged_interval_label(
            row,
            gedlib_commit=authority.gedlib_commit,
            pyged_module_sha256=authority.pyged_module_sha256,
            gedlib_config_sha256=authority.gedlib_config_sha256,
            feature_schema_sha256=authority.feature_schema_sha256,
            pair_sampler_manifest_sha256=str(
                inventory.manifest["manifest_sha256"]
            ),
            gedlib_build_manifest_sha256=authority.gedlib_build_manifest_sha256,
            ged_method=authority.method,
            ged_method_args=authority.method_args,
        )
        for row in selected_observations
    ]
    compact_selected: list[dict[str, Any]] = []
    for official, observation in zip(official_rows, selected_observations):
        compact_selected.append(
            {
                **_compact_row(observation),
                "lower_bound": official["lower_bound"],
                "upper_bound": official["upper_bound"],
                "exact_bound": official["exact_bound"],
                "cache_key": official["cache_key"],
            }
        )
    labels_path = output_root / f"{split}_labels.npz"
    labels_sha = write_compact_npz(labels_path, compact_selected)
    manifest = build_official_pair_labels_manifest(
        official_rows,
        split=split,
        requested_pair_count=inventory.requested_pair_count,
        reserve_selection=reserve,
        compact_storage_format="numpy_npz",
        compact_labels_sha256=labels_sha,
    )
    manifest.update(
        {
            "compact_labels_path": labels_path.name,
            "compact_attempts_path": attempts_path.name,
            "compact_attempts_sha256": attempts_sha,
            "pair_sampler_manifest_file_sha256": inventory.manifest_file_sha256,
            "pair_sampler_root": str(inventory.root),
            "inventory_pair_count": len(inventory.pairs),
            "reserve_available_count": inventory.reserve_available_count,
            "exact_budget_inventory": inventory.reserve_available_count == 0,
        }
    )
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = stable_sha256(manifest)
    atomic_json(output_root / f"{split}_label_manifest.json", manifest)
    cache_hits = sum(bool(row["cache_hit"]) for row in observations)
    return manifest, {
        "split": split,
        "attempt_count": len(observations),
        "success_count": inventory.requested_pair_count,
        "cache_hit_count": cache_hits,
        "cache_miss_count": len(observations) - cache_hits,
        "timeout_count": reserve["timeout_count"],
        "gedlib_error_count": reserve["error_count"],
        "selected_cache_keys_sha256": stable_sha256(
            [row["cache_key"] for row in compact_selected]
        ),
        "reserve_available_count": inventory.reserve_available_count,
    }


def write_fixed_budget_ged_labels(
    *,
    build_manifest_path: str | Path,
    selection_manifest_path: str | Path,
    selection_verifier_receipt_path: str | Path,
    train_pair_root: str | Path,
    validation_pair_root: str | Path,
    output_root: str | Path,
    workers: int = 1,
    pair_timeout_seconds: float = 300.0,
    cache_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Solve, compact, and close exactly 5000 train plus 1000 validation rows."""

    authority, train_inventory, validation_inventory = validate_runtime_authority(
        build_manifest_path=build_manifest_path,
        selection_manifest_path=selection_manifest_path,
        selection_verifier_receipt_path=selection_verifier_receipt_path,
        train_pair_root=train_pair_root,
        validation_pair_root=validation_pair_root,
    )
    train_ids = {
        str(pair.metadata["query_graph_id"]) for pair in train_inventory.pairs
    } | {str(pair.metadata["target_graph_id"]) for pair in train_inventory.pairs}
    validation_ids = {
        str(pair.metadata["query_graph_id"]) for pair in validation_inventory.pairs
    } | {
        str(pair.metadata["target_graph_id"])
        for pair in validation_inventory.pairs
    }
    if train_ids & validation_ids:
        raise NeuroSEDProductionDataError("train/validation pair graph IDs overlap")
    destination = Path(output_root).absolute()
    if destination.exists():
        if any(destination.iterdir()):
            raise NeuroSEDProductionDataError("GED label output root is not fresh")
    else:
        destination.mkdir(parents=True, mode=0o700)
    persistent_free_before_bytes = shutil.disk_usage(destination).free
    if persistent_free_before_bytes < MINIMUM_PERSISTENT_FREE_BYTES:
        raise NeuroSEDProductionDataError(
            "GED labels cannot start with less than 100 GiB persistent free space"
        )
    success_cache = load_verified_selection_cache(
        selection_manifest_path,
        authority=authority,
        train_inventory=train_inventory,
    )
    selected_canary_cache_count = len(success_cache)
    for key, row in load_success_cache(cache_roots).items():
        prior = success_cache.get(key)
        if prior is not None and (
            prior["lower_bound"] != row["lower_bound"]
            or prior["upper_bound"] != row["upper_bound"]
        ):
            raise NeuroSEDProductionDataError("external cache conflicts with canary")
        success_cache[key] = row
    train_observations = solve_inventory(
        train_inventory,
        authority=authority,
        workers=workers,
        pair_timeout_seconds=pair_timeout_seconds,
        success_cache=success_cache,
    )
    validation_observations = solve_inventory(
        validation_inventory,
        authority=authority,
        workers=workers,
        pair_timeout_seconds=pair_timeout_seconds,
        success_cache=success_cache,
    )
    train_manifest, train_cache = _write_split(
        destination, train_inventory, train_observations, authority=authority
    )
    validation_manifest, validation_cache = _write_split(
        destination,
        validation_inventory,
        validation_observations,
        authority=authority,
    )
    persistent_free_bytes = shutil.disk_usage(destination).free
    if persistent_free_bytes < MINIMUM_PERSISTENT_FREE_BYTES:
        raise NeuroSEDProductionDataError(
            "GED labels would leave less than 100 GiB persistent free space"
        )
    split_isolation = {
        "schema_version": "tastemolnet_neurosed_ged_split_isolation_v1",
        "status": "PASS",
        "opened_payload_splits": ["train", "validation"],
        "train_pair_roles_subset_of_train": True,
        "validation_pair_roles_subset_of_validation": True,
        "train_validation_graph_id_intersection_empty": True,
        "query_graph_id_differs_from_target_graph_id": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "train_graph_ids_sha256": stable_sha256(sorted(train_ids)),
        "validation_graph_ids_sha256": stable_sha256(sorted(validation_ids)),
    }
    split_isolation["manifest_sha256"] = stable_sha256(split_isolation)
    atomic_json(destination / "split_isolation.json", split_isolation)
    cache_manifest = {
        "schema_version": "tastemolnet_neurosed_pair_label_cache_v1",
        "status": "PASS",
        "cache_key_directional": True,
        "reverse_cache_shared": False,
        "backend": authority.method,
        "backend_config_hash": authority.gedlib_config_sha256,
        "feature_schema_sha256": authority.feature_schema_sha256,
        "input_cache_roots": [str(Path(root).absolute()) for root in cache_roots],
        "verified_selection_canary_cache_count": selected_canary_cache_count,
        "verified_selection_canary_is_train_prefix": True,
        "train": train_cache,
        "validation": validation_cache,
    }
    cache_manifest["manifest_sha256"] = stable_sha256(cache_manifest)
    atomic_json(destination / "pair_label_cache_manifest.json", cache_manifest)
    aggregate = {
        "schema_version": GED_LABEL_MANIFEST_SCHEMA,
        "status": "PASS",
        "marker": GED_LABEL_PASS_MARKER,
        "dataset": "tastemolnet",
        "train_success_count": authority.train_pair_budget,
        "validation_success_count": authority.validation_pair_budget,
        "ged_backend": authority.method,
        "ged_method_args": authority.method_args,
        "gedlib_commit": authority.gedlib_commit,
        "gedlib_build_manifest_sha256": (
            authority.gedlib_build_manifest_sha256
        ),
        "gedlib_config_sha256": authority.gedlib_config_sha256,
        "pyged_module_sha256": authority.pyged_module_sha256,
        "feature_schema_sha256": authority.feature_schema_sha256,
        "workers": int(workers),
        "bounded_cpu_worker_policy_pass": workers in (1, 2),
        "cpu_contention_evidence": (
            "bounded_one_or_two_worker_policy_completed_all_fixed_labels"
        ),
        "minimum_persistent_free_bytes": MINIMUM_PERSISTENT_FREE_BYTES,
        "persistent_free_before_label_artifacts_bytes": (
            persistent_free_before_bytes
        ),
        "persistent_free_after_label_artifacts_bytes": persistent_free_bytes,
        "disk_reservation_pass": True,
        "independent_pairs": True,
        "query_target_order": "query_to_target",
        "calibration_loaded": False,
        "test_loaded": False,
        "compact_storage_format": "numpy_npz",
        "train_inventory_pair_count": len(train_inventory.pairs),
        "validation_inventory_pair_count": len(validation_inventory.pairs),
        "train_reserve_available_count": train_inventory.reserve_available_count,
        "validation_reserve_available_count": (
            validation_inventory.reserve_available_count
        ),
        "exact_budget_inventory_without_reserve": (
            train_inventory.reserve_available_count == 0
            and validation_inventory.reserve_available_count == 0
        ),
        "train_label_manifest_sha256": train_manifest["manifest_sha256"],
        "validation_label_manifest_sha256": validation_manifest["manifest_sha256"],
        "train_pair_sampler_manifest_sha256": str(
            train_inventory.manifest["manifest_sha256"]
        ),
        "validation_pair_sampler_manifest_sha256": str(
            validation_inventory.manifest["manifest_sha256"]
        ),
        "split_isolation_sha256": split_isolation["manifest_sha256"],
        "cache_manifest_sha256": cache_manifest["manifest_sha256"],
        "non_mip_selection_sha256": authority.selection_sha256,
        "non_mip_selection_manifest_file_sha256": (
            authority.selection_manifest_file_sha256
        ),
        "non_mip_selection_verifier_receipt_sha256": (
            authority.selection_verifier_receipt_sha256
        ),
        "F2_BLP_USED": False,
        "GUROBI_USED": False,
    }
    aggregate["manifest_sha256"] = stable_sha256(aggregate)
    atomic_json(destination / "ged_label_manifest.json", aggregate)
    checksums = []
    for path in sorted(destination.iterdir()):
        if path.is_file() and path.name not in {"ged_label_sha256s.txt", "PASS"}:
            checksums.append(f"{sha256_file(path)}  {path.name}")
    checksum_data = ("\n".join(checksums) + "\n").encode("utf-8")
    descriptor = os.open(
        destination / "ged_label_sha256s.txt",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        os.write(descriptor, checksum_data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    descriptor = os.open(
        destination / "PASS", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        os.write(descriptor, (GED_LABEL_PASS_MARKER + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return aggregate


__all__ = [
    "GED_LABEL_MANIFEST_SCHEMA",
    "GED_LABEL_PASS_MARKER",
    "GEDLabelRuntimeAuthority",
    "load_success_cache",
    "load_verified_selection_cache",
    "select_inventory_labels",
    "solve_inventory",
    "validate_runtime_authority",
    "write_fixed_budget_ged_labels",
]
