"""Frozen full-graph WNode metrics and final-artifact export utilities.

This module is intentionally post-processing only.  It reads an existing
Cartesian parent/candidate evaluation and never calls a teacher, selector, or
distance provider.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.eval.flip_semantics import (
    OLD_WEAK_FLIP_DEFINITION,
    TEACHER_STRICT_FLIP_DEFINITION,
    old_weak_flip,
)
from src.eval.mutagenicity_wnode_selector import (
    ThresholdLevel,
    build_candidate_chemistry,
    build_coverage_redundancy_matrix,
)


FLOAT_TOLERANCE = 1e-12
OFFICIAL_FIELDS = (
    "num_parents",
    "num_candidates",
    "num_valid_pairs",
    "num_close_only_covered",
    "close_only_coverage",
    "num_close_cf_covered",
    "close_cf_coverage",
    "avg_best_distance",
    "median_best_distance",
    "avg_cf_drop_among_covered",
    "flip_rate_among_covered",
)
TABLE_REQUIRED_FIELDS = (
    "method",
    "dataset",
    "source_label",
    "target_label",
    "k",
    "theta",
    "ccrcov",
    "applicable_coverage",
    "any_strict_flip_coverage",
    "flip_rate_among_covered",
    "avg_cf_drop_among_covered",
    "conditional_mean_cost",
    "conditional_median_cost",
    "fixed_capped_mean_cost",
    "fixed_capped_median_cost",
    "coverage_redundancy",
    "structural_redundancy",
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _text(value).lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_distance(row: dict[str, Any]) -> float | None:
    for field in ("distance", "wnode_distance"):
        value = _as_float(row.get(field))
        if value is not None:
            return value
    return None


def _mean(values: Iterable[Any]) -> float | None:
    clean = [number for value in values if (number := _as_float(value)) is not None]
    return float(sum(clean) / len(clean)) if clean else None


def _median(values: Iterable[Any]) -> float | None:
    clean = sorted(
        number for value in values if (number := _as_float(value)) is not None
    )
    if not clean:
        return None
    middle = len(clean) // 2
    if len(clean) % 2:
        return float(clean[middle])
    return float((clean[middle - 1] + clean[middle]) / 2.0)


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _strict_flip(
    row: dict[str, Any],
    *,
    source_label: int | None = None,
    target_label: int | None = None,
) -> bool:
    label = source_label
    if label is None:
        label = _as_int(row.get("label"))
    before = _as_int(row.get("pred_before"))
    after = _as_int(row.get("pred_after"))
    if label is not None and before is not None and after is not None:
        if target_label is not None:
            return before == int(label) and after == int(target_label)
        return before == int(label) and after != int(label)
    return _as_bool(row.get("teacher_strict_flip") or row.get("cf_flip"))


def _applicable(row: dict[str, Any]) -> bool:
    if "applicable" in row and row.get("applicable") not in (None, ""):
        return _as_bool(row.get("applicable"))
    if "match" in row or "delete_valid" in row:
        return _as_bool(row.get("match")) and _as_bool(row.get("delete_valid"))
    return _finite_distance(row) is not None


def _best_strict_row(
    rows: Sequence[dict[str, Any]],
    *,
    threshold: float | None,
    source_label: int | None,
    target_label: int | None,
) -> dict[str, Any] | None:
    best: tuple[float, int, dict[str, Any]] | None = None
    for position, row in enumerate(rows):
        distance = _finite_distance(row)
        if distance is None:
            continue
        if threshold is not None and distance > float(threshold):
            continue
        if not _strict_flip(
            row,
            source_label=source_label,
            target_label=target_label,
        ):
            continue
        key = (distance, position, row)
        if best is None or key[:2] < best[:2]:
            best = key
    return best[2] if best is not None else None


def summarize_wnode_thresholds(
    *,
    method: str,
    details: Sequence[dict[str, Any]],
    threshold_rows: Sequence[dict[str, Any]],
    total_parents: int,
    total_candidates: int,
    source_label: int | None = None,
    target_label: int | None = None,
    feature_cost: str = "cosine",
    node_mass: str = "uniform",
    size_penalty_beta: float = 0.0,
    cf_mode: str = "strict_flip",
    cache_hit_rate: float = 0.0,
    node_embedding_cache_hit_rate: float = 0.0,
    skip_redundancy: bool = True,
    group_audit: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Production WNode threshold aggregation shared by evaluator/exporter."""

    if cf_mode != "strict_flip":
        raise ValueError("summarize_wnode_thresholds requires cf_mode=strict_flip.")
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in details:
        parent_id = _text(row.get("parent_id"))
        if not parent_id:
            raise ValueError("A detail row has an empty parent_id.")
        by_parent.setdefault(parent_id, []).append(dict(row))
    teacher_target = {
        parent_id
        for parent_id, rows in by_parent.items()
        if any(
            _as_int(row.get("pred_before"))
            == (
                int(source_label)
                if source_label is not None
                else _as_int(row.get("label"))
            )
            for row in rows
        )
    }
    valid_pairs = sum(_finite_distance(row) is not None for row in details)
    audit = dict(group_audit or {})
    output: list[dict[str, Any]] = []
    for threshold_row in threshold_rows:
        threshold = float(threshold_row["threshold"])
        if not math.isfinite(threshold):
            continue
        close_only = {
            parent_id
            for parent_id, rows in by_parent.items()
            if any(
                (distance := _finite_distance(row)) is not None
                and distance <= threshold
                for row in rows
            )
        }
        best_rows = [
            best
            for rows in by_parent.values()
            if (
                best := _best_strict_row(
                    rows,
                    threshold=threshold,
                    source_label=source_label,
                    target_label=target_label,
                )
            )
            is not None
        ]
        close_cf = {_text(row.get("parent_id")) for row in best_rows}
        weak = {
            parent_id
            for parent_id, rows in by_parent.items()
            if any(
                (distance := _finite_distance(row)) is not None
                and distance <= threshold
                and old_weak_flip(
                    row.get("pred_after"),
                    int(
                        source_label
                        if source_label is not None
                        else (_as_int(row.get("label")) or 0)
                    ),
                )
                for row in rows
            )
        }
        output.append(
            {
                "method": method,
                "distance_type": "node_wasserstein",
                "distance_line": "MolCLR-Node-Wasserstein",
                "feature_cost": feature_cost,
                "node_mass": node_mass,
                "size_penalty_beta": float(size_penalty_beta),
                "solver": "exact_emd2",
                "threshold": threshold,
                "threshold_source": threshold_row.get("threshold_source"),
                "quantile": threshold_row.get("quantile"),
                "cf_mode": cf_mode,
                "main_ccrcov_uses": "teacher_strict_flip",
                "teacher_strict_flip_definition": TEACHER_STRICT_FLIP_DEFINITION,
                "old_weak_flip_definition": OLD_WEAK_FLIP_DEFINITION,
                "old_weak_ccrcov_status": "audit_only",
                "num_parents": int(total_parents),
                "num_teacher_target_parents": len(teacher_target),
                "num_candidates": int(total_candidates),
                "num_valid_pairs": int(valid_pairs),
                "num_close_only_covered": len(close_only),
                "close_only_coverage": _rate(len(close_only), total_parents),
                "num_close_cf_covered": len(close_cf),
                "close_cf_coverage": _rate(len(close_cf), total_parents),
                "old_weak_num_close_cf_covered": len(weak),
                "old_weak_close_cf_coverage": _rate(len(weak), total_parents),
                "avg_best_distance": _mean(
                    _finite_distance(row) for row in best_rows
                ),
                "median_best_distance": _median(
                    _finite_distance(row) for row in best_rows
                ),
                "avg_cf_drop_among_covered": _mean(
                    row.get("cf_drop") for row in best_rows
                ),
                "flip_rate_among_covered": _mean(
                    1.0
                    if _strict_flip(
                        row,
                        source_label=source_label,
                        target_label=target_label,
                    )
                    else 0.0
                    for row in best_rows
                ),
                "total_pairs": len(details),
                "cache_hit_rate": float(cache_hit_rate),
                "node_embedding_cache_hit_rate": float(
                    node_embedding_cache_hit_rate
                ),
                "skip_redundancy": bool(skip_redundancy),
                **audit,
            }
        )
    return output


def read_csv(path: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {source}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected JSON object at {source}:{line_number}"
                )
            rows.append(dict(value))
    return rows


def stable_json_sha256(payload: Any) -> str:
    """Return the stable JSON identity used by frozen GCF candidate exports."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_smiles(row: dict[str, Any]) -> str:
    for field in (
        "canonical_smiles",
        "candidate_smiles",
        "smiles",
        "graph_smiles",
        "cf_smiles",
        "final_smiles",
    ):
        value = _text(row.get(field))
        if value:
            return value
    return ""


def load_ranked_candidates(
    path: str | Path,
    *,
    expected_count: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows, fields = read_csv(path)
    has_prefix_rank = "rank" in fields
    has_native_rank = "native_rank" in fields
    if not has_prefix_rank and not has_native_rank:
        raise ValueError(
            "Frozen candidate CSV requires rank or native_rank: " f"{path}"
        )
    ranked: list[tuple[int, dict[str, Any]]] = []
    native_ranks: list[int] = []
    for row_index, row in enumerate(rows, start=1):
        rank = _as_int(row.get("rank")) if has_prefix_rank else row_index
        native_rank = _as_int(row.get("native_rank"))
        candidate_id = _text(row.get("candidate_id"))
        smiles = _candidate_smiles(row)
        if rank is None or rank <= 0 or not candidate_id or not smiles:
            raise ValueError(f"Invalid frozen candidate row: {row}")
        if has_native_rank:
            if native_rank is None or native_rank <= 0:
                raise ValueError(f"Invalid native_rank in frozen candidate row: {row}")
            native_ranks.append(native_rank)
        normalized = dict(row)
        normalized["rank"] = rank
        normalized["candidate_id"] = candidate_id
        normalized["candidate_smiles"] = smiles
        ranked.append((rank, normalized))
    if has_prefix_rank:
        ranked.sort(key=lambda item: item[0])
    elif native_ranks != sorted(set(native_ranks)):
        raise ValueError(
            "Frozen native_rank values must be unique and strictly increasing "
            f"in CSV row order: {path}"
        )
    ordered = [row for _, row in ranked]
    expected_ranks = list(range(1, int(expected_count) + 1))
    if [rank for rank, _ in ranked] != expected_ranks:
        raise ValueError(
            f"Frozen candidate ranks must be 1..{expected_count}: "
            f"{[rank for rank, _ in ranked]}"
        )
    ids = [str(row["candidate_id"]) for row in ordered]
    smiles = [str(row["candidate_smiles"]) for row in ordered]
    if len(set(ids)) != len(ids) or len(set(smiles)) != len(smiles):
        raise ValueError("Frozen candidates contain duplicate IDs or SMILES.")
    return ordered, fields


def _manifest_value(payload: Mapping[str, Any], names: set[str]) -> Any:
    value = _deep_find(dict(payload), names)
    if value in (None, ""):
        raise ValueError(
            "Frozen candidate manifest is missing one of: "
            f"{sorted(names)}"
        )
    return value


def _validated_sha256(value: Any, *, description: str) -> str:
    digest = _text(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in digest
    ):
        raise ValueError(
            f"{description} must be a 64-character hexadecimal SHA256."
        )
    return digest.lower()


def _select_manifest_file_entry(
    entries: Sequence[tuple[str, Any]],
    *,
    candidate_relative_path: str | None,
    container_name: str,
) -> tuple[str, Any] | None:
    if candidate_relative_path is not None:
        exact = [entry for entry in entries if entry[0] == candidate_relative_path]
        if len(exact) > 1:
            raise ValueError(
                f"Frozen candidate manifest has duplicate {container_name} entries "
                f"for {candidate_relative_path}."
            )
        if exact:
            return exact[0]

    basename_matches = [
        entry
        for entry in entries
        if PurePosixPath(entry[0]).name == "selected_top20.csv"
    ]
    if len(basename_matches) > 1:
        paths = sorted(path for path, _ in basename_matches)
        raise ValueError(
            "Frozen candidate manifest has ambiguous selected_top20.csv "
            f"entries in {container_name}: {paths}"
        )
    return basename_matches[0] if basename_matches else None


def _manifest_candidate_csv_sha256(
    payload: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    candidate_path: str | Path,
) -> str:
    resolved_manifest = Path(manifest_path).expanduser().resolve()
    resolved_candidate = Path(candidate_path).expanduser().resolve()
    try:
        candidate_relative_path = resolved_candidate.relative_to(
            resolved_manifest.parent
        ).as_posix()
    except ValueError:
        candidate_relative_path = None

    schema_version = _text(payload.get("schema_version"))
    inventory_present = "file_inventory" in payload
    if schema_version == "mut_gcf_frozen_top20_v1" and not inventory_present:
        raise ValueError(
            "mut_gcf_frozen_top20_v1 manifest requires file_inventory."
        )
    if inventory_present:
        inventory = payload.get("file_inventory")
        if not isinstance(inventory, dict):
            raise ValueError("Frozen manifest file_inventory must be a mapping.")
        selected = _select_manifest_file_entry(
            [(str(path), entry) for path, entry in inventory.items()],
            candidate_relative_path=candidate_relative_path,
            container_name="file_inventory",
        )
        if selected is not None:
            relative_path, entry = selected
            if not isinstance(entry, dict):
                raise ValueError(
                    "Frozen manifest file_inventory entry must be a mapping: "
                    f"{relative_path}"
                )
            if entry.get("sha256") in (None, ""):
                raise ValueError(
                    "Frozen manifest file_inventory entry is missing sha256: "
                    f"{relative_path}"
                )
            return _validated_sha256(
                entry.get("sha256"),
                description=f"file_inventory[{relative_path!r}].sha256",
            )
        if schema_version == "mut_gcf_frozen_top20_v1":
            raise ValueError(
                "Frozen candidate manifest file_inventory does not identify "
                f"{candidate_relative_path or 'selected_top20.csv'}."
            )

    legacy_hashes: list[tuple[str, str]] = []
    direct = _deep_find(
        dict(payload),
        {
            "candidate_csv_sha256",
            "selected_top20_csv_sha256",
            "selected_candidates_sha256",
        },
    )
    if direct not in (None, ""):
        legacy_hashes.append(
            (
                "legacy direct field",
                _validated_sha256(
                    direct,
                    description="Frozen candidate manifest direct SHA256",
                ),
            )
        )
    for container_name in ("artifacts", "files", "file_sha256"):
        container = payload.get(container_name)
        if isinstance(container, dict):
            entries = [(str(relative), value) for relative, value in container.items()]
        elif isinstance(container, list):
            entries = [
                (
                    str(item.get("path") or item.get("relative_path") or ""),
                    item,
                )
                for item in container
                if isinstance(item, dict)
            ]
        else:
            continue
        selected = _select_manifest_file_entry(
            entries,
            candidate_relative_path=candidate_relative_path,
            container_name=container_name,
        )
        if selected is None:
            continue
        relative_path, value = selected
        if isinstance(value, dict):
            value = value.get("sha256") or value.get("hash")
        if value in (None, ""):
            raise ValueError(
                f"Frozen manifest {container_name} entry is missing SHA256: "
                f"{relative_path}"
            )
        legacy_hashes.append(
            (
                f"{container_name}[{relative_path!r}]",
                _validated_sha256(
                    value,
                    description=(
                        f"Frozen candidate manifest {container_name} SHA256"
                    ),
                ),
            )
        )
    if legacy_hashes:
        unique_hashes = {digest for _, digest in legacy_hashes}
        if len(unique_hashes) != 1:
            raise ValueError(
                "Frozen candidate legacy manifest contains conflicting "
                f"selected_top20.csv SHA256 values: {legacy_hashes}"
            )
        return legacy_hashes[0][1]
    raise ValueError(
        "Frozen candidate manifest does not identify selected_top20.csv SHA256."
    )


def _manifest_candidate_count(payload: Mapping[str, Any]) -> int | None:
    for name in (
        "candidate_count",
        "selected_candidate_count",
        "selected_top20_rows",
        "selected_count",
    ):
        if name in payload:
            return _as_int(payload.get(name))
    return _as_int(_deep_find(dict(payload), {"candidate_count"}))


def validate_frozen_candidate_contract(
    *,
    candidates_csv: str | Path,
    frozen_manifest_path: str | Path,
    expected_count: int,
    expected_csv_sha256: str,
    expected_order_sha256: str,
    expected_native_ranks: Sequence[int],
    expected_selection_method: str,
) -> dict[str, Any]:
    """Validate a frozen fullgraph CSV without converting or reordering it."""

    candidate_path = Path(candidates_csv).expanduser().resolve()
    manifest_path = Path(frozen_manifest_path).expanduser().resolve()
    candidates, fields = load_ranked_candidates(
        candidate_path,
        expected_count=int(expected_count),
    )
    if "native_rank" not in fields:
        raise ValueError("GCF frozen candidates must retain native_rank.")
    native_ranks = [int(row["native_rank"]) for row in candidates]
    required_native_ranks = [int(value) for value in expected_native_ranks]
    if native_ranks != required_native_ranks:
        raise ValueError(
            "Frozen native candidate order mismatch: "
            f"actual={native_ranks}, expected={required_native_ranks}"
        )
    if not all(_as_bool(row.get("candidate_set_preselected")) for row in candidates):
        raise ValueError("Every frozen candidate must declare candidate_set_preselected=true.")
    if any(_as_bool(row.get("selection_performed_in_eval")) for row in candidates):
        raise ValueError("Frozen candidates declare selection_performed_in_eval=true.")
    if any(not _as_bool(row.get("rdkit_valid")) for row in candidates):
        raise ValueError("Frozen candidates contain rdkit_valid=false rows.")
    if any(_as_int(row.get("rf_pred")) != 0 for row in candidates):
        raise ValueError("Frozen candidates contain a non-target RF prediction.")
    selection_methods = {
        _text(row.get("selection_method")) for row in candidates
    }
    if selection_methods != {str(expected_selection_method)}:
        raise ValueError(
            "Frozen candidate selection_method mismatch: "
            f"actual={sorted(selection_methods)}"
        )
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    order_sha256 = stable_json_sha256(candidate_ids)
    csv_sha256 = sha256_file(candidate_path)
    normalized_expected_csv_sha256 = _validated_sha256(
        expected_csv_sha256,
        description="Expected frozen candidate CSV SHA256",
    )
    normalized_expected_order_sha256 = _validated_sha256(
        expected_order_sha256,
        description="Expected frozen candidate order SHA256",
    )
    if csv_sha256 != normalized_expected_csv_sha256:
        raise ValueError(
            f"Frozen candidate CSV SHA256 mismatch: actual={csv_sha256}"
        )
    if order_sha256 != normalized_expected_order_sha256:
        raise ValueError(
            f"Frozen candidate order SHA256 mismatch: actual={order_sha256}"
        )

    manifest = read_json(manifest_path)
    manifest_csv_sha = _manifest_candidate_csv_sha256(
        manifest,
        manifest_path=manifest_path,
        candidate_path=candidate_path,
    )
    manifest_order_sha = _validated_sha256(
        _manifest_value(manifest, {"selected_candidate_order_sha256"}),
        description="Frozen manifest selected candidate order SHA256",
    )
    manifest_native_ranks = _manifest_value(manifest, {"selected_native_ranks"})
    if not isinstance(manifest_native_ranks, list):
        raise ValueError("Manifest selected_native_ranks must be a list.")
    if manifest_csv_sha != csv_sha256:
        raise ValueError("Frozen manifest candidate CSV SHA256 mismatch.")
    if manifest_order_sha != order_sha256:
        raise ValueError("Frozen manifest candidate order SHA256 mismatch.")
    if [int(value) for value in manifest_native_ranks] != native_ranks:
        raise ValueError("Frozen manifest native rank order mismatch.")

    semantic_checks = {
        "dataset": _text(_manifest_value(manifest, {"dataset"})).lower()
        == "mutagenicity",
        "source_label": _as_int(_manifest_value(manifest, {"source_label"})) == 1,
        "target_label": _as_int(_manifest_value(manifest, {"target_label"})) == 0,
        "candidate_count": _manifest_candidate_count(manifest)
        == int(expected_count),
        "candidate_set_preselected": _as_bool(
            _manifest_value(manifest, {"candidate_set_preselected"})
        ),
        "selection_performed_in_eval": not _as_bool(
            _manifest_value(manifest, {"selection_performed_in_eval"})
        ),
        "rf_reranking_performed": not _as_bool(
            _manifest_value(manifest, {"rf_reranking_performed"})
        ),
        "wnode_reranking_performed": not _as_bool(
            _manifest_value(manifest, {"wnode_reranking_performed"})
        ),
        "selection_method": _text(
            _manifest_value(manifest, {"selection_method"})
        )
        == str(expected_selection_method),
    }
    failed = [name for name, passed in semantic_checks.items() if not passed]
    if failed:
        raise ValueError(f"Frozen candidate manifest semantic mismatch: {failed}")
    return {
        "candidate_count": len(candidates),
        "candidate_ids": candidate_ids,
        "native_ranks": native_ranks,
        "candidate_csv": str(candidate_path),
        "candidate_csv_sha256": csv_sha256,
        "frozen_manifest": str(manifest_path),
        "frozen_manifest_sha256": sha256_file(manifest_path),
        "selected_candidate_order_sha256": order_sha256,
        "candidate_set_preselected": True,
        "selection_performed_in_eval": False,
        "selection_method": str(expected_selection_method),
        "row_order_preserved": True,
        "adapter_used": False,
        "semantic_checks": semantic_checks,
    }


def locate_test_inputs(test_run_dir: str | Path) -> tuple[Path, Path, Path]:
    root = Path(test_run_dir).expanduser().resolve()
    pair_candidates = (
        root / "details" / "pair_details.csv",
        root / "pair_details.csv",
        root / "test_pair_details.csv",
    )
    summary_candidates = (
        root / "combined" / "combined_threshold_summary.csv",
        root / "combined_threshold_summary.csv",
        root / "test_threshold_summary.csv",
    )
    config_candidates = (root / "run_config.json", root / "run_manifest.json")
    pair = next((path for path in pair_candidates if path.is_file()), None)
    summary = next((path for path in summary_candidates if path.is_file()), None)
    config = next((path for path in config_candidates if path.is_file()), None)
    if pair is None or summary is None or config is None:
        raise FileNotFoundError(
            f"Missing pair details, combined summary, or run config under {root}."
        )
    return pair, summary, config


def validate_complete_cartesian(
    details: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    *,
    expected_parent_count: int,
    expected_pair_count: int,
) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    candidate_set = set(candidate_ids)
    parent_order: list[str] = []
    seen_parents: set[str] = set()
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for row in details:
        parent_id = _text(row.get("parent_id"))
        candidate_id = _text(row.get("candidate_id"))
        if not parent_id or not candidate_id:
            raise ValueError("Pair details contain an empty parent/candidate ID.")
        if candidate_id not in candidate_set:
            raise ValueError(f"Pair details contain unfrozen candidate {candidate_id!r}.")
        key = (parent_id, candidate_id)
        if key in seen_pairs:
            raise ValueError(f"Duplicate parent-candidate pair: {key}")
        seen_pairs.add(key)
        if parent_id not in seen_parents:
            seen_parents.add(parent_id)
            parent_order.append(parent_id)
        matrix.setdefault(parent_id, {})[candidate_id] = dict(row)
    if len(parent_order) != int(expected_parent_count):
        raise ValueError(
            f"Parent count mismatch: {len(parent_order)} != {expected_parent_count}"
        )
    expected = int(expected_parent_count) * len(candidate_ids)
    if expected != int(expected_pair_count) or len(seen_pairs) != expected:
        raise ValueError(
            f"Cartesian pair count mismatch: rows={len(seen_pairs)}, "
            f"expected={expected}, CLI_expected={expected_pair_count}."
        )
    missing = [
        (parent_id, candidate_id)
        for parent_id in parent_order
        for candidate_id in candidate_ids
        if candidate_id not in matrix[parent_id]
    ]
    if missing:
        raise ValueError(f"Incomplete Cartesian matrix; missing sample={missing[:5]}")
    return parent_order, matrix


def _pairwise_prefix_mean(matrix: np.ndarray, k: int) -> float:
    if int(k) < 2:
        return 0.0
    selected = matrix[: int(k), : int(k)]
    upper = selected[np.triu_indices(int(k), k=1)]
    return float(np.mean(upper)) if upper.size else 0.0


def compute_prefix_artifacts(
    *,
    details: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    parent_ids: Sequence[str],
    thresholds: Sequence[float],
    theta_star: float,
    cost_cap: float,
    source_label: int,
    target_label: int,
    method_name: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    candidate_index = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    parent_index = {parent_id: index for index, parent_id in enumerate(parent_ids)}
    distances = np.full((len(parent_ids), len(candidates)), np.inf, dtype=np.float64)
    cf_drops = np.full_like(distances, np.nan)
    applicable = np.zeros_like(distances, dtype=bool)
    rows_by_parent: dict[str, list[dict[str, Any]]] = {
        parent_id: [] for parent_id in parent_ids
    }
    for row in details:
        parent_id = _text(row.get("parent_id"))
        candidate_id = _text(row.get("candidate_id"))
        rows_by_parent[parent_id].append(dict(row))
        i = parent_index[parent_id]
        j = candidate_index[candidate_id]
        applicable[i, j] = _applicable(row)
        distance = _finite_distance(row)
        if distance is not None and _strict_flip(
            row,
            source_label=source_label,
            target_label=target_label,
        ):
            distances[i, j] = distance
            drop = _as_float(row.get("cf_drop"))
            if drop is not None:
                cf_drops[i, j] = drop

    coverage_redundancy = build_coverage_redundancy_matrix(
        distances,
        (
            ThresholdLevel(
                threshold_id="theta_star",
                threshold=float(theta_star),
                weight=1.0,
                quantiles=(),
                quantile_labels=(),
            ),
        ),
    )
    chemistry = build_candidate_chemistry(
        [
            {
                "candidate_id": row["candidate_id"],
                "canonical_fragment": row["candidate_smiles"],
            }
            for row in candidates
        ]
    )

    prefix_metrics: list[dict[str, Any]] = []
    threshold_metrics: list[dict[str, Any]] = []
    parent_best_rows: list[dict[str, Any]] = []
    for k in range(1, len(candidates) + 1):
        prefix_details = [
            row
            for row in details
            if candidate_index[_text(row.get("candidate_id"))] < k
        ]
        threshold_rows = [
            {
                "threshold": float(threshold),
                "threshold_source": "frozen_calibration",
                "quantile": None,
            }
            for threshold in thresholds
        ]
        summaries = summarize_wnode_thresholds(
            method=method_name,
            details=prefix_details,
            threshold_rows=threshold_rows,
            total_parents=len(parent_ids),
            total_candidates=k,
            source_label=source_label,
            target_label=target_label,
            group_audit={
                "candidate_set_preselected": True,
                "selection_performed_in_eval": False,
                "selection_method": "frozen_rank_prefix",
                "evaluation_row_unit": "parent_candidate",
                "num_unique_parent_candidate_pairs": len(parent_ids) * k,
                "num_detail_rows": len(prefix_details),
                "num_valid_match_instances": None,
            },
        )
        threshold_metrics.extend({**row, "k": k} for row in summaries)
        theta_summary = next(
            (
                row
                for row in summaries
                if math.isclose(
                    float(row["threshold"]),
                    float(theta_star),
                    rel_tol=0.0,
                    abs_tol=FLOAT_TOLERANCE,
                )
            ),
            None,
        )
        if theta_summary is None:
            theta_summary = summarize_wnode_thresholds(
                method=method_name,
                details=prefix_details,
                threshold_rows=[
                    {
                        "threshold": float(theta_star),
                        "threshold_source": "frozen_calibration_theta_star",
                        "quantile": 0.30,
                    }
                ],
                total_parents=len(parent_ids),
                total_candidates=k,
                source_label=source_label,
                target_label=target_label,
            )[0]
        best = np.min(distances[:, :k], axis=1)
        best_candidate_positions = np.argmin(distances[:, :k], axis=1)
        finite = np.isfinite(best)
        capped = np.minimum(best, float(cost_cap))
        capped[~finite] = float(cost_cap)
        conditional = best[finite]
        applicable_parent = np.any(applicable[:, :k], axis=1)
        prefix_row = {
            "k": k,
            **theta_summary,
            "num_applicable_parents": int(np.count_nonzero(applicable_parent)),
            "applicable_coverage": float(np.mean(applicable_parent)),
            "num_any_strict_flip_parents": int(np.count_nonzero(finite)),
            "any_strict_flip_coverage": float(np.mean(finite)),
            "conditional_mean_cost": (
                float(np.mean(conditional)) if conditional.size else None
            ),
            "conditional_median_cost": (
                float(np.median(conditional)) if conditional.size else None
            ),
            "fixed_capped_mean_cost": float(np.mean(capped)),
            "fixed_capped_median_cost": float(np.median(capped)),
            "coverage_redundancy": _pairwise_prefix_mean(
                coverage_redundancy, k
            ),
            "structural_redundancy": _pairwise_prefix_mean(
                chemistry.structural_similarity, k
            ),
        }
        prefix_metrics.append(prefix_row)
        for i, parent_id in enumerate(parent_ids):
            selected_position = int(best_candidate_positions[i]) if finite[i] else -1
            selected_drop = (
                _as_float(cf_drops[i, selected_position])
                if selected_position >= 0
                else None
            )
            parent_best_rows.append(
                {
                    "k": k,
                    "parent_id": parent_id,
                    "best_candidate_id": (
                        candidate_ids[selected_position]
                        if selected_position >= 0
                        else None
                    ),
                    "best_distance": float(best[i]) if finite[i] else None,
                    "capped_distance": float(capped[i]),
                    "strict_recourse_available": bool(finite[i]),
                    "theta_star_covered": bool(best[i] <= float(theta_star)),
                    "applicable": bool(applicable_parent[i]),
                    "cf_drop": selected_drop,
                }
            )
    return prefix_metrics, threshold_metrics, parent_best_rows


def _same_value(expected: Any, actual: Any, field: str) -> bool:
    if field.startswith("num_"):
        return _as_int(expected) == _as_int(actual)
    left = _as_float(expected)
    right = _as_float(actual)
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE)


def reconstruct_official_summary(
    *,
    recomputed_k20: Sequence[dict[str, Any]],
    official_rows: Sequence[dict[str, Any]],
    thresholds: Sequence[float],
    theta_star: float,
    expected_theta_star_covered: int | None = None,
    recomputed_theta_star_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for threshold in thresholds:
        recomputed = next(
            row
            for row in recomputed_k20
            if math.isclose(
                float(row["threshold"]),
                float(threshold),
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            )
        )
        matches = [
            row
            for row in official_rows
            if (_as_float(row.get("threshold")) is not None)
            and math.isclose(
                float(row["threshold"]),
                float(threshold),
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            )
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Official summary must contain one row for threshold={threshold}; "
                f"found={len(matches)}."
            )
        official = matches[0]
        field_results = {
            field: _same_value(official.get(field), recomputed.get(field), field)
            for field in OFFICIAL_FIELDS
        }
        comparisons.append(
            {
                "threshold": float(threshold),
                "all_fields_match": all(field_results.values()),
                "field_matches": field_results,
                "official": {field: official.get(field) for field in OFFICIAL_FIELDS},
                "recomputed": {
                    field: recomputed.get(field) for field in OFFICIAL_FIELDS
                },
            }
        )
    failures = [row for row in comparisons if not row["all_fields_match"]]
    if failures:
        raise RuntimeError(
            "Official K20 threshold summary reconstruction failed: "
            f"{json.dumps(failures[:2], ensure_ascii=False)}"
        )
    theta_matches = [
        row
        for row in recomputed_k20
        if (_as_float(row.get("threshold")) is not None)
        and math.isclose(
            float(row["threshold"]),
            float(theta_star),
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        )
    ]
    if len(theta_matches) > 1:
        raise RuntimeError(
            "Recomputed K20 summary contains multiple theta-star rows."
        )
    if theta_matches:
        theta_row = theta_matches[0]
        theta_row_source = "frozen_threshold_grid"
        if recomputed_theta_star_row is not None:
            supplied = dict(recomputed_theta_star_row)
            mismatched = [
                field
                for field in OFFICIAL_FIELDS
                if not _same_value(theta_row.get(field), supplied.get(field), field)
            ]
            supplied_threshold = _as_float(supplied.get("threshold"))
            if supplied_threshold is None or not math.isclose(
                supplied_threshold,
                float(theta_star),
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            ):
                mismatched.append("threshold")
            if mismatched:
                raise RuntimeError(
                    "Explicit recomputed theta-star row disagrees with its frozen "
                    f"grid row: {sorted(set(mismatched))}"
                )
    else:
        if recomputed_theta_star_row is None:
            raise RuntimeError(
                "theta_star is not an exact member of the frozen threshold grid; "
                "an explicit recomputed theta-star row is required."
            )
        theta_row = dict(recomputed_theta_star_row)
        theta_row_source = "recomputed_prefix_theta_star"
        supplied_threshold = _as_float(theta_row.get("threshold"))
        if supplied_threshold is None or not math.isclose(
            supplied_threshold,
            float(theta_star),
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ):
            raise RuntimeError(
                "Explicit recomputed theta-star row threshold differs from theta_star."
            )
        if _text(theta_row.get("threshold_source")) != (
            "frozen_calibration_theta_star"
        ):
            raise RuntimeError(
                "Explicit recomputed theta-star row lacks the historical "
                "frozen_calibration_theta_star provenance."
            )
        missing = [
            field
            for field in ("k", "threshold", *OFFICIAL_FIELDS)
            if field not in theta_row
        ]
        if missing:
            raise RuntimeError(
                f"Explicit recomputed theta-star row is incomplete: {missing}"
            )

    num_parents = _as_int(theta_row.get("num_parents"))
    num_candidates = _as_int(theta_row.get("num_candidates"))
    num_covered = _as_int(theta_row.get("num_close_cf_covered"))
    coverage = _as_float(theta_row.get("close_cf_coverage"))
    if (
        num_parents is None
        or num_parents <= 0
        or num_candidates is None
        or num_candidates <= 0
        or num_covered is None
        or num_covered < 0
        or num_covered > num_parents
        or coverage is None
    ):
        raise RuntimeError("Theta-star row has invalid coverage identity fields.")
    if theta_row_source == "recomputed_prefix_theta_star":
        if _as_int(theta_row.get("k")) != num_candidates:
            raise RuntimeError(
                "Explicit recomputed theta-star row does not describe its full K prefix."
            )
        if not recomputed_k20:
            raise RuntimeError("Frozen K20 threshold reconstruction is empty.")
        anchor = recomputed_k20[0]
        for field in ("num_parents", "num_candidates", "num_valid_pairs"):
            if not _same_value(anchor.get(field), theta_row.get(field), field):
                raise RuntimeError(
                    "Explicit recomputed theta-star row differs from the frozen "
                    f"K20 identity field {field}."
                )
    expected_coverage_from_count = num_covered / num_parents
    if not math.isclose(
        coverage,
        expected_coverage_from_count,
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise RuntimeError("Theta-star coverage is not covered/num_parents.")
    if expected_theta_star_covered is not None:
        actual = num_covered
        if actual != int(expected_theta_star_covered):
            raise RuntimeError(
                f"Theta-star covered count mismatch: {actual} != "
                f"{expected_theta_star_covered}."
            )
        expected_coverage = int(expected_theta_star_covered) / num_parents
        if not math.isclose(
            coverage,
            expected_coverage,
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ):
            raise RuntimeError("Theta-star coverage is not covered/num_parents.")
    return {
        "official_summary_reconstruction_passed": True,
        "float_abs_tolerance": FLOAT_TOLERANCE,
        "float_rel_tolerance": 0.0,
        "threshold_count": len(thresholds),
        "comparisons": comparisons,
        "theta_star": float(theta_star),
        "theta_star_row_source": theta_row_source,
        "theta_star_num_close_cf_covered": num_covered,
        "theta_star_close_cf_coverage": coverage,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or [])
    if not fields:
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        json.dumps(row.get(field), ensure_ascii=False)
                        if isinstance(row.get(field), (dict, list, tuple))
                        else ("" if row.get(field) is None else row.get(field))
                    )
                    for field in fields
                }
            )


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _method_slug(method_name: str) -> str:
    lowered = str(method_name).lower()
    for token, slug in (
        ("globalgce", "globalgce"),
        ("gcfexplainer", "gcfexplainer"),
        ("clear", "clear"),
    ):
        if token in lowered:
            return slug
    return "_".join(lowered.replace("-", " ").split())


def _deep_find(payload: Any, names: set[str]) -> Any:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key) in names and value not in (None, ""):
                return value
        for value in payload.values():
            found = _deep_find(value, names)
            if found not in (None, ""):
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _deep_find(value, names)
            if found not in (None, ""):
                return found
    return None


def _inherited_file_identity(
    config: dict[str, Any],
    path_names: set[str],
    hash_names: set[str],
) -> dict[str, Any]:
    path_value = _deep_find(config, path_names)
    inherited_hash = _deep_find(config, hash_names)
    result = {
        "path": str(path_value) if path_value not in (None, "") else None,
        "sha256": (
            str(inherited_hash) if inherited_hash not in (None, "") else None
        ),
        "sha256_source": "run_config",
    }
    if path_value not in (None, ""):
        path = Path(str(path_value)).expanduser()
        if path.is_file():
            result["path"] = str(path.resolve())
            result["sha256"] = sha256_file(path)
            result["sha256_source"] = "file"
    return result


def _resolve_table_fields(
    ours_schema_root: Path,
    k: int,
) -> list[str]:
    reference = ours_schema_root / f"table2_ours_k{k}.csv"
    _, fields = read_csv(reference)
    resolved = list(fields)
    for field in TABLE_REQUIRED_FIELDS:
        if field not in resolved:
            resolved.append(field)
    return resolved


def _extract_threshold_values(payload: dict[str, Any]) -> list[float]:
    raw = payload.get("raw_quantile_thresholds")
    if isinstance(raw, list):
        values = [
            _as_float(item.get("threshold"))
            for item in raw
            if isinstance(item, dict)
        ]
        if values and all(value is not None for value in values):
            return [float(value) for value in values if value is not None]
    raw = payload.get("thresholds")
    if isinstance(raw, list):
        values = [_as_float(value) for value in raw]
        if values and all(value is not None for value in values):
            return [float(value) for value in values if value is not None]
    return []


def load_frozen_threshold_contract(path: str | Path) -> dict[str, Any]:
    """Load the shared Mutagenicity thresholds without fitting or fallback."""

    threshold_path = Path(path).expanduser().resolve()
    payload = read_json(threshold_path)
    thresholds = _extract_threshold_values(payload)
    theta_star = _as_float(payload.get("theta_star"))
    cost_cap = _as_float(payload.get("cost_cap"))
    threshold_source = _text(payload.get("threshold_source"))
    if not thresholds or theta_star is None or cost_cap is None:
        raise ValueError(f"Frozen threshold schema is incomplete: {threshold_path}")
    if any(not math.isfinite(value) or value < 0.0 for value in thresholds):
        raise ValueError("Frozen thresholds must be finite and nonnegative.")
    if thresholds != sorted(thresholds) or len(thresholds) != len(set(thresholds)):
        raise ValueError("Frozen thresholds must be unique and increasing.")
    if float(theta_star) < thresholds[0] or float(theta_star) > thresholds[-1]:
        raise ValueError("Frozen theta_star is outside the threshold grid.")
    if not math.isclose(
        float(cost_cap),
        float(thresholds[-1]),
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise ValueError("Frozen cost_cap must equal the largest threshold.")
    if threshold_source and not threshold_source.startswith("frozen_calibration"):
        raise ValueError(
            f"Unexpected frozen threshold source: {threshold_source!r}"
        )
    return {
        "thresholds_json": str(threshold_path),
        "thresholds_json_sha256": sha256_file(threshold_path),
        "threshold_source": threshold_source or "frozen_calibration",
        "thresholds": thresholds,
        "theta_star": float(theta_star),
        "cost_cap": float(cost_cap),
    }


def validate_frozen_threshold_provenance(
    *,
    ours_schema_root: str | Path,
    calibration_run_dir: str | Path,
    theta_star: float,
    cost_cap: float,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    ours_root = Path(ours_schema_root).expanduser().resolve()
    threshold_path = ours_root / "thresholds.json"
    frozen = read_json(threshold_path)
    frozen_theta = _as_float(frozen.get("theta_star"))
    frozen_cap = _as_float(frozen.get("cost_cap"))
    frozen_thresholds = _extract_threshold_values(frozen)
    if frozen_theta is None or frozen_cap is None or not frozen_thresholds:
        raise ValueError(
            f"Ours frozen thresholds schema is incomplete: {threshold_path}"
        )
    if not math.isclose(
        frozen_theta,
        float(theta_star),
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise ValueError("CLI theta_star differs from frozen calibration theta_star.")
    if not math.isclose(
        frozen_cap,
        float(cost_cap),
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise ValueError("CLI cost_cap differs from frozen calibration cost_cap.")
    requested = [float(value) for value in thresholds]
    if len(requested) != len(frozen_thresholds) or any(
        not math.isclose(
            left,
            right,
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        )
        for left, right in zip(requested, frozen_thresholds)
    ):
        raise ValueError("CLI thresholds differ from frozen calibration thresholds.")

    calibration_root = Path(calibration_run_dir).expanduser().resolve()
    calibration_values: list[float] = []
    quantile_csv = calibration_root / "distance_quantiles.csv"
    if quantile_csv.is_file():
        rows, _ = read_csv(quantile_csv)
        calibration_values = [
            value
            for row in rows
            if (value := _as_float(row.get("threshold"))) is not None
        ]
    else:
        config_path = calibration_root / "run_config.json"
        if config_path.is_file():
            calibration_values = _extract_threshold_values(read_json(config_path))
    if calibration_values and (
        len(calibration_values) != len(requested)
        or any(
            not math.isclose(
                left,
                right,
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            )
            for left, right in zip(calibration_values, requested)
        )
    ):
        raise ValueError(
            "Calibration run thresholds differ from the frozen threshold list."
        )
    return {
        "threshold_source": "frozen_calibration",
        "ours_thresholds_json": str(threshold_path),
        "ours_thresholds_json_sha256": sha256_file(threshold_path),
        "theta_star_matches": True,
        "cost_cap_matches": True,
        "thresholds_match": True,
        "calibration_run_thresholds_checked": bool(calibration_values),
    }


def _dataset_parent_provenance(
    *,
    dataset_path: Path,
    cohort_name: str,
    id_col: str,
    label_col: str,
    source_label: int,
    expected_parent_count: int,
) -> dict[str, Any]:
    if int(source_label) != 1:
        raise ValueError("Mutagenicity fullgraph evaluation requires source_label=1.")
    if cohort_name == "calibration":
        from src.eval.mutagenicity_wnode_matrix import (
            calibration_cohort_hash,
            load_calibration_parents,
        )

        parents = load_calibration_parents(
            dataset_path,
            id_col=id_col,
            smiles_col="smiles",
            label_col=label_col,
            cohort_name="calibration",
            expected_parent_count=expected_parent_count,
        )
        cohort_hash = calibration_cohort_hash(parents)
    elif cohort_name == "test":
        from src.eval.mutagenicity_wnode_frozen_test import (
            load_test_parents,
            test_cohort_hash,
        )

        parents = load_test_parents(
            dataset_path,
            id_col=id_col,
            smiles_col="smiles",
            label_col=label_col,
            cohort_name="test",
            expected_parent_count=expected_parent_count,
        )
        cohort_hash = test_cohort_hash(parents)
    else:  # pragma: no cover - guarded by the caller.
        raise ValueError(f"Unsupported parent cohort: {cohort_name!r}")
    parent_ids = [parent.parent_id for parent in parents]
    return {
        "dataset_csv": str(dataset_path),
        "dataset_csv_sha256": sha256_file(dataset_path),
        "source_parent_ids_sha256": stable_json_sha256(parent_ids),
        "source_parent_count": len(parent_ids),
        "parent_cohort_hash": cohort_hash,
        f"{cohort_name}_cohort_hash": cohort_hash,
        "parent_cohort_hash_source": (
            f"src.eval.mutagenicity_wnode_{'matrix' if cohort_name == 'calibration' else 'frozen_test'}"
        ),
    }


def audit_fullgraph_evaluation_run(
    *,
    run_dir: str | Path,
    frozen_candidates_csv: str | Path,
    frozen_manifest_path: str | Path,
    thresholds_json: str | Path,
    cohort_name: str,
    expected_parent_count: int,
    expected_candidate_count: int,
    expected_pair_count: int,
    expected_candidate_csv_sha256: str,
    expected_candidate_order_sha256: str,
    expected_teacher_sha256: str,
    expected_molclr_checkpoint_sha256: str,
    expected_native_ranks: Sequence[int],
    expected_method: str,
    expected_selection_method: str,
    source_label: int = 1,
    target_label: int = 0,
) -> dict[str, Any]:
    """Audit one existing fullgraph WNode Cartesian run without recomputation."""

    root = Path(run_dir).expanduser().resolve()
    evaluator_marker = root / "_EVALUATOR_COMPLETE.json"
    if not evaluator_marker.is_file():
        raise FileNotFoundError(
            f"Evaluator completion marker is missing: {evaluator_marker}"
        )
    evaluator_completion = read_json(evaluator_marker)
    if not _as_bool(evaluator_completion.get("complete")):
        raise ValueError("Evaluator completion marker does not declare complete=true.")
    if _as_int(
        evaluator_completion.get("num_unique_parent_candidate_pairs")
    ) != int(expected_pair_count):
        raise ValueError("Evaluator completion marker pair count mismatch.")
    candidates_audit = validate_frozen_candidate_contract(
        candidates_csv=frozen_candidates_csv,
        frozen_manifest_path=frozen_manifest_path,
        expected_count=expected_candidate_count,
        expected_csv_sha256=expected_candidate_csv_sha256,
        expected_order_sha256=expected_candidate_order_sha256,
        expected_native_ranks=expected_native_ranks,
        expected_selection_method=expected_selection_method,
    )
    threshold_contract = load_frozen_threshold_contract(thresholds_json)
    candidates, _ = load_ranked_candidates(
        frozen_candidates_csv,
        expected_count=expected_candidate_count,
    )
    pair_path, summary_path, config_path = locate_test_inputs(root)
    details, _ = read_csv(pair_path)
    summary_rows, _ = read_csv(summary_path)
    config = read_json(config_path)
    parent_ids, _ = validate_complete_cartesian(
        details,
        candidates,
        expected_parent_count=expected_parent_count,
        expected_pair_count=expected_pair_count,
    )
    if len(details) != int(expected_pair_count):
        raise ValueError(
            f"Fullgraph detail rows={len(details)} != {expected_pair_count}."
        )
    if any(_finite_distance(row) is None for row in details):
        raise ValueError("Fullgraph Cartesian matrix contains a non-finite distance.")
    if any(_as_int(row.get("pred_before")) != int(source_label) for row in details):
        raise ValueError("Fullgraph source teacher prediction is not source_label=1.")
    if any(_as_int(row.get("pred_after")) != int(target_label) for row in details):
        raise ValueError("Fullgraph candidate teacher prediction is not target_label=0.")
    if any(not _as_bool(row.get("teacher_strict_flip")) for row in details):
        raise ValueError("Fullgraph pair matrix contains a non-strict-flip pair.")

    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    for parent_id in parent_ids:
        observed = [
            _text(row.get("candidate_id"))
            for row in details
            if _text(row.get("parent_id")) == parent_id
        ]
        if observed != candidate_ids:
            raise ValueError(
                f"Evaluator changed frozen candidate order for parent={parent_id!r}."
            )
    if stable_json_sha256(candidate_ids) != str(
        expected_candidate_order_sha256
    ).lower():
        raise ValueError("Evaluation candidate order hash changed.")

    expected_thresholds = [
        float(value) for value in threshold_contract["thresholds"]
    ]
    observed_thresholds: list[float] = []
    for row in summary_rows:
        if _text(row.get("method")) != str(expected_method):
            raise ValueError(
                f"Unexpected method in threshold summary: {row.get('method')!r}"
            )
        value = _as_float(row.get("threshold"))
        if value is None:
            raise ValueError("Threshold summary contains a non-finite threshold.")
        observed_thresholds.append(value)
        if _as_int(row.get("num_parents")) != int(expected_parent_count):
            raise ValueError("Threshold summary parent count mismatch.")
        if _as_int(row.get("num_candidates")) != int(expected_candidate_count):
            raise ValueError("Threshold summary candidate count mismatch.")
        if _as_int(row.get("num_valid_pairs")) != int(expected_pair_count):
            raise ValueError("Threshold summary valid-pair count mismatch.")
        if _text(row.get("cf_mode")) != "strict_flip":
            raise ValueError("Threshold summary does not use strict_flip.")
    if len(observed_thresholds) != len(expected_thresholds) or any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE)
        for left, right in zip(observed_thresholds, expected_thresholds)
    ):
        raise ValueError("Evaluation threshold grid differs from frozen thresholds.")

    if _text(config.get("threshold_source")) != "explicit":
        raise ValueError("Evaluation used fitted or auto-quantile thresholds.")
    configured_thresholds = config.get("thresholds")
    if not isinstance(configured_thresholds, list) or len(configured_thresholds) != len(
        expected_thresholds
    ):
        raise ValueError("Evaluation run_config threshold list is incomplete.")
    if any(
        not math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=FLOAT_TOLERANCE
        )
        for left, right in zip(configured_thresholds, expected_thresholds)
    ):
        raise ValueError("Evaluation run_config thresholds changed.")
    if config.get("candidate_set_preselected") is not True:
        raise ValueError("Evaluation did not recognize the frozen candidate set.")
    if config.get("selection_performed_in_eval") is not False:
        raise ValueError("Evaluation performed candidate selection.")
    if _text(config.get("selection_method")) != str(expected_selection_method):
        raise ValueError("Evaluation selection_method changed.")
    if config.get("run_ours") is not False or config.get("run_fullgraph") is not True:
        raise ValueError("Evaluation did not run in fullgraph-only mode.")
    if _text(config.get("cf_mode")) != "strict_flip":
        raise ValueError("Evaluation run_config does not use strict_flip.")
    if _text(config.get("feature_cost")) != "cosine":
        raise ValueError("Evaluation feature_cost must be cosine.")
    if _text(config.get("node_mass")) != "uniform":
        raise ValueError("Evaluation node_mass must be uniform.")
    if _as_float(config.get("size_penalty_beta")) != 0.0:
        raise ValueError("Evaluation size_penalty_beta must be 0.0.")
    if _as_int(config.get("label")) != int(source_label):
        raise ValueError("Evaluation source label differs from source_label=1.")
    if _as_int(config.get("max_parents")) != int(expected_parent_count):
        raise ValueError("Evaluation max_parents differs from the frozen cohort.")
    if _as_int(config.get("max_candidates")) != int(expected_candidate_count):
        raise ValueError("Evaluation max_candidates differs from frozen Top20.")
    if _as_int(config.get("preselected_topk")) != int(expected_candidate_count):
        raise ValueError("Evaluation preselected_topk differs from frozen Top20.")

    teacher_path = Path(_text(config.get("teacher_path"))).expanduser().resolve()
    molclr_checkpoint = Path(
        _text(config.get("molclr_checkpoint"))
    ).expanduser().resolve()
    for description, path, expected_sha256 in (
        ("RF teacher", teacher_path, expected_teacher_sha256),
        (
            "MolCLR checkpoint",
            molclr_checkpoint,
            expected_molclr_checkpoint_sha256,
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} is missing: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != str(expected_sha256).lower():
            raise ValueError(
                f"{description} SHA256 mismatch: actual={actual_sha256}"
            )

    dataset_path = Path(_text(config.get("dataset_csv"))).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset CSV is missing: {dataset_path}")
    normalized_cohort = str(cohort_name).strip().lower()
    if normalized_cohort not in {"calibration", "test"}:
        raise ValueError(f"Unsupported evaluation cohort: {cohort_name!r}")
    dataset_name = dataset_path.name.lower()
    is_test_input = dataset_name.startswith("test_") or "_test_" in dataset_name
    if normalized_cohort == "calibration" and is_test_input:
        raise ValueError("Calibration WNode run references a test path.")
    if normalized_cohort == "test" and not is_test_input:
        raise ValueError("Frozen test WNode run does not reference the test split.")
    parent_provenance = _dataset_parent_provenance(
        dataset_path=dataset_path,
        cohort_name=normalized_cohort,
        id_col="molecule_id",
        label_col=_text(config.get("label_col")) or "label",
        source_label=int(source_label),
        expected_parent_count=int(expected_parent_count),
    )
    return {
        "audit_passed": True,
        "run_complete": True,
        "cohort": normalized_cohort,
        "source_label": int(source_label),
        "target_label": int(target_label),
        "parent_count": len(parent_ids),
        "candidate_count": len(candidates),
        "pair_count": len(details),
        "complete_cartesian": True,
        "all_pair_distances_finite": True,
        "all_source_teacher_predictions_match": True,
        "all_candidate_teacher_predictions_match": True,
        "all_pairs_strict_flip": True,
        "evaluation_parent_ids_sha256": stable_json_sha256(parent_ids),
        **parent_provenance,
        **candidates_audit,
        "threshold_provenance": threshold_contract,
        "candidate_selection_performed": False,
        "selection_used_calibration": False,
        "selection_used_test": False,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "candidate_set_preselected": True,
        "selection_performed_in_eval": False,
        "strict_flip": True,
        "distance_line": "MolCLR-Node-Wasserstein",
        "feature_cost": "cosine",
        "node_mass": "uniform",
        "size_penalty_beta": 0.0,
        "solver": "exact_emd2",
        "pair_details": str(pair_path),
        "pair_details_sha256": sha256_file(pair_path),
        "threshold_summary": str(summary_path),
        "threshold_summary_sha256": sha256_file(summary_path),
        "run_config": str(config_path),
        "run_config_sha256": sha256_file(config_path),
        "teacher_path": str(teacher_path),
        "teacher_sha256": sha256_file(teacher_path),
        "molclr_checkpoint": str(molclr_checkpoint),
        "molclr_checkpoint_sha256": sha256_file(molclr_checkpoint),
    }


def finalize_fullgraph_evaluation_run(**kwargs: Any) -> dict[str, Any]:
    """Persist a successful read-only audit after the evaluator completes."""

    audit = audit_fullgraph_evaluation_run(**kwargs)
    root = Path(kwargs["run_dir"]).expanduser().resolve()
    if (root / "_RUN_COMPLETE.json").exists():
        raise FileExistsError(f"Run completion marker already exists: {root}")
    _write_json(root / "audit.json", audit)
    _write_json(root / "summary.json", audit)
    _write_json(
        root / "run_manifest.json",
        {
            **audit,
            "run_complete": True,
            "candidate_selection_source": "train_only_frozen_candidates",
            "candidate_order_source": "frozen_csv_row_order",
        },
    )
    _write_json(
        root / "_RUN_COMPLETE.json",
        {
            "run_complete": True,
            "audit_passed": True,
            "cohort": audit["cohort"],
            "parent_count": audit["parent_count"],
            "candidate_count": audit["candidate_count"],
            "pair_count": audit["pair_count"],
            "selected_candidate_order_sha256": audit[
                "selected_candidate_order_sha256"
            ],
        },
    )
    return audit


def _table_row(
    metric: dict[str, Any],
    *,
    method_name: str,
    dataset: str,
    source_label: int,
    target_label: int,
    theta_star: float,
) -> dict[str, Any]:
    return {
        "method": method_name,
        "dataset": dataset,
        "source_label": int(source_label),
        "target_label": int(target_label),
        "k": int(metric["k"]),
        "theta": float(theta_star),
        "coverage": metric["close_cf_coverage"],
        "ccrcov": metric["close_cf_coverage"],
        "applicable_rate": metric["applicable_coverage"],
        "applicable_coverage": metric["applicable_coverage"],
        "any_strict_flip_coverage": metric["any_strict_flip_coverage"],
        "flip_rate_among_covered": metric["flip_rate_among_covered"],
        "avg_cf_drop_among_covered": metric["avg_cf_drop_among_covered"],
        "mean_cf_drop": metric["avg_cf_drop_among_covered"],
        "conditional_mean_cost": metric["conditional_mean_cost"],
        "conditional_median_cost": metric["conditional_median_cost"],
        "fixed_capped_mean_cost": metric["fixed_capped_mean_cost"],
        "fixed_capped_median_cost": metric["fixed_capped_median_cost"],
        "coverage_redundancy": metric["coverage_redundancy"],
        "structural_redundancy": metric["structural_redundancy"],
        "num_test_parents": metric["num_parents"],
        "num_candidates": int(metric["k"]),
        "selected_variant": "frozen_fullgraph_rank",
    }


def _check_run_semantics(
    config: dict[str, Any],
    *,
    forbid_selection: bool,
    forbid_fitting: bool,
) -> None:
    cf_mode = _deep_find(config, {"cf_mode"})
    if cf_mode is not None and _text(cf_mode) != "strict_flip":
        raise ValueError(f"Test run is not strict_flip: cf_mode={cf_mode!r}")
    if forbid_selection:
        preselected = _deep_find(config, {"candidate_set_preselected"})
        selection = _deep_find(config, {"selection_performed_in_eval"})
        if preselected is None or not _as_bool(preselected):
            raise ValueError("Test run does not declare preselected candidates.")
        if selection is None or _as_bool(selection):
            raise ValueError("Test evaluator performed candidate selection.")
    if forbid_fitting:
        source = _deep_find(config, {"threshold_source"})
        if source is None:
            raise ValueError("Test run does not declare its threshold_source.")
        if _text(source).lower() in {
            "auto",
            "auto_quantile",
            "test",
            "test_quantile",
        }:
            raise ValueError(f"Test thresholds were fitted in test run: {source!r}")


def export_final_artifacts(
    *,
    test_run_dir: str | Path,
    calibration_run_dir: str | Path,
    frozen_candidates_csv: str | Path,
    ours_schema_root: str | Path,
    output_dir: str | Path,
    method_name: str,
    dataset: str,
    source_label: int,
    target_label: int,
    test_job_id: str,
    theta_star: float,
    cost_cap: float,
    thresholds: Sequence[float],
    k_values: Sequence[int],
    expected_parent_count: int,
    expected_candidate_count: int,
    expected_pair_count: int,
    forbid_selection: bool,
    forbid_fitting: bool,
    frozen_candidate_manifest: str | Path | None = None,
    expected_candidate_order_sha256: str | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if not thresholds:
        raise ValueError("At least one frozen threshold is required.")
    calibration_root = Path(calibration_run_dir).expanduser().resolve()
    if not calibration_root.is_dir():
        raise FileNotFoundError(
            f"Calibration provenance directory does not exist: {calibration_root}"
        )
    frozen_thresholds = [float(value) for value in thresholds]
    if any(not math.isfinite(value) for value in frozen_thresholds):
        raise ValueError("Thresholds must all be finite.")
    threshold_provenance = validate_frozen_threshold_provenance(
        ours_schema_root=ours_schema_root,
        calibration_run_dir=calibration_root,
        theta_star=theta_star,
        cost_cap=cost_cap,
        thresholds=frozen_thresholds,
    )
    if float(theta_star) < frozen_thresholds[0] or float(theta_star) > frozen_thresholds[-1]:
        raise ValueError("theta_star must be inside the frozen threshold grid.")
    requested_k = sorted(set(int(value) for value in k_values))
    if requested_k != list(range(1, int(expected_candidate_count) + 1)):
        raise ValueError(
            f"k_values must be 1..{expected_candidate_count}; got={requested_k}"
        )

    pair_path, official_path, config_path = locate_test_inputs(test_run_dir)
    details, detail_fields = read_csv(pair_path)
    strict_mismatches = []
    for row in details:
        if row.get("cf_flip") in (None, ""):
            continue
        expected = _strict_flip(
            row,
            source_label=int(source_label),
            target_label=int(target_label),
        )
        if _as_bool(row.get("cf_flip")) != expected:
            strict_mismatches.append(
                (_text(row.get("parent_id")), _text(row.get("candidate_id")))
            )
    if strict_mismatches:
        raise ValueError(
            "pair_details cf_flip does not match strict source-to-target flip; "
            f"sample={strict_mismatches[:5]}"
        )
    official_rows, _ = read_csv(official_path)
    run_config = read_json(config_path)
    _check_run_semantics(
        run_config,
        forbid_selection=forbid_selection,
        forbid_fitting=forbid_fitting,
    )
    candidates, candidate_fields = load_ranked_candidates(
        frozen_candidates_csv,
        expected_count=expected_candidate_count,
    )
    candidate_order_sha256 = stable_json_sha256(
        [str(row["candidate_id"]) for row in candidates]
    )
    if (
        expected_candidate_order_sha256 is not None
        and candidate_order_sha256
        != str(expected_candidate_order_sha256).strip().lower()
    ):
        raise ValueError("Frozen candidate order SHA256 differs from expected value.")
    parent_ids, _ = validate_complete_cartesian(
        details,
        candidates,
        expected_parent_count=expected_parent_count,
        expected_pair_count=expected_pair_count,
    )

    prefix_metrics, threshold_metrics, parent_best_rows = compute_prefix_artifacts(
        details=details,
        candidates=candidates,
        parent_ids=parent_ids,
        thresholds=frozen_thresholds,
        theta_star=float(theta_star),
        cost_cap=float(cost_cap),
        source_label=int(source_label),
        target_label=int(target_label),
        method_name=method_name,
    )
    k20_rows = [
        row for row in threshold_metrics if int(row["k"]) == expected_candidate_count
    ]
    expected_theta_count = (
        69
        if method_name == "GlobalGCE-Frequency-Top20"
        and int(expected_parent_count) == 217
        else None
    )
    reconstruction = reconstruct_official_summary(
        recomputed_k20=k20_rows,
        official_rows=official_rows,
        thresholds=frozen_thresholds,
        theta_star=float(theta_star),
        expected_theta_star_covered=expected_theta_count,
        recomputed_theta_star_row=next(
            row
            for row in prefix_metrics
            if int(row["k"]) == int(expected_candidate_count)
        ),
    )

    ordered_details = sorted(
        details,
        key=lambda row: (
            parent_ids.index(_text(row.get("parent_id"))),
            int(next(
                item["rank"]
                for item in candidates
                if item["candidate_id"] == _text(row.get("candidate_id"))
            )),
        ),
    )
    by_k = {int(row["k"]): row for row in prefix_metrics}
    figure4_rows = [
        row
        for row in threshold_metrics
        if int(row["k"]) in {10, int(expected_candidate_count)}
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        selected_fields = list(candidate_fields)
        for field in ("rank", "candidate_id", "candidate_smiles"):
            if field not in selected_fields:
                selected_fields.append(field)
        _write_csv(temp / "selected_top20.csv", candidates, selected_fields)
        _write_jsonl(temp / "selected_sequence.jsonl", candidates)
        _write_csv(temp / "test_pair_details.csv", ordered_details, detail_fields)
        _write_csv(temp / "test_threshold_summary.csv", k20_rows)
        _write_csv(temp / "parent_best_distances.csv", parent_best_rows)
        _write_csv(temp / "prefix_metrics.csv", prefix_metrics)
        _write_json(temp / "prefix_metrics.json", {"prefix_metrics": prefix_metrics})
        _write_csv(temp / "figure3_coverage_vs_k.csv", prefix_metrics)
        _write_csv(temp / "figure4_coverage_vs_threshold.csv", figure4_rows)
        for k in (10, int(expected_candidate_count)):
            table_row = _table_row(
                by_k[k],
                method_name=method_name,
                dataset=dataset,
                source_label=source_label,
                target_label=target_label,
                theta_star=theta_star,
            )
            fields = _resolve_table_fields(Path(ours_schema_root).resolve(), k)
            slug = _method_slug(method_name)
            _write_csv(temp / f"table2_{slug}_k{k}.csv", [table_row], fields)

        k10 = by_k[10]
        k20 = by_k[int(expected_candidate_count)]
        summary = {
            "method": method_name,
            "dataset": dataset,
            "source_label": int(source_label),
            "target_label": int(target_label),
            "test_parent_count": len(parent_ids),
            "candidate_count": len(candidates),
            "pair_count": len(details),
            "complete_cartesian": True,
            "theta_star": float(theta_star),
            "cost_cap": float(cost_cap),
            "thresholds": frozen_thresholds,
            "threshold_provenance": threshold_provenance,
            "k10_ccrcov_theta_star": float(k10["close_cf_coverage"]),
            "k20_ccrcov_theta_star": float(k20["close_cf_coverage"]),
            "k10_conditional_mean_cost": k10["conditional_mean_cost"],
            "k10_conditional_median_cost": k10["conditional_median_cost"],
            "k10_fixed_capped_mean_cost": k10["fixed_capped_mean_cost"],
            "k10_fixed_capped_median_cost": k10["fixed_capped_median_cost"],
            "k20_conditional_mean_cost": k20["conditional_mean_cost"],
            "k20_conditional_median_cost": k20["conditional_median_cost"],
            "k20_fixed_capped_mean_cost": k20["fixed_capped_mean_cost"],
            "k20_fixed_capped_median_cost": k20["fixed_capped_median_cost"],
            "k10_coverage_redundancy": k10["coverage_redundancy"],
            "k10_structural_redundancy": k10["structural_redundancy"],
            "k20_coverage_redundancy": k20["coverage_redundancy"],
            "k20_structural_redundancy": k20["structural_redundancy"],
            "selection_used_calibration": False,
            "selection_used_test": False,
            "threshold_fitted_on_test": False,
            "test_used_for_selection": False,
            "candidate_selection_performed": False,
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
            "selected_candidate_order_sha256": candidate_order_sha256,
            "official_summary_reconstruction_passed": True,
            "run_complete": True,
        }
        _write_json(temp / "summary.json", summary)
        _write_json(
            temp / "official_summary_reconstruction_audit.json",
            reconstruction,
        )
        repo_root = Path(__file__).resolve().parents[2]
        manifest = {
            "test_job_id": str(test_job_id),
            "generation_input_split": "train",
            "candidate_selection_source": "train_only_frozen_candidates",
            "candidate_selection_performed": False,
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
            "selection_used_calibration": False,
            "selection_used_test": False,
            "threshold_fitted_on_test": False,
            "test_used_for_selection": False,
            "test_parent_count": len(parent_ids),
            "candidate_count": len(candidates),
            "pair_count": len(details),
            "theta_star": float(theta_star),
            "cost_cap": float(cost_cap),
            "thresholds": frozen_thresholds,
            "threshold_provenance": threshold_provenance,
            "candidate_csv": str(Path(frozen_candidates_csv).resolve()),
            "candidate_csv_sha256": sha256_file(frozen_candidates_csv),
            "selected_candidate_order_sha256": candidate_order_sha256,
            "pair_details": str(pair_path),
            "pair_details_sha256": sha256_file(pair_path),
            "official_threshold_summary": str(official_path),
            "official_threshold_summary_sha256": sha256_file(official_path),
            "calibration_run_dir": str(calibration_root),
            "ours_schema_root": str(Path(ours_schema_root).resolve()),
            "teacher": _inherited_file_identity(
                run_config,
                {"teacher_path", "teacher_model_path"},
                {"teacher_sha256", "teacher_hash"},
            ),
            "molclr_checkpoint": _inherited_file_identity(
                run_config,
                {"molclr_checkpoint", "molclr_ckpt"},
                {"molclr_checkpoint_sha256", "molclr_checkpoint_hash"},
            ),
            "git_commit": _git_commit(repo_root),
            "forbid_selection": bool(forbid_selection),
            "forbid_fitting": bool(forbid_fitting),
        }
        if frozen_candidate_manifest is not None:
            frozen_manifest_path = Path(frozen_candidate_manifest).expanduser().resolve()
            manifest["frozen_candidate_manifest"] = str(frozen_manifest_path)
            manifest["frozen_candidate_manifest_sha256"] = sha256_file(
                frozen_manifest_path
            )
        test_audit_path = Path(test_run_dir).expanduser().resolve() / "audit.json"
        calibration_audit_path = calibration_root / "audit.json"
        if test_audit_path.is_file():
            manifest["test_evaluation_audit"] = read_json(test_audit_path)
            manifest["test_evaluation_audit_sha256"] = sha256_file(test_audit_path)
        if calibration_audit_path.is_file():
            manifest["calibration_evaluation_audit"] = read_json(
                calibration_audit_path
            )
            manifest["calibration_evaluation_audit_sha256"] = sha256_file(
                calibration_audit_path
            )
        _write_json(temp / "run_manifest.json", manifest)
        final_audit = audit_final_artifacts(
            run_dir=temp,
            frozen_candidates_csv=frozen_candidates_csv,
            ours_schema_root=ours_schema_root,
            expected_parent_count=expected_parent_count,
            expected_candidate_count=expected_candidate_count,
            expected_pair_count=expected_pair_count,
            theta_star=theta_star,
            cost_cap=cost_cap,
            thresholds=frozen_thresholds,
            check_manifest=False,
        )
        _write_json(temp / "final_artifact_audit.json", final_audit)
        _write_json(temp / "audit.json", final_audit)
        artifact_hashes = {
            path.relative_to(temp).as_posix(): sha256_file(path)
            for path in sorted(temp.rglob("*"))
            if path.is_file()
        }
        _write_json(
            temp / "artifact_manifest.json",
            {
                "files": artifact_hashes,
                "file_count": len(artifact_hashes),
                "all_hashes_generated": True,
                "self_excluded": "artifact_manifest.json",
                "finalization_marker_excluded": "_FINALIZED.json",
                "run_completion_marker_excluded": "_RUN_COMPLETE.json",
            },
        )
        _write_json(
            temp / "_FINALIZED.json",
            {
                "finalized": True,
                "artifact_manifest_sha256": sha256_file(
                    temp / "artifact_manifest.json"
                ),
                "official_summary_reconstruction_passed": True,
                "final_artifact_audit_passed": True,
            },
        )
        _write_json(
            temp / "_RUN_COMPLETE.json",
            {
                "run_complete": True,
                "audit_passed": True,
                "test_parent_count": len(parent_ids),
                "candidate_count": len(candidates),
                "pair_count": len(details),
                "complete_cartesian": True,
                "candidate_selection_performed": False,
                "test_used_for_selection": False,
                "threshold_fitted_on_test": False,
                "selected_candidate_order_sha256": candidate_order_sha256,
            },
        )
        os.replace(temp, output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return summary


def audit_final_artifacts(
    *,
    run_dir: str | Path,
    frozen_candidates_csv: str | Path,
    ours_schema_root: str | Path,
    expected_parent_count: int,
    expected_candidate_count: int,
    expected_pair_count: int,
    theta_star: float,
    cost_cap: float,
    thresholds: Sequence[float],
    check_manifest: bool = True,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    required = (
        "selected_top20.csv",
        "test_pair_details.csv",
        "test_threshold_summary.csv",
        "parent_best_distances.csv",
        "prefix_metrics.csv",
        "prefix_metrics.json",
        "figure3_coverage_vs_k.csv",
        "figure4_coverage_vs_threshold.csv",
        "summary.json",
        "run_manifest.json",
        "official_summary_reconstruction_audit.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Final artifact files missing: {missing}")
    candidates, _ = load_ranked_candidates(
        root / "selected_top20.csv",
        expected_count=expected_candidate_count,
    )
    frozen, _ = load_ranked_candidates(
        frozen_candidates_csv,
        expected_count=expected_candidate_count,
    )
    if [row["candidate_id"] for row in candidates] != [
        row["candidate_id"] for row in frozen
    ]:
        raise ValueError("Exported candidate order differs from frozen rank order.")
    if any("native_rank" in row for row in frozen) and [
        _as_int(row.get("native_rank")) for row in candidates
    ] != [_as_int(row.get("native_rank")) for row in frozen]:
        raise ValueError("Exported candidate native_rank lineage changed.")
    details, _ = read_csv(root / "test_pair_details.csv")
    parent_ids, _ = validate_complete_cartesian(
        details,
        candidates,
        expected_parent_count=expected_parent_count,
        expected_pair_count=expected_pair_count,
    )
    prefix, _ = read_csv(root / "prefix_metrics.csv")
    if [_as_int(row.get("k")) for row in prefix] != list(
        range(1, expected_candidate_count + 1)
    ):
        raise ValueError("Figure 3/prefix metrics do not contain K=1..20.")
    previous_coverage = -math.inf
    previous_cost = math.inf
    for row in prefix:
        coverage = float(row["close_cf_coverage"])
        capped = float(row["fixed_capped_mean_cost"])
        if coverage + FLOAT_TOLERANCE < previous_coverage:
            raise ValueError("Prefix CCRCov decreases with K.")
        if capped > previous_cost + FLOAT_TOLERANCE:
            raise ValueError("Fixed capped mean cost increases with K.")
        previous_coverage = coverage
        previous_cost = capped
    summary = read_json(root / "summary.json")
    recomputed_prefix, recomputed_thresholds, _ = compute_prefix_artifacts(
        details=details,
        candidates=candidates,
        parent_ids=parent_ids,
        thresholds=thresholds,
        theta_star=theta_star,
        cost_cap=cost_cap,
        source_label=int(summary["source_label"]),
        target_label=int(summary["target_label"]),
        method_name=str(summary["method"]),
    )
    recomputed_by_k = {int(row["k"]): row for row in recomputed_prefix}
    metric_fields = (
        *OFFICIAL_FIELDS,
        "num_applicable_parents",
        "applicable_coverage",
        "num_any_strict_flip_parents",
        "any_strict_flip_coverage",
        "conditional_mean_cost",
        "conditional_median_cost",
        "fixed_capped_mean_cost",
        "fixed_capped_median_cost",
        "coverage_redundancy",
        "structural_redundancy",
    )
    for stored in prefix:
        k = int(stored["k"])
        recomputed = recomputed_by_k[k]
        mismatched = [
            field
            for field in metric_fields
            if not _same_value(stored.get(field), recomputed.get(field), field)
        ]
        if mismatched:
            raise ValueError(
                f"Prefix metric reconstruction failed for K={k}: {mismatched}"
            )
    figure4, _ = read_csv(root / "figure4_coverage_vs_threshold.csv")
    expected_figure4 = 2 * len(thresholds)
    if len(figure4) != expected_figure4:
        raise ValueError(
            f"Figure 4 rows={len(figure4)} != expected={expected_figure4}."
        )
    k_values = {_as_int(row.get("k")) for row in figure4}
    if k_values != {10, expected_candidate_count}:
        raise ValueError(f"Figure 4 K values are invalid: {k_values}")
    expected_figure4_rows = [
        row
        for row in recomputed_thresholds
        if int(row["k"]) in {10, expected_candidate_count}
    ]
    for stored, recomputed in zip(figure4, expected_figure4_rows):
        if int(stored["k"]) != int(recomputed["k"]) or not math.isclose(
            float(stored["threshold"]),
            float(recomputed["threshold"]),
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ):
            raise ValueError("Figure 4 row ordering or frozen threshold changed.")
        mismatched = [
            field
            for field in OFFICIAL_FIELDS
            if not _same_value(stored.get(field), recomputed.get(field), field)
        ]
        if mismatched:
            raise ValueError(
                f"Figure 4 metric reconstruction failed: {mismatched}"
            )
    reconstruction = read_json(root / "official_summary_reconstruction_audit.json")
    if not _as_bool(reconstruction.get("official_summary_reconstruction_passed")):
        raise ValueError("Official summary reconstruction did not pass.")
    if _as_bool(summary.get("selection_used_test")):
        raise ValueError("Final artifacts declare test candidate selection.")
    if _as_bool(summary.get("threshold_fitted_on_test")):
        raise ValueError("Final artifacts declare test threshold fitting.")
    if not math.isclose(
        float(summary["theta_star"]),
        float(theta_star),
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    ) or not math.isclose(
        float(summary["cost_cap"]),
        float(cost_cap),
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise ValueError("Frozen theta_star or cost_cap changed.")
    for k in (10, expected_candidate_count):
        table_candidates = list(root.glob(f"table2_*_k{k}.csv"))
        if len(table_candidates) != 1:
            raise ValueError(f"Expected one Table 2 artifact for K={k}.")
        _, fields = read_csv(table_candidates[0])
        reference_fields = _resolve_table_fields(Path(ours_schema_root), k)
        if fields != reference_fields:
            raise ValueError(f"Table 2 schema mismatch for K={k}.")
    manifest_verified = None
    if check_manifest:
        for required_name in (
            "selected_sequence.jsonl",
            "audit.json",
            "_RUN_COMPLETE.json",
        ):
            if not (root / required_name).is_file():
                raise ValueError(f"Final artifact file missing: {required_name}")
        selected_sequence = read_jsonl(root / "selected_sequence.jsonl")
        if [str(row.get("candidate_id")) for row in selected_sequence] != [
            str(row["candidate_id"]) for row in candidates
        ]:
            raise ValueError("selected_sequence.jsonl changed frozen candidate order.")
        complete = read_json(root / "_RUN_COMPLETE.json")
        if not _as_bool(complete.get("run_complete")) or not _as_bool(
            complete.get("audit_passed")
        ):
            raise ValueError("Final run completion marker did not pass audit.")
        finalized = read_json(root / "_FINALIZED.json")
        if not _as_bool(finalized.get("finalized")):
            raise ValueError("Run is not finalized.")
        artifact_manifest = read_json(root / "artifact_manifest.json")
        for relative, digest in artifact_manifest.get("files", {}).items():
            path = root / relative
            if not path.is_file() or sha256_file(path) != str(digest):
                raise ValueError(f"Artifact hash mismatch: {relative}")
        if sha256_file(root / "artifact_manifest.json") != str(
            finalized.get("artifact_manifest_sha256")
        ):
            raise ValueError("artifact_manifest.json hash mismatch.")
        run_manifest = read_json(root / "run_manifest.json")
        threshold_provenance = run_manifest.get("threshold_provenance") or {}
        threshold_source = Path(
            str(threshold_provenance.get("ours_thresholds_json") or "")
        ).expanduser()
        if (
            not threshold_source.is_file()
            or sha256_file(threshold_source)
            != str(threshold_provenance.get("ours_thresholds_json_sha256"))
        ):
            raise ValueError("Frozen threshold provenance hash mismatch.")
        for path_field, hash_field in (
            ("candidate_csv", "candidate_csv_sha256"),
            ("pair_details", "pair_details_sha256"),
            (
                "official_threshold_summary",
                "official_threshold_summary_sha256",
            ),
        ):
            source_path = Path(str(run_manifest[path_field])).expanduser()
            if source_path.is_file() and sha256_file(source_path) != str(
                run_manifest[hash_field]
            ):
                raise ValueError(f"Source provenance hash mismatch: {path_field}")
        official_source = Path(
            str(run_manifest["official_threshold_summary"])
        ).expanduser()
        if official_source.is_file():
            official_rows, _ = read_csv(official_source)
            exported_k20, _ = read_csv(root / "test_threshold_summary.csv")
            reconstruct_official_summary(
                recomputed_k20=exported_k20,
                official_rows=official_rows,
                thresholds=thresholds,
                theta_star=theta_star,
                expected_theta_star_covered=(
                    69
                    if str(summary["method"])
                    == "GlobalGCE-Frequency-Top20"
                    and int(expected_parent_count) == 217
                    else None
                ),
                recomputed_theta_star_row=recomputed_by_k[
                    int(expected_candidate_count)
                ],
            )
        manifest_verified = True
    return {
        "final_artifact_audit_passed": True,
        "parent_count": len(parent_ids),
        "candidate_count": len(candidates),
        "pair_count": len(details),
        "complete_cartesian": True,
        "candidate_order_frozen": True,
        "top10_is_rank_1_to_10": True,
        "top20_is_rank_1_to_20": True,
        "coverage_monotonic_nondecreasing": True,
        "fixed_capped_cost_monotonic_nonincreasing": True,
        "prefix_metrics_recomputed": True,
        "figure4_metrics_recomputed": True,
        "test_selection": False,
        "test_threshold_fitting": False,
        "candidate_selection_performed": False,
        "candidate_set_preselected": True,
        "selection_performed_in_eval": False,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "selected_candidate_order_sha256": stable_json_sha256(
            [str(row["candidate_id"]) for row in candidates]
        ),
        "manifest_hashes_verified": manifest_verified,
    }


__all__ = [
    "FLOAT_TOLERANCE",
    "OFFICIAL_FIELDS",
    "TABLE_REQUIRED_FIELDS",
    "audit_final_artifacts",
    "audit_fullgraph_evaluation_run",
    "compute_prefix_artifacts",
    "export_final_artifacts",
    "finalize_fullgraph_evaluation_run",
    "load_frozen_threshold_contract",
    "load_ranked_candidates",
    "locate_test_inputs",
    "read_jsonl",
    "reconstruct_official_summary",
    "sha256_file",
    "stable_json_sha256",
    "summarize_wnode_thresholds",
    "validate_frozen_candidate_contract",
    "validate_frozen_threshold_provenance",
    "validate_complete_cartesian",
]
