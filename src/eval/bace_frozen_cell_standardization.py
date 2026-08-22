"""Deterministically standardize frozen BACE terminal artifacts.

This module is deliberately downstream of the scientific freeze.  It follows
only SHA256-pinned manifest/artifact identities, verifies the already-frozen
test metrics, and exports the common four-by-four cell schema.  It never opens
the raw held-out test CSV, changes candidate order, fits a threshold, or runs a
classifier/distance model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.baselines.bace_gnn_baseline_contracts import baseline_spec
from src.chem.hard_deletion import (
    CONNECTED_ACTION_SEMANTICS,
    CONNECTED_MATCH_SELECTION_POLICY,
)
from src.eval.bace_frozen_gnn_contracts import stable_sha256


DATASET = "BACE"
ORACLE_BACKEND = "gnn"
CLASSIFIER_FAMILY = "gine"
SOURCE_LABEL = 1
NUM_CLASSES = 2
CF_MODE = "strict_flip"
DISTANCE_LINE = "MolCLR-Node-Wasserstein"
TABLE_K = 10
MAX_K = 20
METHODS = {
    "ours": "Ours",
    "gcfexplainer": "GCFExplainer",
    "comrecgc": "ComRecGC",
}
METHOD_SLUGS = {value: key for key, value in METHODS.items()}
SCHEMA_VERSION = "bace_frozen_cell_standardization_v1"
PASS_MARKER = "[BACE_FROZEN_CELL_STANDARDIZATION_PASS]"
HEX64 = re.compile(r"[0-9a-f]{64}")


class BACECellStandardizationError(ValueError):
    """Frozen BACE evidence cannot be promoted without weakening a gate."""


@dataclass(frozen=True)
class FrozenInputs:
    method: str
    method_slug: str
    source_root: Path
    final_manifest_path: Path
    final_manifest: Mapping[str, Any]
    selection_manifest_path: Path
    selection_manifest: Mapping[str, Any]
    test_manifest_path: Path
    test_manifest: Mapping[str, Any]
    test_merge_manifest_path: Path
    test_merge_manifest: Mapping[str, Any]
    pair_matrix_path: Path
    final_metrics_path: Path
    final_metrics: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise BACECellStandardizationError(f"Expected one physical JSON file: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BACECellStandardizationError(f"Invalid JSON artifact {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise BACECellStandardizationError(f"Expected one JSON object: {source}")
    return dict(value)


def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise BACECellStandardizationError(f"Expected one physical JSONL file: {source}")
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise BACECellStandardizationError(
                    f"Invalid JSONL artifact {source}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise BACECellStandardizationError(
                    f"Expected JSON object at {source}:{line_number}"
                )
            rows.append(dict(value))
    return rows


def _valid_sha(value: Any, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if HEX64.fullmatch(text) is None:
        raise BACECellStandardizationError(f"{label} is not one SHA256: {value!r}")
    return text


def _verify_identity(identity: Any, *, label: str) -> Path:
    if not isinstance(identity, Mapping):
        raise BACECellStandardizationError(f"{label} identity is missing")
    raw_path = str(identity.get("path") or "").strip()
    if not raw_path or not Path(raw_path).expanduser().is_absolute():
        raise BACECellStandardizationError(f"{label} identity path must be absolute")
    path = Path(raw_path).expanduser().resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise BACECellStandardizationError(f"{label} identity is not physical: {path}")
    try:
        expected_size = int(identity.get("size", identity.get("bytes", -1)))
    except (TypeError, ValueError):
        expected_size = -1
    if expected_size != path.stat().st_size:
        raise BACECellStandardizationError(f"{label} identity size changed: {path}")
    expected_sha = _valid_sha(identity.get("sha256"), label=f"{label}.sha256")
    if sha256_file(path) != expected_sha:
        raise BACECellStandardizationError(f"{label} identity SHA256 changed: {path}")
    return path


def _declared_split_identity(identity: Any, *, label: str) -> tuple[str, str, int]:
    """Validate split metadata without opening or statting the raw test CSV."""

    if not isinstance(identity, Mapping):
        raise BACECellStandardizationError(f"{label} split identity is missing")
    path = str(identity.get("path") or "").strip()
    if not path or not Path(path).expanduser().is_absolute():
        raise BACECellStandardizationError(f"{label} split path must be absolute")
    digest = _valid_sha(identity.get("sha256"), label=f"{label}.sha256")
    try:
        size = int(identity.get("size", identity.get("bytes", -1)))
    except (TypeError, ValueError) as exc:
        raise BACECellStandardizationError(f"{label} split size is invalid") from exc
    if size < 0:
        raise BACECellStandardizationError(f"{label} split size is missing")
    return str(Path(path).expanduser()), digest, size


def _method_slug(method: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "", str(method).strip().lower())
    aliases = {
        "ours": "ours",
        "gcfexplainer": "gcfexplainer",
        "gcf": "gcfexplainer",
        "comrecgc": "comrecgc",
    }
    try:
        return aliases[token]
    except KeyError as exc:
        raise BACECellStandardizationError(
            f"Only frozen Ours/GCFExplainer/ComRecGC terminals are supported: {method!r}"
        ) from exc


def _assert_clean_gine(payload: Mapping[str, Any], *, label: str, checkpoint_id: str) -> None:
    backend = str(payload.get("oracle_backend") or "").strip().lower()
    classifier = str(
        payload.get("classifier_family") or payload.get("classifier_type") or ""
    ).strip().lower()
    observed_checkpoint = str(
        payload.get("oracle_checkpoint_hash") or payload.get("checkpoint_id") or ""
    ).strip().lower()
    failures: list[str] = []
    if backend != ORACLE_BACKEND:
        failures.append(f"oracle_backend={backend!r}")
    if classifier not in {CLASSIFIER_FAMILY, "gnn"}:
        failures.append(f"classifier={classifier!r}")
    if payload.get("rf_oracle_used") is not False:
        failures.append(f"rf_oracle_used={payload.get('rf_oracle_used')!r}")
    if observed_checkpoint != checkpoint_id:
        failures.append("oracle_checkpoint_hash_changed")
    if failures:
        raise BACECellStandardizationError(
            f"{label} is not the frozen RF-free BACE GINE contract: "
            + ", ".join(failures)
        )


def _checkpoint_contract(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint_dir).expanduser().resolve(strict=True)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise BACECellStandardizationError(
            f"GINE checkpoint must be one physical bundle directory: {checkpoint}"
        )
    card = _json(checkpoint / "model_card.json")
    split = _json(checkpoint / "split_manifest.json")
    test_status = _json(checkpoint / "test_evaluation_status.json")
    if str(card.get("dataset") or "").strip().lower() != "bace":
        raise BACECellStandardizationError("GINE model card dataset is not BACE")
    if card.get("backbone") != CLASSIFIER_FAMILY or card.get("num_classes") != NUM_CLASSES:
        raise BACECellStandardizationError("GINE model card backbone/class count changed")
    if card.get("source_label") != SOURCE_LABEL:
        raise BACECellStandardizationError("GINE model card source label changed")
    checkpoint_id = _valid_sha(card.get("checkpoint_id"), label="model_card.checkpoint_id")
    _assert_clean_gine(card, label="model_card", checkpoint_id=checkpoint_id)

    sums: dict[str, str] = {}
    for line in (checkpoint / "sha256sums.txt").read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or Path(relative).name != relative:
            raise BACECellStandardizationError("Malformed GINE sha256sums.txt")
        sums[relative] = _valid_sha(digest, label=f"sha256sums[{relative}]")
    if sums.get("model.pt") != checkpoint_id or sha256_file(checkpoint / "model.pt") != checkpoint_id:
        raise BACECellStandardizationError("GINE model.pt no longer matches checkpoint_id")

    if str(split.get("dataset") or "").strip().lower() != "bace":
        raise BACECellStandardizationError("GINE split manifest dataset is not BACE")
    test_file = (split.get("files") or {}).get("test")
    if not isinstance(test_file, Mapping):
        raise BACECellStandardizationError("GINE split manifest lacks held-out test identity")
    test_path = str(test_file.get("path") or "").strip()
    test_hash = _valid_sha(test_file.get("sha256"), label="split_manifest.test.sha256")
    if (
        test_status.get("status") != "NOT_EVALUATED"
        or test_status.get("test_loaded") is not False
        or str(test_status.get("path") or "").strip() != test_path
        or str(test_status.get("sha256") or "").strip().lower() != test_hash
    ):
        raise BACECellStandardizationError(
            "GINE training bundle does not preserve its held-out test boundary"
        )
    return {
        "checkpoint": checkpoint,
        "checkpoint_id": checkpoint_id,
        "dataset_hash": sha256_file(checkpoint / "split_manifest.json"),
        "test_split_path": test_path,
        "test_split_hash": test_hash,
        "temperature_hash": sha256_file(checkpoint / "temperature_scaling.json"),
        "feature_schema_hash": sha256_file(checkpoint / "feature_schema.json"),
    }


def _require_fields(payload: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    failures = [
        f"{key}={payload.get(key)!r}"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    if failures:
        raise BACECellStandardizationError(
            f"{label} frozen contract differs: " + ", ".join(failures)
        )


def _load_frozen_inputs(
    *, method_slug: str, source_root: Path, checkpoint_id: str
) -> FrozenInputs:
    final_path = source_root / "FINAL_PASS.json"
    final = _json(final_path)
    common = {
        "dataset": "bace",
        "status": "PASS",
        "oracle_backend": ORACLE_BACKEND,
        "rf_oracle_used": False,
        "source_label": SOURCE_LABEL,
        "cf_mode": CF_MODE,
        "oracle_checkpoint_hash": checkpoint_id,
        "selector_fitted_on_calibration": True,
        "test_used_only_after_freeze": True,
    }
    if method_slug == "ours":
        _require_fields(
            final,
            {**common, "stage": "B14_FROZEN", "classifier_type": "gnn"},
            label="BACE Ours B14 FINAL_PASS",
        )
        if final.get("final_gate_pass") is not True or final.get("all_hashes_frozen") is not True:
            raise BACECellStandardizationError("BACE Ours B14 final gate is not frozen PASS")
        selection_path = _verify_identity(final.get("b12_manifest_identity"), label="B14 B12")
        test_path = _verify_identity(final.get("b13_manifest_identity"), label="B14 B13")
        metrics_path = _verify_identity(final.get("final_metrics_identity"), label="B14 metrics")
        selection = _json(selection_path)
        test = _json(test_path)
        metrics = _json(metrics_path)
        _require_fields(
            selection,
            {
                "dataset": "bace",
                "stage": "B12_SELECTOR",
                "status": "FROZEN",
                "selection_frozen": True,
                "selector_fitted_on_calibration": True,
                "test_loaded": False,
                "test_used": False,
                "oracle_backend": ORACLE_BACKEND,
                "rf_oracle_used": False,
                "cf_mode": CF_MODE,
                "oracle_checkpoint_hash": checkpoint_id,
            },
            label="BACE Ours B12 selection",
        )
        _require_fields(
            test,
            {
                "dataset": "bace",
                "stage": "B13_FINAL_EVAL",
                "status": "PASS",
                "selection_frozen_before_test": True,
                "test_used_only_after_freeze": True,
                "selector_refit_on_test": False,
                "threshold_refit_on_test": False,
                "oracle_backend": ORACLE_BACKEND,
                "rf_oracle_used": False,
                "cf_mode": CF_MODE,
                "oracle_checkpoint_hash": checkpoint_id,
            },
            label="BACE Ours B13 manifest",
        )
        b13_selection_path = _verify_identity(
            test.get("frozen_selection_manifest_identity"), label="B13 B12"
        )
        if b13_selection_path != selection_path:
            raise BACECellStandardizationError("B13 does not bind the B14-pinned B12 manifest")
        merge_path = _verify_identity(
            test.get("verification_manifest_identity"), label="B13 merge"
        )
        pair_path = _verify_identity(test.get("pair_matrix_identity"), label="B13 pair matrix")
    else:
        spec = baseline_spec(method_slug)
        _require_fields(
            final,
            {
                **common,
                "method": spec.method,
                "method_id": method_slug,
                "stage": "BASELINE_FINAL_FREEZE",
                "classifier_family": CLASSIFIER_FAMILY,
                "selection_frozen_before_test": True,
                "run_complete": True,
                "action_kind": spec.action_kind,
                "action_semantics": spec.action_semantics,
            },
            label=f"BACE {spec.method} final freeze",
        )
        if final.get("all_hashes_frozen") is not True:
            raise BACECellStandardizationError(f"BACE {spec.method} final hashes are not frozen")
        selection_path = _verify_identity(
            final.get("selection_manifest_identity"), label=f"{spec.method} selection"
        )
        merge_path = _verify_identity(
            final.get("test_manifest_identity"), label=f"{spec.method} test merge"
        )
        pair_path = _verify_identity(
            final.get("test_pair_matrix_identity"), label=f"{spec.method} pair matrix"
        )
        metrics_path = _verify_identity(
            final.get("final_metrics_identity"), label=f"{spec.method} metrics"
        )
        selection = _json(selection_path)
        test = _json(merge_path)
        metrics = _json(metrics_path)
        _require_fields(
            selection,
            {
                "dataset": "bace",
                "method": spec.method,
                "method_id": method_slug,
                "stage": "BASELINE_CALIBRATION_SELECTOR",
                "status": "FROZEN",
                "selection_frozen": True,
                "selector_fitted_on_calibration": True,
                "test_loaded": False,
                "oracle_backend": ORACLE_BACKEND,
                "rf_oracle_used": False,
                "cf_mode": CF_MODE,
                "oracle_checkpoint_hash": checkpoint_id,
            },
            label=f"BACE {spec.method} selection",
        )
        _require_fields(
            test,
            {
                "dataset": "bace",
                "method": spec.method,
                "method_id": method_slug,
                "stage": "BASELINE_TEST_EVAL",
                "status": "PASS",
                "selection_frozen_before_test": True,
                "test_loaded": True,
                "oracle_backend": ORACLE_BACKEND,
                "rf_oracle_used": False,
                "cf_mode": CF_MODE,
                "oracle_checkpoint_hash": checkpoint_id,
            },
            label=f"BACE {spec.method} test merge",
        )
        test_path = merge_path

    if str(selection.get("molclr_checkpoint_hash") or "") != str(
        test.get("molclr_checkpoint_hash") or final.get("molclr_checkpoint_hash") or ""
    ):
        raise BACECellStandardizationError("BACE selection/test MolCLR identity changed")
    _assert_clean_gine(final, label="terminal final", checkpoint_id=checkpoint_id)
    _assert_clean_gine(selection, label="frozen selection", checkpoint_id=checkpoint_id)
    _assert_clean_gine(test, label="held-out test manifest", checkpoint_id=checkpoint_id)
    return FrozenInputs(
        method=METHODS[method_slug],
        method_slug=method_slug,
        source_root=source_root,
        final_manifest_path=final_path,
        final_manifest=final,
        selection_manifest_path=selection_path,
        selection_manifest=selection,
        test_manifest_path=test_path,
        test_manifest=test,
        test_merge_manifest_path=merge_path,
        test_merge_manifest=_json(merge_path),
        pair_matrix_path=pair_path,
        final_metrics_path=metrics_path,
        final_metrics=metrics,
    )


def _split_contract(inputs: FrozenInputs, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    manifests = (inputs.test_merge_manifest.get("inputs") or {}).get("shard_manifests")
    if not isinstance(manifests, list) or len(manifests) != 4:
        raise BACECellStandardizationError("Frozen test merge must bind exactly four shards")
    all_parent_hashes: set[str] = set()
    split_paths: set[str] = set()
    split_hashes: set[str] = set()
    split_sizes: set[int] = set()
    shard_ids: list[dict[str, Any]] = []
    for index, identity in enumerate(manifests):
        path = _verify_identity(identity, label=f"test shard {index}")
        shard = _json(path)
        _require_fields(
            shard,
            {
                "dataset": "bace",
                "status": "PASS",
                "test_loaded": True,
                "selection_frozen_before_test": True,
                "oracle_backend": ORACLE_BACKEND,
                "rf_oracle_used": False,
                "cf_mode": CF_MODE,
                "oracle_checkpoint_hash": checkpoint["checkpoint_id"],
            },
            label=f"test shard {index}",
        )
        all_parent_hashes.add(
            _valid_sha(shard.get("all_parent_ids_sha256"), label="all_parent_ids_sha256")
        )
        split_path, split_hash, split_size = _declared_split_identity(
            shard.get("split_identity"), label=f"test shard {index}"
        )
        split_paths.add(split_path)
        split_hashes.add(split_hash)
        split_sizes.add(split_size)
        shard_ids.append(
            {
                "manifest_path": str(path),
                "manifest_sha256": sha256_file(path),
                "shard_index": shard.get("shard_index"),
            }
        )
    if len(all_parent_hashes) != 1 or len(split_paths) != 1 or len(split_hashes) != 1 or len(split_sizes) != 1:
        raise BACECellStandardizationError("Frozen test shard split/cohort identities differ")
    declared_path = next(iter(split_paths))
    if Path(declared_path).expanduser() != Path(str(checkpoint["test_split_path"])).expanduser():
        raise BACECellStandardizationError("Frozen test shards use a different test split path")
    if next(iter(split_hashes)) != checkpoint["test_split_hash"]:
        raise BACECellStandardizationError("Frozen test shards use a different test split hash")
    return {
        "test_parent_set_sha256": next(iter(all_parent_hashes)),
        "test_split_path": declared_path,
        "test_split_hash": next(iter(split_hashes)),
        "test_split_size": next(iter(split_sizes)),
        "shard_manifests": shard_ids,
        "raw_test_opened": False,
    }


def _threshold_contract(selection: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = selection.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise BACECellStandardizationError("Frozen selection lacks a threshold payload")
    if thresholds.get("test_used") is not False:
        raise BACECellStandardizationError("Frozen thresholds do not prove test_used=false")
    source = str(thresholds.get("threshold_source") or "")
    if "calibration" not in source:
        raise BACECellStandardizationError("Frozen thresholds are not calibration-derived")
    levels = thresholds.get("merged_thresholds")
    if not isinstance(levels, list) or not levels:
        raise BACECellStandardizationError("Frozen threshold grid is empty")
    values: list[float] = []
    for row in levels:
        if not isinstance(row, Mapping):
            raise BACECellStandardizationError("Frozen threshold row is not an object")
        try:
            value = float(row["threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BACECellStandardizationError("Frozen threshold value is invalid") from exc
        if not math.isfinite(value) or value < 0.0:
            raise BACECellStandardizationError("Frozen threshold is non-finite/negative")
        values.append(value)
    if any(right <= left for left, right in zip(values, values[1:])):
        raise BACECellStandardizationError("Frozen merged threshold grid is not strictly increasing")
    try:
        theta_star = float(thresholds["theta_star"])
        cost_cap = float(thresholds["cost_cap"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BACECellStandardizationError("Frozen theta_star/cost_cap is invalid") from exc
    if (
        not math.isfinite(theta_star)
        or not math.isfinite(cost_cap)
        or theta_star < 0.0
        or cost_cap < theta_star
        or all(not math.isclose(theta_star, value, rel_tol=0.0, abs_tol=1e-12) for value in values)
    ):
        raise BACECellStandardizationError("Frozen theta_star/cost_cap escaped threshold grid")
    return {
        "values": values,
        "theta_star": theta_star,
        "cost_cap": cost_cap,
        # This is exactly the registry's numeric grid identity.
        "threshold_config_hash": stable_sha256(values),
        "threshold_payload_hash": stable_sha256(dict(thresholds)),
        "threshold_source": source,
    }


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _close(actual: float, expected: Any, *, label: str) -> None:
    try:
        value = float(expected)
    except (TypeError, ValueError) as exc:
        raise BACECellStandardizationError(f"Frozen {label} is missing") from exc
    if not math.isclose(actual, value, rel_tol=0.0, abs_tol=1e-12):
        raise BACECellStandardizationError(
            f"Frozen {label} differs from artifact-only replay: {value} != {actual}"
        )


def _compute_metrics(
    inputs: FrozenInputs,
    *,
    checkpoint_id: str,
    thresholds: Mapping[str, Any],
    test_parent_set_sha256: str,
) -> dict[str, Any]:
    rows = _jsonl(inputs.pair_matrix_path)
    ordered = [str(value) for value in inputs.selection_manifest.get("ordered_rule_ids", [])]
    if len(ordered) != MAX_K or len(set(ordered)) != MAX_K or any(not value for value in ordered):
        raise BACECellStandardizationError("Frozen selector must bind 20 unique ordered rules")
    metric_order = [str(value) for value in inputs.final_metrics.get("ordered_rule_ids", ordered)]
    if metric_order != ordered:
        raise BACECellStandardizationError("Frozen final metrics changed candidate order")
    by_parent: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("dataset") or "").strip().lower() != "bace":
            raise BACECellStandardizationError("Pair matrix dataset escaped BACE")
        if row.get("rf_oracle_used") is not False or row.get("oracle_backend") != ORACLE_BACKEND:
            raise BACECellStandardizationError("Pair matrix contains non-GINE/RF provenance")
        if str(row.get("oracle_checkpoint_hash") or "") != checkpoint_id:
            raise BACECellStandardizationError("Pair matrix oracle hash changed")
        if str(row.get("cf_mode") or "").lower() != CF_MODE:
            raise BACECellStandardizationError("Pair matrix CF mode changed")
        parent = str(row.get("parent_id") or "")
        candidate = str(row.get("candidate_id") or "")
        if not parent or candidate not in ordered:
            raise BACECellStandardizationError("Pair matrix escaped frozen parent/rule identity")
        if candidate in by_parent.setdefault(parent, {}):
            raise BACECellStandardizationError("Pair matrix repeats one parent/rule pair")
        strict = row.get("pair_strict_flip") is True
        distance = _float_or_none(row.get("wnode_distance"))
        if strict:
            if distance is None or distance < 0.0:
                raise BACECellStandardizationError("Strict-flip pair lacks finite WNode")
            if int(row.get("pred_before", -1)) != SOURCE_LABEL or int(row.get("pred_after", -1)) != 0:
                raise BACECellStandardizationError("BACE strict flip is not label 1 -> 0")
        elif distance is not None:
            raise BACECellStandardizationError("Non-strict pair unexpectedly carries WNode")
        if inputs.method_slug == "ours":
            if row.get("action_semantics_version") != CONNECTED_ACTION_SEMANTICS:
                raise BACECellStandardizationError("Ours action semantics changed after B14")
            if row.get("match_selection_policy") != CONNECTED_MATCH_SELECTION_POLICY:
                raise BACECellStandardizationError("Ours match-selection policy changed")
        by_parent[parent][candidate] = dict(row)
    if not by_parent or any(set(values) != set(ordered) for values in by_parent.values()):
        raise BACECellStandardizationError("Frozen pair matrix is not the full Cartesian product")
    parents = sorted(by_parent)
    if stable_sha256(parents) != test_parent_set_sha256:
        raise BACECellStandardizationError("Pair matrix parent set differs from frozen shards")
    if int(inputs.final_metrics.get("parent_count", -1)) != len(parents):
        raise BACECellStandardizationError("Frozen final parent count changed")

    theta_star = float(thresholds["theta_star"])
    cost_cap = float(thresholds["cost_cap"])
    prefix: list[dict[str, Any]] = []
    parent_best: list[dict[str, Any]] = []
    best_by_parent: dict[str, tuple[float, float, str, int | None, bool] | None] = {
        parent: None for parent in parents
    }
    applicable = {parent: False for parent in parents}
    for k, candidate in enumerate(ordered, start=1):
        for parent in parents:
            row = by_parent[parent][candidate]
            applicable[parent] = applicable[parent] or bool(row.get("applicable", False))
            if row.get("pair_strict_flip") is not True:
                continue
            distance = float(row["wnode_distance"])
            cf_drop = _float_or_none(row.get("cf_drop"))
            candidate_value = (
                distance,
                -(cf_drop if cf_drop is not None else -math.inf),
                candidate,
                int(row["pred_after"]),
                bool(row.get("applicable", False)),
            )
            current = best_by_parent[parent]
            if current is None or candidate_value[:3] < current[:3]:
                best_by_parent[parent] = candidate_value
        finite = [value for value in best_by_parent.values() if value is not None]
        covered = [value for value in finite if value[0] <= theta_star]
        capped = [
            min(value[0], cost_cap) if value is not None else cost_cap
            for value in best_by_parent.values()
        ]
        conditional = [value[0] for value in finite]
        covered_cf = [
            -value[1]
            for value in covered
            if math.isfinite(-value[1])
        ]
        row = {
            "dataset": DATASET,
            "method": inputs.method,
            "k": k,
            "SuppCov": len(finite) / len(parents),
            "CCRCov": len(covered) / len(parents),
            "coverage": len(covered) / len(parents),
            "cost": sum(capped) / len(capped),
            "fixed_capped_mean_cost": sum(capped) / len(capped),
            "conditional_mean_cost": (
                sum(conditional) / len(conditional) if conditional else "N/A"
            ),
            "conditional_median_cost": (
                sorted(conditional)[len(conditional) // 2]
                if len(conditional) % 2 == 1
                else (
                    (sorted(conditional)[len(conditional) // 2 - 1] + sorted(conditional)[len(conditional) // 2]) / 2
                    if conditional
                    else "N/A"
                )
            ),
            "CFDrop": sum(covered_cf) / len(covered_cf) if covered_cf else "N/A",
            "FlipRate": len(finite) / len(parents),
            "StructRed": "N/A",
            "CovRed": "N/A",
            "ValidRate": "N/A",
            "AvgSize": "N/A",
            "applicable_rate": sum(applicable.values()) / len(parents),
            "unavailable_metric_reason": (
                "StructRed/CovRed/ValidRate/AvgSize are not present in the frozen "
                "test/prefix/final artifact chain and were not recomputed"
            ),
        }
        prefix.append(row)
        for parent in parents:
            value = best_by_parent[parent]
            parent_best.append(
                {
                    "dataset": DATASET,
                    "method": inputs.method,
                    "k": k,
                    "parent_id": parent,
                    "best_distance": value[0] if value is not None else "N/A",
                    "capped_distance": min(value[0], cost_cap) if value is not None else cost_cap,
                    "best_candidate_id": value[2] if value is not None else "N/A",
                    "destination_label": value[3] if value is not None else "N/A",
                    "strict_recourse_available": value is not None,
                    "theta_star_covered": value is not None and value[0] <= theta_star,
                    "applicable": applicable[parent],
                }
            )

    frozen_rows = inputs.final_metrics.get("prefix_metrics")
    if not isinstance(frozen_rows, list) or len(frozen_rows) != MAX_K:
        raise BACECellStandardizationError("Frozen final metrics lack K=1..20 prefix rows")
    if inputs.method_slug == "ours":
        frozen_curve = inputs.final_metrics.get("ccrcov_theta_star_by_k")
        if not isinstance(frozen_curve, list) or len(frozen_curve) != MAX_K:
            raise BACECellStandardizationError("B13 final metrics lack theta-star curve")
        for index, row in enumerate(prefix):
            _close(float(row["CCRCov"]), frozen_curve[index], label=f"CCRCov K={index + 1}")
            _close(
                float(row["SuppCov"]),
                frozen_rows[index].get("strict_flip_any_rate"),
                label=f"SuppCov K={index + 1}",
            )
    else:
        for index, row in enumerate(prefix):
            recorded = frozen_rows[index]
            if int(recorded.get("K", -1)) != index + 1:
                raise BACECellStandardizationError("Frozen baseline prefix order changed")
            _close(float(row["CCRCov"]), recorded.get("CCRCov"), label=f"CCRCov K={index + 1}")
            _close(float(row["SuppCov"]), recorded.get("SuppCov"), label=f"SuppCov K={index + 1}")
            conditional = row["conditional_mean_cost"]
            if conditional != "N/A":
                _close(float(conditional), recorded.get("avg_cost"), label=f"avg_cost K={index + 1}")

    k10_best = {
        row["parent_id"]: row
        for row in parent_best
        if int(row["k"]) == TABLE_K
    }
    figure4 = []
    for threshold in thresholds["values"]:
        coverage = sum(
            row["best_distance"] != "N/A" and float(row["best_distance"]) <= threshold
            for row in k10_best.values()
        ) / len(k10_best)
        figure4.append(
            {
                "dataset": DATASET,
                "method": inputs.method,
                "k": TABLE_K,
                "threshold": threshold,
                "coverage": coverage,
                "CCRCov": coverage,
            }
        )
    k20_values = [value for value in best_by_parent.values() if value is not None]
    destination_counts = {0: sum(value[3] == 0 for value in k20_values)}
    destination = [
        {
            "dataset": DATASET,
            "method": inputs.method,
            "destination_label": 0,
            "count": destination_counts[0],
            "rate": destination_counts[0] / len(k20_values) if k20_values else "N/A",
            "denominator": len(k20_values),
            "distribution_scope": "K20 finite untargeted strict flips",
        }
    ]
    return {
        "prefix": prefix,
        "parent_best": parent_best,
        "figure3": [
            {
                "dataset": DATASET,
                "method": inputs.method,
                "k": row["k"],
                "coverage": row["CCRCov"],
                "cost": row["cost"],
            }
            for row in prefix
        ],
        "figure4": figure4,
        "table2": [dict(prefix[TABLE_K - 1])],
        "destination": destination,
        "parent_count": len(parents),
        "pair_count": len(rows),
    }


def _atomic_text(path: Path, text: str) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise BACECellStandardizationError(f"Cannot write empty standardized CSV: {path.name}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _expected_hash(actual: str, expected: str | None, *, label: str) -> None:
    if expected is None:
        return
    wanted = _valid_sha(expected, label=f"expected {label}")
    if actual != wanted:
        raise BACECellStandardizationError(f"{label} differs from preregistered expectation")


def standardize_bace_frozen_cell(
    *,
    method: str,
    source_final_root: str | Path,
    gnn_checkpoint: str | Path,
    output_dir: str | Path,
    expected_dataset_hash: str | None = None,
    expected_split_hash: str | None = None,
    expected_molclr_hash: str | None = None,
    expected_threshold_hash: str | None = None,
) -> dict[str, Any]:
    """Export one frozen BACE cell without reopening raw held-out data."""

    method_slug = _method_slug(method)
    source = Path(source_final_root).expanduser().resolve(strict=True)
    if source.is_symlink() or not source.is_dir():
        raise BACECellStandardizationError(f"Frozen source root is not physical: {source}")
    destination = Path(output_dir).expanduser()
    if not destination.is_absolute():
        raise BACECellStandardizationError("Standardized output root must be absolute")
    destination = destination.resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Standardized output root must be fresh: {destination}")

    checkpoint = _checkpoint_contract(gnn_checkpoint)
    inputs = _load_frozen_inputs(
        method_slug=method_slug,
        source_root=source,
        checkpoint_id=str(checkpoint["checkpoint_id"]),
    )
    split = _split_contract(inputs, checkpoint)
    thresholds = _threshold_contract(inputs.selection_manifest)
    molclr_hash = _valid_sha(
        inputs.selection_manifest.get("molclr_checkpoint_hash"),
        label="molclr_checkpoint_hash",
    )
    _expected_hash(str(checkpoint["dataset_hash"]), expected_dataset_hash, label="dataset_hash")
    _expected_hash(str(split["test_split_hash"]), expected_split_hash, label="split_hash")
    _expected_hash(molclr_hash, expected_molclr_hash, label="molclr_checkpoint_hash")
    _expected_hash(
        str(thresholds["threshold_config_hash"]),
        expected_threshold_hash,
        label="threshold_config_hash",
    )
    metrics = _compute_metrics(
        inputs,
        checkpoint_id=str(checkpoint["checkpoint_id"]),
        thresholds=thresholds,
        test_parent_set_sha256=str(split["test_parent_set_sha256"]),
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        table_name = f"table2_{method_slug}_k10.csv"
        _write_csv(temporary / "prefix_metrics.csv", metrics["prefix"])
        _write_json(
            temporary / "prefix_metrics.json",
            {
                "schema_version": SCHEMA_VERSION,
                "dataset": DATASET,
                "method": inputs.method,
                "prefix_metrics": metrics["prefix"],
                "scientific_metrics_recomputed": False,
                "artifact_only_replay": True,
            },
        )
        _write_csv(temporary / "parent_best_distances.csv", metrics["parent_best"])
        _write_csv(temporary / "destination_distribution.csv", metrics["destination"])
        _write_csv(temporary / "figure3_coverage_vs_k.csv", metrics["figure3"])
        _write_csv(temporary / "figure4_coverage_vs_threshold.csv", metrics["figure4"])
        _write_csv(temporary / table_name, metrics["table2"])

        identities = {
            "dataset": DATASET,
            "method": inputs.method,
            "stage": "BACE_FROZEN_CELL_STANDARDIZATION",
            "status": "PASS",
            "frozen": True,
            "finalized": True,
            "oracle_backend": ORACLE_BACKEND,
            "classifier_family": CLASSIFIER_FAMILY,
            "rf_oracle_used": False,
            "source_label": SOURCE_LABEL,
            "num_classes": NUM_CLASSES,
            "cf_mode": CF_MODE,
            "distance_line": DISTANCE_LINE,
            "oracle_checkpoint": str(checkpoint["checkpoint"]),
            "oracle_checkpoint_hash": checkpoint["checkpoint_id"],
            "dataset_hash": checkpoint["dataset_hash"],
            "split_hash": split["test_split_hash"],
            "molclr_checkpoint_hash": molclr_hash,
            "threshold_config_hash": thresholds["threshold_config_hash"],
            "test_used_for_selection": False,
            "threshold_fitted_on_test": False,
            "selector_fitted_on_calibration": True,
            "selection_frozen_before_test": True,
            "test_loaded_only_after_freeze": True,
            "test_used_only_after_freeze": True,
            "raw_test_opened": False,
            "candidate_order_changed": False,
            "selector_refit": False,
            "threshold_refit": False,
            "raw_output_root": str(source),
            "raw_output_complete": True,
            "source_artifacts_complete": True,
        }
        summary = {
            "schema_version": SCHEMA_VERSION,
            **identities,
            "test_parent_count": metrics["parent_count"],
            "pair_count": metrics["pair_count"],
            "k_max": MAX_K,
            "table2_k": TABLE_K,
            "theta_star": thresholds["theta_star"],
            "cost_cap": thresholds["cost_cap"],
            "thresholds": thresholds["values"],
            "k10": metrics["table2"][0],
            "k20": metrics["prefix"][-1],
            "unavailable_metrics": {
                "StructRed": "not present in frozen terminal artifacts",
                "CovRed": "not present in frozen terminal artifacts",
                "ValidRate": "not present in frozen terminal artifacts",
                "AvgSize": "not present in frozen terminal artifacts",
            },
        }
        oracle_manifest = {
            "schema_version": "bace_frozen_cell_oracle_manifest_v1",
            **identities,
            "temperature_scaling_sha256": checkpoint["temperature_hash"],
            "feature_schema_sha256": checkpoint["feature_schema_hash"],
        }
        evaluation_manifest = {
            "schema_version": "bace_frozen_cell_evaluation_manifest_v1",
            **identities,
            "native_action_preserved": True,
            "threshold_values": thresholds["values"],
            "theta_star": thresholds["theta_star"],
            "cost_cap": thresholds["cost_cap"],
            "threshold_source": thresholds["threshold_source"],
            "threshold_payload_hash": thresholds["threshold_payload_hash"],
            "test_parent_set_digest": split["test_parent_set_sha256"],
            "test_split_declared_size": split["test_split_size"],
            "test_shard_manifests": split["shard_manifests"],
            "source_final_manifest": {
                "path": str(inputs.final_manifest_path),
                "sha256": sha256_file(inputs.final_manifest_path),
            },
            "source_selection_manifest": {
                "path": str(inputs.selection_manifest_path),
                "sha256": sha256_file(inputs.selection_manifest_path),
            },
            "source_test_manifest": {
                "path": str(inputs.test_manifest_path),
                "sha256": sha256_file(inputs.test_manifest_path),
            },
            "source_pair_matrix": {
                "path": str(inputs.pair_matrix_path),
                "sha256": sha256_file(inputs.pair_matrix_path),
            },
            "source_final_metrics": {
                "path": str(inputs.final_metrics_path),
                "sha256": sha256_file(inputs.final_metrics_path),
            },
            "scientific_metrics_recomputed": False,
            "deterministic_aggregation_replayed": True,
        }
        _write_json(temporary / "summary.json", summary)
        _write_json(temporary / "oracle_manifest.json", oracle_manifest)
        _write_json(temporary / "evaluation_manifest.json", evaluation_manifest)

        closure_names = (
            "figure3_coverage_vs_k.csv",
            "figure4_coverage_vs_threshold.csv",
            table_name,
            "prefix_metrics.csv",
            "prefix_metrics.json",
            "parent_best_distances.csv",
            "destination_distribution.csv",
            "summary.json",
            "oracle_manifest.json",
            "evaluation_manifest.json",
        )
        files = {
            name: {
                "bytes": (temporary / name).stat().st_size,
                "sha256": sha256_file(temporary / name),
            }
            for name in closure_names
        }
        run_manifest = {
            "schema_version": SCHEMA_VERSION,
            **identities,
            "files": files,
            "created_at": _utc_now(),
            "pass_marker_published_last": True,
        }
        _write_json(temporary / "run_manifest.json", run_manifest)
        artifact_manifest = {
            "schema_version": "bace_frozen_cell_artifact_manifest_v1",
            "dataset": DATASET,
            "method": inputs.method,
            "files": {
                **files,
                "run_manifest.json": {
                    "bytes": (temporary / "run_manifest.json").stat().st_size,
                    "sha256": sha256_file(temporary / "run_manifest.json"),
                },
            },
        }
        _write_json(temporary / "artifact_manifest.json", artifact_manifest)
        freeze_manifest = {
            "schema_version": "bace_frozen_cell_freeze_manifest_v1",
            **identities,
            "files": files,
            "run_manifest_sha256": sha256_file(temporary / "run_manifest.json"),
            "artifact_manifest_sha256": sha256_file(
                temporary / "artifact_manifest.json"
            ),
            "artifact_only_replay": True,
            "scientific_metrics_recomputed": False,
            "raw_test_opened": False,
        }
        _write_json(temporary / "freeze_manifest.json", freeze_manifest)
        _write_json(
            temporary / "_FINALIZED.json",
            {
                "schema_version": "bace_frozen_cell_finalized_v1",
                "dataset": DATASET,
                "method": inputs.method,
                "status": "PASS",
                "finalized": True,
                "gate_passed": True,
                "frozen": True,
                "freeze_manifest_sha256": sha256_file(
                    temporary / "freeze_manifest.json"
                ),
                "raw_test_opened": False,
            },
        )
        final_audit = {
            "schema_version": "bace_frozen_cell_final_artifact_audit_v1",
            **identities,
            "passed": True,
            "audit_passed": True,
            "final_artifact_audit_passed": True,
            "files": files,
            "artifact_manifest_sha256": sha256_file(temporary / "artifact_manifest.json"),
            "freeze_manifest_sha256": sha256_file(temporary / "freeze_manifest.json"),
            "finalized_marker_sha256": sha256_file(temporary / "_FINALIZED.json"),
            "all_required_files_nonempty": all((temporary / name).stat().st_size > 0 for name in closure_names),
            "hash_closure_complete": True,
            "raw_test_opened": False,
            "no_numeric_imputation": True,
            "n_a_fields_have_explicit_reason": True,
        }
        _write_json(temporary / "final_artifact_audit.json", final_audit)
        # PASS is the final file published inside the atomically renamed cell.
        _atomic_text(temporary / "PASS", "PASS\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "status": "PASS",
        "dataset": DATASET,
        "method": inputs.method,
        "standardized_output_root": str(destination),
        "source_final_root": str(source),
        "raw_test_opened": False,
        "oracle_checkpoint_hash": checkpoint["checkpoint_id"],
        "dataset_hash": checkpoint["dataset_hash"],
        "split_hash": split["test_split_hash"],
        "molclr_checkpoint_hash": molclr_hash,
        "threshold_config_hash": thresholds["threshold_config_hash"],
    }


__all__ = [
    "BACECellStandardizationError",
    "CF_MODE",
    "DATASET",
    "DISTANCE_LINE",
    "MAX_K",
    "METHODS",
    "PASS_MARKER",
    "SCHEMA_VERSION",
    "TABLE_K",
    "sha256_file",
    "standardize_bace_frozen_cell",
]
