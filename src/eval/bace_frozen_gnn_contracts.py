"""Shared contracts for the provenance-clean BACE Frozen-GNN downstream route.

This module intentionally contains no model execution.  It owns stable parent
partitioning, atomic artifacts, scientific provenance gates, and split access
authorization shared by B8--B14.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence


SOURCE_LABEL = 1
NUM_SHARDS = 4
DATASET = "bace"
ORACLE_BACKEND = "gnn"
CLASSIFIER_TYPE = "gnn"
CF_MODE = "strict_flip"
NUM_CLASSES = 2


@dataclass(frozen=True, slots=True)
class BACEParent:
    """One immutable source-class parent used by a downstream stage."""

    parent_id: str
    smiles: str
    label: int
    source_row_index: int
    prompt: str | None = None


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def stable_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path_like: str | Path) -> str:
    path = Path(path_like).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path_like: str | Path) -> dict[str, Any]:
    path = Path(path_like).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Artifact identity requires one file: {path}")
    return {
        "path": str(path),
        "size": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: str | Path, payload: Any) -> None:
    _atomic_text(
        Path(path),
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )


def atomic_marker(path: str | Path, marker: str) -> None:
    """Publish one fsynced stage marker as the final commit point."""

    _atomic_text(Path(path), str(marker).rstrip("\n") + "\n")


def atomic_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_text(
        Path(path),
        "".join(
            json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n"
            for row in rows
        ),
    )


def atomic_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    if not fieldnames:
        raise ValueError(f"Cannot write empty CSV artifact: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: (
                            json.dumps(value, sort_keys=True)
                            if isinstance(value, (list, tuple, dict))
                            else ("" if value is None else value)
                        )
                        for key, value in row.items()
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(destination.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path_like: str | Path) -> dict[str, Any]:
    path = Path(path_like).expanduser().resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return payload


def read_jsonl(path_like: str | Path) -> list[dict[str, Any]]:
    path = Path(path_like).expanduser().resolve(strict=True)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def fresh_output_dir(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Scientific output root must be absolute: {path}")
    path = path.resolve(strict=False)
    if path.exists():
        raise FileExistsError(f"Fresh output root already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    return path


def _normalize_parent_id(row: Mapping[str, Any], index: int) -> str:
    for key in ("parent_id", "molecule_id", "compound_id", "id", "record_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return f"BACE_{index:08d}"


def _coerce_label(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    try:
        result = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return result if result in (0, 1) else None


def load_bace_parents(
    path_like: str | Path,
    *,
    source_label: int = SOURCE_LABEL,
) -> list[BACEParent]:
    """Load source-class parents while preserving explicit dataset identities."""

    path = Path(path_like).expanduser().resolve(strict=True)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    elif path.suffix.lower() in {".jsonl", ".json"}:
        rows = read_jsonl(path)
    else:
        raise ValueError(f"Unsupported BACE split format: {path}")
    parents: list[BACEParent] = []
    for index, row in enumerate(rows):
        label = None
        for key in ("label", "target", "TARGET", "y", "class"):
            label = _coerce_label(row.get(key))
            if label is not None:
                break
        if label != int(source_label):
            continue
        smiles = ""
        for key in (
            "parent_smiles",
            "model_smiles",
            "canonical_smiles",
            "smiles",
            "SMILES",
        ):
            smiles = str(row.get(key) or "").strip()
            if smiles:
                break
        if not smiles:
            raise ValueError(f"Source-class BACE row {index} lacks SMILES: {path}")
        parent_id = _normalize_parent_id(row, index)
        prompt = str(row.get("prompt") or "").strip() or None
        parents.append(
            BACEParent(
                parent_id=parent_id,
                smiles=smiles,
                label=label,
                source_row_index=index,
                prompt=prompt,
            )
        )
    if not parents:
        raise ValueError(f"No BACE source-label={source_label} parents found: {path}")
    ids = [parent.parent_id for parent in parents]
    if len(ids) != len(set(ids)):
        duplicates = sorted({value for value in ids if ids.count(value) > 1})[:5]
        raise ValueError(f"BACE split contains duplicate parent IDs: {duplicates}")
    return parents


def fixed_parent_shard_map(
    parent_ids: Sequence[str], *, num_shards: int = NUM_SHARDS
) -> dict[str, int]:
    """Assign sorted parent IDs by positional modulo; independent of GPU count."""

    if int(num_shards) != NUM_SHARDS:
        raise ValueError(f"BACE route is frozen to exactly {NUM_SHARDS} shards")
    normalized = [str(value).strip() for value in parent_ids]
    if any(not value for value in normalized):
        raise ValueError("Parent IDs must be non-empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError("Parent IDs must be unique before fixed sharding")
    return {
        parent_id: position % NUM_SHARDS
        for position, parent_id in enumerate(sorted(normalized))
    }


def select_parent_shard(
    parents: Sequence[BACEParent], shard_index: int
) -> list[BACEParent]:
    if not 0 <= int(shard_index) < NUM_SHARDS:
        raise ValueError(f"shard_index must be in [0, {NUM_SHARDS - 1}]")
    mapping = fixed_parent_shard_map([parent.parent_id for parent in parents])
    return sorted(
        (parent for parent in parents if mapping[parent.parent_id] == int(shard_index)),
        key=lambda parent: parent.parent_id,
    )


def validate_materialized_parent_shard(
    path_like: str | Path,
    *,
    parents: Sequence[BACEParent],
    shard_index: int,
    split: str,
) -> dict[str, Any]:
    """Bind a controller-materialized shard to D's frozen partition rule."""

    payload = read_json(path_like)
    expected_ids = [parent.parent_id for parent in select_parent_shard(parents, shard_index)]
    required = {
        "status": "FROZEN",
        "dataset": DATASET,
        "split": str(split).lower(),
        "shard_id": int(shard_index),
        "shard_count": NUM_SHARDS,
    }
    failures = [
        f"{key}={payload.get(key)!r}"
        for key, expected in required.items()
        if payload.get(key) != expected
    ]
    observed_ids = [str(value) for value in payload.get("parent_ids", [])]
    if observed_ids != expected_ids:
        failures.append("parent_ids_not_exact_fixed_partition")
    if failures:
        raise ValueError(
            "Controller parent shard differs from frozen D contract: "
            + ", ".join(failures)
        )
    return payload


def _walk_mappings(payload: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            yield str(key), value
            yield from _walk_mappings(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            yield from _walk_mappings(value)


def assert_no_rf_provenance(payload: Mapping[str, Any]) -> None:
    """Reject positive RF provenance without rejecting explicit false guards."""

    for key, value in _walk_mappings(payload):
        normalized_key = key.strip().lower()
        normalized_value = str(value).strip().lower() if value is not None else ""
        if normalized_key == "rf_oracle_used" and value is not False:
            raise ValueError(f"RF guard must be explicitly false, observed {value!r}")
        if normalized_key in {"oracle_backend", "teacher_backend", "classifier_type"}:
            if normalized_value in {"rf", "random_forest", "randomforestclassifier"}:
                raise ValueError(f"RF-contaminated {key}={value!r}")
        if normalized_key in {
            "policy_initialization_type",
            "provenance_classification",
            "classification",
        } and normalized_value in {"rf_contaminated", "unknown", "missing"}:
            raise ValueError(f"Unclean BACE provenance {key}={value!r}")
        if isinstance(value, str) and Path(value).name.lower() in {
            "rf_model.pkl",
            "random_forest.pkl",
        }:
            raise ValueError(f"RF artifact is forbidden in BACE route: {value}")


def validate_gnn_provenance(
    payload: Mapping[str, Any], *, expected_stage: str | None = None
) -> None:
    assert_no_rf_provenance(payload)
    required = {
        "dataset": DATASET,
        "oracle_backend": ORACLE_BACKEND,
        "classifier_type": CLASSIFIER_TYPE,
        "rf_oracle_used": False,
        "source_label": SOURCE_LABEL,
        "num_classes": NUM_CLASSES,
    }
    failures = [
        f"{key}={payload.get(key)!r}"
        for key, expected in required.items()
        if payload.get(key) != expected
    ]
    if expected_stage is not None and payload.get("stage") != expected_stage:
        failures.append(f"stage={payload.get('stage')!r}")
    if failures:
        raise ValueError("BACE GNN provenance contract failed: " + ", ".join(failures))


def validate_pass_manifest(
    path_like: str | Path,
    *,
    expected_stage: str | None = None,
    require_no_test: bool = False,
) -> dict[str, Any]:
    payload = read_json(path_like)
    if payload.get("status") not in {"PASS", "FROZEN"}:
        raise ValueError(f"Predecessor stage is not PASS/FROZEN: {path_like}")
    validate_gnn_provenance(payload, expected_stage=expected_stage)
    if require_no_test and (
        payload.get("test_loaded") is not False
        or payload.get("test_used") not in (None, False)
    ):
        raise ValueError(f"Predecessor violates the no-test boundary: {path_like}")
    return payload


def assert_stage_data_boundary(
    *,
    stage: str,
    split_path: str | Path | None,
    frozen_selection_manifest: str | Path | None = None,
) -> dict[str, Any] | None:
    """Authorize raw split access; B13 requires a frozen B12 manifest first."""

    normalized_stage = str(stage).strip().upper()
    if normalized_stage in {"B8_POOL_BASE", "B9_POOL_HIGHTEMP"}:
        if split_path is None:
            raise ValueError(f"{normalized_stage} requires the train split")
        if "test" in Path(split_path).name.lower() or "calib" in Path(split_path).name.lower():
            raise ValueError(f"{normalized_stage} may read train source parents only")
        if "train" not in Path(split_path).name.lower():
            raise ValueError(f"{normalized_stage} requires an explicitly named train split")
        return None
    if normalized_stage == "B11_CROSS_PARENT_VERIFIED":
        if split_path is None:
            raise ValueError("B11 requires the calibration split")
        if "test" in Path(split_path).name.lower():
            raise ValueError("B11 must not access the held-out test split")
        if "calib" not in Path(split_path).name.lower():
            raise ValueError("B11 requires an explicitly named calibration split")
        return None
    if normalized_stage == "B13_FINAL_EVAL":
        if frozen_selection_manifest is None:
            raise ValueError("B13 test access requires a frozen B12 manifest")
        selection = validate_pass_manifest(
            frozen_selection_manifest,
            expected_stage="B12_SELECTOR",
            require_no_test=True,
        )
        if (
            selection.get("selector_fitted_on_calibration") is not True
            or selection.get("selection_frozen") is not True
            or int(selection.get("K", 0)) != 20
        ):
            raise ValueError("B13 rejected an incomplete B12 selection freeze")
        if split_path is None:
            raise ValueError("B13 requires one explicit held-out test split")
        if "test" not in Path(split_path).name.lower():
            raise ValueError("B13 requires an explicitly named held-out test split")
        return selection
    raise ValueError(f"Unsupported downstream data-boundary stage: {stage}")


def require_finite_nonnegative(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


__all__ = [
    "BACEParent",
    "CF_MODE",
    "CLASSIFIER_TYPE",
    "DATASET",
    "NUM_SHARDS",
    "NUM_CLASSES",
    "ORACLE_BACKEND",
    "SOURCE_LABEL",
    "assert_no_rf_provenance",
    "assert_stage_data_boundary",
    "atomic_csv",
    "atomic_json",
    "atomic_jsonl",
    "atomic_marker",
    "file_identity",
    "fixed_parent_shard_map",
    "fresh_output_dir",
    "load_bace_parents",
    "read_json",
    "read_jsonl",
    "require_finite_nonnegative",
    "select_parent_shard",
    "sha256_file",
    "stable_sha256",
    "utc_now",
    "validate_gnn_provenance",
    "validate_materialized_parent_shard",
    "validate_pass_manifest",
]
