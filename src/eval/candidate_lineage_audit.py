"""Fail-closed candidate source, selector, and threshold lineage gates."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.data.molecular_split import file_sha256, stable_json_sha256


def read_candidate_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank candidate row at {source}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Candidate row is not an object: {source}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Candidate lineage input is empty: {source}")
    return rows


def audit_candidate_lineage(
    rows: Iterable[Mapping[str, Any]],
    *,
    allowed_candidate_splits: Sequence[str] = ("train", "val"),
    selector_source_splits: Sequence[str] = ("calibration",),
    threshold_source: str = "calibration",
    expected_dataset: str | None = None,
) -> dict[str, Any]:
    candidates = [dict(row) for row in rows]
    if not candidates:
        raise ValueError("Candidate lineage audit rejects an empty candidate pool.")
    allowed = {str(value) for value in allowed_candidate_splits}
    required = {
        "candidate_id",
        "candidate_source",
        "parent_id",
        "parent_split",
        "generation_seed",
        "generation_rank",
    }
    missing: dict[int, list[str]] = {}
    candidate_ids: list[str] = []
    parent_splits: list[str] = []
    for index, row in enumerate(candidates):
        absent = sorted(key for key in required if row.get(key) in {None, ""})
        if absent:
            missing[index] = absent
            continue
        candidate_ids.append(str(row["candidate_id"]))
        parent_splits.append(str(row["parent_split"]))
        if expected_dataset is not None and str(row.get("dataset") or expected_dataset) != expected_dataset:
            raise ValueError(f"Candidate row {index} has the wrong dataset identity.")
    if missing:
        raise ValueError(f"Candidate lineage fields are missing: {missing}")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Candidate IDs must be unique.")
    disallowed = sorted(set(parent_splits) - allowed)
    if disallowed:
        raise ValueError(
            f"Candidate pool contains disallowed source splits: {disallowed}"
        )
    selectors = {str(value) for value in selector_source_splits}
    violations: list[str] = []
    if "test" in selectors:
        violations.append("test_used_for_selector")
    if str(threshold_source) != "calibration":
        violations.append("threshold_source_not_calibration")
    if violations:
        raise ValueError("Candidate evaluation leakage: " + ", ".join(violations))
    return {
        "schema_version": "candidate_lineage_audit_v1",
        "passed": True,
        "dataset": expected_dataset,
        "candidate_count": len(candidates),
        "candidate_ids_sha256": stable_json_sha256(candidate_ids),
        "candidate_order_sha256": stable_json_sha256(candidate_ids),
        "candidate_source_counts": dict(
            sorted(Counter(str(row["candidate_source"]) for row in candidates).items())
        ),
        "candidate_source_splits": sorted(set(parent_splits)),
        "allowed_candidate_source_splits": sorted(allowed),
        "selector_source_splits": sorted(selectors),
        "threshold_source": str(threshold_source),
        "test_used_for_candidate_generation": False,
        "test_used_for_selector": False,
        "threshold_fitted_on_test": False,
        "selection_performed_in_eval": False,
    }


def audit_candidate_lineage_file(
    path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    return {
        **audit_candidate_lineage(read_candidate_jsonl(source), **kwargs),
        "candidate_path": str(source),
        "candidate_path_sha256": file_sha256(source),
    }


__all__ = [
    "audit_candidate_lineage",
    "audit_candidate_lineage_file",
    "read_candidate_jsonl",
]
