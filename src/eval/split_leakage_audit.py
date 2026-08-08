"""Fail-closed split and threshold leakage audit helpers."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.data.molecular_split import SPLIT_NAMES, audit_split_overlap, file_sha256


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Leakage audit rejects empty split CSV: {path}")
    return rows


def audit_split_files(
    split_paths: Mapping[str, str | Path],
    *,
    protocol: str,
    require_scaffold_disjoint: bool,
    candidate_source_splits: Sequence[str],
    selector_source_splits: Sequence[str],
    threshold_source_split: str,
) -> dict[str, Any]:
    missing = sorted(set(SPLIT_NAMES) - set(split_paths))
    if missing:
        raise ValueError(f"Leakage audit is missing split paths: {missing}")
    candidate_splits = tuple(str(value) for value in candidate_source_splits)
    selector_splits = tuple(str(value) for value in selector_source_splits)
    unknown = sorted(
        (set(candidate_splits) | set(selector_splits) | {threshold_source_split})
        - set(SPLIT_NAMES)
    )
    if unknown:
        raise ValueError(f"Leakage audit received unknown split roles: {unknown}")
    violations: list[str] = []
    if "test" in candidate_splits:
        violations.append("test_used_for_candidate_generation")
    if "test" in selector_splits:
        violations.append("test_used_for_selector")
    if str(threshold_source_split) == "test":
        violations.append("threshold_fitted_on_test")
    if str(threshold_source_split) != "calibration":
        violations.append("threshold_source_not_calibration")
    if violations:
        raise ValueError("Protocol leakage detected: " + ", ".join(violations))
    resolved = {
        split: Path(split_paths[split]).expanduser().resolve()
        for split in SPLIT_NAMES
    }
    rows = {split: _read_csv(resolved[split]) for split in SPLIT_NAMES}
    overlap = audit_split_overlap(
        rows,
        require_scaffold_disjoint=require_scaffold_disjoint,
    )
    return {
        "schema_version": "split_leakage_audit_v1",
        "passed": True,
        "protocol": str(protocol),
        "split_paths": {split: str(resolved[split]) for split in SPLIT_NAMES},
        "split_sha256": {split: file_sha256(resolved[split]) for split in SPLIT_NAMES},
        "candidate_source_splits": list(candidate_splits),
        "selector_source_splits": list(selector_splits),
        "threshold_source": str(threshold_source_split),
        "test_usage": "final_evaluation_only",
        "test_used_for_candidate_generation": False,
        "test_used_for_selector": False,
        "threshold_fitted_on_test": False,
        "require_scaffold_disjoint": bool(require_scaffold_disjoint),
        "overlap_audit": overlap,
    }


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Split manifest must be a JSON object: {source}")
    if payload.get("threshold_source") == "test":
        raise ValueError(f"Split manifest fits thresholds on test: {source}")
    return payload


__all__ = ["audit_split_files", "load_split_manifest"]
