#!/usr/bin/env python3
"""Freeze method-balanced pooled BACE calibration thresholds before final test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chem.hard_deletion import (  # noqa: E402
    CONNECTED_ACTION_SEMANTICS,
    CONNECTED_MATCH_SELECTION_POLICY,
)
from src.eval.bace_paper_artifacts import QUANTILES  # noqa: E402


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _truth(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _weighted_quantile(values: list[tuple[float, float]], quantile: float) -> float:
    ordered = sorted(values, key=lambda item: item[0])
    total = sum(weight for _value, weight in ordered)
    if not ordered or total <= 0.0:
        raise ValueError("Cannot fit pooled thresholds without weighted finite distances.")
    target = float(quantile) * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative + 1e-15 >= target:
            return float(value)
    return float(ordered[-1][0])


def _parent_minima_ours(rows: list[dict[str, Any]]) -> dict[str, float]:
    minima: dict[str, float] = {}
    for row in rows:
        if not _truth(row.get("pair_strict_flip")):
            continue
        if not (
            _truth(row.get("residual_connected"))
            and _truth(row.get("sanitize_ok"))
            and int(row.get("residual_num_components") or 0) == 1
            and not _truth(row.get("contains_dot"))
        ):
            raise ValueError("Ours pooled threshold source contains an invalid connected winner.")
        value = float(row["wnode_distance"])
        if not math.isfinite(value) or value < 0.0:
            continue
        parent = str(row["parent_id"])
        minima[parent] = min(value, minima.get(parent, math.inf))
    return minima


def _parent_minima_gcf(rows: list[dict[str, str]]) -> dict[str, float]:
    minima: dict[str, float] = {}
    for row in rows:
        if not _truth(row.get("teacher_strict_flip")):
            continue
        if not _truth(row.get("delete_valid")):
            raise ValueError("GCF pooled threshold source contains an invalid candidate row.")
        smiles = str(row.get("candidate_smiles") or "")
        if not smiles or "." in smiles:
            raise ValueError("GCF pooled threshold source contains a disconnected candidate.")
        value = float(row["distance"])
        if not math.isfinite(value) or value < 0.0:
            continue
        parent = str(row["parent_id"])
        minima[parent] = min(value, minima.get(parent, math.inf))
    return minima


def freeze_thresholds(
    *,
    ours_matrix_root: Path,
    gcf_calibration_root: Path,
    calibration_csv: Path,
    output_dir: Path,
) -> dict[str, Any]:
    ours_manifest_path = ours_matrix_root / "matrix_manifest.json"
    ours_audit_path = ours_matrix_root / "matrix_audit.json"
    ours_pairs_path = ours_matrix_root / "pair_matrix.jsonl"
    gcf_config_path = gcf_calibration_root / "run_config.json"
    gcf_pairs_path = gcf_calibration_root / "details" / "pair_details.csv"
    for path in (
        ours_manifest_path,
        ours_audit_path,
        ours_pairs_path,
        gcf_config_path,
        gcf_pairs_path,
        calibration_csv,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    ours_manifest = json.loads(ours_manifest_path.read_text())
    ours_audit = json.loads(ours_audit_path.read_text())
    gcf_config = json.loads(gcf_config_path.read_text())
    if ours_manifest.get("test_loaded") is not False:
        raise ValueError("Ours pooled threshold source does not prove test_loaded=false.")
    if ours_manifest.get("action_semantics_version") != CONNECTED_ACTION_SEMANTICS:
        raise ValueError("Ours pooled threshold source uses legacy actions.")
    if ours_audit.get("disconnected_residual_used_count") != 0:
        raise ValueError("Ours pooled threshold source used disconnected residuals.")
    if str(gcf_config.get("cf_mode")) != "strict_flip":
        raise ValueError("GCF pooled threshold source is not strict_flip.")
    calibration_ids = {str(row["molecule_id"]) for row in _csv(calibration_csv)}
    if len(calibration_ids) != 60:
        raise ValueError(f"Expected 60 BACE calibration parents, found {len(calibration_ids)}.")
    ours_minima = _parent_minima_ours(_jsonl(ours_pairs_path))
    gcf_minima = _parent_minima_gcf(_csv(gcf_pairs_path))
    if not set(ours_minima) <= calibration_ids or not set(gcf_minima) <= calibration_ids:
        raise ValueError("Pooled threshold source contains non-calibration parent IDs.")
    if not ours_minima or not gcf_minima:
        raise ValueError("Each method must contribute finite calibration distances.")
    weighted: list[tuple[float, float]] = []
    weighted.extend((value, 0.5 / len(ours_minima)) for value in ours_minima.values())
    weighted.extend((value, 0.5 / len(gcf_minima)) for value in gcf_minima.values())
    thresholds = [_weighted_quantile(weighted, value) for value in QUANTILES]
    if any(right < left for left, right in zip(thresholds, thresholds[1:])):
        raise AssertionError("Pooled thresholds are not monotone.")
    output_dir.mkdir(parents=True, exist_ok=False)
    threshold_path = output_dir / "thresholds.json"
    payload = {
        "schema_version": "bace_wnode_thresholds_v2",
        "dataset": "BACE",
        "distance_line": "MolCLR-Node-Wasserstein",
        "distance_type": "node_wasserstein",
        "cf_mode": "strict_flip",
        "threshold_source": "method_balanced_parent_minimum_pooled_calibration_v1",
        "threshold_protocol_version": "bace_connected_pooled_q30_q50_v4",
        "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
        "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY,
        "quantiles": list(QUANTILES),
        "thresholds": thresholds,
        "theta_star_quantile": 0.30,
        "theta_star": thresholds[3],
        "strict_primary_threshold": thresholds[3],
        "standard_sensitivity_quantile": 0.50,
        "standard_sensitivity_threshold": thresholds[4],
        "paper_primary_rule": "strict_pooled_calibration_q30",
        "cost_cap_quantile": 0.90,
        "cost_cap": thresholds[-1],
        "method_weights": {"ours": 0.5, "gcfexplainer": 0.5},
        "population_unit": "parent_method_minimum_connected_strict_flip_distance",
        "calibration_parent_count": len(calibration_ids),
        "ours_finite_parent_count": len(ours_minima),
        "gcf_finite_parent_count": len(gcf_minima),
        "same_parent_cohort": True,
        "method_specific_threshold": False,
        "shared_across_methods": True,
        "selection_used_test": False,
        "threshold_fitted_on_test": False,
        "calibration_parent_csv": str(calibration_csv),
        "calibration_parent_csv_sha256": _sha(calibration_csv),
        "ours_pair_matrix_sha256": _sha(ours_pairs_path),
        "gcf_pair_details_sha256": _sha(gcf_pairs_path),
    }
    _atomic(threshold_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    rows = [
        {"quantile": q, "threshold": t, "role": "primary_q30" if q == 0.30 else ("sensitivity_q50" if q == 0.50 else "figure4")}
        for q, t in zip(QUANTILES, thresholds, strict=True)
    ]
    csv_path = output_dir / "calibration_distance_quantiles.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("quantile", "threshold", "role"))
        writer.writeheader(); writer.writerows(rows)
    audit = {
        "status": "PASS",
        "Q30_PRE_REGISTERED_ACROSS_DATASETS": False,
        "CROSS_DATASET_RULE": "not_proven; BACE v4 freezes method-balanced pooled calibration before test",
        "THRESHOLD_METHOD_INDEPENDENT": True,
        "THRESHOLD_TEST_INDEPENDENT": True,
        "COMMON_PROTOCOL_GATE_READY": True,
        "primary_threshold": thresholds[3],
        "secondary_sensitivity_threshold": thresholds[4],
        "thresholds_sha256": _sha(threshold_path),
        "test_loaded": False,
    }
    _atomic(
        output_dir / "threshold_protocol_audit.json",
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
    )
    return {**audit, "thresholds": payload}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--ours-matrix-root", required=True)
    parser.add_argument("--gcf-calibration-root", required=True)
    parser.add_argument("--calibration-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = freeze_thresholds(
        ours_matrix_root=Path(args.ours_matrix_root).expanduser().resolve(),
        gcf_calibration_root=Path(args.gcf_calibration_root).expanduser().resolve(),
        calibration_csv=Path(args.calibration_csv).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
    )
    print(json.dumps(result, sort_keys=True))
    print("[BACE_POOLED_CONNECTED_THRESHOLDS_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
