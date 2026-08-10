#!/usr/bin/env python3
"""Summarize calibration-only structural and WNode candidate-pool upper bounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def summarize(matrix_root: Path, thresholds_path: Path, expected_parent_count: int) -> dict[str, Any]:
    pairs = _read_jsonl(matrix_root / "pair_matrix.jsonl")
    manifest = json.loads((matrix_root / "matrix_manifest.json").read_text(encoding="utf-8"))
    thresholds_payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
    thresholds = [float(value) for value in thresholds_payload["thresholds"]]
    theta_star = float(thresholds_payload["theta_star"])
    parent_ids = sorted({str(row["parent_id"]) for row in pairs})
    candidate_ids = sorted({str(row["candidate_id"]) for row in pairs})
    if expected_parent_count and len(parent_ids) != expected_parent_count:
        raise AssertionError(
            f"Parent count mismatch: expected {expected_parent_count}, found {len(parent_ids)}"
        )
    denominator = len(parent_ids)

    def covered(predicate: Any) -> set[str]:
        return {str(row["parent_id"]) for row in pairs if predicate(row)}

    structural = covered(lambda row: bool(row.get("applicable")))
    connected = covered(lambda row: int(row.get("num_connected_valid_matches") or 0) > 0)
    strict = covered(lambda row: bool(row.get("pair_strict_flip")))
    close_sets: dict[str, set[str]] = {}
    for threshold in thresholds:
        close_sets[str(threshold)] = covered(
            lambda row, threshold=threshold: bool(row.get("pair_strict_flip"))
            and row.get("wnode_distance") is not None
            and math.isfinite(float(row["wnode_distance"]))
            and float(row["wnode_distance"]) <= threshold
        )
    theta_key = min(close_sets, key=lambda key: abs(float(key) - theta_star))
    payload = {
        "status": "PASS",
        "dataset": "BACE",
        "split": "calibration",
        "test_loaded": False,
        "action_semantics_version": manifest.get("action_semantics_version"),
        "candidate_universe_policy": (manifest.get("inputs") or {}).get(
            "candidate_universe_policy"
        ),
        "parent_count": denominator,
        "candidate_count": len(candidate_ids),
        "STRUCTURAL_UNION_COVERAGE": len(structural) / denominator,
        "CONNECTED_VALID_UNION_COVERAGE": len(connected) / denominator,
        "CONNECTED_STRICT_FLIP_UNION_COVERAGE": len(strict) / denominator,
        "CLOSE_UNION_AT_PRIMARY_THETA": len(close_sets[theta_key]) / denominator,
        "primary_theta_for_diagnostic": theta_star,
        "primary_theta_source": str(thresholds_path),
        "primary_theta_source_sha256": _sha256(thresholds_path),
        "primary_theta_is_pending_protocol_audit": True,
        "CLOSE_UNION_AT_EACH_FIGURE4_THRESHOLD": [
            {
                "threshold": threshold,
                "covered_count": len(close_sets[str(threshold)]),
                "coverage": len(close_sets[str(threshold)]) / denominator,
            }
            for threshold in thresholds
        ],
        "structural_covered_parent_ids": sorted(structural),
        "connected_valid_parent_ids": sorted(connected),
        "strict_flip_parent_ids": sorted(strict),
        "close_parent_ids_at_diagnostic_theta": sorted(close_sets[theta_key]),
    }
    _atomic_json(matrix_root / "candidate_union_summary.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--thresholds-json", required=True)
    parser.add_argument("--expected-parent-count", type=int, default=0)
    args = parser.parse_args()
    if "test" in f"{args.matrix_root} {args.thresholds_json}".lower():
        raise ValueError("Candidate-union calibration summary forbids test inputs.")
    payload = summarize(
        Path(args.matrix_root).expanduser().resolve(),
        Path(args.thresholds_json).expanduser().resolve(),
        int(args.expected_parent_count),
    )
    print(json.dumps(payload, sort_keys=True), flush=True)
    print("[BACE_CANDIDATE_UNION_SUMMARY_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
