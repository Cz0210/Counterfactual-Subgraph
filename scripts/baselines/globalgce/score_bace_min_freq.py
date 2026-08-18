#!/usr/bin/env python3
"""Score one BACE GlobalGCE min_freq on calibration-only WNode pairs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_bace_action_adapter import (  # noqa: E402
    assert_nonzero_fullgraph_applicability,
)


def _truth(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _thresholds(path: Path) -> tuple[float, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    primary = payload.get("theta_star", payload.get("primary_threshold"))
    grid = payload.get("threshold_grid", payload.get("thresholds"))
    if isinstance(grid, dict):
        grid = list(grid.values())
    if primary is None:
        for key in ("selected", "primary"):
            if isinstance(payload.get(key), dict):
                primary = payload[key].get("theta_star", payload[key].get("threshold"))
                if primary is not None:
                    break
    if grid is None:
        grid = [primary]
    values = sorted({float(value) for value in grid if value is not None})
    if primary is None or not values:
        raise ValueError(f"Threshold manifest is incomplete: {path}")
    return float(primary), values


def _covred(sets: list[set[str]]) -> float:
    values = []
    for left, right in itertools.combinations(sets, 2):
        union = left | right
        values.append(len(left & right) / len(union) if union else 0.0)
    return sum(values) / len(values) if values else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--min-freq", type=int, required=True)
    parser.add_argument("--pool-path", required=True)
    parser.add_argument("--selection-csv", required=True)
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--thresholds-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matrix_root = Path(args.matrix_root).expanduser().resolve()
    fullgraph_pairs = matrix_root / "details" / "pair_details.csv"
    legacy_pairs = matrix_root / "pair_matrix.jsonl"
    if fullgraph_pairs.is_file():
        pairs: list[dict[str, Any]] = _csv(fullgraph_pairs)
        pair_schema = "connected_sanitized_fullgraph_counterfactual_v1"
        pair_audit = assert_nonzero_fullgraph_applicability(pairs)
    elif legacy_pairs.is_file():
        pairs = _jsonl(legacy_pairs)
        pair_schema = "legacy_deletion_matrix"
        pair_audit = None
    else:
        raise FileNotFoundError(
            f"No fullgraph pair_details.csv or legacy pair_matrix.jsonl under {matrix_root}."
        )
    with Path(args.selection_csv).expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        selected = list(csv.DictReader(handle))
    ordered_ids = [str(row.get("candidate_id") or "") for row in selected]
    if not all(ordered_ids) or len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("Selected GlobalGCE candidate IDs must be nonempty and unique.")
    observed_ids = {str(row.get("candidate_id") or "") for row in pairs}
    missing = [value for value in ordered_ids if value not in observed_ids]
    if missing:
        raise ValueError(f"Selected GlobalGCE candidates absent from matrix: {missing}")
    by_candidate: dict[str, dict[str, tuple[bool, float]]] = {}
    parents = sorted({str(row["parent_id"]) for row in pairs})
    for row in pairs:
        candidate_id = str(row["candidate_id"])
        distance = row.get("distance", row.get("wnode_distance"))
        by_candidate.setdefault(candidate_id, {})[str(row["parent_id"])] = (
            _truth(
                row.get(
                    "teacher_strict_flip",
                    row.get("cf_flip", row.get("pair_strict_flip", row.get("strict_flip"))),
                )
            ),
            float(distance) if distance not in (None, "") else math.inf,
        )
    primary, grid = _thresholds(Path(args.thresholds_json).expanduser().resolve())

    def coverage(prefix: list[str], theta: float) -> tuple[float, list[float]]:
        best: list[float] = []
        for parent in parents:
            distances = [
                by_candidate.get(candidate, {}).get(parent, (False, math.inf))[1]
                for candidate in prefix
                if by_candidate.get(candidate, {}).get(parent, (False, math.inf))[0]
            ]
            best.append(min(distances) if distances else math.inf)
        finite_close = [value for value in best if math.isfinite(value) and value <= theta]
        return len(finite_close) / len(parents), finite_close

    primary_curve = []
    multi_curve = []
    candidate_sets: list[set[str]] = []
    for index, candidate in enumerate(ordered_ids, start=1):
        cov, _ = coverage(ordered_ids[:index], primary)
        primary_curve.append(cov)
        multi_curve.append(sum(coverage(ordered_ids[:index], theta)[0] for theta in grid) / len(grid))
        candidate_sets.append({
            parent for parent, (strict, distance) in by_candidate.get(candidate, {}).items()
            if strict and math.isfinite(distance) and distance <= primary
        })
    _, costs = coverage(ordered_ids[:10], primary)
    result = {
        "min_freq": int(args.min_freq),
        "pool_path": str(Path(args.pool_path).expanduser().resolve()),
        "selection_split": "calibration",
        "test_loaded": False,
        "prefix_auc_k1_k10": sum(primary_curve[:10]) / min(10, len(primary_curve)),
        "multi_threshold_prefix_auc": sum(multi_curve[:10]) / min(10, len(multi_curve)),
        "cost": median(costs) if costs else 1.0,
        "coverage_redundancy": _covred(candidate_sets[:10]),
        "rule_count": len(ordered_ids),
        "primary_threshold": primary,
        "calibration_parent_count": len(parents),
        "candidate_native_type": "full_counterfactual_graph",
        "action_adapter": "connected_sanitized_fullgraph_counterfactual_v1",
        "pair_schema": pair_schema,
        "pair_audit": pair_audit,
    }
    if args.validate_only:
        print(json.dumps(result, indent=2, sort_keys=True))
        print("[BACE_GLOBALGCE_MIN_FREQ_SCORE_VALIDATE_OK]")
        return 0
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("[BACE_GLOBALGCE_MIN_FREQ_SCORE_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
