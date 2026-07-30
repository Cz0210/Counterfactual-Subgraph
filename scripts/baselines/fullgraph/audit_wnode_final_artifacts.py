#!/usr/bin/env python3
"""Audit a frozen full-graph WNode final-artifact export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.fullgraph_wnode_artifacts import audit_final_artifacts  # noqa: E402


def _float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated float list.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--frozen-candidates-csv", required=True)
    parser.add_argument("--ours-schema-root", required=True)
    parser.add_argument("--expected-parent-count", type=int, required=True)
    parser.add_argument("--expected-candidate-count", type=int, required=True)
    parser.add_argument("--expected-pair-count", type=int, required=True)
    parser.add_argument("--theta-star", type=float, required=True)
    parser.add_argument("--cost-cap", type=float, required=True)
    parser.add_argument("--thresholds", type=_float_list, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit_final_artifacts(
        run_dir=args.run_dir,
        frozen_candidates_csv=args.frozen_candidates_csv,
        ours_schema_root=args.ours_schema_root,
        expected_parent_count=args.expected_parent_count,
        expected_candidate_count=args.expected_candidate_count,
        expected_pair_count=args.expected_pair_count,
        theta_star=args.theta_star,
        cost_cap=args.cost_cap,
        thresholds=args.thresholds,
        check_manifest=True,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    print("[FULLGRAPH_WNODE_FINAL_ARTIFACT_AUDIT_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
