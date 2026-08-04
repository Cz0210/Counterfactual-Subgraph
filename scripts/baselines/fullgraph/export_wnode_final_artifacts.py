#!/usr/bin/env python3
"""Export frozen full-graph WNode test results as final paper artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.fullgraph_wnode_artifacts import export_final_artifacts  # noqa: E402


def _float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated float list.")
    return values


def _int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated integer list.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test-run-dir", required=True)
    parser.add_argument("--calibration-run-dir", required=True)
    parser.add_argument("--frozen-candidates-csv", required=True)
    parser.add_argument("--frozen-candidate-manifest")
    parser.add_argument("--expected-candidate-order-sha256")
    parser.add_argument("--ours-schema-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--dataset", default="Mutagenicity")
    parser.add_argument("--source-label", type=int, default=1)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--test-job-id", required=True)
    parser.add_argument("--theta-star", type=float, required=True)
    parser.add_argument("--cost-cap", type=float, required=True)
    parser.add_argument("--thresholds", type=_float_list, required=True)
    parser.add_argument(
        "--k-values",
        type=_int_list,
        default=list(range(1, 21)),
        help="Comma-separated frozen prefix K values.",
    )
    parser.add_argument("--expected-parent-count", type=int, required=True)
    parser.add_argument("--expected-candidate-count", type=int, required=True)
    parser.add_argument("--expected-pair-count", type=int, required=True)
    parser.add_argument("--forbid-selection", action="store_true")
    parser.add_argument("--forbid-fitting", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = export_final_artifacts(
        test_run_dir=args.test_run_dir,
        calibration_run_dir=args.calibration_run_dir,
        frozen_candidates_csv=args.frozen_candidates_csv,
        ours_schema_root=args.ours_schema_root,
        output_dir=args.output_dir,
        method_name=args.method_name,
        dataset=args.dataset,
        source_label=args.source_label,
        target_label=args.target_label,
        test_job_id=args.test_job_id,
        theta_star=args.theta_star,
        cost_cap=args.cost_cap,
        thresholds=args.thresholds,
        k_values=args.k_values,
        expected_parent_count=args.expected_parent_count,
        expected_candidate_count=args.expected_candidate_count,
        expected_pair_count=args.expected_pair_count,
        forbid_selection=args.forbid_selection,
        forbid_fitting=args.forbid_fitting,
        frozen_candidate_manifest=args.frozen_candidate_manifest,
        expected_candidate_order_sha256=args.expected_candidate_order_sha256,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    print("[FULLGRAPH_WNODE_FINAL_ARTIFACT_EXPORT_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
