#!/usr/bin/env python3
"""Decode native GCF ranks and export the first 20 RF-valid fullgraphs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.gcfexplainer_mutagenicity_adapter import write_failure_artifacts  # noqa: E402
from src.baselines.gcfexplainer_mutagenicity_runtime import export_rf_valid_native_top20  # noqa: E402
from src.rewards.teacher_semantic import TeacherSemanticScorer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--summary-dir", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--parent-limit", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--forbid-calibration-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.forbid_calibration_test:
        raise ValueError("Candidate export requires --forbid-calibration-test.")
    teacher = TeacherSemanticScorer(args.teacher_path, device="cpu")
    if not teacher.available:
        raise RuntimeError(
            f"Mutagenicity RF teacher unavailable: {teacher.availability_reason}"
        )
    try:
        summary = export_rf_valid_native_top20(
            dataset_dir=args.dataset_dir,
            summary_dir=args.summary_dir,
            teacher=teacher,
            teacher_path=args.teacher_path,
            output_dir=args.output_dir,
            profile=args.profile,
            parent_limit=args.parent_limit,
            top_k=args.top_k,
        )
    except Exception as exc:
        write_failure_artifacts(args.output_dir, error=exc, resolved_config=vars(args))
        raise
    if args.profile == "smoke":
        print(
            "[MUTAGENICITY_GCFEXPLAINER_EXPORT_SMOKE_AUDIT_OK]",
            flush=True,
        )
        print(
            "candidate_yield_gate_passed="
            f"{str(bool(summary['candidate_yield_gate_passed'])).lower()}",
            flush=True,
        )
        if summary["candidate_yield_gate_passed"]:
            reason = "candidate_yield_available"
        elif int(summary["selected_count"]) == 0:
            reason = "no_rf_target_candidate"
        else:
            reason = "insufficient_rf_target_candidates"
        print(f"reason={reason}", flush=True)
        print("full_result_ready=false", flush=True)
    else:
        print("[MUTAGENICITY_GCFEXPLAINER_TOP20_OK]", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
