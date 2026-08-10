#!/usr/bin/env python3
"""Freeze the first twenty RF-target BACE candidates in official native order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.gcfexplainer_bace_runtime import export_bace_rf_valid_top20  # noqa: E402
from src.rewards.teacher_semantic import TeacherSemanticScorer  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--summary-dir", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--parent-limit", type=int, required=True)
    parser.add_argument("--target-k", type=int, default=None)
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Backward-compatible alias for --target-k.",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=0,
        help="Maximum native ranks to inspect; zero scans until K or pool exhaustion.",
    )
    parser.add_argument(
        "--scan-all",
        action="store_true",
        help="Audit every available native rank while freezing the same first valid top-k.",
    )
    parser.add_argument(
        "--require-connected",
        action="store_true",
        help="Reject disconnected full-graph candidates before teacher evaluation.",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.target_k is not None and args.top_k is not None:
        if int(args.target_k) != int(args.top_k):
            parser.error("--target-k and --top-k disagree")
    target_k = int(
        args.target_k if args.target_k is not None else args.top_k or 20
    )
    teacher = TeacherSemanticScorer(args.teacher_path, device="cpu")
    if not teacher.available:
        raise RuntimeError(f"BACE RF teacher unavailable: {teacher.availability_reason}")
    result = export_bace_rf_valid_top20(
        dataset_dir=args.dataset_dir,
        summary_dir=args.summary_dir,
        teacher=teacher,
        teacher_path=args.teacher_path,
        output_dir=args.output_dir,
        profile=args.profile,
        parent_limit=args.parent_limit,
        top_k=target_k,
        scan_limit=args.scan_limit,
        scan_all=bool(args.scan_all),
        require_connected=bool(args.require_connected),
        validate_only=args.validate_only,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[BACE_GCFEXPLAINER_TOP20_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
