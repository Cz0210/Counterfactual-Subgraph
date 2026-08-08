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
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args(argv)
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
        top_k=args.top_k,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[BACE_GCFEXPLAINER_TOP20_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
