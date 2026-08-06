#!/usr/bin/env python3
"""Run a COMRECGC recovery engineering gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.recovery_gate import (  # noqa: E402
    gate_aids_native_full,
    gate_mutagenicity_full,
    gate_mutagenicity_chemistry_smoke,
    gate_project_full,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("aids-native-full", "mut-chemistry-smoke", "mut-full", "project-full"),
        required=True,
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--eval-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=("aids", "mutagenicity"))
    parser.add_argument("--expected-parent-count", type=int)
    parser.add_argument("--expected-teacher-sha256")
    parser.add_argument("--expected-project-commit")
    args = parser.parse_args()
    if args.stage == "aids-native-full":
        result = gate_aids_native_full(
            args.input_dir,
            args.output_dir,
            expected_project_commit=args.expected_project_commit,
        )
    elif args.stage == "mut-chemistry-smoke":
        result = gate_mutagenicity_chemistry_smoke(
            args.input_dir,
            args.output_dir,
            eval_dir=args.eval_dir,
        )
    elif args.stage == "mut-full":
        result = gate_mutagenicity_full(args.input_dir, args.output_dir)
    else:
        if (
            not args.dataset
            or args.expected_parent_count is None
            or not args.expected_teacher_sha256
        ):
            parser.error(
                "project-full requires dataset, expected-parent-count, and teacher SHA256"
            )
        result = gate_project_full(
            args.input_dir,
            args.output_dir,
            dataset=args.dataset,
            expected_parent_count=args.expected_parent_count,
            expected_teacher_sha256=args.expected_teacher_sha256,
            expected_project_commit=args.expected_project_commit,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
