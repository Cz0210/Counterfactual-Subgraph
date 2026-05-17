#!/usr/bin/env python3
"""Filter PPO prompt CSV rows by teacher correctness and confidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.teacher_confidence_filter import (
    TeacherConfidenceFilterConfig,
    filter_prompt_rows_by_teacher_confidence,
    write_filtered_prompt_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config path for HPC wrapper parity. The filter uses explicit CLI paths.",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to the PPO prompt CSV file.",
    )
    parser.add_argument(
        "--teacher-path",
        required=True,
        help="Path to the teacher/oracle model bundle.",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Preferred label column. Falls back to common aliases if missing.",
    )
    parser.add_argument(
        "--smiles-col",
        default="parent_smiles",
        help="Preferred parent-smiles column. Falls back to smiles/prompt aliases if missing.",
    )
    parser.add_argument(
        "--target-label",
        type=int,
        default=1,
        help="Only rows with this label are eligible for retention.",
    )
    parser.add_argument(
        "--min-p-label",
        type=float,
        default=0.5,
        help="Minimum teacher probability on the row label required for retention.",
    )
    parser.add_argument(
        "--require-teacher-correct",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require teacher predicted label to match the row label before keeping the sample.",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Path to write the filtered CSV.",
    )
    parser.add_argument(
        "--out-json",
        required=True,
        help="Path to write the summary JSON.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = filter_prompt_rows_by_teacher_confidence(
        args.dataset_path,
        teacher_path=args.teacher_path,
        config=TeacherConfidenceFilterConfig(
            label_col=str(args.label_col),
            smiles_col=str(args.smiles_col),
            target_label=int(args.target_label),
            min_p_label=float(args.min_p_label),
            require_teacher_correct=bool(args.require_teacher_correct),
        ),
    )
    write_filtered_prompt_outputs(
        result,
        out_csv=args.out_csv,
        out_json=args.out_json,
    )
    print(f"dataset_path: {Path(args.dataset_path).expanduser().resolve()}")
    print(f"teacher_path: {Path(args.teacher_path).expanduser().resolve()}")
    print(f"out_csv: {Path(args.out_csv).expanduser().resolve()}")
    print(f"out_json: {Path(args.out_json).expanduser().resolve()}")
    print(f"resolved_smiles_col: {result.summary['resolved_smiles_col']}")
    print(f"resolved_label_col: {result.summary['resolved_label_col']}")
    print(f"input_count: {result.summary['input_count']}")
    print(f"target_label_count: {result.summary['target_label_count']}")
    print(f"kept_count: {result.summary['kept_count']}")
    print(f"kept_rate: {result.summary['kept_rate']}")
    print(f"teacher_correct_rate_before: {result.summary['teacher_correct_rate_before']}")


if __name__ == "__main__":
    main()
