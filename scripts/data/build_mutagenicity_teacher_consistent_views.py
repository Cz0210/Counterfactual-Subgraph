#!/usr/bin/env python3
"""Build complete Mutagenicity teacher-consistent views from fixed split metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.mutagenicity_teacher_views import (  # noqa: E402
    DEFAULT_EXPECTED_SOURCE_CORRECT_COUNTS,
    TeacherViewConfig,
    build_teacher_consistent_views,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("outputs/hpc/datasets/final/mutagenicity_v1_processed"),
    )
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=Path("outputs/hpc/oracle/final/mutagenicity_rf_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hpc/datasets/mutagenicity_v1_teacher_consistent"),
    )
    parser.add_argument("--source-label", type=int, default=1)
    parser.add_argument("--target-label", type=int, default=0)
    for split, count in DEFAULT_EXPECTED_SOURCE_CORRECT_COUNTS.items():
        parser.add_argument(
            f"--expected-{split}-source-correct",
            type=int,
            default=count,
            help=f"Expected label-1 teacher-correct rows for {split}; <=0 disables the check.",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected = {
        split: int(getattr(args, f"expected_{split}_source_correct"))
        for split in DEFAULT_EXPECTED_SOURCE_CORRECT_COUNTS
    }
    print("[MUTAGENICITY_TEACHER_VIEWS_CONFIG]")
    print(f"processed_root={args.processed_root}")
    print(f"teacher_root={args.teacher_root}")
    print(f"output_dir={args.output_dir}")
    print(f"source_label={args.source_label}")
    print(f"target_label={args.target_label}")
    print(f"expected_source_correct_counts={expected}")
    summary = build_teacher_consistent_views(
        processed_root=args.processed_root,
        teacher_root=args.teacher_root,
        output_dir=args.output_dir,
        config=TeacherViewConfig(
            source_label=int(args.source_label),
            target_label=int(args.target_label),
            expected_source_correct_counts=expected,
        ),
    )
    print(json.dumps(summary["split_counts"], sort_keys=True))
    print("[MUTAGENICITY_TEACHER_VIEWS_BUILD_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
