#!/usr/bin/env python3
"""Train the independent BACE RF teacher using train/validation selection only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bace_rf_teacher import train_bace_teacher  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", default="data/processed/BACE")
    parser.add_argument("--output-dir", default="outputs/hpc/oracle/bace")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--n-estimators-grid", default="300,600")
    parser.add_argument("--max-depth-grid", default="none,20,40")
    parser.add_argument("--min-samples-leaf-grid", default="1,2")
    parser.add_argument(
        "--selection-metric",
        choices=("balanced_accuracy", "auroc", "average_precision"),
        default="balanced_accuracy",
    )
    parser.add_argument("--class-weight", default="balanced_subsample")
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--n-jobs", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = train_bace_teacher(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        radius=args.radius,
        n_bits=args.n_bits,
        n_estimators_grid=args.n_estimators_grid,
        max_depth_grid=args.max_depth_grid,
        min_samples_leaf_grid=args.min_samples_leaf_grid,
        selection_metric=args.selection_metric,
        class_weight=(
            None if str(args.class_weight).lower() in {"none", "null"} else args.class_weight
        ),
        random_seed=args.random_seed,
        n_jobs=args.n_jobs,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    print("[BACE_TEACHER_TRAIN_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
