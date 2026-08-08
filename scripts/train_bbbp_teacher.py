#!/usr/bin/env python3
"""Train the independent BBBP RF teacher using train/validation selection only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bbbp_rf_teacher import (  # noqa: E402
    train_bbbp_teacher,
    validate_bbbp_teacher_paths,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", default="data/processed/BBBP")
    parser.add_argument("--train-csv")
    parser.add_argument("--val-csv")
    parser.add_argument("--calibration-csv")
    parser.add_argument("--test-csv")
    parser.add_argument("--output-dir", default="outputs/hpc/oracle/bbbp")
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
    parser.add_argument("--seed", "--random-seed", dest="random_seed", type=int, default=13)
    parser.add_argument("--n-jobs", type=int, default=7)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir).expanduser().resolve()
    split_paths = {
        "train": Path(args.train_csv).expanduser().resolve()
        if args.train_csv
        else data_dir / "train.csv",
        "val": Path(args.val_csv).expanduser().resolve()
        if args.val_csv
        else data_dir / "val.csv",
        "calibration": Path(args.calibration_csv).expanduser().resolve()
        if args.calibration_csv
        else data_dir / "calibration.csv",
        "test": Path(args.test_csv).expanduser().resolve()
        if args.test_csv
        else data_dir / "test.csv",
    }
    if args.validate_only or args.dry_run:
        audit = validate_bbbp_teacher_paths(split_paths)
        print(
            json.dumps(
                {
                    "status": "VALIDATED_NOT_RUN",
                    "mode": "validate_only" if args.validate_only else "dry_run",
                    "dataset": "BBBP",
                    "split_paths": {key: str(value) for key, value in split_paths.items()},
                    "planned_output_dir": str(Path(args.output_dir).expanduser()),
                    "fit_splits": ["train"],
                    "selection_splits": ["val"],
                    "calibration_loaded_for_fit_or_selection": False,
                    "test_loaded_for_fit_or_selection": False,
                    "leakage_audit": audit,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        print("[BBBP_TEACHER_VALIDATE_OK]", flush=True)
        return 0
    summary = train_bbbp_teacher(
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
        split_paths=split_paths,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    print("[BBBP_TEACHER_TRAIN_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
