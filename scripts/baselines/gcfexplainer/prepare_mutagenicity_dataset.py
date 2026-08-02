#!/usr/bin/env python3
"""Prepare strict train/validation graphs for official GCFExplainer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.gcfexplainer_mutagenicity_adapter import (  # noqa: E402
    prepare_mutagenicity_dataset,
    write_failure_artifacts,
)


DATA_ROOT = "outputs/hpc/datasets/mutagenicity_v1_teacher_consistent"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument(
        "--train-source-csv",
        default=f"{DATA_ROOT}/train_source_label1_teacher_correct.csv",
    )
    parser.add_argument(
        "--train-target-csv",
        default=f"{DATA_ROOT}/train_target_label0_teacher_correct.csv",
    )
    parser.add_argument(
        "--val-source-csv",
        default=f"{DATA_ROOT}/val_source_label1_teacher_correct.csv",
    )
    parser.add_argument(
        "--val-target-csv",
        default=f"{DATA_ROOT}/val_target_label0_teacher_correct.csv",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--forbid-calibration-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.forbid_calibration_test:
        raise ValueError("Strict preparation requires --forbid-calibration-test.")
    config = vars(args).copy()
    try:
        summary = prepare_mutagenicity_dataset(
            train_source_csv=args.train_source_csv,
            train_target_csv=args.train_target_csv,
            val_source_csv=args.val_source_csv,
            val_target_csv=args.val_target_csv,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        write_failure_artifacts(args.output_dir, error=exc, resolved_config=config)
        raise
    print("[MUTAGENICITY_GCFEXPLAINER_DATASET_OK]", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
