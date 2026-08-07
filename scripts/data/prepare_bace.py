#!/usr/bin/env python3
"""Prepare a deterministic BACE graph dataset without downloading data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bace_adapter import DEFAULT_SPLIT_RATIOS, prepare_bace_dataset  # noqa: E402


def _ratios(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(parsed) != 4:
        raise argparse.ArgumentTypeError("Expected train,val,calibration,test ratios.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--raw-csv", default="data/raw/BACE/bace.csv")
    parser.add_argument("--output-dir", default="data/processed/BACE")
    parser.add_argument("--raw-smiles-col", default="smiles")
    parser.add_argument("--raw-label-col", default="label")
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument(
        "--split-ratios",
        type=_ratios,
        default=DEFAULT_SPLIT_RATIOS,
        help="Comma-separated train,val,calibration,test ratios.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = prepare_bace_dataset(
        raw_csv=args.raw_csv,
        output_dir=args.output_dir,
        raw_smiles_col=args.raw_smiles_col,
        raw_label_col=args.raw_label_col,
        split_seed=args.split_seed,
        split_ratios=args.split_ratios,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    print("[BACE_DATASET_PREPARE_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
