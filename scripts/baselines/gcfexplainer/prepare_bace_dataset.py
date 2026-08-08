#!/usr/bin/env python3
"""Prepare frozen BACE train/validation graphs for official GCFExplainer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.gcfexplainer_bace_adapter import prepare_bace_gcf_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--train-source-csv", required=True)
    parser.add_argument("--train-target-csv", required=True)
    parser.add_argument("--val-source-csv", required=True)
    parser.add_argument("--val-target-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_bace_gcf_dataset(
        train_source_csv=args.train_source_csv,
        train_target_csv=args.train_target_csv,
        val_source_csv=args.val_source_csv,
        val_target_csv=args.val_target_csv,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[BACE_GCFEXPLAINER_DATASET_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
