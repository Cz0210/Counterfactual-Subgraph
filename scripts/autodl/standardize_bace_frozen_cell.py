#!/usr/bin/env python3
"""Export one frozen BACE terminal into the four-by-four cell schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.bace_frozen_cell_standardization import (  # noqa: E402
    PASS_MARKER,
    standardize_bace_frozen_cell,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument(
        "--method",
        choices=("Ours", "GCFExplainer", "GlobalGCE", "ComRecGC"),
        required=True,
    )
    parser.add_argument("--source-final-root", required=True)
    parser.add_argument("--gnn-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-dataset-hash")
    parser.add_argument("--expected-split-hash")
    parser.add_argument("--expected-molclr-hash")
    parser.add_argument("--expected-threshold-hash")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = standardize_bace_frozen_cell(
        method=args.method,
        source_final_root=args.source_final_root,
        gnn_checkpoint=args.gnn_checkpoint,
        output_dir=args.output_dir,
        expected_dataset_hash=args.expected_dataset_hash,
        expected_split_hash=args.expected_split_hash,
        expected_molclr_hash=args.expected_molclr_hash,
        expected_threshold_hash=args.expected_threshold_hash,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    print(f"{PASS_MARKER} method={result['method']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
