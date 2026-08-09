#!/usr/bin/env python3
"""Select a frozen BACE WNode-aware action prefix on calibration only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.wnode_prefix_selector import run_bace_wnode_prefix_selector  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=("BACE",), required=True)
    parser.add_argument("--split", choices=("calibration",), required=True)
    parser.add_argument("--matrix-run-dir", required=True)
    parser.add_argument("--thresholds-json", required=True)
    parser.add_argument("--current-selected-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--local-swap-passes", type=int, default=2)
    parser.add_argument("--forbid-test", action="store_true", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.split != "calibration" or not args.forbid_test:
        raise ValueError("BACE selector requires calibration and --forbid-test.")
    summary = run_bace_wnode_prefix_selector(
        matrix_run_dir=args.matrix_run_dir,
        thresholds_json=args.thresholds_json,
        current_selected_csv=args.current_selected_csv,
        output_dir=args.output_dir,
        local_swap_passes=int(args.local_swap_passes),
        fold_count=int(args.fold_count),
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    print("[BACE_WNODE_PREFIX_SELECTION_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
