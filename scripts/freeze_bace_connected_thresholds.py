#!/usr/bin/env python3
"""Freeze the shared BACE q-grid from connected calibration actions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.bace_paper_artifacts import (  # noqa: E402
    freeze_bace_connected_thresholds_from_matrix,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--calibration-matrix-dir", required=True)
    parser.add_argument("--output-path", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = freeze_bace_connected_thresholds_from_matrix(
        calibration_matrix_dir=args.calibration_matrix_dir,
        output_path=args.output_path,
    )
    print(json.dumps(payload, sort_keys=True), flush=True)
    print("[BACE_CONNECTED_THRESHOLDS_FROZEN]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
