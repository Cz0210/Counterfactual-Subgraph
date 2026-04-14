#!/usr/bin/env python3
"""Compute base-model capping and validity metrics from a JSONL prediction log."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.base_metrics import compute_base_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jsonl_path",
        help="Path to the base-model inference JSONL log.",
    )
    parser.add_argument(
        "--prediction-field",
        default="prediction",
        help="JSON field containing the generated SMILES string.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    log_path = Path(args.jsonl_path).expanduser().resolve()
    summary = compute_base_metrics(log_path, prediction_field=args.prediction_field)

    print(f"Base log: {log_path}")
    print(f"Evaluated records: {summary.total_records}")
    print(f"Skipped malformed/blank lines: {summary.skipped_records}")
    print(
        "Base Capping Rate: "
        f"{summary.capping_rate:.2f}% "
        f"({summary.capped_records}/{summary.total_records})"
    )
    print(
        "Base Validity: "
        f"{summary.validity_rate:.2f}% "
        f"({summary.valid_records}/{summary.total_records})"
    )


if __name__ == "__main__":
    main()
