#!/usr/bin/env python3
"""Audit a strict train-only Mutagenicity GlobalGCE candidate pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.globalgce_mutagenicity_adapter import (  # noqa: E402
    DEFAULT_EXPECTED_PARENT_COUNT,
    audit_mutagenicity_train_pool,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument(
        "--expected-parent-count",
        type=int,
        default=DEFAULT_EXPECTED_PARENT_COUNT,
    )
    parser.add_argument(
        "--require-target-label-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-unique-universe",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--forbid-calibration-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--config", default=None, help="HPC compatibility config.")
    parser.add_argument("--set", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit_mutagenicity_train_pool(
        args.run_dir,
        train_csv=args.train_csv,
        expected_parent_count=int(args.expected_parent_count),
        require_target_label_zero=bool(args.require_target_label_zero),
        require_unique_universe=bool(args.require_unique_universe),
        forbid_calibration_test=bool(args.forbid_calibration_test),
        require_complete=bool(args.require_complete),
    )
    output_path = Path(args.run_dir).expanduser().resolve() / "train_pool_audit.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("[MUTAGENICITY_GLOBALGCE_TRAIN_POOL_AUDIT_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
