#!/usr/bin/env python3
"""Audit the frozen train-only BACE GlobalGCE candidate pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_bace_adapter import audit_bace_train_pool  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--expected-parent-count", type=int, default=360)
    args = parser.parse_args(argv)
    audit = audit_bace_train_pool(
        args.run_dir,
        train_csv=args.train_csv,
        expected_parent_count=int(args.expected_parent_count),
        expected_input_train_count=int(args.expected_parent_count),
    )
    output = Path(args.run_dir).expanduser().resolve() / "train_pool_audit.json"
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, sort_keys=True), flush=True)
    print("[BACE_GLOBALGCE_TRAIN_POOL_AUDIT_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
