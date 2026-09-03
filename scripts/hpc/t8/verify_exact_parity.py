#!/usr/bin/env python3
"""Verify event-level parity between official and partitioned T8 gSpan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_hpc_exact import (  # noqa: E402
    validate_hpc_cli_contract,
    verify_exact_parity,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--partition-manifest", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--merged-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate_hpc_cli_contract(args.config, args.set)
    report = verify_exact_parity(
        partition_manifest=args.partition_manifest,
        reference_root=args.reference_root,
        merged_root=args.merged_root,
        output=args.output,
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
