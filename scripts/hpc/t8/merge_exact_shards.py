#!/usr/bin/env python3
"""Merge sealed T8 gSpan shards into the official global DFS preorder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_hpc_exact import (  # noqa: E402
    merge_exact_shards,
    validate_hpc_cli_contract,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--partition-manifest", required=True, type=Path)
    parser.add_argument("--shards-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--scratch-root", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate_hpc_cli_contract(args.config, args.set)
    report = merge_exact_shards(
        partition_manifest=args.partition_manifest,
        shards_root=args.shards_root,
        output_root=args.output_root,
        scratch_root=args.scratch_root,
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
