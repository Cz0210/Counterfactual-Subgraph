#!/usr/bin/env python3
"""Merge exact T8 shards in node-local storage and publish one compact bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_hpc_exact import validate_hpc_cli_contract  # noqa: E402
from src.baselines.globalgce_hpc_storage_safe import (  # noqa: E402
    DEFAULT_MIN_RESERVE_BYTES,
    DEFAULT_RESERVE_FRACTION,
    merge_package_storage_safe,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--partition-manifest", required=True, type=Path)
    parser.add_argument("--shards-root", required=True, type=Path)
    parser.add_argument("--parity-receipt", required=True, type=Path)
    parser.add_argument("--environment-manifest", required=True, type=Path)
    parser.add_argument("--slurm-inventory", required=True, type=Path)
    parser.add_argument("--resource-metrics", required=True, type=Path)
    parser.add_argument("--packaging-commit", required=True)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--minimum-reserve-bytes",
        type=int,
        default=DEFAULT_MIN_RESERVE_BYTES,
    )
    parser.add_argument(
        "--reserve-fraction",
        type=float,
        default=DEFAULT_RESERVE_FRACTION,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate_hpc_cli_contract(args.config, args.set)
    if args.minimum_reserve_bytes < DEFAULT_MIN_RESERVE_BYTES:
        raise SystemExit(
            "--minimum-reserve-bytes may not weaken the 2 GiB production floor"
        )
    if args.reserve_fraction < DEFAULT_RESERVE_FRACTION:
        raise SystemExit(
            "--reserve-fraction may not weaken the 20 percent production floor"
        )
    report = merge_package_storage_safe(
        partition_manifest=args.partition_manifest,
        shards_root=args.shards_root,
        parity_receipt=args.parity_receipt,
        environment_manifest=args.environment_manifest,
        slurm_inventory=args.slurm_inventory,
        resource_metrics=args.resource_metrics,
        packaging_commit=args.packaging_commit,
        scratch_root=args.scratch_root,
        output_root=args.output_root,
        require_distinct_filesystems=True,
        minimum_reserve_bytes=args.minimum_reserve_bytes,
        reserve_fraction=args.reserve_fraction,
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
