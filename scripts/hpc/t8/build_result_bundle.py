#!/usr/bin/env python3
"""Build a deterministic matrix-inert bundle from exact T8 mining output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_hpc_exact import (  # noqa: E402
    build_result_bundle,
    validate_hpc_cli_contract,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--partition-manifest", required=True, type=Path)
    parser.add_argument("--merge-root", required=True, type=Path)
    parser.add_argument("--parity-receipt", required=True, type=Path)
    parser.add_argument("--output-tar", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--environment-manifest", required=True, type=Path)
    parser.add_argument("--slurm-inventory", required=True, type=Path)
    parser.add_argument("--resource-metrics", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate_hpc_cli_contract(args.config, args.set)
    report = build_result_bundle(
        partition_manifest=args.partition_manifest,
        merge_root=args.merge_root,
        parity_receipt=args.parity_receipt,
        output_tar=args.output_tar,
        output_manifest=args.output_manifest,
        environment_manifest=args.environment_manifest,
        slurm_inventory=args.slurm_inventory,
        resource_metrics=args.resource_metrics,
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
