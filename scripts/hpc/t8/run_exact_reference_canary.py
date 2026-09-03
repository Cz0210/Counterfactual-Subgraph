#!/usr/bin/env python3
"""Run the unpartitioned official gSpan reference on selected canary roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_hpc_exact import (  # noqa: E402
    run_exact_reference,
    validate_hpc_cli_contract,
)


def _indices(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(",") if item != "")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("indices must be comma-separated integers") from exc
    if not parsed or tuple(sorted(set(parsed))) != parsed or parsed[0] < 0:
        raise argparse.ArgumentTypeError("indices must be unique and increasing")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--partition-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--root-indices", type=_indices)
    parser.add_argument("--flush-every", default=256, type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate_hpc_cli_contract(args.config, args.set)
    report = run_exact_reference(
        partition_manifest=args.partition_manifest,
        output_root=args.output_root,
        root_indices=args.root_indices,
        flush_every=args.flush_every,
        scratch_root=args.scratch_root,
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
