#!/usr/bin/env python3
"""Build a disjoint/complete official-DFS partition manifest for T8 gSpan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_hpc_exact import (  # noqa: E402
    build_partition_manifest,
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
    parser.add_argument("--graphs-jsonl", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--official-src", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--min-support", required=True, type=int)
    parser.add_argument("--min-vertices", default=3, type=int)
    parser.add_argument("--max-vertices", default=20, type=int)
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--split-root-indices", default=(0,), type=_indices)
    parser.add_argument("--split-depth", default=3, type=int)
    parser.add_argument("--canary-root-indices", default=(0,), type=_indices)
    parser.add_argument("--included-root-indices", type=_indices)
    parser.add_argument(
        "--included-unit-id",
        action="append",
        default=[],
        help="Exact PREFIX_SUBTREE partition ID to include in a bounded canary",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate_hpc_cli_contract(args.config, args.set)
    report = build_partition_manifest(
        graph_jsonl=args.graphs_jsonl,
        input_manifest=args.input_manifest,
        expected_commit=args.expected_commit,
        official_src=args.official_src,
        output=args.output,
        shard_count=args.shard_count,
        min_support=args.min_support,
        min_vertices=args.min_vertices,
        max_vertices=args.max_vertices,
        top_k=args.top_k,
        split_root_indices=args.split_root_indices,
        split_depth=args.split_depth,
        canary_root_indices=args.canary_root_indices,
        included_root_indices=args.included_root_indices,
        included_unit_ids=args.included_unit_id,
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
