#!/usr/bin/env python3
"""Run an isolated real-input T8 gSpan root-sharding exactness canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_root_sharding_canary import (  # noqa: E402
    run_exact_root_sharding_canary,
)


def _root_indices(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(",") if item != "")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("root indices must be comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one root index is required")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--graphs-jsonl", required=True, type=Path)
    parser.add_argument("--official-src", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--root-indices", required=True, type=_root_indices)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--min-support", required=True, type=int)
    parser.add_argument("--min-vertices", default=3, type=int)
    parser.add_argument("--max-vertices", default=20, type=int)
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--expected-production-input-fingerprint")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.config.expanduser().is_file():
        raise SystemExit("--config must identify an existing regular file")
    if args.set not in ([], ["inference.fallback_to_heuristic=false"]):
        raise SystemExit("this canary accepts only inference.fallback_to_heuristic=false")
    report = run_exact_root_sharding_canary(
        graph_jsonl=args.graphs_jsonl,
        official_src=args.official_src,
        output_root=args.output_root,
        scratch_root=args.scratch_root,
        root_indices=args.root_indices,
        shard_count=args.shard_count,
        min_support=args.min_support,
        min_vertices=args.min_vertices,
        max_vertices=args.max_vertices,
        top_k=args.top_k,
        expected_production_input_fingerprint=(
            args.expected_production_input_fingerprint
        ),
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
