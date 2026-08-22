#!/usr/bin/env python3
"""Create fail-closed BACE GCFExplainer equivalence and performance gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.gcfexplainer_acceleration import (  # noqa: E402
    build_acceleration_gate,
    compare_same_gpu_profiles,
    compare_vrrw_equivalence,
    write_fresh_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)

    equivalence = commands.add_parser("equivalence")
    equivalence.add_argument("--legacy-root", type=Path, required=True)
    equivalence.add_argument("--optimized-root", type=Path, required=True)
    equivalence.add_argument("--budget", type=int, choices=(500, 1000), required=True)
    equivalence.add_argument("--output", type=Path, required=True)

    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--legacy-root", type=Path, required=True)
    benchmark.add_argument("--optimized-root", type=Path, required=True)
    benchmark.add_argument("--equivalence-marker", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument(
        "--equivalence-marker", type=Path, action="append", required=True
    )
    aggregate.add_argument("--benchmark-marker", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.action == "equivalence":
        payload = compare_vrrw_equivalence(
            args.legacy_root.resolve(strict=True),
            args.optimized_root.resolve(strict=True),
            budget=args.budget,
        )
        marker = "[BACE_GCF_EQUIVALENCE_PASS]"
    elif args.action == "benchmark":
        payload = compare_same_gpu_profiles(
            legacy_root=args.legacy_root.resolve(strict=True),
            optimized_root=args.optimized_root.resolve(strict=True),
            equivalence_marker=args.equivalence_marker.resolve(strict=True),
        )
        marker = "[BACE_GCF_AB_BENCHMARK_PASS]"
    else:
        if len(args.equivalence_marker) != 2:
            raise ValueError("aggregate requires exactly two equivalence markers")
        benchmark = json.loads(
            args.benchmark_marker.resolve(strict=True).read_text(encoding="utf-8")
        )
        payload = build_acceleration_gate(
            equivalence_markers=[
                value.resolve(strict=True) for value in args.equivalence_marker
            ],
            benchmark=benchmark,
        )
        marker = "[BACE_GCF_ACCELERATION_GATE_PASS]"
    write_fresh_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    if payload.get("status") == "PASS":
        print(marker, flush=True)
        return 0
    print("[BACE_GCF_ACCELERATION_GATE_FAILED]", file=sys.stderr, flush=True)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
