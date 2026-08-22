#!/usr/bin/env python3
"""Publish a fail-closed shared-lowmem A/B benchmark gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_gpu_colocation_gate import (  # noqa: E402
    build_gpu_colocation_gate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument(
        "--single-profile",
        action="append",
        type=Path,
        required=True,
        help="One isolated 10--15 minute profile; provide exactly twice.",
    )
    parser.add_argument("--colocated-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_gpu_colocation_gate(
        single_profile_paths=args.single_profile,
        colocated_profile_path=args.colocated_profile,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    print("[GPU_COLOCATION_BENCHMARK_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
