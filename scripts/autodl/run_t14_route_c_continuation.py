#!/usr/bin/env python3
"""Own T14 Route C postprocess, final verification, and queued publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.tastemolnet_t14_route_c_continuation import (  # noqa: E402
    run_continuation,
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _config(value: str) -> Path:
    if value == "configs/hpc.yaml":
        return PROJECT_ROOT / value
    return _absolute(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=_config, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--continuation-spec", type=_absolute, required=True)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise ValueError("T14 continuation requires fail-closed inference")
    result = run_continuation(args.continuation_spec, once=args.once)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result.get("status") == "PASS":
        print("[T14_ROUTE_C_MATRIX_CELL_PASS]", flush=True)
        return 0
    print("[T14_ROUTE_C_CONTINUATION_WAITING]", flush=True)
    return 75 if args.once else 0


if __name__ == "__main__":
    raise SystemExit(main())
