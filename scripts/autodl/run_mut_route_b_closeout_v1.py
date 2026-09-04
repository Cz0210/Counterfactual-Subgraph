#!/usr/bin/env python3
"""Fail closed until Mut Route B has a reviewed end-to-end closeout adapter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_mut_successor_stages_v1 import (  # noqa: E402
    write_route_b_adapter_blocker,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise argparse.ArgumentTypeError("absolute non-symlink path required")
    return path.resolve(strict=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--decision", type=_absolute)
    parser.add_argument("--output-root", type=_absolute, required=True)
    args = parser.parse_args(argv)
    if args.config not in (None, "configs/hpc.yaml"):
        raise ValueError("--config must be configs/hpc.yaml")
    if args.set not in ([], ["inference.fallback_to_heuristic=false"]):
        raise ValueError("unsupported --set override")
    decision = args.decision
    if decision is None:
        raw = os.environ.get("MUT_NEXT_ACTION_CONSUMED_PATH")
        if not raw:
            raise ValueError(
                "--decision or MUT_NEXT_ACTION_CONSUMED_PATH is required"
            )
        decision = _absolute(raw)
    result = write_route_b_adapter_blocker(
        decision_path=decision,
        output_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[MUT_ROUTE_B_BLOCKED_ADAPTER_MISSING]", flush=True)
    # Exit zero deliberately: the generic executor reads the typed terminal and
    # converts it to a truthful BLOCKED state.  A nonzero exit would discard
    # that typed evidence before terminal inspection.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
