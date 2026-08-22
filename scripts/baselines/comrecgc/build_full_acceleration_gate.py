#!/usr/bin/env python3
"""Aggregate exact 500/1000 BACE ComRecGC replay evidence for full 50k."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.full_acceleration_gate import (  # noqa: E402
    build_full_acceleration_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--m500-root", required=True)
    parser.add_argument("--m1000-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = build_full_acceleration_gate(
        m500_root=args.m500_root,
        m1000_root=args.m1000_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, sort_keys=True))
    print("[COMRECGC_PARALLEL_EQUIVALENCE_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
