#!/usr/bin/env python3
"""Run the read-only BACE ComRecGC 20k/25k resource-cap observer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_bace_comrecgc_resource_cap_observer import ResourceCapObserver


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--convergence-hook-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--science-pid", type=int, required=True)
    parser.add_argument("--science-start-ticks", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    return ResourceCapObserver(
        convergence_hook_root=args.convergence_hook_root,
        state_root=args.state_root,
        science_pid=args.science_pid,
        science_start_ticks=args.science_start_ticks,
        poll_seconds=args.poll_seconds,
    ).run(once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
