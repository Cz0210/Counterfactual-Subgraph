#!/usr/bin/env python3
"""Run the isolated T12 buffered-writer/ordered-collector I/O canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.tastemolnet_gcf_ordered_io_canary import (  # noqa: E402
    run_buffered_ordered_io_canary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--checkpoint-at", default=500, type=int)
    parser.add_argument("--post-reload-records", default=10, type=int)
    parser.add_argument("--buffered-batch-records", default=256, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--executor", choices=("thread", "process"), default="process")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.config.expanduser().is_file():
        raise SystemExit("--config must identify an existing regular file")
    if args.set not in ([], ["inference.fallback_to_heuristic=false"]):
        raise SystemExit("this canary accepts only inference.fallback_to_heuristic=false")
    report = run_buffered_ordered_io_canary(
        input_jsonl=args.input_jsonl,
        output_root=args.output_root,
        checkpoint_at=args.checkpoint_at,
        post_reload_records=args.post_reload_records,
        buffered_batch_records=args.buffered_batch_records,
        workers=args.workers,
        executor_kind=args.executor,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
