#!/usr/bin/env python3
"""Publish a read-only first-divergence audit for two sealed Mut runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_mut_first_divergence_v1 import (  # noqa: E402
    audit_mut_first_divergence,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--legacy-root", type=_absolute, required=True)
    parser.add_argument("--instrumented-root", type=_absolute, required=True)
    parser.add_argument("--task-spec", type=_absolute)
    parser.add_argument("--dataset-summary", type=_absolute)
    parser.add_argument("--output-dir", type=_absolute, required=True)
    parser.add_argument("--timebox-seconds", type=int, default=3600)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timebox_seconds <= 0 or args.timebox_seconds > 3600:
        raise ValueError("Mut divergence audit timebox must be in [1, 3600] seconds")
    started = time.monotonic()
    report = audit_mut_first_divergence(
        legacy_root=args.legacy_root,
        instrumented_root=args.instrumented_root,
        output_dir=args.output_dir,
        task_spec_path=args.task_spec,
        dataset_summary_path=args.dataset_summary,
    )
    elapsed = time.monotonic() - started
    if elapsed > args.timebox_seconds:
        raise TimeoutError(
            f"Mut divergence audit exceeded timebox: {elapsed:.3f}s"
        )
    print(json.dumps({**report, "elapsed_seconds": elapsed}, sort_keys=True))
    print(f"[MUT_FIRST_DIVERGENCE_{report['classification']}_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
