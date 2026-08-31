#!/usr/bin/env python3
"""Adopt completed AIDS exact DBSCAN and run fresh standardized postprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_aids_comrecgc_exact_postprocess_v1 import (  # noqa: E402
    run_aids_exact_postprocess,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--controller-manifest", type=_absolute, required=True)
    parser.add_argument("--exact-receipt", type=_absolute, required=True)
    parser.add_argument("--output-root", type=_absolute, required=True)
    parser.add_argument("--heartbeat-path", type=_absolute, required=True)
    parser.add_argument("--heartbeat-interval-seconds", type=float, default=60.0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if Path(args.config) != Path("configs/hpc.yaml"):
        raise SystemExit("--config must be configs/hpc.yaml")
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise SystemExit(
            "exact postprocess requires exactly "
            "--set inference.fallback_to_heuristic=false"
        )
    result = run_aids_exact_postprocess(
        controller_manifest_path=args.controller_manifest,
        exact_receipt_path=args.exact_receipt,
        output_root=args.output_root,
        heartbeat_path=args.heartbeat_path,
        resume=args.resume,
        max_workers=args.max_workers,
        heartbeat_interval_seconds=args.heartbeat_interval_seconds,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
