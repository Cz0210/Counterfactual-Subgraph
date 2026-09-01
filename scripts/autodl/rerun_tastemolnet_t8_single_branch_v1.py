#!/usr/bin/env python3
"""Fresh bounded recovery for exactly one T8 branch rejected by salvage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.tastemolnet_t8_branch_salvage_v1 import (  # noqa: E402
    run_single_branch_recovery,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--source-attempt-id", required=True)
    parser.add_argument("--target", type=int, choices=(0, 2), required=True)
    parser.add_argument("--t3-output", type=Path, required=True)
    parser.add_argument("--t4-output", type=Path, required=True)
    parser.add_argument("--gnn-checkpoint", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--gspan-scratch-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise ValueError("T8 single-branch recovery requires heuristic fallback disabled")
    result = run_single_branch_recovery(
        config=args.config,
        attempt_id=args.attempt_id,
        recovery_source_attempt_id=args.source_attempt_id,
        target=args.target,
        t3_output=args.t3_output,
        t4_output=args.t4_output,
        gnn_checkpoint=args.gnn_checkpoint,
        train_csv=args.train_csv,
        official_root=args.official_root,
        state_root_path=args.state_root,
        gspan_scratch_root=args.gspan_scratch_root,
        device=args.device,
    )
    print("[TASTE_T8_SINGLE_BRANCH_RECOVERY_PASS]", flush=True)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
