#!/usr/bin/env python3
"""Validate and close two existing read-only TasteMolNet T8 branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.tastemolnet_t8_branch_salvage_v1 import (  # noqa: E402
    SALVAGE_MARKER,
    run_salvage,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--source-attempt-id", required=True)
    parser.add_argument("--target-0-root", type=Path, required=True)
    parser.add_argument("--target-2-root", type=Path, required=True)
    parser.add_argument("--t3-output", type=Path, required=True)
    parser.add_argument("--t4-output", type=Path, required=True)
    parser.add_argument("--gnn-checkpoint", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rerun-request", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise ValueError("T8 salvage requires exactly one heuristic-disable override")
    result = run_salvage(
        config=args.config,
        attempt_id=args.attempt_id,
        recovery_source_attempt_id=args.source_attempt_id,
        target_roots={0: args.target_0_root, 2: args.target_2_root},
        t3_output=args.t3_output,
        t4_output=args.t4_output,
        gnn_checkpoint=args.gnn_checkpoint,
        train_csv=args.train_csv,
        official_root=args.official_root,
        state_root=args.state_root,
        output_root=args.output_root,
        rerun_request=args.rerun_request,
        device=args.device,
        proc_root=args.proc_root,
    )
    print(SALVAGE_MARKER, flush=True)
    print(result["t8_marker"], flush=True)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
