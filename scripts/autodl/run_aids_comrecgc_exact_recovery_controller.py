#!/usr/bin/env python3
"""Run or restart the typed AIDS exact component-recovery controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_aids_comrecgc_exact_recovery_controller_v1 import (  # noqa: E402
    prepare_controller_launch,
    run_controller,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=_absolute, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--launch-mode", choices=("fresh", "resume"))
    parser.add_argument("--context-lines", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds < 0.1 or args.poll_seconds > 60:
        raise ValueError("poll-seconds must be between 0.1 and 60")
    if args.prepare_only:
        if args.resume or args.launch_mode is None:
            raise ValueError(
                "prepare-only requires --launch-mode and does not accept --resume"
            )
        result = prepare_controller_launch(
            args.manifest, resume=args.launch_mode == "resume"
        )
        if args.context_lines:
            for field in (
                "controller_root",
                "cid",
                "launch_id",
                "log_path",
                "pid_path",
                "tmux_session",
                "prelaunch_receipt_path",
                "thread_count",
            ):
                print(result[field])
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
            print("[AIDS_EXACT_RECOVERY_PRELAUNCH_READY]", flush=True)
        return 0
    if args.launch_mode is not None or args.context_lines:
        raise ValueError("launch-mode/context-lines are prepare-only options")
    result = run_controller(
        args.manifest,
        resume=args.resume,
        poll_seconds=args.poll_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("[AIDS_EXACT_RECOVERY_CONTROLLER_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
