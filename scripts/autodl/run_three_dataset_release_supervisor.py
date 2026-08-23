#!/usr/bin/env python3
"""Run or inspect the persistent CPU-only three-dataset release supervisor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.three_dataset_release_supervisor import (  # noqa: E402
    ReleaseSupervisor,
    ReleaseSupervisorError,
    read_supervisor_status,
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
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "once"):
        child = commands.add_parser(name)
        child.add_argument("--spec", type=_absolute, required=True)
    status = commands.add_parser("status")
    status.add_argument("--state-root", type=_absolute, required=True)
    return parser


def _validate_compatibility_args(args: argparse.Namespace) -> None:
    if args.config is not None and not Path(args.config).is_file():
        raise ValueError(f"Missing config: {args.config}")
    unsupported = [
        value
        for value in args.set
        if value != "inference.fallback_to_heuristic=false"
    ]
    if unsupported:
        raise ValueError(f"Unsupported --set values: {unsupported}")


def _print_state(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
    state_payload = payload.get("state.json")
    state = str(
        state_payload.get("state") if isinstance(state_payload, dict) else ""
    )
    if state == "PASS":
        print("[MATRIX_12_OF_16_PASS]")
        print("[THREE_DATASET_RELEASE_SUPERVISOR_PASS]")
    elif state == "WAITING_DEPENDENCY":
        print("[THREE_DATASET_RELEASE_WAITING_DEPENDENCY]")
    elif state:
        print(f"[THREE_DATASET_RELEASE_{state}]")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_compatibility_args(args)
        if args.command == "status":
            _print_state(read_supervisor_status(args.state_root))
            return 0
        with ReleaseSupervisor(args.spec) as supervisor:
            return_code = supervisor.run(once=args.command == "once")
        _print_state(read_supervisor_status(supervisor.state_root))
        return return_code
    except (OSError, ReleaseSupervisorError, ValueError) as exc:
        print(
            f"[THREE_DATASET_RELEASE_SUPERVISOR_BLOCKED] "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
