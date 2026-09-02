#!/usr/bin/env python3
"""Wait for the sole T8/T13-grade recovery and route its terminal outcome."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import re
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.tastemolnet_globalgce_full import atomic_json  # noqa: E402
from src.eval.tastemolnet_globalgce_valid_zero import (  # noqa: E402
    build_authorization_receipt,
    validate_attempt_receipt,
)
from src.eval.tastemolnet_globalgce_valid_zero_relay import (  # noqa: E402
    wait_and_finalize,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def _commit(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise argparse.ArgumentTypeError("exact lowercase Git SHA required")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=_absolute, required=True)
    parser.add_argument("--attempt-receipt", type=_absolute, required=True)
    parser.add_argument("--authorization-receipt", type=_absolute, required=True)
    parser.add_argument("--test-csv", type=_absolute, required=True)
    parser.add_argument("--threshold-contract", type=_absolute, required=True)
    parser.add_argument("--valid-zero-output-root", type=_absolute, required=True)
    parser.add_argument("--control-root", type=_absolute, required=True)
    parser.add_argument("--lease-path", type=_absolute, required=True)
    parser.add_argument("--science-pid", type=int, required=True)
    parser.add_argument("--science-start-ticks", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    parser.add_argument("--execution-commit", type=_commit, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.science_pid <= 0 or args.science_start_ticks <= 0:
        raise ValueError("positive science PID and start ticks are required")
    source = args.source_root.resolve(strict=True)
    attempt_path = args.attempt_receipt.resolve(strict=True)
    test_csv = args.test_csv.resolve(strict=True)
    threshold = args.threshold_contract.resolve(strict=True)
    control = args.control_root
    if control.is_symlink():
        raise ValueError("control root cannot be a symlink")
    control.mkdir(parents=True, exist_ok=True)
    control = control.resolve(strict=True)
    lease = args.lease_path
    if lease.is_symlink():
        raise ValueError("relay lease cannot be a symlink")
    lease.parent.mkdir(parents=True, exist_ok=True)
    with lease.open("a+b") as lease_handle:
        try:
            fcntl.flock(lease_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another valid-zero relay owns the canonical lease") from exc
        terminal_path = control / "terminal.json"
        if terminal_path.is_file():
            terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
            print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)
            return 0 if terminal.get("status") in {"PASS", "NORMAL_PATH"} else 2
        attempt = validate_attempt_receipt(attempt_path, source_root=source)
        authorization_path = args.authorization_receipt
        if not authorization_path.exists():
            if authorization_path.is_symlink():
                raise ValueError("authorization output cannot be a symlink")
            receipt = build_authorization_receipt(
                source_root=source,
                attempt_receipt=attempt,
                execution_commit=args.execution_commit,
            )
            atomic_json(authorization_path, receipt)
        terminal = wait_and_finalize(
            source_root=source,
            attempt_receipt_path=attempt_path,
            authorization_receipt_path=authorization_path.resolve(strict=True),
            test_csv=test_csv,
            threshold_contract=threshold,
            output_root=args.valid_zero_output_root,
            control_root=control,
            execution_commit=args.execution_commit,
            science_pid=args.science_pid,
            science_start_ticks=args.science_start_ticks,
            poll_seconds=args.poll_seconds,
            proc_root=args.proc_root.resolve(strict=True),
        )
        print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)
        if terminal.get("status") == "PASS":
            print("[TASTE_GLOBALGCE_VALID_ZERO_RESULT_PASS]", flush=True)
            print("[TASTE_GLOBALGCE_PASS]", flush=True)
            return 0
        if terminal.get("status") == "NORMAL_PATH":
            print("[TASTE_GLOBALGCE_NORMAL_PATH]", flush=True)
            return 0
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
