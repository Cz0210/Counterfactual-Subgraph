#!/usr/bin/env python3
"""Authorize or publish the one-attempt Taste GlobalGCE valid-zero terminal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.tastemolnet_globalgce_valid_zero import (  # noqa: E402
    build_authorization_receipt,
    publish_valid_zero_result,
    validate_attempt_receipt,
    validate_authorization_receipt,
    validate_terminal_observation,
    validate_valid_zero_source,
)
from src.baselines.tastemolnet_globalgce_full import (  # noqa: E402
    atomic_json,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def _commit(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise argparse.ArgumentTypeError("exact lowercase 40-character Git SHA required")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--source-root", type=_absolute, required=True)
    authorize.add_argument("--attempt-receipt", type=_absolute, required=True)
    authorize.add_argument("--authorization-receipt", type=_absolute, required=True)
    authorize.add_argument("--execution-commit", type=_commit, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--source-root", type=_absolute, required=True)
    finalize.add_argument("--attempt-receipt", type=_absolute, required=True)
    finalize.add_argument("--authorization-receipt", type=_absolute, required=True)
    finalize.add_argument("--recovery-observation", type=_absolute, required=True)
    finalize.add_argument("--test-csv", type=_absolute, required=True)
    finalize.add_argument("--threshold-contract", type=_absolute, required=True)
    finalize.add_argument("--output-root", type=_absolute, required=True)
    finalize.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    finalize.add_argument("--execution-commit", type=_commit, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.source_root.resolve(strict=True)
    attempt = validate_attempt_receipt(args.attempt_receipt, source_root=source)
    if args.command == "authorize":
        destination = args.authorization_receipt
        if destination.exists() or destination.is_symlink():
            raise ValueError("authorization receipt output must be fresh")
        receipt = build_authorization_receipt(
            source_root=source,
            attempt_receipt=attempt,
            execution_commit=args.execution_commit,
        )
        atomic_json(destination, receipt)
        print("[TASTE_GLOBALGCE_VALID_ZERO_RESULT_AUTHORIZED]", flush=True)
        print(f"authorization_receipt={destination}")
        return 0

    authorization = validate_authorization_receipt(
        args.authorization_receipt,
        source_root=source,
        attempt_receipt=attempt,
    )
    observation = validate_terminal_observation(
        args.recovery_observation,
        source_root=source,
        attempt_id=attempt["attempt_id"],
    )
    source_audit = validate_valid_zero_source(source, proc_root=args.proc_root)
    result = publish_valid_zero_result(
        source_audit=source_audit,
        attempt_receipt=attempt,
        authorization=authorization,
        observation=observation,
        test_csv=args.test_csv,
        threshold_contract=args.threshold_contract,
        output_root=args.output_root,
        execution_commit=args.execution_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(result["marker"], flush=True)
    print("[TASTE_GLOBALGCE_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
