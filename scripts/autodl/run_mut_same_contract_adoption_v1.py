#!/usr/bin/env python3
"""Seal the historical Mut 50k adoption after the current same-contract A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_mut_same_contract_adoption_v1 import (  # noqa: E402
    publish_same_contract_adoption,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise argparse.ArgumentTypeError("absolute non-symlink path required")
    return path.resolve(strict=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--ab-task-spec", type=_absolute, required=True)
    parser.add_argument("--ab-owner-terminal", type=_absolute, required=True)
    parser.add_argument("--same-contract-gate", type=_absolute, required=True)
    parser.add_argument("--authorization-receipt", type=_absolute, required=True)
    parser.add_argument("--historical-source-root", type=_absolute, required=True)
    parser.add_argument("--completed-common-root", type=_absolute, required=True)
    parser.add_argument("--output-root", type=_absolute, required=True)
    parser.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = publish_same_contract_adoption(
        task_spec_path=args.ab_task_spec,
        owner_terminal_path=args.ab_owner_terminal,
        gate_path=args.same_contract_gate,
        authorization_receipt_path=args.authorization_receipt,
        historical_source_root=args.historical_source_root,
        completed_common_root=args.completed_common_root,
        output_root=args.output_root,
        proc_root=args.proc_root,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[MUT_TRACE_ON_50K_ADOPTION_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
