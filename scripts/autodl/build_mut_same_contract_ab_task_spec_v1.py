#!/usr/bin/env python3
"""Seal or validate the bounded Mut same-contract A/B task spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_mut_first_divergence_v1 import atomic_json  # noqa: E402
from src.utils.autodl_mut_same_contract_ab_v1 import (  # noqa: E402
    build_same_contract_ab_spec,
    same_contract_ab_command,
    validate_same_contract_ab_spec,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("expected one JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="action", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--template", type=_absolute, required=True)
    build.add_argument("--output", type=_absolute, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--spec", type=_absolute, required=True)
    command = subparsers.add_parser("command")
    command.add_argument("--spec", type=_absolute, required=True)
    args = parser.parse_args(argv)
    if args.action == "build":
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"A/B task spec output must be fresh: {args.output}")
        result = build_same_contract_ab_spec(_json(args.template))
        atomic_json(args.output, result)
        payload: object = result
    elif args.action == "validate":
        payload = validate_same_contract_ab_spec(_json(args.spec))
    else:
        payload = same_contract_ab_command(
            validate_same_contract_ab_spec(_json(args.spec))
        )
    print(json.dumps(payload, sort_keys=True))
    print("[MUT_SAME_CONTRACT_AB_TASK_SPEC_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
