#!/usr/bin/env python3
"""Validate/build the fresh terminal-only AIDS ComRecGC v5 controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_aids_comrecgc_exact_route_v5 import (  # noqa: E402
    build_manifest,
    build_payload,
    validate_payload,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="action", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--spec", type=_absolute, required=True)
    build = commands.add_parser("build")
    build.add_argument("--spec", type=_absolute, required=True)
    build.add_argument("--output", type=_absolute, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "validate":
        payload, summary = build_payload(spec_path=args.spec)
        result = {**summary, **validate_payload(payload)}
        marker = "[AIDS_COMRECGC_EXACT_ROUTE_V5_VALIDATE_PASS]"
    else:
        result = build_manifest(spec_path=args.spec, output_path=args.output)
        marker = "[AIDS_COMRECGC_EXACT_ROUTE_V5_BUILD_PASS]"
    print(json.dumps(result, indent=2, sort_keys=True))
    print(marker, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
