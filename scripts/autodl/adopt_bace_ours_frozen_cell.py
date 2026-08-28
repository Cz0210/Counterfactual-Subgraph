#!/usr/bin/env python3
"""Publish or reopen the checksum-pinned BACE Ours freeze-adoption receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.bace_ours_freeze_adoption import (  # noqa: E402
    PASS_MARKER,
    adopt_bace_ours_frozen_cell,
    validate_adoption_receipt,
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
    parser.add_argument("action", choices=("adopt", "validate"))
    parser.add_argument("--matrix-root", type=_absolute)
    parser.add_argument("--output-root", type=_absolute, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "adopt":
        if args.matrix_root is None:
            raise ValueError("adopt requires --matrix-root")
        result = adopt_bace_ours_frozen_cell(
            matrix_root=args.matrix_root,
            output_root=args.output_root,
        )
    else:
        if args.matrix_root is not None:
            raise ValueError("validate does not accept --matrix-root")
        result = validate_adoption_receipt(args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(PASS_MARKER, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
