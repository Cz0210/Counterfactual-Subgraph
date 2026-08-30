#!/usr/bin/env python3
"""Independently verify and receipt one stopped AIDS exact checkpoint."""

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
    publish_exact_checkpoint_adoption_receipt,
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
    parser.add_argument("--expected-progress-rows", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_progress_rows <= 0:
        raise ValueError("expected-progress-rows must be positive")
    result = publish_exact_checkpoint_adoption_receipt(
        args.manifest,
        expected_progress_rows=args.expected_progress_rows,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("[AIDS_EXACT_CHECKPOINT_ADOPTION_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
