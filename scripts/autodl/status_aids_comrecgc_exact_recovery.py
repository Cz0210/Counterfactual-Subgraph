#!/usr/bin/env python3
"""Read-only status for the typed AIDS exact component-recovery controller."""

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
    RecoveryControllerError,
    controller_status,
    load_bound_controller_manifest,
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
    parser.add_argument("--require-launchable", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = controller_status(args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_launchable:
        try:
            load_bound_controller_manifest(args.manifest)
        except RecoveryControllerError as exc:
            print(
                f"[AIDS_EXACT_RECOVERY_NOT_LAUNCHABLE] {exc}", file=sys.stderr
            )
            return 78
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
