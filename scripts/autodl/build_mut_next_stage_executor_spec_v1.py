#!/usr/bin/env python3
"""Seal an immutable Mut post-A/B successor task specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_mut_next_stage_executor_v1 import (  # noqa: E402
    atomic_json,
    build_successor_spec,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise argparse.ArgumentTypeError("absolute non-symlink path required")
    return path.resolve(strict=False)


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"template is absent or indirect: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("template must contain one JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--template", type=_absolute, required=True)
    parser.add_argument("--output", type=_absolute, required=True)
    parser.add_argument("--no-file-check", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"successor spec output must be fresh: {args.output}")
    value = build_successor_spec(
        _json(args.template), check_files=not args.no_file_check
    )
    atomic_json(args.output, value)
    print(json.dumps(value, sort_keys=True), flush=True)
    print("[MUT_SUCCESSOR_CHAIN_PREDEPLOYED]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
