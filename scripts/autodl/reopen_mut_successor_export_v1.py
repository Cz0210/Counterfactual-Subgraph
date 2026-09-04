#!/usr/bin/env python3
"""Strictly reopen an existing Mut terminal and seal its figure/table exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_mut_successor_stages_v1 import (  # noqa: E402
    reopen_completed_export,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise argparse.ArgumentTypeError("absolute non-symlink path required")
    return path.resolve(strict=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--terminal-root", type=_absolute, required=True)
    parser.add_argument("--output-root", type=_absolute, required=True)
    parser.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    args = parser.parse_args(argv)
    if args.config not in (None, "configs/hpc.yaml"):
        raise ValueError("--config must be configs/hpc.yaml")
    if args.set not in ([], ["inference.fallback_to_heuristic=false"]):
        raise ValueError("unsupported --set override")
    result = reopen_completed_export(
        terminal_root=args.terminal_root,
        output_root=args.output_root,
        proc_root=args.proc_root,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[MUT_SUCCESSOR_EXPORT_REOPEN_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
