"""Exec the fixed AIDS v5 continuation as a PID-stable private process group."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project = args.project_root.expanduser().resolve(strict=True)
    script = args.script.expanduser()
    if script.is_symlink():
        raise RuntimeError("science continuation may not be a symlink")
    script = script.resolve(strict=True)
    expected = (
        project / "scripts/autodl/run_comrecgc_standardized_continuation.sh"
    ).resolve(strict=True)
    if script != expected or not script.is_file() or not os.access(script, os.X_OK):
        raise RuntimeError("science continuation identity changed")
    os.setsid()
    os.execv("/bin/bash", ["bash", str(script)])
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
