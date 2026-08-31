#!/usr/bin/env python3
"""Independently close the real-GPU T12 uninterrupted/resume replay gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.tastemolnet_gcf_full_resume import (  # noqa: E402
    CANARY_PASS_MARKER,
    TasteGCFFullResumeError,
    write_canary_gate,
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--uninterrupted", type=_absolute, required=True)
    parser.add_argument("--cross-process-resumed", type=_absolute, required=True)
    parser.add_argument("--output", type=_absolute, required=True)
    return parser


def _read(path: Path, *, label: str) -> dict[str, object]:
    if path.resolve(strict=True) != path or path.is_symlink():
        raise TasteGCFFullResumeError(f"{label} observation is an alias")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteGCFFullResumeError(f"{label} observation is unreadable") from exc
    if type(value) is not dict:
        raise TasteGCFFullResumeError(f"{label} observation is not one object")
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.config.resolve(strict=True) != (REPO_ROOT / "configs/hpc.yaml").resolve(
        strict=True
    ):
        raise TasteGCFFullResumeError("T12 canary requires configs/hpc.yaml")
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise TasteGCFFullResumeError(
            "T12 canary requires fail-closed inference override"
        )
    gate = write_canary_gate(
        args.output,
        _read(args.uninterrupted, label="uninterrupted"),
        _read(args.cross_process_resumed, label="cross-process resumed"),
    )
    print(json.dumps(gate, sort_keys=True), flush=True)
    print(CANARY_PASS_MARKER, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
