#!/usr/bin/env python3
"""Validate live owners and atomically publish the canonical final16 registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.final16_owner_registry_v1 import (  # noqa: E402
    atomic_write_owner_registry,
    build_owner_registry,
    validate_owner_registry,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise argparse.ArgumentTypeError("absolute non-symlink path required")
    return path.resolve(strict=False)


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"registry input is absent or indirect: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("registry input must contain one JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--input", type=_absolute, required=True)
    parser.add_argument("--output", type=_absolute, required=True)
    parser.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    parser.add_argument("--no-process-check", action="store_true")
    args = parser.parse_args(argv)
    if args.config not in (None, "configs/hpc.yaml"):
        raise ValueError("--config must be configs/hpc.yaml")
    if args.set not in ([], ["inference.fallback_to_heuristic=false"]):
        raise ValueError("unsupported --set override")
    raw = _json(args.input)
    required = {
        "registry_id",
        "matrix_authority_root",
        "tasks",
        "publishers",
        "gpu_leases",
    }
    if set(raw) != required:
        raise ValueError(
            f"registry input keys changed: missing={required - set(raw)}, "
            f"extra={set(raw) - required}"
        )
    value = build_owner_registry(
        registry_id=str(raw["registry_id"]),
        matrix_authority_root=str(raw["matrix_authority_root"]),
        tasks=raw["tasks"],
        publishers=raw["publishers"],
        gpu_leases=raw["gpu_leases"],
        proc_root=args.proc_root,
        check_processes=not args.no_process_check,
    )
    validate_owner_registry(
        value,
        proc_root=args.proc_root,
        check_processes=not args.no_process_check,
    )
    atomic_write_owner_registry(args.output, value)
    print(json.dumps(value, sort_keys=True), flush=True)
    print("[FINAL16_OWNER_REGISTRY_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
