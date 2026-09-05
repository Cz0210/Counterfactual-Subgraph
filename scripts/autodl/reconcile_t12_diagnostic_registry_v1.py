#!/usr/bin/env python3
"""Atomically retire the dead T12 accelerated owner and stale GPU lease."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.tastemolnet_t12_diagnostic_reconcile_v1 import (  # noqa: E402
    reconcile_registry_file_after_diagnostic,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise argparse.ArgumentTypeError("absolute non-symlink path required")
    return path.resolve(strict=False)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"expected one JSON object: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--registry", type=_absolute, required=True)
    parser.add_argument("--output", type=_absolute, required=True)
    parser.add_argument("--expected-registry-file-sha256", required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--expected-owner-pid", type=int, required=True)
    parser.add_argument("--expected-owner-start-ticks", type=int, required=True)
    parser.add_argument("--diagnostic-terminal", type=_absolute, required=True)
    parser.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    args = parser.parse_args(argv)
    if args.config not in (None, "configs/hpc.yaml"):
        raise ValueError("--config must be configs/hpc.yaml")
    if args.set not in ([], ["inference.fallback_to_heuristic=false"]):
        raise ValueError("unsupported --set override")
    if args.output != args.registry:
        raise ValueError(
            "registry reconciliation is an in-place CAS; --output must equal --registry"
        )
    value = reconcile_registry_file_after_diagnostic(
        registry_path=args.registry,
        expected_registry_file_sha256=args.expected_registry_file_sha256,
        expected_registry_sha256=args.expected_registry_sha256,
        task_id=args.task_id,
        expected_owner_pid=args.expected_owner_pid,
        expected_owner_start_ticks=args.expected_owner_start_ticks,
        diagnostic_terminal=_json(args.diagnostic_terminal),
        proc_root=args.proc_root,
    )
    print(json.dumps(value, sort_keys=True), flush=True)
    print("[T12_ACCELERATED_OWNER_TERMINAL_RECONCILED]", flush=True)
    print("[T12_STALE_GPU1_LEASE_RELEASED]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
