#!/usr/bin/env python3
"""Reopen sealed T12 diagnostic checkpoints and write a terminal overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.autodl.run_t12_accelerated_from250_v1 import _configure_profile  # noqa: E402
from src.utils.tastemolnet_t12_diagnostic_reconcile_v1 import (  # noqa: E402
    reconcile_diagnostic_510,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise argparse.ArgumentTypeError("absolute non-symlink path required")
    return path.resolve(strict=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--task-spec", type=_absolute, required=True)
    parser.add_argument("--source-terminal", type=_absolute, required=True)
    parser.add_argument("--segment-510-log", type=_absolute, required=True)
    parser.add_argument("--overlay-root", type=_absolute, required=True)
    parser.add_argument("--expected-owner-pid", type=int, required=True)
    parser.add_argument("--expected-owner-start-ticks", type=int, required=True)
    parser.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    args = parser.parse_args(argv)
    if args.config.resolve(strict=True) != (PROJECT_ROOT / "configs/hpc.yaml").resolve(strict=True):
        raise ValueError("T12 diagnostic reconcile requires configs/hpc.yaml")
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise ValueError("T12 diagnostic reconcile requires fail-closed inference")
    import torch
    from src.baselines.tastemolnet_gcf_full_resume import reopen_checkpoint

    _configure_profile()

    def reopener(path: Path, *, expected_identity: dict) -> dict:
        return reopen_checkpoint(path, expected_identity=expected_identity, torch=torch)

    result = reconcile_diagnostic_510(
        task_spec_path=args.task_spec,
        source_terminal_path=args.source_terminal,
        segment_510_log_path=args.segment_510_log,
        overlay_root=args.overlay_root,
        expected_owner_pid=args.expected_owner_pid,
        expected_owner_start_ticks=args.expected_owner_start_ticks,
        proc_root=args.proc_root,
        checkpoint_reopener=reopener,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[T12_DIAGNOSTIC_510_RECONCILIATION_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
