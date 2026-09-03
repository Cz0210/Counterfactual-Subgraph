#!/usr/bin/env python3
"""Execute the strictly gated BACE five-backbone two-lane ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.gnn.five_backbone_execution import (  # noqa: E402
    FiveBackboneExecutionError,
    load_execution_spec,
    load_launch_evidence,
    run_five_backbone_execution,
)


def _commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip().lower()


def _runtime_config(path: Path, overrides: list[str]) -> Path:
    candidate = path if path.is_absolute() else PROJECT_ROOT / path
    if candidate.is_symlink():
        raise FiveBackboneExecutionError("runtime config must be a physical file")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise FiveBackboneExecutionError("runtime config is not a file")
    if "inference.fallback_to_heuristic=false" not in set(overrides):
        raise FiveBackboneExecutionError(
            "five-backbone execution requires heuristic fallback disabled"
        )
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hpc.yaml"))
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--status-sha256", required=True)
    parser.add_argument("--run-spec", type=Path, required=True)
    parser.add_argument("--run-spec-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--main-ready-gpu-tasks", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    _runtime_config(args.config, list(args.overrides))
    spec = load_execution_spec(
        args.run_spec,
        args.run_spec_sha256,
        project_root=PROJECT_ROOT,
    )
    if Path(spec.output_root) != args.output_root.expanduser():
        raise FiveBackboneExecutionError("CLI output_root differs from the immutable run spec")
    if spec.execution_commit != _commit():
        raise FiveBackboneExecutionError("execution commit differs from deployed checkout")
    launch = load_launch_evidence(args.status, args.status_sha256)
    return run_five_backbone_execution(
        spec,
        launch,
        main_ready_gpu_tasks=args.main_ready_gpu_tasks,
        resume=bool(args.resume),
        poll_seconds=float(args.poll_seconds),
    )


def main() -> int:
    result = run(parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("state") in {"PASS", "PAUSED_MAIN_PRIORITY"} else 75


if __name__ == "__main__":
    raise SystemExit(main())
