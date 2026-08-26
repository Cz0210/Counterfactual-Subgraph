#!/usr/bin/env python3
"""Run or inspect the dedicated persistent TasteMolNet GINE controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_tastemolnet_gine_controller_v1 import (  # noqa: E402
    TasteGINEControllerSpec,
    inspect_tastemolnet_gine_controller,
    run_tastemolnet_gine_controller,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--cid", required=True)
    run.add_argument("--controller-root", type=Path, required=True)
    run.add_argument("--project-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--training-state-root", type=Path, required=True)
    run.add_argument("--worker-wrapper", type=Path, required=True)
    run.add_argument("--resume-controller", action="store_true")
    run.add_argument("--poll-seconds", type=float, default=30.0)
    run.add_argument("--terminal-stability-seconds", type=float, default=2.0)
    status = subparsers.add_parser("status")
    status.add_argument("--controller-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "status":
        result = inspect_tastemolnet_gine_controller(args.controller_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    worker = args.worker_wrapper.expanduser().resolve(strict=True)
    spec = TasteGINEControllerSpec.build(
        cid=args.cid,
        controller_root=args.controller_root,
        project_root=args.project_root,
        output_dir=args.output_dir,
        training_state_root=args.training_state_root,
        worker_argv=("bash", str(worker)),
        poll_seconds=args.poll_seconds,
        terminal_stability_seconds=args.terminal_stability_seconds,
    )
    return run_tastemolnet_gine_controller(
        spec, resume=bool(args.resume_controller)
    )


if __name__ == "__main__":
    raise SystemExit(main())
