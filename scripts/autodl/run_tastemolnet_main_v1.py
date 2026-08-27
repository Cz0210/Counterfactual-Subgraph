#!/usr/bin/env python3
"""Prepare or run the fresh TasteMolNet policy-v2 main controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_tastemolnet_main_v1 import (  # noqa: E402
    TasteMainControllerError,
    TasteMainSpec,
    inspect_tastemolnet_main,
    prepare_tastemolnet_main,
    run_tastemolnet_main,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def _add_spec_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--controller-id", required=True)
    parser.add_argument("--control-root", required=True, type=_absolute)
    parser.add_argument("--runtime-root", required=True, type=_absolute)
    parser.add_argument("--controller-root", required=True, type=_absolute)
    parser.add_argument("--old-source-manifest", required=True, type=_absolute)
    parser.add_argument("--old-task-root", required=True, type=_absolute)
    parser.add_argument("--policy", required=True, type=_absolute)
    parser.add_argument("--policy-receipt", required=True, type=_absolute)
    parser.add_argument("--prepared-root", required=True, type=_absolute)
    parser.add_argument("--graph-cache-root", required=True, type=_absolute)
    parser.add_argument("--project-root", required=True, type=_absolute)
    parser.add_argument("--gine-controller-root", required=True, type=_absolute)
    parser.add_argument("--gine-output-root", required=True, type=_absolute)
    parser.add_argument("--gine-training-state-root", required=True, type=_absolute)
    parser.add_argument("--reservation-gb", type=int, default=20)
    parser.add_argument("--minimum-free-after-reservations-gb", type=int, default=100)


def _spec(args: argparse.Namespace) -> TasteMainSpec:
    config = Path(args.config).expanduser().resolve(strict=True)
    if args.project_root.resolve(strict=True) != PROJECT_ROOT.resolve(strict=True):
        raise TasteMainControllerError(
            "Taste main project root must equal the loaded immutable CLI root"
        )
    if config != (args.project_root / "configs/hpc.yaml").resolve(strict=True):
        raise TasteMainControllerError("Taste main route freezes configs/hpc.yaml")
    return TasteMainSpec(
        controller_id=args.controller_id,
        control_root=args.control_root.resolve(strict=True),
        runtime_root=args.runtime_root.resolve(strict=True),
        controller_root=args.controller_root,
        old_source_manifest=args.old_source_manifest.resolve(strict=True),
        old_task_root=args.old_task_root.resolve(strict=True),
        policy_path=args.policy.resolve(strict=True),
        policy_receipt=args.policy_receipt.resolve(strict=True),
        prepared_root=args.prepared_root.resolve(strict=True),
        graph_cache_root=args.graph_cache_root.resolve(strict=True),
        project_root=args.project_root.resolve(strict=True),
        gine_controller_root=args.gine_controller_root,
        gine_output_root=args.gine_output_root,
        gine_training_state_root=args.gine_training_state_root,
        reservation_gb=args.reservation_gb,
        minimum_free_after_reservations_gb=args.minimum_free_after_reservations_gb,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "run"):
        child = subparsers.add_parser(action)
        _add_spec_arguments(child)
        if action == "run":
            child.add_argument("--resume", action="store_true")
    status = subparsers.add_parser("status")
    status.add_argument("--config", required=True)
    status.add_argument("--controller-root", required=True, type=_absolute)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "status":
            result = inspect_tastemolnet_main(args.controller_root)
        elif args.action == "prepare":
            result = prepare_tastemolnet_main(_spec(args))
        else:
            return run_tastemolnet_main(_spec(args), resume=bool(args.resume))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"TASTEMOLNET_MAIN_V1_FAILED: {exc}", file=sys.stderr, flush=True)
        return 65
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
