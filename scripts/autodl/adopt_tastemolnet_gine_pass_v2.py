#!/usr/bin/env python3
"""Independently verify and adopt the completed TasteMolNet T2 GINE result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.tastemolnet_t2_adoption_v2 import (  # noqa: E402
    EXPECTED_SOURCE_COMMIT,
    TasteT2AdoptionError,
    adopt_tastemolnet_t2_scientific_result,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return Path(os.path.normpath(str(path)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--control-root", type=_absolute, required=True)
    parser.add_argument("--artifact-root", type=_absolute, required=True)
    parser.add_argument("--controller-root", type=_absolute, required=True)
    parser.add_argument("--training-state-root", type=_absolute, required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--source-controller-id", required=True)
    parser.add_argument("--expected-source-commit", default=EXPECTED_SOURCE_COMMIT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected_config = PROJECT_ROOT / "configs/hpc.yaml"
    supplied_config = Path(os.path.abspath(args.config.expanduser()))
    if supplied_config != expected_config or not supplied_config.is_file():
        print("TASTE_T2_ADOPTION_BLOCKED: --config must be this checkout's configs/hpc.yaml", file=sys.stderr)
        return 2
    try:
        result = adopt_tastemolnet_t2_scientific_result(
            control_root=args.control_root,
            artifact_root=args.artifact_root,
            controller_root=args.controller_root,
            training_state_root=args.training_state_root,
            project_root=PROJECT_ROOT,
            source_run_id=args.source_run_id,
            source_controller_id=args.source_controller_id,
            expected_source_commit=args.expected_source_commit,
            device=args.device,
            batch_size=args.batch_size,
        )
    except (TasteT2AdoptionError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"T2_ADOPTION_BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(result["marker"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
