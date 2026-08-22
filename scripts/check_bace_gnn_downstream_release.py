#!/usr/bin/env python3
"""Write a READY-only B8--B14 dependency decision; never a scientific PASS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train.bace_stage_boundaries import (  # noqa: E402
    STAGE_DEPENDENCY_CONTRACT,
    STAGE_SPLIT_CONTRACT,
    validate_pass_dependencies,
    validate_stage_data_access,
)
from src.train.bace_policy_init import atomic_text  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--stage", choices=tuple(STAGE_SPLIT_CONTRACT), required=True)
    parser.add_argument(
        "--requested-split",
        choices=("train", "calibration", "test", "manifest_only"),
        required=True,
    )
    parser.add_argument("--dependency", action="append", type=Path, default=[])
    parser.add_argument("--expected-dependency-stage", action="append", default=[])
    parser.add_argument("--selector-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected_dependencies = STAGE_DEPENDENCY_CONTRACT[args.stage]
    if tuple(args.expected_dependency_stage) != tuple(expected_dependencies):
        raise ValueError(
            "BACE release dependency stages must exactly equal the frozen contract: "
            f"{expected_dependencies}"
        )
    data_access = validate_stage_data_access(
        stage=args.stage,
        requested_split=args.requested_split,
        selector_manifest=args.selector_manifest,
    )
    dependencies = validate_pass_dependencies(
        args.dependency, expected_stages=expected_dependencies
    )
    payload = {
        "schema_version": "bace_downstream_release_v1",
        "stage": args.stage,
        "status": "READY",
        "scientific_stage_pass": False,
        "must_run_stage_payload": True,
        "data_access": data_access,
        "dependencies": dependencies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"BACE release decision output already exists: {args.output}")
    atomic_text(args.output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[BACE_{args.stage}_READY]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
