#!/usr/bin/env python3
"""Inspect or build a SEALED-candidate Taste T5 clean-base adoption."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

PROJECT_ROOT = Path(__file__).resolve(strict=True).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.tastemolnet_t5_clean_base_adoption_v2 import (  # noqa: E402
    TasteT5CleanBaseAdoptionError,
    build_clean_base_candidate,
    inspect_clean_chemllm_base,
)
from src.utils.process_identity_v2 import require_auto_termination_disabled  # noqa: E402


def _config(value: str) -> Path:
    selected = Path(value)
    expected = PROJECT_ROOT / "configs/hpc.yaml"
    if selected.resolve(strict=True) != expected:
        raise argparse.ArgumentTypeError(
            "--config must be this checkout's configs/hpc.yaml"
        )
    info = os.lstat(selected)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise argparse.ArgumentTypeError("--config must be one physical file")
    return selected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--config", type=_config, required=True)
    inspect_parser.add_argument("--source-model", type=Path, required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--config", type=_config, required=True)
    build_parser.add_argument("--source-model", type=Path, required=True)
    build_parser.add_argument("--expected-source-inventory-sha256", required=True)
    build_parser.add_argument("--artifact-root", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        require_auto_termination_disabled()
        if args.action == "inspect":
            result = inspect_clean_chemllm_base(args.source_model)
        else:
            artifact_root = args.artifact_root or os.environ.get(
                "MANAGED_ARTIFACT_ROOT"
            )
            attempt_id = os.environ.get("MANAGED_ATTEMPT_ID")
            generation_token = os.environ.get("MANAGED_GENERATION_TOKEN")
            if not artifact_root or not attempt_id or not generation_token:
                raise TasteT5CleanBaseAdoptionError(
                    "managed-v2 worker environment is incomplete"
                )
            result = build_clean_base_candidate(
                source_model=args.source_model,
                artifact_root=artifact_root,
                attempt_id=attempt_id,
                generation_token=generation_token,
                config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                expected_source_inventory_sha256=(
                    args.expected_source_inventory_sha256
                ),
            )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except (TasteT5CleanBaseAdoptionError, OSError, ValueError) as exc:
        print(f"T5_CLEAN_BASE_WORKER_BLOCKED: {exc}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
