#!/usr/bin/env python3
"""Independently verify and atomically publish a Taste T5 base adoption."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve(strict=True).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.tastemolnet_t5_clean_base_adoption_v2 import (  # noqa: E402
    PASS_MARKER,
    TasteT5CleanBaseAdoptionError,
    verify_and_publish_clean_base_adoption,
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


def _clean_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise TasteT5CleanBaseAdoptionError("independent verifier checkout is dirty")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=_config, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--final-path", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--expected-attempt-id", required=True)
    parser.add_argument("--expected-generation-token", required=True)
    parser.add_argument("--expected-controller-id", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-source-inventory-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        require_auto_termination_disabled()
        if _clean_commit() != args.expected_git_commit:
            raise TasteT5CleanBaseAdoptionError(
                "verifier checkout commit differs from execution authority"
            )
        publication, verification = verify_and_publish_clean_base_adoption(
            sealed_path=args.sealed,
            final_path=args.final_path,
            source_model=args.source_model,
            expected_attempt_id=args.expected_attempt_id,
            expected_generation_token=args.expected_generation_token,
            expected_controller_id=args.expected_controller_id,
            expected_git_commit=args.expected_git_commit,
            expected_config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
            expected_source_inventory_sha256=(
                args.expected_source_inventory_sha256
            ),
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "final_path": str(publication.final_path),
                    "attempt_id": publication.attempt_id,
                    "generation_token": publication.generation_token,
                    "verification_sha256": publication.verification_sha256,
                    "gate_sha256": publication.gate_sha256,
                    "publish_mode": publication.publish_mode,
                    **verification,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        print(PASS_MARKER, flush=True)
        return 0
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        print(
            f"T5_CLEAN_BASE_VERIFIER_BLOCKED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
