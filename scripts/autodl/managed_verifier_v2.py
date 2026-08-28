#!/usr/bin/env python3
"""Independently verify one SEALED managed-v2 artifact and publish it."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from src.utils.managed_execution_v2 import ManagedExecutionV2Error
from src.utils.process_identity_v2 import require_auto_termination_disabled
from src.utils.terminal_publisher_v2 import (
    open_sealed_worker_artifact,
    verify_and_publish_sealed_attempt,
)


def _load_verification(path: Path) -> dict[str, Any]:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if before.st_dev != named.st_dev or before.st_ino != named.st_ino:
            raise ManagedExecutionV2Error(
                "verification input changed before independent open"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, 64 * 1024)
            if not block:
                break
            total += len(block)
            if total > 4 * 1024 * 1024:
                raise ManagedExecutionV2Error("verification input is too large")
            chunks.append(block)
        after = os.fstat(descriptor)
        named_after = os.stat(path, follow_symlinks=False)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or after.st_dev != named_after.st_dev
            or after.st_ino != named_after.st_ino
        ):
            raise ManagedExecutionV2Error(
                "verification input changed while read"
            )
    finally:
        os.close(descriptor)
    raw = json.loads(b"".join(chunks).decode("utf-8"))
    if not isinstance(raw, dict):
        raise ManagedExecutionV2Error("verification input must be a JSON object")
    return raw


def run(args: argparse.Namespace) -> int:
    require_auto_termination_disabled()
    verification = _load_verification(args.verification_json)
    with open_sealed_worker_artifact(
        args.sealed,
        expected_attempt_id=args.expected_attempt_id,
        expected_generation_token=args.expected_generation_token,
    ) as held:
        publication = verify_and_publish_sealed_attempt(
            held,
            final_path=args.final_path,
            verification=verification,
            force_cross_filesystem=args.force_cross_filesystem,
        )
    print(
        json.dumps(
            {
                "schema_version": publication.schema_version,
                "state": "PASS",
                "final_path": str(publication.final_path),
                "attempt_id": publication.attempt_id,
                "generation_token": publication.generation_token,
                "sealed_sha256": publication.sealed_sha256,
                "source_inventory_sha256": publication.source_inventory_sha256,
                "published_inventory_sha256": publication.published_inventory_sha256,
                "verification_sha256": publication.verification_sha256,
                "gate_sha256": publication.gate_sha256,
                "publish_mode": publication.publish_mode,
            },
            sort_keys=True,
        )
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--final-path", type=Path, required=True)
    parser.add_argument("--verification-json", type=Path, required=True)
    parser.add_argument("--expected-attempt-id")
    parser.add_argument("--expected-generation-token")
    parser.add_argument("--force-cross-filesystem", action="store_true")
    parser.add_argument("--config", help="accepted for paired Slurm invocation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        print(
            json.dumps(
                {
                    "state": "QUARANTINED",
                    "quarantine_reason": f"{type(exc).__name__}: {exc}",
                    "science_adopted": False,
                    "downstream_released": False,
                    "manual_review_required": True,
                    "auto_terminate_uncontrolled_children": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
