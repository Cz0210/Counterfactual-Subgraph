#!/usr/bin/env python3
"""Run one UUID-scoped managed-v2 worker and seal only raw evidence.

This entrypoint never writes PASS, a verifier gate, an adoption receipt, or a
release marker, and it never sends a signal to a child.  A successful worker
produces only artifacts plus ``raw_evidence.json``, ``worker_exit.json``, and
``SEALED.json`` for a separate verifier process.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from src.utils.managed_execution_v2 import (
    ManagedExecutionV2Error,
    create_managed_attempt,
    create_worker_staging,
    utc_now,
    write_worker_exit,
    write_worker_raw_evidence,
)
from src.utils.process_identity_v2 import (
    audit_process_lineage,
    capture_process_snapshot,
    register_process_lineage,
    require_auto_termination_disabled,
)
from src.utils.terminal_publisher_v2 import seal_worker_staging


def _input_hash(value: str) -> tuple[str, str]:
    name, separator, digest = value.partition("=")
    if not separator or not name or len(digest) != 64:
        raise argparse.ArgumentTypeError("input hash must be NAME=SHA256")
    return name, digest


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        info = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if info.st_dev != named.st_dev or info.st_ino != named.st_ino:
            raise ManagedExecutionV2Error("raw evidence input changed while opened")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, 64 * 1024)
            if not block:
                break
            total += len(block)
            if total > 4 * 1024 * 1024:
                raise ManagedExecutionV2Error("raw evidence input is too large")
            chunks.append(block)
    finally:
        os.close(descriptor)
    payload = json.loads(b"".join(chunks).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ManagedExecutionV2Error("raw evidence input must be a JSON object")
    return payload


def run(args: argparse.Namespace) -> int:
    require_auto_termination_disabled()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ManagedExecutionV2Error("managed worker command is absent")
    input_hashes = dict(args.input_hash)
    with create_managed_attempt(
        stage_root=args.stage_root,
        controller_id=args.controller_id,
        task_id=args.task_id,
        git_commit=args.git_commit,
        config_hash=args.config_hash,
        input_hashes=input_hashes,
        attempt_id=args.attempt_id,
    ) as attempt:
        with create_worker_staging(attempt) as staging:
            launcher = capture_process_snapshot(os.getpid())
            environment = dict(os.environ)
            environment["AUTO_TERMINATE_UNCONTROLLED_CHILDREN"] = "0"
            environment["MANAGED_ATTEMPT_ID"] = attempt.attempt_id
            environment["MANAGED_GENERATION_TOKEN"] = staging.generation_token
            environment["MANAGED_ARTIFACT_ROOT"] = str(staging.artifact_root)
            child = subprocess.Popen(
                command,
                cwd=args.cwd or staging.artifact_root,
                env=environment,
            )
            worker = capture_process_snapshot(child.pid)
            lineage = register_process_lineage(
                controller_id=args.controller_id,
                attempt_id=attempt.attempt_id,
                launcher=launcher,
                worker=worker,
                registered_at=utc_now(),
            )
            raw = _load_json(args.raw_evidence_json)
            raw.update(
                {
                    "attempt_manifest": dict(attempt.manifest.payload),
                    "process_lineage": lineage.to_dict(),
                    "scientific_command": command,
                    "artifact_root": str(staging.artifact_root),
                }
            )
            raw_document = write_worker_raw_evidence(staging, raw)
            raw_document.close()
            exit_code = int(child.wait())
            observed = audit_process_lineage(
                lineage,
                observed_worker=None,
                launcher_alive=True,
                last_heartbeat=None,
                output_root=staging.artifact_root,
                observed_at=utc_now(),
            )
            exit_document = write_worker_exit(
                staging,
                {
                    "exit_code": exit_code,
                    "process_audit": observed,
                    "worker_closed_artifact_writers": True,
                },
            )
            exit_document.close()
            if exit_code != 0:
                print(
                    json.dumps(
                        {
                            "state": "QUARANTINED",
                            "quarantine_reason": "WORKER_NONZERO_EXIT",
                            "attempt_id": attempt.attempt_id,
                            "output_root": str(staging.artifact_root),
                            "manual_review_required": True,
                            "auto_terminate_uncontrolled_children": False,
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                )
                return exit_code
            sealed = seal_worker_staging(staging)
            print(
                json.dumps(
                    {
                        "state": "SEALED",
                        "attempt_id": sealed.attempt_id,
                        "generation_token": sealed.generation_token,
                        "staging_path": str(sealed.staging_path),
                        "artifact_root": str(sealed.artifact_root),
                        "sealed_path": str(sealed.seal_path),
                        "sealed_sha256": sealed.seal_sha256,
                        "inventory_sha256": sealed.inventory_sha256,
                        "independent_verification_required": True,
                    },
                    sort_keys=True,
                )
            )
            return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--controller-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--input-hash", action="append", type=_input_hash, default=[])
    parser.add_argument("--attempt-id")
    parser.add_argument("--raw-evidence-json", type=Path)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--config", help="accepted for paired Slurm invocation")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (ManagedExecutionV2Error, OSError, ValueError) as exc:
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
