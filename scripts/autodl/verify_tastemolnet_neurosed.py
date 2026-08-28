#!/usr/bin/env python3
"""Independently verify and atomically publish a Taste NeuroSED bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.tastemolnet_neurosed_gate import verify_bundle  # noqa: E402
from src.utils.process_identity_v2 import require_auto_termination_disabled  # noqa: E402
from src.utils.managed_execution_v2 import (  # noqa: E402
    WORKER_EXIT_SCHEMA,
    WORKER_RAW_EVIDENCE_SCHEMA,
)
from src.utils.terminal_publisher_v2 import (  # noqa: E402
    open_sealed_worker_artifact,
    verify_and_publish_sealed_attempt,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--final-path", type=Path, required=True)
    parser.add_argument("--expected-attempt-id", required=True)
    parser.add_argument("--expected-generation-token")
    parser.add_argument("--require-cuda-tolerance", action="store_true")
    return parser.parse_args(argv)


def _held_json(held: object, relative_path: str) -> dict[str, object]:
    matches = [
        item
        for item in held.files  # type: ignore[attr-defined]
        if item.evidence.relative_path == relative_path
    ]
    if len(matches) != 1:
        raise ValueError(f"SEALED managed metadata is absent: {relative_path}")
    item = matches[0]
    remaining = int(item.evidence.size)
    offset = 0
    chunks: list[bytes] = []
    while remaining:
        block = os.pread(item.descriptor, min(64 * 1024, remaining), offset)
        if not block:
            raise ValueError(f"SEALED managed metadata is short: {relative_path}")
        chunks.append(block)
        remaining -= len(block)
        offset += len(block)
    payload = json.loads(b"".join(chunks).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"SEALED managed metadata is not JSON: {relative_path}")
    return payload


def _verify_managed_binding(
    held: object, scientific_verification: dict[str, object]
) -> dict[str, object]:
    raw = _held_json(held, "raw_evidence.json")
    worker_exit = _held_json(held, "worker_exit.json")
    if (
        raw.get("schema_version") != WORKER_RAW_EVIDENCE_SCHEMA
        or worker_exit.get("schema_version") != WORKER_EXIT_SCHEMA
        or raw.get("attempt_id") != held.sealed.attempt_id  # type: ignore[attr-defined]
        or worker_exit.get("attempt_id") != held.sealed.attempt_id  # type: ignore[attr-defined]
        or raw.get("generation_token") != held.sealed.generation_token  # type: ignore[attr-defined]
        or worker_exit.get("generation_token") != held.sealed.generation_token  # type: ignore[attr-defined]
    ):
        raise ValueError("managed worker raw/exit generation binding changed")
    evidence = raw.get("evidence")
    exit_evidence = worker_exit.get("exit")
    if not isinstance(evidence, dict) or not isinstance(exit_evidence, dict):
        raise ValueError("managed worker evidence payload is malformed")
    attempt = evidence.get("attempt_manifest")
    command = evidence.get("scientific_command")
    process_lineage = evidence.get("process_lineage")
    process_audit = exit_evidence.get("process_audit")
    if (
        not isinstance(attempt, dict)
        or attempt.get("task_id") != "TASTE_GCF_NEUROSED"
        or attempt.get("attempt_id") != held.sealed.attempt_id  # type: ignore[attr-defined]
        or attempt.get("auto_terminate_uncontrolled_children") is not False
        or not isinstance(command, list)
        or not command
        or not isinstance(process_lineage, dict)
        or not isinstance(process_audit, dict)
        or process_audit.get("state") != "EXITED"
        or process_audit.get("attempt_id") != held.sealed.attempt_id  # type: ignore[attr-defined]
        or exit_evidence.get("exit_code") != 0
        or exit_evidence.get("worker_closed_artifact_writers") is not True
    ):
        raise ValueError("managed Taste NeuroSED attempt/exit contract changed")
    command_text = "\0".join(str(value) for value in command)
    if "calibration.csv" in command_text or "test.csv" in command_text:
        raise ValueError("managed Taste NeuroSED command references a forbidden split")
    binding = scientific_verification.get("managed_input_binding")
    if not isinstance(binding, dict):
        raise ValueError("scientific bundle lacks managed input binding")
    expected_inputs = {
        "train_csv": binding.get("train_csv_sha256"),
        "validation_csv": binding.get("validation_csv_sha256"),
        "preparation_split_manifest": binding.get(
            "preparation_split_manifest_sha256"
        ),
    }
    if (
        attempt.get("input_hashes") != expected_inputs
        or attempt.get("config_hash")
        != binding.get("source_execution_config_sha256")
        or attempt.get("git_commit") != binding.get("execution_git_commit")
    ):
        raise ValueError("managed attempt inputs differ from the scientific bundle")
    return {
        "controller_id": attempt.get("controller_id"),
        "task_id": "TASTE_GCF_NEUROSED",
        "attempt_id": held.sealed.attempt_id,  # type: ignore[attr-defined]
        "generation_token": held.sealed.generation_token,  # type: ignore[attr-defined]
        "input_hashes": expected_inputs,
        "config_hash": attempt.get("config_hash"),
        "git_commit": attempt.get("git_commit"),
        "worker_exit_code": 0,
        "worker_closed_artifact_writers": True,
        "auto_terminate_uncontrolled_children": False,
    }


def run(args: argparse.Namespace) -> int:
    require_auto_termination_disabled()
    with open_sealed_worker_artifact(
        args.sealed,
        expected_attempt_id=args.expected_attempt_id,
        expected_generation_token=args.expected_generation_token,
    ) as held:
        verification = verify_bundle(
            held.sealed.artifact_root,
            require_cuda_tolerance=args.require_cuda_tolerance,
        )
        verification["managed_attempt_binding"] = _verify_managed_binding(
            held, verification
        )
        held.revalidate()
        publication = verify_and_publish_sealed_attempt(
            held,
            final_path=args.final_path,
            verification=verification,
        )
    print(
        json.dumps(
            {
                "state": "PASS",
                "final_path": str(publication.final_path),
                "bundle_root": str(publication.final_path / "artifacts"),
                "attempt_id": publication.attempt_id,
                "generation_token": publication.generation_token,
                "best_checkpoint": str(publication.final_path / "artifacts" / "best.pt"),
                "verification_sha256": publication.verification_sha256,
                "gate_sha256": publication.gate_sha256,
                "publish_mode": publication.publish_mode,
                "scientific_verification": verification,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("[TASTE_GCF_NEUROSED_PASS]", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
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
