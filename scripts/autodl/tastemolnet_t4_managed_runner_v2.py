#!/usr/bin/env python3
"""Hold GPU1 across the T4 worker, SEALED handoff, verifier, and release ACK."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve(strict=True).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.tastemolnet_t4_oracle_smoke_v2 import (  # noqa: E402
    PHYSICAL_GPU_INDEX,
    TASK_ID,
    TasteT4OracleSmokeError,
    collect_t4_managed_input_hashes,
)
from src.utils.autodl_tastemolnet_main_v2 import (  # noqa: E402
    TasteMainV2AuthorityError,
    create_gpu_lease_activation,
    create_gpu_lease_renewal,
    hold_taste_main_v2_controller_authority,
    inspect_clean_git,
    release_registered_runner_gpu_lock_after_ack,
)
from src.utils.managed_execution_v2 import (  # noqa: E402
    ManagedExecutionV2Error,
    create_managed_attempt,
    create_worker_staging,
    utc_now,
    write_worker_exit,
    write_worker_raw_evidence,
)
from src.utils.process_identity_v2 import (  # noqa: E402
    ProcessIdentityV2Error,
    ProcessSnapshotV2,
    audit_process_lineage,
    capture_process_snapshot,
    register_process_lineage,
    require_auto_termination_disabled,
)
from src.utils.terminal_publisher_v2 import seal_worker_staging  # noqa: E402


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


class _NaturalChildLifetime:
    """Keep the runner and its GPU lock alive until every child exits naturally."""

    def __init__(self) -> None:
        self._children: list[subprocess.Popen[Any]] = []

    def __enter__(self) -> "_NaturalChildLifetime":
        return self

    def spawn(self, *args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        child = subprocess.Popen(*args, **kwargs)
        self._children.append(child)
        return child

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        # No signal authority exists.  On every success/error/interruption path,
        # retain this runner generation (and its flock FD) until children stop
        # by themselves.
        for child in self._children:
            child.wait()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=_absolute, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--stage-root", type=_absolute, required=True)
    parser.add_argument("--final-path", type=_absolute, required=True)
    parser.add_argument("--t3-root", type=_absolute, required=True)
    parser.add_argument("--graph-cache-root", type=_absolute, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--controller-launcher-receipt", type=_absolute, required=True)
    parser.add_argument("--controller-receipt", type=_absolute, required=True)
    parser.add_argument("--controller-anchor-heartbeat", type=_absolute, required=True)
    parser.add_argument("--expected-controller-id", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-git-tree", required=True)
    parser.add_argument("--expected-controller-launcher-receipt-sha256", required=True)
    parser.add_argument("--expected-controller-receipt-sha256", required=True)
    parser.add_argument("--expected-controller-anchor-heartbeat-sha256", required=True)
    parser.add_argument("--gpu-lease", type=_absolute, required=True)
    parser.add_argument("--expected-gpu-lease-uuid", required=True)
    parser.add_argument("--expected-gpu-lease-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args(argv)


def _wait_phase(
    args: argparse.Namespace,
    *,
    attempt_id: str,
    generation_token: str,
    runner: ProcessSnapshotV2,
    phase: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + 45
    while True:
        try:
            with hold_taste_main_v2_controller_authority(
                args.controller_receipt,
                args.controller_anchor_heartbeat,
                args.expected_controller_id,
                args.expected_git_commit,
                args.expected_git_tree,
                35,
                expected_launcher_receipt_path=args.controller_launcher_receipt,
                expected_launcher_receipt_sha256=args.expected_controller_launcher_receipt_sha256,
                expected_receipt_sha256=args.expected_controller_receipt_sha256,
                expected_heartbeat_sha256=args.expected_controller_anchor_heartbeat_sha256,
                expected_task_id=TASK_ID,
                expected_gpu_index=PHYSICAL_GPU_INDEX,
                expected_gpu_uuid=args.gpu_uuid,
                expected_lease_uuid=args.expected_gpu_lease_uuid,
                expected_lease_sha256=args.expected_gpu_lease_sha256,
                expected_attempt_id=attempt_id,
                expected_generation_token=generation_token,
                expected_activation_phase=phase,
                expected_worker_process=runner,
            ) as authority:
                evidence = authority.revalidate()
                if evidence.get("anchor_heartbeat_sequence") != 1:
                    raise TasteMainV2AuthorityError(
                        "T4 controller anchor must be heartbeat sequence 1"
                    )
                return evidence
        except (OSError, ValueError, TasteMainV2AuthorityError) as exc:
            if time.monotonic() >= deadline:
                raise TasteMainV2AuthorityError(
                    f"controller did not acknowledge {phase}: {exc}"
                ) from exc
            time.sleep(0.25)


def _common_child_args(args: argparse.Namespace) -> list[str]:
    return [
        "--config", str(args.config),
        "--set", "inference.fallback_to_heuristic=false",
        "--t3-root", str(args.t3_root),
        "--graph-cache-root", str(args.graph_cache_root),
        "--gpu-uuid", args.gpu_uuid,
        "--controller-launcher-receipt", str(args.controller_launcher_receipt),
        "--controller-receipt", str(args.controller_receipt),
        "--controller-anchor-heartbeat", str(args.controller_anchor_heartbeat),
        "--expected-controller-id", args.expected_controller_id,
        "--expected-git-commit", args.expected_git_commit,
        "--expected-git-tree", args.expected_git_tree,
        "--expected-controller-launcher-receipt-sha256", args.expected_controller_launcher_receipt_sha256,
        "--expected-controller-receipt-sha256", args.expected_controller_receipt_sha256,
        "--expected-controller-anchor-heartbeat-sha256", args.expected_controller_anchor_heartbeat_sha256,
        "--expected-gpu-lease-uuid", args.expected_gpu_lease_uuid,
        "--expected-gpu-lease-sha256", args.expected_gpu_lease_sha256,
        "--batch-size", str(args.batch_size),
    ]


def run(args: argparse.Namespace) -> int:
    require_auto_termination_disabled()
    if args.config.resolve(strict=True) != PROJECT_ROOT / "configs/hpc.yaml":
        raise TasteMainV2AuthorityError("--config must be this checkout's configs/hpc.yaml")
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise TasteMainV2AuthorityError("fail-closed inference override is required")
    commit, tree = inspect_clean_git(PROJECT_ROOT)
    if (commit, tree) != (args.expected_git_commit, args.expected_git_tree):
        raise TasteMainV2AuthorityError("T4 runner checkout differs from controller")
    input_hashes = collect_t4_managed_input_hashes(
        t3_root=args.t3_root,
        graph_cache_root=args.graph_cache_root,
        controller_launcher_receipt_sha256=args.expected_controller_launcher_receipt_sha256,
        controller_receipt_sha256=args.expected_controller_receipt_sha256,
        controller_anchor_heartbeat_sha256=args.expected_controller_anchor_heartbeat_sha256,
        gpu_lease_sha256=args.expected_gpu_lease_sha256,
    )
    runner = capture_process_snapshot(os.getpid())
    with create_managed_attempt(
        stage_root=args.stage_root,
        controller_id=args.expected_controller_id,
        task_id=TASK_ID,
        git_commit=commit,
        config_hash=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        input_hashes=input_hashes,
    ) as attempt, create_worker_staging(attempt) as staging, _NaturalChildLifetime() as children:
        environment = {
            **os.environ,
            "AUTO_TERMINATE_UNCONTROLLED_CHILDREN": "0",
            "MANAGED_ATTEMPT_ID": attempt.attempt_id,
            "MANAGED_GENERATION_TOKEN": staging.generation_token,
            "MANAGED_ARTIFACT_ROOT": str(staging.artifact_root),
            "CUDA_VISIBLE_DEVICES": "1",
            "AUTODL_PHYSICAL_GPU_INDEX": "1",
            "AUTODL_PHYSICAL_GPU_UUID": args.gpu_uuid,
        }
        worker_command = [
            sys.executable,
            "-I",
            "-B",
            str(PROJECT_ROOT / "scripts/autodl/tastemolnet_t4_oracle_smoke_worker_v2.py"),
            *_common_child_args(args),
            "--artifact-root", str(staging.artifact_root),
        ]
        worker = children.spawn(worker_command, cwd=PROJECT_ROOT, env=environment)
        worker_snapshot = capture_process_snapshot(worker.pid)
        try:
            activation = create_gpu_lease_activation(
                controller_receipt_path=args.controller_receipt,
                lease_path=args.gpu_lease,
                expected_lease_sha256=args.expected_gpu_lease_sha256,
                attempt_id=attempt.attempt_id,
                generation_token=staging.generation_token,
                managed_worker=runner,
                training_child=worker_snapshot,
                phase="WORKER_ACTIVE",
            )
            renewal = create_gpu_lease_renewal(
                controller_receipt_path=args.controller_receipt,
                lease_path=args.gpu_lease,
                expected_lease_sha256=args.expected_gpu_lease_sha256,
                attempt_id=attempt.attempt_id,
                generation_token=staging.generation_token,
                sequence=1,
                previous_renewal_sha256=None,
            )
            _wait_phase(
                args,
                attempt_id=attempt.attempt_id,
                generation_token=staging.generation_token,
                runner=runner,
                phase="WORKER_ACTIVE",
            )
        except BaseException:
            worker.wait()
            raise
        lineage = register_process_lineage(
            controller_id=args.expected_controller_id,
            attempt_id=attempt.attempt_id,
            launcher=runner,
            worker=worker_snapshot,
            registered_at=utc_now(),
        )
        raw = write_worker_raw_evidence(
            staging,
            {
                "attempt_manifest": dict(attempt.manifest.payload),
                "process_lineage": lineage.to_dict(),
                "scientific_command": worker_command,
                "artifact_root": str(staging.artifact_root),
                "controller_activation_sha256": activation.sha256,
                "controller_renewal_sha256": renewal.sha256,
            },
        )
        raw.close()
        worker_exit = int(worker.wait())
        audit = audit_process_lineage(
            lineage,
            observed_worker=None,
            launcher_alive=True,
            last_heartbeat=None,
            output_root=staging.artifact_root,
            observed_at=utc_now(),
        )
        exited = write_worker_exit(
            staging,
            {
                "exit_code": worker_exit,
                "process_audit": audit,
                "worker_closed_artifact_writers": True,
            },
        )
        exited.close()
        if worker_exit != 0:
            raise TasteMainV2AuthorityError(
                f"scientific worker exited {worker_exit}; no release was requested"
            )
        sealed = seal_worker_staging(staging)
        waiting = create_gpu_lease_activation(
            controller_receipt_path=args.controller_receipt,
            lease_path=args.gpu_lease,
            expected_lease_sha256=args.expected_gpu_lease_sha256,
            attempt_id=attempt.attempt_id,
            generation_token=staging.generation_token,
            managed_worker=runner,
            activation_sequence=2,
            previous_activation_sha256=activation.sha256,
            phase="WAITING_VERIFIER",
        )
        _wait_phase(
            args,
            attempt_id=attempt.attempt_id,
            generation_token=staging.generation_token,
            runner=runner,
            phase="WAITING_VERIFIER",
        )
        verifier_command = [
            sys.executable,
            "-I",
            "-B",
            str(PROJECT_ROOT / "scripts/autodl/tastemolnet_t4_oracle_smoke_verifier_v2.py"),
            *_common_child_args(args),
            "--sealed", str(sealed.staging_path),
            "--final-path", str(args.final_path),
            "--expected-attempt-id", attempt.attempt_id,
            "--expected-generation-token", staging.generation_token,
        ]
        verifier = children.spawn(verifier_command, cwd=PROJECT_ROOT, env=environment)
        verifier_snapshot = capture_process_snapshot(verifier.pid)
        try:
            verifying = create_gpu_lease_activation(
                controller_receipt_path=args.controller_receipt,
                lease_path=args.gpu_lease,
                expected_lease_sha256=args.expected_gpu_lease_sha256,
                attempt_id=attempt.attempt_id,
                generation_token=staging.generation_token,
                managed_worker=runner,
                training_child=verifier_snapshot,
                activation_sequence=3,
                previous_activation_sha256=waiting.sha256,
                phase="VERIFIER_ACTIVE",
            )
            _wait_phase(
                args,
                attempt_id=attempt.attempt_id,
                generation_token=staging.generation_token,
                runner=runner,
                phase="VERIFIER_ACTIVE",
            )
        except BaseException:
            verifier.wait()
            raise
        verifier_exit = int(verifier.wait())
        if verifier_exit != 0 or not args.final_path.exists():
            raise TasteMainV2AuthorityError(
                f"independent verifier exited {verifier_exit}; no release was requested"
            )
        releasing = create_gpu_lease_activation(
            controller_receipt_path=args.controller_receipt,
            lease_path=args.gpu_lease,
            expected_lease_sha256=args.expected_gpu_lease_sha256,
            attempt_id=attempt.attempt_id,
            generation_token=staging.generation_token,
            managed_worker=runner,
            activation_sequence=4,
            previous_activation_sha256=verifying.sha256,
            phase="RELEASE_REQUESTED",
        )
        release = release_registered_runner_gpu_lock_after_ack(
            controller_receipt_path=args.controller_receipt,
            lease_path=args.gpu_lease,
            expected_lease_sha256=args.expected_gpu_lease_sha256,
            release_activation=releasing,
        )
        print(
            json.dumps(
                {
                    "state": "PASS",
                    "attempt_id": attempt.attempt_id,
                    "generation_token": staging.generation_token,
                    "final_path": str(args.final_path),
                    **release,
                    "auto_terminate_uncontrolled_children": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (
        OSError,
        ValueError,
        ManagedExecutionV2Error,
        ProcessIdentityV2Error,
        TasteMainV2AuthorityError,
        TasteT4OracleSmokeError,
    ) as exc:
        print(
            json.dumps(
                {
                    "state": "QUARANTINED",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "manual_review_required": True,
                    "signal_sent": False,
                    "auto_terminate_uncontrolled_children": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
