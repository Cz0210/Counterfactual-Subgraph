#!/usr/bin/env python3
"""Own or verify the T12 checkpoint-250 accelerated diagnostic branch."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.main_ready_task_specs import atomic_json, load_spec, process_identity  # noqa: E402
from src.utils.tastemolnet_t12_accelerated_from250 import (  # noqa: E402
    compare_checkpoint_payloads,
    file_sha256,
    fork_step250_prefix,
    validate_reference_step250,
)


ACCELERATED_TOTAL_STEPS = 510
# The sealed step-250 checkpoint authenticates this diagnostic schedule.  It
# cannot be relabelled to add intermediate durable boundaries without a full
# journal re-emission, so the authorized parallel arm retains the exact source
# schedule and writes lightweight progress separately.
ACCELERATED_CHECKPOINT_CURSORS = (250, 500, 510)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _configure_profile() -> None:
    from src.baselines import tastemolnet_gcf_full as full
    from src.baselines import tastemolnet_gcf_full_resume as resume
    from src.baselines import tastemolnet_gcf_production_state as state
    from src.baselines import tastemolnet_gcf_transition_store as transitions

    state.PINNED_TOTAL_STEPS = ACCELERATED_TOTAL_STEPS
    state.PINNED_CHECKPOINT_CURSORS = ACCELERATED_CHECKPOINT_CURSORS
    state.HISTORY_MAX_SEGMENTS = len(ACCELERATED_CHECKPOINT_CURSORS)
    transitions.TRANSITION_MAX_SEGMENTS = len(ACCELERATED_CHECKPOINT_CURSORS)
    resume.PRODUCTION_TOTAL_STEPS = ACCELERATED_TOTAL_STEPS
    resume.PRODUCTION_CHECKPOINT_CURSORS = frozenset(
        ACCELERATED_CHECKPOINT_CURSORS
    )
    full.PRODUCTION_TOTAL_STEPS = ACCELERATED_TOTAL_STEPS
    full.PRODUCTION_CHECKPOINT_CURSORS = frozenset(
        ACCELERATED_CHECKPOINT_CURSORS
    )


def _expected_identity(run_identity_path: Path, cursor: int) -> dict[str, Any]:
    from src.baselines.tastemolnet_gcf_full_resume import production_checkpoint_identity

    value = json.loads(run_identity_path.read_text(encoding="utf-8"))
    return production_checkpoint_identity(
        value["identity_template"], checkpoint_cursor=cursor
    )


def _segment(args: argparse.Namespace) -> int:
    spec = load_spec(args.task_spec)
    contract = spec["science_contract"]
    _configure_profile()
    from src.baselines.tastemolnet_gcf_full import run_t12_generation_segment

    result = run_t12_generation_segment(
        mode="resume",
        output_root=spec["output_root"],
        checkpoint_manifest=args.checkpoint_manifest,
        attempt_id=contract["source_science_attempt_id"],
        generation_token=contract["generation_token"],
        gpu_uuid=spec["gpu_request"]["uuid"],
        managed_neurosed_root=contract["managed_neurosed_root"],
        t3_root=contract["t3_root"],
        official_root=contract["official_root"],
        threshold_authority_path=contract["threshold_authority"],
        replay_gate_path=contract["replay_gate"],
        resume_run_identity_authority=(Path(spec["output_root"]) / "run_identity.json"),
        disposable_index_root=contract["disposable_index_root"],
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def _parity(args: argparse.Namespace) -> int:
    import torch
    from src.baselines.tastemolnet_gcf_full_resume import reopen_checkpoint

    spec = load_spec(args.task_spec)
    contract = spec["science_contract"]
    _configure_profile()
    authority = Path(contract["source_reference_root"]) / "run_identity.json"
    reports: dict[str, Any] = {}
    for cursor in (500, 510):
        expected = _expected_identity(authority, cursor)
        reference_path = Path(contract[f"reference_checkpoint_{cursor}"])
        accelerated_path = Path(contract[f"accelerated_checkpoint_{cursor}"])
        reference = reopen_checkpoint(reference_path, expected_identity=expected, torch=torch)
        accelerated = reopen_checkpoint(
            accelerated_path, expected_identity=expected, torch=torch
        )
        reports[str(cursor)] = compare_checkpoint_payloads(
            reference=reference, accelerated=accelerated
        )
    receipt = {
        "schema_version": "tastemolnet_t12_accelerated_250_500_510_endpoint_v1",
        "status": "ENDPOINT_ONLY_PASS_PROMOTION_BLOCKED",
        "comparison_scope": "ENDPOINT_CHECKPOINT_STATE_ONLY",
        "source_checkpoint_250_manifest_sha256": contract[
            "source_checkpoint_250_manifest_sha256"
        ],
        "source_checkpoint_250_payload_sha256": contract[
            "source_checkpoint_250_payload_sha256"
        ],
        "source_checkpoint_250_rng_sha256": contract[
            "source_checkpoint_250_rng_sha256"
        ],
        "step_251_500": reports["500"],
        "reload_501_510": reports["510"],
        "reference_root": contract["source_reference_root"],
        "accelerated_root": spec["output_root"],
        "per_step_251_500_parity_proven": False,
        "production_identity_reframe_implemented": False,
        "promotion_allowed": False,
        "blocked_reason": (
            "PER_STEP_TRANSCRIPT_AND_PRODUCTION_IDENTITY_REFRAME_NOT_IMPLEMENTED"
        ),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(spec["output_root"]) / "endpoint_250_500_510_comparison.json"
    if path.exists() or path.is_symlink():
        previous = json.loads(path.read_text(encoding="utf-8"))
        previous.pop("written_at", None)
        observed = dict(receipt)
        observed.pop("written_at", None)
        if previous != observed:
            raise RuntimeError("T12 parity receipt already exists with different science")
    else:
        atomic_json(path, receipt)
    print(
        json.dumps(
            {
                "status": "ENDPOINT_ONLY_PASS_PROMOTION_BLOCKED",
                "receipt": str(path),
                "sha256": file_sha256(path),
                "promotion_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _owner(args: argparse.Namespace) -> int:
    spec = load_spec(args.task_spec)
    if spec["task_kind"] != "T12_ACCELERATED_FROM_CHECKPOINT250":
        raise ValueError("wrong T12 accelerated task kind")
    contract = spec["science_contract"]
    if spec["gpu_request"].get("index") != 1:
        raise ValueError("T12 accelerated branch must use authorized GPU1")
    if (
        spec.get("required_environment", {}).get(
            "ALLOW_T12_ACCELERATED_FROM_CHECKPOINT250_NOW"
        )
        != "1"
    ):
        raise ValueError("T12 accelerated GPU1 dispatch is not authorized")
    validate_reference_step250(task_spec_path=Path(contract["source_reference_task_spec"]))
    root = Path(spec["output_root"])
    if root.exists() or root.is_symlink():
        raise FileExistsError("T12 accelerated root must be fresh")
    identity = process_identity(os.getpid())
    if not identity or not identity["alive"]:
        raise RuntimeError("T12 accelerated owner identity is unavailable")
    pid = int(identity["pid"])
    ticks = int(identity["start_ticks"])
    heartbeat_path = Path(spec["expected_heartbeat_path"])
    pid_path = Path(spec["expected_pid_file"])
    lease_path = Path(spec["gpu_request"]["lease_path"])
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease = lease_path.open("a+b")
    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic_json(pid_path, {"task_id": spec["task_id"], "owner_pid": pid, "owner_start_ticks": ticks})
    state: dict[str, Any] = {"phase": "FORKING_STEP250", "step": 250, "child_pid": None, "stop": False}

    def heartbeat() -> None:
        sequence = 0
        while not state["stop"]:
            sequence += 1
            atomic_json(
                heartbeat_path,
                {
                    "schema_version": "tastemolnet_t12_accelerated_owner_heartbeat_v1",
                    "task_id": spec["task_id"],
                    "owner_pid": pid,
                    "owner_start_ticks": ticks,
                    "science_pid": state["child_pid"],
                    "phase": state["phase"],
                    "completed_step": state["step"],
                    "output_root": str(root),
                    "reference_root": contract["source_reference_root"],
                    "reference_signaled": False,
                    "dispatch_authorization": (
                        "ALLOW_T12_ACCELERATED_FROM_CHECKPOINT250_NOW"
                    ),
                    "gpu": spec["gpu_request"],
                    "gpu_lock_held": True,
                    "sequence": sequence,
                    "written_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            time.sleep(30)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    environment = dict(os.environ)
    environment.update(spec["required_environment"])
    environment["CUDA_VISIBLE_DEVICES"] = str(spec["gpu_request"]["index"])
    script = Path(__file__).resolve()
    exit_code = 1
    try:
        import torch

        _configure_profile()
        source_run_identity = Path(contract["source_reference_root"]) / "run_identity.json"
        fork_step250_prefix(
            source_root=Path(contract["source_reference_root"]),
            target_root=root,
            source_checkpoint_manifest=Path(contract["source_checkpoint_250"]),
            expected_identity=_expected_identity(source_run_identity, 250),
            torch=torch,
        )
        previous = Path(contract["accelerated_checkpoint_250"])
        log_root = heartbeat_path.parent / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        for cursor in ACCELERATED_CHECKPOINT_CURSORS[1:]:
            state["phase"] = f"ACCELERATED_RESUME_TO_{cursor}"
            command = [
                str(Path(spec["python"])),
                "-I",
                "-B",
                str(script),
                "--config",
                spec["config_path"],
                "segment",
                "--task-spec",
                str(args.task_spec),
                "--checkpoint-manifest",
                str(previous),
            ]
            with (log_root / f"segment-{cursor:08d}.log").open("xb") as stream:
                child = subprocess.Popen(
                    command,
                    cwd=spec["repo_root"],
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                state["child_pid"] = child.pid
                returncode = child.wait()
            state["child_pid"] = None
            if returncode != 0:
                state["phase"] = f"FAILED_AT_{cursor}"
                return returncode
            previous = Path(contract[f"accelerated_checkpoint_{cursor}"])
            if not previous.is_file():
                raise RuntimeError(f"T12 accelerated checkpoint {cursor} is absent")
            state["step"] = cursor
        state["phase"] = "ACCELERATED_510_ENDPOINT_READY_PROMOTION_BLOCKED"
        atomic_json(
            root / "accelerated_510_ready.json",
            {
                "schema_version": "tastemolnet_t12_accelerated_510_ready_v1",
                "status": "ENDPOINT_READY_PROMOTION_BLOCKED",
                "checkpoint_250": contract["accelerated_checkpoint_250"],
                "checkpoint_500": contract["accelerated_checkpoint_500"],
                "checkpoint_510": contract["accelerated_checkpoint_510"],
                "reference_signaled": False,
                "promotion_allowed": False,
                "per_step_251_500_parity_proven": False,
                "production_identity_reframe_implemented": False,
                "dispatch_authorization": (
                    "ALLOW_T12_ACCELERATED_FROM_CHECKPOINT250_NOW"
                ),
            },
        )
        exit_code = 0
        return 0
    finally:
        state["stop"] = True
        thread.join(timeout=2)
        lease.close()
        atomic_json(
            Path(spec["expected_terminal_path"]),
            {
                "schema_version": "tastemolnet_t12_accelerated_owner_terminal_v1",
                "task_id": spec["task_id"],
                "status": state["phase"],
                "exit_code": exit_code,
                "completed_step": state["step"],
                "owner_pid": pid,
                "owner_start_ticks": ticks,
                "output_root": str(root),
                "gpu_lock_held": False,
                "reference_signaled": False,
                "dispatch_authorization": (
                    "ALLOW_T12_ACCELERATED_FROM_CHECKPOINT250_NOW"
                ),
                "written_at": datetime.now(timezone.utc).isoformat(),
            },
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=_absolute, required=True)
    parser.add_argument("--set", action="append", default=[])
    subparsers = parser.add_subparsers(dest="action", required=True)
    owner = subparsers.add_parser("owner")
    owner.add_argument("--task-spec", type=_absolute, required=True)
    segment = subparsers.add_parser("segment")
    segment.add_argument("--task-spec", type=_absolute, required=True)
    segment.add_argument("--checkpoint-manifest", type=_absolute, required=True)
    parity = subparsers.add_parser("parity")
    parity.add_argument("--task-spec", type=_absolute, required=True)
    args = parser.parse_args(argv)
    if args.action == "segment":
        return _segment(args)
    if args.action == "parity":
        return _parity(args)
    return _owner(args)


if __name__ == "__main__":
    raise SystemExit(main())
