#!/usr/bin/env python3
"""Run/own the real 3,778-parent T12 reference 500 + reload tail.

The profile uses the production planner, official sample size and candidate
capacity.  It changes only the diagnostic stop schedule to 250, 500 and the
501--510 reload proof.  Each segment is a fresh process.
"""

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

from src.utils.main_ready_task_specs import (  # noqa: E402
    atomic_json,
    load_spec,
    process_identity,
)


TOTAL_STEPS = 510
CHECKPOINT_CURSORS = (250, 500, 510)


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

    state.PINNED_TOTAL_STEPS = TOTAL_STEPS
    state.PINNED_CHECKPOINT_CURSORS = CHECKPOINT_CURSORS
    state.HISTORY_MAX_SEGMENTS = len(CHECKPOINT_CURSORS)
    transitions.TRANSITION_MAX_SEGMENTS = len(CHECKPOINT_CURSORS)
    resume.PRODUCTION_TOTAL_STEPS = TOTAL_STEPS
    resume.PRODUCTION_CHECKPOINT_CURSORS = frozenset(CHECKPOINT_CURSORS)
    full.PRODUCTION_TOTAL_STEPS = TOTAL_STEPS
    full.PRODUCTION_CHECKPOINT_CURSORS = frozenset(CHECKPOINT_CURSORS)


def _segment(args: argparse.Namespace) -> int:
    spec = load_spec(args.task_spec)
    contract = spec["science_contract"]
    _configure_profile()
    from src.baselines.tastemolnet_gcf_full import run_t12_generation_segment

    result = run_t12_generation_segment(
        mode=args.mode,
        output_root=spec["output_root"],
        checkpoint_manifest=args.checkpoint_manifest,
        attempt_id=spec["attempt_uuid"],
        generation_token=contract["generation_token"],
        gpu_uuid=spec["gpu_request"]["uuid"],
        managed_neurosed_root=contract["managed_neurosed_root"],
        t3_root=contract["t3_root"],
        official_root=contract["official_root"],
        threshold_authority_path=contract["threshold_authority"],
        replay_gate_path=contract["replay_gate"],
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def _owner_identity() -> tuple[int, int]:
    identity = process_identity(os.getpid())
    if identity is None or not identity["alive"] or identity["start_ticks"] <= 0:
        raise RuntimeError("T12 owner process identity is unavailable")
    return int(identity["pid"]), int(identity["start_ticks"])


def _owner_terminal(
    *, spec: dict[str, Any], state: dict[str, Any], owner_pid: int, owner_ticks: int
) -> dict[str, Any]:
    return {
        "schema_version": "t12_reference_500_owner_terminal_v1",
        "task_id": spec["task_id"],
        "status": state["phase"],
        "owner_pid": owner_pid,
        "owner_start_ticks": owner_ticks,
        "output_root": spec["output_root"],
        "completed_step": state["step"],
        "gpu_lock_held": False,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }


def _owner(args: argparse.Namespace) -> int:
    spec = load_spec(args.task_spec)
    if spec["task_kind"] != "T12_REFERENCE_ACCELERATED_PARITY_AND_FULL":
        raise ValueError("wrong immutable task kind")
    root = Path(spec["output_root"])
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"fresh T12 reference root already exists: {root}")
    pid_path = Path(spec["expected_pid_file"])
    heartbeat_path = Path(spec["expected_heartbeat_path"])
    gpu = spec.get("gpu_request")
    if not isinstance(gpu, dict) or "lease_path" not in gpu:
        raise ValueError("T12 task spec has no bound GPU lease")
    lease_path = _absolute(str(gpu["lease_path"]))
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    if lease_path.is_symlink():
        raise ValueError("T12 GPU lease cannot be a symlink")
    pid, ticks = _owner_identity()
    lease = lease_path.open("a+b")
    try:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lease.close()
        raise
    try:
        atomic_json(pid_path, {"task_id": spec["task_id"], "owner_pid": pid, "owner_start_ticks": ticks})
    except BaseException:
        lease.close()
        raise
    state: dict[str, Any] = {"phase": "STARTING_REFERENCE", "child_pid": None, "step": 0, "stop": False}

    def heartbeat() -> None:
        sequence = 0
        while not state["stop"]:
            sequence += 1
            atomic_json(heartbeat_path, {
                "schema_version": "t12_reference_500_owner_heartbeat_v1",
                "task_id": spec["task_id"],
                "owner_pid": pid,
                "owner_start_ticks": ticks,
                "science_pid": state["child_pid"],
                "phase": state["phase"],
                "completed_step": state["step"],
                "target_step": 500,
                "reload_tail": "501-510",
                "parent_count": 3778,
                "sample_size": 10000,
                "candidate_capacity": 100000,
                "gpu": spec["gpu_request"],
                "gpu_lock_held": True,
                "output_root": str(root),
                "calibration_loaded": False,
                "test_loaded": False,
                "written_at": datetime.now(timezone.utc).isoformat(),
                "sequence": sequence,
            })
            time.sleep(30)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    environment = dict(os.environ)
    environment.update(spec["required_environment"])
    environment["CUDA_VISIBLE_DEVICES"] = str(spec["gpu_request"]["index"])
    script = Path(__file__).resolve()
    log_parent = heartbeat_path.parent / "logs"
    log_parent.mkdir(parents=True, exist_ok=True)
    previous: Path | None = None
    try:
        for cursor in CHECKPOINT_CURSORS:
            mode = "fresh" if previous is None else "resume"
            state["phase"] = f"REFERENCE_{mode.upper()}_TO_{cursor}"
            command = [
                str(Path(spec["python"])), "-I", "-B", str(script),
                "--config", spec["config_path"], "segment",
                "--task-spec", str(args.task_spec), "--mode", mode,
            ]
            if previous is not None:
                command.extend(["--checkpoint-manifest", str(previous)])
            with (log_parent / f"segment-{cursor:08d}.log").open("xb") as stream:
                child = subprocess.Popen(command, cwd=spec["repo_root"], env=environment, stdout=stream, stderr=subprocess.STDOUT)
                state["child_pid"] = child.pid
                returncode = child.wait()
            state["child_pid"] = None
            if returncode != 0:
                state["phase"] = f"FAILED_AT_{cursor}"
                return returncode
            previous = root / "checkpoints" / f"checkpoint-{cursor:08d}.manifest.json"
            if not previous.is_file():
                raise RuntimeError(f"durable T12 checkpoint {cursor} is absent")
            state["step"] = cursor
        state["phase"] = "REFERENCE_500_AND_RELOAD_510_PASS"
        atomic_json(root / "reference_500_receipt.json", {
            "schema_version": "t12_reference_500_receipt_v1",
            "status": "PASS",
            "reference_steps": 500,
            "reload_steps": [501, 510],
            "checkpoint_250": str(root / "checkpoints/checkpoint-00000250.manifest.json"),
            "checkpoint_500": str(root / "checkpoints/checkpoint-00000500.manifest.json"),
            "checkpoint_510": str(root / "checkpoints/checkpoint-00000510.manifest.json"),
            "parent_count": 3778,
            "sample_size": 10000,
            "calibration_loaded": False,
            "test_loaded": False,
        })
        return 0
    finally:
        state["stop"] = True
        thread.join(timeout=2)
        lease.close()
        atomic_json(
            heartbeat_path.parent / "terminal.json",
            _owner_terminal(
                spec=spec,
                state=state,
                owner_pid=pid,
                owner_ticks=ticks,
            ),
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
    segment.add_argument("--mode", choices=("fresh", "resume"), required=True)
    segment.add_argument("--checkpoint-manifest", type=_absolute)
    args = parser.parse_args(argv)
    if args.action == "segment":
        if (args.mode == "resume") != (args.checkpoint_manifest is not None):
            raise ValueError("T12 segment resume/checkpoint arguments disagree")
        return _segment(args)
    return _owner(args)


if __name__ == "__main__":
    raise SystemExit(main())
