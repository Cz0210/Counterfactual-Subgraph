#!/usr/bin/env python3
"""Run the bounded, sequential same-contract Mut trace-on/off A/B."""

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

from src.utils.autodl_mut_first_divergence_v1 import atomic_json, file_sha256  # noqa: E402
from src.utils.autodl_mut_post_ab_continuation_v1 import (  # noqa: E402
    classify_same_contract_gate,
)
from src.utils.autodl_mut_same_contract_ab_v1 import (  # noqa: E402
    same_contract_ab_command,
    validate_same_contract_ab_spec,
)
from src.utils.autodl_runtime import GPUFileLock, query_gpu_inventory  # noqa: E402


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected one JSON object: {path}")
    return value


def _start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    closing = raw.rfind(")")
    if closing < 0:
        raise RuntimeError("owner process stat is malformed")
    return int(raw[closing + 2 :].split()[19])


def _lease(path: Path):
    if path.is_symlink():
        raise RuntimeError("Mut A/B lease cannot be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        stream.close()
        raise
    return stream


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--task-spec", type=_absolute, required=True)
    args = parser.parse_args(argv)
    spec = validate_same_contract_ab_spec(_json(args.task_spec), check_files=True)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(spec["gpu_index"]):
        raise RuntimeError("Mut A/B owner requires the exact assigned physical GPU")
    for field in ("run_root", "output_dir", "control_root"):
        target = Path(spec[field])
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Mut A/B fresh target already exists: {target}")
    gpu_lock = GPUFileLock(
        Path(spec["gpu_lock_root"]),
        gpu_index=0,
        gpu_uuid=str(spec["gpu_uuid"]),
        owner={"task_id": spec["task_id"], "role": "mut_same_contract_ab"},
    ).acquire()
    try:
        observations = {row.index: row for row in query_gpu_inventory()}
        physical = observations.get(0)
        if physical is None or physical.uuid != spec["gpu_uuid"]:
            raise RuntimeError("Mut A/B physical GPU0 UUID changed")
        if physical.processes:
            raise RuntimeError("Mut A/B physical GPU0 already has compute processes")
        lease = _lease(Path(spec["lease_path"]))
    except BaseException:
        gpu_lock.release()
        raise
    control = Path(spec["control_root"])
    try:
        control.mkdir(parents=True, exist_ok=False)
    except BaseException:
        lease.close()
        gpu_lock.release()
        raise
    owner_pid = os.getpid()
    owner_ticks = _start_ticks(owner_pid)
    state: dict[str, Any] = {
        "phase": "STARTING_FRESH_SAME_CONTRACT_A_B",
        "science_pid": None,
        "stop": False,
    }
    atomic_json(
        control / "owner_pid.json",
        {
            "schema_version": "mut_same_contract_ab_owner_pid_v1",
            "task_id": spec["task_id"],
            "owner_pid": owner_pid,
            "owner_start_ticks": owner_ticks,
            "task_spec": str(args.task_spec),
            "task_spec_file_sha256": file_sha256(args.task_spec),
            "gpu_uuid": spec["gpu_uuid"],
            "gpu_lock_path": str(gpu_lock.path),
        },
    )

    def heartbeat() -> None:
        sequence = 0
        while not state["stop"]:
            sequence += 1
            active = Path(spec["run_root"]) / "active_arm.json"
            active_state: Any = None
            if active.is_file() and not active.is_symlink():
                try:
                    active_state = _json(active)
                except (OSError, ValueError, json.JSONDecodeError):
                    active_state = "UNREADABLE_TRANSIENT"
            atomic_json(
                control / "heartbeat.json",
                {
                    "schema_version": "mut_same_contract_ab_owner_heartbeat_v1",
                    "task_id": spec["task_id"],
                    "owner_pid": owner_pid,
                    "owner_start_ticks": owner_ticks,
                    "science_pid": state["science_pid"],
                    "phase": state["phase"],
                    "active_arm": active_state,
                    "gpu_index": spec["gpu_index"],
                    "gpu_uuid": spec["gpu_uuid"],
                    "global_gpu_uuid_lock_held": True,
                    "arms_sequential": True,
                    "comparison_steps": 500,
                    "post_reload_steps": 10,
                    "fresh_50k_started": False,
                    "sequence": sequence,
                    "written_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            time.sleep(30)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    child: subprocess.Popen[bytes] | None = None
    returncode = 1
    try:
        command = same_contract_ab_command(spec)
        environment = dict(os.environ)
        environment.update(spec["required_environment"])
        state["phase"] = "RUNNING_FRESH_SAME_CONTRACT_A_B"
        with (control / "science.log").open("xb") as stream:
            child = subprocess.Popen(
                command,
                cwd=spec["controller_project_root"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            state["science_pid"] = child.pid
            returncode = child.wait()
        state["science_pid"] = None
        gate = Path(spec["output_dir"]) / "trace_on_off_500_step_equivalence.json"
        if gate.is_file():
            state["phase"] = classify_same_contract_gate(_json(gate))
            if state["phase"] == "PASS_TRACE_MODE_EQUIVALENCE" and returncode == 0:
                return 0
            if state["phase"] == "SCIENTIFIC_STATE_DIVERGENCE_CONFIRMED":
                return 4
            return returncode if returncode > 0 else 5
        state["phase"] = f"FAILED_A_B_EXIT_{returncode}"
        return returncode if returncode > 0 else 128 - returncode
    finally:
        state["stop"] = True
        thread.join(timeout=2)
        lease.close()
        gpu_lock.release()
        gate = Path(spec["output_dir"]) / "trace_on_off_500_step_equivalence.json"
        atomic_json(
            control / "terminal.json",
            {
                "schema_version": "mut_same_contract_ab_owner_terminal_v1",
                "task_id": spec["task_id"],
                "status": state["phase"],
                "owner_pid": owner_pid,
                "owner_start_ticks": owner_ticks,
                "science_pid": state["science_pid"],
                "science_returncode": returncode,
                "equivalence_gate": str(gate) if gate.is_file() else None,
                "equivalence_gate_sha256": file_sha256(gate) if gate.is_file() else None,
                "fresh_50k_started": False,
                "written_at": datetime.now(timezone.utc).isoformat(),
            },
        )


if __name__ == "__main__":
    raise SystemExit(main())
