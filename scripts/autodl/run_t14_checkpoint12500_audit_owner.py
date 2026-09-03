#!/usr/bin/env python3
"""Own the fail-closed T14 step-12,500 convergence/archive audit.

When live cgroup headroom exceeds the historical measured requirement, the
child sequentially loads only sealed train-side checkpoints for convergence.
Otherwise it falls back to a metadata/archive audit that never imports torch.
Neither path opens SQLite or claims that a GPU science canary ran.
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


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _owner_identity() -> tuple[int, int]:
    identity = process_identity(os.getpid())
    if identity is None or not identity["alive"] or identity["start_ticks"] <= 0:
        raise RuntimeError("T14 audit owner process identity is unavailable")
    return int(identity["pid"]), int(identity["start_ticks"])


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_counter(path: Path) -> int:
    raw = path.read_text(encoding="utf-8").strip()
    if raw == "max":
        return (1 << 63) - 1
    value = int(raw)
    if value < 0:
        raise ValueError(f"negative cgroup counter: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=_absolute, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument(
        "--task-spec",
        type=_absolute,
        default=os.environ.get("MAIN_READY_TASK_SPEC"),
        required=os.environ.get("MAIN_READY_TASK_SPEC") is None,
    )
    args = parser.parse_args(argv)
    spec = load_spec(args.task_spec)
    if spec["task_kind"] != "T14_CONVERGENCE_AUDIT_OR_LOW_MEMORY_RESUME":
        raise ValueError("wrong immutable T14 task kind")
    contract = spec.get("science_contract")
    if not isinstance(contract, dict):
        raise ValueError("T14 science contract is absent")
    resume_spec = _absolute(str(contract.get("resume_spec", "")))
    if not resume_spec.is_file() or resume_spec.is_symlink():
        raise ValueError("bound T14 resume spec is absent or indirect")

    output = Path(spec["output_root"])
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"fresh T14 audit root already exists: {output}")
    pid_path = Path(spec["expected_pid_file"])
    heartbeat_path = Path(spec["expected_heartbeat_path"])
    owner_pid, owner_ticks = _owner_identity()
    atomic_json(
        pid_path,
        {
            "task_id": spec["task_id"],
            "owner_pid": owner_pid,
            "owner_start_ticks": owner_ticks,
        },
    )
    state: dict[str, Any] = {
        "phase": "STARTING_SEALED_CHECKPOINT_AUDIT",
        "child_pid": None,
        "stop": False,
    }

    def heartbeat() -> None:
        sequence = 0
        while not state["stop"]:
            sequence += 1
            atomic_json(
                heartbeat_path,
                {
                    "schema_version": "t14_main_ready_owner_heartbeat_v1",
                    "task_id": spec["task_id"],
                    "owner_pid": owner_pid,
                    "owner_start_ticks": owner_ticks,
                    "science_pid": None,
                    "audit_child_pid": state["child_pid"],
                    "phase": state["phase"],
                    "output_root": str(output),
                    "requested_gpu": spec["gpu_request"],
                    "gpu_lock_held": False,
                    "science_started": False,
                    "torch_load_invoked": False,
                    "sqlite_payload_opened": False,
                    "sequence": sequence,
                    "written_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            time.sleep(10)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    resume_payload = _read_json(resume_spec)
    memory = resume_payload.get("memory")
    if not isinstance(memory, dict):
        raise ValueError("bound T14 resume spec has no memory contract")
    limit = _read_counter(_absolute(str(memory["cgroup_limit_path"])))
    current = _read_counter(_absolute(str(memory["cgroup_current_path"])))
    required = int(memory["historical_required_headroom_bytes"])
    headroom = limit - current
    full_state_lock = _absolute(str(resume_payload["full_state_lock_path"]))
    full_state_lock.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = full_state_lock.open("a+b")
    log_root = heartbeat_path.parent / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    returncode = 3
    child_terminal: dict[str, Any] = {}
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            state["phase"] = "WAITING_FULL_STATE_CONSUMER_SERIALIZATION"
            return 75
        if headroom >= required:
            state["phase"] = "RUNNING_SERIAL_TRAIN_SIDE_CONVERGENCE_AUDIT"
            audit_script = (
                PROJECT_ROOT
                / "scripts/autodl/run_t14_external_convergence_auditor_v1.py"
            )
            command = [
                str(Path(spec["python"])),
                "-I",
                "-B",
                str(audit_script),
                "--config",
                spec["config_path"],
                "--set",
                "inference.fallback_to_heuristic=false",
                "--checkpoint-root",
                str(resume_payload["checkpoint_root"]),
                "--output-root",
                str(output),
                "--execution-commit",
                str(spec["execution_commit"]),
            ]
        else:
            state["phase"] = "AUDITING_ARCHIVE_WITHOUT_DESERIALIZATION"
            audit_script = (
                PROJECT_ROOT
                / "src/baselines/tastemolnet_t14_checkpoint12500_audit_owner.py"
            )
            command = [
                str(Path(spec["python"])),
                "-I",
                "-B",
                str(audit_script),
                "--resume-spec",
                str(resume_spec),
                "--owner-root",
                str(output),
            ]
        environment = dict(os.environ)
        environment.update(spec["required_environment"])
        with (log_root / "t14-checkpoint-audit.log").open("xb") as stream:
            child = subprocess.Popen(
                command,
                cwd=spec["repo_root"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            state["child_pid"] = child.pid
            returncode = child.wait()
        state["child_pid"] = None
        convergence_path = output / "t14_external_convergence_audit.json"
        terminal_path = output / "terminal.json"
        if convergence_path.is_file():
            child_terminal = _read_json(convergence_path)
            if returncode == 0 and child_terminal.get("converged") is True:
                state["phase"] = "T14_12500_CONVERGENCE_PASS"
            elif returncode == 0 and child_terminal.get("status") == "CONTINUE_T14":
                state["phase"] = "CONTINUE_T14_LOW_MEMORY_CANARY_REQUIRED"
            else:
                state["phase"] = f"FAILED_T14_CONVERGENCE_AUDIT_EXIT_{returncode}"
        elif terminal_path.is_file():
            child_terminal = _read_json(terminal_path)
            reason = str(child_terminal.get("reason_code", "AUDIT_TERMINAL_MISSING"))
            if returncode == 75 and reason in {
                "BLOCKED_LOW_MEMORY_CANARY_UNAVAILABLE",
                "BLOCKED_T14_CGROUP_HEADROOM",
            }:
                state["phase"] = reason
            elif returncode == 0 and child_terminal.get("status") == "PASS":
                state["phase"] = "SEALED_CHECKPOINT_AUDIT_PASS"
            else:
                state["phase"] = f"FAILED_T14_AUDIT_EXIT_{returncode}"
        else:
            state["phase"] = f"FAILED_T14_AUDIT_EXIT_{returncode}"
        return returncode
    finally:
        state["stop"] = True
        thread.join(timeout=2)
        atomic_json(
            heartbeat_path.parent / "terminal.json",
            {
                "schema_version": "t14_main_ready_owner_terminal_v1",
                "task_id": spec["task_id"],
                "status": state["phase"],
                "owner_pid": owner_pid,
                "owner_start_ticks": owner_ticks,
                "output_root": str(output),
                "science_started": False,
                "gpu_lock_held": False,
                "child_returncode": returncode,
                "child_terminal": child_terminal,
                "cgroup_limit_bytes": limit,
                "cgroup_current_bytes_at_admission": current,
                "cgroup_headroom_bytes_at_admission": headroom,
                "historical_required_headroom_bytes": required,
                "written_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
