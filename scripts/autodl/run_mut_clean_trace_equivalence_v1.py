#!/usr/bin/env python3
"""Own one fresh, fail-closed Mut trace adoption continuation.

This is only an ownership bridge for the already reviewed Mut adoption worker.
It deliberately omits the old completed A arm, causing both checkpoint parity
and trace-on/off parity to be generated from fresh roots.  The reviewed worker
then performs the historical 50k binding, WNode closeout, and matrix queueing.
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
        raise RuntimeError("Mut owner process identity is unavailable")
    return int(identity["pid"]), int(identity["start_ticks"])


def _acquire_gpu_lease(spec: dict[str, Any]):
    gpu = spec.get("gpu_request")
    if not isinstance(gpu, dict) or "lease_path" not in gpu:
        raise ValueError("Mut task spec has no bound dispatch-owner lease")
    if gpu.get("lease_scope") != "MAIN_READY_DISPATCH_OWNER":
        raise ValueError(
            "Mut wrapper lease must use the independent dispatch-owner namespace"
        )
    lease_path = _absolute(str(gpu["lease_path"]))
    if "main-ready-dispatch-leases" not in lease_path.parts:
        raise ValueError(
            "Mut wrapper lease path is not in main-ready-dispatch-leases"
        )
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    if lease_path.is_symlink():
        raise ValueError("Mut GPU lease cannot be a symlink")
    lease = lease_path.open("a+b")
    try:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lease.close()
        raise
    return lease


def _heartbeat_payload(
    spec: dict[str, Any],
    *,
    owner_pid: int,
    owner_ticks: int,
    output: Path,
    state: dict[str, Any],
    sequence: int,
) -> dict[str, Any]:
    return {
        "schema_version": "mut_main_ready_owner_heartbeat_v1",
        "task_id": spec["task_id"],
        "owner_pid": owner_pid,
        "owner_start_ticks": owner_ticks,
        "science_pid": state["child_pid"],
        "phase": state["phase"],
        "output_root": str(output),
        "gpu_request": spec["gpu_request"],
        "dispatch_lease_held": True,
        "worker_gpu_lock_managed_internally": True,
        "science_gpu_selected_by_reviewed_worker": True,
        "fresh_trace_on_arm": True,
        "fresh_trace_off_arm": True,
        "old_mixed_commit_result_adopted": False,
        "pair_store_recomputed": False,
        "dbscan_recomputed": False,
        "sequence": sequence,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
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
    if spec["task_kind"] != "MUT_TRACE_EQUIVALENCE_AND_ADOPTION":
        raise ValueError("wrong immutable Mut task kind")
    contract = spec.get("science_contract")
    if not isinstance(contract, dict):
        raise ValueError("Mut science contract is absent")
    required = {
        "legacy_adoption_spec",
        "authorization_receipt",
        "protected_manifest",
        "historical_project_root",
        "instrumentation_project_root",
        "semantic_finalizer_project_root",
        "terminal_controller_evidence",
        "controller_pid",
        "controller_start_ticks",
    }
    missing = required - set(contract)
    if missing:
        raise ValueError(f"Mut science contract is incomplete: {sorted(missing)}")

    output = Path(spec["output_root"])
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"fresh Mut output already exists: {output}")
    owner_pid, owner_ticks = _owner_identity()
    lease = _acquire_gpu_lease(spec)
    pid_path = Path(spec["expected_pid_file"])
    heartbeat_path = Path(spec["expected_heartbeat_path"])
    try:
        atomic_json(
            pid_path,
            {
                "task_id": spec["task_id"],
                "owner_pid": owner_pid,
                "owner_start_ticks": owner_ticks,
            },
        )
    except BaseException:
        lease.close()
        raise
    state: dict[str, Any] = {
        "phase": "STARTING_FRESH_EQUIVALENCE",
        "child_pid": None,
        "stop": False,
    }

    def heartbeat() -> None:
        sequence = 0
        while not state["stop"]:
            sequence += 1
            atomic_json(
                heartbeat_path,
                _heartbeat_payload(
                    spec,
                    owner_pid=owner_pid,
                    owner_ticks=owner_ticks,
                    output=output,
                    state=state,
                    sequence=sequence,
                ),
            )
            time.sleep(30)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    worker = PROJECT_ROOT / "scripts/autodl/run_mut_trace_on_adoption_worker.py"
    command = [
        str(Path(spec["python"])), "-I", "-B", str(worker),
        "--config", spec["config_path"], "run",
        "--spec", str(contract["legacy_adoption_spec"]),
        "--authorization-receipt", str(contract["authorization_receipt"]),
        "--protected-manifest", str(contract["protected_manifest"]),
        "--historical-project-root", str(contract["historical_project_root"]),
        "--instrumentation-project-root", str(contract["instrumentation_project_root"]),
        "--semantic-finalizer-project-root", str(contract["semantic_finalizer_project_root"]),
        "--output-root", str(output),
        "--controller-pid", str(int(contract["controller_pid"])),
        "--controller-start-ticks", str(int(contract["controller_start_ticks"])),
        "--terminal-controller-evidence", str(contract["terminal_controller_evidence"]),
        "--successor-guard-script", "run_mut_checkpoint_instrumentation_equivalence.py",
        "--successor-guard-action", "run-pair",
        "--throttle-profile", "legacy-v1",
    ]
    environment = dict(os.environ)
    environment.update(spec["required_environment"])
    log_root = heartbeat_path.parent / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    try:
        state["phase"] = "RUNNING_REVIEWED_MUT_ADOPTION_WORKER"
        with (log_root / "mut-adoption-worker.log").open("xb") as stream:
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
        if returncode != 0:
            state["phase"] = f"FAILED_REVIEWED_WORKER_EXIT_{returncode}"
            return returncode
        terminal = _json(output / "worker_terminal.json")
        if terminal.get("status") != "PASS_MATRIX_PUBLISHER_SUBMITTED":
            state["phase"] = "FAILED_REVIEWED_WORKER_TERMINAL"
            return 3
        state["phase"] = "PASS_MATRIX_PUBLISHER_SUBMITTED"
        return 0
    finally:
        state["stop"] = True
        thread.join(timeout=2)
        lease.close()
        atomic_json(
            heartbeat_path.parent / "terminal.json",
            {
                "schema_version": "mut_main_ready_owner_terminal_v1",
                "task_id": spec["task_id"],
                "status": state["phase"],
                "owner_pid": owner_pid,
                "owner_start_ticks": owner_ticks,
                "output_root": str(output),
                "dispatch_lease_held": False,
                "worker_gpu_lock_managed_internally": True,
                "science_gpu_selected_by_reviewed_worker": True,
                "written_at": datetime.now(timezone.utc).isoformat(),
            },
        )


if __name__ == "__main__":
    raise SystemExit(main())
