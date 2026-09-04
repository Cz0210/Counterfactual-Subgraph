#!/usr/bin/env python3
"""Own one explicitly authorized Mut trace-off Route-B generation."""

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

from src.utils.autodl_mut_first_divergence_v1 import atomic_json  # noqa: E402
from src.utils.autodl_mut_route_b_v1 import (  # noqa: E402
    M_MAX,
    route_b_generation_command,
    validate_route_b_spec,
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
        raise RuntimeError("owner /proc stat is malformed")
    return int(raw[closing + 2 :].split()[19])


def _acquire_lease(path: Path):
    if path.is_symlink():
        raise RuntimeError("Route-B lease cannot be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        stream.close()
        raise
    return stream


def _resume_required(spec: dict[str, Any]) -> bool:
    output = Path(spec["output_root"])
    if not output.exists():
        return False
    if output.is_symlink() or not output.is_dir():
        raise RuntimeError("Route-B output is indirect or not a directory")
    if (output / "_RUN_COMPLETE.json").exists():
        raise FileExistsError("Route-B generation is already complete")
    checkpoint = Path(spec["checkpoint_root"]) / "LATEST"
    mirror = Path(spec["checkpoint_mirror_root"]) / "LATEST"
    if not checkpoint.is_file() or not mirror.is_file():
        raise RuntimeError("existing Route-B output has no mirrored committed checkpoint")
    return True


def _validate_completed_output(spec: dict[str, Any]) -> dict[str, Any]:
    output = Path(spec["output_root"])
    complete = _json(output / "_RUN_COMPLETE.json")
    manifest = _json(output / "run_manifest.json")
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError("Route-B run manifest has no generation parameters")
    if (
        complete.get("run_complete") is not True
        or manifest.get("run_complete") is not True
        or complete.get("counterfactuals_sha256")
        != manifest.get("counterfactuals_sha256")
        or manifest.get("dataset") != "mutagenicity"
        or manifest.get("trace_enabled") is not False
        or int(parameters.get("steps", -1)) != M_MAX
        or int(parameters.get("candidate_capacity", -1)) != 100_000
        or manifest.get("test_loaded") is not False
        or manifest.get("calibration_loaded") is not False
    ):
        raise RuntimeError("Route-B completed output violates the frozen contract")
    if (output / "trace").exists():
        raise RuntimeError("trace-off Route B unexpectedly produced a trace directory")
    return {"complete": complete, "run_manifest": manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--task-spec", type=_absolute, required=True)
    args = parser.parse_args(argv)
    spec = validate_route_b_spec(_json(args.task_spec), check_files=True)
    gpu = str(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if gpu != str(spec["gpu_index"]):
        raise RuntimeError(
            "Route-B owner requires its exact physical GPU in CUDA_VISIBLE_DEVICES"
        )
    gpu_lock = GPUFileLock(
        Path(spec["gpu_lock_root"]),
        gpu_index=0,
        gpu_uuid=str(spec["gpu_uuid"]),
        owner={"task_id": spec["task_id"], "role": "mut_traceoff_route_b"},
    ).acquire()
    try:
        observations = {row.index: row for row in query_gpu_inventory()}
        physical = observations.get(0)
        if physical is None or physical.uuid != spec["gpu_uuid"]:
            raise RuntimeError("Route-B physical GPU0 UUID changed")
        if physical.processes:
            raise RuntimeError("Route-B physical GPU0 already has compute processes")
        lease = _acquire_lease(Path(spec["lease_path"]))
    except BaseException:
        gpu_lock.release()
        raise
    runtime = Path(spec["owner_runtime_root"])
    try:
        runtime.mkdir(parents=True, exist_ok=False)
    except BaseException:
        lease.close()
        gpu_lock.release()
        raise
    owner_pid = os.getpid()
    owner_ticks = _start_ticks(owner_pid)
    resumed = _resume_required(spec)
    command = route_b_generation_command(spec)
    if resumed:
        command.append("--resume")
    state: dict[str, Any] = {
        "phase": "STARTING_ROUTE_B_RESUME" if resumed else "STARTING_ROUTE_B_FRESH",
        "science_pid": None,
        "stop": False,
    }
    atomic_json(
        runtime / "owner_pid.json",
        {
            "schema_version": "mut_route_b_owner_pid_v1",
            "task_id": spec["task_id"],
            "owner_pid": owner_pid,
            "owner_start_ticks": owner_ticks,
            "gpu_uuid": spec["gpu_uuid"],
            "gpu_lock_path": str(gpu_lock.path),
        },
    )

    def heartbeat() -> None:
        sequence = 0
        while not state["stop"]:
            sequence += 1
            atomic_json(
                runtime / "heartbeat.json",
                {
                    "schema_version": "mut_route_b_owner_heartbeat_v1",
                    "task_id": spec["task_id"],
                    "owner_pid": owner_pid,
                    "owner_start_ticks": owner_ticks,
                    "science_pid": state["science_pid"],
                    "phase": state["phase"],
                    "output_root": spec["output_root"],
                    "trace_enabled": False,
                    "M_MAX": 50_000,
                    "candidate_capacity": 100_000,
                    "pair_store_reuse_allowed": False,
                    "dbscan_reuse_allowed": False,
                    "resumed": resumed,
                    "gpu_uuid": spec["gpu_uuid"],
                    "global_gpu_uuid_lock_held": True,
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
        state["phase"] = "RUNNING_ROUTE_B_TRACE_OFF_50K"
        environment = dict(os.environ)
        environment.update(spec["required_environment"])
        with (runtime / "science.log").open("xb") as stream:
            child = subprocess.Popen(
                command,
                cwd=spec["repo_root"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            state["science_pid"] = child.pid
            returncode = child.wait()
        state["science_pid"] = None
        if returncode != 0:
            state["phase"] = f"FAILED_SCIENCE_EXIT_{returncode}"
            return returncode if returncode > 0 else 128 - returncode
        _validate_completed_output(spec)
        state["phase"] = "PASS_GENERATION_COMPLETE_AWAITING_NEW_PAIR_DBSCAN"
        return 0
    finally:
        state["stop"] = True
        thread.join(timeout=2)
        lease.close()
        gpu_lock.release()
        atomic_json(
            runtime / "terminal.json",
            {
                "schema_version": "mut_route_b_owner_terminal_v1",
                "task_id": spec["task_id"],
                "status": state["phase"],
                "owner_pid": owner_pid,
                "owner_start_ticks": owner_ticks,
                "science_pid": state["science_pid"],
                "science_returncode": returncode,
                "output_root": spec["output_root"],
                "fresh_50k_started": child is not None,
                "trace_enabled": False,
                "pair_store_reused": False,
                "dbscan_reused": False,
                "written_at": datetime.now(timezone.utc).isoformat(),
            },
        )


if __name__ == "__main__":
    raise SystemExit(main())
