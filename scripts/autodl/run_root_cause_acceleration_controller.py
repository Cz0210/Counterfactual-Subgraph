#!/usr/bin/env python3
"""Persistent read-only supervisor for root-cause acceleration routes."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any, Mapping

from src.utils.autodl_progress_health import ProgressPolicy, update_progress_health
from src.utils.autodl_runtime import atomic_write_json, read_json_object, utc_now


SCHEMA = "autodl_root_cause_acceleration_monitor_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _proc_identity(pid: int) -> dict[str, Any] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        end = raw.rfind(")")
        fields = raw[end + 2 :].split()
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", "replace"
        )
        io_values: dict[str, int] = {}
        for line in Path(f"/proc/{pid}/io").read_text(encoding="utf-8").splitlines():
            name, value = line.split(":", 1)
            io_values[name] = int(value.strip())
        return {
            "pid": pid,
            "state": fields[0],
            "ppid": int(fields[1]),
            "utime_ticks": int(fields[11]),
            "stime_ticks": int(fields[12]),
            "start_ticks": int(fields[19]),
            "rss_bytes": int(fields[21]) * int(os.sysconf("SC_PAGE_SIZE")),
            "read_bytes": io_values.get("read_bytes", 0),
            "write_bytes": io_values.get("write_bytes", 0),
            "command": command,
        }
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError):
        return None


def _read_pointer(path: Path, pointer: list[str]) -> Any:
    value: Any = read_json_object(path)
    for name in pointer:
        if not isinstance(value, Mapping) or name not in value:
            raise KeyError(f"Missing JSON progress pointer component: {name}")
        value = value[name]
    return value


def _progress(task: Mapping[str, Any]) -> int:
    probe = task.get("progress")
    if not isinstance(probe, Mapping):
        raise ValueError("Each task requires a progress probe.")
    path = Path(str(probe.get("path") or "")).expanduser().resolve(strict=True)
    kind = str(probe.get("kind"))
    if kind == "json":
        pointer = probe.get("pointer")
        if not isinstance(pointer, list) or not pointer:
            raise ValueError("JSON progress probes require a nonempty pointer.")
        return int(_read_pointer(path, [str(value) for value in pointer]))
    if kind == "regex_tail":
        pattern = re.compile(str(probe.get("pattern") or ""))
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - int(probe.get("tail_bytes", 2_000_000))))
            text = handle.read().decode("utf-8", "replace").replace("\r", "\n")
        matches = list(pattern.finditer(text))
        if not matches:
            raise ValueError(f"Progress regex has no match: {path}")
        return int(matches[-1].group(int(probe.get("group", 1))))
    raise ValueError(f"Unsupported progress probe kind: {kind}")


def _gpu_snapshot() -> dict[int, dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
    rows: dict[int, dict[str, Any]] = {}
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 4:
            raise RuntimeError("Unexpected nvidia-smi GPU row.")
        rows[int(values[0])] = {
            "index": int(values[0]),
            "uuid": values[1],
            "memory_used_mib": int(values[2]),
            "utilization_percent": int(values[3]),
        }
    return rows


def _validate_spec(path: Path) -> dict[str, Any]:
    spec = read_json_object(path)
    if spec.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported root-cause supervisor spec.")
    if not str(spec.get("controller_id") or "").strip():
        raise ValueError("controller_id is required.")
    tasks = spec.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("At least one monitored task is required.")
    ids = [str(task.get("task_id") or "") for task in tasks]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Monitored task IDs must be unique and nonempty.")
    for task in tasks:
        if task.get("ownership") != "external_read_only":
            raise ValueError("The root-cause supervisor cannot own scientific tasks.")
        if int(task.get("pid", 0)) <= 0 or int(task.get("start_ticks", 0)) <= 0:
            raise ValueError("Every monitored PID requires frozen start_ticks.")
        if int(task.get("total", 0)) <= 0:
            raise ValueError("Every monitored task requires a positive total.")
    return spec


def run_once(*, spec: Mapping[str, Any], root: Path, spec_sha256: str) -> dict[str, Any]:
    prior_path = root / "state.json"
    prior = read_json_object(prior_path) if prior_path.is_file() else {"tasks": {}}
    prior_tasks = prior.get("tasks") if isinstance(prior.get("tasks"), Mapping) else {}
    gpu = _gpu_snapshot()
    observed_at = utc_now()
    output_tasks: dict[str, Any] = {}
    for task in spec["tasks"]:
        task_id = str(task["task_id"])
        identity = _proc_identity(int(task["pid"]))
        alive = bool(
            identity
            and int(identity["start_ticks"]) == int(task["start_ticks"])
            and str(task.get("command_contains") or "") in str(identity["command"])
        )
        try:
            completed = _progress(task)
            health = update_progress_health(
                prior_tasks.get(task_id),
                completed=completed,
                total=int(task["total"]),
                pid_alive=alive,
                observed_at=observed_at,
                policy=ProgressPolicy(
                    stalled_after_seconds=float(task.get("stalled_after_seconds", 1800)),
                    slow_eta_hours=float(task.get("slow_eta_hours", 24)),
                    unviable_eta_hours=float(task.get("unviable_eta_hours", 168)),
                ),
            )
            error = None
        except Exception as exc:
            health = dict(prior_tasks.get(task_id) or {})
            health.update(
                {
                    "health_state": "OBSERVATION_FAILED",
                    "observed_at": observed_at,
                    "pid_alive": alive,
                    "automatic_signal_allowed": False,
                }
            )
            error = f"{type(exc).__name__}:{exc}"
        gpu_index = task.get("gpu_index")
        output_tasks[task_id] = {
            **health,
            "task_id": task_id,
            "pid": int(task["pid"]),
            "start_ticks": int(task["start_ticks"]),
            "process": identity,
            "gpu": None if gpu_index is None else gpu.get(int(gpu_index)),
            "output_root": str(task.get("output_root") or ""),
            "ownership": "external_read_only",
            "observation_error": error,
        }
    payload = {
        "schema_version": SCHEMA,
        "controller_id": spec["controller_id"],
        "controller_pid": os.getpid(),
        "spec_sha256": spec_sha256,
        "updated_at": observed_at,
        "tasks": output_tasks,
        "gpu_inventory": gpu,
        "automatic_signal_allowed": False,
        "scientific_task_ownership": "none",
    }
    atomic_write_json(prior_path, payload)
    atomic_write_json(
        root / "heartbeat.json",
        {
            "schema_version": SCHEMA,
            "controller_id": spec["controller_id"],
            "pid": os.getpid(),
            "updated_at": observed_at,
            "spec_sha256": spec_sha256,
            "health_counts": {
                state: sum(
                    row.get("health_state") == state for row in output_tasks.values()
                )
                for state in sorted({row.get("health_state") for row in output_tasks.values()})
            },
        },
    )
    _append_jsonl(root / "status_updates.jsonl", payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec_path = args.spec.expanduser().resolve(strict=True)
    spec = _validate_spec(spec_path)
    spec_sha = _sha256(spec_path)
    root = args.control_root.expanduser().resolve(strict=False) / str(spec["controller_id"])
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "controller.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Root-cause controller is already running.") from exc
    atomic_write_json(
        root / "ownership.json",
        {
            "schema_version": SCHEMA,
            "controller_id": spec["controller_id"],
            "controller_pid": os.getpid(),
            "scientific_task_ownership": "none",
            "external_tasks_are_read_only": True,
            "automatic_signal_allowed": False,
            "spec_path": str(spec_path),
            "spec_sha256": spec_sha,
        },
    )
    atomic_write_json(root / "queue.json", {"controller_ids": spec.get("component_controller_ids", []), "scientific_tasks_scheduled_here": []})
    if not (root / "runs.jsonl").exists():
        _append_jsonl(
            root / "runs.jsonl",
            {
                "record_type": "monitor_started",
                "controller_id": spec["controller_id"],
                "controller_pid": os.getpid(),
                "spec_path": str(spec_path),
                "spec_sha256": spec_sha,
                "started_at": utc_now(),
            },
        )
    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    while not stop:
        run_once(spec=spec, root=root, spec_sha256=spec_sha)
        if args.once:
            break
        for _ in range(max(1, args.poll_seconds)):
            if stop:
                break
            time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
