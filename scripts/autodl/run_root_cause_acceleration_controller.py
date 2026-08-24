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
import stat
import subprocess
import time
from typing import Any, Mapping

from src.utils.autodl_progress_health import (
    OBSERVATION_FAILED,
    ProgressPolicy,
    SUPERSESSION_RECEIPT_SCHEMA,
    SUPERSESSION_RECEIPT_STATE,
    mark_superseded,
    route_viability_for_progress_state,
    update_progress_health,
)
from src.utils.autodl_runtime import atomic_write_json, read_json_object, utc_now


SCHEMA = "autodl_root_cause_acceleration_monitor_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TERMINAL_PROC_STATES = frozenset({"Z", "X", "x"})
SUPERSESSION_REPLACEMENT_FIELDS = (
    "controller_id",
    "task_id",
    "output_root",
    "controller_manifest_path",
    "expected_controller_manifest_sha256",
    "task_gate_path",
    "expected_task_gate_sha256",
    "task_gate_status_field",
    "final_manifest_path",
    "expected_final_manifest_sha256",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _absolute_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty absolute path.")
    expanded = Path(value).expanduser()
    if not expanded.is_absolute():
        raise ValueError(f"{label} must be absolute.")
    return Path(os.path.abspath(os.fspath(expanded)))


def _physical_path(path: Path, *, label: str, kind: str) -> Path:
    """Require an existing path whose complete path chain has no symlinks."""

    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        try:
            info = os.lstat(current)
        except OSError as exc:
            raise ValueError(f"{label} is not readable: {current}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{label} cannot contain symlinks: {current}")
    info = os.lstat(path)
    if kind == "file" and not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a physical regular file.")
    if kind == "dir" and not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} must be a physical directory.")
    return path


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a lowercase SHA256 digest string.")
    digest = value
    if not SHA256_RE.fullmatch(digest):
        raise ValueError(f"{label} must be a lowercase SHA256 digest.")
    return digest


def _positive_json_int(value: Any, *, label: str) -> int:
    """Accept a JSON integer only; bool and integral floats are forbidden."""

    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive JSON integer.")
    return value


def _read_physical_json(
    value: Any,
    *,
    label: str,
    expected_sha256: str,
) -> tuple[Path, dict[str, Any], str]:
    """Open one no-symlink JSON file and validate the exact bytes once."""

    path = _absolute_path(value, label=label)
    if path.suffix.lower() != ".json":
        raise ValueError(f"{label} must use a .json suffix.")
    _physical_path(path, label=label, kind="file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} could not be opened safely.") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{label} must be a physical regular file.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            content = handle.read()
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(content).hexdigest()
    if digest != _require_sha256(expected_sha256, label=f"expected {label} SHA256"):
        raise ValueError(f"{label} SHA256 mismatch.")
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must contain UTF-8 JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must contain an object.")
    return path, payload, digest


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


def _scientific_worker_alive(
    identity: Mapping[str, Any] | None, task: Mapping[str, Any]
) -> bool:
    """Match the frozen worker generation and exclude terminal proc states."""

    if not identity or str(identity.get("state") or "") in TERMINAL_PROC_STATES:
        return False
    identity_pid = identity.get("pid")
    identity_start = identity.get("start_ticks")
    task_pid = task.get("pid")
    task_start = task.get("start_ticks")
    if any(
        type(value) is not int or value <= 0
        for value in (identity_pid, identity_start, task_pid, task_start)
    ):
        return False
    return bool(
        identity_pid == task_pid
        and identity_start == task_start
        and str(task.get("command_contains") or "")
        in str(identity.get("command") or "")
    )


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
        _positive_json_int(task.get("pid"), label="monitored task pid")
        _positive_json_int(
            task.get("start_ticks"), label="monitored task start_ticks"
        )
        if int(task.get("total", 0)) <= 0:
            raise ValueError("Every monitored task requires a positive total.")
        supersession = task.get("supersession")
        if supersession is not None:
            _validate_supersession_spec(supersession)
    return spec


def _validate_supersession_spec(raw: Any) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("Task supersession must be a mapping when provided.")
    if not isinstance(raw.get("reason"), str) or not raw["reason"].strip():
        raise ValueError("Task supersession requires a nonempty reason.")
    _absolute_path(raw.get("receipt_path"), label="supersession receipt_path")
    _require_sha256(
        raw.get("expected_receipt_sha256"),
        label="supersession expected_receipt_sha256",
    )
    replacement = raw.get("replacement")
    if not isinstance(replacement, Mapping):
        raise ValueError("Task supersession requires a replacement mapping.")
    for field in SUPERSESSION_REPLACEMENT_FIELDS:
        if field not in replacement:
            raise ValueError(f"Task supersession replacement is missing {field}.")
    for field in ("controller_id", "task_id"):
        if not isinstance(replacement.get(field), str) or not replacement[field].strip():
            raise ValueError(f"Replacement {field} must be nonempty.")
    for field in (
        "output_root",
        "controller_manifest_path",
        "task_gate_path",
        "final_manifest_path",
    ):
        _absolute_path(replacement.get(field), label=f"replacement {field}")
    for field in (
        "expected_controller_manifest_sha256",
        "expected_task_gate_sha256",
        "expected_final_manifest_sha256",
    ):
        _require_sha256(replacement.get(field), label=f"replacement {field}")
    if replacement.get("task_gate_status_field") not in {"state", "status"}:
        raise ValueError("Replacement task_gate_status_field must be state or status.")
    return raw


def _receipt_field(
    payload: Mapping[str, Any], field: str, expected: Any, *, label: str
) -> None:
    if field not in payload:
        raise ValueError(f"{label} is missing {field}.")
    actual = payload[field]
    if isinstance(expected, bool):
        matches = actual is expected
    else:
        matches = type(actual) is type(expected) and actual == expected
    if not matches:
        raise ValueError(f"{label} {field} does not match the frozen spec.")


def _supersession_receipt(
    task: Mapping[str, Any], *, scientific_worker_alive: bool
) -> dict[str, Any] | None:
    """Validate a hash-pinned graceful handover receipt; never perform it.

    The receipt binds the frozen old PID generation/output root and a physical,
    hash-closed replacement PASS gate.  A still-live old worker is a hard
    failure even when every receipt byte is otherwise valid.
    """

    raw = task.get("supersession")
    if raw is None:
        return None
    raw = _validate_supersession_spec(raw)
    if scientific_worker_alive:
        raise ValueError("A live scientific worker cannot be marked SUPERSEDED.")
    reason = raw["reason"].strip()
    receipt_path, receipt, receipt_sha = _read_physical_json(
        raw.get("receipt_path"),
        label="supersession receipt",
        expected_sha256=str(raw.get("expected_receipt_sha256")),
    )
    _receipt_field(
        receipt,
        "schema_version",
        SUPERSESSION_RECEIPT_SCHEMA,
        label="supersession receipt",
    )
    _receipt_field(
        receipt,
        "state",
        SUPERSESSION_RECEIPT_STATE,
        label="supersession receipt",
    )
    _receipt_field(receipt, "reason", reason, label="supersession receipt")
    for field in (
        "graceful_checkpoint_completed",
        "graceful_stop_completed",
        "old_worker_exited",
    ):
        _receipt_field(receipt, field, True, label="supersession receipt")
    _receipt_field(receipt, "sigkill_used", False, label="supersession receipt")

    old_task = receipt.get("old_task")
    if not isinstance(old_task, Mapping):
        raise ValueError("Supersession receipt requires an old_task mapping.")
    expected_old = {
        "task_id": str(task["task_id"]),
        "pid": _positive_json_int(task.get("pid"), label="old task pid"),
        "start_ticks": _positive_json_int(
            task.get("start_ticks"), label="old task start_ticks"
        ),
        "output_root": str(
            _absolute_path(task.get("output_root"), label="old task output_root")
        ),
    }
    for field, expected in expected_old.items():
        _receipt_field(old_task, field, expected, label="receipt old_task")

    replacement_spec = raw["replacement"]
    assert isinstance(replacement_spec, Mapping)
    replacement_receipt = receipt.get("replacement")
    if not isinstance(replacement_receipt, Mapping):
        raise ValueError("Supersession receipt requires a replacement mapping.")
    normalized_replacement = {
        "controller_id": str(replacement_spec["controller_id"]),
        "task_id": str(replacement_spec["task_id"]),
        "output_root": str(
            _absolute_path(
                replacement_spec["output_root"], label="replacement output_root"
            )
        ),
        "controller_manifest_path": str(
            _absolute_path(
                replacement_spec["controller_manifest_path"],
                label="replacement controller_manifest_path",
            )
        ),
        "controller_manifest_sha256": str(
            replacement_spec["expected_controller_manifest_sha256"]
        ),
        "task_gate_path": str(
            _absolute_path(
                replacement_spec["task_gate_path"],
                label="replacement task_gate_path",
            )
        ),
        "task_gate_sha256": str(replacement_spec["expected_task_gate_sha256"]),
        "task_gate_status_field": str(replacement_spec["task_gate_status_field"]),
        "task_gate_state": "PASS",
        "final_manifest_path": str(
            _absolute_path(
                replacement_spec["final_manifest_path"],
                label="replacement final_manifest_path",
            )
        ),
        "final_manifest_sha256": str(
            replacement_spec["expected_final_manifest_sha256"]
        ),
    }
    for field, expected in normalized_replacement.items():
        _receipt_field(
            replacement_receipt,
            field,
            expected,
            label="receipt replacement",
        )

    replacement_root = _physical_path(
        Path(normalized_replacement["output_root"]),
        label="replacement output_root",
        kind="dir",
    )
    _, controller_manifest, controller_manifest_sha = _read_physical_json(
        normalized_replacement["controller_manifest_path"],
        label="replacement controller manifest",
        expected_sha256=normalized_replacement["controller_manifest_sha256"],
    )
    if controller_manifest.get("controller_id") != normalized_replacement["controller_id"]:
        raise ValueError("Replacement controller manifest controller_id mismatch.")
    _, task_gate, task_gate_sha = _read_physical_json(
        normalized_replacement["task_gate_path"],
        label="replacement task gate",
        expected_sha256=normalized_replacement["task_gate_sha256"],
    )
    if task_gate.get("task_id") != normalized_replacement["task_id"]:
        raise ValueError("Replacement task gate task_id mismatch.")
    gate_field = normalized_replacement["task_gate_status_field"]
    if task_gate.get(gate_field) != "PASS":
        raise ValueError("Replacement task gate is not PASS.")
    final_manifest_path, _, final_manifest_sha = _read_physical_json(
        normalized_replacement["final_manifest_path"],
        label="replacement final manifest",
        expected_sha256=normalized_replacement["final_manifest_sha256"],
    )
    try:
        final_manifest_path.relative_to(replacement_root)
    except ValueError as exc:
        raise ValueError("Replacement final manifest must be inside output_root.") from exc

    return {
        "schema_version": SUPERSESSION_RECEIPT_SCHEMA,
        "state": SUPERSESSION_RECEIPT_STATE,
        "reason": reason,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "graceful_checkpoint_completed": True,
        "graceful_stop_completed": True,
        "old_worker_exited": True,
        "sigkill_used": False,
        "old_task": expected_old,
        "replacement": {
            **normalized_replacement,
            "controller_manifest_sha256": controller_manifest_sha,
            "task_gate_sha256": task_gate_sha,
            "final_manifest_sha256": final_manifest_sha,
        },
    }


def _observation_failure(
    previous: Mapping[str, Any] | None,
    *,
    observed_at: str,
    scientific_worker_alive: bool,
) -> dict[str, Any]:
    """Return a schema-complete failed observation without asserting PASS."""

    health = dict(previous or {})
    health.update(
        {
            "health_state": OBSERVATION_FAILED,
            "scientific_progress_state": OBSERVATION_FAILED,
            "route_viability": route_viability_for_progress_state(
                OBSERVATION_FAILED
            ),
            "observed_at": observed_at,
            # ``pid_alive`` is an old monitor-v1 alias retained for clients.
            "pid_alive": scientific_worker_alive,
            "scientific_worker_alive": scientific_worker_alive,
            "automatic_signal_allowed": False,
        }
    )
    return health


def run_once(*, spec: Mapping[str, Any], root: Path, spec_sha256: str) -> dict[str, Any]:
    prior_path = root / "state.json"
    prior = read_json_object(prior_path) if prior_path.is_file() else {"tasks": {}}
    prior_tasks = prior.get("tasks") if isinstance(prior.get("tasks"), Mapping) else {}
    gpu = _gpu_snapshot()
    observed_at = utc_now()
    controller_process = _proc_identity(os.getpid())
    controller_process_alive = controller_process is not None
    output_tasks: dict[str, Any] = {}
    for task in spec["tasks"]:
        task_id = str(task["task_id"])
        identity = _proc_identity(int(task["pid"]))
        alive = _scientific_worker_alive(identity, task)
        try:
            supersession = _supersession_receipt(
                task, scientific_worker_alive=alive
            )
            if supersession is not None:
                # This skips a potentially removed progress file after the
                # independently approved graceful handover.  It never sends
                # a signal to the frozen worker PID.
                health = mark_superseded(
                    prior_tasks.get(task_id),
                    total=int(task["total"]),
                    scientific_worker_alive=alive,
                    observed_at=observed_at,
                    supersession=supersession,
                )
            else:
                completed = _progress(task)
                health = update_progress_health(
                    prior_tasks.get(task_id),
                    completed=completed,
                    total=int(task["total"]),
                    pid_alive=alive,
                    observed_at=observed_at,
                    policy=ProgressPolicy(
                        stalled_after_seconds=float(
                            task.get("stalled_after_seconds", 1800)
                        ),
                        slow_eta_hours=float(task.get("slow_eta_hours", 24)),
                        unviable_eta_hours=float(
                            task.get("unviable_eta_hours", 168)
                        ),
                    ),
                )
            error = None
        except Exception as exc:
            health = _observation_failure(
                prior_tasks.get(task_id),
                observed_at=observed_at,
                scientific_worker_alive=alive,
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
            # Explicitly disambiguate monitor liveness from observed science.
            "controller_process_alive": controller_process_alive,
            "scientific_worker_alive": alive,
            "observation_error": error,
        }
    payload = {
        "schema_version": SCHEMA,
        "controller_id": spec["controller_id"],
        "controller_pid": os.getpid(),
        "controller_process_alive": controller_process_alive,
        "controller_process": controller_process,
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
            "controller_process_alive": controller_process_alive,
            "updated_at": observed_at,
            "spec_sha256": spec_sha256,
            "health_counts": {
                state: sum(
                    row.get("health_state") == state for row in output_tasks.values()
                )
                for state in sorted({row.get("health_state") for row in output_tasks.values()})
            },
            "route_viability_counts": {
                state: sum(
                    row.get("route_viability") == state
                    for row in output_tasks.values()
                )
                for state in sorted(
                    {row.get("route_viability") for row in output_tasks.values()}
                )
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
