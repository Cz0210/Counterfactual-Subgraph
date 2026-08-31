#!/usr/bin/env python3
"""Monitor exactly the eight remaining 16-of-16 science stages."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time
from typing import Any, Mapping


SCHEMA = "fast_16of16_v2_spec_v1"
STATE_SCHEMA = "fast_16of16_v2_state_v1"
TASKS = (
    "aids_postprocess",
    "mut_exact",
    "bace_comrecgc",
    "bace_globalgce",
    "taste_t6",
    "taste_t7",
    "taste_t8",
    "taste_t14",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _absolute(value: Any, *, field: str) -> Path:
    if type(value) is not str or not value or "\0" in value:
        raise ValueError(f"{field} must be one absolute path")
    path = Path(value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise ValueError(f"{field} must be normalized and absolute")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(data)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise OSError("short controller write")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _heartbeat_json(path: Path, value: Mapping[str, Any]) -> None:
    """Replaceable FUSE heartbeat; science artifacts never use this path."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()


def load_spec(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    if set(value) != {
        "schema_version",
        "controller_id",
        "state_root",
        "execution_commit",
        "execution_tree",
        "poll_seconds",
        "run_gnn_ablation",
        "tasks",
    } or value.get("schema_version") != SCHEMA:
        raise ValueError("fast-v2 spec shape changed")
    if value.get("run_gnn_ablation") is not False:
        raise ValueError("GNN ablation must remain disabled")
    if type(value.get("controller_id")) is not str or not value["controller_id"]:
        raise ValueError("controller_id is required")
    _absolute(value.get("state_root"), field="state_root")
    for field in ("execution_commit", "execution_tree"):
        token = value.get(field)
        if (
            type(token) is not str
            or len(token) != 40
            or any(character not in "0123456789abcdef" for character in token)
        ):
            raise ValueError(f"{field} must be one Git object ID")
    if value.get("poll_seconds") != 60:
        raise ValueError("fast-v2 poll interval must equal 60")
    tasks = value.get("tasks")
    if type(tasks) is not dict or tuple(tasks) != TASKS:
        raise ValueError("fast-v2 fixed task order changed")
    for name, row in tasks.items():
        if type(row) is not dict or set(row) != {
            "root",
            "pid",
            "start_ticks",
            "command_token",
            "progress_files",
            "terminal_files",
        }:
            raise ValueError(f"{name} binding shape changed")
        _absolute(row["root"], field=f"{name}.root")
        pid = row["pid"]
        ticks = row["start_ticks"]
        token = row["command_token"]
        if pid is None:
            if ticks is not None or token is not None:
                raise ValueError(f"{name} queued PID fields are inconsistent")
        elif (
            type(pid) is not int
            or pid <= 1
            or type(ticks) is not int
            or ticks <= 0
            or type(token) is not str
            or not token
        ):
            raise ValueError(f"{name} live PID binding is invalid")
        for field in ("progress_files", "terminal_files"):
            paths = row[field]
            if type(paths) is not list or len(paths) > 8:
                raise ValueError(f"{name}.{field} changed")
            for index, item in enumerate(paths):
                _absolute(item, field=f"{name}.{field}[{index}]")
    return value


def _process(row: Mapping[str, Any]) -> dict[str, Any]:
    if row["pid"] is None:
        return {"state": "QUEUED", "pid": None, "identity_match": None}
    pid = int(row["pid"])
    proc = Path("/proc") / str(pid)
    if not proc.is_dir():
        return {"state": "EXITED", "pid": pid, "identity_match": False}
    try:
        raw_stat = (proc / "stat").read_text(encoding="utf-8")
        _head, separator, tail = raw_stat.rpartition(")")
        fields = tail.strip().split()
        start_ticks = int(fields[19])
        raw_cmd = (proc / "cmdline").read_bytes()
        command = raw_cmd.replace(b"\0", b" ").decode("utf-8")
    except (OSError, UnicodeDecodeError, ValueError, IndexError):
        return {"state": "UNREADABLE", "pid": pid, "identity_match": False}
    identity = (
        bool(separator)
        and start_ticks == row["start_ticks"]
        and row["command_token"] in command
    )
    return {
        "state": "RUNNING" if identity else "PID_IDENTITY_MISMATCH",
        "pid": pid,
        "identity_match": identity,
        "start_ticks": start_ticks,
        "cmdline_sha256": hashlib.sha256(raw_cmd).hexdigest(),
    }


def _artifact(path: Path, *, parse_json: bool) -> dict[str, Any]:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return {"path": str(path), "exists": False}
    regular = stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode)
    result: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "regular": regular,
        "bytes": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
    }
    if regular and parse_json and info.st_size <= 2 * 1024 * 1024:
        try:
            result["payload"] = _load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            result["json_error"] = f"{type(exc).__name__}: {exc}"
    return result


def observe(spec: Mapping[str, Any], sequence: int) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for name in TASKS:
        row = spec["tasks"][name]
        terminal = [
            _artifact(_absolute(path, field="terminal"), parse_json=False)
            for path in row["terminal_files"]
        ]
        tasks[name] = {
            "root": row["root"],
            "root_exists": Path(row["root"]).is_dir(),
            "process": _process(row),
            "progress": [
                _artifact(_absolute(path, field="progress"), parse_json=True)
                for path in row["progress_files"]
            ],
            "terminal": terminal,
            "terminal_complete": bool(terminal)
            and all(item.get("regular") is True for item in terminal),
        }
    return {
        "schema_version": STATE_SCHEMA,
        "controller_id": spec["controller_id"],
        "controller_pid": os.getpid(),
        "sequence": sequence,
        "written_at": _utc_now(),
        "execution_commit": spec["execution_commit"],
        "execution_tree": spec["execution_tree"],
        "run_gnn_ablation": False,
        "scope": "FIXED_EIGHT_STAGE_OBSERVER",
        "science_launch_allowed": False,
        "process_termination_allowed": False,
        "tasks": tasks,
    }


def run(spec_path: Path, *, once: bool) -> int:
    spec = load_spec(spec_path)
    state_root = _absolute(spec["state_root"], field="state_root")
    state_root.mkdir(parents=True, exist_ok=True)
    lock = (state_root / "controller.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("another fast-v2 controller owns this root") from exc
    spec_sha = _sha256(spec_path)
    receipt = state_root / "controller_receipt.json"
    if receipt.exists():
        if _load_json(receipt).get("spec_sha256") != spec_sha:
            raise RuntimeError("fast-v2 controller spec changed")
    else:
        _atomic_json(
            receipt,
            {
                "schema_version": "fast_16of16_v2_controller_receipt_v1",
                "status": "RUNNING",
                "controller_id": spec["controller_id"],
                "controller_pid": os.getpid(),
                "spec_path": str(spec_path),
                "spec_sha256": spec_sha,
                "created_at": _utc_now(),
                "scope": "FIXED_EIGHT_STAGE_OBSERVER",
                "run_gnn_ablation": False,
            },
        )
    stopped = False

    def _stop(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, _stop)
    heartbeat = state_root / "heartbeat.json"
    sequence = 0
    if heartbeat.is_file():
        sequence = int(_load_json(heartbeat).get("sequence", 0))
    try:
        while True:
            sequence += 1
            _heartbeat_json(heartbeat, observe(spec, sequence))
            if once or stopped:
                return 0
            time.sleep(60)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    expected = Path(__file__).resolve().parents[2] / "configs/hpc.yaml"
    if args.config.resolve(strict=True) != expected:
        raise SystemExit("--config must be this checkout's configs/hpc.yaml")
    try:
        return run(args.spec.resolve(strict=True), once=args.once)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[FAST_16OF16_V2_BLOCKED] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
