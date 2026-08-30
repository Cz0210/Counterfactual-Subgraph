#!/usr/bin/env python3
"""Persist a narrow heartbeat for the authorized 4x4 deadline continuation.

This sidecar does not replace any science controller.  It only records the
exact processes and terminal artifacts named by its immutable spec so the
already-running AIDS/Mut, BACE, and Taste continuations remain restartable.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any


SCHEMA = "deadline_main_completion_spec_v1"
STATE_SCHEMA = "deadline_main_completion_state_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _absolute(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError(f"{label} must be one absolute path")
    path = Path(value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise ValueError(f"{label} must be normalized and absolute")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_spec(path: Path) -> dict[str, Any]:
    spec = _load_json(path)
    expected = {
        "schema_version",
        "controller_id",
        "state_root",
        "execution_commit",
        "execution_tree",
        "poll_seconds",
        "run_gnn_ablation",
        "observed_processes",
        "observed_artifacts",
    }
    if set(spec) != expected or spec.get("schema_version") != SCHEMA:
        raise ValueError("deadline sidecar spec shape changed")
    if spec.get("run_gnn_ablation") is not False:
        raise ValueError("GNN ablation must remain disabled")
    if type(spec.get("controller_id")) is not str or not spec["controller_id"]:
        raise ValueError("controller_id is required")
    _absolute(spec.get("state_root"), label="state_root")
    for key in ("execution_commit", "execution_tree"):
        value = spec.get(key)
        if (
            type(value) is not str
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{key} must be one Git object ID")
    if type(spec.get("poll_seconds")) is not int or spec["poll_seconds"] != 60:
        raise ValueError("deadline sidecar poll_seconds must equal 60")
    processes = spec.get("observed_processes")
    artifacts = spec.get("observed_artifacts")
    if type(processes) is not dict or type(artifacts) is not dict:
        raise ValueError("observed process/artifact maps are required")
    for name, row in processes.items():
        if type(name) is not str or type(row) is not dict or set(row) != {
            "pid",
            "start_ticks",
            "command_token",
        }:
            raise ValueError("observed process binding changed")
        if type(row["pid"]) is not int or row["pid"] <= 0:
            raise ValueError("observed PID must be positive")
        if type(row["start_ticks"]) is not int or row["start_ticks"] <= 0:
            raise ValueError("observed start_ticks must be positive")
        if type(row["command_token"]) is not str or not row["command_token"]:
            raise ValueError("observed command token is required")
    for name, value in artifacts.items():
        if type(name) is not str:
            raise ValueError("observed artifact name changed")
        _absolute(value, label=f"artifact {name}")
    return spec


def _process(spec: dict[str, Any]) -> dict[str, Any]:
    pid = int(spec["pid"])
    proc = Path("/proc") / str(pid)
    if not proc.is_dir():
        return {"pid": pid, "alive": False, "identity_match": False}
    try:
        stat_fields = (proc / "stat").read_text(encoding="utf-8").split()
        start_ticks = int(stat_fields[21])
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (OSError, UnicodeDecodeError, ValueError, IndexError):
        return {"pid": pid, "alive": False, "identity_match": False}
    identity = (
        start_ticks == spec["start_ticks"]
        and spec["command_token"] in command
    )
    return {
        "pid": pid,
        "alive": True,
        "identity_match": identity,
        "start_ticks": start_ticks,
        "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
    }


def observe(spec: dict[str, Any], *, sequence: int) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for name, raw_path in spec["observed_artifacts"].items():
        path = _absolute(raw_path, label=f"artifact {name}")
        row: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            row.update(size=path.stat().st_size, sha256=_sha256(path))
        artifacts[name] = row
    return {
        "schema_version": STATE_SCHEMA,
        "controller_id": spec["controller_id"],
        "controller_pid": os.getpid(),
        "sequence": sequence,
        "written_at": _utc_now(),
        "execution_commit": spec["execution_commit"],
        "execution_tree": spec["execution_tree"],
        "run_gnn_ablation": False,
        "processes": {
            name: _process(binding)
            for name, binding in spec["observed_processes"].items()
        },
        "artifacts": artifacts,
    }


def run(spec_path: Path, *, once: bool) -> int:
    spec = load_spec(spec_path)
    state_root = _absolute(spec["state_root"], label="state_root")
    state_root.mkdir(parents=True, exist_ok=True)
    spec_sha256 = _sha256(spec_path)
    receipt_path = state_root / "controller_receipt.json"
    if receipt_path.is_file():
        receipt = _load_json(receipt_path)
        if receipt.get("spec_sha256") != spec_sha256:
            raise RuntimeError("deadline controller receipt is bound to another spec")
    else:
        _atomic_json(
            receipt_path,
            {
                "schema_version": "deadline_main_completion_controller_receipt_v1",
                "controller_id": spec["controller_id"],
                "created_at": _utc_now(),
                "execution_commit": spec["execution_commit"],
                "execution_tree": spec["execution_tree"],
                "spec_path": str(spec_path),
                "spec_sha256": spec_sha256,
                "scope": "READ_ONLY_DEADLINE_CONTINUATION_HEARTBEAT",
                "process_termination_allowed": False,
                "matrix_publication_allowed": False,
                "run_gnn_ablation": False,
            },
        )
    lock = (state_root / "controller.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("another deadline sidecar owns this state root") from exc
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    sequence = 0
    heartbeat_path = state_root / "heartbeat.json"
    if heartbeat_path.is_file():
        sequence = int(_load_json(heartbeat_path).get("sequence", 0))
    try:
        while True:
            sequence += 1
            state = observe(spec, sequence=sequence)
            _atomic_json(state_root / "state.json", state)
            _atomic_json(heartbeat_path, state)
            if once or stop:
                return 0
            time.sleep(spec["poll_seconds"])
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    action = parser.add_subparsers(dest="action", required=True)
    for name in ("run", "once"):
        command = action.add_parser(name)
        command.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    expected = Path(__file__).resolve().parents[2] / "configs/hpc.yaml"
    if args.config.resolve(strict=True) != expected:
        raise SystemExit("--config must be this checkout's configs/hpc.yaml")
    try:
        return run(args.spec.resolve(strict=True), once=args.action == "once")
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[DEADLINE_MAIN_COMPLETION_BLOCKED] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
