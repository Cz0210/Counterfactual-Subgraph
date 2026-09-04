#!/usr/bin/env python3
"""Run the read-only final-four closeout observer and durable heartbeat."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.final_four_cells_observer import (  # noqa: E402
    FinalFourObserverError,
    atomic_json,
    snapshot,
    stable_sha256,
    utc_now,
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path


def _start_ticks() -> int:
    fields = Path("/proc/self/stat").read_text(encoding="ascii").split()
    return int(fields[21])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--state-root", type=_absolute, required=True)
    parser.add_argument("--matrix-authority", type=_absolute, required=True)
    parser.add_argument("--task-spec", type=_absolute, action="append", default=[])
    parser.add_argument("--hpc-t8-pointer", type=_absolute)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.poll_seconds < 10.0:
        raise ValueError("poll interval must be at least 10 seconds")
    state_root = args.state_root.resolve(strict=False)
    state_root.mkdir(parents=True, exist_ok=True)
    lock_path = state_root / ".observer.lock"
    stopping = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FinalFourObserverError("another final-four observer owns the lock") from exc
        pid = os.getpid()
        start_ticks = _start_ticks()
        command = [sys.executable, *sys.argv]
        identity = {
            "controller_id": f"final_four_cells_v1_{pid}_{start_ticks}",
            "controller_pid": pid,
            "controller_start_ticks": start_ticks,
            "controller_command_sha256": stable_sha256(command),
            "repo_root": str(PROJECT_ROOT),
            "cwd": os.getcwd(),
        }
        atomic_json(state_root / "owner.json", {**identity, "started_at": utc_now()})
        sequence = 0
        while True:
            sequence += 1
            observed = snapshot(
                matrix_authority=args.matrix_authority,
                task_specs=args.task_spec,
                hpc_t8_pointer=args.hpc_t8_pointer,
            )
            heartbeat = {
                **observed,
                **identity,
                "sequence": sequence,
                "written_at": utc_now(),
                "observer_only": True,
            }
            heartbeat["heartbeat_sha256"] = stable_sha256(heartbeat)
            atomic_json(state_root / "heartbeat.json", heartbeat)
            if args.once or stopping or observed["state"] == "PASS":
                break
            time.sleep(args.poll_seconds)
        terminal = {
            **identity,
            "state": "PASS" if observed["state"] == "PASS" else "STOPPED",
            "matrix_complete_cells": observed["matrix_complete_cells"],
            "stopped_at": utc_now(),
            "signal_received": stopping,
        }
        terminal["terminal_sha256"] = stable_sha256(terminal)
        atomic_json(state_root / "terminal.json", terminal)
    print(json.dumps(heartbeat, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
