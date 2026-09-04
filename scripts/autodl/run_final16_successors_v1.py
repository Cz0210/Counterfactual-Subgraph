#!/usr/bin/env python3
"""Persist a read-only heartbeat over the canonical final16 owner registry.

Science launch and publication remain owned by their existing sealed one-shot
binders. This sidecar never executes a task command, takes a GPU lock, signals
a process, or writes the matrix authority.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Sequence
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.final16_successors_v1 import (  # noqa: E402
    HEARTBEAT_SCHEMA,
    OWNER_SCHEMA,
    TERMINAL_SCHEMA,
    build_snapshot,
)
from src.utils.final_four_cells_observer import (  # noqa: E402
    atomic_json,
    stable_sha256,
    utc_now,
)


class Final16ControllerError(RuntimeError):
    """The read-only controller cannot establish exclusive ownership."""


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise argparse.ArgumentTypeError("absolute non-symlink path required")
    return path.resolve(strict=False)


def _self_start_ticks() -> int:
    raw = Path("/proc/self/stat").read_text(encoding="ascii")
    closing = raw.rfind(")")
    if closing < 0:
        raise Final16ControllerError("cannot parse /proc/self/stat")
    return int(raw[closing + 2 :].split()[19])


def _ensure_state_root(path: Path) -> Path:
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise Final16ControllerError("state root must be a physical directory")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise Final16ControllerError("state root may not be a symlink")
    return path.resolve(strict=True)


def _controller_identity(controller_id: str | None) -> dict[str, Any]:
    pid = os.getpid()
    ticks = _self_start_ticks()
    identity = controller_id or f"final16-successors-{uuid4()}"
    if not identity or "/" in identity:
        raise Final16ControllerError("controller id is empty or unsafe")
    return {
        "controller_id": identity,
        "controller_pid": pid,
        "controller_start_ticks": ticks,
        "controller_command_sha256": stable_sha256([sys.executable, *sys.argv]),
        "repo_root": str(PROJECT_ROOT),
        "cwd": os.getcwd(),
    }


def run_controller(
    *,
    state_root: Path,
    matrix_authority_root: Path,
    owner_registry: Path,
    poll_seconds: float,
    controller_id: str | None = None,
    once: bool = False,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    state_root = _ensure_state_root(state_root)
    if os.environ.get("RUN_LLM_ABLATION", "0") not in {"", "0"}:
        raise Final16ControllerError("LLM ablation dispatch belongs to its strict gate")
    if os.environ.get("RUN_GNN_ABLATION", "0") not in {"", "0"}:
        raise Final16ControllerError("GNN ablation dispatch belongs to its strict gate")
    stopping = False

    def _stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    previous = {
        signum: signal.signal(signum, _stop)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    lock_path = state_root / "controller.lock"
    identity = _controller_identity(controller_id)
    last_heartbeat: dict[str, Any] = {}
    try:
        with lock_path.open("a+b") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise Final16ControllerError(
                    "another final16 successors controller owns this state root"
                ) from exc
            owner = {
                "schema_version": OWNER_SCHEMA,
                **identity,
                "owner_registry": str(owner_registry),
                "matrix_authority_root": str(matrix_authority_root),
                "controller_launches_science": False,
                "controller_launches_publishers": False,
                "started_at": utc_now(),
            }
            owner["owner_sha256"] = stable_sha256(owner)
            atomic_json(state_root / "owner.json", owner)
            sequence = 0
            while True:
                sequence += 1
                try:
                    observed = build_snapshot(
                        matrix_authority_root=matrix_authority_root,
                        owner_registry_path=owner_registry,
                        proc_root=proc_root,
                    )
                    heartbeat: dict[str, Any] = {
                        "schema_version": HEARTBEAT_SCHEMA,
                        **identity,
                        "sequence": sequence,
                        "state": observed["state"],
                        "snapshot": observed,
                        "evidence_error": None,
                        "written_at": utc_now(),
                    }
                except Exception as exc:
                    heartbeat = {
                        "schema_version": HEARTBEAT_SCHEMA,
                        **identity,
                        "sequence": sequence,
                        "state": "BLOCKED_EVIDENCE",
                        "snapshot": None,
                        "evidence_error": f"{type(exc).__name__}: {exc}",
                        "controller_launches_science": False,
                        "controller_launches_publishers": False,
                        "matrix_write_performed": False,
                        "signal_sent": False,
                        "written_at": utc_now(),
                    }
                heartbeat["heartbeat_sha256"] = stable_sha256(heartbeat)
                atomic_json(state_root / "heartbeat.json", heartbeat)
                last_heartbeat = heartbeat
                if once or stopping:
                    break
                time.sleep(poll_seconds)
            terminal = {
                "schema_version": TERMINAL_SCHEMA,
                **identity,
                "state": "STOPPED_GRACEFULLY" if stopping else "SINGLE_SNAPSHOT_COMPLETE",
                "last_sequence": sequence,
                "science_restart_performed": False,
                "matrix_write_performed": False,
                "signal_sent_to_science": False,
                "stopped_at": utc_now(),
            }
            terminal["terminal_sha256"] = stable_sha256(terminal)
            atomic_json(state_root / "terminal.json", terminal)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return last_heartbeat


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--state-root", type=_absolute, required=True)
    parser.add_argument("--matrix-authority-root", type=_absolute, required=True)
    parser.add_argument("--owner-registry", type=_absolute, required=True)
    parser.add_argument("--controller-id")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.config not in (None, "configs/hpc.yaml", str(PROJECT_ROOT / "configs/hpc.yaml")):
        raise ValueError("--config must be configs/hpc.yaml")
    if args.set not in ([], ["inference.fallback_to_heuristic=false"]):
        raise ValueError("unsupported --set override")
    if not 10.0 <= args.poll_seconds <= 3600.0:
        raise ValueError("poll interval must be in [10, 3600] seconds")
    value = run_controller(
        state_root=args.state_root,
        matrix_authority_root=args.matrix_authority_root,
        owner_registry=args.owner_registry,
        poll_seconds=args.poll_seconds,
        controller_id=args.controller_id,
        once=args.once,
        proc_root=args.proc_root,
    )
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
