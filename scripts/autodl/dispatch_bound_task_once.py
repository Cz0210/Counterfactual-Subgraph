#!/usr/bin/env python3
"""Launch missing immutable main tasks once, then exit after owner evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.main_ready_task_specs import (  # noqa: E402
    OWNER_SCHEMA,
    atomic_json,
    command_from_spec,
    conflicting_output_writers,
    load_spec,
    probe_owner,
    probe_terminal,
    process_identity,
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _existing_owner(
    spec: dict[str, Any], *, proc_root: Path = Path("/proc")
) -> dict[str, Any] | None:
    evidence = probe_owner(spec, proc_root=proc_root)
    return evidence if evidence["state"] == "OWNER_CONFIRMED" else None


def _conflicting_output_writer(
    spec: dict[str, Any], *, proc_root: Path = Path("/proc")
) -> dict[str, Any] | None:
    """Return an unowned process naming this exact root; never adopt it."""

    conflicts = conflicting_output_writers(spec["output_root"], proc_root=proc_root)
    return conflicts[0] if conflicts else None


def _wait_owner(
    spec: dict[str, Any],
    child: subprocess.Popen[bytes],
    timeout: int,
    *,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        owner = _existing_owner(spec, proc_root=proc_root)
        if owner:
            return owner
        if child.poll() is not None:
            return None
        time.sleep(2)
    return None


def _launcher_state(
    child: subprocess.Popen[bytes], *, proc_root: Path = Path("/proc")
) -> dict[str, Any]:
    """Reap an exited launcher or block retries while it is still live."""

    returncode = child.poll()
    if returncode is None:
        return {
            "state": "BLOCKED_UNCONFIRMED_LIVE_LAUNCHER",
            "launcher": process_identity(child.pid, proc_root=proc_root),
            "returncode": None,
            "cleanup": {
                "launcher_reaped": False,
                "signal_sent": False,
                "retry_allowed": False,
            },
        }
    child.wait()
    return {
        "state": "LAUNCHER_EXITED_WITHOUT_OWNER",
        "launcher_pid": child.pid,
        "returncode": returncode,
        "cleanup": {
            "launcher_reaped": True,
            "process_absent": process_identity(child.pid, proc_root=proc_root) is None,
            "signal_sent": False,
            "retry_allowed": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--task-spec", type=_absolute, action="append", required=True)
    parser.add_argument("--existing-control-root", type=_absolute, required=True)
    parser.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    args = parser.parse_args(argv)
    supplied_root = args.existing_control_root
    root = supplied_root.resolve(strict=True)
    if supplied_root.is_symlink() or root != supplied_root or not root.is_dir():
        raise ValueError("existing control root must be one physical directory")
    owner_root = root / "main-ready-owners"
    owner_root.mkdir(exist_ok=True)
    if owner_root.is_symlink() or owner_root.resolve(strict=True) != owner_root:
        raise ValueError("main-ready owner root must be one physical directory")
    lock_path = root / "main-ready-bound-dispatch.lock"
    if lock_path.is_symlink():
        raise ValueError("main-ready dispatch lock cannot be a symlink")
    lock = lock_path.open("a+b")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 75
    outcomes: list[dict[str, Any]] = []
    with lock:
        specs = [load_spec(path) for path in args.task_spec]
        authorities = {spec["matrix_authority_root"] for spec in specs}
        if len(authorities) != 1:
            raise RuntimeError("one-shot tasks do not share one matrix authority")
        for spec_path, spec in zip(args.task_spec, specs, strict=True):
            task_root = owner_root / spec["task_id"]
            task_root.mkdir(exist_ok=True)
            if task_root.is_symlink() or task_root.resolve(strict=True) != task_root:
                raise ValueError("main-ready task owner root must be physical")
            existing = _existing_owner(spec, proc_root=args.proc_root)
            if existing:
                evidence = {
                    "schema_version": OWNER_SCHEMA,
                    "task_id": spec["task_id"],
                    "status": "OWNER_CONFIRMED",
                    "owner": existing,
                    "duplicate_started": False,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                }
                atomic_json(task_root / "owner_evidence.json", evidence)
                outcomes.append(evidence)
                continue
            terminal = probe_terminal(spec)
            if terminal is not None:
                evidence = {
                    "schema_version": OWNER_SCHEMA,
                    "task_id": spec["task_id"],
                    "status": (
                        "TERMINAL_OBSERVED"
                        if terminal.get("state") == "TERMINAL"
                        else "BLOCKED_INVALID_TERMINAL"
                    ),
                    "owner": None,
                    "terminal": terminal,
                    "duplicate_started": False,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                }
                atomic_json(task_root / "owner_evidence.json", evidence)
                outcomes.append(evidence)
                continue
            conflict = _conflicting_output_writer(spec, proc_root=args.proc_root)
            if conflict is not None:
                evidence = {
                    "schema_version": OWNER_SCHEMA,
                    "task_id": spec["task_id"],
                    "status": "BLOCKED_UNVERIFIED_OUTPUT_WRITER",
                    "owner": None,
                    "conflicting_process": conflict,
                    "duplicate_started": False,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                }
                atomic_json(task_root / "owner_evidence.json", evidence)
                outcomes.append(evidence)
                continue
            command = command_from_spec(spec)
            environment = dict(os.environ)
            environment.update(spec["required_environment"])
            environment["MAIN_READY_TASK_SPEC"] = str(spec_path)
            log_path = task_root / "launcher.log"
            if log_path.is_symlink():
                raise ValueError("main-ready launcher log cannot be a symlink")
            log = log_path.open("ab", buffering=0)
            owner = None
            failures: list[dict[str, Any]] = []
            # Initial launch, followed by the three authorized retries.
            try:
                for attempt, delay in enumerate((0, 60, 120, 300), start=1):
                    if delay:
                        time.sleep(delay)
                    existing = _existing_owner(spec, proc_root=args.proc_root)
                    if existing:
                        owner = existing
                        break
                    terminal = probe_terminal(spec)
                    if terminal is not None:
                        failures.append(
                            {
                                "attempt": attempt,
                                "state": "TERMINAL_OBSERVED",
                                "terminal": terminal,
                            }
                        )
                        break
                    conflict = _conflicting_output_writer(spec, proc_root=args.proc_root)
                    if conflict is not None:
                        failures.append(
                            {
                                "attempt": attempt,
                                "state": "BLOCKED_UNVERIFIED_OUTPUT_WRITER",
                                "conflicting_process": conflict,
                            }
                        )
                        break
                    try:
                        child = subprocess.Popen(
                            command,
                            cwd=spec["repo_root"],
                            env=environment,
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    except OSError as exc:
                        failures.append(
                            {
                                "attempt": attempt,
                                "state": "LAUNCHER_CREATE_FAILED",
                                "error": f"{type(exc).__name__}: {exc}",
                                "cleanup": {
                                    "child_created": False,
                                    "retry_allowed": False,
                                },
                            }
                        )
                        break
                    owner = _wait_owner(
                        spec,
                        child,
                        int(spec.get("owner_timeout_seconds", 60)),
                        proc_root=args.proc_root,
                    )
                    if owner:
                        break
                    launcher = _launcher_state(child, proc_root=args.proc_root)
                    if launcher["state"] == "BLOCKED_UNCONFIRMED_LIVE_LAUNCHER":
                        failures.append({"attempt": attempt, **launcher})
                        # An unconfirmed child may already own scientific
                        # state. Never overlap it with a retry merely because
                        # its heartbeat contract is incomplete or delayed.
                        break
                    terminal = probe_terminal(spec)
                    failures.append(
                        {
                            "attempt": attempt,
                            **launcher,
                            "state": (
                                "TERMINAL_OBSERVED"
                                if terminal is not None
                                else "LAUNCHER_EXITED_WITHOUT_OWNER"
                            ),
                            "terminal": terminal,
                        }
                    )
                    if terminal is not None:
                        break
            finally:
                log.close()
            final_terminal = probe_terminal(spec)
            if owner:
                status = "OWNER_CONFIRMED"
            elif final_terminal is not None:
                status = (
                    "TERMINAL_OBSERVED"
                    if final_terminal.get("state") == "TERMINAL"
                    else "BLOCKED_INVALID_TERMINAL"
                )
            elif failures and failures[-1].get("state") == "BLOCKED_UNCONFIRMED_LIVE_LAUNCHER":
                status = "BLOCKED_UNCONFIRMED_LIVE_LAUNCHER"
            elif failures and failures[-1].get("state") == "BLOCKED_UNVERIFIED_OUTPUT_WRITER":
                status = "BLOCKED_UNVERIFIED_OUTPUT_WRITER"
            else:
                status = "BLOCKED_LAUNCH_RETRY_EXHAUSTED"
            evidence = {
                "schema_version": OWNER_SCHEMA,
                "task_id": spec["task_id"],
                "status": status,
                "owner": owner,
                "terminal": final_terminal,
                "failures": failures,
                "duplicate_started": False,
                "matrix_authority_root": spec["matrix_authority_root"],
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(task_root / "owner_evidence.json", evidence)
            outcomes.append(evidence)
    def accepted(row: dict[str, Any]) -> bool:
        if row["status"] == "OWNER_CONFIRMED":
            return True
        terminal = row.get("terminal")
        if row["status"] != "TERMINAL_OBSERVED" or not isinstance(terminal, dict):
            return False
        status = str(terminal.get("status", ""))
        return "PASS" in status and "FAILED" not in status and "BLOCKED" not in status

    result = {
        "status": "PASS" if all(accepted(row) for row in outcomes) else "BLOCKED",
        "outcomes": outcomes,
    }
    atomic_json(root / "main_ready_one_shot_dispatch.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
