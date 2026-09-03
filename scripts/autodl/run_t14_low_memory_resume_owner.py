#!/usr/bin/env python3
"""Serialize, admit, and own one low-memory Taste T14 resume process."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.tastemolnet_t14_resume import (  # noqa: E402
    T14ResumeError,
    assert_auditor_serialized,
    evaluate_memory_admission,
    inspect_process_identity,
    load_canary_receipt,
    load_resume_spec,
    read_cgroup_counter,
)


WAITING_EXIT = 75
OWNER_SCHEMA = "tastemolnet_t14_low_memory_resume_owner_v1"


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _physical_file(path: Path, *, label: str, executable: bool = False) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise T14ResumeError(f"T14 {label} must be one absolute physical file")
    resolved = path.resolve(strict=True)
    if resolved != path or not path.is_file():
        raise T14ResumeError(f"T14 {label} path is aliased or not a file")
    if executable and not os.access(path, os.X_OK):
        raise T14ResumeError(f"T14 {label} is not executable")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=_absolute, required=True)
    parser.add_argument("--resume-spec", type=_absolute, required=True)
    parser.add_argument("--canary-receipt", type=_absolute, required=True)
    parser.add_argument("--owner-root", type=_absolute, required=True)
    parser.add_argument("--science-wrapper", type=_absolute, required=True)
    parser.add_argument("--cgroup-limit-file", type=_absolute, required=True)
    parser.add_argument("--cgroup-current-file", type=_absolute, required=True)
    parser.add_argument("--auditor-pid", type=int, required=True)
    parser.add_argument("--auditor-start-ticks", type=int, required=True)
    parser.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _physical_file(args.config, label="config")
    wrapper = _physical_file(args.science_wrapper, label="science wrapper", executable=True)
    spec = load_resume_spec(args.resume_spec)
    owner_root = args.owner_root
    if not owner_root.is_absolute() or owner_root.is_symlink():
        raise T14ResumeError("T14 owner root must be one absolute physical path")
    owner_root.mkdir(parents=True, exist_ok=True)
    if owner_root.resolve(strict=True) != owner_root:
        raise T14ResumeError("T14 owner root contains an alias")
    if not args.canary_receipt.is_file() or args.canary_receipt.is_symlink():
        _atomic_json(
            owner_root / "admission.json",
            {
                "schema_version": OWNER_SCHEMA,
                "status": "WAITING_T14_PARITY_CANARY",
                "checkpoint_digest": spec["checkpoint_digest"],
                "required_canary_steps_max": 50,
                "required_forced_checkpoint_save_reload": True,
                "science_started": False,
                "written_at": _utc_now(),
            },
        )
        return WAITING_EXIT
    receipt = load_canary_receipt(args.canary_receipt, spec=spec)
    observed_auditor = assert_auditor_serialized(
        auditor_pid=args.auditor_pid,
        auditor_start_ticks=args.auditor_start_ticks,
        proc_root=args.proc_root,
    )
    lock_path = Path(str(spec["full_state_lock_path"]))
    if (
        not lock_path.is_absolute()
        or lock_path.is_symlink()
        or lock_path.parent.resolve(strict=True)
        != Path(str(spec["checkpoint_root"])).resolve(strict=True)
    ):
        raise T14ResumeError("T14 full-state consumer lock is not spec-bound")
    lock_handle = lock_path.open("a+b")
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _atomic_json(
                owner_root / "admission.json",
                {
                    "schema_version": OWNER_SCHEMA,
                    "status": "WAITING_FULL_STATE_CONSUMER_SERIALIZATION",
                    "written_at": _utc_now(),
                },
            )
            return WAITING_EXIT

        limit = read_cgroup_counter(args.cgroup_limit_file, allow_max=True)
        current = read_cgroup_counter(args.cgroup_current_file)
        if (
            str(args.cgroup_limit_file) != spec["memory"]["cgroup_limit_path"]
            or str(args.cgroup_current_file) != spec["memory"]["cgroup_current_path"]
        ):
            raise T14ResumeError("T14 live cgroup counters differ from the canary contract")
        admission = evaluate_memory_admission(
            spec,
            cgroup_limit_bytes=limit,
            cgroup_current_bytes=current,
            optimized_canary_receipt=receipt,
        )
        admission_payload = {
            "schema_version": OWNER_SCHEMA,
            "status": admission.state,
            "basis": admission.basis,
            "cgroup_limit_bytes": limit,
            "cgroup_current_bytes": current,
            "available_headroom_bytes": admission.available_headroom_bytes,
            "required_headroom_bytes": admission.required_headroom_bytes,
            "safety_margin_bytes": spec["memory"]["safety_margin_bytes"],
            "checkpoint_digest": spec["checkpoint_digest"],
            "resume_execution_commit": spec["resume_execution_commit"],
            "canary_receipt_sha256": receipt["receipt_sha256"],
            "auditor_pid": args.auditor_pid,
            "auditor_expected_start_ticks": args.auditor_start_ticks,
            "auditor_observed_start_ticks": observed_auditor.start_ticks,
            "auditor_observed_state": observed_auditor.state,
            "full_state_consumers_serialized": True,
            "science_started": False,
            "written_at": _utc_now(),
        }
        _atomic_json(owner_root / "admission.json", admission_payload)
        if not admission.admitted:
            return WAITING_EXIT
        if args.dry_run:
            print(json.dumps(admission_payload, sort_keys=True), flush=True)
            return 0

        environment = dict(os.environ)
        environment.update(
            {
                "TASTEMOLNET_T14_RESUME": "1",
                "T14_RESUME_SPEC": str(args.resume_spec),
                "T14_FULL_STATE_CONSUMER_LOCK": str(lock_path),
                "T14_FULL_STATE_CONSUMER_LOCK_FD": str(lock_handle.fileno()),
            }
        )
        child = subprocess.Popen(
            [str(wrapper)],
            env=environment,
            pass_fds=(lock_handle.fileno(),),
        )
        owner_identity = inspect_process_identity(os.getpid(), proc_root=args.proc_root)
        child_identity = inspect_process_identity(child.pid, proc_root=args.proc_root)
        command_sha256 = hashlib.sha256(wrapper.read_bytes()).hexdigest()
        stop_requested = False

        def request_stop(_signum: int, _frame: object) -> None:
            nonlocal stop_requested
            stop_requested = True
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        while child.poll() is None:
            _atomic_json(
                owner_root / "heartbeat.json",
                {
                    **admission_payload,
                    "status": "OWNER_CONFIRMED",
                    "science_started": True,
                    "owner_pid": owner_identity.pid,
                    "owner_start_ticks": owner_identity.start_ticks,
                    "science_pid": child_identity.pid,
                    "science_start_ticks": child_identity.start_ticks,
                    "science_wrapper": str(wrapper),
                    "science_wrapper_sha256": command_sha256,
                    "stop_requested": stop_requested,
                    "written_at": _utc_now(),
                },
            )
            time.sleep(30)
        return_code = int(child.wait())
        _atomic_json(
            owner_root / "terminal.json",
            {
                **admission_payload,
                "status": "PASS" if return_code == 0 else "FAILED",
                "science_started": True,
                "owner_pid": owner_identity.pid,
                "owner_start_ticks": owner_identity.start_ticks,
                "science_pid": child_identity.pid,
                "science_start_ticks": child_identity.start_ticks,
                "science_exit_code": return_code,
                "stop_requested": stop_requested,
                "written_at": _utc_now(),
            },
        )
        return return_code
    finally:
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
