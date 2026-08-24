"""Hold the global ComRecGC high-memory lock across the AIDS v5 supervisor.

The helper queues on the same physical flock before the v5 science child is
started.  Once the protected repair-v4 reader exits naturally, this helper
acquires the lock and retains it until the exact supervisor generation exits.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
from typing import Any, Sequence


STATE_SCHEMA = "aids_comrecgc_v5_highmem_handover_v1"


def _proc_generation(proc_root: Path, pid: int) -> tuple[str, int] | None:
    try:
        raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None
    closing = raw.rfind(")")
    if closing < 0:
        raise RuntimeError("malformed supervisor proc stat")
    fields = raw[closing + 2 :].split()
    if len(fields) <= 19:
        raise RuntimeError("truncated supervisor proc stat")
    return fields[0], int(fields[19])


def _atomic_state(path: Path, payload: dict[str, Any]) -> None:
    if path.is_symlink():
        raise RuntimeError("handover state path may not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def run_handover(
    *,
    lock_path: str | Path,
    state_path: str | Path,
    supervisor_pid: int,
    proc_root: str | Path = "/proc",
    poll_seconds: float = 1.0,
) -> int:
    if supervisor_pid <= 0 or poll_seconds <= 0:
        raise RuntimeError("invalid handover process contract")
    proc = Path(proc_root).expanduser().resolve(strict=True)
    generation = _proc_generation(proc, supervisor_pid)
    if generation is None or generation[0] == "Z":
        raise RuntimeError("supervisor generation is not alive")
    supervisor_start_ticks = generation[1]
    lock = Path(lock_path).expanduser()
    state_file = Path(state_path).expanduser()
    if not lock.is_absolute() or not state_file.is_absolute():
        raise RuntimeError("handover paths must be absolute")
    lock.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    lock_fd = os.open(lock, flags, 0o600)
    try:
        lock_stat = os.fstat(lock_fd)
        if not stat.S_ISREG(lock_stat.st_mode):
            raise RuntimeError("high-memory lock is not a regular file")
        identity = {
            "schema_version": STATE_SCHEMA,
            "lock_path": str(lock.resolve(strict=True)),
            "lock_device": int(lock_stat.st_dev),
            "lock_inode": int(lock_stat.st_ino),
            "supervisor_pid": int(supervisor_pid),
            "supervisor_start_ticks": int(supervisor_start_ticks),
            "helper_pid": int(os.getpid()),
        }
        _atomic_state(state_file, {**identity, "status": "QUEUED"})

        def fail_closed(error: str) -> None:
            try:
                _atomic_state(
                    state_file,
                    {
                        **identity,
                        "status": "FAILED",
                        "error": error,
                    },
                )
            finally:
                os._exit(75)

        def watchdog() -> None:
            while True:
                try:
                    current_lock = os.stat(lock, follow_symlinks=False)
                    if (
                        not stat.S_ISREG(current_lock.st_mode)
                        or (int(current_lock.st_dev), int(current_lock.st_ino))
                        != (int(lock_stat.st_dev), int(lock_stat.st_ino))
                    ):
                        fail_closed("HIGHMEM_LOCK_PATH_IDENTITY_CHANGED")
                    current = _proc_generation(proc, supervisor_pid)
                    if (
                        current is None
                        or current[0] == "Z"
                        or int(current[1]) != int(supervisor_start_ticks)
                    ):
                        os._exit(0)
                except BaseException as exc:
                    fail_closed(f"HANDOVER_WATCHDOG:{type(exc).__name__}:{exc}")
                time.sleep(min(float(poll_seconds), 0.1))

        monitor = threading.Thread(
            target=watchdog,
            name="aids-v5-highmem-lock-watchdog",
            daemon=True,
        )
        monitor.start()
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        after = os.stat(lock, follow_symlinks=False)
        if (
            not stat.S_ISREG(after.st_mode)
            or (int(after.st_dev), int(after.st_ino))
            != (int(lock_stat.st_dev), int(lock_stat.st_ino))
        ):
            fail_closed("HIGHMEM_LOCK_PATH_IDENTITY_CHANGED_AT_ACQUIRE")
        _atomic_state(state_file, {**identity, "status": "ACQUIRED"})
        while True:
            time.sleep(poll_seconds)
    except BaseException as exc:
        try:
            _atomic_state(
                state_file,
                {
                    "schema_version": STATE_SCHEMA,
                    "status": "FAILED",
                    "helper_pid": int(os.getpid()),
                    "error": f"{type(exc).__name__}:{exc}",
                },
            )
        except BaseException:
            pass
        raise
    finally:
        os.close(lock_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--supervisor-pid", type=int, required=True)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_handover(
        lock_path=args.lock_path,
        state_path=args.state_path,
        supervisor_pid=args.supervisor_pid,
        proc_root=args.proc_root,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
