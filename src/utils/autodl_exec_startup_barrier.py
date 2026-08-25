"""Fail-closed, durable startup barrier for an ``exec``-based worker.

The parent owns two capabilities while a worker is being registered:

* an exclusive ``flock`` on a physical, mode-0600 lock file; and
* the write end of an anonymous release pipe.

The child starts this module with the pipe read descriptor and a duplicate of
the locked descriptor.  It executes the requested target only after reading
the *exact* one-time release token followed by EOF.  EOF, malformed input, a
changed durable record, or a changed lock-file identity all fail closed.

This module deliberately has no controller-specific policy.  A controller can
arm and launch a wrapper, durably publish its PID/identity, and only then call
``release``.  If the controller dies first, kernel descriptor teardown sends
EOF and the target is never executed.

The lock protects only the pre-exec interval: the wrapper is required to close
it immediately before ``execvpe``.  Thus a free historical barrier lock is not
proof that a released target has exited.  Resume authority must come from a
caller-owned, fsynced PRE_RELEASE phase plus target-process identity checks.
The caller also owns trust in the selected Python executable, working
directory, and environment used to import this module; ``-S`` suppresses site
initialization, but this utility is not a sandbox against a hostile same-UID
Python path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


RECORD_SCHEMA = "autodl_exec_startup_barrier_v1"
RECORD_KIND = "durable_exec_startup_barrier"
RECORD_STATE = "ARMED_UNRELEASED"
MODULE_NAME = "src.utils.autodl_exec_startup_barrier"
RELEASE_TOKEN_BYTES = 32
MAX_RECORD_BYTES = 64 * 1024


class StartupBarrierError(RuntimeError):
    """Base exception for a fail-closed startup barrier."""


class StartupBarrierBusy(StartupBarrierError):
    """The recorded barrier lock is still owned by another open description."""


class StartupBarrierValidationError(StartupBarrierError):
    """The durable record, launcher, or physical lock binding drifted."""


def _argv_tuple(argv: Sequence[str], *, field: str) -> tuple[str, ...]:
    values = tuple(argv)
    if not values:
        raise StartupBarrierValidationError(f"{field} must not be empty")
    if any(not isinstance(value, str) or not value or "\x00" in value for value in values):
        raise StartupBarrierValidationError(
            f"{field} must contain only non-empty NUL-free strings"
        )
    return values


def argv_sha256(argv: Sequence[str]) -> str:
    """Hash an argv without ambiguous shell joining or locale conversion."""

    values = _argv_tuple(argv, field="argv")
    payload = json.dumps(
        list(values), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _launcher_argv(
    *,
    python_executable: str,
    record_path: str,
    release_read_fd: int,
    lock_fd: int,
    target_argv: Sequence[str],
) -> tuple[str, ...]:
    return (
        python_executable,
        "-S",
        "-m",
        MODULE_NAME,
        "--record",
        record_path,
        "--release-read-fd",
        str(release_read_fd),
        "--lock-fd",
        str(lock_fd),
        "--",
        *_argv_tuple(target_argv, field="target_argv"),
    )


@dataclass(frozen=True, slots=True)
class StartupBarrierRecord:
    schema: str
    kind: str
    state: str
    lock_path: str
    lock_dev: int
    lock_inode: int
    lock_mode: int
    lock_uid: int
    lock_nlink: int
    record_path: str
    python_executable: str
    release_read_fd: int
    lock_fd: int
    release_token_bytes: int
    release_token_sha256: str
    target_argv: tuple[str, ...]
    target_argv_sha256: str
    launcher_argv: tuple[str, ...]
    launcher_argv_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind,
            "state": self.state,
            "lock_path": self.lock_path,
            "lock_dev": self.lock_dev,
            "lock_inode": self.lock_inode,
            "lock_mode": self.lock_mode,
            "lock_uid": self.lock_uid,
            "lock_nlink": self.lock_nlink,
            "record_path": self.record_path,
            "python_executable": self.python_executable,
            "release_read_fd": self.release_read_fd,
            "lock_fd": self.lock_fd,
            "release_token_bytes": self.release_token_bytes,
            "release_token_sha256": self.release_token_sha256,
            "target_argv": list(self.target_argv),
            "target_argv_sha256": self.target_argv_sha256,
            "launcher_argv": list(self.launcher_argv),
            "launcher_argv_sha256": self.launcher_argv_sha256,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "StartupBarrierRecord":
        expected = set(cls.__dataclass_fields__)
        if set(raw) != expected:
            missing = sorted(expected - set(raw))
            extra = sorted(set(raw) - expected)
            raise StartupBarrierValidationError(
                f"barrier record fields differ (missing={missing}, extra={extra})"
            )
        if not isinstance(raw["target_argv"], list) or not isinstance(
            raw["launcher_argv"], list
        ):
            raise StartupBarrierValidationError(
                "recorded target and launcher argv must be JSON arrays"
            )
        try:
            record = cls(
                schema=str(raw["schema"]),
                kind=str(raw["kind"]),
                state=str(raw["state"]),
                lock_path=str(raw["lock_path"]),
                lock_dev=_strict_nonnegative_int(raw["lock_dev"], "lock_dev"),
                lock_inode=_strict_nonnegative_int(raw["lock_inode"], "lock_inode"),
                lock_mode=_strict_nonnegative_int(raw["lock_mode"], "lock_mode"),
                lock_uid=_strict_nonnegative_int(raw["lock_uid"], "lock_uid"),
                lock_nlink=_strict_nonnegative_int(raw["lock_nlink"], "lock_nlink"),
                record_path=str(raw["record_path"]),
                python_executable=str(raw["python_executable"]),
                release_read_fd=_strict_nonnegative_int(
                    raw["release_read_fd"], "release_read_fd"
                ),
                lock_fd=_strict_nonnegative_int(raw["lock_fd"], "lock_fd"),
                release_token_bytes=_strict_nonnegative_int(
                    raw["release_token_bytes"], "release_token_bytes"
                ),
                release_token_sha256=str(raw["release_token_sha256"]),
                target_argv=_argv_tuple(raw["target_argv"], field="target_argv"),
                target_argv_sha256=str(raw["target_argv_sha256"]),
                launcher_argv=_argv_tuple(raw["launcher_argv"], field="launcher_argv"),
                launcher_argv_sha256=str(raw["launcher_argv_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StartupBarrierValidationError("malformed barrier record") from exc
        return record


def _strict_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StartupBarrierValidationError(f"{field} must be a non-negative integer")
    return value


def _require_posix_primitives() -> None:
    if not hasattr(os, "O_NOFOLLOW"):
        raise StartupBarrierError("O_NOFOLLOW is required for startup barriers")
    if not hasattr(fcntl, "flock"):
        raise StartupBarrierError("flock is required for startup barriers")


def _validate_physical_lock_stat(
    observed: os.stat_result,
    *,
    expected: StartupBarrierRecord | None = None,
) -> None:
    if not stat.S_ISREG(observed.st_mode):
        raise StartupBarrierValidationError("barrier lock is not a regular file")
    mode = stat.S_IMODE(observed.st_mode)
    if mode != 0o600:
        raise StartupBarrierValidationError(
            f"barrier lock mode must be 0600, observed {mode:04o}"
        )
    if observed.st_uid != os.getuid():
        raise StartupBarrierValidationError("barrier lock is not owned by this uid")
    if observed.st_nlink != 1:
        raise StartupBarrierValidationError("barrier lock link count must equal one")
    if expected is not None:
        binding = (
            observed.st_dev,
            observed.st_ino,
            mode,
            observed.st_uid,
            observed.st_nlink,
        )
        recorded = (
            expected.lock_dev,
            expected.lock_inode,
            expected.lock_mode,
            expected.lock_uid,
            expected.lock_nlink,
        )
        if binding != recorded:
            raise StartupBarrierValidationError(
                "barrier lock identity differs from durable record"
            )


def _validate_lock_fd_and_path(fd: int, record: StartupBarrierRecord) -> None:
    try:
        fd_stat = os.fstat(fd)
    except OSError as exc:
        raise StartupBarrierValidationError("barrier lock fd is not open") from exc
    _validate_physical_lock_stat(fd_stat, expected=record)
    try:
        path_stat = os.stat(record.lock_path, follow_symlinks=False)
    except OSError as exc:
        raise StartupBarrierValidationError("barrier lock path cannot be stat'ed") from exc
    if stat.S_ISLNK(path_stat.st_mode):
        raise StartupBarrierValidationError("barrier lock path became a symlink")
    _validate_physical_lock_stat(path_stat, expected=record)
    if (fd_stat.st_dev, fd_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino):
        raise StartupBarrierValidationError(
            "barrier lock fd and path no longer name the same inode"
        )


def _assert_exclusive_lock(fd: int) -> None:
    """Ensure this open description owns the exclusive barrier lock."""

    try:
        # For an inherited descriptor this is a no-op on the same open-file
        # description.  If the lock was accidentally dropped, it reacquires it
        # only when no competing owner exists; contention is fail-closed.
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            raise StartupBarrierValidationError(
                "barrier lock fd does not own an exclusive lock"
            ) from exc
        raise


def _open_nofollow(path: str, flags: int, mode: int | None = None) -> int:
    _require_posix_primitives()
    open_flags = flags | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        open_flags |= os.O_CLOEXEC
    try:
        if mode is None:
            return os.open(path, open_flags)
        return os.open(path, open_flags, mode)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise StartupBarrierValidationError(f"refusing symlink path: {path}") from exc
        raise


def _fsync_directory(path: str) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publication_temp_paths(record_path: str) -> list[str]:
    parent = os.path.dirname(record_path)
    prefix = os.path.basename(record_path) + ".tmp."
    return sorted(
        os.path.join(parent, name)
        for name in os.listdir(parent)
        if name.startswith(prefix)
    )


def _reconcile_interrupted_record_publication(record_path: str) -> None:
    """Remove only non-authoritative temp links while holding the lock.

    A fresh publication uses ``link(temp, final)`` for no-replace semantics.
    SIGKILL can therefore leave either an unlinked temp or final+temp names for
    one inode.  Neither state can have launched a wrapper because ``arm`` has
    not returned.  Under the barrier's exclusive lock we reject unsafe temp
    objects, remove the private mode-0600 names, and fsync the directory.  For
    the final+temp case this restores the authoritative final record to
    ``nlink == 1`` before it is parsed.
    """

    temporary_paths = _publication_temp_paths(record_path)
    try:
        final_stat = os.stat(record_path, follow_symlinks=False)
    except FileNotFoundError:
        final_stat = None
    if final_stat is not None:
        if stat.S_ISLNK(final_stat.st_mode) or not stat.S_ISREG(final_stat.st_mode):
            raise StartupBarrierValidationError(
                "barrier record publication target is not a physical file"
            )
        if stat.S_IMODE(final_stat.st_mode) != 0o600 or final_stat.st_uid != os.getuid():
            raise StartupBarrierValidationError(
                "barrier record publication target security fields are unsafe"
            )
        if final_stat.st_nlink not in {1, 2}:
            raise StartupBarrierValidationError(
                "barrier record publication target has unexpected link count"
            )
    same_inode_temps = 0
    for temporary in temporary_paths:
        observed = os.stat(temporary, follow_symlinks=False)
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
            raise StartupBarrierValidationError(
                "barrier publication temp is not a physical regular file"
            )
        if (
            stat.S_IMODE(observed.st_mode) != 0o600
            or observed.st_uid != os.getuid()
            or observed.st_nlink not in {1, 2}
        ):
            raise StartupBarrierValidationError(
                "barrier publication temp security fields are unsafe"
            )
        if final_stat is not None and (observed.st_dev, observed.st_ino) == (
            final_stat.st_dev,
            final_stat.st_ino,
        ):
            same_inode_temps += 1
    if final_stat is not None and final_stat.st_nlink == 2 and same_inode_temps != 1:
        raise StartupBarrierValidationError(
            "cannot reconcile linked barrier record publication"
        )
    if final_stat is not None and final_stat.st_nlink == 1 and same_inode_temps:
        raise StartupBarrierValidationError(
            "single-link barrier record unexpectedly aliases a temp"
        )
    if temporary_paths:
        for temporary in temporary_paths:
            os.unlink(temporary)
        _fsync_directory(os.path.dirname(record_path))
    if final_stat is not None:
        normalized = os.stat(record_path, follow_symlinks=False)
        _validate_record_file_stat(normalized)


def reconcile_interrupted_startup_barrier_publication(
    *,
    lock_path: str | os.PathLike[str],
    record_path: str | os.PathLike[str],
    timeout_seconds: float = 0.0,
) -> bool:
    """Normalize a pre-``arm`` record publication after parent hard failure.

    Callers durably in PRE_ARM cannot use the strict record reader yet: a
    killed publisher may have left either a temp-only inode or final+temp hard
    links.  This public operation acquires the exact physical barrier lock,
    invokes the same reconciliation used by :func:`arm_exec_startup_barrier`,
    and returns whether an authoritative final record remains.  It never
    creates a missing lock, so arbitrary temp names cannot bootstrap authority.
    """

    _require_posix_primitives()
    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be non-negative")
    absolute_lock = os.path.abspath(os.fspath(lock_path))
    absolute_record = os.path.abspath(os.fspath(record_path))
    if absolute_lock == absolute_record:
        raise StartupBarrierValidationError("lock and record paths must differ")
    try:
        os.stat(absolute_lock, follow_symlinks=False)
    except FileNotFoundError:
        try:
            record_exists = os.path.lexists(absolute_record)
            temporary_exists = bool(_publication_temp_paths(absolute_record))
        except FileNotFoundError:
            record_exists = False
            temporary_exists = False
        if record_exists or temporary_exists:
            raise StartupBarrierValidationError(
                "barrier publication exists without its physical lock"
            )
        return False
    lock_fd = _open_nofollow(absolute_lock, os.O_RDWR)
    try:
        before = os.fstat(lock_fd)
        _validate_physical_lock_stat(before)
        current = os.stat(absolute_lock, follow_symlinks=False)
        _validate_physical_lock_stat(current)
        if (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino):
            raise StartupBarrierValidationError(
                "barrier lock fd/path inode race during reconciliation"
            )
        _acquire_exclusive_lock(lock_fd, timeout_seconds=timeout_seconds)
        locked = os.fstat(lock_fd)
        current = os.stat(absolute_lock, follow_symlinks=False)
        if (locked.st_dev, locked.st_ino) != (current.st_dev, current.st_ino):
            raise StartupBarrierValidationError(
                "barrier lock changed after reconciliation lock acquisition"
            )
        _reconcile_interrupted_record_publication(absolute_record)
        os.fsync(lock_fd)
        try:
            final = os.stat(absolute_record, follow_symlinks=False)
        except FileNotFoundError:
            return False
        _validate_record_file_stat(final)
        return True
    finally:
        os.close(lock_fd)


def _acquire_exclusive_lock(fd: int, *, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
        else:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise StartupBarrierBusy("startup barrier lock is already held")
        time.sleep(min(0.05, remaining))


def _record_file_binding(observed: os.stat_result) -> tuple[int, ...]:
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_uid,
        observed.st_nlink,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _validate_record_file_stat(observed: os.stat_result) -> None:
    if not stat.S_ISREG(observed.st_mode):
        raise StartupBarrierValidationError("barrier record is not a regular file")
    if stat.S_IMODE(observed.st_mode) != 0o600:
        raise StartupBarrierValidationError("barrier record mode must be 0600")
    if observed.st_uid != os.getuid() or observed.st_nlink != 1:
        raise StartupBarrierValidationError(
            "barrier record ownership/link count is unsafe"
        )


def _write_record_durable(
    path: str,
    record: StartupBarrierRecord,
    *,
    replace_expected: StartupBarrierRecord | None,
) -> None:
    parent = os.path.dirname(path)
    if not parent or not os.path.isdir(parent):
        raise StartupBarrierValidationError("barrier record parent must already exist")
    if replace_expected is None:
        try:
            os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise StartupBarrierValidationError(
                "fresh barrier refuses to overwrite an existing record"
            )
    else:
        current_record = validate_startup_barrier_record(
            path,
            expected_target_argv=replace_expected.target_argv,
            expected_launcher_argv=replace_expected.launcher_argv,
            validate_lock_path=False,
        )
        if current_record != replace_expected:
            raise StartupBarrierValidationError(
                "rearm record changed after lock acquisition"
            )
    payload = (
        json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if len(payload) > MAX_RECORD_BYTES:
        raise StartupBarrierValidationError(
            "barrier record exceeds the frozen publication byte limit"
        )
    temporary = f"{path}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    fd = _open_nofollow(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise StartupBarrierError("short write while persisting barrier record")
            offset += written
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    else:
        os.close(fd)
    try:
        if replace_expected is None:
            # ``link`` provides no-replace publication.  A crash before the
            # following unlink leaves nlink=2, which the reader rejects rather
            # than mistaking a partial fresh publication for authority.
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError as exc:
                raise StartupBarrierValidationError(
                    "fresh barrier record appeared concurrently"
                ) from exc
            os.unlink(temporary)
            temporary = ""
        else:
            os.replace(temporary, path)
            temporary = ""
        _fsync_directory(parent)
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _load_record(path: str) -> StartupBarrierRecord:
    absolute = os.path.abspath(path)
    fd = _open_nofollow(absolute, os.O_RDONLY)
    try:
        before_fd = os.fstat(fd)
        _validate_record_file_stat(before_fd)
        try:
            before_path = os.stat(absolute, follow_symlinks=False)
        except OSError as exc:
            raise StartupBarrierValidationError(
                "barrier record path cannot be stat'ed"
            ) from exc
        _validate_record_file_stat(before_path)
        if (before_fd.st_dev, before_fd.st_ino) != (
            before_path.st_dev,
            before_path.st_ino,
        ):
            raise StartupBarrierValidationError(
                "barrier record fd/path inode race before read"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(8192, MAX_RECORD_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_RECORD_BYTES:
                raise StartupBarrierValidationError("barrier record is too large")
        try:
            raw = json.loads(b"".join(chunks).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StartupBarrierValidationError("barrier record is not valid JSON") from exc
        after_fd = os.fstat(fd)
        try:
            after_path = os.stat(absolute, follow_symlinks=False)
        except OSError as exc:
            raise StartupBarrierValidationError(
                "barrier record path disappeared during read"
            ) from exc
        _validate_record_file_stat(after_fd)
        _validate_record_file_stat(after_path)
        if _record_file_binding(before_fd) != _record_file_binding(after_fd):
            raise StartupBarrierValidationError("barrier record changed during fd read")
        if _record_file_binding(before_path) != _record_file_binding(after_path):
            raise StartupBarrierValidationError("barrier record path changed during read")
        if (after_fd.st_dev, after_fd.st_ino) != (
            after_path.st_dev,
            after_path.st_ino,
        ):
            raise StartupBarrierValidationError(
                "barrier record fd/path inode race after read"
            )
    finally:
        os.close(fd)
    if not isinstance(raw, dict):
        raise StartupBarrierValidationError("barrier record must be a JSON object")
    record = StartupBarrierRecord.from_mapping(raw)
    if record.record_path != absolute:
        raise StartupBarrierValidationError("barrier record path binding drifted")
    return record


def _validate_record_semantics(record: StartupBarrierRecord) -> None:
    if (
        record.schema != RECORD_SCHEMA
        or record.kind != RECORD_KIND
        or record.state != RECORD_STATE
    ):
        raise StartupBarrierValidationError("barrier record authority fields are invalid")
    if not os.path.isabs(record.lock_path) or not os.path.isabs(record.record_path):
        raise StartupBarrierValidationError("barrier paths must be absolute")
    if not os.path.isabs(record.python_executable):
        raise StartupBarrierValidationError("python executable must be absolute")
    if record.lock_mode != 0o600 or record.lock_uid != os.getuid() or record.lock_nlink != 1:
        raise StartupBarrierValidationError("recorded lock security fields are invalid")
    if record.release_token_bytes != RELEASE_TOKEN_BYTES:
        raise StartupBarrierValidationError("release-token length contract drifted")
    if len(record.release_token_sha256) != 64:
        raise StartupBarrierValidationError("release-token digest is malformed")
    if record.target_argv_sha256 != argv_sha256(record.target_argv):
        raise StartupBarrierValidationError("target argv digest mismatch")
    expected_launcher = _launcher_argv(
        python_executable=record.python_executable,
        record_path=record.record_path,
        release_read_fd=record.release_read_fd,
        lock_fd=record.lock_fd,
        target_argv=record.target_argv,
    )
    if record.launcher_argv != expected_launcher:
        raise StartupBarrierValidationError("launcher argv does not match record fields")
    if record.launcher_argv_sha256 != argv_sha256(record.launcher_argv):
        raise StartupBarrierValidationError("launcher argv digest mismatch")


def validate_startup_barrier_record(
    record_path: str | os.PathLike[str],
    *,
    expected_target_argv: Sequence[str] | None = None,
    expected_launcher_argv: Sequence[str] | None = None,
    validate_lock_path: bool = True,
) -> StartupBarrierRecord:
    """Load and fail-closed validate a durable startup-barrier record."""

    record = _load_record(os.fspath(record_path))
    _validate_record_semantics(record)
    if expected_target_argv is not None:
        expected = _argv_tuple(expected_target_argv, field="expected_target_argv")
        if expected != record.target_argv:
            raise StartupBarrierValidationError("expected target argv differs from record")
    if expected_launcher_argv is not None:
        expected = _argv_tuple(expected_launcher_argv, field="expected_launcher_argv")
        if expected != record.launcher_argv:
            raise StartupBarrierValidationError("expected launcher argv differs from record")
    if validate_lock_path:
        fd = _open_nofollow(record.lock_path, os.O_RDWR)
        try:
            _validate_lock_fd_and_path(fd, record)
        finally:
            os.close(fd)
    return record


def wait_for_unreleased_barrier_lock_quiescent(
    record_path: str | os.PathLike[str],
    *,
    timeout_seconds: float,
    poll_seconds: float = 0.05,
    expected_target_argv: Sequence[str] | None = None,
    expected_launcher_argv: Sequence[str] | None = None,
) -> StartupBarrierRecord:
    """Wait until an old unreleased wrapper no longer holds its lock.

    The lock path and inode are revalidated both before and after acquiring a
    new exclusive open description.  A replaced inode or symlink is a hard
    error, never evidence of quiescence.
    """

    if timeout_seconds < 0 or poll_seconds <= 0:
        raise ValueError("timeout must be non-negative and poll interval positive")
    record = validate_startup_barrier_record(
        record_path,
        expected_target_argv=expected_target_argv,
        expected_launcher_argv=expected_launcher_argv,
        validate_lock_path=True,
    )
    deadline = time.monotonic() + timeout_seconds
    while True:
        fd = _open_nofollow(record.lock_path, os.O_RDWR)
        try:
            _validate_lock_fd_and_path(fd, record)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                locked = False
            else:
                locked = True
            if locked:
                _validate_lock_fd_and_path(fd, record)
                fcntl.flock(fd, fcntl.LOCK_UN)
                return validate_startup_barrier_record(
                    record_path,
                    expected_target_argv=expected_target_argv,
                    expected_launcher_argv=expected_launcher_argv,
                    validate_lock_path=True,
                )
        finally:
            os.close(fd)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise StartupBarrierBusy("unreleased startup barrier is still locked")
        time.sleep(min(poll_seconds, remaining))


def validate_reopenable_unreleased_barrier(
    record_path: str | os.PathLike[str],
    *,
    expected_target_argv: Sequence[str],
    timeout_seconds: float = 0.0,
    poll_seconds: float = 0.05,
) -> StartupBarrierRecord:
    """Validate provenance and quiescence before replacing an old arm record."""

    record = validate_startup_barrier_record(
        record_path,
        expected_target_argv=expected_target_argv,
        validate_lock_path=True,
    )
    return wait_for_unreleased_barrier_lock_quiescent(
        record_path,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
        expected_target_argv=expected_target_argv,
        expected_launcher_argv=record.launcher_argv,
    )


class ArmedExecStartupBarrier:
    """Parent-side capability returned by :func:`arm_exec_startup_barrier`."""

    def __init__(
        self,
        *,
        record: StartupBarrierRecord,
        release_token: bytes,
        release_read_fd: int,
        release_write_fd: int,
        lock_fd: int,
    ) -> None:
        self.record = record
        self._release_token = release_token
        self._release_read_fd = release_read_fd
        self._release_write_fd = release_write_fd
        self._lock_fd = lock_fd
        self._spawned = False
        self._terminal_action: str | None = None

    @property
    def launcher_argv(self) -> tuple[str, ...]:
        return self.record.launcher_argv

    @property
    def pass_fds(self) -> tuple[int, int]:
        return (self.record.release_read_fd, self.record.lock_fd)

    def _validate_live_parent_capability(self) -> None:
        if self._terminal_action is not None:
            raise StartupBarrierError(
                f"startup barrier already completed via {self._terminal_action}"
            )
        record = validate_startup_barrier_record(
            self.record.record_path,
            expected_target_argv=self.record.target_argv,
            expected_launcher_argv=self.record.launcher_argv,
            validate_lock_path=False,
        )
        if record != self.record:
            raise StartupBarrierValidationError("durable barrier record changed after arm")
        _validate_lock_fd_and_path(self._lock_fd, self.record)
        _assert_exclusive_lock(self._lock_fd)
        try:
            os.fstat(self._release_write_fd)
        except OSError as exc:
            raise StartupBarrierValidationError("release write fd is not open") from exc

    def mark_spawned(self) -> None:
        """Close the parent's read duplicate after a successful external spawn."""

        if self._spawned:
            raise StartupBarrierError("startup wrapper was already marked spawned")
        self._validate_live_parent_capability()
        os.close(self._release_read_fd)
        self._release_read_fd = -1
        self._spawned = True

    def launch(self, **popen_kwargs: Any) -> subprocess.Popen[Any]:
        """Launch the wrapper with exactly the two recorded inherited fds."""

        forbidden = {
            "args",
            "pass_fds",
            "close_fds",
            "executable",
            "shell",
            "preexec_fn",
        }.intersection(popen_kwargs)
        if forbidden:
            raise TypeError(f"launch owns subprocess arguments: {sorted(forbidden)}")
        self._validate_live_parent_capability()
        try:
            process = subprocess.Popen(
                self.launcher_argv,
                pass_fds=self.pass_fds,
                close_fds=True,
                **popen_kwargs,
            )
        except BaseException:
            self.abort()
            raise
        try:
            self.mark_spawned()
        except BaseException:
            self.abort()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
            raise
        return process

    def _close_parent_capabilities(self) -> None:
        for attribute in ("_release_read_fd", "_release_write_fd", "_lock_fd"):
            fd = getattr(self, attribute)
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, attribute, -1)

    def release(self) -> None:
        """Send the exact release token and relinquish parent capabilities."""

        if not self._spawned:
            raise StartupBarrierError("cannot release before the wrapper is spawned")
        try:
            self._validate_live_parent_capability()
            offset = 0
            while offset < len(self._release_token):
                written = os.write(self._release_write_fd, self._release_token[offset:])
                if written <= 0:
                    raise StartupBarrierError("short write to startup release pipe")
                offset += written
        except BaseException:
            self._terminal_action = "ABORTED_ON_RELEASE_ERROR"
            self._close_parent_capabilities()
            raise
        else:
            # Closing the write end supplies the EOF that is part of the exact
            # token framing.  The wrapper closes its lock descriptor only after
            # it observes this EOF and revalidates every durable binding.
            os.close(self._release_write_fd)
            self._release_write_fd = -1
            os.close(self._lock_fd)
            self._lock_fd = -1
            self._terminal_action = "RELEASED"

    def abort(self) -> None:
        """Close without a token; the wrapper must fail closed on EOF."""

        if self._terminal_action is not None:
            return
        self._terminal_action = "ABORTED"
        self._close_parent_capabilities()

    def __enter__(self) -> "ArmedExecStartupBarrier":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._terminal_action is None:
            self.abort()


def _make_pipe() -> tuple[int, int]:
    flags = getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "pipe2"):
        return os.pipe2(flags)
    read_fd, write_fd = os.pipe()
    os.set_inheritable(read_fd, False)
    os.set_inheritable(write_fd, False)
    return read_fd, write_fd


def arm_exec_startup_barrier(
    *,
    lock_path: str | os.PathLike[str],
    record_path: str | os.PathLike[str],
    target_argv: Sequence[str],
    python_executable: str | os.PathLike[str] = sys.executable,
    record_policy: str = "fresh",
    rearm_timeout_seconds: float = 0.0,
    expected_unreleased_record: StartupBarrierRecord | None = None,
) -> ArmedExecStartupBarrier:
    """Create and lock a physical barrier, pipe, and durable record.

    ``record_policy="fresh"`` requires the record path not to exist and uses
    no-replace publication.  ``record_policy="resume"`` is only for a caller
    whose own durable state proves the prior barrier was *unreleased*.  It
    validates the old target/lock binding, waits for lock quiescence, acquires
    that same inode, and re-reads the exact old record while holding the lock
    before replacing it.  Lock freedom alone does not prove an already
    released target process is gone.  Consequently resume also requires the
    exact ``expected_unreleased_record`` projected from that caller-owned
    durable PRE_RELEASE state; a bare ``record_policy="resume"`` is rejected.
    """

    _require_posix_primitives()
    absolute_lock = os.path.abspath(os.fspath(lock_path))
    absolute_record = os.path.abspath(os.fspath(record_path))
    absolute_python = os.path.abspath(os.fspath(python_executable))
    target = _argv_tuple(target_argv, field="target_argv")
    if record_policy not in {"fresh", "resume"}:
        raise ValueError("record_policy must be 'fresh' or 'resume'")
    if record_policy == "fresh" and expected_unreleased_record is not None:
        raise StartupBarrierValidationError(
            "fresh barrier cannot consume prior-unreleased authority"
        )
    if record_policy == "resume" and not isinstance(
        expected_unreleased_record, StartupBarrierRecord
    ):
        raise StartupBarrierValidationError(
            "resume requires typed expected_unreleased_record authority"
        )
    if rearm_timeout_seconds < 0:
        raise ValueError("rearm_timeout_seconds must be non-negative")
    if absolute_lock == absolute_record:
        raise StartupBarrierValidationError("lock and record paths must differ")
    if not os.path.isfile(absolute_python):
        raise StartupBarrierValidationError("python executable is not a regular file")
    if not os.path.isdir(os.path.dirname(absolute_lock)):
        raise StartupBarrierValidationError("barrier lock parent must already exist")
    previous_record: StartupBarrierRecord | None = None
    if record_policy == "fresh":
        try:
            os.stat(absolute_record, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise StartupBarrierValidationError(
                "fresh barrier refuses any existing durable record"
            )
    lock_fd = _open_nofollow(absolute_lock, os.O_RDWR | os.O_CREAT, 0o600)
    read_fd = -1
    write_fd = -1
    try:
        lock_stat = os.fstat(lock_fd)
        _validate_physical_lock_stat(lock_stat)
        path_stat = os.stat(absolute_lock, follow_symlinks=False)
        _validate_physical_lock_stat(path_stat)
        if (lock_stat.st_dev, lock_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino):
            raise StartupBarrierValidationError("lock fd/path inode race during arm")
        _acquire_exclusive_lock(
            lock_fd,
            timeout_seconds=(rearm_timeout_seconds if record_policy == "resume" else 0.0),
        )
        _reconcile_interrupted_record_publication(absolute_record)
        if record_policy == "fresh":
            try:
                os.stat(absolute_record, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise StartupBarrierValidationError(
                    "fresh barrier record appeared before lock acquisition"
                )
        else:
            try:
                previous_record = validate_startup_barrier_record(
                    absolute_record,
                    expected_target_argv=target,
                    validate_lock_path=False,
                )
            except FileNotFoundError as exc:
                raise StartupBarrierValidationError(
                    "resume requires an existing durable record"
                ) from exc
            if previous_record.lock_path != absolute_lock:
                raise StartupBarrierValidationError(
                    "resume lock path differs from previous record"
                )
            if previous_record.python_executable != absolute_python:
                raise StartupBarrierValidationError(
                    "resume Python executable differs from previous record"
                )
            if previous_record != expected_unreleased_record:
                raise StartupBarrierValidationError(
                    "resume record differs from caller PRE_RELEASE authority"
                )
            _validate_lock_fd_and_path(lock_fd, previous_record)
            locked_record = validate_startup_barrier_record(
                absolute_record,
                expected_target_argv=target,
                expected_launcher_argv=previous_record.launcher_argv,
                validate_lock_path=False,
            )
            if locked_record != previous_record:
                raise StartupBarrierValidationError(
                    "resume record changed before locked replacement"
                )
        os.fsync(lock_fd)
        _fsync_directory(os.path.dirname(absolute_lock))
        read_fd, write_fd = _make_pipe()
        token = secrets.token_bytes(RELEASE_TOKEN_BYTES)
        launcher = _launcher_argv(
            python_executable=absolute_python,
            record_path=absolute_record,
            release_read_fd=read_fd,
            lock_fd=lock_fd,
            target_argv=target,
        )
        record = StartupBarrierRecord(
            schema=RECORD_SCHEMA,
            kind=RECORD_KIND,
            state=RECORD_STATE,
            lock_path=absolute_lock,
            lock_dev=lock_stat.st_dev,
            lock_inode=lock_stat.st_ino,
            lock_mode=stat.S_IMODE(lock_stat.st_mode),
            lock_uid=lock_stat.st_uid,
            lock_nlink=lock_stat.st_nlink,
            record_path=absolute_record,
            python_executable=absolute_python,
            release_read_fd=read_fd,
            lock_fd=lock_fd,
            release_token_bytes=len(token),
            release_token_sha256=hashlib.sha256(token).hexdigest(),
            target_argv=target,
            target_argv_sha256=argv_sha256(target),
            launcher_argv=launcher,
            launcher_argv_sha256=argv_sha256(launcher),
        )
        _validate_record_semantics(record)
        _write_record_durable(
            absolute_record,
            record,
            replace_expected=previous_record,
        )
        return ArmedExecStartupBarrier(
            record=record,
            release_token=token,
            release_read_fd=read_fd,
            release_write_fd=write_fd,
            lock_fd=lock_fd,
        )
    except BaseException:
        for fd in (read_fd, write_fd, lock_fd):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        raise


def _read_exact_release_token(fd: int, record: StartupBarrierRecord) -> bytes:
    received = bytearray()
    maximum = record.release_token_bytes + 1
    while True:
        try:
            chunk = os.read(fd, maximum - len(received))
        except OSError as exc:
            raise StartupBarrierValidationError("release pipe read failed") from exc
        if not chunk:
            break
        received.extend(chunk)
        if len(received) >= maximum:
            raise StartupBarrierValidationError("release pipe contained extra bytes")
    token = bytes(received)
    if len(token) != record.release_token_bytes:
        raise StartupBarrierValidationError("release pipe reached EOF without exact token")
    if not secrets.compare_digest(
        hashlib.sha256(token).hexdigest(), record.release_token_sha256
    ):
        raise StartupBarrierValidationError("release token digest mismatch")
    return token


def _wrapper_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True)
    parser.add_argument("--release-read-fd", required=True, type=int)
    parser.add_argument("--lock-fd", required=True, type=int)
    parser.add_argument("target", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    target = list(args.target)
    if target and target[0] == "--":
        target = target[1:]
    read_fd = args.release_read_fd
    lock_fd = args.lock_fd
    try:
        record = validate_startup_barrier_record(
            args.record,
            expected_target_argv=target,
            validate_lock_path=False,
        )
        if read_fd != record.release_read_fd or lock_fd != record.lock_fd:
            raise StartupBarrierValidationError("launcher fd numbers differ from record")
        if os.path.abspath(sys.executable) != record.python_executable:
            raise StartupBarrierValidationError("launcher Python executable drifted")
        actual_launcher = _launcher_argv(
            python_executable=os.path.abspath(sys.executable),
            record_path=os.path.abspath(args.record),
            release_read_fd=read_fd,
            lock_fd=lock_fd,
            target_argv=target,
        )
        if actual_launcher != record.launcher_argv:
            raise StartupBarrierValidationError("runtime launcher argv drifted")
        _validate_lock_fd_and_path(lock_fd, record)
        _assert_exclusive_lock(lock_fd)
        _read_exact_release_token(read_fd, record)
        # Re-open the record and repeat every binding after release.  The
        # target never runs if either durable file changed while blocked.
        released_record = validate_startup_barrier_record(
            args.record,
            expected_target_argv=target,
            expected_launcher_argv=actual_launcher,
            validate_lock_path=False,
        )
        if released_record != record:
            raise StartupBarrierValidationError("barrier record changed while armed")
        _validate_lock_fd_and_path(lock_fd, record)
        _assert_exclusive_lock(lock_fd)
        os.close(read_fd)
        read_fd = -1
        os.close(lock_fd)
        lock_fd = -1
        os.execvpe(target[0], target, os.environ.copy())
    except BaseException as exc:
        # This wrapper is intentionally terse: its only safe failure action is
        # to close capabilities and exit without executing the target.
        try:
            sys.stderr.write(f"startup barrier refused exec: {exc}\n")
            sys.stderr.flush()
        except BaseException:
            pass
        return 73
    finally:
        for fd in (read_fd, lock_fd):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
    return 73


if __name__ == "__main__":
    raise SystemExit(_wrapper_main())
