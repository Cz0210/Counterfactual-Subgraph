"""Fail-closed launcher/worker identity evidence for managed execution v2.

Process identity is runtime audit evidence.  It is deliberately separate from
scientific artifact adoption: an exited or legitimately re-parented process
does not invalidate a sealed artifact that an independent verifier can prove.
This module never sends signals and exposes no termination primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any, Mapping, MutableMapping, Sequence
import uuid


PROCESS_IDENTITY_SCHEMA = "managed_process_identity_v2"
PROCESS_LINEAGE_SCHEMA = "managed_process_lineage_v2"
QUARANTINE_SCHEMA = "managed_quarantine_v2"
AUTO_TERMINATE_UNCONTROLLED_CHILDREN = False


class ProcessIdentityV2Error(RuntimeError):
    """Raised when exact process-generation evidence cannot be obtained."""


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def stable_json_sha256(value: Any) -> str:
    data = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def require_uuid4(value: str, *, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProcessIdentityV2Error(f"{label} is not a UUID") from exc
    if parsed.version != 4 or parsed.variant != uuid.RFC_4122:
        raise ProcessIdentityV2Error(f"{label} must be an RFC-4122 UUIDv4")
    if str(parsed) != value.lower():
        raise ProcessIdentityV2Error(f"{label} must use canonical UUID text")
    return str(parsed)


def require_auto_termination_disabled(
    environment: Mapping[str, str] | None = None,
) -> None:
    """Reject any attempt to grant signal authority to the controller."""

    source = os.environ if environment is None else environment
    raw = source.get("AUTO_TERMINATE_UNCONTROLLED_CHILDREN", "0")
    if raw != "0":
        raise ProcessIdentityV2Error(
            "AUTO_TERMINATE_UNCONTROLLED_CHILDREN must remain exactly 0"
        )


def _read_all(descriptor: int, *, maximum_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        block = os.read(descriptor, min(64 * 1024, maximum_bytes - total + 1))
        if not block:
            return b"".join(chunks)
        chunks.append(block)
        total += len(block)
        if total > maximum_bytes:
            raise ProcessIdentityV2Error("process evidence exceeds its read bound")


def _read_proc_file(proc_descriptor: int, name: str, *, maximum_bytes: int) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=proc_descriptor,
    )
    try:
        return _read_all(descriptor, maximum_bytes=maximum_bytes)
    finally:
        os.close(descriptor)


def _parse_linux_stat(raw: bytes) -> tuple[int, int]:
    try:
        text = raw.decode("utf-8")
        right = text.rfind(")")
        if right < 0:
            raise ValueError("missing comm terminator")
        fields = text[right + 2 :].split()
        # fields[0] is state (field 3), fields[1] is PPID (field 4), and
        # fields[19] is starttime (field 22).
        ppid = int(fields[1])
        start_ticks = int(fields[19])
    except (IndexError, UnicodeDecodeError, ValueError) as exc:
        raise ProcessIdentityV2Error("malformed Linux /proc stat") from exc
    if ppid < 0 or start_ticks <= 0:
        raise ProcessIdentityV2Error("invalid Linux process generation")
    return ppid, start_ticks


def linux_boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
        return str(uuid.UUID(value))
    except (OSError, ValueError) as exc:
        raise ProcessIdentityV2Error("Linux boot_id is unavailable") from exc


_FALLBACK_LOCK = threading.Lock()
_FALLBACK_START: MutableMapping[int, int] = {}


@dataclass(frozen=True, slots=True)
class ProcessSnapshotV2:
    pid: int
    ppid: int
    pid_start_ticks: int
    boot_id: str
    executable_realpath: str
    command: tuple[str, ...]
    command_hash: str
    cwd_realpath: str
    cgroup_path: str | None

    def __post_init__(self) -> None:
        if isinstance(self.pid, bool) or self.pid <= 0:
            raise ProcessIdentityV2Error("process PID is invalid")
        if isinstance(self.ppid, bool) or self.ppid < 0:
            raise ProcessIdentityV2Error("process PPID is invalid")
        if isinstance(self.pid_start_ticks, bool) or self.pid_start_ticks <= 0:
            raise ProcessIdentityV2Error("process start ticks are invalid")
        if not self.boot_id:
            raise ProcessIdentityV2Error("process boot_id is absent")
        if not Path(self.executable_realpath).is_absolute():
            raise ProcessIdentityV2Error("process executable is not absolute")
        if not Path(self.cwd_realpath).is_absolute():
            raise ProcessIdentityV2Error("process cwd is not absolute")
        if not self.command or stable_json_sha256(list(self.command)) != self.command_hash:
            raise ProcessIdentityV2Error("process command hash is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROCESS_IDENTITY_SCHEMA,
            "pid": self.pid,
            "ppid": self.ppid,
            "pid_start_ticks": self.pid_start_ticks,
            "boot_id": self.boot_id,
            "executable_realpath": self.executable_realpath,
            "command": list(self.command),
            "command_hash": self.command_hash,
            "cwd_realpath": self.cwd_realpath,
            "cgroup_path": self.cgroup_path,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ProcessSnapshotV2":
        if raw.get("schema_version") != PROCESS_IDENTITY_SCHEMA:
            raise ProcessIdentityV2Error("process identity schema is invalid")
        command = raw.get("command")
        if not isinstance(command, list) or not all(
            isinstance(item, str) for item in command
        ):
            raise ProcessIdentityV2Error("process command is malformed")
        cgroup = raw.get("cgroup_path")
        if cgroup is not None and not isinstance(cgroup, str):
            raise ProcessIdentityV2Error("process cgroup is malformed")
        return cls(
            pid=raw.get("pid"),
            ppid=raw.get("ppid"),
            pid_start_ticks=raw.get("pid_start_ticks"),
            boot_id=raw.get("boot_id"),
            executable_realpath=raw.get("executable_realpath"),
            command=tuple(command),
            command_hash=raw.get("command_hash"),
            cwd_realpath=raw.get("cwd_realpath"),
            cgroup_path=cgroup,
        )

    def same_generation(self, other: "ProcessSnapshotV2") -> bool:
        return (
            self.pid == other.pid
            and self.pid_start_ticks == other.pid_start_ticks
            and self.boot_id == other.boot_id
        )

    def same_runtime_identity(self, other: "ProcessSnapshotV2") -> bool:
        return (
            self.same_generation(other)
            and self.executable_realpath == other.executable_realpath
            and self.command_hash == other.command_hash
            and self.cwd_realpath == other.cwd_realpath
            and self.cgroup_path == other.cgroup_path
        )


def capture_process_snapshot(pid: int) -> ProcessSnapshotV2:
    """Capture one exact process snapshot; Linux is the production route."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ProcessIdentityV2Error("PID is invalid")
    if sys.platform.startswith("linux"):
        proc_path = Path("/proc") / str(pid)
        proc_descriptor = os.open(
            proc_path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            ppid, start_ticks = _parse_linux_stat(
                _read_proc_file(proc_descriptor, "stat", maximum_bytes=64 * 1024)
            )
            raw_command = _read_proc_file(
                proc_descriptor, "cmdline", maximum_bytes=1024 * 1024
            )
            command = tuple(
                item.decode("utf-8", errors="surrogateescape")
                for item in raw_command.rstrip(b"\0").split(b"\0")
                if item
            )
            if not command:
                raise ProcessIdentityV2Error("process command is empty")
            raw_cgroup = _read_proc_file(
                proc_descriptor, "cgroup", maximum_bytes=1024 * 1024
            )
            cgroup_lines = sorted(
                line
                for line in raw_cgroup.decode("utf-8").splitlines()
                if line
            )
            cgroup = "\n".join(cgroup_lines) if cgroup_lines else None
            executable = os.path.realpath(os.readlink(f"/proc/{pid}/exe"))
            cwd = os.path.realpath(os.readlink(f"/proc/{pid}/cwd"))
            return ProcessSnapshotV2(
                pid=pid,
                ppid=ppid,
                pid_start_ticks=start_ticks,
                boot_id=linux_boot_id(),
                executable_realpath=executable,
                command=command,
                command_hash=stable_json_sha256(list(command)),
                cwd_realpath=cwd,
                cgroup_path=cgroup,
            )
        except (OSError, UnicodeDecodeError) as exc:
            raise ProcessIdentityV2Error(
                f"cannot capture Linux process generation {pid}"
            ) from exc
        finally:
            os.close(proc_descriptor)

    # Non-Linux exists only for local protocol tests.  Cache a monotonic token
    # for the current process so repeated captures remain a stable generation.
    if pid != os.getpid():
        raise ProcessIdentityV2Error(
            "non-Linux capture supports only the current review process"
        )
    with _FALLBACK_LOCK:
        start_ticks = _FALLBACK_START.setdefault(pid, time.monotonic_ns())
    executable = str(Path(sys.executable).resolve(strict=True))
    cwd = str(Path.cwd().resolve(strict=True))
    command = tuple(sys.argv) or (executable,)
    return ProcessSnapshotV2(
        pid=pid,
        ppid=os.getppid(),
        pid_start_ticks=start_ticks,
        boot_id=f"non-linux-review-{socket.gethostname()}",
        executable_realpath=executable,
        command=command,
        command_hash=stable_json_sha256(list(command)),
        cwd_realpath=cwd,
        cgroup_path=None,
    )


@dataclass(frozen=True, slots=True)
class ManagedProcessLineageV2:
    controller_id: str
    attempt_id: str
    launcher: ProcessSnapshotV2
    worker: ProcessSnapshotV2
    relationship: str
    registered_at: str

    def __post_init__(self) -> None:
        require_uuid4(self.attempt_id, label="attempt_id")
        if not self.controller_id:
            raise ProcessIdentityV2Error("controller_id is absent")
        if self.relationship not in {
            "LAUNCHER_EXEC_WORKER",
            "SINGLE_MANAGED_CHILD",
            "LEGITIMATE_REPARENTING",
        }:
            raise ProcessIdentityV2Error("launcher/worker relationship is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROCESS_LINEAGE_SCHEMA,
            "controller_id": self.controller_id,
            "attempt_id": self.attempt_id,
            "launcher_pid": self.launcher.pid,
            "launcher_pid_start_ticks": self.launcher.pid_start_ticks,
            "worker_pid": self.worker.pid,
            "worker_pid_start_ticks": self.worker.pid_start_ticks,
            "boot_id": self.worker.boot_id,
            "executable_realpath": self.worker.executable_realpath,
            "command_hash": self.worker.command_hash,
            "cwd_realpath": self.worker.cwd_realpath,
            "cgroup_path": self.worker.cgroup_path,
            "relationship": self.relationship,
            "registered_at": self.registered_at,
            "launcher": self.launcher.to_dict(),
            "worker": self.worker.to_dict(),
        }


def register_process_lineage(
    *,
    controller_id: str,
    attempt_id: str,
    launcher: ProcessSnapshotV2,
    worker: ProcessSnapshotV2,
    registered_at: str,
    launcher_exit_observed: bool = False,
) -> ManagedProcessLineageV2:
    """Register exec, one-child, or already-observed legal re-parenting."""

    require_auto_termination_disabled()
    if launcher.boot_id != worker.boot_id:
        raise ProcessIdentityV2Error("launcher and worker boot IDs differ")
    if launcher.same_generation(worker):
        relationship = "LAUNCHER_EXEC_WORKER"
    elif worker.ppid == launcher.pid:
        relationship = "SINGLE_MANAGED_CHILD"
    elif launcher_exit_observed:
        relationship = "LEGITIMATE_REPARENTING"
    else:
        raise ProcessIdentityV2Error("worker is outside the allowed lineage")
    return ManagedProcessLineageV2(
        controller_id=controller_id,
        attempt_id=attempt_id,
        launcher=launcher,
        worker=worker,
        relationship=relationship,
        registered_at=registered_at,
    )


@dataclass(frozen=True, slots=True)
class QuarantineEvidenceV2:
    controller_id: str
    attempt_id: str
    quarantine_reason: str
    last_known_pid: int | None
    last_known_start_ticks: int | None
    last_heartbeat: str | None
    output_root: str
    observed_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": QUARANTINE_SCHEMA,
            "state": "QUARANTINED",
            "controller_id": self.controller_id,
            "attempt_id": self.attempt_id,
            "science_adopted": False,
            "downstream_released": False,
            "quarantine_reason": self.quarantine_reason,
            "last_known_pid": self.last_known_pid,
            "last_known_start_ticks": self.last_known_start_ticks,
            "last_heartbeat": self.last_heartbeat,
            "output_root": self.output_root,
            "manual_review_required": True,
            "auto_terminate_uncontrolled_children": False,
            "observed_at": self.observed_at,
        }


def quarantine_runtime_anomaly(
    *,
    controller_id: str,
    attempt_id: str,
    reason: str,
    output_root: str | Path,
    observed_at: str,
    last_known_pid: int | None = None,
    last_known_start_ticks: int | None = None,
    last_heartbeat: str | None = None,
) -> dict[str, Any]:
    """Quarantine a drift/orphan/heartbeat/publish anomaly without signalling."""

    require_auto_termination_disabled()
    require_uuid4(attempt_id, label="attempt_id")
    allowed = {
        "WORKER_PROCESS_IDENTITY_DRIFT",
        "UNEXPECTED_CHILD",
        "ORPHAN_PROCESS",
        "HEARTBEAT_LOSS",
        "TERMINAL_PUBLISH_MISMATCH",
        "UNEXPECTED_REPARENTING",
    }
    if reason not in allowed:
        raise ProcessIdentityV2Error("quarantine reason is not a closed v2 kind")
    return QuarantineEvidenceV2(
        controller_id=controller_id,
        attempt_id=attempt_id,
        quarantine_reason=reason,
        last_known_pid=last_known_pid,
        last_known_start_ticks=last_known_start_ticks,
        last_heartbeat=last_heartbeat,
        output_root=str(Path(output_root).absolute()),
        observed_at=observed_at,
    ).to_dict()


def audit_process_lineage(
    lineage: ManagedProcessLineageV2,
    *,
    observed_worker: ProcessSnapshotV2 | None,
    launcher_alive: bool,
    last_heartbeat: str | None,
    output_root: str | Path,
    observed_at: str,
) -> dict[str, Any]:
    """Return RUNNING/EXITED audit or fail closed to QUARANTINED.

    PPID alone is not an identity: a matching worker generation may be
    legitimately re-parented after its launcher exits.  Every other component
    of the registered runtime identity remains exact.
    """

    require_auto_termination_disabled()
    if observed_worker is None:
        return {
            "schema_version": "managed_process_audit_v2",
            "state": "EXITED",
            "controller_id": lineage.controller_id,
            "attempt_id": lineage.attempt_id,
            "science_adopted": False,
            "downstream_released": False,
        }
    expected = lineage.worker
    exact = expected.same_runtime_identity(observed_worker)
    ppid_allowed = observed_worker.ppid == expected.ppid or not launcher_alive
    if exact and ppid_allowed:
        return {
            "schema_version": "managed_process_audit_v2",
            "state": (
                "RUNNING"
                if observed_worker.ppid == expected.ppid
                else "RUNNING_LEGITIMATELY_REPARENTED"
            ),
            "controller_id": lineage.controller_id,
            "attempt_id": lineage.attempt_id,
            "launcher_alive": launcher_alive,
            "worker": observed_worker.to_dict(),
            "science_adopted": False,
            "downstream_released": False,
        }
    reason = (
        "UNEXPECTED_REPARENTING" if exact else "WORKER_PROCESS_IDENTITY_DRIFT"
    )
    return quarantine_runtime_anomaly(
        controller_id=lineage.controller_id,
        attempt_id=lineage.attempt_id,
        reason=reason,
        last_known_pid=observed_worker.pid,
        last_known_start_ticks=observed_worker.pid_start_ticks,
        last_heartbeat=last_heartbeat,
        output_root=str(Path(output_root).absolute()),
        observed_at=observed_at,
    )


__all__ = [
    "AUTO_TERMINATE_UNCONTROLLED_CHILDREN",
    "ManagedProcessLineageV2",
    "PROCESS_IDENTITY_SCHEMA",
    "PROCESS_LINEAGE_SCHEMA",
    "ProcessIdentityV2Error",
    "ProcessSnapshotV2",
    "QuarantineEvidenceV2",
    "audit_process_lineage",
    "canonical_json_bytes",
    "capture_process_snapshot",
    "linux_boot_id",
    "register_process_lineage",
    "quarantine_runtime_anomaly",
    "require_auto_termination_disabled",
    "require_uuid4",
    "stable_json_sha256",
]
