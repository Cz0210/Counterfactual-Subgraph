"""Managed Taste release-v3 authority under compatibility ``main_v2`` names.

This is not the complete ``main_completion_v4`` scheduler. The controller owns an immutable process receipt,
publishes append-only heartbeat generations, and acknowledges immutable GPU
lease requests.  Consumers retain all authority files with ``O_NOFOLLOW`` and
revalidate the live controller process generation before a scientific release.

This module contains no process-termination primitive.  Loss or drift of
authority only makes a consumer fail closed or makes the controller publish a
``QUARANTINED`` heartbeat generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

from src.utils.process_identity_v2 import (
    ProcessIdentityV2Error,
    ProcessSnapshotV2,
    canonical_json_bytes,
    capture_process_snapshot,
    require_auto_termination_disabled,
    require_uuid4,
)
from src.utils.autodl_runtime import (
    AutoDLRuntimeError,
    GPUFileLock,
    GPULockError,
    RuntimeLayout,
    build_runtime_layout,
)


CONTROLLER_RECEIPT_SCHEMA = "tastemolnet_main_v2_controller_receipt_v1"
LAUNCHER_RECEIPT_SCHEMA = "tastemolnet_main_v2_launcher_receipt_v1"
LAUNCHER_READY_SCHEMA = "tastemolnet_main_v2_launcher_ready_v1"
HEARTBEAT_SCHEMA = "tastemolnet_main_v2_controller_heartbeat_v1"
GPU_LEASE_SCHEMA = "tastemolnet_main_v2_gpu_lease_v1"
GPU_LEASE_ACTIVATION_SCHEMA = "tastemolnet_main_v2_gpu_lease_activation_v1"
GPU_LEASE_RENEWAL_SCHEMA = "tastemolnet_main_v2_gpu_lease_renewal_v1"
GPU_LEASE_RELEASE_SCHEMA = "tastemolnet_main_v2_gpu_lease_release_v1"
GPU_LOCK_SCHEMA = "tastemolnet_main_v2_gpu_lock_v1"
STATUS_SCHEMA = "tastemolnet_main_v2_status_v1"
MANAGED_TASTE_RELEASE_VERSION = 3
MANAGED_TASTE_RELEASE_MARKER = "[MANAGED_TASTE_RELEASE_V3_PASS]"
CONTROLLER_RECEIPT_NAME = "controller_receipt.json"
LAUNCHER_RECEIPT_NAME = "launcher_receipt.json"
LAUNCHER_READY_NAME = "launcher_ready.json"
HEARTBEAT_DIRECTORY = "heartbeats"
GPU_LEASE_DIRECTORY = "gpu_leases"
GPU_LEASE_ACTIVATION_DIRECTORY = "gpu_lease_activations"
GPU_LEASE_RENEWAL_DIRECTORY = "gpu_lease_renewals"
GPU_LEASE_RELEASE_DIRECTORY = "gpu_lease_releases"
GPU_LOCK_DIRECTORY = "gpu_locks"
PUBLICATION_STAGING_DIRECTORY = ".publication_staging"
HEARTBEAT_INTERVAL_SECONDS = 10
DEFAULT_MAX_HEARTBEAT_AGE_SECONDS = 35
ALLOWED_GPU_INDICES = frozenset({1, 2})
PROTECTED_GPU_INDICES = frozenset({0, 3})
MAX_ACTIVE_TASKS = 2
MINIMUM_PERSISTENT_FREE_GIB = 100
SCHEDULER_POLL_SECONDS = 60
TASK_GPU_BINDINGS = {
    "T4_ORACLE_SMOKE": 1,
    "TASTE_GCF_NEUROSED": 2,
}
_GPU_LOCKS: dict[str, tuple[GPUFileLock, str]] = {}
_WORKER_GPU_LOCKS: dict[str, GPUFileLock] = {}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEARTBEAT_NAME = re.compile(
    r"^(?P<sequence>[0-9]{20})-(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12})\.json$"
)
_LEASE_NAME = re.compile(
    r"^(?P<task>[A-Z][A-Z0-9_]*)-(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12})\.json$"
)
_ACTIVATION_NAME = re.compile(
    r"^(?P<lease>[0-9a-f-]{36})-(?P<sequence>[0-9]{20})-(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12})\.json$"
)
_RENEWAL_NAME = re.compile(
    r"^(?P<lease>[0-9a-f-]{36})-(?P<sequence>[0-9]{20})-(?P<uuid>[0-9a-f]{8}-"
    r"[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})\.json$"
)
_RELEASE_NAME = re.compile(
    r"^(?P<lease>[0-9a-f-]{36})-(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-"
    r"4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})\.json$"
)
_TASK_ID = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")
_GPU_UUID = re.compile(r"^GPU-[0-9A-Fa-f-]+$")


class TasteMainV2AuthorityError(RuntimeError):
    """Controller authority is absent, stale, mutable, or inconsistent."""


def capture_policy_facts(
    *,
    persistent_storage_root: str | Path,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate launch gates and record the observed persistent-space floor."""

    source = os.environ if environment is None else environment
    required = {
        "RUN_TASTEMOLNET": "1",
        "TASTE_RESEARCH_COMPUTE_ALLOWED": "1",
        "TASTE_PAPER_RESULTS_ALLOWED": "1",
        "TASTE_DATA_REDISTRIBUTION_ALLOWED": "0",
        "PRIMARY_TASTE_SOURCE_LABEL": "1",
        "MIN_FREE_AFTER_RESERVATIONS_GB": "100",
        "SCHEDULER_POLL_SECONDS": "60",
        "AUTODL_MAX_GPUS": "4",
        "MAX_CONCURRENT_TASTE_FULL": "2",
        "RUN_GNN_ABLATION": "0",
    }
    for name, expected in required.items():
        if source.get(name) != expected:
            raise TasteMainV2AuthorityError(f"{name} must be exactly {expected}")
    for name in ("AUTODL_DATA_ROOT", "AUTODL_RUNTIME_ROOT", "AUTODL_CONTROL_ROOT"):
        if not source.get(name):
            raise TasteMainV2AuthorityError(f"{name} must be explicitly set")
    storage = _require_absolute_physical(
        Path(persistent_storage_root), label="persistent storage root"
    )
    data_root = _require_absolute_physical(
        Path(source["AUTODL_DATA_ROOT"]), label="persistent data root"
    )
    configured_runtime = Path(source["AUTODL_RUNTIME_ROOT"])
    configured_control = Path(source["AUTODL_CONTROL_ROOT"])
    _ensure_physical_child_directory(storage, "control")
    _ensure_physical_child_directory(storage, "locks")
    try:
        layout = build_runtime_layout(
            project_root=Path(__file__).resolve(strict=True).parents[2],
            data_root=data_root,
            control_root=configured_control,
        )
    except (AutoDLRuntimeError, OSError) as exc:
        raise TasteMainV2AuthorityError(
            "canonical AutoDL runtime layout is invalid"
        ) from exc
    if configured_runtime != layout.runtime_root or storage != layout.runtime_root:
        raise TasteMainV2AuthorityError(
            "AUTODL_RUNTIME_ROOT must be the canonical data-root runtime"
        )
    if configured_control != layout.control_root or configured_control != storage / "control":
        raise TasteMainV2AuthorityError(
            "AUTODL_CONTROL_ROOT must be the canonical runtime control root"
        )
    free_bytes = shutil.disk_usage(storage).free
    minimum_bytes = MINIMUM_PERSISTENT_FREE_GIB * 1024**3
    if free_bytes < minimum_bytes:
        raise TasteMainV2AuthorityError(
            "persistent storage has less than the required 100 GiB free"
        )
    return {
        "run_tastemolnet": True,
        "taste_research_compute_allowed": True,
        "taste_paper_results_allowed": True,
        "taste_data_redistribution_allowed": False,
        "primary_taste_source_label": 1,
        "minimum_free_after_reservations_gib": MINIMUM_PERSISTENT_FREE_GIB,
        "scheduler_poll_seconds": SCHEDULER_POLL_SECONDS,
        "max_physical_gpus": 4,
        "max_concurrent_taste_full": 2,
        "gnn_ablation_enabled": False,
        "persistent_data_root": str(layout.data_root),
        "persistent_storage_root": str(storage),
        "persistent_control_root": str(layout.control_root),
        "canonical_gpu_lock_root": str(layout.locks_dir),
        "persistent_free_bytes_at_launch": free_bytes,
        "persistent_free_gib_at_launch": free_bytes / 1024**3,
        "observed_at": _utc_now(),
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_nlink),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _require_absolute_physical(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise TasteMainV2AuthorityError(f"{label} must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise TasteMainV2AuthorityError(f"{label} is unavailable") from exc
    if resolved != path:
        raise TasteMainV2AuthorityError(f"{label} must be an exact physical path")
    return path


def _ensure_physical_child_directory(parent: Path, name: str) -> Path:
    """Create/reopen one fixed child without following a symlink."""

    physical_parent = _require_absolute_physical(parent, label="directory parent")
    if not name or "/" in name or name in {".", ".."}:
        raise TasteMainV2AuthorityError("directory child name is malformed")
    parent_fd = os.open(
        physical_parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    child_fd = -1
    try:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        child_fd = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        held = os.fstat(child_fd)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(held.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise TasteMainV2AuthorityError("directory child is not physical")
        os.fsync(parent_fd)
    finally:
        if child_fd >= 0:
            os.close(child_fd)
        os.close(parent_fd)
    return _require_absolute_physical(
        physical_parent / name, label="directory child"
    )


def _write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise TasteMainV2AuthorityError(f"short write for {path.name}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one UUID target without replacement."""

    if destination.exists() or destination.is_symlink():
        raise TasteMainV2AuthorityError(f"immutable target is already burned: {destination.name}")
    if os.name != "posix" or not hasattr(os, "uname"):
        raise TasteMainV2AuthorityError("atomic no-replace publication requires POSIX")
    system = os.uname().sysname
    libc = ctypes.CDLL(None, use_errno=True)
    if system == "Linux":
        function = getattr(libc, "renameat2", None)
        if function is None:
            raise TasteMainV2AuthorityError("Linux renameat2 is unavailable")
        result = function(
            ctypes.c_int(-100),
            ctypes.c_char_p(os.fsencode(source)),
            ctypes.c_int(-100),
            ctypes.c_char_p(os.fsencode(destination)),
            ctypes.c_uint(1),
        )
    elif system == "Darwin":
        function = getattr(libc, "renamex_np", None)
        if function is None:
            raise TasteMainV2AuthorityError("Darwin renamex_np is unavailable")
        result = function(
            ctypes.c_char_p(os.fsencode(source)),
            ctypes.c_char_p(os.fsencode(destination)),
            ctypes.c_uint(4),
        )
    else:
        raise TasteMainV2AuthorityError("atomic no-replace publication is unsupported")
    if result != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise TasteMainV2AuthorityError(
                f"immutable target is already burned: {destination.name}"
            )
        raise OSError(code, os.strerror(code), str(destination))


def _publish_immutable(path: Path, data: bytes, *, staging_root: Path) -> None:
    """Write/fsync privately, then expose one complete file atomically."""

    staging = _require_absolute_physical(staging_root, label="publication staging")
    staged = staging / f"{uuid.uuid4()}.json"
    _write_exclusive(staged, data)
    _rename_noreplace(staged, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_fresh_namespace(root: Path, *, children: Sequence[str]) -> None:
    """Create one UUID namespace through held parent/root descriptors."""

    if not root.is_absolute() or root.exists() or root.is_symlink():
        raise TasteMainV2AuthorityError("namespace root must be a fresh absolute path")
    parent = _require_absolute_physical(root.parent, label="namespace parent")
    parent_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.mkdir(root.name, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        root_fd = os.open(
            root.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            for name in children:
                if "/" in name or name in {"", ".", ".."}:
                    raise TasteMainV2AuthorityError("namespace child is malformed")
                os.mkdir(name, 0o700, dir_fd=root_fd)
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
    finally:
        os.close(parent_fd)


def ensure_controller_namespace_parents(
    control_root: str | Path,
) -> tuple[Path, Path]:
    """Create only the fixed parent hierarchy through retained descriptors."""

    requested_root = Path(control_root)
    if not requested_root.is_absolute() or requested_root.name != "control":
        raise TasteMainV2AuthorityError(
            "AutoDL control root must be the canonical runtime control path"
        )
    runtime_root = _require_absolute_physical(
        requested_root.parent, label="AutoDL runtime root"
    )
    if runtime_root.name != "counterfactual-subgraph-runtime":
        raise TasteMainV2AuthorityError("AutoDL runtime root name changed")
    _ensure_physical_child_directory(runtime_root, "control")
    root = _require_absolute_physical(requested_root, label="AutoDL control root")
    descriptor = os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    opened: list[int] = [descriptor]
    try:
        for name in ("taste-main-v2",):
            try:
                os.mkdir(name, 0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            opened.append(child)
            descriptor = child
        for name in ("controllers", "launches"):
            try:
                os.mkdir(name, 0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
        os.fsync(descriptor)
        base = root / "taste-main-v2"
        controllers = _require_absolute_physical(
            base / "controllers", label="controller namespace parent"
        )
        launches = _require_absolute_physical(
            base / "launches", label="launcher namespace parent"
        )
        return controllers, launches
    finally:
        for item in reversed(opened):
            os.close(item)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(payload))


def _decode_json(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteMainV2AuthorityError(f"{label} is not valid JSON") from exc
    if type(value) is not dict:
        raise TasteMainV2AuthorityError(f"{label} must be one JSON object")
    return value


class _HeldFile:
    """One retained single-link regular file opened without following links."""

    def __init__(self, path: Path, *, label: str) -> None:
        self.path = _require_absolute_physical(path, label=label)
        self.label = label
        self.descriptor = -1
        descriptor = os.open(
            self.path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            info = os.fstat(descriptor)
            named = os.stat(self.path, follow_symlinks=False)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino)
            ):
                raise TasteMainV2AuthorityError(
                    f"{label} must be one physical single-link file"
                )
            self.descriptor = descriptor
            self.identity = _identity(info)
            self.data = self._read()
            self.sha256 = _sha256(self.data)
        except BaseException:
            os.close(descriptor)
            raise

    def _read(self) -> bytes:
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(self.descriptor, 64 * 1024)
            if not block:
                break
            total += len(block)
            if total > 4 * 1024 * 1024:
                raise TasteMainV2AuthorityError(f"{self.label} is too large")
            chunks.append(block)
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        return b"".join(chunks)

    def json(self) -> dict[str, Any]:
        return _decode_json(self.data, label=self.label)

    def revalidate(self) -> None:
        info = os.fstat(self.descriptor)
        try:
            named = os.stat(self.path, follow_symlinks=False)
        except OSError as exc:
            raise TasteMainV2AuthorityError(
                f"{self.label} path disappeared while held"
            ) from exc
        if (
            _identity(info) != self.identity
            or (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino)
            or _sha256(self._read()) != self.sha256
        ):
            raise TasteMainV2AuthorityError(f"{self.label} changed while held")

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


def _snapshot_fields(snapshot: ProcessSnapshotV2) -> dict[str, Any]:
    return {
        "pid": snapshot.pid,
        "ppid": snapshot.ppid,
        "pid_start_ticks": snapshot.pid_start_ticks,
        "boot_id": snapshot.boot_id,
        "exe": snapshot.executable_realpath,
        "command": list(snapshot.command),
        "command_hash": snapshot.command_hash,
        "cwd": snapshot.cwd_realpath,
        "cgroup": snapshot.cgroup_path,
    }


def _snapshot_from_payload(payload: Mapping[str, Any]) -> ProcessSnapshotV2:
    return ProcessSnapshotV2.from_mapping(
        {
            "schema_version": "managed_process_identity_v2",
            "pid": payload.get("pid"),
            "ppid": payload.get("ppid"),
            "pid_start_ticks": payload.get("pid_start_ticks"),
            "boot_id": payload.get("boot_id"),
            "executable_realpath": payload.get("exe"),
            "command": payload.get("command"),
            "command_hash": payload.get("command_hash"),
            "cwd_realpath": payload.get("cwd"),
            "cgroup_path": payload.get("cgroup"),
        }
    )


def _validate_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise TasteMainV2AuthorityError(f"{label} is not SHA-256")
    return value


def _validate_controller_id(controller_id: Any, controller_uuid: Any) -> None:
    if not isinstance(controller_id, str) or not controller_id:
        raise TasteMainV2AuthorityError("controller_id is absent")
    require_uuid4(controller_uuid, label="controller_uuid")
    if controller_uuid not in controller_id:
        raise TasteMainV2AuthorityError("controller_id is not bound to controller_uuid")


def inspect_clean_git(project_root: str | Path) -> tuple[str, str]:
    """Return the exact commit/tree only for a clean immutable checkout."""

    root = Path(project_root).resolve(strict=True)
    if not root.is_dir():
        raise TasteMainV2AuthorityError("project root is not a directory")
    try:
        from src.utils.tastemolnet_gine_pass_adoption_v1 import (
            T2PassAdoptionError,
            _git_identity,
        )

        evidence = _git_identity(
            root,
            critical_paths=(
                "src/utils/autodl_tastemolnet_main_v2.py",
                "scripts/autodl/run_taste_main_v2.py",
                "scripts/autodl/launch_taste_main_v2.sh",
            ),
        )
    except (OSError, ValueError, T2PassAdoptionError) as exc:
        raise TasteMainV2AuthorityError("hardened immutable Git audit failed") from exc
    return str(evidence["commit"]), str(evidence["tree"])


@dataclass(frozen=True, slots=True)
class LauncherCreation:
    launcher_root: Path
    receipt_path: Path
    receipt_sha256: str
    payload: dict[str, Any]


def create_launcher_receipt(
    *,
    launcher_root: str | Path,
    controller_id: str,
    controller_uuid: str,
    controller_snapshot: ProcessSnapshotV2,
    project_root: str | Path,
    git_identity: tuple[str, str] | None = None,
    policy_facts: Mapping[str, Any],
    launcher_snapshot: ProcessSnapshotV2 | None = None,
) -> LauncherCreation:
    """Externally bind a newly spawned controller before it can self-receipt."""

    require_auto_termination_disabled()
    controller_uuid = require_uuid4(controller_uuid, label="controller_uuid")
    _validate_controller_id(controller_id, controller_uuid)
    launcher = launcher_snapshot or capture_process_snapshot(os.getpid())
    if launcher.pid != os.getpid():
        raise TasteMainV2AuthorityError("launcher receipt writer identity changed")
    if (
        launcher.same_runtime_identity(controller_snapshot)
        or controller_snapshot.ppid != launcher.pid
    ):
        raise TasteMainV2AuthorityError(
            "external launcher must be the controller's distinct live parent"
        )
    project = Path(project_root).resolve(strict=True)
    commit, tree = git_identity or inspect_clean_git(project)
    policy = dict(policy_facts)
    _validate_policy_facts(policy)
    root = Path(launcher_root)
    expected_parent = (
        Path(str(policy["persistent_control_root"]))
        / "taste-main-v2"
        / "launches"
    )
    if root.parent != expected_parent or root.name != controller_uuid:
        raise TasteMainV2AuthorityError(
            "launcher root is outside its canonical UUID namespace"
        )
    _create_fresh_namespace(root, children=(PUBLICATION_STAGING_DIRECTORY,))
    generation_token = str(uuid.uuid4())
    payload = {
        "schema_version": LAUNCHER_RECEIPT_SCHEMA,
        "managed_taste_release_version": MANAGED_TASTE_RELEASE_VERSION,
        "controller_id": controller_id,
        "controller_uuid": controller_uuid,
        "launcher_generation_token": generation_token,
        "launcher_process": launcher.to_dict(),
        "controller_process": controller_snapshot.to_dict(),
        "git_commit": commit,
        "git_tree": tree,
        "project_root": str(project),
        "policy_facts": policy,
        "policy_facts_sha256": _sha256(_json_bytes(policy)),
        "state": "CONTROLLER_SPAWNED",
        "created_at": _utc_now(),
        "created_at_ns": time.time_ns(),
        "auto_terminate_uncontrolled_children": False,
        "signal_authority": False,
    }
    receipt_path = root / LAUNCHER_RECEIPT_NAME
    data = _json_bytes(payload)
    _publish_immutable(
        receipt_path,
        data,
        staging_root=root / PUBLICATION_STAGING_DIRECTORY,
    )
    _fsync_directory(root)
    return LauncherCreation(root, receipt_path, _sha256(data), payload)


def _validate_launcher_receipt(
    payload: Mapping[str, Any],
    *,
    expected_controller_id: str | None,
    expected_git_commit: str | None,
    expected_git_tree: str | None,
) -> ProcessSnapshotV2:
    if (
        payload.get("schema_version") != LAUNCHER_RECEIPT_SCHEMA
        or payload.get("managed_taste_release_version")
        != MANAGED_TASTE_RELEASE_VERSION
    ):
        raise TasteMainV2AuthorityError("launcher receipt schema changed")
    _validate_controller_id(payload.get("controller_id"), payload.get("controller_uuid"))
    require_uuid4(payload.get("launcher_generation_token"), label="launcher generation")
    if expected_controller_id is not None and payload.get("controller_id") != expected_controller_id:
        raise TasteMainV2AuthorityError("launcher controller ID changed")
    if expected_git_commit is not None and payload.get("git_commit") != expected_git_commit:
        raise TasteMainV2AuthorityError("launcher Git commit changed")
    if expected_git_tree is not None and payload.get("git_tree") != expected_git_tree:
        raise TasteMainV2AuthorityError("launcher Git tree changed")
    if (
        payload.get("state") != "CONTROLLER_SPAWNED"
        or payload.get("auto_terminate_uncontrolled_children") is not False
        or payload.get("signal_authority") is not False
    ):
        raise TasteMainV2AuthorityError("launcher safety contract changed")
    if not Path(str(payload.get("project_root"))).is_absolute():
        raise TasteMainV2AuthorityError("launcher project root is invalid")
    launcher_created_ns = payload.get("created_at_ns")
    if (
        isinstance(launcher_created_ns, bool)
        or not isinstance(launcher_created_ns, int)
        or launcher_created_ns <= 0
    ):
        raise TasteMainV2AuthorityError("launcher creation generation is invalid")
    policy = payload.get("policy_facts")
    if type(policy) is not dict:
        raise TasteMainV2AuthorityError("launcher policy facts are absent")
    _validate_policy_facts(policy)
    if payload.get("policy_facts_sha256") != _sha256(_json_bytes(policy)):
        raise TasteMainV2AuthorityError("launcher policy SHA changed")
    launcher_process = payload.get("launcher_process")
    controller_process = payload.get("controller_process")
    if type(launcher_process) is not dict or type(controller_process) is not dict:
        raise TasteMainV2AuthorityError("launcher process evidence is absent")
    launcher = ProcessSnapshotV2.from_mapping(launcher_process)
    controller = ProcessSnapshotV2.from_mapping(controller_process)
    if launcher.same_runtime_identity(controller) or controller.ppid != launcher.pid:
        raise TasteMainV2AuthorityError(
            "launcher receipt does not bind a distinct parent/controller lineage"
        )
    return controller


def immutable_authority_sha256(path: str | Path, *, label: str) -> str:
    """Return a SHA only after opening one immutable authority file safely."""

    held = _HeldFile(Path(path), label=label)
    try:
        held.revalidate()
        return held.sha256
    finally:
        held.close()


def read_launcher_policy_facts(
    path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    """Read the already externally validated launch policy from its trust root."""

    held = _HeldFile(Path(path), label="external launcher receipt")
    try:
        if held.sha256 != expected_sha256:
            raise TasteMainV2AuthorityError("external launcher receipt SHA changed")
        payload = held.json()
        _validate_launcher_receipt(
            payload,
            expected_controller_id=None,
            expected_git_commit=None,
            expected_git_tree=None,
        )
        policy = payload.get("policy_facts")
        if type(policy) is not dict:
            raise TasteMainV2AuthorityError("external launcher policy is absent")
        held.revalidate()
        return dict(policy)
    finally:
        held.close()


def publish_launcher_ready(
    *,
    launcher_receipt_path: str | Path,
    controller_receipt_path: str | Path,
    controller_anchor_heartbeat_path: str | Path,
    authority_evidence: Mapping[str, Any],
) -> Path:
    """Publish the supervisor's immutable post-validation readiness receipt."""

    launcher = _HeldFile(Path(launcher_receipt_path), label="external launcher receipt")
    controller = _HeldFile(Path(controller_receipt_path), label="controller receipt")
    heartbeat = _HeldFile(
        Path(controller_anchor_heartbeat_path),
        label="controller anchor heartbeat",
    )
    try:
        if launcher.path.parent.name == "" or launcher.path.name != LAUNCHER_RECEIPT_NAME:
            raise TasteMainV2AuthorityError("launcher receipt path is malformed")
        if authority_evidence.get("launcher_receipt_sha256") != launcher.sha256:
            raise TasteMainV2AuthorityError("launcher readiness authority changed")
        if authority_evidence.get("receipt_sha256") != controller.sha256:
            raise TasteMainV2AuthorityError("controller readiness authority changed")
        if (
            authority_evidence.get("anchor_heartbeat_sha256") != heartbeat.sha256
            or authority_evidence.get("anchor_heartbeat_path")
            != str(heartbeat.path)
            or authority_evidence.get("anchor_heartbeat_sequence") != 1
        ):
            raise TasteMainV2AuthorityError(
                "sequence-1 heartbeat readiness anchor changed"
            )
        payload = {
            "schema_version": LAUNCHER_READY_SCHEMA,
            "managed_taste_release_version": MANAGED_TASTE_RELEASE_VERSION,
            "state": "RUNNING",
            "controller_id": authority_evidence["controller_id"],
            "controller_uuid": authority_evidence["controller_uuid"],
            "controller_pid": authority_evidence["pid"],
            "controller_pid_start_ticks": authority_evidence["pid_start_ticks"],
            "launcher_receipt_path": str(launcher.path),
            "launcher_receipt_sha256": launcher.sha256,
            "controller_receipt_path": str(controller.path),
            "controller_receipt_sha256": controller.sha256,
            "controller_anchor_heartbeat_path": str(heartbeat.path),
            "controller_anchor_heartbeat_sha256": heartbeat.sha256,
            "anchor_heartbeat_sequence": 1,
            "controller_terminal_heartbeat_path": authority_evidence[
                "heartbeat_path"
            ],
            "controller_terminal_heartbeat_sha256": authority_evidence[
                "heartbeat_sha256"
            ],
            "terminal_heartbeat_sequence": authority_evidence["sequence"],
            "verified_at": _utc_now(),
            "science_released": False,
            "auto_terminate_uncontrolled_children": False,
            "signal_authority": False,
        }
        path = launcher.path.parent / LAUNCHER_READY_NAME
        _publish_immutable(
            path,
            _json_bytes(payload),
            staging_root=launcher.path.parent / PUBLICATION_STAGING_DIRECTORY,
        )
        launcher.revalidate()
        controller.revalidate()
        heartbeat.revalidate()
        return path
    finally:
        heartbeat.close()
        controller.close()
        launcher.close()


@dataclass(frozen=True, slots=True)
class ControllerCreation:
    controller_root: Path
    receipt_path: Path
    receipt_sha256: str
    payload: dict[str, Any]


def create_controller_receipt(
    *,
    controller_root: str | Path,
    project_root: str | Path,
    controller_id: str,
    controller_uuid: str,
    expected_git_commit: str | None = None,
    expected_git_tree: str | None = None,
    launcher_receipt_path: str | Path,
    expected_launcher_receipt_sha256: str,
    process_snapshot: ProcessSnapshotV2 | None = None,
    git_identity: tuple[str, str] | None = None,
    policy_facts: Mapping[str, Any] | None = None,
) -> ControllerCreation:
    """Create a fresh controller namespace and its immutable receipt."""

    require_auto_termination_disabled()
    controller_uuid = require_uuid4(controller_uuid, label="controller_uuid")
    _validate_controller_id(controller_id, controller_uuid)
    launcher_receipt = _HeldFile(
        Path(launcher_receipt_path), label="external launcher receipt"
    )
    launcher_payload = launcher_receipt.json()
    launched_controller = _validate_launcher_receipt(
        launcher_payload,
        expected_controller_id=controller_id,
        expected_git_commit=expected_git_commit,
        expected_git_tree=expected_git_tree,
    )
    if launcher_receipt.sha256 != expected_launcher_receipt_sha256:
        launcher_receipt.close()
        raise TasteMainV2AuthorityError("external launcher receipt SHA changed")
    if policy_facts is None:
        launcher_receipt.close()
        raise TasteMainV2AuthorityError("validated controller policy facts are required")
    policy = dict(policy_facts)
    _validate_policy_facts(policy)
    root = Path(controller_root)
    expected_controller_parent = (
        Path(str(policy["persistent_control_root"]))
        / "taste-main-v2"
        / "controllers"
    )
    expected_launcher_parent = (
        Path(str(policy["persistent_control_root"]))
        / "taste-main-v2"
        / "launches"
    )
    if root.parent != expected_controller_parent or root.name != controller_uuid:
        launcher_receipt.close()
        raise TasteMainV2AuthorityError(
            "controller root is outside its canonical UUID namespace"
        )
    if (
        launcher_receipt.path.parent.parent != expected_launcher_parent
        or launcher_receipt.path.parent.name != controller_uuid
    ):
        launcher_receipt.close()
        raise TasteMainV2AuthorityError(
            "launcher receipt is outside its canonical UUID namespace"
        )
    if not root.is_absolute() or root.exists() or root.is_symlink():
        raise TasteMainV2AuthorityError("controller root must be a fresh absolute path")
    _require_absolute_physical(root.parent, label="controller parent")
    project = Path(project_root).resolve(strict=True)
    if launcher_payload.get("project_root") != str(project):
        launcher_receipt.close()
        raise TasteMainV2AuthorityError(
            "controller project root differs from external launcher"
        )
    commit, tree = git_identity or inspect_clean_git(project)
    if expected_git_commit is not None and commit != expected_git_commit:
        raise TasteMainV2AuthorityError("controller Git commit differs from authority")
    if expected_git_tree is not None and tree != expected_git_tree:
        raise TasteMainV2AuthorityError("controller Git tree differs from authority")
    snapshot = process_snapshot or capture_process_snapshot(os.getpid())
    if snapshot.pid != os.getpid():
        raise TasteMainV2AuthorityError("controller receipt must describe its writer")
    if not launched_controller.same_runtime_identity(snapshot):
        launcher_receipt.close()
        raise TasteMainV2AuthorityError("controller differs from external launcher target")
    _create_fresh_namespace(
        root,
        children=(
            HEARTBEAT_DIRECTORY,
            GPU_LEASE_DIRECTORY,
            GPU_LEASE_ACTIVATION_DIRECTORY,
            GPU_LEASE_RENEWAL_DIRECTORY,
            GPU_LEASE_RELEASE_DIRECTORY,
            GPU_LOCK_DIRECTORY,
            PUBLICATION_STAGING_DIRECTORY,
        ),
    )
    if _sha256(_json_bytes(policy)) != launcher_payload.get("policy_facts_sha256"):
        launcher_receipt.close()
        raise TasteMainV2AuthorityError("controller policy differs from launcher authority")
    payload = {
        "schema_version": CONTROLLER_RECEIPT_SCHEMA,
        "managed_taste_release_version": MANAGED_TASTE_RELEASE_VERSION,
        "controller_id": controller_id,
        "controller_uuid": controller_uuid,
        **_snapshot_fields(snapshot),
        "git_commit": commit,
        "git_tree": tree,
        "project_root": str(project),
        "launcher_receipt_path": str(launcher_receipt.path),
        "launcher_receipt_sha256": launcher_receipt.sha256,
        "launcher_generation_token": launcher_payload["launcher_generation_token"],
        "state": "RUNNING",
        "controller_state": "WAITING_DEPENDENCIES",
        "created_at": _utc_now(),
        "created_at_ns": time.time_ns(),
        "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
        "allowed_physical_gpu_indices": sorted(ALLOWED_GPU_INDICES),
        "protected_physical_gpu_indices": sorted(PROTECTED_GPU_INDICES),
        "max_active_tasks": MAX_ACTIVE_TASKS,
        "task_gpu_bindings": dict(TASK_GPU_BINDINGS),
        "policy_facts": policy,
        "policy_facts_sha256": _sha256(_json_bytes(policy)),
        "auto_terminate_uncontrolled_children": False,
        "signal_authority": False,
        "append_only_heartbeat_generations": True,
    }
    receipt_path = root / CONTROLLER_RECEIPT_NAME
    data = _json_bytes(payload)
    _publish_immutable(
        receipt_path,
        data,
        staging_root=root / PUBLICATION_STAGING_DIRECTORY,
    )
    launcher_receipt.revalidate()
    launcher_receipt.close()
    _fsync_directory(root)
    return ControllerCreation(root, receipt_path, _sha256(data), payload)


def _validate_policy_facts(policy: Mapping[str, Any]) -> None:
    required = {
        "run_tastemolnet": True,
        "taste_research_compute_allowed": True,
        "taste_paper_results_allowed": True,
        "taste_data_redistribution_allowed": False,
        "primary_taste_source_label": 1,
        "minimum_free_after_reservations_gib": MINIMUM_PERSISTENT_FREE_GIB,
        "scheduler_poll_seconds": SCHEDULER_POLL_SECONDS,
        "max_physical_gpus": 4,
        "max_concurrent_taste_full": 2,
        "gnn_ablation_enabled": False,
    }
    if any(policy.get(name) != value for name, value in required.items()):
        raise TasteMainV2AuthorityError("controller policy facts changed")
    free = policy.get("persistent_free_bytes_at_launch")
    if isinstance(free, bool) or not isinstance(free, int) or free < 100 * 1024**3:
        raise TasteMainV2AuthorityError("controller persistent free-space fact is invalid")
    data_root = Path(str(policy.get("persistent_data_root")))
    runtime_root = Path(str(policy.get("persistent_storage_root")))
    control_root = Path(str(policy.get("persistent_control_root")))
    lock_root = Path(str(policy.get("canonical_gpu_lock_root")))
    if not all(path.is_absolute() for path in (data_root, runtime_root, control_root, lock_root)):
        raise TasteMainV2AuthorityError("controller runtime layout is not absolute")
    if (
        runtime_root != data_root / "counterfactual-subgraph-runtime"
        or control_root != runtime_root / "control"
        or lock_root != runtime_root / "locks"
    ):
        raise TasteMainV2AuthorityError(
            "controller runtime/control/GPU-lock layout is not canonical"
        )


def _runtime_layout_from_receipt(
    receipt_payload: Mapping[str, Any],
) -> RuntimeLayout:
    policy = receipt_payload.get("policy_facts")
    if type(policy) is not dict:
        raise TasteMainV2AuthorityError("controller policy facts are absent")
    _validate_policy_facts(policy)
    try:
        layout = build_runtime_layout(
            project_root=Path(str(receipt_payload["project_root"])),
            data_root=Path(str(policy["persistent_data_root"])),
            control_root=Path(str(policy["persistent_control_root"])),
        )
    except (AutoDLRuntimeError, KeyError, OSError) as exc:
        raise TasteMainV2AuthorityError(
            "controller canonical runtime layout cannot be rebuilt"
        ) from exc
    if (
        layout.runtime_root != Path(str(policy["persistent_storage_root"]))
        or layout.locks_dir != Path(str(policy["canonical_gpu_lock_root"]))
    ):
        raise TasteMainV2AuthorityError("controller runtime layout changed")
    for path, label in (
        (layout.data_root, "persistent data root"),
        (layout.runtime_root, "persistent runtime root"),
        (layout.control_root, "persistent control root"),
        (layout.locks_dir, "canonical GPU lock root"),
    ):
        _require_absolute_physical(path, label=label)
    return layout


def _validate_controller_namespace_binding(
    receipt_path: Path, receipt_payload: Mapping[str, Any]
) -> RuntimeLayout:
    layout = _runtime_layout_from_receipt(receipt_payload)
    controller_uuid = require_uuid4(
        receipt_payload.get("controller_uuid"), label="controller_uuid"
    )
    expected_receipt = (
        layout.control_root
        / "taste-main-v2"
        / "controllers"
        / controller_uuid
        / CONTROLLER_RECEIPT_NAME
    )
    expected_launcher = (
        layout.control_root
        / "taste-main-v2"
        / "launches"
        / controller_uuid
        / LAUNCHER_RECEIPT_NAME
    )
    if receipt_path != expected_receipt:
        raise TasteMainV2AuthorityError(
            "controller receipt is outside its canonical UUID namespace"
        )
    if Path(str(receipt_payload.get("launcher_receipt_path"))) != expected_launcher:
        raise TasteMainV2AuthorityError(
            "launcher receipt is outside its canonical UUID namespace"
        )
    return layout


def _validate_receipt(
    payload: Mapping[str, Any],
    *,
    expected_controller_id: str | None,
    expected_git_commit: str | None,
    expected_git_tree: str | None,
) -> ProcessSnapshotV2:
    if (
        payload.get("schema_version") != CONTROLLER_RECEIPT_SCHEMA
        or payload.get("managed_taste_release_version")
        != MANAGED_TASTE_RELEASE_VERSION
    ):
        raise TasteMainV2AuthorityError("controller receipt schema changed")
    _validate_controller_id(payload.get("controller_id"), payload.get("controller_uuid"))
    if expected_controller_id is not None and payload.get("controller_id") != expected_controller_id:
        raise TasteMainV2AuthorityError("controller receipt ID differs from authority")
    if expected_git_commit is not None and payload.get("git_commit") != expected_git_commit:
        raise TasteMainV2AuthorityError("controller receipt commit differs from authority")
    if expected_git_tree is not None and payload.get("git_tree") != expected_git_tree:
        raise TasteMainV2AuthorityError("controller receipt tree differs from authority")
    if (
        payload.get("state") != "RUNNING"
        or payload.get("auto_terminate_uncontrolled_children") is not False
        or payload.get("signal_authority") is not False
        or payload.get("append_only_heartbeat_generations") is not True
        or payload.get("allowed_physical_gpu_indices") != [1, 2]
        or payload.get("protected_physical_gpu_indices") != [0, 3]
        or payload.get("max_active_tasks") != 2
        or payload.get("task_gpu_bindings") != TASK_GPU_BINDINGS
        or payload.get("heartbeat_interval_seconds") != 10
    ):
        raise TasteMainV2AuthorityError("controller receipt safety contract changed")
    policy = payload.get("policy_facts")
    if type(policy) is not dict:
        raise TasteMainV2AuthorityError("controller policy facts are absent")
    _validate_policy_facts(policy)
    if payload.get("policy_facts_sha256") != _sha256(_json_bytes(policy)):
        raise TasteMainV2AuthorityError("controller policy facts SHA changed")
    _validate_sha256(
        payload.get("launcher_receipt_sha256"), label="launcher receipt"
    )
    if not Path(str(payload.get("launcher_receipt_path"))).is_absolute():
        raise TasteMainV2AuthorityError("launcher receipt path is invalid")
    require_uuid4(
        payload.get("launcher_generation_token"), label="launcher generation"
    )
    if not re.fullmatch(r"[0-9a-f]{40}", str(payload.get("git_commit"))):
        raise TasteMainV2AuthorityError("controller receipt commit is malformed")
    if not re.fullmatch(r"[0-9a-f]{40}", str(payload.get("git_tree"))):
        raise TasteMainV2AuthorityError("controller receipt tree is malformed")
    receipt_created_ns = payload.get("created_at_ns")
    if (
        isinstance(receipt_created_ns, bool)
        or not isinstance(receipt_created_ns, int)
        or receipt_created_ns <= 0
    ):
        raise TasteMainV2AuthorityError("controller creation generation is invalid")
    return _snapshot_from_payload(payload)


@dataclass(frozen=True, slots=True)
class GpuLeaseCreation:
    path: Path
    sha256: str
    lease_uuid: str
    payload: dict[str, Any]


def create_gpu_lease_request(
    *,
    controller_receipt_path: str | Path,
    task_id: str,
    physical_gpu_index: int,
    physical_gpu_uuid: str,
    lease_uuid: str | None = None,
    lifetime_seconds: int = 21600,
) -> GpuLeaseCreation:
    """Create an immutable request; authority starts after a heartbeat acks it."""

    require_auto_termination_disabled()
    if not _TASK_ID.fullmatch(task_id):
        raise TasteMainV2AuthorityError("GPU lease task_id is malformed")
    if TASK_GPU_BINDINGS.get(task_id) != physical_gpu_index:
        raise TasteMainV2AuthorityError("task/GPU pair is not a fixed Taste binding")
    if physical_gpu_index not in ALLOWED_GPU_INDICES:
        raise TasteMainV2AuthorityError("GPU lease targets a non-Taste GPU")
    if physical_gpu_index in PROTECTED_GPU_INDICES:
        raise TasteMainV2AuthorityError("GPU lease targets a protected GPU")
    if not _GPU_UUID.fullmatch(physical_gpu_uuid):
        raise TasteMainV2AuthorityError("GPU lease UUID is malformed")
    if isinstance(lifetime_seconds, bool) or not 30 <= lifetime_seconds <= 86400:
        raise TasteMainV2AuthorityError("GPU lease lifetime is outside 30..86400 seconds")
    lease_uuid = require_uuid4(
        lease_uuid or str(uuid.uuid4()), label="gpu_lease_uuid"
    )
    receipt = _HeldFile(Path(controller_receipt_path), label="controller receipt")
    try:
        receipt_payload = receipt.json()
        _validate_receipt(
            receipt_payload,
            expected_controller_id=None,
            expected_git_commit=None,
            expected_git_tree=None,
        )
        _validate_controller_namespace_binding(receipt.path, receipt_payload)
        root = receipt.path.parent
        if receipt.path.name != CONTROLLER_RECEIPT_NAME:
            raise TasteMainV2AuthorityError("controller receipt filename changed")
        lease_dir = _require_absolute_physical(
            root / GPU_LEASE_DIRECTORY, label="GPU lease directory"
        )
        now_ns = time.time_ns()
        payload = {
            "schema_version": GPU_LEASE_SCHEMA,
            "lease_uuid": lease_uuid,
            "task_id": task_id,
            "controller_id": receipt_payload["controller_id"],
            "controller_uuid": receipt_payload["controller_uuid"],
            "controller_receipt_sha256": receipt.sha256,
            "policy_facts_sha256": receipt_payload["policy_facts_sha256"],
            "physical_gpu_index": physical_gpu_index,
            "physical_gpu_uuid": physical_gpu_uuid,
            "state": "REQUESTED",
            "created_at": _utc_now(),
            "created_at_ns": now_ns,
            "expires_at_ns": now_ns + lifetime_seconds * 1_000_000_000,
            "auto_terminate_uncontrolled_children": False,
        }
        path = lease_dir / f"{task_id}-{lease_uuid}.json"
        data = _json_bytes(payload)
        _publish_immutable(
            path,
            data,
            staging_root=root / PUBLICATION_STAGING_DIRECTORY,
        )
        receipt.revalidate()
        return GpuLeaseCreation(path, _sha256(data), lease_uuid, payload)
    finally:
        receipt.close()


@dataclass(frozen=True, slots=True)
class GpuLeaseActivationCreation:
    path: Path
    sha256: str
    activation_uuid: str
    payload: dict[str, Any]


def create_gpu_lease_activation(
    *,
    controller_receipt_path: str | Path,
    lease_path: str | Path,
    expected_lease_sha256: str,
    attempt_id: str,
    generation_token: str,
    managed_launcher: ProcessSnapshotV2 | None = None,
    managed_worker: ProcessSnapshotV2 | None = None,
    training_child: ProcessSnapshotV2 | None = None,
    activation_uuid: str | None = None,
    activation_sequence: int = 1,
    previous_activation_sha256: str | None = None,
    phase: str = "WORKER_ACTIVE",
) -> GpuLeaseActivationCreation:
    """Register a live managed lineage; controller alone can acknowledge it."""

    require_auto_termination_disabled()
    require_uuid4(attempt_id, label="managed attempt_id")
    require_uuid4(generation_token, label="managed generation_token")
    if isinstance(activation_sequence, bool) or activation_sequence <= 0:
        raise TasteMainV2AuthorityError("activation sequence is invalid")
    if activation_sequence == 1 and previous_activation_sha256 is not None:
        raise TasteMainV2AuthorityError("first activation cannot have a predecessor")
    if activation_sequence > 1:
        _validate_sha256(previous_activation_sha256, label="previous activation")
    if phase not in {
        "WORKER_ACTIVE",
        "WAITING_VERIFIER",
        "VERIFIER_ACTIVE",
        "RELEASE_REQUESTED",
    }:
        raise TasteMainV2AuthorityError("activation phase is invalid")
    if phase in {"WAITING_VERIFIER", "RELEASE_REQUESTED"} and training_child is not None:
        raise TasteMainV2AuthorityError(
            "non-scientific activation phase cannot retain a child"
        )
    writer = capture_process_snapshot(os.getpid())
    worker = managed_worker or writer
    lineage = [item for item in (managed_launcher, worker, training_child) if item]
    if not any(writer.same_runtime_identity(item) for item in lineage):
        raise TasteMainV2AuthorityError(
            "lease activation writer is outside the declared managed lineage"
        )
    if not writer.same_runtime_identity(worker):
        raise TasteMainV2AuthorityError(
            "global GPU UUID lock must be held by the registered managed runner"
        )
    if training_child is not None and training_child.ppid not in {
        worker.pid,
        managed_launcher.pid if managed_launcher is not None else worker.pid,
    }:
        raise TasteMainV2AuthorityError("training child is outside permitted lineage")
    activation_uuid = require_uuid4(
        activation_uuid or str(uuid.uuid4()), label="activation_uuid"
    )
    receipt = _HeldFile(Path(controller_receipt_path), label="controller receipt")
    lease = _HeldFile(Path(lease_path), label="GPU lease request")
    try:
        receipt_payload = receipt.json()
        _validate_receipt(
            receipt_payload,
            expected_controller_id=None,
            expected_git_commit=None,
            expected_git_tree=None,
        )
        _validate_controller_namespace_binding(receipt.path, receipt_payload)
        if lease.sha256 != expected_lease_sha256:
            raise TasteMainV2AuthorityError("GPU lease request SHA changed")
        lease_payload = _validate_lease(
            lease,
            receipt_payload=receipt_payload,
            receipt_sha256=receipt.sha256,
            now_ns=time.time_ns(),
        )
        expected_lease_dir = receipt.path.parent / GPU_LEASE_DIRECTORY
        if lease.path.parent != expected_lease_dir:
            raise TasteMainV2AuthorityError("GPU lease request is outside controller root")
        lock_root = _runtime_layout_from_receipt(receipt_payload).locks_dir
        lock_key = str(lease_payload["lease_uuid"])
        lock = _WORKER_GPU_LOCKS.get(lock_key)
        acquired_new_lock = False
        if lock is None:
            lock = GPUFileLock(
                lock_root,
                gpu_index=lease_payload["physical_gpu_index"],
                gpu_uuid=lease_payload["physical_gpu_uuid"],
                owner={
                    "controller_id": receipt_payload["controller_id"],
                    "controller_receipt_sha256": receipt.sha256,
                    "lease_uuid": lease_payload["lease_uuid"],
                    "attempt_id": attempt_id,
                    "generation_token": generation_token,
                    "pid_start_ticks": writer.pid_start_ticks,
                    "boot_id": writer.boot_id,
                    "command_hash": writer.command_hash,
                },
            )
            try:
                lock.acquire()
                acquired_new_lock = True
            except GPULockError as exc:
                raise TasteMainV2AuthorityError(
                    "registered managed runner cannot acquire shared GPU UUID lock"
                ) from exc
        worker_lock = _HeldFile(
            lock.path.resolve(strict=True), label="managed runner GPU UUID lock"
        )
        payload = {
            "schema_version": GPU_LEASE_ACTIVATION_SCHEMA,
            "activation_uuid": activation_uuid,
            "activation_sequence": activation_sequence,
            "previous_activation_sha256": previous_activation_sha256,
            "phase": phase,
            "lease_uuid": lease_payload["lease_uuid"],
            "lease_path": str(lease.path),
            "lease_sha256": lease.sha256,
            "controller_id": receipt_payload["controller_id"],
            "controller_receipt_sha256": receipt.sha256,
            "task_id": lease_payload["task_id"],
            "physical_gpu_index": lease_payload["physical_gpu_index"],
            "physical_gpu_uuid": lease_payload["physical_gpu_uuid"],
            "attempt_id": attempt_id,
            "generation_token": generation_token,
            "registration_writer": writer.to_dict(),
            "managed_launcher": (
                managed_launcher.to_dict() if managed_launcher is not None else None
            ),
            "managed_worker": worker.to_dict(),
            "training_child": (
                training_child.to_dict() if training_child is not None else None
            ),
            "permitted_training_child_count": 1 if training_child is not None else 0,
            "worker_gpu_lock_path": str(worker_lock.path),
            "worker_gpu_lock_sha256": worker_lock.sha256,
            "state": "ACTIVATION_REQUESTED",
            "created_at": _utc_now(),
            "created_at_ns": time.time_ns(),
            "auto_terminate_uncontrolled_children": False,
        }
        directory = _require_absolute_physical(
            receipt.path.parent / GPU_LEASE_ACTIVATION_DIRECTORY,
            label="GPU lease activation directory",
        )
        path = directory / (
            f"{lease_payload['lease_uuid']}-{activation_sequence:020d}-"
            f"{activation_uuid}.json"
        )
        data = _json_bytes(payload)
        _publish_immutable(
            path,
            data,
            staging_root=receipt.path.parent / PUBLICATION_STAGING_DIRECTORY,
        )
        receipt.revalidate()
        lease.revalidate()
        worker_lock.revalidate()
        worker_lock.close()
        _WORKER_GPU_LOCKS[lock_key] = lock
        return GpuLeaseActivationCreation(path, _sha256(data), activation_uuid, payload)
    except BaseException:
        if locals().get("acquired_new_lock") and "lock" in locals():
            lock.release()
        if "worker_lock" in locals():
            worker_lock.close()
        raise
    finally:
        lease.close()
        receipt.close()


def _validate_activation(
    held: _HeldFile,
    *,
    lease_payload: Mapping[str, Any],
    lease_sha256: str,
    receipt_payload: Mapping[str, Any],
    receipt_sha256: str,
    snapshot_reader: SnapshotReader,
    check_live: bool = True,
) -> dict[str, Any]:
    payload = held.json()
    matched = _ACTIVATION_NAME.fullmatch(held.path.name)
    if matched is None:
        raise TasteMainV2AuthorityError("GPU lease activation filename is malformed")
    if (
        payload.get("schema_version") != GPU_LEASE_ACTIVATION_SCHEMA
        or payload.get("activation_uuid") != matched.group("uuid")
        or payload.get("activation_sequence") != int(matched.group("sequence"))
        or payload.get("lease_uuid") != matched.group("lease")
        or payload.get("lease_uuid") != lease_payload.get("lease_uuid")
        or payload.get("lease_sha256") != lease_sha256
        or payload.get("controller_id") != receipt_payload.get("controller_id")
        or payload.get("controller_receipt_sha256") != receipt_sha256
        or payload.get("task_id") != lease_payload.get("task_id")
        or payload.get("physical_gpu_index")
        != lease_payload.get("physical_gpu_index")
        or payload.get("physical_gpu_uuid") != lease_payload.get("physical_gpu_uuid")
        or payload.get("state") != "ACTIVATION_REQUESTED"
        or payload.get("phase")
        not in {
            "WORKER_ACTIVE",
            "WAITING_VERIFIER",
            "VERIFIER_ACTIVE",
            "RELEASE_REQUESTED",
        }
        or payload.get("auto_terminate_uncontrolled_children") is not False
    ):
        raise TasteMainV2AuthorityError("GPU lease activation binding changed")
    require_uuid4(payload.get("attempt_id"), label="activation attempt_id")
    require_uuid4(payload.get("generation_token"), label="activation generation_token")
    snapshots: list[ProcessSnapshotV2] = []
    for field in ("registration_writer", "managed_launcher", "managed_worker", "training_child"):
        raw = payload.get(field)
        if raw is None and field in {"managed_launcher", "training_child"}:
            continue
        if type(raw) is not dict:
            raise TasteMainV2AuthorityError("activation process lineage is malformed")
        expected = ProcessSnapshotV2.from_mapping(raw)
        if check_live:
            observed = snapshot_reader(expected.pid)
            if not expected.same_runtime_identity(observed):
                raise TasteMainV2AuthorityError("activation process generation is not live")
        snapshots.append(expected)
    writer = ProcessSnapshotV2.from_mapping(payload["registration_writer"])
    if not any(writer.same_runtime_identity(item) for item in snapshots[1:]):
        raise TasteMainV2AuthorityError("activation writer is outside live lineage")
    child = payload.get("training_child")
    if payload["phase"] in {"WAITING_VERIFIER", "RELEASE_REQUESTED"} and child is not None:
        raise TasteMainV2AuthorityError("non-scientific activation retained a child")
    expected_child_count = 1 if child is not None else 0
    if payload.get("permitted_training_child_count") != expected_child_count:
        raise TasteMainV2AuthorityError("activation child policy changed")
    worker = ProcessSnapshotV2.from_mapping(payload["managed_worker"])
    worker_lock_path = Path(str(payload.get("worker_gpu_lock_path")))
    if not worker_lock_path.is_absolute():
        raise TasteMainV2AuthorityError("managed runner GPU lock path is malformed")
    worker_lock = _HeldFile(worker_lock_path, label="managed runner GPU UUID lock")
    try:
        if worker_lock.sha256 != payload.get("worker_gpu_lock_sha256"):
            raise TasteMainV2AuthorityError("managed runner GPU lock SHA changed")
        metadata = worker_lock.json()
        if (
            metadata.get("state") != "LOCKED"
            or metadata.get("gpu_index") != lease_payload.get("physical_gpu_index")
            or metadata.get("gpu_uuid") != lease_payload.get("physical_gpu_uuid")
            or metadata.get("pid") != worker.pid
            or metadata.get("pid_start_ticks") != worker.pid_start_ticks
            or metadata.get("boot_id") != worker.boot_id
            or metadata.get("command_hash") != worker.command_hash
            or metadata.get("lease_uuid") != lease_payload.get("lease_uuid")
            or metadata.get("attempt_id") != payload.get("attempt_id")
            or metadata.get("generation_token") != payload.get("generation_token")
        ):
            raise TasteMainV2AuthorityError(
                "managed runner GPU lock metadata differs from live generation"
            )
        probe = os.open(
            worker_lock.path,
            os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        acquired = False
        try:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                pass
            if acquired and check_live:
                fcntl.flock(probe, fcntl.LOCK_UN)
                raise TasteMainV2AuthorityError(
                    "managed runner does not hold the global GPU UUID lock"
                )
            if acquired:
                fcntl.flock(probe, fcntl.LOCK_UN)
        finally:
            os.close(probe)
    finally:
        worker_lock.close()
    return payload


@dataclass(frozen=True, slots=True)
class GpuLeaseRenewalCreation:
    path: Path
    sha256: str
    sequence: int
    payload: dict[str, Any]


def create_gpu_lease_renewal(
    *,
    controller_receipt_path: str | Path,
    lease_path: str | Path,
    expected_lease_sha256: str,
    attempt_id: str,
    generation_token: str,
    sequence: int,
    previous_renewal_sha256: str | None,
    lifetime_seconds: int = 21600,
    renewal_uuid: str | None = None,
) -> GpuLeaseRenewalCreation:
    """Append a runner-authored renewal request for controller acknowledgement."""

    require_auto_termination_disabled()
    require_uuid4(attempt_id, label="renewal attempt_id")
    require_uuid4(generation_token, label="renewal generation_token")
    if isinstance(sequence, bool) or sequence <= 0:
        raise TasteMainV2AuthorityError("renewal sequence is invalid")
    if sequence == 1 and previous_renewal_sha256 is not None:
        raise TasteMainV2AuthorityError("first renewal cannot have a predecessor")
    if sequence > 1:
        _validate_sha256(previous_renewal_sha256, label="previous renewal")
    if isinstance(lifetime_seconds, bool) or not 30 <= lifetime_seconds <= 86400:
        raise TasteMainV2AuthorityError("renewal lifetime is outside 30..86400 seconds")
    renewal_uuid = require_uuid4(
        renewal_uuid or str(uuid.uuid4()), label="renewal_uuid"
    )
    receipt = _HeldFile(Path(controller_receipt_path), label="controller receipt")
    lease = _HeldFile(Path(lease_path), label="GPU lease request")
    try:
        receipt_payload = receipt.json()
        _validate_receipt(
            receipt_payload,
            expected_controller_id=None,
            expected_git_commit=None,
            expected_git_tree=None,
        )
        _validate_controller_namespace_binding(receipt.path, receipt_payload)
        if lease.sha256 != expected_lease_sha256:
            raise TasteMainV2AuthorityError("renewal lease SHA changed")
        lease_payload = _validate_lease(
            lease,
            receipt_payload=receipt_payload,
            receipt_sha256=receipt.sha256,
            now_ns=time.time_ns(),
            allow_expired=True,
        )
        retained = _WORKER_GPU_LOCKS.get(str(lease_payload["lease_uuid"]))
        if retained is None:
            raise TasteMainV2AuthorityError(
                "only the registered lock-holding runner may renew a lease"
            )
        writer = capture_process_snapshot(os.getpid())
        lock = _HeldFile(retained.path.resolve(strict=True), label="runner GPU lock")
        try:
            metadata = lock.json()
            if (
                metadata.get("state") != "LOCKED"
                or metadata.get("pid") != writer.pid
                or metadata.get("pid_start_ticks") != writer.pid_start_ticks
                or metadata.get("boot_id") != writer.boot_id
                or metadata.get("command_hash") != writer.command_hash
                or metadata.get("lease_uuid") != lease_payload["lease_uuid"]
                or metadata.get("attempt_id") != attempt_id
                or metadata.get("generation_token") != generation_token
            ):
                raise TasteMainV2AuthorityError("renewal runner generation changed")
            now_ns = time.time_ns()
            prior_expiry = int(lease_payload["expires_at_ns"])
            renewal_directory = _require_absolute_physical(
                receipt.path.parent / GPU_LEASE_RENEWAL_DIRECTORY,
                label="GPU lease renewal directory",
            )
            prior_entries: list[tuple[int, Path]] = []
            for entry in os.scandir(renewal_directory):
                matched = _RENEWAL_NAME.fullmatch(entry.name)
                if matched is None:
                    raise TasteMainV2AuthorityError(
                        "unexpected GPU renewal directory entry"
                    )
                if matched.group("lease") == lease_payload["lease_uuid"]:
                    prior_entries.append(
                        (int(matched.group("sequence")), renewal_directory / entry.name)
                    )
            prior_entries.sort()
            if len(prior_entries) != sequence - 1 or any(
                number != expected
                for expected, (number, _path) in enumerate(prior_entries, start=1)
            ):
                raise TasteMainV2AuthorityError("GPU renewal chain has a jump")
            if prior_entries:
                prior = _HeldFile(prior_entries[-1][1], label="previous GPU renewal")
                try:
                    if prior.sha256 != previous_renewal_sha256:
                        raise TasteMainV2AuthorityError(
                            "previous GPU renewal SHA changed"
                        )
                    prior_payload = prior.json()
                    prior_expiry = int(prior_payload["requested_expires_at_ns"])
                finally:
                    prior.close()
            payload = {
                "schema_version": GPU_LEASE_RENEWAL_SCHEMA,
                "renewal_uuid": renewal_uuid,
                "sequence": sequence,
                "previous_renewal_sha256": previous_renewal_sha256,
                "lease_uuid": lease_payload["lease_uuid"],
                "lease_path": str(lease.path),
                "lease_sha256": lease.sha256,
                "controller_id": receipt_payload["controller_id"],
                "controller_receipt_sha256": receipt.sha256,
                "task_id": lease_payload["task_id"],
                "attempt_id": attempt_id,
                "generation_token": generation_token,
                "registration_writer": writer.to_dict(),
                "requested_at": _utc_now(),
                "requested_at_ns": now_ns,
                "requested_expires_at_ns": (
                    max(now_ns, prior_expiry)
                    + lifetime_seconds * 1_000_000_000
                ),
                "state": "RENEWAL_REQUESTED",
                "auto_terminate_uncontrolled_children": False,
            }
            directory = renewal_directory
            path = directory / (
                f"{lease_payload['lease_uuid']}-{sequence:020d}-{renewal_uuid}.json"
            )
            data = _json_bytes(payload)
            _publish_immutable(
                path,
                data,
                staging_root=receipt.path.parent / PUBLICATION_STAGING_DIRECTORY,
            )
            receipt.revalidate()
            lease.revalidate()
            lock.revalidate()
            return GpuLeaseRenewalCreation(path, _sha256(data), sequence, payload)
        finally:
            lock.close()
    finally:
        lease.close()
        receipt.close()


def _renewal_chain(
    controller_root: Path,
    *,
    lease_payload: Mapping[str, Any],
    lease_sha256: str,
    receipt_payload: Mapping[str, Any],
    receipt_sha256: str,
    expected_worker: ProcessSnapshotV2,
    snapshot_reader: SnapshotReader,
) -> tuple[int, dict[str, Any] | None, Path | None, str | None]:
    directory = _require_absolute_physical(
        controller_root / GPU_LEASE_RENEWAL_DIRECTORY,
        label="GPU lease renewal directory",
    )
    lease_uuid = str(lease_payload["lease_uuid"])
    entries: list[tuple[int, Path]] = []
    for entry in os.scandir(directory):
        matched = _RENEWAL_NAME.fullmatch(entry.name)
        if matched is None:
            raise TasteMainV2AuthorityError("unexpected GPU renewal directory entry")
        if matched.group("lease") == lease_uuid:
            entries.append((int(matched.group("sequence")), directory / entry.name))
    entries.sort()
    previous: str | None = None
    effective_expiry = int(lease_payload["expires_at_ns"])
    terminal_payload: dict[str, Any] | None = None
    terminal_path: Path | None = None
    terminal_sha: str | None = None
    for expected_sequence, (sequence, path) in enumerate(entries, start=1):
        held = _HeldFile(path, label="GPU lease renewal")
        try:
            payload = held.json()
            matched = _RENEWAL_NAME.fullmatch(path.name)
            writer_raw = payload.get("registration_writer")
            if (
                matched is None
                or sequence != expected_sequence
                or payload.get("schema_version") != GPU_LEASE_RENEWAL_SCHEMA
                or payload.get("renewal_uuid") != matched.group("uuid")
                or payload.get("sequence") != expected_sequence
                or payload.get("previous_renewal_sha256") != previous
                or payload.get("lease_uuid") != lease_uuid
                or payload.get("lease_sha256") != lease_sha256
                or payload.get("controller_id") != receipt_payload["controller_id"]
                or payload.get("controller_receipt_sha256") != receipt_sha256
                or payload.get("task_id") != lease_payload["task_id"]
                or payload.get("attempt_id") is None
                or payload.get("generation_token") is None
                or payload.get("state") != "RENEWAL_REQUESTED"
                or payload.get("auto_terminate_uncontrolled_children") is not False
                or type(writer_raw) is not dict
            ):
                raise TasteMainV2AuthorityError("GPU renewal binding changed")
            require_uuid4(payload["attempt_id"], label="renewal attempt_id")
            require_uuid4(payload["generation_token"], label="renewal generation")
            writer = ProcessSnapshotV2.from_mapping(writer_raw)
            if not writer.same_runtime_identity(expected_worker):
                raise TasteMainV2AuthorityError("GPU renewal runner changed")
            requested_expiry = payload.get("requested_expires_at_ns")
            if (
                isinstance(requested_expiry, bool)
                or not isinstance(requested_expiry, int)
                or requested_expiry <= effective_expiry
            ):
                raise TasteMainV2AuthorityError("GPU renewal expiry did not advance")
            if expected_sequence == len(entries):
                observed = snapshot_reader(writer.pid)
                if not writer.same_runtime_identity(observed):
                    raise TasteMainV2AuthorityError("terminal renewal runner is not live")
                terminal_payload = payload
                terminal_path = path
                terminal_sha = held.sha256
            effective_expiry = requested_expiry
            previous = held.sha256
        finally:
            held.close()
    return effective_expiry, terminal_payload, terminal_path, terminal_sha


def _validate_lease(
    lease: _HeldFile,
    *,
    receipt_payload: Mapping[str, Any],
    receipt_sha256: str,
    now_ns: int,
    allow_expired: bool = False,
) -> dict[str, Any]:
    payload = lease.json()
    matched = _LEASE_NAME.fullmatch(lease.path.name)
    if matched is None:
        raise TasteMainV2AuthorityError("GPU lease filename is malformed")
    if (
        payload.get("schema_version") != GPU_LEASE_SCHEMA
        or payload.get("lease_uuid") != matched.group("uuid")
        or payload.get("task_id") != matched.group("task")
        or payload.get("controller_id") != receipt_payload.get("controller_id")
        or payload.get("controller_uuid") != receipt_payload.get("controller_uuid")
        or payload.get("controller_receipt_sha256") != receipt_sha256
        or payload.get("policy_facts_sha256")
        != receipt_payload.get("policy_facts_sha256")
        or payload.get("state") != "REQUESTED"
        or payload.get("auto_terminate_uncontrolled_children") is not False
    ):
        raise TasteMainV2AuthorityError("GPU lease binding changed")
    index = payload.get("physical_gpu_index")
    if isinstance(index, bool) or index not in ALLOWED_GPU_INDICES or index in PROTECTED_GPU_INDICES:
        raise TasteMainV2AuthorityError("GPU lease targets an unauthorized GPU")
    if TASK_GPU_BINDINGS.get(str(payload.get("task_id"))) != index:
        raise TasteMainV2AuthorityError("GPU lease task/GPU binding changed")
    if not _GPU_UUID.fullmatch(str(payload.get("physical_gpu_uuid"))):
        raise TasteMainV2AuthorityError("GPU lease physical UUID is malformed")
    created = payload.get("created_at_ns")
    expires = payload.get("expires_at_ns")
    if (
        isinstance(created, bool)
        or not isinstance(created, int)
        or isinstance(expires, bool)
        or not isinstance(expires, int)
        or expires <= created
        or (not allow_expired and now_ns >= expires)
    ):
        raise TasteMainV2AuthorityError("GPU lease is expired or malformed")
    return payload


def _release_acknowledgement(
    controller_root: Path,
    *,
    receipt_payload: Mapping[str, Any],
    receipt_sha256: str,
    lease_payload: Mapping[str, Any],
    lease_sha256: str,
    activation_payload: Mapping[str, Any],
    activation_path: Path,
    activation_sha256: str,
) -> tuple[Path, str]:
    if (
        activation_payload.get("phase") != "RELEASE_REQUESTED"
        or activation_payload.get("training_child") is not None
    ):
        raise TasteMainV2AuthorityError("lease release request is not clean")
    release_uuid = str(activation_payload["activation_uuid"])
    path = (
        controller_root
        / GPU_LEASE_RELEASE_DIRECTORY
        / f"{lease_payload['lease_uuid']}-{release_uuid}.json"
    )
    payload = {
        "schema_version": GPU_LEASE_RELEASE_SCHEMA,
        "release_uuid": release_uuid,
        "controller_id": receipt_payload["controller_id"],
        "controller_receipt_sha256": receipt_sha256,
        "task_id": lease_payload["task_id"],
        "lease_uuid": lease_payload["lease_uuid"],
        "lease_sha256": lease_sha256,
        "activation_path": str(activation_path),
        "activation_sha256": activation_sha256,
        "attempt_id": activation_payload["attempt_id"],
        "generation_token": activation_payload["generation_token"],
        "managed_worker": activation_payload["managed_worker"],
        "state": "RELEASE_ACKNOWLEDGED",
        "acknowledged_at": _utc_now(),
        "auto_terminate_uncontrolled_children": False,
        "signal_authority": False,
    }
    data = _json_bytes(payload)
    if not path.exists():
        _publish_immutable(
            path,
            data,
            staging_root=controller_root / PUBLICATION_STAGING_DIRECTORY,
        )
    held = _HeldFile(path, label="GPU lease release acknowledgement")
    try:
        actual = held.json()
        stable = set(payload) - {"acknowledged_at"}
        if any(actual.get(field) != payload[field] for field in stable):
            raise TasteMainV2AuthorityError("GPU release acknowledgement changed")
        return path, held.sha256
    finally:
        held.close()


def _validate_release_ack(
    path: Path,
    *,
    receipt_payload: Mapping[str, Any],
    receipt_sha256: str,
    lease_payload: Mapping[str, Any],
    lease_sha256: str,
) -> dict[str, Any]:
    held = _HeldFile(path, label="GPU lease release acknowledgement")
    try:
        payload = held.json()
        matched = _RELEASE_NAME.fullmatch(path.name)
        if (
            matched is None
            or payload.get("schema_version") != GPU_LEASE_RELEASE_SCHEMA
            or payload.get("release_uuid") != matched.group("uuid")
            or payload.get("lease_uuid") != matched.group("lease")
            or payload.get("lease_uuid") != lease_payload["lease_uuid"]
            or payload.get("lease_sha256") != lease_sha256
            or payload.get("controller_id") != receipt_payload["controller_id"]
            or payload.get("controller_receipt_sha256") != receipt_sha256
            or payload.get("task_id") != lease_payload["task_id"]
            or payload.get("state") != "RELEASE_ACKNOWLEDGED"
            or payload.get("auto_terminate_uncontrolled_children") is not False
            or payload.get("signal_authority") is not False
        ):
            raise TasteMainV2AuthorityError("GPU release acknowledgement binding changed")
        return payload
    finally:
        held.close()


def _release_controller_coordination(
    *, receipt_payload: Mapping[str, Any], lease_payload: Mapping[str, Any]
) -> None:
    lock_root = _runtime_layout_from_receipt(receipt_payload).locks_dir
    candidate = GPUFileLock(
        lock_root,
        gpu_index=int(lease_payload["physical_gpu_index"]),
        gpu_uuid=f"{lease_payload['physical_gpu_uuid']}.taste-main-v2-controller",
    )
    retained = _GPU_LOCKS.get(str(candidate.path))
    if retained is None or retained[1] != lease_payload["lease_uuid"]:
        raise TasteMainV2AuthorityError("controller coordination lock is absent")
    lock, _lease_uuid = retained
    lock.release()
    _GPU_LOCKS.pop(str(candidate.path), None)


def release_registered_runner_gpu_lock_after_ack(
    *,
    controller_receipt_path: str | Path,
    lease_path: str | Path,
    expected_lease_sha256: str,
    release_activation: GpuLeaseActivationCreation,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Release the runner UUID lock only after a controller release ACK exists."""

    require_auto_termination_disabled()
    if timeout_seconds != 60:
        raise TasteMainV2AuthorityError("production release ACK timeout must be 60 seconds")
    if release_activation.payload.get("phase") != "RELEASE_REQUESTED":
        raise TasteMainV2AuthorityError("release activation phase changed")
    receipt = _HeldFile(Path(controller_receipt_path), label="controller receipt")
    lease = _HeldFile(Path(lease_path), label="GPU lease request")
    try:
        receipt_payload = receipt.json()
        _validate_receipt(
            receipt_payload,
            expected_controller_id=None,
            expected_git_commit=None,
            expected_git_tree=None,
        )
        _validate_controller_namespace_binding(receipt.path, receipt_payload)
        if lease.sha256 != expected_lease_sha256:
            raise TasteMainV2AuthorityError("release lease SHA changed")
        lease_payload = _validate_lease(
            lease,
            receipt_payload=receipt_payload,
            receipt_sha256=receipt.sha256,
            now_ns=time.time_ns(),
            allow_expired=True,
        )
        if release_activation.payload.get("lease_uuid") != lease_payload["lease_uuid"]:
            raise TasteMainV2AuthorityError("release activation lease changed")
        expected_path = (
            receipt.path.parent
            / GPU_LEASE_RELEASE_DIRECTORY
            / (
                f"{lease_payload['lease_uuid']}-"
                f"{release_activation.payload['activation_uuid']}.json"
            )
        )
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if not expected_path.exists():
                time.sleep(0.25)
                continue
            try:
                ack = _validate_release_ack(
                    expected_path,
                    receipt_payload=receipt_payload,
                    receipt_sha256=receipt.sha256,
                    lease_payload=lease_payload,
                    lease_sha256=lease.sha256,
                )
                if (
                    ack.get("activation_sha256") != release_activation.sha256
                    or ack.get("attempt_id")
                    != release_activation.payload["attempt_id"]
                    or ack.get("generation_token")
                    != release_activation.payload["generation_token"]
                ):
                    raise TasteMainV2AuthorityError(
                        "release acknowledgement generation changed"
                    )
                lock = _WORKER_GPU_LOCKS.pop(str(lease_payload["lease_uuid"]), None)
                if lock is None:
                    raise TasteMainV2AuthorityError(
                        "registered runner GPU lock is absent at release"
                    )
                lock.release()
                receipt.revalidate()
                lease.revalidate()
                return {
                    "release_path": str(expected_path),
                    "release_sha256": immutable_authority_sha256(
                        expected_path, label="GPU lease release acknowledgement"
                    ),
                    "lease_uuid": lease_payload["lease_uuid"],
                    "state": "RELEASED",
                    "signal_sent": False,
                }
            except FileNotFoundError:
                time.sleep(0.25)
        raise TasteMainV2AuthorityError(
            "controller did not acknowledge clean GPU release within 60 seconds"
        )
    finally:
        lease.close()
        receipt.close()


def _active_leases(
    controller_root: Path,
    *,
    receipt_payload: Mapping[str, Any],
    receipt_sha256: str,
    now_ns: int,
    gpu_inventory: Mapping[int, str] | None,
    snapshot_reader: SnapshotReader,
) -> list[dict[str, Any]]:
    lease_dir = _require_absolute_physical(
        controller_root / GPU_LEASE_DIRECTORY, label="GPU lease directory"
    )
    activation_dir = _require_absolute_physical(
        controller_root / GPU_LEASE_ACTIVATION_DIRECTORY,
        label="GPU lease activation directory",
    )
    release_dir = _require_absolute_physical(
        controller_root / GPU_LEASE_RELEASE_DIRECTORY,
        label="GPU lease release directory",
    )
    activation_names: dict[str, list[str]] = {}
    for entry in os.scandir(activation_dir):
        matched = _ACTIVATION_NAME.fullmatch(entry.name)
        if matched is None:
            raise TasteMainV2AuthorityError("unexpected GPU activation directory entry")
        activation_names.setdefault(matched.group("lease"), []).append(entry.name)
    release_names: dict[str, list[str]] = {}
    for entry in os.scandir(release_dir):
        matched = _RELEASE_NAME.fullmatch(entry.name)
        if matched is None:
            raise TasteMainV2AuthorityError("unexpected GPU release directory entry")
        release_names.setdefault(matched.group("lease"), []).append(entry.name)
    active: list[dict[str, Any]] = []
    reserved_indices: set[int] = set()
    held_files: list[_HeldFile] = []
    try:
        for entry in sorted(os.scandir(lease_dir), key=lambda item: item.name):
            if not _LEASE_NAME.fullmatch(entry.name):
                raise TasteMainV2AuthorityError("unexpected GPU lease directory entry")
            held = _HeldFile(lease_dir / entry.name, label="GPU lease")
            held_files.append(held)
            payload = _validate_lease(
                held,
                receipt_payload=receipt_payload,
                receipt_sha256=receipt_sha256,
                now_ns=now_ns,
                allow_expired=True,
            )
            releases = release_names.get(payload["lease_uuid"], [])
            if len(releases) > 1:
                raise TasteMainV2AuthorityError("GPU lease has multiple release acknowledgements")
            if releases:
                _validate_release_ack(
                    release_dir / releases[0],
                    receipt_payload=receipt_payload,
                    receipt_sha256=receipt_sha256,
                    lease_payload=payload,
                    lease_sha256=held.sha256,
                )
                continue
            names = activation_names.get(payload["lease_uuid"], [])
            if not names:
                if now_ns >= payload["expires_at_ns"]:
                    continue
            index = int(payload["physical_gpu_index"])
            if index in reserved_indices:
                raise TasteMainV2AuthorityError(
                    "duplicate physical GPU lease requested"
                )
            reserved_indices.add(index)
            if not names:
                continue
            names.sort(key=lambda name: int(_ACTIVATION_NAME.fullmatch(name).group("sequence")))  # type: ignore[union-attr]
            previous_activation_sha256: str | None = None
            activation: _HeldFile | None = None
            activation_payload: dict[str, Any] | None = None
            activation_chain: list[dict[str, Any]] = []
            expected_phases = (
                "WORKER_ACTIVE",
                "WAITING_VERIFIER",
                "VERIFIER_ACTIVE",
                "RELEASE_REQUESTED",
            )
            chain_attempt_id: str | None = None
            chain_generation_token: str | None = None
            chain_worker: ProcessSnapshotV2 | None = None
            for activation_sequence, name in enumerate(names, start=1):
                if activation_sequence > len(expected_phases):
                    raise TasteMainV2AuthorityError(
                        "GPU activation chain has too many phases"
                    )
                candidate = _HeldFile(
                    activation_dir / name, label="GPU lease activation"
                )
                try:
                    candidate_payload = _validate_activation(
                        candidate,
                        lease_payload=payload,
                        lease_sha256=held.sha256,
                        receipt_payload=receipt_payload,
                        receipt_sha256=receipt_sha256,
                        snapshot_reader=snapshot_reader,
                        check_live=activation_sequence == len(names),
                    )
                    if (
                        candidate_payload["activation_sequence"]
                        != activation_sequence
                        or candidate_payload["previous_activation_sha256"]
                        != previous_activation_sha256
                    ):
                        raise TasteMainV2AuthorityError(
                            "GPU activation chain has a jump or bad predecessor"
                        )
                    candidate_worker = ProcessSnapshotV2.from_mapping(
                        candidate_payload["managed_worker"]
                    )
                    if candidate_payload["phase"] != expected_phases[activation_sequence - 1]:
                        raise TasteMainV2AuthorityError(
                            "GPU activation phase chain has a jump"
                        )
                    if activation_sequence == 1:
                        chain_attempt_id = candidate_payload["attempt_id"]
                        chain_generation_token = candidate_payload["generation_token"]
                        chain_worker = candidate_worker
                    elif (
                        candidate_payload["attempt_id"] != chain_attempt_id
                        or candidate_payload["generation_token"]
                        != chain_generation_token
                        or chain_worker is None
                        or not candidate_worker.same_runtime_identity(chain_worker)
                    ):
                        raise TasteMainV2AuthorityError(
                            "GPU activation attempt/runner generation changed"
                        )
                    activation_chain.append(candidate_payload)
                    previous_activation_sha256 = candidate.sha256
                    if activation_sequence == len(names):
                        activation = candidate
                        activation_payload = candidate_payload
                        candidate = None
                finally:
                    if candidate is not None:
                        candidate.close()
            if activation is None or activation_payload is None:
                raise TasteMainV2AuthorityError("GPU activation terminal is absent")
            held_files.append(activation)
            for historical in activation_chain[:-1]:
                child_raw = historical.get("training_child")
                if child_raw is None:
                    continue
                child = ProcessSnapshotV2.from_mapping(child_raw)
                try:
                    observed_child = snapshot_reader(child.pid)
                except (FileNotFoundError, ProcessLookupError):
                    continue
                except OSError as exc:
                    if exc.errno in {errno.ENOENT, errno.ESRCH}:
                        continue
                    raise TasteMainV2AuthorityError(
                        "historical science child liveness is indeterminate"
                    ) from exc
                if child.same_runtime_identity(observed_child):
                    raise TasteMainV2AuthorityError(
                        "prior science child remains live across an activation phase"
                    )
            worker = ProcessSnapshotV2.from_mapping(
                activation_payload["managed_worker"]
            )
            (
                effective_expires_at_ns,
                renewal_payload,
                renewal_path,
                renewal_sha256,
            ) = _renewal_chain(
                controller_root,
                lease_payload=payload,
                lease_sha256=held.sha256,
                receipt_payload=receipt_payload,
                receipt_sha256=receipt_sha256,
                expected_worker=worker,
                snapshot_reader=snapshot_reader,
            )
            if renewal_payload is not None and (
                renewal_payload["attempt_id"] != activation_payload["attempt_id"]
                or renewal_payload["generation_token"]
                != activation_payload["generation_token"]
            ):
                raise TasteMainV2AuthorityError("GPU renewal generation changed")
            if now_ns >= effective_expires_at_ns:
                raise TasteMainV2AuthorityError(
                    "activated GPU lease expired while its runner may remain live"
                )
            index = payload["physical_gpu_index"]
            if gpu_inventory is None:
                gpu_inventory = probe_physical_gpus()
            if gpu_inventory.get(index) != payload["physical_gpu_uuid"]:
                raise TasteMainV2AuthorityError(
                    "nvidia-smi physical index/UUID differs from lease"
                )
            gpu_lock = _ensure_gpu_lock(
                controller_root=controller_root,
                receipt_payload=receipt_payload,
                receipt_sha256=receipt_sha256,
                lease_payload=payload,
                lease_sha256=held.sha256,
                activation_payload=activation_payload,
                activation_path=activation.path,
                activation_sha256=activation.sha256,
            )
            if activation_payload["phase"] == "RELEASE_REQUESTED":
                _release_acknowledgement(
                    controller_root,
                    receipt_payload=receipt_payload,
                    receipt_sha256=receipt_sha256,
                    lease_payload=payload,
                    lease_sha256=held.sha256,
                    activation_payload=activation_payload,
                    activation_path=activation.path,
                    activation_sha256=activation.sha256,
                )
                _release_controller_coordination(
                    receipt_payload=receipt_payload,
                    lease_payload=payload,
                )
                continue
            active.append(
                {
                    "task_id": payload["task_id"],
                    "lease_uuid": payload["lease_uuid"],
                    "lease_path": str(held.path),
                    "lease_sha256": held.sha256,
                    "physical_gpu_index": payload["physical_gpu_index"],
                    "physical_gpu_uuid": payload["physical_gpu_uuid"],
                    "expires_at_ns": effective_expires_at_ns,
                    "base_expires_at_ns": payload["expires_at_ns"],
                    "renewal_path": str(renewal_path) if renewal_path else None,
                    "renewal_sha256": renewal_sha256,
                    "renewal_sequence": (
                        renewal_payload["sequence"] if renewal_payload else 0
                    ),
                    "state": "ACTIVE",
                    "activation_path": str(activation.path),
                    "activation_sha256": activation.sha256,
                    "activation_uuid": activation_payload["activation_uuid"],
                    "activation_sequence": activation_payload[
                        "activation_sequence"
                    ],
                    "previous_activation_sha256": activation_payload[
                        "previous_activation_sha256"
                    ],
                    "phase": activation_payload["phase"],
                    "attempt_id": activation_payload["attempt_id"],
                    "generation_token": activation_payload["generation_token"],
                    "process_lineage": {
                        name: activation_payload[name]
                        for name in (
                            "registration_writer",
                            "managed_launcher",
                            "managed_worker",
                            "training_child",
                        )
                    },
                    "gpu_lock_path": gpu_lock["path"],
                    "gpu_lock_sha256": gpu_lock["sha256"],
                    "shared_gpu_lock_path": gpu_lock["shared_path"],
                    "shared_gpu_lock_sha256": gpu_lock["shared_sha256"],
                }
            )
        if len(active) > MAX_ACTIVE_TASKS:
            raise TasteMainV2AuthorityError("more than two Taste tasks requested")
        indices = [item["physical_gpu_index"] for item in active]
        if len(set(indices)) != len(indices):
            raise TasteMainV2AuthorityError("duplicate physical GPU lease requested")
        for held in held_files:
            held.revalidate()
        return active
    finally:
        for held in held_files:
            held.close()


def probe_physical_gpus() -> dict[int, str]:
    """Read the controller-visible physical GPU index/UUID mapping."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TasteMainV2AuthorityError("nvidia-smi GPU inventory is unavailable") from exc
    result: dict[int, str] = {}
    for line in completed.stdout.splitlines():
        left, separator, right = line.partition(",")
        if not separator:
            raise TasteMainV2AuthorityError("nvidia-smi GPU inventory is malformed")
        try:
            index = int(left.strip())
        except ValueError as exc:
            raise TasteMainV2AuthorityError("nvidia-smi GPU index is malformed") from exc
        gpu_uuid = right.strip()
        if index in result or not _GPU_UUID.fullmatch(gpu_uuid):
            raise TasteMainV2AuthorityError("nvidia-smi GPU UUID mapping is malformed")
        result[index] = gpu_uuid
    if set(result) != {0, 1, 2, 3}:
        raise TasteMainV2AuthorityError("controller requires exact physical GPUs 0..3")
    return result


def _ensure_gpu_lock(
    *,
    controller_root: Path,
    receipt_payload: Mapping[str, Any],
    receipt_sha256: str,
    lease_payload: Mapping[str, Any],
    lease_sha256: str,
    activation_payload: Mapping[str, Any],
    activation_path: Path,
    activation_sha256: str,
) -> dict[str, str]:
    index = lease_payload["physical_gpu_index"]
    lease_uuid = str(lease_payload["lease_uuid"])
    ownership_path = (
        controller_root
        / GPU_LOCK_DIRECTORY
        / f"gpu{index}-{lease_uuid}-{activation_payload['activation_uuid']}.json"
    )
    lock_root = _runtime_layout_from_receipt(receipt_payload).locks_dir
    lock_root.mkdir(mode=0o700, exist_ok=True)
    if lock_root.is_symlink() or lock_root.resolve(strict=True) != lock_root:
        raise TasteMainV2AuthorityError("shared GPU lock root is not physical")
    coordination_uuid = (
        f"{lease_payload['physical_gpu_uuid']}.taste-main-v2-controller"
    )
    coordination = GPUFileLock(
        lock_root,
        gpu_index=index,
        gpu_uuid=coordination_uuid,
        owner={
            "controller_id": receipt_payload["controller_id"],
            "controller_receipt_sha256": receipt_sha256,
            "lease_uuid": lease_uuid,
            "coordination_for_gpu_uuid": lease_payload["physical_gpu_uuid"],
        },
    )
    lock_key = str(coordination.path)
    retained = _GPU_LOCKS.get(lock_key)
    if retained is None:
        try:
            coordination.acquire()
        except GPULockError as exc:
            raise TasteMainV2AuthorityError(
                "Taste controller GPU coordination lock is already owned"
            ) from exc
        _GPU_LOCKS[lock_key] = (coordination, lease_uuid)
    else:
        coordination, retained_lease_uuid = retained
        if retained_lease_uuid != lease_uuid:
            raise TasteMainV2AuthorityError(
                "shared GPU UUID lock remains reserved by another live/quarantined lease"
            )
    shared_path = Path(activation_payload["worker_gpu_lock_path"])
    shared_held = _HeldFile(shared_path, label="managed runner GPU UUID lock")
    try:
        shared_payload = shared_held.json()
        if (
            shared_payload.get("state") != "LOCKED"
            or shared_payload.get("gpu_index") != index
            or shared_payload.get("gpu_uuid") != lease_payload["physical_gpu_uuid"]
            or shared_payload.get("controller_id") != receipt_payload["controller_id"]
            or shared_payload.get("lease_uuid") != lease_uuid
            or shared_payload.get("attempt_id") != activation_payload["attempt_id"]
            or shared_payload.get("generation_token")
            != activation_payload["generation_token"]
        ):
            raise TasteMainV2AuthorityError("shared project GPU lock metadata changed")
        shared_sha256 = shared_held.sha256
    finally:
        shared_held.close()
    payload = {
        "schema_version": GPU_LOCK_SCHEMA,
        "controller_id": receipt_payload["controller_id"],
        "controller_receipt_sha256": receipt_sha256,
        "physical_gpu_index": index,
        "physical_gpu_uuid": lease_payload["physical_gpu_uuid"],
        "task_id": lease_payload["task_id"],
        "lease_uuid": lease_uuid,
        "lease_sha256": lease_sha256,
        "activation_uuid": activation_payload["activation_uuid"],
        "activation_path": str(activation_path),
        "activation_sha256": activation_sha256,
        "attempt_id": activation_payload["attempt_id"],
        "generation_token": activation_payload["generation_token"],
        "shared_gpu_lock_path": str(shared_path),
        "shared_gpu_lock_sha256": shared_sha256,
        "state": "ACTIVE",
        "created_at": _utc_now(),
        "auto_terminate_uncontrolled_children": False,
    }
    if not ownership_path.exists():
        _publish_immutable(
            ownership_path,
            _json_bytes(payload),
            staging_root=controller_root / PUBLICATION_STAGING_DIRECTORY,
        )
    held = _HeldFile(ownership_path, label="controller GPU ownership generation")
    try:
        actual = held.json()
        stable_fields = set(payload) - {"created_at"}
        if any(actual.get(field) != payload[field] for field in stable_fields):
            raise TasteMainV2AuthorityError("controller GPU lock is owned by another task")
        return {
            "path": str(ownership_path),
            "sha256": held.sha256,
            "shared_path": str(shared_path),
            "shared_sha256": shared_sha256,
        }
    finally:
        held.close()


@dataclass(frozen=True, slots=True)
class HeartbeatCreation:
    path: Path
    sha256: str
    sequence: int
    heartbeat_uuid: str
    payload: dict[str, Any]


def write_heartbeat_generation(
    *,
    controller_receipt_path: str | Path,
    sequence: int,
    previous_heartbeat_sha256: str | None,
    controller_state: str = "WAITING_DEPENDENCIES",
    heartbeat_uuid: str | None = None,
    now_ns: int | None = None,
    gpu_inventory: Mapping[int, str] | None = None,
    snapshot_reader: SnapshotReader = capture_process_snapshot,
) -> HeartbeatCreation:
    """Publish exactly one never-reused heartbeat generation."""

    require_auto_termination_disabled()
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
        raise TasteMainV2AuthorityError("heartbeat sequence must be positive")
    if sequence == 1 and previous_heartbeat_sha256 is not None:
        raise TasteMainV2AuthorityError("first heartbeat cannot have a predecessor")
    if sequence > 1:
        _validate_sha256(previous_heartbeat_sha256, label="previous heartbeat")
    if controller_state not in {"WAITING_DEPENDENCIES", "MONITORING", "QUARANTINED"}:
        raise TasteMainV2AuthorityError("controller state is invalid")
    heartbeat_uuid = require_uuid4(
        heartbeat_uuid or str(uuid.uuid4()), label="heartbeat_uuid"
    )
    receipt = _HeldFile(Path(controller_receipt_path), label="controller receipt")
    try:
        receipt_payload = receipt.json()
        expected_snapshot = _validate_receipt(
            receipt_payload,
            expected_controller_id=None,
            expected_git_commit=None,
            expected_git_tree=None,
        )
        _validate_controller_namespace_binding(receipt.path, receipt_payload)
        if expected_snapshot.pid != os.getpid():
            raise TasteMainV2AuthorityError(
                "only the immutable controller process may publish heartbeats"
            )
        observed = snapshot_reader(expected_snapshot.pid)
        if not expected_snapshot.same_runtime_identity(observed):
            raise TasteMainV2AuthorityError("controller process generation drifted")
        timestamp_ns = time.time_ns() if now_ns is None else now_ns
        if controller_state == "QUARANTINED":
            active_tasks: list[dict[str, Any]] = []
        else:
            active_tasks = _active_leases(
                receipt.path.parent,
                receipt_payload=receipt_payload,
                receipt_sha256=receipt.sha256,
                now_ns=timestamp_ns,
                gpu_inventory=(dict(gpu_inventory) if gpu_inventory is not None else None),
                snapshot_reader=snapshot_reader,
            )
        effective_state = (
            "MONITORING" if active_tasks and controller_state != "QUARANTINED" else controller_state
        )
        payload = {
            "schema_version": HEARTBEAT_SCHEMA,
            "managed_taste_release_version": MANAGED_TASTE_RELEASE_VERSION,
            "heartbeat_uuid": heartbeat_uuid,
            "sequence": sequence,
            "controller_id": receipt_payload["controller_id"],
            "controller_uuid": receipt_payload["controller_uuid"],
            **_snapshot_fields(expected_snapshot),
            "git_commit": receipt_payload["git_commit"],
            "git_tree": receipt_payload["git_tree"],
            "receipt_sha256": receipt.sha256,
            "policy_facts_sha256": receipt_payload["policy_facts_sha256"],
            "previous_heartbeat_sha256": previous_heartbeat_sha256,
            "state": "RUNNING" if effective_state != "QUARANTINED" else "QUARANTINED",
            "controller_state": effective_state,
            "active_tasks": active_tasks,
            "heartbeat_at": _utc_now(),
            "heartbeat_at_ns": timestamp_ns,
            "auto_terminate_uncontrolled_children": False,
            "signal_authority": False,
        }
        directory = _require_absolute_physical(
            receipt.path.parent / HEARTBEAT_DIRECTORY,
            label="heartbeat directory",
        )
        path = directory / f"{sequence:020d}-{heartbeat_uuid}.json"
        data = _json_bytes(payload)
        _publish_immutable(
            path,
            data,
            staging_root=receipt.path.parent / PUBLICATION_STAGING_DIRECTORY,
        )
        receipt.revalidate()
        return HeartbeatCreation(path, _sha256(data), sequence, heartbeat_uuid, payload)
    finally:
        receipt.close()


def _validate_heartbeat(
    payload: Mapping[str, Any],
    *,
    path: Path,
    receipt_payload: Mapping[str, Any],
    receipt_sha256: str,
    max_age_seconds: float,
    now_ns: int,
    check_freshness: bool = True,
) -> ProcessSnapshotV2:
    matched = _HEARTBEAT_NAME.fullmatch(path.name)
    if matched is None:
        raise TasteMainV2AuthorityError("heartbeat generation filename is malformed")
    if (
        payload.get("schema_version") != HEARTBEAT_SCHEMA
        or payload.get("managed_taste_release_version")
        != MANAGED_TASTE_RELEASE_VERSION
        or payload.get("heartbeat_uuid") != matched.group("uuid")
        or payload.get("sequence") != int(matched.group("sequence"))
        or payload.get("controller_id") != receipt_payload.get("controller_id")
        or payload.get("controller_uuid") != receipt_payload.get("controller_uuid")
        or payload.get("git_commit") != receipt_payload.get("git_commit")
        or payload.get("git_tree") != receipt_payload.get("git_tree")
        or payload.get("receipt_sha256") != receipt_sha256
        or payload.get("policy_facts_sha256")
        != receipt_payload.get("policy_facts_sha256")
        or payload.get("state") != "RUNNING"
        or payload.get("controller_state") not in {"WAITING_DEPENDENCIES", "MONITORING"}
        or payload.get("auto_terminate_uncontrolled_children") is not False
        or payload.get("signal_authority") is not False
    ):
        raise TasteMainV2AuthorityError("heartbeat authority binding changed")
    timestamp_ns = payload.get("heartbeat_at_ns")
    if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
        raise TasteMainV2AuthorityError("heartbeat timestamp is malformed")
    if check_freshness:
        age_ns = now_ns - timestamp_ns
        if age_ns < -5_000_000_000 or age_ns > int(max_age_seconds * 1_000_000_000):
            raise TasteMainV2AuthorityError("terminal heartbeat generation is stale")
    if timestamp_ns < int(receipt_payload["created_at_ns"]):
        raise TasteMainV2AuthorityError("heartbeat predates its controller receipt")
    previous = payload.get("previous_heartbeat_sha256")
    if payload["sequence"] == 1:
        if previous is not None:
            raise TasteMainV2AuthorityError("first heartbeat predecessor changed")
    else:
        _validate_sha256(previous, label="previous heartbeat")
    snapshot = _snapshot_from_payload(payload)
    receipt_snapshot = _snapshot_from_payload(receipt_payload)
    if not receipt_snapshot.same_runtime_identity(snapshot):
        raise TasteMainV2AuthorityError("heartbeat process differs from receipt")
    active = payload.get("active_tasks")
    if not isinstance(active, list) or len(active) > MAX_ACTIVE_TASKS:
        raise TasteMainV2AuthorityError("heartbeat active task list is malformed")
    indices: list[int] = []
    lease_ids: set[str] = set()
    for task in active:
        if type(task) is not dict:
            raise TasteMainV2AuthorityError("heartbeat active task is malformed")
        if not _TASK_ID.fullmatch(str(task.get("task_id"))):
            raise TasteMainV2AuthorityError("heartbeat task id is malformed")
        require_uuid4(task.get("lease_uuid"), label="heartbeat lease_uuid")
        _validate_sha256(task.get("lease_sha256"), label="heartbeat lease")
        if task.get("state") != "ACTIVE":
            raise TasteMainV2AuthorityError("heartbeat GPU lease is not ACTIVE")
        require_uuid4(task.get("activation_uuid"), label="heartbeat activation_uuid")
        sequence = task.get("activation_sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            raise TasteMainV2AuthorityError("heartbeat activation sequence is malformed")
        if task.get("phase") not in {
            "WORKER_ACTIVE",
            "WAITING_VERIFIER",
            "VERIFIER_ACTIVE",
        }:
            raise TasteMainV2AuthorityError("heartbeat activation phase is malformed")
        predecessor = task.get("previous_activation_sha256")
        if sequence == 1:
            if predecessor is not None:
                raise TasteMainV2AuthorityError(
                    "first heartbeat activation has a predecessor"
                )
        else:
            _validate_sha256(predecessor, label="heartbeat activation predecessor")
        require_uuid4(task.get("attempt_id"), label="heartbeat attempt_id")
        require_uuid4(task.get("generation_token"), label="heartbeat generation_token")
        _validate_sha256(task.get("activation_sha256"), label="heartbeat activation")
        _validate_sha256(task.get("gpu_lock_sha256"), label="heartbeat GPU lock")
        _validate_sha256(
            task.get("shared_gpu_lock_sha256"), label="shared GPU lock"
        )
        expires = task.get("expires_at_ns")
        base_expires = task.get("base_expires_at_ns")
        if (
            isinstance(expires, bool)
            or not isinstance(expires, int)
            or isinstance(base_expires, bool)
            or not isinstance(base_expires, int)
            or expires < base_expires
        ):
            raise TasteMainV2AuthorityError("heartbeat lease expiry is malformed")
        renewal_sequence = task.get("renewal_sequence")
        if isinstance(renewal_sequence, bool) or not isinstance(renewal_sequence, int) or renewal_sequence < 0:
            raise TasteMainV2AuthorityError("heartbeat renewal sequence is malformed")
        if renewal_sequence == 0:
            if task.get("renewal_path") is not None or task.get("renewal_sha256") is not None:
                raise TasteMainV2AuthorityError("heartbeat has a spurious renewal")
        else:
            if not Path(str(task.get("renewal_path"))).is_absolute():
                raise TasteMainV2AuthorityError("heartbeat renewal path is malformed")
            _validate_sha256(task.get("renewal_sha256"), label="heartbeat renewal")
        for field in (
            "lease_path",
            "activation_path",
            "gpu_lock_path",
            "shared_gpu_lock_path",
        ):
            if not Path(str(task.get(field))).is_absolute():
                raise TasteMainV2AuthorityError(f"heartbeat {field} is malformed")
        lineage = task.get("process_lineage")
        if type(lineage) is not dict or type(lineage.get("managed_worker")) is not dict:
            raise TasteMainV2AuthorityError("heartbeat process lineage is absent")
        index = task.get("physical_gpu_index")
        if isinstance(index, bool) or index not in ALLOWED_GPU_INDICES or index in PROTECTED_GPU_INDICES:
            raise TasteMainV2AuthorityError("heartbeat includes an unauthorized GPU")
        if TASK_GPU_BINDINGS.get(str(task.get("task_id"))) != index:
            raise TasteMainV2AuthorityError("heartbeat task/GPU binding changed")
        if not _GPU_UUID.fullmatch(str(task.get("physical_gpu_uuid"))):
            raise TasteMainV2AuthorityError("heartbeat GPU UUID is malformed")
        if task["lease_uuid"] in lease_ids:
            raise TasteMainV2AuthorityError("heartbeat repeats a GPU lease")
        lease_ids.add(task["lease_uuid"])
        indices.append(index)
    if len(indices) != len(set(indices)):
        raise TasteMainV2AuthorityError("heartbeat repeats a physical GPU")
    return snapshot


def _heartbeat_generation_paths(controller_root: str | Path) -> list[Path]:
    root = _require_absolute_physical(Path(controller_root), label="controller root")
    directory = _require_absolute_physical(
        root / HEARTBEAT_DIRECTORY, label="heartbeat directory"
    )
    names: list[str] = []
    for entry in os.scandir(directory):
        if _HEARTBEAT_NAME.fullmatch(entry.name) is None:
            raise TasteMainV2AuthorityError("unexpected heartbeat directory entry")
        names.append(entry.name)
    names.sort()
    if not names:
        raise TasteMainV2AuthorityError("controller has no heartbeat generation")
    sequences = [int(_HEARTBEAT_NAME.fullmatch(name).group("sequence")) for name in names]  # type: ignore[union-attr]
    if len(sequences) != len(set(sequences)):
        raise TasteMainV2AuthorityError("heartbeat sequence is reused")
    return [directory / name for name in names]


def latest_heartbeat_path(controller_root: str | Path) -> Path:
    return _heartbeat_generation_paths(controller_root)[-1]


def initial_heartbeat_path(controller_root: str | Path) -> Path:
    """Return the immutable sequence-1 trust anchor, never the moving terminal."""

    path = _heartbeat_generation_paths(controller_root)[0]
    matched = _HEARTBEAT_NAME.fullmatch(path.name)
    if matched is None or int(matched.group("sequence")) != 1:
        raise TasteMainV2AuthorityError("heartbeat chain does not start at sequence 1")
    return path


SnapshotReader = Callable[[int], ProcessSnapshotV2]


class HeldTasteMainV2ControllerAuthority:
    """Held external receipt plus full-chain terminal controller authority."""

    def __init__(
        self,
        receipt_path: str | Path,
        heartbeat_path: str | Path,
        expected_controller_id: str | None,
        expected_git_commit: str | None,
        expected_git_tree: str | None,
        max_age_seconds: float,
        *,
        expected_launcher_receipt_path: str | Path | None = None,
        expected_launcher_receipt_sha256: str | None = None,
        expected_receipt_sha256: str | None = None,
        expected_heartbeat_sha256: str | None = None,
        expected_task_id: str | None = None,
        expected_gpu_index: int | None = None,
        expected_gpu_uuid: str | None = None,
        expected_lease_uuid: str | None = None,
        expected_lease_sha256: str | None = None,
        expected_attempt_id: str | None = None,
        expected_generation_token: str | None = None,
        expected_activation_phase: str | None = None,
        expected_worker_process: ProcessSnapshotV2 | None = None,
        snapshot_reader: SnapshotReader = capture_process_snapshot,
        now_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        require_auto_termination_disabled()
        if max_age_seconds < 10 or max_age_seconds > 300:
            raise TasteMainV2AuthorityError("heartbeat max age must be within 10..300 seconds")
        self.max_age_seconds = float(max_age_seconds)
        self.snapshot_reader = snapshot_reader
        self.now_ns = now_ns
        self.receipt = _HeldFile(Path(receipt_path), label="controller receipt")
        self.launcher_receipt: _HeldFile | None = None
        self.terminal_heartbeat: _HeldFile | None = None
        self.lease: _HeldFile | None = None
        self.activation: _HeldFile | None = None
        self.gpu_lock: _HeldFile | None = None
        self.shared_gpu_lock: _HeldFile | None = None
        self.anchor_heartbeat_path = Path(heartbeat_path)
        self.expected_anchor_heartbeat_sha256 = expected_heartbeat_sha256
        self.expected_task_id = expected_task_id
        self.expected_gpu_index = expected_gpu_index
        self.expected_gpu_uuid = expected_gpu_uuid
        self.expected_lease_uuid = expected_lease_uuid
        self.expected_lease_sha256 = expected_lease_sha256
        self.expected_attempt_id = expected_attempt_id
        self.expected_generation_token = expected_generation_token
        self.expected_activation_phase = expected_activation_phase
        self.expected_worker_process = expected_worker_process
        _validate_sha256(expected_receipt_sha256, label="expected controller receipt")
        _validate_sha256(
            expected_heartbeat_sha256, label="expected anchor heartbeat"
        )
        gpu_pins = (
            expected_gpu_index,
            expected_gpu_uuid,
            expected_lease_uuid,
            expected_lease_sha256,
            expected_attempt_id,
            expected_generation_token,
            expected_activation_phase,
            expected_worker_process,
        )
        if self.expected_task_id is None:
            if any(value is not None for value in gpu_pins):
                raise TasteMainV2AuthorityError(
                    "no-lease authority cannot carry GPU/task-generation pins"
                )
        else:
            if any(
                value is None
                for value in (
                    expected_gpu_index,
                    expected_gpu_uuid,
                    expected_lease_uuid,
                    expected_lease_sha256,
                    expected_attempt_id,
                    expected_generation_token,
                    expected_activation_phase,
                )
            ):
                raise TasteMainV2AuthorityError(
                    "GPU authority requires complete task/GPU/lease/generation/phase pins"
                )
            require_uuid4(expected_attempt_id, label="expected attempt_id")
            require_uuid4(expected_generation_token, label="expected generation_token")
        try:
            if expected_receipt_sha256 is not None and self.receipt.sha256 != expected_receipt_sha256:
                raise TasteMainV2AuthorityError("controller receipt SHA differs from authority")
            self.receipt_payload = self.receipt.json()
            self.receipt_snapshot = _validate_receipt(
                self.receipt_payload,
                expected_controller_id=expected_controller_id,
                expected_git_commit=expected_git_commit,
                expected_git_tree=expected_git_tree,
            )
            _validate_controller_namespace_binding(
                self.receipt.path, self.receipt_payload
            )
            if self.receipt.path.name != CONTROLLER_RECEIPT_NAME:
                raise TasteMainV2AuthorityError("controller receipt filename changed")
            receipt_launcher_path = Path(self.receipt_payload["launcher_receipt_path"])
            if (
                expected_launcher_receipt_path is None
                or expected_launcher_receipt_sha256 is None
                or receipt_launcher_path != Path(expected_launcher_receipt_path)
                or self.receipt_payload["launcher_receipt_sha256"]
                != expected_launcher_receipt_sha256
            ):
                raise TasteMainV2AuthorityError(
                    "external launcher path/SHA is not explicitly pinned"
                )
            self.launcher_receipt = _HeldFile(
                receipt_launcher_path, label="external launcher receipt"
            )
            if self.launcher_receipt.sha256 != expected_launcher_receipt_sha256:
                raise TasteMainV2AuthorityError("external launcher receipt SHA changed")
            launcher_payload = self.launcher_receipt.json()
            launched = _validate_launcher_receipt(
                launcher_payload,
                expected_controller_id=expected_controller_id,
                expected_git_commit=expected_git_commit,
                expected_git_tree=expected_git_tree,
            )
            if not launched.same_runtime_identity(self.receipt_snapshot):
                raise TasteMainV2AuthorityError(
                    "controller receipt differs from external launcher target"
                )
            if launcher_payload.get("project_root") != self.receipt_payload.get(
                "project_root"
            ):
                raise TasteMainV2AuthorityError(
                    "controller project root differs from external launcher"
                )
            if int(launcher_payload["created_at_ns"]) >= int(
                self.receipt_payload["created_at_ns"]
            ):
                raise TasteMainV2AuthorityError(
                    "controller receipt does not follow its external launcher"
                )
            if self.anchor_heartbeat_path.parent != self.receipt.path.parent / HEARTBEAT_DIRECTORY:
                raise TasteMainV2AuthorityError("heartbeat is outside controller authority")
            self._scan_full_chain()
        except BaseException:
            self.close()
            raise

    @property
    def current_heartbeat(self) -> _HeldFile:
        if self.terminal_heartbeat is None:
            raise TasteMainV2AuthorityError("terminal heartbeat is not held")
        return self.terminal_heartbeat

    def _scan_full_chain(self) -> None:
        """Validate seq1..terminal; freshness/live checks apply only terminal."""

        paths = _heartbeat_generation_paths(self.receipt.path.parent)
        if len(paths) > 100_000:
            raise TasteMainV2AuthorityError("heartbeat chain exceeds its audit bound")
        previous_sha256: str | None = None
        anchor_seen = False
        anchor_sequence: int | None = None
        chain_rows: list[dict[str, Any]] = []
        new_terminal: _HeldFile | None = None
        try:
            for offset, path in enumerate(paths, start=1):
                held = _HeldFile(path, label="controller heartbeat")
                keep = offset == len(paths)
                try:
                    payload = held.json()
                    snapshot = _validate_heartbeat(
                        payload,
                        path=path,
                        receipt_payload=self.receipt_payload,
                        receipt_sha256=self.receipt.sha256,
                        max_age_seconds=self.max_age_seconds,
                        now_ns=self.now_ns(),
                        check_freshness=keep,
                    )
                    if (
                        payload["sequence"] != offset
                        or payload["previous_heartbeat_sha256"] != previous_sha256
                    ):
                        raise TasteMainV2AuthorityError(
                            "heartbeat chain has a jump or bad predecessor"
                        )
                    if path == self.anchor_heartbeat_path:
                        if (
                            self.expected_anchor_heartbeat_sha256 is not None
                            and held.sha256
                            != self.expected_anchor_heartbeat_sha256
                        ):
                            raise TasteMainV2AuthorityError(
                                "historical anchor heartbeat SHA changed"
                            )
                        anchor_seen = True
                        anchor_sequence = offset
                    chain_rows.append(
                        {
                            "sequence": offset,
                            "name": path.name,
                            "sha256": held.sha256,
                        }
                    )
                    previous_sha256 = held.sha256
                    if keep:
                        self.heartbeat_payload = payload
                        self.heartbeat_snapshot = snapshot
                        new_terminal = held
                finally:
                    if not keep:
                        held.close()
            if not anchor_seen:
                raise TasteMainV2AuthorityError(
                    "declared historical heartbeat is outside the full chain"
                )
            if anchor_sequence is None:
                raise TasteMainV2AuthorityError(
                    "declared historical heartbeat sequence is absent"
                )
            if new_terminal is None:
                raise TasteMainV2AuthorityError("terminal heartbeat is absent")
            old_terminal = self.terminal_heartbeat
            self.terminal_heartbeat = new_terminal
            new_terminal = None
            if old_terminal is not None:
                old_terminal.close()
            self.chain_closure_sha256 = _sha256(_json_bytes({"heartbeats": chain_rows}))
            self.chain_length = len(chain_rows)
            self.anchor_sequence = anchor_sequence
            self._validate_live_process()
            self._bind_expected_lease()
            self.receipt.revalidate()
            if self.launcher_receipt is None:
                raise TasteMainV2AuthorityError("external launcher receipt is absent")
            self.launcher_receipt.revalidate()
            self._refresh_evidence()
        finally:
            if new_terminal is not None:
                new_terminal.close()

    def _validate_live_process(self) -> None:
        observed = self.snapshot_reader(self.receipt_snapshot.pid)
        if not self.receipt_snapshot.same_runtime_identity(observed):
            raise TasteMainV2AuthorityError("live controller process generation changed")

    def _matching_active_task(self) -> dict[str, Any] | None:
        if self.expected_task_id is None:
            return None
        matches = [
            item
            for item in self.heartbeat_payload["active_tasks"]
            if item.get("task_id") == self.expected_task_id
            and (self.expected_gpu_index is None or item.get("physical_gpu_index") == self.expected_gpu_index)
            and (self.expected_gpu_uuid is None or item.get("physical_gpu_uuid") == self.expected_gpu_uuid)
            and (self.expected_lease_uuid is None or item.get("lease_uuid") == self.expected_lease_uuid)
            and (self.expected_lease_sha256 is None or item.get("lease_sha256") == self.expected_lease_sha256)
            and item.get("attempt_id") == self.expected_attempt_id
            and item.get("generation_token") == self.expected_generation_token
            and (
                self.expected_activation_phase is None
                or item.get("phase") == self.expected_activation_phase
            )
        ]
        if len(matches) != 1:
            raise TasteMainV2AuthorityError("expected GPU lease is not acknowledged")
        return matches[0]

    def _bind_expected_lease(self) -> None:
        task = self._matching_active_task()
        if task is None:
            return
        path = Path(task["lease_path"])
        expected_directory = self.receipt.path.parent / GPU_LEASE_DIRECTORY
        if path.parent != expected_directory:
            raise TasteMainV2AuthorityError("acknowledged lease is outside controller root")
        held = _HeldFile(path, label="acknowledged GPU lease")
        if held.sha256 != task["lease_sha256"]:
            held.close()
            raise TasteMainV2AuthorityError("acknowledged GPU lease SHA changed")
        _validate_lease(
            held,
            receipt_payload=self.receipt_payload,
            receipt_sha256=self.receipt.sha256,
            now_ns=self.now_ns(),
            allow_expired=True,
        )
        if self.now_ns() >= int(task["expires_at_ns"]):
            held.close()
            raise TasteMainV2AuthorityError("acknowledged effective GPU lease expired")
        activation_path = Path(task["activation_path"])
        if activation_path.parent != self.receipt.path.parent / GPU_LEASE_ACTIVATION_DIRECTORY:
            held.close()
            raise TasteMainV2AuthorityError("activation is outside controller root")
        activation = _HeldFile(activation_path, label="acknowledged GPU activation")
        if activation.sha256 != task["activation_sha256"]:
            activation.close()
            held.close()
            raise TasteMainV2AuthorityError("acknowledged activation SHA changed")
        activation_payload = _validate_activation(
            activation,
            lease_payload=held.json(),
            lease_sha256=held.sha256,
            receipt_payload=self.receipt_payload,
            receipt_sha256=self.receipt.sha256,
            snapshot_reader=self.snapshot_reader,
        )
        worker = ProcessSnapshotV2.from_mapping(activation_payload["managed_worker"])
        if self.expected_worker_process is not None and not self.expected_worker_process.same_runtime_identity(worker):
            activation.close()
            held.close()
            raise TasteMainV2AuthorityError("ACTIVE lease worker generation changed")
        lock_path = Path(task["gpu_lock_path"])
        if lock_path.parent != self.receipt.path.parent / GPU_LOCK_DIRECTORY:
            activation.close()
            held.close()
            raise TasteMainV2AuthorityError("GPU lock is outside controller root")
        gpu_lock = _HeldFile(lock_path, label="controller-held GPU UUID lock")
        if gpu_lock.sha256 != task["gpu_lock_sha256"]:
            gpu_lock.close()
            activation.close()
            held.close()
            raise TasteMainV2AuthorityError("controller GPU lock SHA changed")
        lock_payload = gpu_lock.json()
        if (
            lock_payload.get("schema_version") != GPU_LOCK_SCHEMA
            or lock_payload.get("state") != "ACTIVE"
            or lock_payload.get("lease_sha256") != held.sha256
            or lock_payload.get("activation_sha256") != activation.sha256
            or lock_payload.get("physical_gpu_index") != self.expected_gpu_index
            or lock_payload.get("physical_gpu_uuid") != self.expected_gpu_uuid
            or lock_payload.get("attempt_id") != self.expected_attempt_id
            or lock_payload.get("generation_token") != self.expected_generation_token
            or lock_payload.get("shared_gpu_lock_path")
            != task.get("shared_gpu_lock_path")
            or lock_payload.get("shared_gpu_lock_sha256")
            != task.get("shared_gpu_lock_sha256")
        ):
            gpu_lock.close()
            activation.close()
            held.close()
            raise TasteMainV2AuthorityError("controller GPU lock binding changed")
        shared_gpu_lock = _HeldFile(
            Path(task["shared_gpu_lock_path"]),
            label="shared project GPU UUID lock",
        )
        if shared_gpu_lock.sha256 != task["shared_gpu_lock_sha256"]:
            shared_gpu_lock.close()
            gpu_lock.close()
            activation.close()
            held.close()
            raise TasteMainV2AuthorityError("shared GPU lock SHA changed")
        shared_payload = shared_gpu_lock.json()
        if (
            shared_payload.get("state") != "LOCKED"
            or shared_payload.get("gpu_index") != self.expected_gpu_index
            or shared_payload.get("gpu_uuid") != self.expected_gpu_uuid
            or shared_payload.get("controller_id")
            != self.receipt_payload["controller_id"]
            or shared_payload.get("lease_uuid") != self.expected_lease_uuid
            or shared_payload.get("attempt_id") != self.expected_attempt_id
            or shared_payload.get("generation_token")
            != self.expected_generation_token
        ):
            shared_gpu_lock.close()
            gpu_lock.close()
            activation.close()
            held.close()
            raise TasteMainV2AuthorityError("shared GPU lock metadata changed")
        if self.lease is not None:
            self.lease.close()
        if self.activation is not None:
            self.activation.close()
        if self.gpu_lock is not None:
            self.gpu_lock.close()
        if self.shared_gpu_lock is not None:
            self.shared_gpu_lock.close()
        self.lease = held
        self.activation = activation
        self.gpu_lock = gpu_lock
        self.shared_gpu_lock = shared_gpu_lock
        self._refresh_evidence()

    def _refresh_evidence(self) -> None:
        heartbeat = self.current_heartbeat
        payload = self.heartbeat_payload
        evidence: dict[str, Any] = {
            "controller_id": self.receipt_payload["controller_id"],
            "managed_taste_release_version": MANAGED_TASTE_RELEASE_VERSION,
            "controller_uuid": self.receipt_payload["controller_uuid"],
            "pid": self.receipt_snapshot.pid,
            "pid_start_ticks": self.receipt_snapshot.pid_start_ticks,
            "boot_id": self.receipt_snapshot.boot_id,
            "exe": self.receipt_snapshot.executable_realpath,
            "command_hash": self.receipt_snapshot.command_hash,
            "cwd": self.receipt_snapshot.cwd_realpath,
            "cgroup": self.receipt_snapshot.cgroup_path,
            "git_commit": self.receipt_payload["git_commit"],
            "git_tree": self.receipt_payload["git_tree"],
            "launcher_receipt_path": str(self.launcher_receipt.path),
            "launcher_receipt_sha256": self.launcher_receipt.sha256,
            "state": payload["state"],
            "controller_state": payload["controller_state"],
            "heartbeat_at": payload["heartbeat_at"],
            "heartbeat_at_ns": payload["heartbeat_at_ns"],
            "sequence": payload["sequence"],
            "receipt_path": str(self.receipt.path),
            "receipt_sha256": self.receipt.sha256,
            "heartbeat_path": str(heartbeat.path),
            "heartbeat_sha256": heartbeat.sha256,
            "heartbeat_uuid": payload["heartbeat_uuid"],
            "anchor_heartbeat_path": str(self.anchor_heartbeat_path),
            "anchor_heartbeat_sha256": self.expected_anchor_heartbeat_sha256,
            "anchor_heartbeat_sequence": self.anchor_sequence,
            "heartbeat_chain_length": self.chain_length,
            "heartbeat_chain_closure_sha256": self.chain_closure_sha256,
            "active_tasks": payload["active_tasks"],
        }
        if self.lease is not None:
            lease = self.lease.json()
            active_task = self._matching_active_task()
            if active_task is None:
                raise TasteMainV2AuthorityError("held GPU task disappeared")
            evidence.update(
                {
                    "lease_path": str(self.lease.path),
                    "lease_sha256": self.lease.sha256,
                    "lease_uuid": lease["lease_uuid"],
                    "task_id": lease["task_id"],
                    "physical_gpu_index": lease["physical_gpu_index"],
                    "physical_gpu_uuid": lease["physical_gpu_uuid"],
                    "activation_path": str(self.activation.path),
                    "activation_sha256": self.activation.sha256,
                    "activation_phase": active_task["phase"],
                    "activation_sequence": active_task["activation_sequence"],
                    "effective_expires_at_ns": active_task["expires_at_ns"],
                    "base_expires_at_ns": active_task["base_expires_at_ns"],
                    "renewal_path": active_task["renewal_path"],
                    "renewal_sha256": active_task["renewal_sha256"],
                    "renewal_sequence": active_task["renewal_sequence"],
                    "attempt_id": self.expected_attempt_id,
                    "generation_token": self.expected_generation_token,
                    "gpu_lock_path": str(self.gpu_lock.path),
                    "gpu_lock_sha256": self.gpu_lock.sha256,
                    "shared_gpu_lock_path": str(self.shared_gpu_lock.path),
                    "shared_gpu_lock_sha256": self.shared_gpu_lock.sha256,
                }
            )
        self.evidence = evidence

    def revalidate(self) -> dict[str, Any]:
        """Recheck held bytes, adopt a newer generation, and recapture /proc."""

        self._scan_full_chain()
        return dict(self.evidence)

    def __enter__(self) -> "HeldTasteMainV2ControllerAuthority":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self.lease is not None:
            self.lease.close()
            self.lease = None
        if self.activation is not None:
            self.activation.close()
            self.activation = None
        if self.gpu_lock is not None:
            self.gpu_lock.close()
            self.gpu_lock = None
        if self.shared_gpu_lock is not None:
            self.shared_gpu_lock.close()
            self.shared_gpu_lock = None
        if self.terminal_heartbeat is not None:
            self.terminal_heartbeat.close()
            self.terminal_heartbeat = None
        if self.launcher_receipt is not None:
            self.launcher_receipt.close()
            self.launcher_receipt = None
        if hasattr(self, "receipt"):
            self.receipt.close()


def hold_taste_main_v2_controller_authority(
    receipt_path: str | Path,
    heartbeat_path: str | Path,
    expected_controller_id: str | None,
    expected_git_commit: str | None,
    expected_git_tree: str | None,
    max_age_seconds: float = DEFAULT_MAX_HEARTBEAT_AGE_SECONDS,
    **kwargs: Any,
) -> HeldTasteMainV2ControllerAuthority:
    """Compatibility constructor shared by every Taste method verifier."""

    return HeldTasteMainV2ControllerAuthority(
        receipt_path,
        heartbeat_path,
        expected_controller_id,
        expected_git_commit,
        expected_git_tree,
        max_age_seconds,
        **kwargs,
    )


def controller_status(
    *,
    controller_root: str | Path,
    max_age_seconds: float = DEFAULT_MAX_HEARTBEAT_AGE_SECONDS,
) -> dict[str, Any]:
    """Read-only controller status.  This function never repairs state."""

    try:
        root = _require_absolute_physical(
            Path(controller_root), label="controller root"
        )
        receipt_path = root / CONTROLLER_RECEIPT_NAME
        receipt_probe = _HeldFile(receipt_path, label="controller receipt")
        try:
            receipt_payload = receipt_probe.json()
            receipt_sha256 = receipt_probe.sha256
        finally:
            receipt_probe.close()
        heartbeat_path = latest_heartbeat_path(root)
        heartbeat_probe = _HeldFile(heartbeat_path, label="controller heartbeat")
        try:
            heartbeat_sha256 = heartbeat_probe.sha256
        finally:
            heartbeat_probe.close()
        with hold_taste_main_v2_controller_authority(
            receipt_path,
            heartbeat_path,
            None,
            None,
            None,
            max_age_seconds,
            expected_launcher_receipt_path=receipt_payload[
                "launcher_receipt_path"
            ],
            expected_launcher_receipt_sha256=receipt_payload[
                "launcher_receipt_sha256"
            ],
            expected_receipt_sha256=receipt_sha256,
            expected_heartbeat_sha256=heartbeat_sha256,
        ) as authority:
            evidence = authority.revalidate()
            return {
                "schema_version": STATUS_SCHEMA,
                "status": "RUNNING",
                "science_released": False,
                "authority": evidence,
                "read_only": True,
                "auto_terminate_uncontrolled_children": False,
            }
    except (
        OSError,
        ValueError,
        ProcessIdentityV2Error,
        TasteMainV2AuthorityError,
    ) as exc:
        return {
            "schema_version": STATUS_SCHEMA,
            "status": "QUARANTINED",
            "reason": f"{type(exc).__name__}: {exc}",
            "science_released": False,
            "read_only": True,
            "auto_terminate_uncontrolled_children": False,
        }


def run_controller_loop(
    *,
    receipt_path: str | Path,
    heartbeat_count: int = 0,
    heartbeat_interval_seconds: int = HEARTBEAT_INTERVAL_SECONDS,
) -> int:
    """Append heartbeats forever (or a bounded count for protocol tests)."""

    require_auto_termination_disabled()
    if heartbeat_interval_seconds != HEARTBEAT_INTERVAL_SECONDS:
        raise TasteMainV2AuthorityError("production heartbeat interval must be 10 seconds")
    if isinstance(heartbeat_count, bool) or heartbeat_count < 0:
        raise TasteMainV2AuthorityError("heartbeat count is invalid")
    sequence = 1
    previous: str | None = None
    quarantined = False
    while heartbeat_count == 0 or sequence <= heartbeat_count:
        state = "QUARANTINED" if quarantined else "WAITING_DEPENDENCIES"
        try:
            created = write_heartbeat_generation(
                controller_receipt_path=receipt_path,
                sequence=sequence,
                previous_heartbeat_sha256=previous,
                controller_state=state,
            )
        except (
            OSError,
            ValueError,
            ProcessIdentityV2Error,
            TasteMainV2AuthorityError,
        ):
            receipt_root = Path(receipt_path).parent
            heartbeat_directory = receipt_root / HEARTBEAT_DIRECTORY
            published_same_sequence = [
                entry.name
                for entry in os.scandir(heartbeat_directory)
                if (
                    (matched := _HEARTBEAT_NAME.fullmatch(entry.name)) is not None
                    and int(matched.group("sequence")) == sequence
                )
            ]
            if published_same_sequence:
                # Never reuse a sequence after an error that occurred after
                # immutable publication. Read-only status will quarantine any
                # invalid terminal authority; the controller must not create a
                # second, ambiguous generation for the same sequence.
                raise
            # Once a previously healthy controller observes lease/identity
            # drift, it keeps every retained reservation and publishes only a
            # fail-closed quarantine generation.  It never signals a worker.
            if quarantined:
                raise
            quarantined = True
            created = write_heartbeat_generation(
                controller_receipt_path=receipt_path,
                sequence=sequence,
                previous_heartbeat_sha256=previous,
                controller_state="QUARANTINED",
            )
        previous = created.sha256
        sequence += 1
        if heartbeat_count != 0 and sequence > heartbeat_count:
            break
        time.sleep(HEARTBEAT_INTERVAL_SECONDS)
    return 0


__all__ = [
    "ALLOWED_GPU_INDICES",
    "CONTROLLER_RECEIPT_NAME",
    "CONTROLLER_RECEIPT_SCHEMA",
    "ControllerCreation",
    "DEFAULT_MAX_HEARTBEAT_AGE_SECONDS",
    "GPU_LEASE_ACTIVATION_SCHEMA",
    "GPU_LEASE_RELEASE_SCHEMA",
    "GPU_LEASE_RENEWAL_SCHEMA",
    "GPU_LEASE_SCHEMA",
    "GpuLeaseActivationCreation",
    "GpuLeaseCreation",
    "GpuLeaseRenewalCreation",
    "HEARTBEAT_DIRECTORY",
    "HEARTBEAT_INTERVAL_SECONDS",
    "HEARTBEAT_SCHEMA",
    "HeartbeatCreation",
    "HeldTasteMainV2ControllerAuthority",
    "LAUNCHER_READY_NAME",
    "LAUNCHER_READY_SCHEMA",
    "LAUNCHER_RECEIPT_NAME",
    "LAUNCHER_RECEIPT_SCHEMA",
    "LauncherCreation",
    "MANAGED_TASTE_RELEASE_MARKER",
    "MANAGED_TASTE_RELEASE_VERSION",
    "MAX_ACTIVE_TASKS",
    "PROTECTED_GPU_INDICES",
    "STATUS_SCHEMA",
    "TASK_GPU_BINDINGS",
    "TasteMainV2AuthorityError",
    "capture_policy_facts",
    "controller_status",
    "create_controller_receipt",
    "create_gpu_lease_activation",
    "create_gpu_lease_request",
    "create_gpu_lease_renewal",
    "create_launcher_receipt",
    "ensure_controller_namespace_parents",
    "hold_taste_main_v2_controller_authority",
    "immutable_authority_sha256",
    "initial_heartbeat_path",
    "inspect_clean_git",
    "latest_heartbeat_path",
    "probe_physical_gpus",
    "publish_launcher_ready",
    "read_launcher_policy_facts",
    "release_registered_runner_gpu_lock_after_ack",
    "run_controller_loop",
    "write_heartbeat_generation",
]
