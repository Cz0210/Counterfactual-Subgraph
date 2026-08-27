"""Read-only adoption of the completed TasteMolNet T2 GINE result.

This module deliberately does *not* repair the failed v2 controller.  It holds
the old controller, run-state, registry, training-state, output, and source
identities read-only and can publish a five-file receipt only into one exact,
fresh, non-scientific control namespace.  Publication is release-frozen in
this source revision.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping


ADOPTION_VERSION = "v1"
ADOPTION_MARKER = "T2_GINE_FULL_PASS_ADOPTED"
ADOPTION_NAMESPACE = "tastemolnet-t2-gine-pass-adoption-v1"
RELEASE_CONFIG_RELATIVE = (
    "configs/autodl/tastemolnet_t2_pass_adoption_release_v1.json"
)
RELEASE_CONFIG_SCHEMA = "tastemolnet_t2_gine_pass_release_config_v1"
EXTERNAL_RELEASE_SCHEMA = (
    "tastemolnet_t2_gine_pass_external_release_authority_v1"
)
GIT_BINARY = Path("/usr/bin/git")
IMPLEMENTATION_CRITICAL_BLOBS = (
    RELEASE_CONFIG_RELATIVE,
    "docs/AUTODL_TASTEMOLNET_T2_PASS_ADOPTION_V1.md",
    "scripts/autodl/adopt_tastemolnet_gine_pass_v1.py",
    "scripts/slurm/adopt_tastemolnet_gine_pass_v1.sh",
    "src/utils/tastemolnet_gine_pass_adoption_v1.py",
)
FIVE_FILE_SET = (
    "input_hashes.json",
    "state.json",
    "manifest.json",
    "output_hashes.json",
    "gate.json",
)

SOURCE_CID = "tastemolnet_gine_v2_20260827T160626Z_583bf668"
SOURCE_RUN_ID = (
    "20260827T160732Z-tastemolnet-"
    "TASTEMOLNET_GINE_FULL_RESEARCH_V1-87809"
)
SOURCE_STAGE = "TASTEMOLNET_GINE_FULL_RESEARCH_V1"
SOURCE_EXECUTION_COMMIT = "583bf668896142d8cc292cd624fbbffc20faf688"
SOURCE_IDENTITY_FIX_COMMIT = "3a90fd8697b58bad4f95f3be9347b327d5c51043"
SOURCE_FAILED_REASON = "WORKER_PROCESS_IDENTITY_DRIFT"
SOURCE_OUTPUT_RELATIVE = Path(
    "outputs/gnn_oracles/tastemolnet/gine/seed7/"
    "full-20260827T160626Z"
)

CONTROLLER_SCHEMA = "autodl_tastemolnet_gine_persistent_controller_v2"
CONTROLLER_CLAIM_SCHEMA = "autodl_tastemolnet_gine_controller_root_claim_v2"
CONTROLLER_STATE_SCHEMA = "autodl_tastemolnet_gine_controller_state_v2"
TRAINER_AUTHORITY_SCHEMA = "autodl_exp_run_trainer_child_authority_v1"
TRAINING_STATE_SCHEMA = "molecular_gnn_epoch_resume_v1"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")
UTC_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
FILE_IDENTITY_FIELDS = frozenset(
    {
        "device",
        "inode",
        "mode",
        "uid",
        "nlink",
        "size",
        "mtime_ns",
        "ctime_ns",
    }
)
TRAINER_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "run_id",
        "dataset",
        "stage",
        "controller_cid",
        "controller_root",
        "project_root",
        "authority_path",
        "parent_exp_run",
        "child_registered",
        "trainer_command",
        "trainer_command_sha256",
        "barrier_record",
    }
)
PROCESS_SNAPSHOT_FIELDS = frozenset(
    {
        "pid",
        "linux_start_ticks",
        "ppid",
        "argv",
        "argv_sha256",
        "cmdline_sha256",
        "cwd",
        "exe",
        "exe_identity",
    }
)
FAILED_CONTROLLER_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "cid",
        "spec_sha256",
        "root_claim_sha256",
        "phase",
        "updated_at",
        "resource_deadline_sha256",
        "resource_deadline_epoch_seconds",
        "attempt",
        "launch_index",
        "retries_used",
        "reason",
    }
)
STARTUP_BARRIER_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "state",
        "lock_path",
        "lock_dev",
        "lock_inode",
        "lock_mode",
        "lock_uid",
        "lock_nlink",
        "record_path",
        "python_executable",
        "release_read_fd",
        "lock_fd",
        "release_token_bytes",
        "release_token_sha256",
        "target_argv",
        "target_argv_sha256",
        "launcher_argv",
        "launcher_argv_sha256",
    }
)

CHECKPOINT_FILES = frozenset(
    {
        "model.pt",
        "config.yaml",
        "model_card.json",
        "feature_schema.json",
        "label_map.json",
        "split_manifest.json",
        "training_metrics.json",
        "validation_predictions.csv",
        "test_evaluation_status.json",
        "temperature_scaling.json",
        "environment.json",
        "git_state.json",
        "sha256sums.txt",
        "data_use_policy_binding.json",
        "graph_cache_usage.json",
        "oracle_manifest.json",
        "last.pt",
        "last_checkpoint.json",
        "checkpoint_reload.json",
    }
)
HASHED_CHECKPOINT_FILES = CHECKPOINT_FILES - {"sha256sums.txt"}
CONTROLLER_REQUIRED_OUTPUT_FILES = (
    "model.pt",
    "last.pt",
    "last_checkpoint.json",
    "checkpoint_reload.json",
    "model_card.json",
    "feature_schema.json",
    "training_metrics.json",
    "test_evaluation_status.json",
    "temperature_scaling.json",
    "data_use_policy_binding.json",
    "graph_cache_usage.json",
    "oracle_manifest.json",
    "sha256sums.txt",
)

# The implementation commit is permanently release-disabled.  A later clean
# one-parent release commit may change only RELEASE_CONFIG_RELATIVE.  That
# config can point at a separately reviewed, physically held external receipt;
# neither CLI arguments nor environment variables are release capabilities.


class T2PassAdoptionError(RuntimeError):
    """The old scientific result cannot be adopted without ambiguity."""


class T2PassAdoptionReleaseDisabled(T2PassAdoptionError):
    """Publication is frozen until reviewed release pins are filled."""


def _strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _exact_int(value: Any, expected: int) -> bool:
    return _strict_int(value) and value == expected


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _file_identity(info: os.stat_result) -> dict[str, int]:
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(info.st_mode),
        "uid": int(info.st_uid),
        "nlink": int(info.st_nlink),
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def _directory_identity(info: os.stat_result) -> dict[str, int]:
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(info.st_mode),
        "uid": int(info.st_uid),
    }


def _publication_directory_identity(info: os.stat_result) -> dict[str, int]:
    """Stable fields that can be committed before the final directory update."""

    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(info.st_mode),
        "uid": int(info.st_uid),
        "gid": int(info.st_gid),
    }


def _publication_file_identity(info: os.stat_result) -> dict[str, int]:
    """Physical leaf identity recorded by the terminal gate."""

    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(info.st_mode),
        "uid": int(info.st_uid),
        "gid": int(info.st_gid),
        "nlink": int(info.st_nlink),
        "size": int(info.st_size),
    }


def _production_proc_root() -> Path:
    """Private test seam; production has no CLI/environment proc override."""

    return Path("/proc")


def _assert_linux_procfs(directory_fd: int) -> None:
    """Require Linux PROC_SUPER_MAGIC for the already-held literal /proc fd."""

    buffer = ctypes.create_string_buffer(256)
    libc = ctypes.CDLL(None, use_errno=True)
    fstatfs = getattr(libc, "fstatfs", None)
    if fstatfs is None:
        raise T2PassAdoptionError("Linux fstatfs is unavailable for /proc audit")
    fstatfs.argtypes = [ctypes.c_int, ctypes.c_void_p]
    fstatfs.restype = ctypes.c_int
    if fstatfs(directory_fd, ctypes.byref(buffer)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    fs_type = ctypes.c_long.from_buffer_copy(buffer.raw[: ctypes.sizeof(ctypes.c_long)])
    if int(fs_type.value) & 0xFFFFFFFF != 0x00009FA0:
        raise T2PassAdoptionError("literal /proc is not a physical procfs mount")


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _absolute(path: Any, *, label: str) -> Path:
    if not isinstance(path, (str, Path)):
        raise T2PassAdoptionError(f"{label} must be an absolute path")
    raw = Path(path).expanduser()
    if not raw.is_absolute() or "\x00" in str(raw):
        raise T2PassAdoptionError(f"{label} must be an absolute path")
    normalized = Path(os.path.normpath(str(raw)))
    if normalized != raw:
        raise T2PassAdoptionError(f"{label} must be lexically normalized")
    return normalized


def _relative_parts(relative: str | PurePosixPath) -> tuple[str, ...]:
    value = PurePosixPath(relative)
    if value.is_absolute() or not value.parts or any(
        part in {"", ".", ".."} or "\x00" in part for part in value.parts
    ):
        raise T2PassAdoptionError(f"unsafe relative path: {relative!s}")
    return tuple(value.parts)


class HeldDirectory:
    """Retain every absolute pathname edge with openat/O_NOFOLLOW authority."""

    def __init__(self, path: str | Path, *, label: str) -> None:
        self.path = _absolute(path, label=label)
        self.label = label
        self._edges: list[tuple[int, int, str, dict[str, int]]] = []
        self.fd: int | None = None

    def open(self) -> "HeldDirectory":
        if self.fd is not None:
            raise T2PassAdoptionError(f"{self.label} authority is already open")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        parent_fd = os.open("/", flags)
        try:
            for name in self.path.parts[1:]:
                child_fd = os.open(name, flags, dir_fd=parent_fd)
                child_info = os.fstat(child_fd)
                named_info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if (
                    not stat.S_ISDIR(child_info.st_mode)
                    or stat.S_ISLNK(named_info.st_mode)
                    or not _same_inode(child_info, named_info)
                ):
                    os.close(child_fd)
                    raise T2PassAdoptionError(
                        f"{self.label} contains a non-physical or foreign directory"
                    )
                self._edges.append(
                    (parent_fd, child_fd, name, _directory_identity(child_info))
                )
                parent_fd = child_fd
            if not self._edges:
                raise T2PassAdoptionError(f"{self.label} may not be filesystem root")
            self.fd = self._edges[-1][1]
            if os.fstat(self.fd).st_uid != os.getuid():
                raise T2PassAdoptionError(
                    f"{self.label} final directory is not owned by the current user"
                )
            self.verify()
            return self
        except BaseException:
            if not self._edges:
                os.close(parent_fd)
            self.close()
            raise

    @property
    def identity(self) -> dict[str, int]:
        if self.fd is None:
            raise T2PassAdoptionError(f"{self.label} authority is closed")
        return _directory_identity(os.fstat(self.fd))

    def verify(self) -> None:
        if self.fd is None:
            raise T2PassAdoptionError(f"{self.label} authority is closed")
        for parent_fd, child_fd, name, expected in self._edges:
            held = os.fstat(child_fd)
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISDIR(held.st_mode)
                or stat.S_ISLNK(named.st_mode)
                or not _same_inode(held, named)
                or _directory_identity(held) != expected
                or _directory_identity(named) != expected
            ):
                raise T2PassAdoptionError(f"{self.label} pathname authority changed")

    def close(self) -> None:
        seen: set[int] = set()
        for parent_fd, child_fd, _name, _identity in reversed(self._edges):
            for descriptor in (child_fd, parent_fd):
                if descriptor not in seen:
                    try:
                        os.close(descriptor)
                    except BaseException:
                        pass
                    seen.add(descriptor)
        self._edges.clear()
        self.fd = None

    def __enter__(self) -> "HeldDirectory":
        return self.open()

    def __exit__(self, *_args: object) -> None:
        self.close()


class HeldFile:
    def __init__(
        self,
        *,
        parent_fd: int,
        name: str,
        relative: str,
        max_read_bytes: int = 64 * 1024**3,
    ) -> None:
        self.parent_fd = parent_fd
        self.name = name
        self.relative = relative
        self.fd = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            before = os.fstat(self.fd)
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(named.st_mode)
                or before.st_nlink != 1
                or before.st_uid != os.getuid()
                or not _same_inode(before, named)
            ):
                raise T2PassAdoptionError(
                    f"{relative} is not one owner-bound physical regular file"
                )
            self.identity = _file_identity(before)
            self.sha256 = self._hash(max_read_bytes=max_read_bytes)
            self.verify()
        except BaseException:
            self.close()
            raise

    def _hash(self, *, max_read_bytes: int) -> str:
        assert self.fd is not None
        os.lseek(self.fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(self.fd, 8 * 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > max_read_bytes:
                raise T2PassAdoptionError(f"{self.relative} exceeds its audit bound")
            digest.update(block)
        return digest.hexdigest()

    def bytes(self, *, limit: int = 32 * 1024**2) -> bytes:
        if self.fd is None:
            raise T2PassAdoptionError(f"{self.relative} authority is closed")
        if self.identity["size"] > limit:
            raise T2PassAdoptionError(f"{self.relative} exceeds its typed-read bound")
        os.lseek(self.fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        remaining = int(self.identity["size"])
        while remaining:
            block = os.read(self.fd, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        data = b"".join(chunks)
        if len(data) != self.identity["size"] or hashlib.sha256(data).hexdigest() != self.sha256:
            raise T2PassAdoptionError(f"{self.relative} held bytes changed")
        self.verify()
        return data

    def json(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise T2PassAdoptionError(f"{self.relative} is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise T2PassAdoptionError(f"{self.relative} must contain one JSON object")
        return payload

    def verify(self) -> None:
        if self.fd is None:
            raise T2PassAdoptionError(f"{self.relative} authority is closed")
        for parent_fd, child_fd, name, expected in getattr(
            self, "_selected_parent_edges", []
        ):
            held_directory = os.fstat(child_fd)
            named_directory = os.stat(
                name, dir_fd=parent_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISDIR(held_directory.st_mode)
                or stat.S_ISLNK(named_directory.st_mode)
                or not _same_inode(held_directory, named_directory)
                or _directory_identity(held_directory) != expected
                or _directory_identity(named_directory) != expected
            ):
                raise T2PassAdoptionError(
                    f"{self.relative} selected pathname authority changed"
                )
        held = os.fstat(self.fd)
        named = os.stat(self.name, dir_fd=self.parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(held.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or not _same_inode(held, named)
            or _file_identity(held) != self.identity
            or _file_identity(named) != self.identity
        ):
            raise T2PassAdoptionError(f"{self.relative} named authority changed")

    def close(self) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except BaseException:
                pass
            self.fd = None


class HeldTree:
    """Retain every directory and file descriptor in a bounded physical tree."""

    def __init__(
        self,
        path: str | Path,
        *,
        label: str,
        max_files: int = 4096,
        max_bytes: int = 64 * 1024**3,
    ) -> None:
        self.root = HeldDirectory(path, label=label)
        self.label = label
        self.max_files = max_files
        self.max_bytes = max_bytes
        self._directory_rows: list[dict[str, Any]] = []
        self._directory_fds: list[tuple[int, int, str, dict[str, int], str]] = []
        self._directory_listings: list[tuple[int, tuple[str, ...], str]] = []
        self.files: dict[str, HeldFile] = {}
        self._total_bytes = 0

    def open(self) -> "HeldTree":
        try:
            self.root.open()
            assert self.root.fd is not None
            self._scan(self.root.fd, "")
            self.verify()
            return self
        except BaseException:
            self.close()
            raise

    def _scan(self, directory_fd: int, prefix: str) -> None:
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as exc:
            raise T2PassAdoptionError(f"{self.label} cannot be inventoried") from exc
        self._directory_listings.append((directory_fd, tuple(names), prefix or "."))
        for name in names:
            relative = f"{prefix}/{name}" if prefix else name
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                child_fd = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
                held = os.fstat(child_fd)
                if not _same_inode(info, held) or held.st_uid != os.getuid():
                    os.close(child_fd)
                    raise T2PassAdoptionError(f"{self.label}/{relative} changed while opened")
                identity = _directory_identity(held)
                self._directory_fds.append((directory_fd, child_fd, name, identity, relative))
                self._directory_rows.append(
                    {"path": relative, "kind": "directory", "identity": identity}
                )
                self._scan(child_fd, relative)
            elif stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                held_file = HeldFile(
                    parent_fd=directory_fd,
                    name=name,
                    relative=relative,
                    max_read_bytes=self.max_bytes,
                )
                self.files[relative] = held_file
                self._total_bytes += int(held_file.identity["size"])
                if len(self.files) > self.max_files:
                    raise T2PassAdoptionError(f"{self.label} exceeds its file-count bound")
                if self._total_bytes > self.max_bytes:
                    raise T2PassAdoptionError(f"{self.label} exceeds its byte bound")
            else:
                raise T2PassAdoptionError(
                    f"{self.label}/{relative} is symlinked or special"
                )

    @property
    def inventory(self) -> list[dict[str, Any]]:
        rows = list(self._directory_rows)
        rows.extend(
            {
                "path": relative,
                "kind": "file",
                "identity": held.identity,
                "sha256": held.sha256,
            }
            for relative, held in self.files.items()
        )
        return sorted(rows, key=lambda row: (str(row["path"]), str(row["kind"])))

    @property
    def inventory_sha256(self) -> str:
        return _stable_sha256(self.inventory)

    def file(self, relative: str) -> HeldFile:
        try:
            return self.files[relative]
        except KeyError as exc:
            raise T2PassAdoptionError(f"{self.label} lacks required file {relative}") from exc

    def verify(self) -> None:
        self.root.verify()
        for parent_fd, child_fd, name, expected, relative in self._directory_fds:
            held = os.fstat(child_fd)
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISDIR(held.st_mode)
                or stat.S_ISLNK(named.st_mode)
                or not _same_inode(held, named)
                or _directory_identity(held) != expected
                or _directory_identity(named) != expected
            ):
                raise T2PassAdoptionError(f"{self.label}/{relative} authority changed")
        for held in self.files.values():
            held.verify()
        for directory_fd, expected_names, relative in self._directory_listings:
            if tuple(sorted(os.listdir(directory_fd))) != expected_names:
                raise T2PassAdoptionError(
                    f"{self.label}/{relative} directory inventory changed"
                )

    def close(self) -> None:
        for held in self.files.values():
            held.close()
        self.files.clear()
        seen: set[int] = set()
        for _parent, child, _name, _identity, _relative in reversed(self._directory_fds):
            if child not in seen:
                try:
                    os.close(child)
                except BaseException:
                    pass
                seen.add(child)
        self._directory_fds.clear()
        self._directory_listings.clear()
        self._total_bytes = 0
        self.root.close()

    def __enter__(self) -> "HeldTree":
        return self.open()

    def __exit__(self, *_args: object) -> None:
        self.close()


def _open_selected_file(root: HeldDirectory, relative: str) -> HeldFile:
    if root.fd is None:
        raise T2PassAdoptionError("selected-file root authority is closed")
    parts = _relative_parts(relative)
    parent_fd = root.fd
    opened: list[tuple[int, int, str, dict[str, int]]] = []
    result: HeldFile | None = None
    try:
        for part in parts[:-1]:
            child = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
            held = os.fstat(child)
            named = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISDIR(held.st_mode)
                or stat.S_ISLNK(named.st_mode)
                or not _same_inode(held, named)
                or held.st_uid != os.getuid()
            ):
                os.close(child)
                raise T2PassAdoptionError(
                    f"selected path {relative} contains a foreign directory"
                )
            opened.append(
                (parent_fd, child, part, _directory_identity(held))
            )
            parent_fd = child
        # HeldFile needs its parent descriptor for its full lifetime, so these
        # intermediate descriptors intentionally remain open on the object.
        result = HeldFile(parent_fd=parent_fd, name=parts[-1], relative=relative)
        setattr(result, "_selected_parent_edges", opened)
        result.verify()
        return result
    except BaseException:
        if result is not None:
            result.close()
        for _parent, child, _name, _identity in reversed(opened):
            os.close(child)
        raise


def _close_selected_file(value: HeldFile) -> None:
    value.close()
    for _parent, child, _name, _identity in reversed(
        getattr(value, "_selected_parent_edges", [])
    ):
        try:
            os.close(child)
        except BaseException:
            pass


@dataclass(frozen=True)
class T2PassAdoptionSources:
    control_root: Path
    controller_root: Path
    output_root: Path
    training_state_root: Path
    execution_project_root: Path
    identity_fix_project_root: Path
    proc_root: Path
    _test_proc_override: bool = False

    @classmethod
    def build(
        cls,
        *,
        control_root: str | Path,
        controller_root: str | Path,
        output_root: str | Path,
        training_state_root: str | Path,
        execution_project_root: str | Path,
        identity_fix_project_root: str | Path,
    ) -> "T2PassAdoptionSources":
        return cls(
            control_root=_absolute(control_root, label="control root"),
            controller_root=_absolute(controller_root, label="old controller root"),
            output_root=_absolute(output_root, label="old scientific output root"),
            training_state_root=_absolute(
                training_state_root, label="old training-state root"
            ),
            execution_project_root=_absolute(
                execution_project_root, label="deployed execution project root"
            ),
            identity_fix_project_root=_absolute(
                identity_fix_project_root, label="identity-fix project root"
            ),
            proc_root=_absolute(_production_proc_root(), label="production proc root"),
        )

    @classmethod
    def _build_for_tests(
        cls,
        *,
        control_root: str | Path,
        controller_root: str | Path,
        output_root: str | Path,
        training_state_root: str | Path,
        execution_project_root: str | Path,
        identity_fix_project_root: str | Path,
        proc_root: str | Path,
    ) -> "T2PassAdoptionSources":
        """Construct a synthetic procfs authority; never exposed by the CLI."""

        production = cls.build(
            control_root=control_root,
            controller_root=controller_root,
            output_root=output_root,
            training_state_root=training_state_root,
            execution_project_root=execution_project_root,
            identity_fix_project_root=identity_fix_project_root,
        )
        return cls(
            **{
                **production.__dict__,
                "proc_root": _absolute(proc_root, label="test proc root"),
                "_test_proc_override": True,
            }
        )

    @property
    def runtime_root(self) -> Path:
        return self.control_root.parent

    @property
    def run_state_root(self) -> Path:
        return self.control_root / "experiment_registry" / "run_state" / SOURCE_RUN_ID

    @property
    def registry_path(self) -> Path:
        return self.control_root / "experiment_registry" / "runs.jsonl"

    @property
    def adoption_root(self) -> Path:
        return self.control_root / ADOPTION_NAMESPACE / SOURCE_CID


def adoption_output_root(control_root: str | Path) -> Path:
    root = _absolute(control_root, label="control root")
    return root / ADOPTION_NAMESPACE / SOURCE_CID


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _descriptor_path(descriptor: int, *, non_linux_fallback: Path) -> str:
    candidates = (
        f"/proc/self/fd/{descriptor}",
        f"/dev/fd/{descriptor}",
    )
    expected = os.fstat(descriptor)
    for candidate in candidates:
        try:
            observed = os.stat(candidate)
        except OSError:
            continue
        if _same_inode(expected, observed):
            return candidate
    if not sys.platform.startswith("linux"):
        return str(non_linux_fallback)
    raise T2PassAdoptionError("descriptor-backed Git paths are unavailable")


def _git_binary_authority() -> tuple[int, dict[str, int], str]:
    descriptor = os.open(GIT_BINARY, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        held = os.fstat(descriptor)
        named = os.stat(GIT_BINARY, follow_symlinks=False)
        if (
            not stat.S_ISREG(held.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or not _same_inode(held, named)
            or held.st_uid != 0
            or held.st_mode & 0o022
        ):
            raise T2PassAdoptionError(
                "fixed /usr/bin/git is not one root-owned non-writable binary"
            )
        identity = _file_identity(held)
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if _file_identity(os.fstat(descriptor)) != identity:
            raise T2PassAdoptionError("fixed /usr/bin/git changed while hashed")
        return descriptor, identity, digest.hexdigest()
    except BaseException:
        os.close(descriptor)
        raise


def _git_identity(
    path: Path,
    *,
    critical_paths: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Audit one checkout with fixed Git, isolated config, and physical fds."""

    root = HeldDirectory(path, label="Git worktree")
    dot_git_file: HeldFile | None = None
    git_directory: HeldDirectory | None = None
    git_binary_fd: int | None = None
    try:
        root.open()
        assert root.fd is not None
        dot_git_info = os.stat(".git", dir_fd=root.fd, follow_symlinks=False)
        if stat.S_ISDIR(dot_git_info.st_mode) and not stat.S_ISLNK(dot_git_info.st_mode):
            git_path = path / ".git"
        elif stat.S_ISREG(dot_git_info.st_mode) and not stat.S_ISLNK(dot_git_info.st_mode):
            dot_git_file = _open_selected_file(root, ".git")
            try:
                pointer = dot_git_file.bytes(limit=4096).decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                raise T2PassAdoptionError("Git worktree pointer is not UTF-8") from exc
            if not pointer.startswith("gitdir: ") or "\n" in pointer:
                raise T2PassAdoptionError("Git worktree pointer is malformed")
            raw_git_path = Path(pointer[len("gitdir: ") :])
            git_path = (
                raw_git_path
                if raw_git_path.is_absolute()
                else Path(os.path.abspath(path / raw_git_path))
            )
            git_path = _absolute(git_path, label="Git directory")
        else:
            raise T2PassAdoptionError(".git is symlinked, special, or absent")
        git_directory = HeldDirectory(git_path, label="Git directory")
        git_directory.open()
        assert git_directory.fd is not None
        git_binary_fd, git_binary_identity, git_binary_sha = _git_binary_authority()
        worktree_fd_path = _descriptor_path(
            root.fd,
            non_linux_fallback=path,
        )
        gitdir_fd_path = _descriptor_path(
            git_directory.fd,
            non_linux_fallback=git_path,
        )
        environment = {
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "LANG": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_NO_REPLACE_OBJECTS": "1",
        }
        prefix = [
            str(GIT_BINARY),
            "--no-optional-locks",
            f"--git-dir={gitdir_fd_path}",
            f"--work-tree={worktree_fd_path}",
            "-c",
            f"core.worktree={worktree_fd_path}",
            "-c",
            "core.filemode=true",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.attributesFile=/dev/null",
            "-c",
            "core.excludesFile=/dev/null",
            "-c",
            "core.untrackedCache=false",
        ]
        pass_fds = (root.fd, git_directory.fd, git_binary_fd)

        def run(*args: str, text: bool = True) -> str | bytes:
            try:
                completed = subprocess.run(
                    [*prefix, *args],
                    check=True,
                    capture_output=True,
                    text=text,
                    env=environment,
                    pass_fds=pass_fds,
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                raise T2PassAdoptionError(
                    f"hardened Git audit failed for {path}"
                ) from exc
            return completed.stdout

        commit = str(run("rev-parse", "HEAD")).strip()
        tree = str(run("rev-parse", "HEAD^{tree}")).strip()
        if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(
            r"[0-9a-f]{40}", tree
        ):
            raise T2PassAdoptionError("Git commit/tree identity is not full SHA-1")
        status_output = str(
            run(
                "status",
                "--porcelain=v2",
                "--untracked-files=all",
                "--ignored=matching",
            )
        )
        if status_output:
            raise T2PassAdoptionError(
                "Git worktree is not fully clean (tracked/staged/untracked/ignored)"
            )
        index_rows = bytes(run("ls-files", "-v", "-z", text=False)).split(b"\0")
        for row in index_rows:
            if not row:
                continue
            if len(row) < 3 or row[1:2] != b" " or row[:1] != b"H":
                raise T2PassAdoptionError(
                    "Git index contains skip-worktree/assume-unchanged metadata"
                )
        if str(run("for-each-ref", "--format=%(refname)", "refs/replace")).strip():
            raise T2PassAdoptionError("Git replacement refs are forbidden")
        critical_hashes: dict[str, str] = {}
        for relative in critical_paths:
            _relative_parts(relative)
            working = _open_selected_file(root, relative)
            try:
                working_bytes = working.bytes(limit=64 * 1024**2)
                committed = bytes(run("show", f"HEAD:{relative}", text=False))
                if working_bytes != committed:
                    raise T2PassAdoptionError(
                        f"critical Git blob differs from HEAD: {relative}"
                    )
                critical_hashes[relative] = working.sha256
            finally:
                _close_selected_file(working)
        lineage = str(run("rev-list", "--parents", "-n", "1", "HEAD")).strip().split()
        if not lineage or lineage[0] != commit or any(
            not re.fullmatch(r"[0-9a-f]{40}", value) for value in lineage
        ):
            raise T2PassAdoptionError("Git commit lineage is malformed")
        parent_tree: str | None = None
        changed_from_parent: list[str] = []
        parent_critical_blobs: dict[str, str] = {}
        if len(lineage) == 2:
            parent_tree = str(run("rev-parse", f"{lineage[1]}^{{tree}}")).strip()
            changed_from_parent = sorted(
                value
                for value in str(
                    run(
                        "diff",
                        "--name-only",
                        "--no-renames",
                        lineage[1],
                        "HEAD",
                    )
                ).splitlines()
                if value
            )
            if changed_from_parent == [RELEASE_CONFIG_RELATIVE]:
                for relative in critical_paths:
                    parent_bytes = bytes(
                        run("show", f"{lineage[1]}:{relative}", text=False)
                    )
                    parent_critical_blobs[relative] = hashlib.sha256(
                        parent_bytes
                    ).hexdigest()
        root.verify()
        git_directory.verify()
        held_git_binary = os.fstat(git_binary_fd)
        named_git_binary = os.stat(GIT_BINARY, follow_symlinks=False)
        if (
            _file_identity(held_git_binary) != git_binary_identity
            or _file_identity(named_git_binary) != git_binary_identity
        ):
            raise T2PassAdoptionError("fixed /usr/bin/git authority changed")
        return {
            "commit": commit,
            "tree": tree,
            "status_porcelain": "",
            "critical_blobs": critical_hashes,
            "parents": lineage[1:],
            "parent_tree": parent_tree,
            "changed_from_parent": changed_from_parent,
            "parent_critical_blobs": parent_critical_blobs,
            "git_binary_path": str(GIT_BINARY),
            "git_binary_identity": git_binary_identity,
            "git_binary_sha256": git_binary_sha,
            "git_environment_policy": "fixed_allowlist_no_replace_no_config_hooks",
        }
    finally:
        if git_binary_fd is not None:
            try:
                os.close(git_binary_fd)
            except OSError:
                pass
        if git_directory is not None:
            git_directory.close()
        if dot_git_file is not None:
            try:
                _close_selected_file(dot_git_file)
            except OSError:
                pass
        root.close()


def _json_line_rows(file: HeldFile) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = file.bytes(limit=128 * 1024**2).decode("utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError
            rows.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise T2PassAdoptionError(
            f"run registry is malformed near line {number if 'number' in locals() else 1}"
        ) from exc
    return rows


def _validate_process_snapshot(value: Any, *, label: str) -> tuple[int, int]:
    if not isinstance(value, Mapping) or set(value) != PROCESS_SNAPSHOT_FIELDS:
        raise T2PassAdoptionError(f"{label} fields changed")
    pid = value.get("pid")
    start = value.get("linux_start_ticks")
    if not _strict_int(pid) or pid <= 0 or not _strict_int(start) or start <= 0:
        raise T2PassAdoptionError(f"{label} PID/start identity is untyped")
    argv = value.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or not item or "\x00" in item for item in argv)
        or value.get("argv_sha256") != _stable_sha256(argv)
        or not SHA256_RE.fullmatch(str(value.get("cmdline_sha256", "")))
    ):
        raise T2PassAdoptionError(f"{label} argv identity changed")
    if not _strict_int(value.get("ppid")) or value["ppid"] < 0:
        raise T2PassAdoptionError(f"{label} parent PID is untyped")
    for field in ("cwd", "exe"):
        raw = value.get(field)
        if (
            not isinstance(raw, str)
            or not raw
            or "\x00" in raw
            or not os.path.isabs(raw)
            or os.path.normpath(raw) != raw
        ):
            raise T2PassAdoptionError(f"{label} {field} identity changed")
    executable = value.get("exe_identity")
    if (
        not isinstance(executable, Mapping)
        or set(executable) != FILE_IDENTITY_FIELDS
        or any(not _strict_int(executable[field]) for field in FILE_IDENTITY_FIELDS)
        or executable["device"] < 0
        or executable["inode"] <= 0
        or executable["uid"] < 0
        or executable["nlink"] <= 0
        or executable["size"] <= 0
        or executable["mtime_ns"] < 0
        or executable["ctime_ns"] < 0
        or not stat.S_ISREG(executable["mode"])
    ):
        raise T2PassAdoptionError(f"{label} executable identity changed")
    return int(pid), int(start)


def _assert_pid_absent(proc: HeldDirectory, pid: int) -> None:
    if proc.fd is None:
        raise T2PassAdoptionError("proc authority is closed")
    try:
        os.stat(str(pid), dir_fd=proc.fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise T2PassAdoptionError(f"PID {pid} liveness is unreadable") from exc
    raise T2PassAdoptionError(f"PID {pid} is still present; all source PIDs must be dead")


def _parse_checkpoint_hashes(file: HeldFile) -> dict[str, str]:
    try:
        lines = file.bytes().decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise T2PassAdoptionError("checkpoint hash inventory is not UTF-8") from exc
    if len(lines) != 18:
        raise T2PassAdoptionError("checkpoint SHA closure must contain exactly 18 hashes")
    result: dict[str, str] = {}
    for line in lines:
        digest, separator, relative = line.partition("  ")
        if (
            separator != "  "
            or not SHA256_RE.fullmatch(digest)
            or not relative
            or PurePosixPath(relative).name != relative
            or relative in result
        ):
            raise T2PassAdoptionError("checkpoint SHA closure is malformed or duplicated")
        result[relative] = digest
    if set(result) != HASHED_CHECKPOINT_FILES:
        raise T2PassAdoptionError("checkpoint SHA closure file set changed")
    if list(result) != sorted(HASHED_CHECKPOINT_FILES):
        raise T2PassAdoptionError("checkpoint SHA closure order changed")
    return result


class _SourceHold:
    def __init__(self, sources: T2PassAdoptionSources) -> None:
        self.sources = sources
        self.control = HeldDirectory(sources.control_root, label="control root")
        self.proc = HeldDirectory(sources.proc_root, label="proc root")
        self.controller = HeldTree(sources.controller_root, label="old controller tree")
        self.output = HeldTree(sources.output_root, label="scientific output tree")
        self.training = HeldTree(
            sources.training_state_root, label="training-state tree"
        )
        self.run_state = HeldTree(sources.run_state_root, label="run-state tree")
        self.execution_project = HeldDirectory(
            sources.execution_project_root, label="execution project root"
        )
        self.identity_fix_project = HeldDirectory(
            sources.identity_fix_project_root, label="identity-fix project root"
        )
        self.adoption_project = HeldDirectory(
            Path(__file__).resolve(strict=True).parents[2],
            label="adoption implementation project root",
        )
        self.registry: HeldFile | None = None
        self.runtime_log_parent: HeldDirectory | None = None
        self.runtime_log: HeldFile | None = None
        self.external_release_parent: HeldDirectory | None = None
        self._selected: list[HeldFile] = []
        self._dead_pids: tuple[int, ...] = ()
        self._execution_git: dict[str, Any] | None = None
        self._identity_fix_git: dict[str, Any] | None = None
        self._adoption_git: dict[str, Any] | None = None
        self._release_config: dict[str, Any] | None = None
        self._external_release_file: HeldFile | None = None
        self.evidence: dict[str, Any] | None = None

    def __enter__(self) -> "_SourceHold":
        try:
            self._validate_paths()
            self.control.open()
            self.proc.open()
            if not self.sources._test_proc_override:
                assert self.proc.fd is not None
                _assert_linux_procfs(self.proc.fd)
            self.controller.open()
            self.output.open()
            self.training.open()
            self.run_state.open()
            self.execution_project.open()
            self.identity_fix_project.open()
            self.adoption_project.open()
            self.registry = _open_selected_file(
                self.control, "experiment_registry/runs.jsonl"
            )
            self._selected.append(self.registry)
            try:
                self.evidence = self._validate_and_build()
            except T2PassAdoptionError:
                raise
            except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
                raise T2PassAdoptionError(
                    "source evidence is malformed or untyped"
                ) from exc
            self.verify()
            return self
        except BaseException:
            self.close()
            raise

    def _validate_paths(self) -> None:
        source = self.sources
        adoption_project_root = Path(__file__).resolve(strict=True).parents[2]
        if not source._test_proc_override:
            if not sys.platform.startswith("linux"):
                raise T2PassAdoptionError("production PID audit requires Linux procfs")
            if source.proc_root != Path("/proc"):
                raise T2PassAdoptionError("production proc authority must be exactly /proc")
        elif source.proc_root == Path("/proc"):
            raise T2PassAdoptionError("test proc override must not alias production /proc")
        expected_controller = (
            source.control_root / "tastemolnet-gine-training-v2" / SOURCE_CID
        )
        expected_output = source.runtime_root / SOURCE_OUTPUT_RELATIVE
        if source.controller_root != expected_controller:
            raise T2PassAdoptionError("old controller root is not the deployed canonical root")
        if source.output_root != expected_output:
            raise T2PassAdoptionError("scientific output root is not the deployed canonical root")
        mutually_disjoint = (
            source.controller_root,
            source.training_state_root,
            source.output_root,
            source.execution_project_root,
            source.identity_fix_project_root,
            adoption_project_root,
        )
        for index, left in enumerate(mutually_disjoint):
            for right in mutually_disjoint[index + 1 :]:
                if _paths_overlap(left, right):
                    raise T2PassAdoptionError(
                        "controller/state/output/source authorities must be disjoint"
                    )
        protected = (
            source.controller_root,
            source.training_state_root,
            source.output_root,
            source.execution_project_root,
            source.identity_fix_project_root,
            adoption_project_root,
        )
        if any(_paths_overlap(source.adoption_root, value) for value in protected):
            raise T2PassAdoptionError("adoption output overlaps a retained source authority")

    def _selected_project_file(
        self, root: HeldDirectory, absolute_value: Any, *, label: str
    ) -> HeldFile:
        if not isinstance(absolute_value, str):
            raise T2PassAdoptionError(f"{label} path is untyped")
        absolute = _absolute(absolute_value, label=label)
        try:
            relative = absolute.relative_to(root.path).as_posix()
        except ValueError as exc:
            raise T2PassAdoptionError(f"{label} is outside its immutable project") from exc
        held = _open_selected_file(root, relative)
        self._selected.append(held)
        return held

    @staticmethod
    def _validate_release_config(value: Any) -> None:
        expected_fields = {
            "schema_version",
            "stage",
            "dataset",
            "authorization",
            "external_authority_path",
            "external_authority_sha256",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected_fields
            or value.get("schema_version") != RELEASE_CONFIG_SCHEMA
            or value.get("stage") != "T2_GINE_FULL"
            or value.get("dataset") != "tastemolnet"
            or type(value.get("authorization")) is not bool
        ):
            raise T2PassAdoptionError("T2 release config schema/type changed")
        if value["authorization"] is False:
            if (
                value.get("external_authority_path") is not None
                or value.get("external_authority_sha256") is not None
            ):
                raise T2PassAdoptionError(
                    "disabled T2 release config must retain null authority pins"
                )
            return
        authority_path = value.get("external_authority_path")
        authority_sha = value.get("external_authority_sha256")
        if type(authority_path) is not str or _absolute(
            authority_path, label="external release authority"
        ) != Path(authority_path):
            raise T2PassAdoptionError(
                "external release authority path is not exact absolute str"
            )
        if type(authority_sha) is not str or not SHA256_RE.fullmatch(authority_sha):
            raise T2PassAdoptionError(
                "external release authority SHA-256 is not exact native str"
            )

    def _validate_and_build(self) -> dict[str, Any]:
        assert self.registry is not None
        source = self.sources
        controller_spec_file = self.controller.file("controller_spec.json")
        controller_state_file = self.controller.file("controller_state.json")
        controller_claim_file = self.controller.file("controller_root_claim.json")
        controller_sentinel_file = self.controller.file(".controller-root-identity")
        controller_lock_file = self.controller.file(".controller.lock")
        resource_deadline_file = self.controller.file("resource_wait_deadline.json")
        spec = controller_spec_file.json()
        state = controller_state_file.json()
        claim = controller_claim_file.json()
        resource_deadline = resource_deadline_file.json()
        spec_canonical_sha = _stable_sha256(spec)
        source_identity = spec.get("source_identity")
        verified_model_route = spec.get("verified_model_route")
        if (
            not isinstance(source_identity, Mapping)
            or not isinstance(verified_model_route, Mapping)
            or spec.get("schema_version") != CONTROLLER_SCHEMA
            or spec.get("cid") != SOURCE_CID
            or spec.get("controller_root") != str(source.controller_root)
            or spec.get("project_root") != str(source.execution_project_root)
            or spec.get("output_dir") != str(source.output_root)
            or spec.get("training_state_root") != str(source.training_state_root)
            or source_identity.get("commit") != SOURCE_EXECUTION_COMMIT
            or verified_model_route.get("backbone") != "gine"
            or not _exact_int(verified_model_route.get("seed"), 7)
            or not _exact_int(spec.get("physical_gpu_index"), 1)
            or spec.get("terminal_marker") != "[TASTE_GINE_THREE_CLASS_PASS]"
            or spec.get("required_output_files")
            != list(CONTROLLER_REQUIRED_OUTPUT_FILES)
            or spec.get("worker_argv_sha256")
            != _stable_sha256(spec.get("worker_argv"))
            or not _strict_int(spec.get("resource_wait_deadline_seconds"))
            or spec.get("resource_wait_deadline_seconds") <= 0
        ):
            raise T2PassAdoptionError("deployed controller spec identity changed")
        environment = spec.get("environment_authority")
        if not isinstance(environment, Mapping) or (
            environment.get("PRIMARY_GNN_BACKBONE") != "gine"
            or environment.get("PRIMARY_SEED") != "7"
            or environment.get("RUN_TASTEMOLNET") != "1"
            or environment.get("TASTE_RESEARCH_COMPUTE_ALLOWED") != "1"
            or environment.get("TASTE_PAPER_RESULTS_ALLOWED") != "1"
            or environment.get("TASTE_DATA_REDISTRIBUTION_ALLOWED") != "0"
            or environment.get("TASTE_UPSTREAM_LICENSE_STATUS")
            != "NOT_EXPLICITLY_STATED"
            or environment.get("AUTODL_MAX_GPUS") != "4"
            or environment.get("TASTEMOLNET_GPU_INDEX") != "1"
            or not isinstance(environment.get("AUTODL_DATA_ROOT"), str)
            or not os.path.isabs(environment["AUTODL_DATA_ROOT"])
            or os.path.normpath(environment["AUTODL_DATA_ROOT"])
            != environment["AUTODL_DATA_ROOT"]
            or environment.get("AUTODL_CONTROL_ROOT") != str(source.control_root)
            or environment.get("TASTEMOLNET_GNN_FULL_OUTPUT")
            != str(source.output_root)
            or environment.get("TASTEMOLNET_GNN_TRAINING_STATE_ROOT")
            != str(source.training_state_root)
            or environment.get("TASTEMOLNET_GINE_CONTROLLER_CID") != SOURCE_CID
            or environment.get("TASTEMOLNET_GINE_CONTROLLER_ROOT")
            != str(source.controller_root)
        ):
            raise T2PassAdoptionError("deployed controller environment authority changed")
        if spec.get("environment_authority_sha256") != _stable_sha256(dict(environment)):
            raise T2PassAdoptionError("controller environment authority hash changed")
        if (
            set(state) != FAILED_CONTROLLER_STATE_FIELDS
            or not isinstance(state.get("updated_at"), str)
            or not UTC_TIMESTAMP_RE.fullmatch(state["updated_at"])
            or state.get("schema_version") != CONTROLLER_STATE_SCHEMA
            or state.get("cid") != SOURCE_CID
            or state.get("spec_sha256") != spec_canonical_sha
            or state.get("phase") != "FAILED"
            or state.get("reason") != SOURCE_FAILED_REASON
            or not _strict_int(state.get("resource_deadline_epoch_seconds"))
            or not _exact_int(state.get("attempt"), 0)
            or not _exact_int(state.get("launch_index"), 0)
            or not _exact_int(state.get("retries_used"), 0)
        ):
            raise T2PassAdoptionError("controller is not the exact failed identity-drift state")
        if "controller_terminal.json" in self.controller.files or "PASS" in self.controller.files:
            raise T2PassAdoptionError("old controller unexpectedly has terminal/PASS publication")
        claim_sentinel = claim.get("sentinel")
        claim_lock = claim.get("lock")
        if (
            set(claim)
            != {
                "schema_version",
                "cid",
                "spec_sha256",
                "root",
                "root_identity",
                "sentinel",
                "lock",
            }
            or not isinstance(claim_sentinel, Mapping)
            or set(claim_sentinel) != {"identity", "sha256"}
            or not isinstance(claim_lock, Mapping)
            or set(claim_lock) != {"identity"}
            or claim.get("schema_version") != CONTROLLER_CLAIM_SCHEMA
            or claim.get("cid") != SOURCE_CID
            or claim.get("root") != str(source.controller_root)
            or claim.get("spec_sha256") != spec_canonical_sha
            or claim.get("root_identity") != self.controller.root.identity
            or state.get("root_claim_sha256") != controller_claim_file.sha256
            or claim_sentinel.get("sha256") != controller_sentinel_file.sha256
            or claim_sentinel.get("identity") != controller_sentinel_file.identity
            or claim_lock.get("identity") != controller_lock_file.identity
            or resource_deadline.get("schema_version")
            != "autodl_tastemolnet_gine_resource_deadline_v1"
            or resource_deadline.get("cid") != SOURCE_CID
            or resource_deadline.get("spec_sha256") != spec_canonical_sha
            or resource_deadline.get("duration_seconds")
            != spec.get("resource_wait_deadline_seconds")
            or not _strict_int(resource_deadline.get("started_epoch_seconds"))
            or not _strict_int(resource_deadline.get("deadline_epoch_seconds"))
            or resource_deadline["deadline_epoch_seconds"]
            - resource_deadline["started_epoch_seconds"]
            != resource_deadline["duration_seconds"]
            or state.get("resource_deadline_sha256")
            != resource_deadline_file.sha256
            or state.get("resource_deadline_epoch_seconds")
            != resource_deadline.get("deadline_epoch_seconds")
        ):
            raise T2PassAdoptionError("controller root/spec/claim closure changed")

        execution_git = _git_identity(source.execution_project_root)
        identity_fix_git = _git_identity(source.identity_fix_project_root)
        adoption_git = _git_identity(
            self.adoption_project.path,
            critical_paths=IMPLEMENTATION_CRITICAL_BLOBS,
        )
        self._execution_git = execution_git
        self._identity_fix_git = identity_fix_git
        self._adoption_git = adoption_git
        if (
            execution_git.get("commit") != SOURCE_EXECUTION_COMMIT
            or execution_git.get("tree") != source_identity.get("tree")
            or execution_git.get("status_porcelain") != ""
            or identity_fix_git.get("commit") != SOURCE_IDENTITY_FIX_COMMIT
            or identity_fix_git.get("status_porcelain") != ""
            or adoption_git.get("status_porcelain") != ""
            or set(adoption_git.get("critical_blobs", {}))
            != set(IMPLEMENTATION_CRITICAL_BLOBS)
        ):
            raise T2PassAdoptionError(
                "583bf/3a90/adoption immutable Git identity changed"
            )
        selected_source_hashes: dict[str, str] = {}
        raw_autodl_python = environment.get("AUTODL_PYTHON")
        if (
            not isinstance(raw_autodl_python, str)
            or not os.path.isabs(raw_autodl_python)
            or os.path.normpath(raw_autodl_python) != raw_autodl_python
            or not isinstance(source_identity.get("python_executable"), str)
            or not os.path.isabs(source_identity["python_executable"])
            or os.path.normpath(source_identity["python_executable"])
            != source_identity["python_executable"]
            or not SHA256_RE.fullmatch(
                str(source_identity.get("python_executable_sha256", ""))
            )
            or not isinstance(spec.get("worker_argv"), list)
            or len(spec["worker_argv"]) != 2
            or source_identity.get("worker_program_path") != spec["worker_argv"][0]
            or spec["worker_argv"][1]
            != str(
                source.execution_project_root
                / "scripts/autodl/run_tastemolnet_gnn_full.sh"
            )
            or not SHA256_RE.fullmatch(
                str(source_identity.get("worker_program_sha256", ""))
            )
        ):
            raise T2PassAdoptionError("controller executable source identity changed")
        expected_source_paths = {
            "worker_wrapper_path": source.execution_project_root
            / "scripts/autodl/run_tastemolnet_gnn_full.sh",
            "controller_module_path": source.execution_project_root
            / "src/utils/autodl_tastemolnet_gine_controller_v1.py",
            "verified_backbone_config_path": source.execution_project_root
            / "configs/gnn/gine.yaml",
        }
        for path_field, sha_field in (
            ("worker_wrapper_path", "worker_wrapper_sha256"),
            ("controller_module_path", "controller_module_sha256"),
            ("verified_backbone_config_path", "verified_backbone_config_sha256"),
        ):
            if path_field not in source_identity:
                if path_field == "controller_module_path":
                    absolute_value = str(
                        source.execution_project_root
                        / "src/utils/autodl_tastemolnet_gine_controller_v1.py"
                    )
                else:
                    raise T2PassAdoptionError(f"controller source lacks {path_field}")
            else:
                absolute_value = source_identity[path_field]
            if absolute_value != str(expected_source_paths[path_field]):
                raise T2PassAdoptionError(
                    f"controller source path changed: {path_field}"
                )
            held = self._selected_project_file(
                self.execution_project, absolute_value, label=path_field
            )
            if held.sha256 != source_identity.get(sha_field):
                raise T2PassAdoptionError(f"controller source hash changed: {path_field}")
            selected_source_hashes[path_field] = held.sha256
        config_evidence: list[dict[str, str]] = []
        config_files = spec.get("config_files")
        if not isinstance(config_files, list) or len(config_files) != 3:
            raise T2PassAdoptionError("controller config inventory changed")
        expected_config_paths = (
            source.execution_project_root / "configs/hpc.yaml",
            source.execution_project_root / "configs/gnn/gine.yaml",
            source.execution_project_root
            / "configs/autodl/tastemolnet_gine_research_v1.yaml",
        )
        for row, expected_config_path in zip(
            config_files, expected_config_paths, strict=True
        ):
            if not isinstance(row, Mapping):
                raise T2PassAdoptionError("controller config entry is untyped")
            if row.get("path") != str(expected_config_path):
                raise T2PassAdoptionError("controller config path changed")
            held = self._selected_project_file(
                self.execution_project, row.get("path"), label="controller config"
            )
            if held.sha256 != row.get("sha256"):
                raise T2PassAdoptionError("controller config hash changed")
            config_evidence.append({"path": str(row["path"]), "sha256": held.sha256})
        fix_module = _open_selected_file(
            self.identity_fix_project,
            "src/utils/autodl_tastemolnet_gine_controller_v1.py",
        )
        self._selected.append(fix_module)
        adoption_module = _open_selected_file(
            self.adoption_project,
            "src/utils/tastemolnet_gine_pass_adoption_v1.py",
        )
        self._selected.append(adoption_module)
        release_config_file = _open_selected_file(
            self.adoption_project,
            RELEASE_CONFIG_RELATIVE,
        )
        self._selected.append(release_config_file)
        release_config = release_config_file.json()
        self._validate_release_config(release_config)
        self._release_config = release_config

        authority_file = self.run_state.file("trainer_child_authority.json")
        runtime_marker_file = self.run_state.file("state.json")
        launch_spec_file = self.run_state.file("launch_spec.json")
        authority = authority_file.json()
        runtime_marker = runtime_marker_file.json()
        launch_spec = launch_spec_file.json()
        if set(authority) != TRAINER_AUTHORITY_FIELDS or (
            authority.get("schema_version") != TRAINER_AUTHORITY_SCHEMA
            or authority.get("status") != "RELEASE_AUTHORIZED"
            or authority.get("run_id") != SOURCE_RUN_ID
            or authority.get("dataset") != "tastemolnet"
            or authority.get("stage") != SOURCE_STAGE
            or authority.get("controller_cid") != SOURCE_CID
            or authority.get("controller_root") != str(source.controller_root)
            or authority.get("project_root") != str(source.execution_project_root)
            or authority.get("authority_path")
            != str(source.run_state_root / "trainer_child_authority.json")
        ):
            raise T2PassAdoptionError("trainer authority identity changed")
        command = authority.get("trainer_command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(item, str) or not item or "\x00" in item for item in command)
            or authority.get("trainer_command_sha256") != _stable_sha256(command)
        ):
            raise T2PassAdoptionError("trainer command binding changed")
        parent_pid, _parent_start = _validate_process_snapshot(
            authority.get("parent_exp_run"), label="exp_run parent"
        )
        child_pid, _child_start = _validate_process_snapshot(
            authority.get("child_registered"), label="trainer child"
        )
        if authority["child_registered"].get("ppid") != parent_pid:
            raise T2PassAdoptionError("trainer ancestry changed")
        barrier = authority.get("barrier_record")
        barrier_lock_file = self.run_state.file("trainer-startup.lock")
        barrier_record_file = self.run_state.file("trainer-startup.json")
        launcher_argv = (
            barrier.get("launcher_argv") if isinstance(barrier, Mapping) else None
        )
        expected_launcher_argv = (
            [
                barrier.get("python_executable"),
                "-S",
                "-m",
                "src.utils.autodl_exec_startup_barrier",
                "--record",
                barrier.get("record_path"),
                "--release-read-fd",
                str(barrier.get("release_read_fd")),
                "--lock-fd",
                str(barrier.get("lock_fd")),
                "--",
                *command,
            ]
            if isinstance(barrier, Mapping)
            else None
        )
        if (
            not isinstance(barrier, Mapping)
            or set(barrier) != STARTUP_BARRIER_FIELDS
            or barrier.get("schema") != "autodl_exec_startup_barrier_v1"
            or barrier.get("kind") != "durable_exec_startup_barrier"
            or barrier.get("state") != "ARMED_UNRELEASED"
            or barrier.get("lock_path")
            != str(source.run_state_root / "trainer-startup.lock")
            or barrier.get("record_path")
            != str(source.run_state_root / "trainer-startup.json")
            or barrier_record_file.json() != dict(barrier)
            or barrier.get("lock_dev") != barrier_lock_file.identity["device"]
            or barrier.get("lock_inode") != barrier_lock_file.identity["inode"]
            or barrier.get("lock_mode")
            != stat.S_IMODE(barrier_lock_file.identity["mode"])
            or barrier.get("lock_mode") != 0o600
            or barrier.get("lock_uid") != barrier_lock_file.identity["uid"]
            or barrier.get("lock_uid") != os.getuid()
            or barrier.get("lock_nlink") != barrier_lock_file.identity["nlink"]
            or barrier.get("lock_nlink") != 1
            or not _strict_int(barrier.get("release_read_fd"))
            or barrier["release_read_fd"] < 0
            or not _strict_int(barrier.get("lock_fd"))
            or barrier["lock_fd"] < 0
            or not _strict_int(barrier.get("release_token_bytes"))
            or barrier["release_token_bytes"] != 32
            or not SHA256_RE.fullmatch(str(barrier.get("release_token_sha256", "")))
            or barrier.get("target_argv") != command
            or barrier.get("target_argv_sha256") != _stable_sha256(command)
            or not isinstance(launcher_argv, list)
            or not launcher_argv
            or any(
                not isinstance(item, str) or not item or "\x00" in item
                for item in launcher_argv
            )
            or launcher_argv != expected_launcher_argv
            or barrier.get("launcher_argv_sha256")
            != _stable_sha256(launcher_argv)
            or authority["child_registered"].get("argv") != launcher_argv
            or authority["parent_exp_run"].get("cwd")
            != str(source.execution_project_root)
            or authority["child_registered"].get("cwd")
            != str(source.execution_project_root)
            or barrier.get("python_executable") != raw_autodl_python
            or authority["parent_exp_run"].get("exe")
            != source_identity.get("python_executable")
            or authority["child_registered"].get("exe")
            != source_identity.get("python_executable")
            or authority["parent_exp_run"].get("exe_identity")
            != authority["child_registered"].get("exe_identity")
        ):
            raise T2PassAdoptionError("trainer startup barrier binding changed")
        gpu_uuid = launch_spec.get("gpu_uuid")
        launch_input_manifest = launch_spec.get("input_manifest")
        if (
            not _exact_int(launch_spec.get("schema_version"), 1)
            or launch_spec.get("run_id") != SOURCE_RUN_ID
            or launch_spec.get("project_root") != str(source.execution_project_root)
            or launch_spec.get("data_root") != environment.get("AUTODL_DATA_ROOT")
            or launch_spec.get("control_root") != str(source.control_root)
            or launch_spec.get("dataset") != "tastemolnet"
            or launch_spec.get("stage") != SOURCE_STAGE
            or launch_spec.get("command") != command
            or launch_spec.get("python_executable")
            != source_identity.get("python_executable")
            or not _exact_int(launch_spec.get("gpu_index"), 1)
            or not isinstance(gpu_uuid, str)
            or not GPU_UUID_RE.fullmatch(gpu_uuid)
            or not _exact_int(launch_spec.get("max_gpus"), 4)
            or not _exact_int(launch_spec.get("gpu_hard_limit"), 4)
            or launch_spec.get("git_commit") != SOURCE_EXECUTION_COMMIT
            or launch_spec.get("config_files")
            != [str(path) for path in expected_config_paths]
            or (
                launch_input_manifest is not None
                and (
                    not isinstance(launch_input_manifest, str)
                    or not os.path.isabs(launch_input_manifest)
                    or os.path.normpath(launch_input_manifest)
                    != launch_input_manifest
                )
            )
            or launch_spec.get("expected_output") != str(source.output_root)
            or launch_spec.get("resume_published_output_receipt") is not None
            or launch_spec.get("resume_published_output_receipt_sha256") is not None
            or launch_spec.get("required_output_files")
            != list(CONTROLLER_REQUIRED_OUTPUT_FILES)
            or launch_spec.get("required_output_any") != []
            or launch_spec.get("required_absolute_output_files") != []
            or launch_spec.get("required_log_marker")
            != "[TASTE_GINE_THREE_CLASS_PASS]"
            or launch_spec.get("heavy") is not True
        ):
            raise T2PassAdoptionError("exp_run launch/runtime contract changed")
        expected_parent_argv = [
            raw_autodl_python,
            str(source.execution_project_root / "scripts/autodl/exp_run.py"),
            "--project-root",
            str(source.execution_project_root),
            "--data-root",
            environment["AUTODL_DATA_ROOT"],
            "launch",
            "--dataset",
            "tastemolnet",
            "--stage",
            SOURCE_STAGE,
            "--gpu-index",
            "1",
            "--gpu-uuid",
            gpu_uuid,
            "--gpu-required",
            "--heavy",
            "--max-gpus",
            "4",
            "--gpu-hard-limit",
            "4",
            "--foreground",
        ]
        for config_path in expected_config_paths:
            expected_parent_argv.extend(("--config-file", str(config_path)))
        if launch_input_manifest is not None:
            expected_parent_argv.extend(("--input-manifest", launch_input_manifest))
        expected_parent_argv.extend(
            ("--expected-output", str(source.output_root))
        )
        for required_name in CONTROLLER_REQUIRED_OUTPUT_FILES:
            expected_parent_argv.extend(("--required-output-file", required_name))
        expected_parent_argv.extend(
            (
                "--required-log-marker",
                "[TASTE_GINE_THREE_CLASS_PASS]",
                "--",
                *command,
            )
        )
        if authority["parent_exp_run"].get("argv") != expected_parent_argv:
            raise T2PassAdoptionError("exp_run parent argv differs from deployed wrapper")
        runtime_log_path = _absolute(
            launch_spec.get("log_path"), label="source runtime log"
        )
        try:
            runtime_log_path.relative_to(source.runtime_root / "logs")
        except ValueError as exc:
            raise T2PassAdoptionError("source runtime log is outside runtime logs") from exc
        self.runtime_log_parent = HeldDirectory(
            runtime_log_path.parent, label="source runtime log parent"
        )
        self.runtime_log_parent.open()
        self.runtime_log = _open_selected_file(
            self.runtime_log_parent, runtime_log_path.name
        )
        self._selected.append(self.runtime_log)
        try:
            runtime_log_text = self.runtime_log.bytes(limit=64 * 1024**2).decode(
                "utf-8"
            )
        except UnicodeDecodeError as exc:
            raise T2PassAdoptionError("source runtime log is not UTF-8") from exc
        for marker in (
            "[TASTE_GINE_THREE_CLASS_PASS]",
            "[MOLECULAR_GNN_TRAIN_OK]",
            "[AUTODL_RUN_EXIT] exit_code=0",
        ):
            if marker not in runtime_log_text:
                raise T2PassAdoptionError(
                    f"source runtime log lacks required marker {marker}"
                )
        if (
            not _exact_int(runtime_marker.get("schema_version"), 1)
            or runtime_marker.get("run_id") != SOURCE_RUN_ID
            or runtime_marker.get("dataset") != "tastemolnet"
            or runtime_marker.get("stage") != SOURCE_STAGE
            or runtime_marker.get("state") != "PASS"
            or not _exact_int(runtime_marker.get("exit_code"), 0)
            or not _strict_int(runtime_marker.get("pid"))
            or runtime_marker.get("pid") != parent_pid
            or not _strict_int(runtime_marker.get("child_pid"))
            or runtime_marker.get("child_pid") != child_pid
            or not _exact_int(runtime_marker.get("gpu_index"), 1)
            or runtime_marker.get("gpu_uuid") != gpu_uuid
            or runtime_marker.get("failures") != []
            or runtime_marker.get("log_path") != str(runtime_log_path)
        ):
            raise T2PassAdoptionError("runtime PASS marker changed")
        registry_rows = [
            row for row in _json_line_rows(self.registry) if row.get("run_id") == SOURCE_RUN_ID
        ]
        if not registry_rows:
            raise T2PassAdoptionError("run registry lacks the source run")
        final_registry = registry_rows[-1]
        if (
            not _exact_int(final_registry.get("schema_version"), 1)
            or final_registry.get("dataset") != "tastemolnet"
            or final_registry.get("stage") != SOURCE_STAGE
            or final_registry.get("state") != "PASS"
            or not _exact_int(final_registry.get("exit_code"), 0)
            or not _strict_int(final_registry.get("pid"))
            or final_registry.get("pid") != parent_pid
            or not _exact_int(final_registry.get("gpu_index"), 1)
            or final_registry.get("gpu_uuid") != gpu_uuid
            or final_registry.get("backend") != "autodl"
            or final_registry.get("command") != command
            or final_registry.get("expected_output") != str(source.output_root)
            or final_registry.get("git_commit") != SOURCE_EXECUTION_COMMIT
            or final_registry.get("log_path") != str(runtime_log_path)
        ):
            raise T2PassAdoptionError("run registry terminal event changed")
        for row in registry_rows:
            if "pid" in row and (
                not _strict_int(row["pid"]) or int(row["pid"]) <= 0
            ):
                raise T2PassAdoptionError("run registry declares an untyped PID")
        self._dead_pids = tuple(sorted({parent_pid, child_pid}))
        for pid in self._dead_pids:
            _assert_pid_absent(self.proc, pid)

        if set(self.output.files) != CHECKPOINT_FILES or any(
            row["kind"] == "directory" for row in self.output.inventory
        ):
            raise T2PassAdoptionError("scientific output inventory is not the exact 19-file bundle")
        checkpoint_hashes = _parse_checkpoint_hashes(
            self.output.file("sha256sums.txt")
        )
        for relative, expected in checkpoint_hashes.items():
            if self.output.file(relative).sha256 != expected:
                raise T2PassAdoptionError(f"checkpoint hash mismatch: {relative}")
        output_closure = self._validate_output_bundle()
        training_closure = self._validate_training_state(output_closure)

        evidence = {
            "schema_version": "tastemolnet_t2_gine_pass_source_evidence_v1",
            "source_result": {
                "dataset": "tastemolnet",
                "stage": "T2_GINE_FULL",
                "source_stage": SOURCE_STAGE,
                "source_cid": SOURCE_CID,
                "source_run_id": SOURCE_RUN_ID,
                "backbone": "gine",
                "seed": 7,
                "num_classes": 3,
                "label_map": {"0": "Bitter", "1": "Sweet", "2": "Tasteless"},
                "source_label": 1,
                "source_name": "Sweet",
                "strict_flip": "pred_before == 1 and pred_after != 1",
            },
            "failed_controller": {
                "root": str(source.controller_root),
                "phase": "FAILED",
                "reason": SOURCE_FAILED_REASON,
                "attempt": state["attempt"],
                "launch_index": state["launch_index"],
                "spec_sha256": controller_spec_file.sha256,
                "state_sha256": controller_state_file.sha256,
                "claim_sha256": controller_claim_file.sha256,
                "inventory": self.controller.inventory,
                "inventory_sha256": self.controller.inventory_sha256,
                "scientific_false_negative": True,
                "old_controller_mutated": False,
            },
            "source_code": {
                "execution_project_root": str(source.execution_project_root),
                "execution_git": execution_git,
                "identity_fix_project_root": str(source.identity_fix_project_root),
                "identity_fix_git": identity_fix_git,
                "adoption_project_root": str(self.adoption_project.path),
                "adoption_git": adoption_git,
                "selected_source_hashes": selected_source_hashes,
                "config_files": config_evidence,
                "identity_fix_controller_module_sha256": fix_module.sha256,
                "adoption_module_sha256": adoption_module.sha256,
                "release_config_path": str(
                    self.adoption_project.path / RELEASE_CONFIG_RELATIVE
                ),
                "release_config_sha256": release_config_file.sha256,
                "release_config": release_config,
                "release_model": "external_receipt_plus_one_parent_config_delta",
            },
            "run_authority": {
                "run_state_root": str(source.run_state_root),
                "run_state_inventory": self.run_state.inventory,
                "run_state_inventory_sha256": self.run_state.inventory_sha256,
                "registry_path": str(source.registry_path),
                "registry_matching_rows": registry_rows,
                "registry_run_closure_sha256": _stable_sha256(registry_rows),
                "registry_status": "PASS",
                "runtime_marker_path": str(source.run_state_root / "state.json"),
                "runtime_marker_sha256": runtime_marker_file.sha256,
                "runtime_status": "PASS",
                "runtime_log_path": str(runtime_log_path),
                "runtime_log_sha256": self.runtime_log.sha256,
                "runtime_log_markers_verified": [
                    "[TASTE_GINE_THREE_CLASS_PASS]",
                    "[MOLECULAR_GNN_TRAIN_OK]",
                    "[AUTODL_RUN_EXIT] exit_code=0",
                ],
                "trainer_authority_path": str(
                    source.run_state_root / "trainer_child_authority.json"
                ),
                "trainer_authority_sha256": authority_file.sha256,
                "trainer_python_raw_argv_token": barrier["python_executable"],
                "trainer_python_physical_executable": source_identity[
                    "python_executable"
                ],
                "raw_and_physical_python_identities_kept_distinct": True,
                "all_declared_pids_dead": True,
                "dead_pids": list(self._dead_pids),
                "physical_gpu_index": 1,
                "physical_gpu_uuid": gpu_uuid,
            },
            "scientific_output": {
                "root": str(source.output_root),
                "status": "PASS",
                "inventory": self.output.inventory,
                "inventory_sha256": self.output.inventory_sha256,
                "sha256sums_sha256": self.output.file("sha256sums.txt").sha256,
                "hash_count": 18,
                **output_closure,
                "old_output_mutated": False,
            },
            "training_state": {
                "root": str(source.training_state_root),
                "training_complete_status": "PASS",
                "inventory": self.training.inventory,
                "inventory_sha256": self.training.inventory_sha256,
                **training_closure,
                "old_training_state_mutated": False,
            },
            "adoption_boundary": {
                "control_root": str(source.control_root),
                "adoption_root": str(source.adoption_root),
                "exact_formula": "<control_root>/tastemolnet-t2-gine-pass-adoption-v1/<source_cid>",
                "non_scientific": True,
                "science_executed": False,
                "main_controller_mutated": False,
                "matrix_mutated": False,
                "gpu_lock_mutated": False,
                "old_source_authorities_revalidated_through_gate_commit": True,
            },
        }
        evidence["source_evidence_sha256"] = _stable_sha256(evidence)
        return evidence

    def _validate_output_bundle(self) -> dict[str, Any]:
        model_card = self.output.file("model_card.json").json()
        label_map = self.output.file("label_map.json").json()
        oracle = self.output.file("oracle_manifest.json").json()
        policy = self.output.file("data_use_policy_binding.json").json()
        cache = self.output.file("graph_cache_usage.json").json()
        reload_receipt = self.output.file("checkpoint_reload.json").json()
        last = self.output.file("last_checkpoint.json").json()
        test = self.output.file("test_evaluation_status.json").json()
        metrics = self.output.file("training_metrics.json").json()
        config = self.output.file("config.yaml").json()
        git_state = self.output.file("git_state.json").json()
        policy_evidence = policy.get("policy")
        expected_labels = {"0": "Bitter", "1": "Sweet", "2": "Tasteless"}
        health = oracle.get("health_gate")
        predicted = health.get("predicted_classes") if isinstance(health, Mapping) else None
        gnn_config = config.get("gnn")
        training_config = config.get("training")
        final_validation = metrics.get("final_validation")
        if (
            model_card.get("dataset") != "tastemolnet"
            or model_card.get("profile") != "full"
            or model_card.get("backbone") != "gine"
            or model_card.get("oracle_backend") != "gnn"
            or model_card.get("rf_oracle_used") is not False
            or not _exact_int(model_card.get("num_classes"), 3)
            or not _exact_int(model_card.get("source_label"), 1)
            or not _exact_int(model_card.get("seed"), 7)
            or model_card.get("training_commit") != SOURCE_EXECUTION_COMMIT
            or model_card.get("test_loaded_during_training") is not False
            or model_card.get("test_used_for_model_fit_or_selection") is not False
            or model_card.get("selection_split") != "validation"
            or model_card.get("selection_metric") != "macro_ovr_roc_auc"
            or model_card.get("selection_tiebreak_metric") != "macro_f1"
            or model_card.get("checkpoint_id")
            != self.output.file("model.pt").sha256
            or model_card.get("paper_result_reporting_allowed") is not True
            or model_card.get("dataset_redistributed") is not False
            or model_card.get("upstream_license_not_explicit") is not True
            or model_card.get("license_pass_claimed") is not False
            or not isinstance(gnn_config, Mapping)
            or gnn_config.get("backbone") != "gine"
            or not _exact_int(gnn_config.get("num_classes"), 3)
            or not isinstance(training_config, Mapping)
            or not _exact_int(training_config.get("primary_seed"), 7)
            or training_config.get("class_weighted_loss") is not True
            or training_config.get("weighted_sampler") is not False
            or label_map != expected_labels
            or oracle.get("schema_version")
            != "tastemolnet_three_class_gine_oracle_manifest_v1"
            or oracle.get("status") != "PASS"
            or oracle.get("classifier_family") != "gine"
            or oracle.get("oracle_backend") != "gnn"
            or oracle.get("rf_oracle_used") is not False
            or oracle.get("checkpoint_id")
            != self.output.file("model.pt").sha256
            or not _exact_int(oracle.get("num_classes"), 3)
            or oracle.get("label_map") != expected_labels
            or not _exact_int(oracle.get("source_label"), 1)
            or oracle.get("source_label_name") != "Sweet"
            or oracle.get("test_loaded") is not False
            or oracle.get("test_evaluated") is not False
            or not isinstance(health, Mapping)
            or health.get("status") != "PASS"
            or predicted != [0, 1, 2]
            or not isinstance(predicted, list)
            or any(not _strict_int(value) for value in predicted)
            or model_card.get("health_gate") != health
            or metrics.get("health_gate") != health
            or metrics.get("selection_metric") != "macro_ovr_roc_auc"
            or metrics.get("selection_tiebreak_metric") != "macro_f1"
            or metrics.get("class_weighted_loss") is not True
            or metrics.get("weighted_sampler") is not False
            or not isinstance(final_validation, Mapping)
        ):
            raise T2PassAdoptionError("three-class GINE/health-gate closure changed")
        for metric_name in ("macro_ovr_roc_auc", "macro_f1"):
            metric_value = final_validation.get(metric_name)
            if (
                not isinstance(metric_value, (int, float))
                or isinstance(metric_value, bool)
                or not math.isfinite(float(metric_value))
            ):
                raise T2PassAdoptionError(
                    f"Taste validation {metric_name} is unavailable or non-finite"
                )
        per_class = final_validation.get("per_class")
        if not isinstance(per_class, Mapping):
            raise T2PassAdoptionError("Taste validation per-class metrics are unavailable")
        for label in ("0", "1", "2"):
            row = per_class.get(label)
            recall = row.get("recall") if isinstance(row, Mapping) else None
            if (
                not isinstance(recall, (int, float))
                or isinstance(recall, bool)
                or not math.isfinite(float(recall))
                or float(recall) <= 0.0
            ):
                raise T2PassAdoptionError(
                    "one Taste validation class has non-positive recall"
                )
        if (
            policy.get("schema_version") != "tastemolnet_training_policy_binding_v1"
            or policy.get("dataset") != "tastemolnet"
            or policy.get("status") != "NOT_EXPLICITLY_STATED"
            or policy.get("authorization_status")
            != "RESEARCH_REPORTING_ALLOWED_NO_REDISTRIBUTION"
            or not isinstance(policy_evidence, Mapping)
            or not _exact_int(policy_evidence.get("policy_version"), 2)
            or policy_evidence.get("authorization_state")
            != "ACTIVE_SCOPED_AUTHORIZATION"
            or policy_evidence.get("authorization_status")
            != "RESEARCH_REPORTING_ALLOWED_NO_REDISTRIBUTION"
            or policy_evidence.get("research_execution_allowed") is not True
            or policy_evidence.get("paper_reporting_allowed") is not True
            or policy_evidence.get("research_compute_allowed") is not True
            or policy_evidence.get("paper_result_reporting_allowed") is not True
            or policy_evidence.get("data_redistribution_allowed") is not False
            or policy_evidence.get("dataset_redistributed") is not False
            or policy_evidence.get("main_route_state") != "READY_FOR_MAIN_ROUTE"
            or policy_evidence.get("license_conclusion") != "NOT_GRANTED_OR_INFERRED"
            or policy.get("paper_result_reporting_allowed") is not True
            or policy.get("paper_results_reporting_allowed_by_project_policy")
            is not True
            or policy.get("data_redistribution_allowed") is not False
            or policy.get("dataset_redistributed") is not False
            or policy.get("upstream_license_not_explicit") is not True
            or policy.get("upstream_license_status") != "NOT_EXPLICITLY_STATED"
            or policy.get("upstream_license_claimed_resolved") is not False
            or policy.get("license_pass_claimed") is not False
            or policy.get("public_artifact_audit_required") is not True
            or policy.get("hpc_execution_authorized") is not False
            or cache.get("schema_version") != "tastemolnet_graph_cache_usage_v1"
            or cache.get("dataset") != "tastemolnet"
            or cache.get("mode") != "read_only_existing_cache"
            or cache.get("loaded_splits") != ["train", "validation"]
            or cache.get("calibration_loaded") is not False
            or cache.get("test_loaded") is not False
            or cache.get("test_metadata_hash_only") is not True
            or cache.get("graph_cache_rebuilt") is not False
            or cache.get("data_reprepared") is not False
            or test.get("status") != "NOT_EVALUATED"
            or test.get("test_loaded") is not False
            or oracle.get("selection_split") != "validation"
            or oracle.get("selection_metric") != "macro_ovr_roc_auc"
            or oracle.get("selection_tiebreak_metric") != "macro_f1"
            or oracle.get("temperature_calibration_split") != "validation"
            or oracle.get("paper_result_reporting_allowed") is not True
            or oracle.get("dataset_redistributed") is not False
            or oracle.get("upstream_license_not_explicit") is not True
        ):
            raise T2PassAdoptionError("Taste policy/cache/test boundary changed")
        serialized = json.dumps(
            [model_card, oracle, policy, cache], sort_keys=True, ensure_ascii=True
        )
        if "TASTE_LICENSE_PASS" in serialized or "LICENSE_PASS" in serialized:
            raise T2PassAdoptionError("scientific output makes a forbidden licence claim")
        if (
            reload_receipt.get("schema_version")
            != "tastemolnet_gine_checkpoint_reload_v1"
            or reload_receipt.get("status") != "PASS"
            or reload_receipt.get("checkpoint_reload_pass") is not True
            or reload_receipt.get("batch_single_probability_equivalence") is not True
            or reload_receipt.get("all_probabilities_finite") is not True
            or not _exact_int(reload_receipt.get("num_classes"), 3)
            or not _exact_int(reload_receipt.get("source_label"), 1)
            or reload_receipt.get("checkpoint_id")
            != self.output.file("model.pt").sha256
            or reload_receipt.get("last_checkpoint") != last
            or last.get("schema_version") != "tastemolnet_last_training_checkpoint_v1"
            or last.get("checkpoint_file") != "last.pt"
            or last.get("same_bytes_as_latest_epoch_checkpoint") is not True
            or not _strict_int(last.get("completed_epoch"))
            or last.get("completed_epoch") < 1
            or last.get("checkpoint_sha256") != self.output.file("last.pt").sha256
            or last.get("source_checkpoint_sha256") != last.get("checkpoint_sha256")
        ):
            raise T2PassAdoptionError("model/last checkpoint reload closure changed")
        if git_state.get("commit") != SOURCE_EXECUTION_COMMIT:
            raise T2PassAdoptionError("output Git commit differs from deployed execution")
        return {
            "model_sha256": self.output.file("model.pt").sha256,
            "last_sha256": self.output.file("last.pt").sha256,
            "checkpoint_id": reload_receipt["checkpoint_id"],
            "training_resume_contract_sha256": model_card.get(
                "training_resume_contract_sha256"
            ),
            "completed_epoch": last["completed_epoch"],
            "health_gate_status": "PASS",
            "test_loaded": False,
            "calibration_cache_loaded": False,
            "rf_oracle_used": False,
            "research_compute_allowed": True,
            "paper_result_reporting_allowed": True,
            "data_redistribution_allowed": False,
        }

    def _validate_training_state(self, output: Mapping[str, Any]) -> dict[str, Any]:
        if stat.S_IMODE(self.training.root.identity["mode"]) != 0o700:
            raise T2PassAdoptionError("training-state root is not owner-private mode 0700")
        if self.training.root.fd is None:
            raise T2PassAdoptionError("training-state root authority is closed")
        root_claim_file = self.training.file("root_claim.json")
        root_claim = root_claim_file.json()
        training_root_claim_identity = root_claim.get("root_identity")
        state_sentinel = self.training.file(".root_identity")
        writer_lock = self.training.file(".writer.lock")
        completion_file = self.training.file("training_complete.json")
        completion = completion_file.json()
        contract_file = self.training.file("training_contract.json")
        contract_payload = contract_file.json()
        latest = self.training.file("latest_checkpoint.json").json()
        contract = contract_payload.get("contract")
        if not isinstance(contract, Mapping):
            raise T2PassAdoptionError("training contract is untyped")
        contract_source = contract.get("source_identity")
        contract_model = contract.get("model_config")
        contract_training = contract.get("training")
        contract_taste = contract.get("tastemolnet_scoped_authority")
        contract_policy = (
            contract_taste.get("policy")
            if isinstance(contract_taste, Mapping)
            else None
        )
        if (
            contract.get("schema_version")
            != "molecular_gnn_training_resume_contract_v1"
            or contract.get("dataset") != "tastemolnet"
            or contract.get("profile") != "full"
            or contract.get("output_dir") != str(self.sources.output_root)
            or not isinstance(contract_source, Mapping)
            or contract_source.get("commit") != SOURCE_EXECUTION_COMMIT
            or not isinstance(contract_model, Mapping)
            or contract_model.get("backbone") != "gine"
            or not _exact_int(contract_model.get("num_classes"), 3)
            or not _exact_int(contract_model.get("num_layers"), 5)
            or not _exact_int(contract_model.get("hidden_dim"), 256)
            or contract_model.get("dropout") != 0.2
            or contract_model.get("pooling") != "mean"
            or not _exact_int(contract_model.get("readout_layers"), 2)
            or contract_model.get("normalization") != "batch_norm"
            or contract_model.get("residual") is not True
            or contract_model.get("edge_feature_mode")
            != "native_edge_conditioned_message"
            or not isinstance(contract_training, Mapping)
            or not _exact_int(contract_training.get("max_epochs"), 200)
            or not _exact_int(
                contract_training.get("early_stopping_patience"), 20
            )
            or not _exact_int(contract_training.get("batch_size"), 64)
            or contract_training.get("learning_rate") != 0.001
            or contract_training.get("weight_decay") != 0.00001
            or not _exact_int(contract_training.get("seed"), 7)
            or contract_training.get("class_weighted_loss") is not True
            or contract_training.get("weighted_sampler") is not False
            or contract_training.get("selection_metric")
            != "macro_ovr_roc_auc"
            or contract_training.get("selection_tiebreak_metric") != "macro_f1"
            or contract_training.get("gradient_clip_norm") != 5.0
            or not isinstance(contract_taste, Mapping)
            or not isinstance(contract_policy, Mapping)
            or not _exact_int(contract_policy.get("policy_version"), 2)
            or contract_policy.get("authorization_state")
            != "ACTIVE_SCOPED_AUTHORIZATION"
            or contract_policy.get("research_compute_allowed") is not True
            or contract_policy.get("paper_result_reporting_allowed") is not True
            or contract_policy.get("data_redistribution_allowed") is not False
        ):
            raise T2PassAdoptionError(
                "training contract no longer binds the exact Taste GINE route"
            )
        contract_sha = _stable_sha256(dict(contract))
        contract_evidence = {
            "schema_version": "molecular_gnn_training_contract_physical_v1",
            "name": "training_contract.json",
            "identity": contract_file.identity,
            "file_sha256": contract_file.sha256,
            "canonical_sha256": contract_sha,
            "content": contract_payload,
        }
        latest_name = latest.get("checkpoint_file")
        if (
            root_claim.get("schema_version") != TRAINING_STATE_SCHEMA
            or root_claim.get("artifact_kind")
            != "molecular_gnn_training_state_root_claim"
            or root_claim.get("root") != str(self.sources.training_state_root)
            or not isinstance(training_root_claim_identity, Mapping)
            or set(training_root_claim_identity) != set(
                _file_identity(os.fstat(self.training.root.fd))
            )
            or any(
                not _strict_int(value)
                for value in training_root_claim_identity.values()
            )
            or {
                key: training_root_claim_identity.get(key)
                for key in self.training.root.identity
            }
            != self.training.root.identity
            or root_claim.get("sentinel", {}).get("name") != ".root_identity"
            or root_claim.get("sentinel", {}).get("sha256") != state_sentinel.sha256
            or root_claim.get("sentinel", {}).get("identity")
            != state_sentinel.identity
            or root_claim.get("lock", {}).get("name") != ".writer.lock"
            or root_claim.get("lock", {}).get("identity") != writer_lock.identity
            or stat.S_IMODE(state_sentinel.identity["mode"]) != 0o600
            or stat.S_IMODE(writer_lock.identity["mode"]) != 0o600
            or contract_payload.get("root_claim_sha256") != root_claim_file.sha256
            or set(contract_payload)
            != {
                "schema_version",
                "artifact_kind",
                "contract",
                "contract_sha256",
                "root_claim_sha256",
            }
            or contract_payload.get("schema_version") != TRAINING_STATE_SCHEMA
            or contract_payload.get("artifact_kind") != "molecular_gnn_training_contract"
            or contract_payload.get("contract_sha256") != contract_sha
            or completion.get("schema_version") != TRAINING_STATE_SCHEMA
            or completion.get("artifact_kind") != "molecular_gnn_training_complete"
            or completion.get("status") != "PASS"
            or completion.get("contract_sha256") != contract_sha
            or completion.get("training_contract_evidence") != contract_evidence
            or completion.get("output_dir") != str(self.sources.output_root)
            or latest.get("schema_version") != TRAINING_STATE_SCHEMA
            or latest.get("status") != "CHECKPOINT_COMPLETE"
            or latest.get("contract_sha256") != contract_sha
            or latest.get("training_contract_evidence") != contract_evidence
            or not isinstance(latest_name, str)
            or PurePosixPath(latest_name).name != latest_name
            or latest_name not in self.training.files
            or not _strict_int(latest.get("completed_epoch"))
            or latest["completed_epoch"] < 1
            or latest_name != f"checkpoint-{latest['completed_epoch']:06d}.pt"
            or latest.get("next_epoch") != latest["completed_epoch"] + 1
            or latest.get("checkpoint_bytes")
            != self.training.file(latest_name).identity["size"]
            or latest.get("checkpoint_sha256") != self.training.file(latest_name).sha256
            or latest.get("checkpoint_sha256") != output.get("last_sha256")
            or latest.get("completed_epoch") != output.get("completed_epoch")
            or output.get("training_resume_contract_sha256") != contract_sha
        ):
            raise T2PassAdoptionError("training-complete/latest checkpoint closure changed")
        output_identity = completion.get("output_identity")
        if not isinstance(output_identity, Mapping) or (
            output_identity.get("model_sha256") != output.get("model_sha256")
            or output_identity.get("model_card_sha256")
            != self.output.file("model_card.json").sha256
            or output_identity.get("sha256s_sha256")
            != self.output.file("sha256sums.txt").sha256
            or output_identity.get("checkpoint_id") != output.get("checkpoint_id")
            or output_identity.get("training_resume_contract_sha256") != contract_sha
        ):
            raise T2PassAdoptionError("training completion output identity changed")
        return {
            "training_complete_sha256": completion_file.sha256,
            "contract_sha256": contract_sha,
            "latest_checkpoint_file": latest_name,
            "latest_checkpoint_sha256": latest["checkpoint_sha256"],
            "completed_epoch": latest["completed_epoch"],
        }

    def verify(self) -> None:
        self.control.verify()
        self.proc.verify()
        self.controller.verify()
        self.output.verify()
        self.training.verify()
        self.run_state.verify()
        self.execution_project.verify()
        self.identity_fix_project.verify()
        self.adoption_project.verify()
        if self.runtime_log_parent is not None:
            self.runtime_log_parent.verify()
        if self.external_release_parent is not None:
            self.external_release_parent.verify()
        if (
            self._execution_git is not None
            and _git_identity(self.sources.execution_project_root)
            != self._execution_git
        ):
            raise T2PassAdoptionError("execution Git identity changed while held")
        if (
            self._identity_fix_git is not None
            and _git_identity(self.sources.identity_fix_project_root)
            != self._identity_fix_git
        ):
            raise T2PassAdoptionError("identity-fix Git identity changed while held")
        if (
            self._adoption_git is not None
            and _git_identity(
                self.adoption_project.path,
                critical_paths=IMPLEMENTATION_CRITICAL_BLOBS,
            )
            != self._adoption_git
        ):
            raise T2PassAdoptionError("adoption implementation Git identity changed while held")
        for pid in self._dead_pids:
            _assert_pid_absent(self.proc, pid)
        assert self.registry is not None
        for held in self._selected:
            held.verify()

    def close(self) -> None:
        for held in reversed(self._selected):
            try:
                _close_selected_file(held)
            except BaseException:
                pass
        self._selected.clear()
        self.registry = None
        self.runtime_log = None
        self._external_release_file = None
        for tree in (self.run_state, self.training, self.output, self.controller):
            tree.close()
        for directory in (
            self.adoption_project,
            self.identity_fix_project,
            self.execution_project,
            self.proc,
            self.control,
        ):
            directory.close()
        if self.runtime_log_parent is not None:
            self.runtime_log_parent.close()
            self.runtime_log_parent = None
        if self.external_release_parent is not None:
            self.external_release_parent.close()
            self.external_release_parent = None

    def __exit__(self, *_args: object) -> None:
        self.close()


def preflight_t2_gine_pass_adoption(
    sources: T2PassAdoptionSources,
) -> dict[str, Any]:
    """Read and close a complete source snapshot; never creates output."""

    with _SourceHold(sources) as hold:
        assert hold.evidence is not None
        evidence = json.loads(json.dumps(hold.evidence))
        hold.verify()
    return evidence


def _observed_release_values(evidence: Mapping[str, Any]) -> dict[str, str]:
    return {
        "control_root": evidence["adoption_boundary"]["control_root"],
        "controller_root": evidence["failed_controller"]["root"],
        "scientific_output_root": evidence["scientific_output"]["root"],
        "training_state_root": evidence["training_state"]["root"],
        "execution_project_root": evidence["source_code"]["execution_project_root"],
        "identity_fix_project_root": evidence["source_code"]["identity_fix_project_root"],
        "adoption_project_root": evidence["source_code"]["adoption_project_root"],
        "runtime_log_path": evidence["run_authority"]["runtime_log_path"],
        "controller_inventory_sha256": evidence["failed_controller"]["inventory_sha256"],
        "run_state_inventory_sha256": evidence["run_authority"][
            "run_state_inventory_sha256"
        ],
        "output_inventory_sha256": evidence["scientific_output"]["inventory_sha256"],
        "training_state_inventory_sha256": evidence["training_state"]["inventory_sha256"],
        "registry_run_closure_sha256": evidence["run_authority"]["registry_run_closure_sha256"],
        "runtime_marker_sha256": evidence["run_authority"]["runtime_marker_sha256"],
        "runtime_log_sha256": evidence["run_authority"]["runtime_log_sha256"],
        "trainer_authority_sha256": evidence["run_authority"]["trainer_authority_sha256"],
        "training_complete_sha256": evidence["training_state"]["training_complete_sha256"],
        "execution_commit": evidence["source_code"]["execution_git"]["commit"],
        "execution_tree": evidence["source_code"]["execution_git"]["tree"],
        "identity_fix_commit": evidence["source_code"]["identity_fix_git"]["commit"],
        "identity_fix_tree": evidence["source_code"]["identity_fix_git"]["tree"],
        "git_binary_sha256": evidence["source_code"]["adoption_git"][
            "git_binary_sha256"
        ],
    }


_RELEASE_PATH_FIELDS = frozenset(
    {
        "control_root",
        "controller_root",
        "scientific_output_root",
        "training_state_root",
        "execution_project_root",
        "identity_fix_project_root",
        "adoption_project_root",
        "runtime_log_path",
    }
)
_RELEASE_SHA256_FIELDS = frozenset(
    {
        "controller_inventory_sha256",
        "run_state_inventory_sha256",
        "output_inventory_sha256",
        "training_state_inventory_sha256",
        "registry_run_closure_sha256",
        "runtime_marker_sha256",
        "runtime_log_sha256",
        "trainer_authority_sha256",
        "training_complete_sha256",
        "git_binary_sha256",
    }
)
_RELEASE_SHA1_FIELDS = frozenset(
    {
        "execution_commit",
        "execution_tree",
        "identity_fix_commit",
        "identity_fix_tree",
    }
)


def _validate_source_release_pins(value: Any) -> dict[str, str]:
    expected = _RELEASE_PATH_FIELDS | _RELEASE_SHA256_FIELDS | _RELEASE_SHA1_FIELDS
    if not isinstance(value, Mapping) or set(value) != expected:
        raise T2PassAdoptionReleaseDisabled("external source-pin schema changed")
    result: dict[str, str] = {}
    for key in sorted(expected):
        item = value[key]
        if type(item) is not str:
            raise T2PassAdoptionReleaseDisabled(
                f"external release pin {key} is not native str"
            )
        if key in _RELEASE_PATH_FIELDS:
            if _absolute(item, label=f"release pin {key}") != Path(item):
                raise T2PassAdoptionReleaseDisabled(
                    f"external release path pin {key} is not canonical"
                )
        elif key in _RELEASE_SHA256_FIELDS:
            if not SHA256_RE.fullmatch(item):
                raise T2PassAdoptionReleaseDisabled(
                    f"external release SHA-256 pin {key} is malformed"
                )
        elif not re.fullmatch(r"[0-9a-f]{40}", item):
            raise T2PassAdoptionReleaseDisabled(
                f"external release Git pin {key} is malformed"
            )
        result[key] = item
    return result


def reviewed_release_candidate(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return independent-review material; it cannot authorize publication."""

    adoption_git = evidence["source_code"]["adoption_git"]
    return {
        "schema_version": EXTERNAL_RELEASE_SCHEMA,
        "stage": "T2_GINE_FULL",
        "dataset": "tastemolnet",
        "status": "UNREVIEWED_RELEASE_CANDIDATE",
        "authorization": False,
        "implementation": {
            "commit": adoption_git["commit"],
            "tree": adoption_git["tree"],
            "critical_blobs": dict(adoption_git["critical_blobs"]),
        },
        "source_pins": _observed_release_values(evidence),
    }


def _validate_external_release(
    value: Any,
    *,
    evidence: Mapping[str, Any],
) -> dict[str, str]:
    expected_fields = {
        "schema_version",
        "stage",
        "dataset",
        "status",
        "authorization",
        "implementation",
        "source_pins",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_fields
        or value.get("schema_version") != EXTERNAL_RELEASE_SCHEMA
        or value.get("stage") != "T2_GINE_FULL"
        or value.get("dataset") != "tastemolnet"
        or value.get("status") != "REVIEWED_RELEASE_AUTHORIZED"
        or value.get("authorization") is not True
    ):
        raise T2PassAdoptionReleaseDisabled(
            "external T2 release authority is not exact reviewed authorization"
        )
    implementation = value.get("implementation")
    if not isinstance(implementation, Mapping) or set(implementation) != {
        "commit",
        "tree",
        "critical_blobs",
    }:
        raise T2PassAdoptionReleaseDisabled(
            "external implementation authority schema changed"
        )
    commit = implementation.get("commit")
    tree = implementation.get("tree")
    blobs = implementation.get("critical_blobs")
    if (
        type(commit) is not str
        or not re.fullmatch(r"[0-9a-f]{40}", commit)
        or type(tree) is not str
        or not re.fullmatch(r"[0-9a-f]{40}", tree)
        or not isinstance(blobs, Mapping)
        or set(blobs) != set(IMPLEMENTATION_CRITICAL_BLOBS)
        or any(
            type(path) is not str
            or type(digest) is not str
            or not SHA256_RE.fullmatch(digest)
            for path, digest in blobs.items()
        )
    ):
        raise T2PassAdoptionReleaseDisabled(
            "external implementation commit/tree/blob pins are malformed"
        )
    pins = _validate_source_release_pins(value.get("source_pins"))
    if pins != _observed_release_values(evidence):
        raise T2PassAdoptionReleaseDisabled(
            "external release source pins do not match held evidence"
        )
    adoption_git = evidence["source_code"]["adoption_git"]
    if (
        adoption_git.get("parents") != [commit]
        or adoption_git.get("parent_tree") != tree
        or adoption_git.get("changed_from_parent") != [RELEASE_CONFIG_RELATIVE]
        or adoption_git.get("parent_critical_blobs") != dict(blobs)
        or any(
            adoption_git.get("critical_blobs", {}).get(path) != digest
            for path, digest in blobs.items()
            if path != RELEASE_CONFIG_RELATIVE
        )
    ):
        raise T2PassAdoptionReleaseDisabled(
            "current checkout is not the exact clean one-config release child"
        )
    return pins


def _require_release(
    hold: _SourceHold,
    evidence: Mapping[str, Any],
) -> dict[str, str]:
    config = hold._release_config
    if not isinstance(config, Mapping) or config.get("authorization") is not True:
        raise T2PassAdoptionReleaseDisabled(
            "T2 PASS adoption publication is stage-frozen: authorization is false"
        )
    authority_path = _absolute(
        config.get("external_authority_path"),
        label="external release authority",
    )
    if any(
        _paths_overlap(authority_path, protected)
        for protected in (
            hold.sources.adoption_root,
            hold.sources.controller_root,
            hold.sources.output_root,
            hold.sources.training_state_root,
            hold.sources.execution_project_root,
            hold.sources.identity_fix_project_root,
            hold.adoption_project.path,
        )
    ):
        raise T2PassAdoptionReleaseDisabled(
            "external release authority overlaps a protected source/destination"
        )
    hold.external_release_parent = HeldDirectory(
        authority_path.parent,
        label="external release authority parent",
    )
    hold.external_release_parent.open()
    hold._external_release_file = _open_selected_file(
        hold.external_release_parent,
        authority_path.name,
    )
    hold._selected.append(hold._external_release_file)
    expected_sha = config.get("external_authority_sha256")
    if (
        type(expected_sha) is not str
        or not SHA256_RE.fullmatch(expected_sha)
        or hold._external_release_file.sha256 != expected_sha
    ):
        raise T2PassAdoptionReleaseDisabled(
            "external release authority runtime SHA pin changed"
        )
    value = hold._external_release_file.json()
    pins = _validate_external_release(value, evidence=evidence)
    hold.verify()
    return pins


def _write_new_at(directory_fd: int, name: str, data: bytes) -> HeldFile:
    if name not in FIVE_FILE_SET[:-1]:
        raise T2PassAdoptionError("publisher attempted a non-canonical adoption file")
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise T2PassAdoptionError("short write while publishing adoption receipt")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)
    held = HeldFile(
        parent_fd=directory_fd,
        name=name,
        relative=f"adoption/{name}",
    )
    if held.bytes() != data:
        held.close()
        raise T2PassAdoptionError(f"published adoption readback changed: {name}")
    return held


def _write_prepared_gate(directory_fd: int, data: bytes) -> HeldFile:
    name = ".gate.json.prepared"
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise T2PassAdoptionError("short write while preparing terminal gate")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)
    held = HeldFile(
        parent_fd=directory_fd,
        name=name,
        relative="adoption/.gate.json.prepared",
    )
    if held.bytes() != data:
        held.close()
        raise T2PassAdoptionError("prepared terminal gate readback changed")
    return held


def _rename_gate_noreplace(
    directory_fd: int,
    *,
    retained_closure: Callable[[], None],
) -> None:
    """Make gate visible with the final publication syscall."""

    source = b".gate.json.prepared"
    destination = b"gate.json"
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise T2PassAdoptionError("Linux renameat2(RENAME_NOREPLACE) is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        # The caller-supplied retained closure executes inside the terminal
        # primitive, after the syscall entrypoint is prepared and immediately
        # before the publication syscall.  A wrapper-entry pathname swap can
        # therefore never redirect the gate to a detached old inode.
        retained_closure()
        if renameat2(directory_fd, source, directory_fd, destination, 1) != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise T2PassAdoptionError("terminal gate destination already exists")
            raise OSError(error, os.strerror(error))
        return
    # Portable test-only fallback.  Production is already Linux-/proc-gated.
    retained_closure()
    try:
        os.stat("gate.json", dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise T2PassAdoptionError("terminal gate destination already exists")
    os.rename(
        ".gate.json.prepared",
        "gate.json",
        src_dir_fd=directory_fd,
        dst_dir_fd=directory_fd,
    )


def _publication_binding(
    output_fd: int,
    documents: Mapping[str, HeldFile],
) -> dict[str, Any]:
    if set(documents) != set(FIVE_FILE_SET[:-1]):
        raise T2PassAdoptionError("terminal gate lacks the exact four documents")
    output_info = os.fstat(output_fd)
    if (
        not stat.S_ISDIR(output_info.st_mode)
        or output_info.st_uid != os.getuid()
        or stat.S_IMODE(output_info.st_mode) != 0o700
    ):
        raise T2PassAdoptionError("adoption root identity changed before terminal gate")
    rows: dict[str, Any] = {}
    for name in FIVE_FILE_SET[:-1]:
        held = documents[name]
        held.verify()
        info = os.fstat(held.fd) if held.fd is not None else None
        if info is None:
            raise T2PassAdoptionError(f"terminal document authority closed: {name}")
        rows[name] = {
            "identity": _publication_file_identity(info),
            "sha256": held.sha256,
        }
    return {
        "schema_version": "tastemolnet_t2_gate_physical_binding_v1",
        "adoption_root_identity": _publication_directory_identity(output_info),
        "documents": rows,
    }


def _open_or_create_namespace(control: HeldDirectory) -> tuple[int, dict[str, int]]:
    if control.fd is None:
        raise T2PassAdoptionError("control root authority is closed")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        os.mkdir(ADOPTION_NAMESPACE, 0o700, dir_fd=control.fd)
        os.fsync(control.fd)
    except FileExistsError:
        pass
    namespace_fd = os.open(ADOPTION_NAMESPACE, flags, dir_fd=control.fd)
    info = os.fstat(namespace_fd)
    named = os.stat(ADOPTION_NAMESPACE, dir_fd=control.fd, follow_symlinks=False)
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or not _same_inode(info, named)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        os.close(namespace_fd)
        raise T2PassAdoptionError("adoption namespace is not owner-bound mode 0700")
    return namespace_fd, _directory_identity(info)


def _verify_destination_binding(
    *,
    control: HeldDirectory,
    namespace_fd: int,
    namespace_identity: Mapping[str, int],
    output_fd: int,
    output_identity: Mapping[str, int],
    expected_names: tuple[str, ...],
) -> None:
    """Retain the fresh namespace/CID path edges throughout publication."""

    if control.fd is None:
        raise T2PassAdoptionError("control root authority is closed")
    held_namespace = os.fstat(namespace_fd)
    named_namespace = os.stat(
        ADOPTION_NAMESPACE, dir_fd=control.fd, follow_symlinks=False
    )
    held_output = os.fstat(output_fd)
    named_output = os.stat(
        SOURCE_CID, dir_fd=namespace_fd, follow_symlinks=False
    )
    if (
        not stat.S_ISDIR(held_namespace.st_mode)
        or stat.S_ISLNK(named_namespace.st_mode)
        or not _same_inode(held_namespace, named_namespace)
        or _directory_identity(held_namespace) != dict(namespace_identity)
        or _directory_identity(named_namespace) != dict(namespace_identity)
        or tuple(sorted(os.listdir(namespace_fd))) != (SOURCE_CID,)
        or not stat.S_ISDIR(held_output.st_mode)
        or stat.S_ISLNK(named_output.st_mode)
        or not _same_inode(held_output, named_output)
        or _directory_identity(held_output) != dict(output_identity)
        or _directory_identity(named_output) != dict(output_identity)
        or held_output.st_uid != os.getuid()
        or stat.S_IMODE(held_output.st_mode) != 0o700
        or tuple(sorted(os.listdir(output_fd))) != tuple(sorted(expected_names))
    ):
        raise T2PassAdoptionError("fresh adoption destination authority changed")


def _build_receipt_payloads(
    evidence: Mapping[str, Any], *, pins: Mapping[str, str]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Build the four preterminal documents and their acyclic hash DAG."""

    source_sha = str(evidence["source_evidence_sha256"])
    completion_semantics = {
        "old_controller_state": "FAILED",
        "old_controller_reason": SOURCE_FAILED_REASON,
        "old_controller_is_scientific_false_negative": True,
        "old_controller_record_retained_as_control_plane_truth": True,
        "scientific_bundle_status": "PASS",
        "run_registry_status": "PASS",
        "runtime_status": "PASS",
        "training_complete_status": "PASS",
        "scientific_pass_adopted_without_controller_reconciliation": True,
        "old_controller_rewritten": False,
        "old_scientific_output_rewritten": False,
    }
    input_hashes = {
        "schema_version": "tastemolnet_t2_gine_pass_input_hashes_v1",
        "stage": "T2_GINE_FULL",
        "status": "VERIFIED_READ_ONLY",
        "marker": ADOPTION_MARKER,
        "source_evidence": dict(evidence),
        "source_evidence_sha256": source_sha,
        "release_pins": dict(pins),
    }
    input_sha = hashlib.sha256(_json_bytes(input_hashes)).hexdigest()
    state = {
        "schema_version": "tastemolnet_t2_gine_pass_adoption_state_v1",
        "stage": "T2_GINE_FULL",
        "status": "PASS",
        "state": ADOPTION_MARKER,
        "marker": ADOPTION_MARKER,
        "source_cid": SOURCE_CID,
        "source_run_id": SOURCE_RUN_ID,
        "input_hashes_sha256": input_sha,
        "source_evidence_sha256": source_sha,
        "scientific_execution_performed": False,
        "old_controller_mutated": False,
        "old_output_mutated": False,
        "old_training_state_mutated": False,
        "main_controller_mutated": False,
        "matrix_mutated": False,
        "source_identity_revalidated_through_gate_commit": True,
        "completion_semantics": completion_semantics,
    }
    state_sha = hashlib.sha256(_json_bytes(state)).hexdigest()
    manifest = {
        "schema_version": "tastemolnet_t2_gine_pass_adoption_receipt_v1",
        "receipt_kind": ADOPTION_MARKER,
        "stage": "T2_GINE_FULL",
        "status": "PASS",
        "marker": ADOPTION_MARKER,
        "source_cid": SOURCE_CID,
        "source_run_id": SOURCE_RUN_ID,
        "source_execution_commit": SOURCE_EXECUTION_COMMIT,
        "source_identity_fix_commit": SOURCE_IDENTITY_FIX_COMMIT,
        "input_hashes_sha256": input_sha,
        "state_sha256": state_sha,
        "source_evidence_sha256": source_sha,
        "receipt_file": "manifest.json",
        "downstream_consumers": ["T3_GINE_CALIBRATED", "T4_ORACLE_SMOKE"],
        "downstream_must_bind_receipt_sha256": True,
        "five_file_set": list(FIVE_FILE_SET),
        "release_authorized": True,
        "science_executed_by_adoption": False,
        "completion_semantics": completion_semantics,
        "publication_boundary": {
            "only_write_root": evidence["adoption_boundary"]["adoption_root"],
            "old_controller": "RETAINED_READ_ONLY",
            "old_scientific_output": "RETAINED_READ_ONLY",
            "old_training_state": "RETAINED_READ_ONLY",
            "main_controller": "NOT_OPENED_NOT_WRITTEN",
            "main_matrix": "NOT_OPENED_NOT_WRITTEN",
            "source_identity_revalidated_through_gate_commit": True,
        },
        "t3_dependency_contract": {
            "required_t2_gate_file": "gate.json",
            "required_t2_receipt_file": "manifest.json",
            "receipt_binding": (
                "gate.receipt_sha256 == SHA256(manifest.json)"
            ),
            "required_formal_bundle_root": evidence["scientific_output"]["root"],
            "required_formal_bundle_inventory_sha256": evidence[
                "scientific_output"
            ]["inventory_sha256"],
            "required_model_sha256": evidence["scientific_output"][
                "model_sha256"
            ],
            "other_t2_authorities_allowed": False,
        },
        "t4_dependency_contract": {
            "receipt_binding": (
                "gate.receipt_sha256 == SHA256(manifest.json)"
            ),
            "t3_gate_still_required": True,
        },
    }
    manifest_sha = hashlib.sha256(_json_bytes(manifest)).hexdigest()
    output_hashes = {
        "schema_version": "tastemolnet_t2_gine_pass_output_hashes_v1",
        "stage": "T2_GINE_FULL",
        "status": "PASS_PENDING_GATE",
        "marker": ADOPTION_MARKER,
        "receipt_file": "manifest.json",
        "receipt_sha256": manifest_sha,
        "files": {
            "input_hashes.json": input_sha,
            "state.json": state_sha,
            "manifest.json": manifest_sha,
        },
        "gate_excluded_to_avoid_hash_cycle": True,
    }
    output_sha = hashlib.sha256(_json_bytes(output_hashes)).hexdigest()
    payloads = {
        "input_hashes.json": input_hashes,
        "state.json": state,
        "manifest.json": manifest,
        "output_hashes.json": output_hashes,
    }
    hashes = {
        name: hashlib.sha256(_json_bytes(payload)).hexdigest()
        for name, payload in payloads.items()
    }
    return payloads, hashes


def _build_gate_payload(
    evidence: Mapping[str, Any],
    *,
    document_hashes: Mapping[str, str],
    physical_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if set(document_hashes) != set(FIVE_FILE_SET[:-1]):
        raise T2PassAdoptionError("terminal gate document-hash schema changed")
    manifest_sha = document_hashes["manifest.json"]
    return {
        "schema_version": "tastemolnet_t2_gine_pass_adoption_gate_v1",
        "stage": "T2_GINE_FULL",
        "status": "PASS",
        "state": ADOPTION_MARKER,
        "marker": ADOPTION_MARKER,
        "failures": [],
        "receipt_file": "manifest.json",
        "receipt_sha256": manifest_sha,
        "source_evidence_sha256": evidence["source_evidence_sha256"],
        "files": dict(document_hashes),
        "physical_binding": dict(physical_binding),
        "gate_published_last": True,
        "no_fallible_operation_after_gate_publication": True,
        "downstream_release_scope": ["T3_GINE_CALIBRATED", "T4_ORACLE_SMOKE"],
    }


def publish_t2_gine_pass_adoption(
    sources: T2PassAdoptionSources,
) -> dict[str, Any]:
    """One-shot publish the exact five-file receipt into the derived fresh root."""

    if sources.adoption_root.exists() or sources.adoption_root.is_symlink():
        raise T2PassAdoptionError(
            "adoption root already exists; one-shot publication never resumes or reconciles"
        )
    with _SourceHold(sources) as hold:
        assert hold.evidence is not None
        evidence = hold.evidence
        pins = _require_release(hold, evidence)
        hold.verify()
        namespace_fd, namespace_identity = _open_or_create_namespace(hold.control)
        output_fd: int | None = None
        documents: dict[str, HeldFile] = {}
        prepared_gate: HeldFile | None = None
        try:
            if os.stat(
                ADOPTION_NAMESPACE,
                dir_fd=hold.control.fd,
                follow_symlinks=False,
            ).st_ino != namespace_identity["inode"]:
                raise T2PassAdoptionError("adoption namespace changed before mkdirat")
            if os.listdir(namespace_fd):
                raise T2PassAdoptionError(
                    "adoption namespace is not fresh and empty before CID creation"
                )
            os.mkdir(SOURCE_CID, 0o700, dir_fd=namespace_fd)
            os.fsync(namespace_fd)
            output_fd = os.open(
                SOURCE_CID,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=namespace_fd,
            )
            output_info = os.fstat(output_fd)
            output_identity = _directory_identity(output_info)
            if (
                output_info.st_uid != os.getuid()
                or stat.S_IMODE(output_info.st_mode) != 0o700
            ):
                raise T2PassAdoptionError("fresh adoption root lost owner authority")
            _verify_destination_binding(
                control=hold.control,
                namespace_fd=namespace_fd,
                namespace_identity=namespace_identity,
                output_fd=output_fd,
                output_identity=output_identity,
                expected_names=(),
            )
            payloads, expected_hashes = _build_receipt_payloads(
                evidence, pins=pins
            )
            source_sha = str(evidence["source_evidence_sha256"])
            written_names: list[str] = []
            for name in FIVE_FILE_SET[:-1]:
                hold.verify()
                _verify_destination_binding(
                    control=hold.control,
                    namespace_fd=namespace_fd,
                    namespace_identity=namespace_identity,
                    output_fd=output_fd,
                    output_identity=output_identity,
                    expected_names=tuple(written_names),
                )
                held_document = _write_new_at(
                    output_fd,
                    name,
                    _json_bytes(payloads[name]),
                )
                documents[name] = held_document
                if held_document.sha256 != expected_hashes[name]:
                    raise T2PassAdoptionError(
                        f"published adoption hash changed while writing: {name}"
                    )
                written_names.append(name)
            hold.verify()
            _verify_destination_binding(
                control=hold.control,
                namespace_fd=namespace_fd,
                namespace_identity=namespace_identity,
                output_fd=output_fd,
                output_identity=output_identity,
                expected_names=tuple(written_names),
            )
            physical_binding = _publication_binding(output_fd, documents)
            gate_payload = _build_gate_payload(
                evidence,
                document_hashes=expected_hashes,
                physical_binding=physical_binding,
            )
            gate_bytes = _json_bytes(gate_payload)
            gate_sha = hashlib.sha256(gate_bytes).hexdigest()
            prepared_gate = _write_prepared_gate(output_fd, gate_bytes)
            if prepared_gate.sha256 != gate_sha:
                raise T2PassAdoptionError("prepared terminal gate hash changed")
            hold.verify()
            _verify_destination_binding(
                control=hold.control,
                namespace_fd=namespace_fd,
                namespace_identity=namespace_identity,
                output_fd=output_fd,
                output_identity=output_identity,
                expected_names=tuple([*written_names, ".gate.json.prepared"]),
            )
            if prepared_gate.bytes() != gate_bytes:
                raise T2PassAdoptionError("prepared terminal gate bytes changed")
            if _publication_binding(output_fd, documents) != physical_binding:
                raise T2PassAdoptionError(
                    "preterminal document physical binding changed"
                )
            hold.verify()
            _verify_destination_binding(
                control=hold.control,
                namespace_fd=namespace_fd,
                namespace_identity=namespace_identity,
                output_fd=output_fd,
                output_identity=output_identity,
                expected_names=tuple([*written_names, ".gate.json.prepared"]),
            )
            if _publication_binding(output_fd, documents) != physical_binding:
                raise T2PassAdoptionError(
                    "terminal document binding changed at commit boundary"
                )
            prepared_gate.verify()
            result = {
                "status": "PASS",
                "state": ADOPTION_MARKER,
                "adoption_root": str(sources.adoption_root),
                "receipt_path": str(sources.adoption_root / "manifest.json"),
                "receipt_sha256": expected_hashes["manifest.json"],
                "gate_sha256": gate_sha,
                "source_evidence_sha256": source_sha,
                "five_file_set": list(FIVE_FILE_SET),
            }

            def retained_gate_closure() -> None:
                """Revalidate every retained authority at the commit boundary."""

                hold.verify()
                if _publication_binding(output_fd, documents) != physical_binding:
                    raise T2PassAdoptionError(
                        "terminal document binding changed inside commit primitive"
                    )
                prepared_gate.verify()
                if prepared_gate.bytes() != gate_bytes:
                    raise T2PassAdoptionError(
                        "prepared terminal gate changed inside commit primitive"
                    )
                _verify_destination_binding(
                    control=hold.control,
                    namespace_fd=namespace_fd,
                    namespace_identity=namespace_identity,
                    output_fd=output_fd,
                    output_identity=output_identity,
                    expected_names=tuple(
                        [*written_names, ".gate.json.prepared"]
                    ),
                )

            # The retained closure runs *inside* the terminal primitive after
            # its syscall entrypoint is prepared.  The rename itself is the
            # final publication syscall; only non-throwing cleanup follows.
            _rename_gate_noreplace(
                output_fd,
                retained_closure=retained_gate_closure,
            )
            return result
        finally:
            # Cleanup is deliberately best-effort.  In particular, no close
            # failure may turn a visible terminal gate into a reported failure.
            if prepared_gate is not None:
                try:
                    prepared_gate.close()
                except BaseException:
                    pass
            for held_document in reversed(tuple(documents.values())):
                try:
                    held_document.close()
                except BaseException:
                    pass
            if output_fd is not None:
                try:
                    os.close(output_fd)
                except BaseException:
                    pass
            try:
                os.close(namespace_fd)
            except BaseException:
                pass


def validate_t2_gine_pass_adoption(
    sources: T2PassAdoptionSources,
) -> dict[str, Any]:
    """Read-only status validation of the exact five-file receipt and sources."""

    with _SourceHold(sources) as source_hold:
        assert source_hold.evidence is not None
        with HeldTree(sources.adoption_root, label="T2 adoption root") as adoption:
            if set(adoption.files) != set(FIVE_FILE_SET) or any(
                row["kind"] == "directory" for row in adoption.inventory
            ):
                raise T2PassAdoptionError("adoption root is not the exact five-file set")
            if stat.S_IMODE(adoption.root.identity["mode"]) != 0o700 or any(
                stat.S_IMODE(adoption.file(name).identity["mode"]) != 0o600
                for name in FIVE_FILE_SET
            ):
                raise T2PassAdoptionError(
                    "adoption root/files are not owner-private mode 0700/0600"
                )
            values = {name: adoption.file(name).json() for name in FIVE_FILE_SET}
            for name, value in values.items():
                if adoption.file(name).bytes() != _json_bytes(value):
                    raise T2PassAdoptionError(
                        f"{name} is not the deterministic canonical receipt encoding"
                    )
            hashes = {name: adoption.file(name).sha256 for name in FIVE_FILE_SET}
            evidence = source_hold.evidence
            source_sha = evidence["source_evidence_sha256"]
            pins = _require_release(source_hold, evidence)
            expected_documents, expected_document_hashes = _build_receipt_payloads(
                evidence, pins=pins
            )
            assert adoption.root.fd is not None
            binding = _publication_binding(
                adoption.root.fd,
                {
                    name: adoption.file(name)
                    for name in FIVE_FILE_SET[:-1]
                },
            )
            expected_gate = _build_gate_payload(
                evidence,
                document_hashes=expected_document_hashes,
                physical_binding=binding,
            )
            expected_values = {**expected_documents, "gate.json": expected_gate}
            expected_hashes = {
                **expected_document_hashes,
                "gate.json": hashlib.sha256(_json_bytes(expected_gate)).hexdigest(),
            }
            if values != expected_values or hashes != expected_hashes:
                raise T2PassAdoptionError("T2 adoption receipt hash/type closure changed")
            source_hold.verify()
            adoption.verify()
            return {
                "status": "PASS",
                "state": ADOPTION_MARKER,
                "adoption_root": str(sources.adoption_root),
                "receipt_path": str(sources.adoption_root / "manifest.json"),
                "receipt_sha256": hashes["manifest.json"],
                "gate_sha256": hashes["gate.json"],
                "source_evidence_sha256": source_sha,
                "five_file_set": list(FIVE_FILE_SET),
                "read_only_validation": True,
            }


__all__ = [
    "ADOPTION_MARKER",
    "FIVE_FILE_SET",
    "SOURCE_CID",
    "SOURCE_EXECUTION_COMMIT",
    "SOURCE_IDENTITY_FIX_COMMIT",
    "SOURCE_RUN_ID",
    "T2PassAdoptionError",
    "T2PassAdoptionReleaseDisabled",
    "T2PassAdoptionSources",
    "adoption_output_root",
    "preflight_t2_gine_pass_adoption",
    "publish_t2_gine_pass_adoption",
    "reviewed_release_candidate",
    "validate_t2_gine_pass_adoption",
]
