"""Atomic epoch checkpoints for long-running molecular-GNN training.

The checkpoint root is deliberately separate from the immutable classifier
bundle.  A failed training process can therefore resume in the same science
campaign while the final output path remains fresh for PASS-last publication.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import ctypes
import copy
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "molecular_gnn_epoch_resume_v1"
ROOT_CLAIM_NAME = "root_claim.json"
ROOT_SENTINEL_NAME = ".root_identity"
LOCK_NAME = ".writer.lock"
CONTRACT_NAME = "training_contract.json"
LATEST_NAME = "latest_checkpoint.json"
HEARTBEAT_NAME = "training_heartbeat.json"
COMPLETE_NAME = "training_complete.json"
CLEANUP_NAME = "checkpoint_cleanup.json"
OUTPUT_PARENT_SCHEMA = "molecular_gnn_output_parent_authority_v1"
FINALIZATION_SCHEMA = "molecular_gnn_finalization_workspace_v1"
FINALIZATION_MAX_FILES = 2048
FINALIZATION_MAX_BYTES = 32 * 1024**3
CONTRACT_PHYSICAL_SCHEMA = "molecular_gnn_training_contract_physical_v1"
MAX_CONTRACT_BYTES = 16 * 1024**2


class MolecularGNNResumeError(RuntimeError):
    """Raised when a persistent training state is not safe to resume."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_held_json_physical(
    path: Path,
    descriptor: int,
    *,
    label: str,
) -> tuple[dict[str, Any], bytes, os.stat_result]:
    """Read one held regular-file inode and prove its named binding stayed fixed."""

    try:
        before = os.fstat(descriptor)
        named_before = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or _stat_identity(before) != _stat_identity(named_before)
        ):
            raise MolecularGNNResumeError(
                f"{label} must remain one named physical regular file"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, MAX_CONTRACT_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_CONTRACT_BYTES:
                raise MolecularGNNResumeError(f"{label} exceeds its physical byte limit")
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        named_after = os.stat(path, follow_symlinks=False)
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(named_after)
        ):
            raise MolecularGNNResumeError(f"{label} changed while read")
        payload = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MolecularGNNResumeError(f"{label} is not valid held JSON") from exc
    if not isinstance(payload, dict):
        raise MolecularGNNResumeError(f"{label} must contain one JSON object")
    return payload, data, after


def _validate_contract_payload(
    payload: Mapping[str, Any],
    *,
    expected_contract_sha256: str,
    expected_contract: Mapping[str, Any] | None = None,
) -> None:
    contract = payload.get("contract")
    if (
        set(payload)
        != {
            "schema_version",
            "artifact_kind",
            "contract",
            "contract_sha256",
            "root_claim_sha256",
        }
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("artifact_kind") != "molecular_gnn_training_contract"
        or not isinstance(contract, Mapping)
        or canonical_sha256(contract) != expected_contract_sha256
        or payload.get("contract_sha256") != expected_contract_sha256
    ):
        raise MolecularGNNResumeError(
            "training contract canonical content/hash changed"
        )
    if expected_contract is not None and dict(contract) != dict(expected_contract):
        raise MolecularGNNResumeError("training resume contract content changed")


def _contract_physical_evidence(
    *,
    payload: Mapping[str, Any],
    data: bytes,
    info: os.stat_result,
) -> dict[str, Any]:
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        raise MolecularGNNResumeError("training contract content is untyped")
    return {
        "schema_version": CONTRACT_PHYSICAL_SCHEMA,
        "name": CONTRACT_NAME,
        "identity": _stat_identity(info),
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "canonical_sha256": canonical_sha256(contract),
        "content": copy.deepcopy(dict(payload)),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json_replace(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                dict(payload),
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_noclobber(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                dict(payload),
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _stat_identity(info: os.stat_result) -> dict[str, int]:
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


def _require_regular(path: Path, *, label: str) -> os.stat_result:
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise MolecularGNNResumeError(f"{label} is missing: {path}") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise MolecularGNNResumeError(f"{label} must be one physical regular file")
    return info


def _require_physical_directory(path: Path, *, label: str) -> os.stat_result:
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise MolecularGNNResumeError(f"{label} is missing: {path}") from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise MolecularGNNResumeError(f"{label} must be one physical directory")
    return info


def _directory_binding(info: os.stat_result) -> dict[str, int]:
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(info.st_mode),
        "uid": int(info.st_uid),
    }


def _bound_regular_file_evidence(
    path: Path, *, label: str
) -> tuple[os.stat_result, str]:
    """Hash one named file while proving the fd/path inode stayed identical."""

    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        before = os.fstat(descriptor)
        named_before = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or _stat_identity(before) != _stat_identity(named_before)
        ):
            raise MolecularGNNResumeError(
                f"{label} must remain one named physical regular file"
            )
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        named_after = os.stat(path, follow_symlinks=False)
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(named_after)
        ):
            raise MolecularGNNResumeError(f"{label} changed while hashed")
        return after, digest.hexdigest()
    finally:
        os.close(descriptor)


def assert_no_symlink_components(path: str | Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(Path(path).expanduser()))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            return
        if stat.S_ISLNK(info.st_mode):
            raise MolecularGNNResumeError(f"{label} may not contain symlink components")


def paths_overlap(left: str | Path, right: str | Path) -> bool:
    """Return whether two absolute lexical trees contain one another."""

    first = Path(os.path.abspath(Path(left).expanduser()))
    second = Path(os.path.abspath(Path(right).expanduser()))
    return first == second or first in second.parents or second in first.parents


def atomic_rename_directory_noreplace(
    source: str | Path,
    target: str | Path,
    *,
    directory_fd: int | None = None,
) -> None:
    """Atomically publish one directory without replacing an existing name.

    Production AutoDL uses Linux ``renameat2(RENAME_NOREPLACE)``.  Darwin's
    equivalent ``renamex_np(RENAME_EXCL)`` is used by local tests; every other
    platform fails closed instead of emulating no-replace with a racy check.
    """

    source_path = Path(source)
    target_path = Path(target)
    if source_path.parent != target_path.parent:
        raise MolecularGNNResumeError("atomic directory publication requires siblings")
    _require_physical_directory(source_path, label="finalization staging directory")
    if directory_fd is not None:
        held_parent = os.fstat(directory_fd)
        named_parent = _require_physical_directory(
            source_path.parent, label="atomic publication parent"
        )
        if (
            not stat.S_ISDIR(held_parent.st_mode)
            or (held_parent.st_dev, held_parent.st_ino)
            != (named_parent.st_dev, named_parent.st_ino)
        ):
            raise MolecularGNNResumeError(
                "atomic publication parent directory identity changed"
            )
        source_directory_fd = directory_fd
        target_directory_fd = directory_fd
        encoded_source = os.fsencode(source_path.name)
        encoded_target = os.fsencode(target_path.name)
    else:
        source_directory_fd = -100
        target_directory_fd = -100
        encoded_source = os.fsencode(source_path)
        encoded_target = os.fsencode(target_path)
    library = ctypes.CDLL(None, use_errno=True)
    result: int
    if sys.platform.startswith("linux"):
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise MolecularGNNResumeError("Linux renameat2 is required for publication")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = int(
            renameat2(
                source_directory_fd,
                encoded_source,
                target_directory_fd,
                encoded_target,
                1,
            )
        )
    elif sys.platform == "darwin":  # pragma: no cover - exercised on macOS CI.
        if directory_fd is not None:
            renameatx_np = getattr(library, "renameatx_np", None)
            if renameatx_np is None:
                raise MolecularGNNResumeError("Darwin renameatx_np is unavailable")
            renameatx_np.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            renameatx_np.restype = ctypes.c_int
            result = int(
                renameatx_np(
                    source_directory_fd,
                    encoded_source,
                    target_directory_fd,
                    encoded_target,
                    0x00000004,
                )
            )
        else:
            renamex_np = getattr(library, "renamex_np", None)
            if renamex_np is None:
                raise MolecularGNNResumeError("Darwin renamex_np is unavailable")
            renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            renamex_np.restype = ctypes.c_int
            result = int(renamex_np(encoded_source, encoded_target, 0x00000004))
    else:  # pragma: no cover - production is Linux.
        raise MolecularGNNResumeError(
            "atomic no-replace directory publication is unsupported on this platform"
        )
    if result != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(error, os.strerror(error), str(target_path))
        raise OSError(error, os.strerror(error), str(target_path))
    if directory_fd is None:
        _fsync_directory(target_path.parent)
    else:
        os.fsync(directory_fd)


class OutputParentAuthority:
    """Hold and repeatedly verify the physical parent used for final publish."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        contract_sha256: str,
        resume: bool,
        read_only: bool = False,
    ) -> None:
        self.output_dir = Path(os.path.abspath(Path(output_dir).expanduser()))
        self.parent = self.output_dir.parent
        self.contract_sha256 = str(contract_sha256)
        token = hashlib.sha256(str(self.output_dir).encode("utf-8")).hexdigest()[:20]
        prefix = f".{self.output_dir.name}.parent-authority-{token}"
        self.sentinel_path = self.parent / f"{prefix}.sentinel"
        self.lock_path = self.parent / f"{prefix}.lock"
        self.claim_path = self.parent / f"{prefix}.json"
        self.resume = bool(resume)
        self.read_only = bool(read_only)
        self._parent_fd: int | None = None
        self._lock_fd: int | None = None
        self._claim: dict[str, Any] | None = None

    def open(self) -> None:
        assert_no_symlink_components(self.parent, label="output parent")
        if not self.parent.exists():
            if self.resume or self.read_only:
                raise MolecularGNNResumeError("resume output parent is absent")
            self.parent.mkdir(parents=True, mode=0o700)
            _fsync_directory(self.parent.parent)
        _require_physical_directory(self.parent, label="output parent")
        self._parent_fd = os.open(
            self.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        lock_flags = os.O_RDONLY if self.read_only else os.O_RDWR | os.O_CREAT
        self._lock_fd = os.open(
            self.lock_path,
            lock_flags | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        lock_info = os.fstat(self._lock_fd)
        if (
            not stat.S_ISREG(lock_info.st_mode)
            or lock_info.st_nlink != 1
            or lock_info.st_uid != os.getuid()
            or stat.S_IMODE(lock_info.st_mode) != 0o600
        ):
            self.close()
            raise MolecularGNNResumeError("output-parent lock is not a single regular file")
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.close()
            raise MolecularGNNResumeError("another writer owns the output parent claim") from exc
        finalization_prefix = (
            f".{self.output_dir.name}.finalizing-{self.contract_sha256}"
        )
        publication_paths = (
            self.output_dir,
            self.parent / finalization_prefix,
            self.parent / f"{finalization_prefix}.claim.json",
            self.parent / f"{finalization_prefix}.complete.json",
        )
        if not self.sentinel_path.exists():
            if self.read_only:
                self.close()
                raise MolecularGNNResumeError(
                    "read-only output-parent sentinel is absent"
                )
            if self.claim_path.exists() or any(
                path.exists() or path.is_symlink() for path in publication_paths
            ):
                self.close()
                raise MolecularGNNResumeError(
                    "output-parent sentinel is absent after publication began"
                )
            descriptor = os.open(
                self.sentinel_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                os.write(descriptor, secrets.token_bytes(32))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _fsync_directory(self.parent)
        sentinel_info = _require_regular(
            self.sentinel_path, label="output-parent sentinel"
        )
        if (
            sentinel_info.st_uid != os.getuid()
            or stat.S_IMODE(sentinel_info.st_mode) != 0o600
        ):
            self.close()
            raise MolecularGNNResumeError(
                "output-parent sentinel must be owner-bound mode 0600"
            )
        if self.claim_path.exists():
            claim = _load_json(self.claim_path, label="output-parent claim")
        else:
            if self.read_only:
                self.close()
                raise MolecularGNNResumeError(
                    "read-only output-parent claim is absent"
                )
            if any(path.exists() or path.is_symlink() for path in publication_paths):
                self.close()
                raise MolecularGNNResumeError(
                    "output-parent claim is absent after publication began"
                )
            claim = {
                "schema_version": OUTPUT_PARENT_SCHEMA,
                "output_dir": str(self.output_dir),
                "contract_sha256": self.contract_sha256,
                "parent_identity": _directory_binding(os.fstat(self._parent_fd)),
                "sentinel": {
                    "path": str(self.sentinel_path),
                    "identity": _stat_identity(sentinel_info),
                    "sha256": sha256_file(self.sentinel_path),
                },
                "lock": {
                    "path": str(self.lock_path),
                    "identity": _stat_identity(lock_info),
                },
            }
            _atomic_json_noclobber(self.claim_path, claim)
        self._claim = claim
        self.verify()

    @property
    def directory_fd(self) -> int:
        """Return the held publication parent descriptor after revalidation."""

        self.verify()
        assert self._parent_fd is not None
        return self._parent_fd

    def verify(self) -> None:
        if self._parent_fd is None or self._lock_fd is None or self._claim is None:
            raise MolecularGNNResumeError("output-parent authority is not open")
        parent_now = _require_physical_directory(self.parent, label="output parent")
        parent_held = os.fstat(self._parent_fd)
        if (
            (parent_now.st_dev, parent_now.st_ino)
            != (parent_held.st_dev, parent_held.st_ino)
            or _directory_binding(parent_held) != self._claim.get("parent_identity")
            or self._claim.get("schema_version") != OUTPUT_PARENT_SCHEMA
            or self._claim.get("output_dir") != str(self.output_dir)
            or self._claim.get("contract_sha256") != self.contract_sha256
        ):
            raise MolecularGNNResumeError("output-parent physical identity changed")
        sentinel_info = _require_regular(
            self.sentinel_path, label="output-parent sentinel"
        )
        expected_sentinel = self._claim.get("sentinel", {})
        if (
            _stat_identity(sentinel_info) != expected_sentinel.get("identity")
            or sha256_file(self.sentinel_path) != expected_sentinel.get("sha256")
        ):
            raise MolecularGNNResumeError("output-parent sentinel changed")
        lock_now = _require_regular(self.lock_path, label="output-parent lock")
        lock_held = os.fstat(self._lock_fd)
        if (
            (lock_now.st_dev, lock_now.st_ino) != (lock_held.st_dev, lock_held.st_ino)
            or _stat_identity(lock_held) != self._claim.get("lock", {}).get("identity")
        ):
            raise MolecularGNNResumeError("output-parent lock identity changed")
        if _load_json(self.claim_path, label="output-parent claim") != self._claim:
            raise MolecularGNNResumeError("output-parent claim changed")

    def close(self) -> None:
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None
        if self._parent_fd is not None:
            os.close(self._parent_fd)
            self._parent_fd = None


def _finalization_inventory(root: Path, *, hash_files: bool) -> list[dict[str, Any]]:
    _require_physical_directory(root, label="finalization staging directory")
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for directory, directories, files in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        for name in sorted(directories):
            info = os.lstat(base / name)
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise MolecularGNNResumeError("finalization staging contains a symlink/special directory")
        for name in sorted(files):
            path = base / name
            info, digest = _bound_regular_file_evidence(
                path, label="finalization staging file"
            )
            total_bytes += int(info.st_size)
            row: dict[str, Any] = {
                "path": path.relative_to(root).as_posix(),
                "size": int(info.st_size),
            }
            if hash_files:
                row["sha256"] = digest
            rows.append(row)
            if len(rows) > FINALIZATION_MAX_FILES or total_bytes > FINALIZATION_MAX_BYTES:
                raise MolecularGNNResumeError("finalization staging exceeds bounded inventory limits")
    return sorted(rows, key=lambda row: row["path"])


def _finalization_inventory_fd(root_fd: int, *, hash_files: bool) -> list[dict[str, Any]]:
    """Inventory a staging tree relative to one already-held physical root."""

    rows: list[dict[str, Any]] = []
    total_bytes = 0

    def walk(directory_fd: int, prefix: str) -> None:
        nonlocal total_bytes
        entries = sorted(os.scandir(directory_fd), key=lambda entry: entry.name)
        for entry in entries:
            info = entry.stat(follow_symlinks=False)
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                child_fd = os.open(
                    entry.name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
                try:
                    held = os.fstat(child_fd)
                    if (held.st_dev, held.st_ino) != (info.st_dev, info.st_ino):
                        raise MolecularGNNResumeError(
                            "finalization staging directory changed while inventoried"
                        )
                    walk(child_fd, relative)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise MolecularGNNResumeError(
                    "finalization staging contains a symlink/special file"
                )
            descriptor = os.open(
                entry.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                before = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or _stat_identity(before) != _stat_identity(info)
                ):
                    raise MolecularGNNResumeError(
                        "finalization staging file changed while inventoried"
                    )
                digest = hashlib.sha256()
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                after = os.fstat(descriptor)
                named_after = os.stat(
                    entry.name, dir_fd=directory_fd, follow_symlinks=False
                )
                if (
                    _stat_identity(before) != _stat_identity(after)
                    or _stat_identity(after) != _stat_identity(named_after)
                ):
                    raise MolecularGNNResumeError(
                        "finalization staging file changed while inventoried"
                    )
            finally:
                os.close(descriptor)
            total_bytes += int(after.st_size)
            row: dict[str, Any] = {"path": relative, "size": int(after.st_size)}
            if hash_files:
                row["sha256"] = digest.hexdigest()
            rows.append(row)
            if len(rows) > FINALIZATION_MAX_FILES or total_bytes > FINALIZATION_MAX_BYTES:
                raise MolecularGNNResumeError(
                    "finalization staging exceeds bounded inventory limits"
                )

    walk(root_fd, "")
    return sorted(rows, key=lambda row: row["path"])


def _clear_finalization_directory_fd(
    root_fd: int, *, expected_inventory: Sequence[Mapping[str, Any]]
) -> None:
    """Delete only the already-inventoried contents below one held directory fd."""

    expected = {str(row["path"]): dict(row) for row in expected_inventory}
    allowed_directories: set[str] = set()
    for relative in expected:
        parent = Path(relative).parent
        while parent != Path("."):
            allowed_directories.add(parent.as_posix())
            parent = parent.parent

    def clear(directory_fd: int, prefix: str) -> None:
        entries = sorted(os.scandir(directory_fd), key=lambda entry: entry.name)
        for entry in entries:
            info = entry.stat(follow_symlinks=False)
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                if relative not in allowed_directories:
                    raise MolecularGNNResumeError(
                        "finalization cleanup encountered an unreceipted directory"
                    )
                child_fd = os.open(
                    entry.name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
                try:
                    held = os.fstat(child_fd)
                    if (held.st_dev, held.st_ino) != (info.st_dev, info.st_ino):
                        raise MolecularGNNResumeError(
                            "finalization staging directory changed during cleanup"
                        )
                    clear(child_fd, relative)
                    os.fsync(child_fd)
                finally:
                    os.close(child_fd)
                os.rmdir(entry.name, dir_fd=directory_fd)
                continue
            row = expected.get(relative)
            if row is None or not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise MolecularGNNResumeError(
                    "finalization cleanup encountered content absent from its receipt"
                )
            descriptor = os.open(
                entry.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                digest = hashlib.sha256()
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                held = os.fstat(descriptor)
                named = os.stat(
                    entry.name, dir_fd=directory_fd, follow_symlinks=False
                )
                if (
                    _stat_identity(held) != _stat_identity(named)
                    or int(row.get("size", -1)) != held.st_size
                    or row.get("sha256") != digest.hexdigest()
                ):
                    raise MolecularGNNResumeError(
                        "finalization cleanup content changed after its receipt"
                    )
            finally:
                os.close(descriptor)
            os.unlink(entry.name, dir_fd=directory_fd)
        os.fsync(directory_fd)

    clear(root_fd, "")


class FinalizationWorkspace:
    """One deterministic, contract-owned staging directory per final output."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        contract_sha256: str,
        resume: bool,
        parent_authority: OutputParentAuthority,
        training_state_root: str | Path,
    ) -> None:
        self.output_dir = Path(os.path.abspath(Path(output_dir).expanduser()))
        self.contract_sha256 = str(contract_sha256)
        self.resume = bool(resume)
        self.parent_authority = parent_authority
        self.training_state_root = Path(training_state_root)
        prefix = f".{self.output_dir.name}.finalizing-{self.contract_sha256}"
        self.staging = self.output_dir.parent / prefix
        self.claim_path = self.output_dir.parent / f"{prefix}.claim.json"
        self.ready_path = self.output_dir.parent / f"{prefix}.complete.json"
        self.cleanup_receipt_path = (
            self.training_state_root / "finalization_cleanup.json"
        )
        self._claim: dict[str, Any] | None = None
        self._staging_fd: int | None = None

    def _open_staging_fd(self) -> None:
        if self._staging_fd is not None:
            return
        self._staging_fd = os.open(
            self.staging,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )

    def _verify_staging_identity(self, *, published: bool = False) -> None:
        if self._claim is None or self._staging_fd is None:
            raise MolecularGNNResumeError("finalization staging authority is not open")
        held = os.fstat(self._staging_fd)
        path = self.output_dir if published else self.staging
        current = _require_physical_directory(
            path,
            label=("published output directory" if published else "finalization staging directory"),
        )
        if (
            (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino)
            or _directory_binding(held) != self._claim.get("staging_identity")
        ):
            raise MolecularGNNResumeError("finalization staging directory identity changed")

    def prepare(self) -> tuple[Path, bool]:
        self.parent_authority.verify()
        if self.output_dir.exists() or self.output_dir.is_symlink():
            raise FileExistsError(f"immutable output already exists: {self.output_dir}")
        if self.staging.exists():
            if not self.resume:
                raise MolecularGNNResumeError("finalization staging exists without resume authority")
            info = _require_physical_directory(
                self.staging, label="finalization staging directory"
            )
            if not self.claim_path.exists():
                if self.ready_path.exists():
                    raise MolecularGNNResumeError(
                        "finalization completion exists without a staging claim"
                    )
                if _finalization_inventory(self.staging, hash_files=False):
                    raise MolecularGNNResumeError(
                        "unclaimed finalization staging directory is not empty"
                    )
                claim = {
                    "schema_version": FINALIZATION_SCHEMA,
                    "contract_sha256": self.contract_sha256,
                    "output_dir": str(self.output_dir),
                    "staging_dir": str(self.staging),
                    "staging_identity": _directory_binding(info),
                    "output_parent_claim_sha256": sha256_file(
                        self.parent_authority.claim_path
                    ),
                }
                _atomic_json_noclobber(self.claim_path, claim)
            else:
                claim = _load_json(self.claim_path, label="finalization staging claim")
            expected = {
                "schema_version": FINALIZATION_SCHEMA,
                "contract_sha256": self.contract_sha256,
                "output_dir": str(self.output_dir),
                "staging_dir": str(self.staging),
                "staging_identity": _directory_binding(info),
                "output_parent_claim_sha256": sha256_file(
                    self.parent_authority.claim_path
                ),
            }
            if claim != expected:
                raise MolecularGNNResumeError("finalization staging claim changed")
            self._claim = claim
            self._open_staging_fd()
            self._verify_staging_identity()
            inventory = _finalization_inventory_fd(
                self._staging_fd, hash_files=True
            )
            if self.ready_path.exists():
                ready = _load_json(self.ready_path, label="finalization completion")
                if ready != self._ready_payload(inventory):
                    raise MolecularGNNResumeError("completed finalization inventory changed")
                return self.staging, True
            if self.cleanup_receipt_path.exists():
                cleanup_payload = _load_json(
                    self.cleanup_receipt_path,
                    label="finalization cleanup receipt",
                )
                original_inventory = cleanup_payload.get("inventory")
                if (
                    cleanup_payload.get("schema_version") != FINALIZATION_SCHEMA
                    or cleanup_payload.get("status")
                    not in {"CLEANUP_PREPARED", "CLEANUP_COMPLETE"}
                    or cleanup_payload.get("contract_sha256") != self.contract_sha256
                    or cleanup_payload.get("staging_dir") != str(self.staging)
                    or cleanup_payload.get("staging_identity")
                    != claim["staging_identity"]
                    or not isinstance(original_inventory, list)
                    or cleanup_payload.get("inventory_sha256")
                    != hashlib.sha256(
                        json.dumps(
                            original_inventory,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                ):
                    raise MolecularGNNResumeError(
                        "finalization cleanup receipt changed"
                    )
                original_by_path = {
                    str(row.get("path")): row
                    for row in original_inventory
                    if isinstance(row, Mapping)
                }
                if any(
                    original_by_path.get(str(row["path"])) != row
                    for row in inventory
                ):
                    raise MolecularGNNResumeError(
                        "finalization cleanup resumed with unreceipted content"
                    )
                if cleanup_payload["status"] == "CLEANUP_COMPLETE":
                    if inventory:
                        raise MolecularGNNResumeError(
                            "completed finalization cleanup is no longer empty"
                        )
                    return self.staging, False
            else:
                cleanup_payload = {
                    "schema_version": FINALIZATION_SCHEMA,
                    "status": "CLEANUP_PREPARED",
                    "contract_sha256": self.contract_sha256,
                    "staging_dir": str(self.staging),
                    "staging_identity": claim["staging_identity"],
                    "inventory": inventory,
                    "inventory_sha256": hashlib.sha256(
                        json.dumps(
                            inventory, sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")
                    ).hexdigest(),
                }
                _atomic_json_noclobber(
                    self.cleanup_receipt_path, cleanup_payload
                )
            self._verify_staging_identity()
            _clear_finalization_directory_fd(
                self._staging_fd,
                expected_inventory=cleanup_payload["inventory"],
            )
            self._verify_staging_identity()
            complete_payload = {**cleanup_payload, "status": "CLEANUP_COMPLETE"}
            _atomic_json_replace(self.cleanup_receipt_path, complete_payload)
            return self.staging, False
        if self.claim_path.exists() or self.ready_path.exists():
            raise MolecularGNNResumeError("orphan finalization authority has no staging root")
        self.staging.mkdir(mode=0o700)
        _fsync_directory(self.staging.parent)
        claim = {
            "schema_version": FINALIZATION_SCHEMA,
            "contract_sha256": self.contract_sha256,
            "output_dir": str(self.output_dir),
            "staging_dir": str(self.staging),
            "staging_identity": _directory_binding(os.lstat(self.staging)),
            "output_parent_claim_sha256": sha256_file(
                self.parent_authority.claim_path
            ),
        }
        _atomic_json_noclobber(self.claim_path, claim)
        self._claim = claim
        self._open_staging_fd()
        self._verify_staging_identity()
        self.parent_authority.verify()
        return self.staging, False

    def _ready_payload(self, inventory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if self._claim is None:
            raise MolecularGNNResumeError("finalization workspace is not prepared")
        normalized = [dict(row) for row in inventory]
        return {
            "schema_version": FINALIZATION_SCHEMA,
            "status": "BUNDLE_COMPLETE",
            "contract_sha256": self.contract_sha256,
            "output_dir": str(self.output_dir),
            "staging_dir": str(self.staging),
            "staging_identity": self._claim["staging_identity"],
            "inventory": normalized,
            "inventory_sha256": hashlib.sha256(
                json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }

    def mark_ready(self) -> dict[str, Any]:
        self.parent_authority.verify()
        self._verify_staging_identity()
        inventory = _finalization_inventory(self.staging, hash_files=True)
        payload = self._ready_payload(inventory)
        if self.ready_path.exists():
            if _load_json(self.ready_path, label="finalization completion") != payload:
                raise MolecularGNNResumeError("finalization completion changed")
        else:
            _atomic_json_noclobber(self.ready_path, payload)
        self.parent_authority.verify()
        self._verify_staging_identity()
        return payload

    def publish(self) -> None:
        self.parent_authority.verify()
        self._verify_staging_identity()
        inventory = _finalization_inventory(self.staging, hash_files=True)
        ready = _load_json(self.ready_path, label="finalization completion")
        if ready != self._ready_payload(inventory):
            raise MolecularGNNResumeError("finalization inventory drifted before publish")
        self._verify_staging_identity()
        atomic_rename_directory_noreplace(
            self.staging,
            self.output_dir,
            directory_fd=self.parent_authority.directory_fd,
        )
        self._verify_staging_identity(published=True)
        self.parent_authority.verify()

    def verify_published(self) -> dict[str, Any]:
        """Reopen and close the claim/ready/output triangle after publication."""

        self.parent_authority.verify()
        if self.staging.exists() or self.staging.is_symlink():
            raise MolecularGNNResumeError("published output still has a staging sibling")
        observed_claim = _load_json(
            self.claim_path, label="finalization staging claim"
        )
        expected_claim = {
            "schema_version": FINALIZATION_SCHEMA,
            "contract_sha256": self.contract_sha256,
            "output_dir": str(self.output_dir),
            "staging_dir": str(self.staging),
            "staging_identity": _directory_binding(
                _require_physical_directory(
                    self.output_dir, label="published output directory"
                )
            ),
            "output_parent_claim_sha256": sha256_file(
                self.parent_authority.claim_path
            ),
        }
        if observed_claim != expected_claim:
            raise MolecularGNNResumeError(
                "published finalization staging claim changed"
            )
        self._claim = observed_claim
        if self._staging_fd is None:
            self._staging_fd = os.open(
                self.output_dir,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
        self._verify_staging_identity(published=True)
        inventory = _finalization_inventory(self.output_dir, hash_files=True)
        ready = _load_json(self.ready_path, label="finalization completion")
        if ready != self._ready_payload(inventory):
            raise MolecularGNNResumeError("published finalization closure changed")
        return {
            "claim_sha256": sha256_file(self.claim_path),
            "completion_sha256": sha256_file(self.ready_path),
            "inventory_sha256": ready["inventory_sha256"],
        }

    def close(self) -> None:
        if self._staging_fd is not None:
            os.close(self._staging_fd)
            self._staging_fd = None


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        before = os.fstat(descriptor)
        named_before = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or _stat_identity(before) != _stat_identity(named_before)
        ):
            raise MolecularGNNResumeError(
                f"{label} must remain one named physical regular file"
            )
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as handle:
            payload = json.load(handle)
        after = os.fstat(descriptor)
        named_after = os.stat(path, follow_symlinks=False)
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(named_after)
        ):
            raise MolecularGNNResumeError(f"{label} changed while read")
    except (OSError, json.JSONDecodeError) as exc:
        raise MolecularGNNResumeError(f"{label} is not valid JSON") from exc
    finally:
        os.close(descriptor)
    if not isinstance(payload, dict):
        raise MolecularGNNResumeError(f"{label} must contain one JSON object")
    return payload


def _atomic_torch_noclobber(
    torch_module: Any, payload: Mapping[str, Any], path: Path
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch_module.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _torch_load(torch_module: Any, path: Path) -> Mapping[str, Any]:
    try:
        payload = torch_module.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older production PyTorch.
        payload = torch_module.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise MolecularGNNResumeError("training checkpoint payload is not a mapping")
    return payload


@dataclass(frozen=True)
class MolecularGNNResumeSnapshot:
    completed_epoch: int
    next_epoch: int
    best_epoch: int
    best_primary: float
    best_tiebreak: float | None
    epochs_without_improvement: int
    history: list[dict[str, Any]]
    best_state: Mapping[str, Any] | None


class MolecularGNNResumeStore:
    """Invocation-wide single-writer state store with inode-bound authority."""

    def __init__(
        self,
        root: str | Path,
        *,
        resume: bool,
        contract: Mapping[str, Any],
        torch_module: Any,
    ) -> None:
        unresolved = Path(os.path.abspath(Path(root).expanduser()))
        if not unresolved.is_absolute():
            raise MolecularGNNResumeError("training-state root must be absolute")
        current = Path(unresolved.anchor)
        for part in unresolved.parts[1:]:
            current = current / part
            try:
                info = os.lstat(current)
            except FileNotFoundError:
                break
            if stat.S_ISLNK(info.st_mode):
                raise MolecularGNNResumeError(
                    "training-state path may not contain symlink components"
                )
        self.root = unresolved
        self.resume = bool(resume)
        self.contract = dict(contract)
        self.contract_sha256 = canonical_sha256(self.contract)
        self.torch = torch_module
        self._root_fd: int | None = None
        self._lock_fd: int | None = None
        self._claim: dict[str, Any] | None = None
        self._contract_fd: int | None = None
        self._contract_payload: dict[str, Any] | None = None
        self._contract_evidence: dict[str, Any] | None = None

    def __enter__(self) -> "MolecularGNNResumeStore":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def open(self) -> None:
        if self._root_fd is not None:
            raise MolecularGNNResumeError("training-state store is already open")
        if self.root.exists():
            if not self.resume:
                raise MolecularGNNResumeError(
                    f"training-state root must be fresh: {self.root}"
                )
            if not self.root.is_dir() or self.root.is_symlink():
                raise MolecularGNNResumeError(
                    "training-state root must remain one physical directory"
                )
        else:
            if self.resume:
                raise MolecularGNNResumeError(
                    f"resume requested but training-state root is absent: {self.root}"
                )
            self.root.parent.mkdir(parents=True, exist_ok=True)
            self.root.mkdir(mode=0o700)
            _fsync_directory(self.root.parent)

        root_info = _require_physical_directory(
            self.root, label="training-state root"
        )
        if (
            root_info.st_uid != os.getuid()
            or stat.S_IMODE(root_info.st_mode) != 0o700
        ):
            raise MolecularGNNResumeError(
                "training-state root must be owner-bound mode 0700"
            )

        self._root_fd = os.open(
            self.root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        sentinel = self.root / ROOT_SENTINEL_NAME
        lock = self.root / LOCK_NAME
        claim_path = self.root / ROOT_CLAIM_NAME
        if not sentinel.exists():
            if any(self.root.iterdir()):
                self.close()
                raise MolecularGNNResumeError(
                    "unclaimed training-state root contains unexpected artifacts"
                )
            descriptor = os.open(
                sentinel,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                os.write(descriptor, secrets.token_bytes(32))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _fsync_directory(self.root)
        sentinel_info = _require_regular(sentinel, label="training-state sentinel")
        if (
            sentinel_info.st_uid != os.getuid()
            or stat.S_IMODE(sentinel_info.st_mode) != 0o600
        ):
            self.close()
            raise MolecularGNNResumeError(
                "training-state sentinel must be owner-bound mode 0600"
            )

        self._lock_fd = os.open(
            lock,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        lock_info = os.fstat(self._lock_fd)
        if (
            not stat.S_ISREG(lock_info.st_mode)
            or lock_info.st_nlink != 1
            or lock_info.st_uid != os.getuid()
            or stat.S_IMODE(lock_info.st_mode) != 0o600
        ):
            self.close()
            raise MolecularGNNResumeError("training-state writer lock is not regular")
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.close()
            raise MolecularGNNResumeError(
                "another molecular-GNN training writer owns this state root"
            ) from exc

        if claim_path.exists():
            claim = _load_json(claim_path, label="training-state root claim")
        else:
            allowed = {ROOT_SENTINEL_NAME, LOCK_NAME}
            unexpected = sorted(path.name for path in self.root.iterdir() if path.name not in allowed)
            if unexpected:
                self.close()
                raise MolecularGNNResumeError(
                    f"unclaimed training-state root has unexpected files: {unexpected}"
                )
            claim = {
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": "molecular_gnn_training_state_root_claim",
                "root": str(self.root),
                "root_identity": _stat_identity(os.fstat(self._root_fd)),
                "sentinel": {
                    "name": ROOT_SENTINEL_NAME,
                    "sha256": sha256_file(sentinel),
                    "identity": _stat_identity(sentinel_info),
                },
                "lock": {
                    "name": LOCK_NAME,
                    "identity": _stat_identity(lock_info),
                },
                "claim_nonce": secrets.token_hex(32),
            }
            _atomic_json_noclobber(claim_path, claim)
        self._claim = claim
        self.verify_writer_authority()

        contract_path = self.root / CONTRACT_NAME
        contract_payload = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "molecular_gnn_training_contract",
            "contract": self.contract,
            "contract_sha256": self.contract_sha256,
            "root_claim_sha256": sha256_file(claim_path),
        }
        try:
            if contract_path.exists():
                observed = _load_json(contract_path, label="training contract")
                if observed != contract_payload:
                    raise MolecularGNNResumeError(
                        "training resume contract changed"
                    )
            else:
                _atomic_json_noclobber(contract_path, contract_payload)
            self._contract_fd = os.open(
                contract_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            observed, data, info = _read_held_json_physical(
                contract_path, self._contract_fd, label="training contract"
            )
            _validate_contract_payload(
                observed,
                expected_contract_sha256=self.contract_sha256,
                expected_contract=self.contract,
            )
            if observed.get("root_claim_sha256") != sha256_file(claim_path):
                raise MolecularGNNResumeError(
                    "training contract root claim changed"
                )
            self._contract_payload = observed
            self._contract_evidence = _contract_physical_evidence(
                payload=observed, data=data, info=info
            )
            self.verify_writer_authority()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._contract_fd is not None:
            os.close(self._contract_fd)
            self._contract_fd = None
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None
        if self._root_fd is not None:
            os.close(self._root_fd)
            self._root_fd = None

    def verify_writer_authority(self) -> None:
        if self._root_fd is None or self._lock_fd is None or self._claim is None:
            raise MolecularGNNResumeError("training-state writer authority is not open")
        if self.root.is_symlink() or not self.root.is_dir():
            raise MolecularGNNResumeError("training-state root path changed")
        root_now = os.lstat(self.root)
        root_held = os.fstat(self._root_fd)
        expected_root = self._claim.get("root_identity", {})
        if (
            (root_now.st_dev, root_now.st_ino) != (root_held.st_dev, root_held.st_ino)
            or int(expected_root.get("device", -1)) != root_held.st_dev
            or int(expected_root.get("inode", -1)) != root_held.st_ino
            or int(expected_root.get("mode", -1)) != root_held.st_mode
            or int(expected_root.get("uid", -1)) != root_held.st_uid
            or self._claim.get("schema_version") != SCHEMA_VERSION
            or self._claim.get("artifact_kind")
            != "molecular_gnn_training_state_root_claim"
            or self._claim.get("root") != str(self.root)
        ):
            raise MolecularGNNResumeError("training-state root inode changed")
        sentinel = self.root / ROOT_SENTINEL_NAME
        sentinel_now = _require_regular(sentinel, label="training-state sentinel")
        expected_sentinel = self._claim.get("sentinel", {})
        if (
            expected_sentinel.get("name") != ROOT_SENTINEL_NAME
            or _stat_identity(sentinel_now) != expected_sentinel.get("identity")
            or sha256_file(sentinel) != expected_sentinel.get("sha256")
        ):
            raise MolecularGNNResumeError("training-state sentinel changed")
        lock = self.root / LOCK_NAME
        lock_now = _require_regular(lock, label="training-state writer lock")
        lock_held = os.fstat(self._lock_fd)
        expected_lock = self._claim.get("lock", {}).get("identity")
        if (
            self._claim.get("lock", {}).get("name") != LOCK_NAME
            or (lock_now.st_dev, lock_now.st_ino)
            != (lock_held.st_dev, lock_held.st_ino)
            or _stat_identity(lock_held) != expected_lock
        ):
            raise MolecularGNNResumeError("training-state writer lock inode changed")
        claim_path = self.root / ROOT_CLAIM_NAME
        observed_claim = _load_json(claim_path, label="training-state root claim")
        if observed_claim != self._claim:
            raise MolecularGNNResumeError("training-state root claim changed")
        if self._contract_fd is not None:
            if self._contract_payload is None or self._contract_evidence is None:
                raise MolecularGNNResumeError("training contract authority is incomplete")
            observed, data, info = _read_held_json_physical(
                self.root / CONTRACT_NAME,
                self._contract_fd,
                label="training contract",
            )
            _validate_contract_payload(
                observed,
                expected_contract_sha256=self.contract_sha256,
                expected_contract=self.contract,
            )
            evidence = _contract_physical_evidence(
                payload=observed, data=data, info=info
            )
            if (
                observed != self._contract_payload
                or evidence != self._contract_evidence
                or observed.get("root_claim_sha256") != sha256_file(claim_path)
            ):
                raise MolecularGNNResumeError(
                    "training contract physical inode/content/hash changed"
                )

    @property
    def contract_evidence(self) -> dict[str, Any]:
        self.verify_writer_authority()
        if self._contract_evidence is None:
            raise MolecularGNNResumeError("training contract evidence is unavailable")
        return copy.deepcopy(self._contract_evidence)

    def has_checkpoint(self) -> bool:
        self.verify_writer_authority()
        return (self.root / LATEST_NAME).is_file()

    def _reconcile_unpublished_checkpoint(self) -> None:
        """Promote one fully fsynced state whose JSON publication was interrupted."""

        latest_path = self.root / LATEST_NAME
        current: dict[str, Any] | None = None
        current_epoch = 0
        current_sha: str | None = None
        committed_names: set[str] = set()
        if latest_path.exists():
            current = _load_json(latest_path, label="latest training checkpoint")
            if (
                current.get("schema_version") != SCHEMA_VERSION
                or current.get("status") != "CHECKPOINT_COMPLETE"
                or current.get("contract_sha256") != self.contract_sha256
                or current.get("training_contract_evidence")
                != self.contract_evidence
            ):
                raise MolecularGNNResumeError("latest training checkpoint changed")
            current_epoch = int(current.get("completed_epoch", -1))
            current_sha = str(current.get("checkpoint_sha256"))
            committed_names.add(str(current.get("checkpoint_file")))
            previous_name = current.get("previous_checkpoint_file")
            if isinstance(previous_name, str):
                committed_names.add(previous_name)
        candidates = [
            path
            for path in sorted(self.root.glob("checkpoint-*.pt"))
            if path.name not in committed_names
        ]
        newer: list[tuple[Path, Mapping[str, Any]]] = []
        for path in candidates:
            state = _torch_load(self.torch, path)
            epoch = int(state.get("completed_epoch", -1))
            if epoch <= current_epoch:
                continue
            if (
                state.get("schema_version") != SCHEMA_VERSION
                or state.get("contract_sha256") != self.contract_sha256
                or state.get("training_contract_evidence")
                != self.contract_evidence
                or int(state.get("next_epoch", -1)) != epoch + 1
                or state.get("previous_checkpoint_sha256") != current_sha
                or not isinstance(state.get("history"), list)
                or len(state["history"]) != epoch
                or not isinstance(state.get("metrics"), Mapping)
            ):
                raise MolecularGNNResumeError(
                    "unpublished training checkpoint cannot be reconciled"
                )
            newer.append((path, state))
        if not newer:
            return
        if len(newer) != 1 or int(newer[0][1]["completed_epoch"]) != current_epoch + 1:
            raise MolecularGNNResumeError(
                "training-state root has ambiguous unpublished checkpoints"
            )
        checkpoint_path, state = newer[0]
        checkpoint_sha = sha256_file(checkpoint_path)
        recovered = {
            "schema_version": SCHEMA_VERSION,
            "status": "CHECKPOINT_COMPLETE",
            "contract_sha256": self.contract_sha256,
            "training_contract_evidence": self.contract_evidence,
            "completed_epoch": int(state["completed_epoch"]),
            "next_epoch": int(state["next_epoch"]),
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "previous_checkpoint_file": (
                None if current is None else current.get("checkpoint_file")
            ),
            "previous_checkpoint_sha256": current_sha,
            "metrics": dict(state["metrics"]),
            "updated_at": _utc_now(),
            "reconciled_after_interrupted_publication": True,
        }
        self.verify_writer_authority()
        _atomic_json_replace(latest_path, recovered)
        _atomic_json_replace(
            self.root / HEARTBEAT_NAME,
            {
                **recovered,
                "artifact_kind": "molecular_gnn_training_heartbeat",
                "writer_pid": os.getpid(),
            },
        )
        self.verify_writer_authority()

    def load(self, *, model: Any, optimizer: Any) -> MolecularGNNResumeSnapshot | None:
        self.verify_writer_authority()
        self._reconcile_unpublished_checkpoint()
        manifest_path = self.root / LATEST_NAME
        if not manifest_path.exists():
            if self.resume:
                allowed = {
                    ROOT_SENTINEL_NAME,
                    LOCK_NAME,
                    ROOT_CLAIM_NAME,
                    CONTRACT_NAME,
                }
                unexpected = sorted(
                    path.name
                    for path in self.root.iterdir()
                    if path.name not in allowed and not path.name.endswith(".tmp")
                )
                if unexpected:
                    raise MolecularGNNResumeError(
                        "resume root has no committed checkpoint but contains artifacts: "
                        f"{unexpected}"
                    )
            return None
        manifest = _load_json(manifest_path, label="latest training checkpoint")
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("status") != "CHECKPOINT_COMPLETE"
            or manifest.get("contract_sha256") != self.contract_sha256
            or manifest.get("training_contract_evidence")
            != self.contract_evidence
        ):
            raise MolecularGNNResumeError("latest training checkpoint contract changed")
        relative = manifest.get("checkpoint_file")
        if not isinstance(relative, str) or Path(relative).name != relative:
            raise MolecularGNNResumeError("training checkpoint path is not root-relative")
        checkpoint_path = self.root / relative
        info = _require_regular(checkpoint_path, label="training checkpoint state")
        if (
            info.st_size != int(manifest.get("checkpoint_bytes", -1))
            or sha256_file(checkpoint_path) != manifest.get("checkpoint_sha256")
        ):
            raise MolecularGNNResumeError("training checkpoint state hash changed")
        state = _torch_load(self.torch, checkpoint_path)
        completed_epoch = int(manifest.get("completed_epoch", -1))
        if (
            state.get("schema_version") != SCHEMA_VERSION
            or state.get("contract_sha256") != self.contract_sha256
            or state.get("training_contract_evidence")
            != self.contract_evidence
            or int(state.get("completed_epoch", -1)) != completed_epoch
            or int(state.get("next_epoch", -1)) != completed_epoch + 1
        ):
            raise MolecularGNNResumeError("training checkpoint payload is inconsistent")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        self.torch.set_rng_state(state["torch_cpu_rng_state"])
        cuda_count = int(state.get("torch_cuda_device_count", 0))
        cuda_states = state.get("torch_cuda_rng_states", [])
        if cuda_count:
            if not self.torch.cuda.is_available() or self.torch.cuda.device_count() != cuda_count:
                raise MolecularGNNResumeError("CUDA device count changed across GNN resume")
            self.torch.cuda.set_rng_state_all(cuda_states)
        history = state.get("history")
        if not isinstance(history, list) or len(history) != completed_epoch:
            raise MolecularGNNResumeError("training checkpoint history is inconsistent")
        self.verify_writer_authority()
        return MolecularGNNResumeSnapshot(
            completed_epoch=completed_epoch,
            next_epoch=completed_epoch + 1,
            best_epoch=int(state.get("best_epoch", 0)),
            best_primary=float(state.get("best_primary", float("-inf"))),
            best_tiebreak=(
                None
                if state.get("best_tiebreak") is None
                else float(state["best_tiebreak"])
            ),
            epochs_without_improvement=int(state.get("epochs_without_improvement", 0)),
            history=[dict(row) for row in history],
            best_state=state.get("best_state"),
        )

    def save(
        self,
        *,
        completed_epoch: int,
        model: Any,
        optimizer: Any,
        best_state: Mapping[str, Any] | None,
        best_epoch: int,
        best_primary: float,
        best_tiebreak: float | None,
        epochs_without_improvement: int,
        history: Sequence[Mapping[str, Any]],
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        self.verify_writer_authority()
        if completed_epoch <= 0 or len(history) != completed_epoch:
            raise MolecularGNNResumeError("checkpoint epoch/history is inconsistent")
        previous_sha: str | None = None
        previous_file: str | None = None
        latest_path = self.root / LATEST_NAME
        if latest_path.exists():
            previous = _load_json(latest_path, label="previous training checkpoint")
            previous_relative = previous.get("checkpoint_file")
            if (
                previous.get("schema_version") != SCHEMA_VERSION
                or previous.get("status") != "CHECKPOINT_COMPLETE"
                or previous.get("contract_sha256") != self.contract_sha256
                or previous.get("training_contract_evidence")
                != self.contract_evidence
                or not isinstance(previous_relative, str)
                or Path(previous_relative).name != previous_relative
            ):
                raise MolecularGNNResumeError("previous training checkpoint changed")
            previous_path = self.root / previous_relative
            previous_info = _require_regular(
                previous_path, label="previous training checkpoint state"
            )
            if (
                previous_info.st_size != int(previous.get("checkpoint_bytes", -1))
                or sha256_file(previous_path) != previous.get("checkpoint_sha256")
            ):
                raise MolecularGNNResumeError(
                    "previous training checkpoint state hash changed"
                )
            if int(previous.get("completed_epoch", -1)) >= completed_epoch:
                raise MolecularGNNResumeError("training checkpoint epoch did not advance")
            previous_sha = str(previous.get("checkpoint_sha256"))
            previous_file = str(previous.get("checkpoint_file"))
        checkpoint_name = f"checkpoint-{completed_epoch:06d}.pt"
        checkpoint_path = self.root / checkpoint_name
        payload = {
            "schema_version": SCHEMA_VERSION,
            "contract_sha256": self.contract_sha256,
            "training_contract_evidence": self.contract_evidence,
            "completed_epoch": int(completed_epoch),
            "next_epoch": int(completed_epoch + 1),
            "previous_checkpoint_sha256": previous_sha,
            "model_state": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "best_state": (
                None
                if best_state is None
                else {
                    key: value.detach().cpu().clone()
                    for key, value in best_state.items()
                }
            ),
            "best_epoch": int(best_epoch),
            "best_primary": float(best_primary),
            "best_tiebreak": (
                None if best_tiebreak is None else float(best_tiebreak)
            ),
            "epochs_without_improvement": int(epochs_without_improvement),
            "history": [dict(row) for row in history],
            "metrics": dict(metrics),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_cpu_rng_state": self.torch.get_rng_state(),
            "torch_cuda_rng_states": (
                self.torch.cuda.get_rng_state_all()
                if self.torch.cuda.is_available()
                else []
            ),
            "torch_cuda_device_count": (
                self.torch.cuda.device_count() if self.torch.cuda.is_available() else 0
            ),
        }
        _atomic_torch_noclobber(self.torch, payload, checkpoint_path)
        checkpoint_sha = sha256_file(checkpoint_path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "CHECKPOINT_COMPLETE",
            "contract_sha256": self.contract_sha256,
            "training_contract_evidence": self.contract_evidence,
            "completed_epoch": int(completed_epoch),
            "next_epoch": int(completed_epoch + 1),
            "checkpoint_file": checkpoint_name,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "previous_checkpoint_file": previous_file,
            "previous_checkpoint_sha256": previous_sha,
            "metrics": dict(metrics),
            "updated_at": _utc_now(),
        }
        self.verify_writer_authority()
        _atomic_json_replace(latest_path, manifest)
        _atomic_json_replace(
            self.root / HEARTBEAT_NAME,
            {
                **manifest,
                "artifact_kind": "molecular_gnn_training_heartbeat",
                "writer_pid": os.getpid(),
            },
        )
        self.verify_writer_authority()
        self._cleanup_old_checkpoints(keep={checkpoint_name, previous_file})
        return manifest

    def _cleanup_old_checkpoints(self, *, keep: set[str | None]) -> None:
        records: list[dict[str, Any]] = []
        cleanup_path = self.root / CLEANUP_NAME
        if cleanup_path.exists():
            payload = _load_json(cleanup_path, label="checkpoint cleanup manifest")
            raw_records = payload.get("removed")
            if not isinstance(raw_records, list):
                raise MolecularGNNResumeError("checkpoint cleanup manifest changed")
            records = [dict(row) for row in raw_records]
        for path in sorted(self.root.glob("checkpoint-*.pt")):
            if path.name in keep:
                continue
            info = _require_regular(path, label="obsolete training checkpoint")
            record = {
                "path": path.name,
                "size": info.st_size,
                "sha256": sha256_file(path),
                "reason": "superseded_by_newer_atomic_epoch_checkpoint",
                "reconstructable": True,
                "referenced_by": None,
            }
            records.append(record)
            _atomic_json_replace(
                cleanup_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_kind": "molecular_gnn_checkpoint_cleanup",
                    "removed": records,
                },
            )
            self.verify_writer_authority()
            path.unlink()
            _fsync_directory(self.root)
            self.verify_writer_authority()

    def mark_complete(
        self, *, output_dir: str | Path, output_identity: Mapping[str, Any]
    ) -> dict[str, Any]:
        self.verify_writer_authority()
        output = Path(output_dir).expanduser().resolve(strict=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "molecular_gnn_training_complete",
            "status": "PASS",
            "contract_sha256": self.contract_sha256,
            "training_contract_evidence": self.contract_evidence,
            "output_dir": str(output),
            "output_identity": dict(output_identity),
            "completed_at": _utc_now(),
        }
        complete_path = self.root / COMPLETE_NAME
        if complete_path.exists():
            observed = _load_json(complete_path, label="training completion manifest")
            stable_fields = {
                key: observed.get(key)
                for key in (
                    "schema_version",
                    "artifact_kind",
                    "status",
                    "contract_sha256",
                    "training_contract_evidence",
                    "output_dir",
                    "output_identity",
                )
            }
            expected_fields = {key: payload[key] for key in stable_fields}
            if stable_fields != expected_fields:
                raise MolecularGNNResumeError("training completion manifest changed")
            payload = observed
        else:
            _atomic_json_noclobber(complete_path, payload)
        self.verify_writer_authority()
        return payload

    def completion(self) -> dict[str, Any] | None:
        self.verify_writer_authority()
        path = self.root / COMPLETE_NAME
        if not path.exists():
            return None
        payload = _load_json(path, label="training completion manifest")
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("artifact_kind") != "molecular_gnn_training_complete"
            or payload.get("status") != "PASS"
            or payload.get("contract_sha256") != self.contract_sha256
            or payload.get("training_contract_evidence")
            != self.contract_evidence
        ):
            raise MolecularGNNResumeError("training completion manifest is invalid")
        self.verify_writer_authority()
        return payload


class MolecularGNNStateReadAuthority:
    """Hold the typed state-root/lock/claim closure without loading PyTorch."""

    def __init__(self, root: str | Path, *, contract_sha256: str) -> None:
        self.root = Path(os.path.abspath(Path(root).expanduser()))
        self.contract_sha256 = str(contract_sha256)
        self._root_fd: int | None = None
        self._lock_fd: int | None = None
        self._claim: dict[str, Any] | None = None
        self._contract_fd: int | None = None
        self._contract_payload: dict[str, Any] | None = None
        self._contract_evidence: dict[str, Any] | None = None

    def open(self) -> None:
        assert_no_symlink_components(self.root, label="training-state read authority")
        root_info = _require_physical_directory(
            self.root, label="training-state root"
        )
        if (
            root_info.st_uid != os.getuid()
            or stat.S_IMODE(root_info.st_mode) != 0o700
        ):
            raise MolecularGNNResumeError(
                "training-state root must be owner-bound mode 0700"
            )
        self._root_fd = os.open(
            self.root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        self._lock_fd = os.open(
            self.root / LOCK_NAME,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        lock_info = os.fstat(self._lock_fd)
        if (
            not stat.S_ISREG(lock_info.st_mode)
            or lock_info.st_nlink != 1
            or lock_info.st_uid != os.getuid()
            or stat.S_IMODE(lock_info.st_mode) != 0o600
        ):
            self.close()
            raise MolecularGNNResumeError(
                "training-state read lock must be owner-bound mode 0600"
            )
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.close()
            raise MolecularGNNResumeError("training-state writer remains active") from exc
        try:
            self._claim = _load_json(
                self.root / ROOT_CLAIM_NAME, label="training-state root claim"
            )
            if (
                self._claim.get("schema_version") != SCHEMA_VERSION
                or self._claim.get("artifact_kind")
                != "molecular_gnn_training_state_root_claim"
                or self._claim.get("root") != str(self.root)
                or self._claim.get("sentinel", {}).get("name")
                != ROOT_SENTINEL_NAME
                or self._claim.get("lock", {}).get("name") != LOCK_NAME
            ):
                raise MolecularGNNResumeError(
                    "training-state root claim schema/path changed"
                )
            contract_path = self.root / CONTRACT_NAME
            self._contract_fd = os.open(
                contract_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            contract, data, contract_info = _read_held_json_physical(
                contract_path,
                self._contract_fd,
                label="training-state contract",
            )
            _validate_contract_payload(
                contract, expected_contract_sha256=self.contract_sha256
            )
            if (
                contract.get("root_claim_sha256")
                != sha256_file(self.root / ROOT_CLAIM_NAME)
            ):
                raise MolecularGNNResumeError(
                    "training-state read contract changed"
                )
            self._contract_payload = contract
            self._contract_evidence = _contract_physical_evidence(
                payload=contract, data=data, info=contract_info
            )
            self.verify()
        except BaseException:
            self.close()
            raise

    def verify(self) -> None:
        if self._root_fd is None or self._lock_fd is None or self._claim is None:
            raise MolecularGNNResumeError("training-state read authority is not open")
        root_now = _require_physical_directory(self.root, label="training-state root")
        root_held = os.fstat(self._root_fd)
        expected_root = self._claim.get("root_identity", {})
        if (
            (root_now.st_dev, root_now.st_ino) != (root_held.st_dev, root_held.st_ino)
            or int(expected_root.get("device", -1)) != root_held.st_dev
            or int(expected_root.get("inode", -1)) != root_held.st_ino
            or int(expected_root.get("mode", -1)) != root_held.st_mode
            or int(expected_root.get("uid", -1)) != root_held.st_uid
        ):
            raise MolecularGNNResumeError("training-state read root inode changed")
        sentinel = self.root / ROOT_SENTINEL_NAME
        sentinel_info = _require_regular(sentinel, label="training-state sentinel")
        expected_sentinel = self._claim.get("sentinel", {})
        if (
            sentinel_info.st_uid != os.getuid()
            or stat.S_IMODE(sentinel_info.st_mode) != 0o600
            or expected_sentinel.get("name") != ROOT_SENTINEL_NAME
            or _stat_identity(sentinel_info)
            != expected_sentinel.get("identity")
            or sha256_file(sentinel) != expected_sentinel.get("sha256")
        ):
            raise MolecularGNNResumeError("training-state read sentinel changed")
        lock_path = self.root / LOCK_NAME
        lock_now = _require_regular(lock_path, label="training-state writer lock")
        lock_held = os.fstat(self._lock_fd)
        if (
            (lock_now.st_dev, lock_now.st_ino) != (lock_held.st_dev, lock_held.st_ino)
            or _stat_identity(lock_held) != self._claim.get("lock", {}).get("identity")
        ):
            raise MolecularGNNResumeError("training-state read lock inode changed")
        if _load_json(
            self.root / ROOT_CLAIM_NAME, label="training-state root claim"
        ) != self._claim:
            raise MolecularGNNResumeError("training-state read root claim changed")
        if (
            self._contract_fd is None
            or self._contract_payload is None
            or self._contract_evidence is None
        ):
            raise MolecularGNNResumeError("training-state read contract is not held")
        contract, data, info = _read_held_json_physical(
            self.root / CONTRACT_NAME,
            self._contract_fd,
            label="training-state contract",
        )
        _validate_contract_payload(
            contract, expected_contract_sha256=self.contract_sha256
        )
        evidence = _contract_physical_evidence(
            payload=contract, data=data, info=info
        )
        if (
            contract != self._contract_payload
            or evidence != self._contract_evidence
            or contract.get("root_claim_sha256")
            != sha256_file(self.root / ROOT_CLAIM_NAME)
        ):
            raise MolecularGNNResumeError(
                "training-state read contract inode/content/hash changed"
            )

    @property
    def contract_evidence(self) -> dict[str, Any]:
        self.verify()
        if self._contract_evidence is None:
            raise MolecularGNNResumeError("training-state contract evidence is absent")
        return copy.deepcopy(self._contract_evidence)

    def close(self) -> None:
        if self._contract_fd is not None:
            os.close(self._contract_fd)
            self._contract_fd = None
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None
        if self._root_fd is not None:
            os.close(self._root_fd)
            self._root_fd = None


__all__ = [
    "COMPLETE_NAME",
    "FinalizationWorkspace",
    "LATEST_NAME",
    "MolecularGNNResumeError",
    "MolecularGNNResumeSnapshot",
    "MolecularGNNResumeStore",
    "MolecularGNNStateReadAuthority",
    "OutputParentAuthority",
    "SCHEMA_VERSION",
    "atomic_rename_directory_noreplace",
    "assert_no_symlink_components",
    "canonical_sha256",
    "paths_overlap",
    "sha256_file",
]
