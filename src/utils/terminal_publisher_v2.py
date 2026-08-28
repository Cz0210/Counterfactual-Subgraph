"""SEALED-to-verifier atomic terminal publication for managed execution v2.

The worker closes its artifact writers and creates ``SEALED.json``.  A
different verifier process then reopens every file with ``O_NOFOLLOW``, proves
the recorded inode/token/hash inventory, writes verification/gate/PASS, and
publishes the complete directory with a no-replace atomic rename.  Cross-filesystem
publication copies into a unique directory on the destination filesystem,
fsyncs and rehashes it, then performs the same atomic rename.  No file-linking
primitive is used by this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any, Mapping
import uuid

from src.utils.managed_execution_v2 import (
    FAILED_MARKER,
    GATE_SCHEMA,
    HeldJSONV2,
    HeldWorkerStagingV2,
    ManagedExecutionV2Error,
    PASS_MARKER,
    VERIFICATION_SCHEMA,
    WORKER_EXIT_SCHEMA,
    WORKER_RAW_EVIDENCE_SCHEMA,
    WORKER_SEALED_SCHEMA,
    _open_directory,
    _open_directory_at,
    _physical_identity,
    _read_descriptor,
    _same_physical,
    _write_bytes_exclusive_at,
    _write_json_exclusive_at,
    utc_now,
    write_worker_metadata,
)
from src.utils.process_identity_v2 import canonical_json_bytes, require_uuid4


TERMINAL_PUBLICATION_SCHEMA = "managed_terminal_publication_v2"
FILE_EVIDENCE_SCHEMA = "managed_file_evidence_v2"
DIRECTORY_EVIDENCE_SCHEMA = "managed_directory_evidence_v2"

_ROOT_SEAL_NAME = "SEALED.json"
_VERIFIER_ROOT_NAMES = frozenset(
    {"verification.json", "gate.json", PASS_MARKER, FAILED_MARKER}
)


class TerminalPublisherV2Error(ManagedExecutionV2Error):
    """Raised when SEALED evidence or atomic publication is not exact."""


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise TerminalPublisherV2Error("inventory path is unsafe")
    return path


def _sha256_descriptor(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return digest.hexdigest()
        digest.update(block)


def _identity_fields(info: os.stat_result) -> dict[str, int]:
    return {
        "st_dev": int(info.st_dev),
        "st_ino": int(info.st_ino),
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
    }


def _directory_digest(
    *, relative_path: str, names: list[str], generation_token: str
) -> str:
    payload = {
        "relative_path": relative_path,
        "names": sorted(names),
        "generation_token": generation_token,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class FileEvidenceV2:
    relative_path: str
    st_dev: int
    st_ino: int
    size: int
    mtime_ns: int
    sha256: str
    attempt_id: str
    generation_token: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FILE_EVIDENCE_SCHEMA,
            "relative_path": self.relative_path,
            "st_dev": self.st_dev,
            "st_ino": self.st_ino,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "sha256": self.sha256,
            "attempt_id": self.attempt_id,
            "generation_token": self.generation_token,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FileEvidenceV2":
        if raw.get("schema_version") != FILE_EVIDENCE_SCHEMA:
            raise TerminalPublisherV2Error("file evidence schema is invalid")
        relative = str(_safe_relative_path(str(raw.get("relative_path"))))
        require_uuid4(str(raw.get("attempt_id")), label="attempt_id")
        require_uuid4(
            str(raw.get("generation_token")), label="generation_token"
        )
        sha = raw.get("sha256")
        if (
            not isinstance(sha, str)
            or len(sha) != 64
            or any(char not in "0123456789abcdef" for char in sha)
        ):
            raise TerminalPublisherV2Error("file evidence SHA256 is invalid")
        integers: dict[str, int] = {}
        for name in ("st_dev", "st_ino", "size", "mtime_ns"):
            value = raw.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TerminalPublisherV2Error(
                    f"file evidence {name} is invalid"
                )
            integers[name] = value
        return cls(
            relative_path=relative,
            sha256=sha,
            attempt_id=str(raw["attempt_id"]),
            generation_token=str(raw["generation_token"]),
            **integers,
        )


@dataclass(frozen=True, slots=True)
class DirectoryEvidenceV2:
    relative_path: str
    st_dev: int
    st_ino: int
    size: int
    mtime_ns: int
    sha256: str
    attempt_id: str
    generation_token: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DIRECTORY_EVIDENCE_SCHEMA,
            "relative_path": self.relative_path,
            "st_dev": self.st_dev,
            "st_ino": self.st_ino,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "sha256": self.sha256,
            "attempt_id": self.attempt_id,
            "generation_token": self.generation_token,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DirectoryEvidenceV2":
        if raw.get("schema_version") != DIRECTORY_EVIDENCE_SCHEMA:
            raise TerminalPublisherV2Error(
                "directory evidence schema is invalid"
            )
        relative = str(_safe_relative_path(str(raw.get("relative_path"))))
        require_uuid4(str(raw.get("attempt_id")), label="attempt_id")
        require_uuid4(
            str(raw.get("generation_token")), label="generation_token"
        )
        sha = raw.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise TerminalPublisherV2Error("directory evidence SHA is invalid")
        integers: dict[str, int] = {}
        for name in ("st_dev", "st_ino", "size", "mtime_ns"):
            value = raw.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TerminalPublisherV2Error(
                    f"directory evidence {name} is invalid"
                )
            integers[name] = value
        return cls(
            relative_path=relative,
            sha256=sha,
            attempt_id=str(raw["attempt_id"]),
            generation_token=str(raw["generation_token"]),
            **integers,
        )


@dataclass(frozen=True, slots=True)
class _ScannedInventory:
    files: tuple[FileEvidenceV2, ...]
    directories: tuple[DirectoryEvidenceV2, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "files": [item.to_dict() for item in self.files],
            "directories": [item.to_dict() for item in self.directories],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.payload())).hexdigest()


def _scan_directory(
    descriptor: int,
    *,
    relative: PurePosixPath | None,
    attempt_id: str,
    generation_token: str,
    skip_root_seal: bool,
) -> _ScannedInventory:
    before = os.fstat(descriptor)
    if not stat.S_ISDIR(before.st_mode) or before.st_uid != os.getuid():
        raise TerminalPublisherV2Error("inventory directory is unsafe")
    try:
        names = sorted(os.listdir(descriptor))
    except OSError as exc:
        raise TerminalPublisherV2Error("inventory directory is unreadable") from exc
    files: list[FileEvidenceV2] = []
    directories: list[DirectoryEvidenceV2] = []
    for name in names:
        if relative is None and skip_root_seal and name == _ROOT_SEAL_NAME:
            continue
        if relative is None and name in _VERIFIER_ROOT_NAMES:
            raise TerminalPublisherV2Error(
                "worker staging already contains verifier-only output"
            )
        if name in {"", ".", ".."} or "/" in name:
            raise TerminalPublisherV2Error("inventory entry name is unsafe")
        info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        child_relative = PurePosixPath(name) if relative is None else relative / name
        if stat.S_ISLNK(info.st_mode):
            raise TerminalPublisherV2Error(
                f"sealed artifact contains symlink: {child_relative}"
            )
        if stat.S_ISDIR(info.st_mode):
            child_descriptor = _open_directory_at(
                descriptor, name, label=f"inventory directory {child_relative}"
            )
            try:
                nested = _scan_directory(
                    child_descriptor,
                    relative=child_relative,
                    attempt_id=attempt_id,
                    generation_token=generation_token,
                    skip_root_seal=False,
                )
            finally:
                os.close(child_descriptor)
            files.extend(nested.files)
            directories.extend(nested.directories)
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise TerminalPublisherV2Error(
                f"sealed artifact contains non-regular file: {child_relative}"
            )
        if info.st_nlink != 1:
            raise TerminalPublisherV2Error(
                f"sealed artifact contains multiply-linked file: {child_relative}"
            )
        file_descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=descriptor,
        )
        try:
            opened = os.fstat(file_descriptor)
            if not _same_physical(info, opened):
                raise TerminalPublisherV2Error(
                    f"sealed artifact inode changed: {child_relative}"
                )
            digest = _sha256_descriptor(file_descriptor)
            after = os.fstat(file_descriptor)
            named_after = os.stat(
                name, dir_fd=descriptor, follow_symlinks=False
            )
            if (
                _physical_identity(opened) != _physical_identity(after)
                or not _same_physical(after, named_after)
            ):
                raise TerminalPublisherV2Error(
                    f"sealed artifact changed while hashed: {child_relative}"
                )
            identity = _identity_fields(after)
            files.append(
                FileEvidenceV2(
                    relative_path=str(child_relative),
                    sha256=digest,
                    attempt_id=attempt_id,
                    generation_token=generation_token,
                    **identity,
                )
            )
        finally:
            os.close(file_descriptor)
    after_directory = os.fstat(descriptor)
    if _physical_identity(before) != _physical_identity(after_directory):
        raise TerminalPublisherV2Error("inventory directory changed while scanned")
    if relative is not None:
        identity = _identity_fields(after_directory)
        directories.append(
            DirectoryEvidenceV2(
                relative_path=str(relative),
                sha256=_directory_digest(
                    relative_path=str(relative),
                    names=names,
                    generation_token=generation_token,
                ),
                attempt_id=attempt_id,
                generation_token=generation_token,
                **identity,
            )
        )
    return _ScannedInventory(
        files=tuple(sorted(files, key=lambda item: item.relative_path)),
        directories=tuple(
            sorted(directories, key=lambda item: item.relative_path)
        ),
    )


def _inventory_from_seal(raw: Mapping[str, Any]) -> _ScannedInventory:
    files_raw = raw.get("files")
    directories_raw = raw.get("directories")
    if not isinstance(files_raw, list) or not isinstance(directories_raw, list):
        raise TerminalPublisherV2Error("SEALED inventory is malformed")
    inventory = _ScannedInventory(
        files=tuple(FileEvidenceV2.from_mapping(item) for item in files_raw),
        directories=tuple(
            DirectoryEvidenceV2.from_mapping(item) for item in directories_raw
        ),
    )
    paths = [item.relative_path for item in inventory.files] + [
        item.relative_path for item in inventory.directories
    ]
    if len(paths) != len(set(paths)):
        raise TerminalPublisherV2Error("SEALED inventory repeats a path")
    return inventory


@dataclass(frozen=True, slots=True)
class SealedWorkerArtifactV2:
    staging_path: Path
    artifact_root: Path
    seal_path: Path
    attempt_id: str
    generation_token: str
    seal_sha256: str
    inventory_sha256: str


def seal_worker_staging(
    staging: HeldWorkerStagingV2,
) -> SealedWorkerArtifactV2:
    """Worker final action: hash closed files and publish immutable SEALED."""

    staging.revalidate()
    raw = HeldJSONV2.open_at(
        staging.descriptor,
        staging.path,
        "raw_evidence.json",
        label="worker raw evidence",
    )
    worker_exit = HeldJSONV2.open_at(
        staging.descriptor,
        staging.path,
        "worker_exit.json",
        label="worker exit evidence",
    )
    try:
        if (
            raw.payload.get("schema_version") != WORKER_RAW_EVIDENCE_SCHEMA
            or worker_exit.payload.get("schema_version") != WORKER_EXIT_SCHEMA
            or raw.payload.get("attempt_id") != staging.attempt.attempt_id
            or worker_exit.payload.get("attempt_id") != staging.attempt.attempt_id
            or raw.payload.get("generation_token") != staging.generation_token
            or worker_exit.payload.get("generation_token") != staging.generation_token
        ):
            raise TerminalPublisherV2Error(
                "worker raw/exit evidence is not bound to staging"
            )
        inventory = _scan_directory(
            staging.descriptor,
            relative=None,
            attempt_id=staging.attempt.attempt_id,
            generation_token=staging.generation_token,
            skip_root_seal=False,
        )
        root = os.fstat(staging.descriptor)
        payload = {
            "schema_version": WORKER_SEALED_SCHEMA,
            "status": "SEALED",
            "attempt_id": staging.attempt.attempt_id,
            "generation_token": staging.generation_token,
            "staging_id": staging.staging_id,
            "staging_path": str(staging.path),
            "artifact_root": str(staging.artifact_root),
            "sealed_at": utc_now(),
            "staging_identity": {
                "st_dev": int(root.st_dev),
                "st_ino": int(root.st_ino),
                "size_before_seal": int(root.st_size),
                "mtime_ns_before_seal": int(root.st_mtime_ns),
            },
            **inventory.payload(),
            "inventory_sha256": inventory.sha256,
            "worker_raw_evidence_sha256": raw.sha256,
            "worker_exit_sha256": worker_exit.sha256,
            "worker_closed": True,
            "independent_verification_required": True,
        }
        seal = write_worker_metadata(
            staging, name=_ROOT_SEAL_NAME, payload=payload
        )
        try:
            staging.revalidate()
            seal.revalidate()
            return SealedWorkerArtifactV2(
                staging_path=staging.path,
                artifact_root=staging.artifact_root,
                seal_path=seal.path,
                attempt_id=staging.attempt.attempt_id,
                generation_token=staging.generation_token,
                seal_sha256=seal.sha256,
                inventory_sha256=inventory.sha256,
            )
        finally:
            seal.close()
    finally:
        worker_exit.close()
        raw.close()


@dataclass(slots=True)
class _HeldInventoryFile:
    evidence: FileEvidenceV2
    parent_descriptor: int
    name: str
    descriptor: int
    _closed: bool = False

    def revalidate(self) -> None:
        if self._closed:
            raise TerminalPublisherV2Error("held inventory file is closed")
        held = os.fstat(self.descriptor)
        named = os.stat(
            self.name,
            dir_fd=self.parent_descriptor,
            follow_symlinks=False,
        )
        evidence_identity = {
            "st_dev": self.evidence.st_dev,
            "st_ino": self.evidence.st_ino,
            "size": self.evidence.size,
            "mtime_ns": self.evidence.mtime_ns,
        }
        if (
            _identity_fields(held) != evidence_identity
            or not _same_physical(held, named)
            or held.st_nlink != 1
            or _sha256_descriptor(self.descriptor) != self.evidence.sha256
        ):
            raise TerminalPublisherV2Error(
                f"sealed file changed: {self.evidence.relative_path}"
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            os.close(self.descriptor)
        except BaseException:
            pass
        self.descriptor = -1


@dataclass(slots=True)
class HeldSealedArtifactV2:
    sealed: SealedWorkerArtifactV2
    root_descriptor: int
    root_parent_descriptor: int
    root_name: str
    root_identity: Mapping[str, int]
    seal_document: HeldJSONV2
    generation_document: HeldJSONV2
    inventory: _ScannedInventory
    files: tuple[_HeldInventoryFile, ...]
    directory_descriptors: tuple[int, ...]
    _closed: bool = False

    @property
    def staging_path(self) -> Path:
        return self.sealed.staging_path

    @property
    def artifact_root(self) -> Path:
        return self.sealed.artifact_root

    def revalidate(self) -> Mapping[str, Any]:
        if self._closed:
            raise TerminalPublisherV2Error("held SEALED authority is closed")
        root = os.fstat(self.root_descriptor)
        named = os.stat(
            self.root_name,
            dir_fd=self.root_parent_descriptor,
            follow_symlinks=False,
        )
        if (
            root.st_dev != self.root_identity["st_dev"]
            or root.st_ino != self.root_identity["st_ino"]
            or not _same_physical(root, named)
        ):
            raise TerminalPublisherV2Error("SEALED staging directory suffered ABA")
        seal = self.seal_document.revalidate()
        token = self.generation_document.revalidate()
        if (
            seal.get("attempt_id") != self.sealed.attempt_id
            or seal.get("generation_token") != self.sealed.generation_token
            or seal.get("inventory_sha256") != self.sealed.inventory_sha256
            or token.get("attempt_id") != self.sealed.attempt_id
            or token.get("generation_token") != self.sealed.generation_token
        ):
            raise TerminalPublisherV2Error("SEALED generation binding changed")
        for held in self.files:
            held.revalidate()
        for descriptor, evidence in zip(
            self.directory_descriptors, self.inventory.directories
        ):
            info = os.fstat(descriptor)
            expected = {
                "st_dev": evidence.st_dev,
                "st_ino": evidence.st_ino,
                "size": evidence.size,
                "mtime_ns": evidence.mtime_ns,
            }
            if _identity_fields(info) != expected:
                raise TerminalPublisherV2Error(
                    f"sealed directory changed: {evidence.relative_path}"
                )
            names = sorted(os.listdir(descriptor))
            if _directory_digest(
                relative_path=evidence.relative_path,
                names=names,
                generation_token=self.sealed.generation_token,
            ) != evidence.sha256:
                raise TerminalPublisherV2Error(
                    f"sealed directory listing changed: {evidence.relative_path}"
                )
        return seal

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for held in reversed(self.files):
            held.close()
        for descriptor in reversed(self.directory_descriptors):
            try:
                os.close(descriptor)
            except BaseException:
                pass
        self.generation_document.close()
        self.seal_document.close()
        for descriptor in (self.root_descriptor, self.root_parent_descriptor):
            try:
                os.close(descriptor)
            except BaseException:
                pass
        self.root_descriptor = self.root_parent_descriptor = -1

    def __enter__(self) -> "HeldSealedArtifactV2":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _open_relative_directory_chain(
    root_descriptor: int, parts: tuple[str, ...]
) -> tuple[int, list[int]]:
    current = root_descriptor
    opened: list[int] = []
    for index, part in enumerate(parts):
        child = _open_directory_at(
            current, part, label=f"sealed directory {'/'.join(parts[: index + 1])}"
        )
        opened.append(child)
        current = child
    return current, opened


def open_sealed_worker_artifact(
    sealed_path: str | Path,
    *,
    expected_attempt_id: str | None = None,
    expected_generation_token: str | None = None,
) -> HeldSealedArtifactV2:
    """Independent verifier opener retaining the complete SEALED closure."""

    path = Path(sealed_path)
    if path.name == _ROOT_SEAL_NAME:
        root_path = path.parent
    else:
        root_path = path
    if not root_path.is_absolute():
        root_path = Path(os.path.abspath(root_path))
    try:
        resolved_root = root_path.resolve(strict=True)
    except OSError as exc:
        raise TerminalPublisherV2Error("SEALED staging root is unavailable") from exc
    if resolved_root != root_path:
        raise TerminalPublisherV2Error("SEALED staging root contains a symlink")
    parent_path = root_path.parent
    parent_descriptor = _open_directory(parent_path, label="SEALED parent")
    root_descriptor = -1
    seal: HeldJSONV2 | None = None
    token: HeldJSONV2 | None = None
    held_files: list[_HeldInventoryFile] = []
    held_directories: list[int] = []
    try:
        root_descriptor = _open_directory_at(
            parent_descriptor, root_path.name, label="SEALED staging root"
        )
        seal = HeldJSONV2.open_at(
            root_descriptor, root_path, _ROOT_SEAL_NAME, label="SEALED manifest"
        )
        raw = seal.payload
        if raw.get("schema_version") != WORKER_SEALED_SCHEMA or raw.get(
            "status"
        ) != "SEALED":
            raise TerminalPublisherV2Error("SEALED manifest status/schema is invalid")
        attempt_id = require_uuid4(str(raw.get("attempt_id")), label="attempt_id")
        generation_token = require_uuid4(
            str(raw.get("generation_token")), label="generation_token"
        )
        if expected_attempt_id is not None and attempt_id != require_uuid4(
            expected_attempt_id, label="expected_attempt_id"
        ):
            raise TerminalPublisherV2Error("SEALED attempt_id differs from expected")
        if expected_generation_token is not None and generation_token != require_uuid4(
            expected_generation_token, label="expected_generation_token"
        ):
            raise TerminalPublisherV2Error(
                "SEALED generation_token differs from expected"
            )
        staging_identity = raw.get("staging_identity")
        if not isinstance(staging_identity, Mapping):
            raise TerminalPublisherV2Error("SEALED staging identity is absent")
        root_info = os.fstat(root_descriptor)
        if (
            staging_identity.get("st_dev") != root_info.st_dev
            or staging_identity.get("st_ino") != root_info.st_ino
        ):
            raise TerminalPublisherV2Error("SEALED staging inode changed")
        token = HeldJSONV2.open_at(
            root_descriptor,
            root_path,
            ".generation_token.json",
            label="SEALED generation token",
        )
        inventory = _inventory_from_seal(raw)
        if raw.get("inventory_sha256") != inventory.sha256:
            raise TerminalPublisherV2Error("SEALED inventory SHA differs")
        observed_inventory = _scan_directory(
            root_descriptor,
            relative=None,
            attempt_id=attempt_id,
            generation_token=generation_token,
            skip_root_seal=True,
        )
        if observed_inventory.payload() != inventory.payload():
            raise TerminalPublisherV2Error(
                "SEALED physical inventory differs from its manifest"
            )
        directory_by_path: dict[str, int] = {}
        for evidence in inventory.directories:
            relative = _safe_relative_path(evidence.relative_path)
            descriptor, opened = _open_relative_directory_chain(
                root_descriptor, tuple(relative.parts)
            )
            for intermediate in opened[:-1]:
                os.close(intermediate)
            held_directories.append(descriptor)
            info = os.fstat(descriptor)
            if _identity_fields(info) != {
                "st_dev": evidence.st_dev,
                "st_ino": evidence.st_ino,
                "size": evidence.size,
                "mtime_ns": evidence.mtime_ns,
            }:
                raise TerminalPublisherV2Error(
                    f"SEALED directory identity differs: {relative}"
                )
            directory_by_path[str(relative)] = descriptor
        for evidence in inventory.files:
            relative = _safe_relative_path(evidence.relative_path)
            parent_relative = str(relative.parent)
            if parent_relative == ".":
                file_parent = root_descriptor
            else:
                file_parent = directory_by_path.get(parent_relative)
                if file_parent is None:
                    raise TerminalPublisherV2Error(
                        f"SEALED file parent is absent: {relative}"
                    )
            descriptor = os.open(
                relative.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=file_parent,
            )
            held_files.append(
                _HeldInventoryFile(
                    evidence=evidence,
                    parent_descriptor=file_parent,
                    name=relative.name,
                    descriptor=descriptor,
                )
            )
        sealed = SealedWorkerArtifactV2(
            staging_path=root_path,
            artifact_root=root_path / "artifacts",
            seal_path=root_path / _ROOT_SEAL_NAME,
            attempt_id=attempt_id,
            generation_token=generation_token,
            seal_sha256=seal.sha256,
            inventory_sha256=inventory.sha256,
        )
        held = HeldSealedArtifactV2(
            sealed=sealed,
            root_descriptor=root_descriptor,
            root_parent_descriptor=parent_descriptor,
            root_name=root_path.name,
            root_identity=_identity_fields(root_info),
            seal_document=seal,
            generation_document=token,
            inventory=inventory,
            files=tuple(held_files),
            directory_descriptors=tuple(held_directories),
        )
        held.revalidate()
        return held
    except BaseException:
        for item in reversed(held_files):
            item.close()
        for descriptor in reversed(held_directories):
            try:
                os.close(descriptor)
            except BaseException:
                pass
        if token is not None:
            token.close()
        if seal is not None:
            seal.close()
        for descriptor in (root_descriptor, parent_descriptor):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except BaseException:
                    pass
        raise


def _atomic_rename_noreplace(
    *,
    source_parent_descriptor: int,
    source_name: str,
    destination_parent_descriptor: int,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise TerminalPublisherV2Error("renameat2 is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            source_parent_descriptor,
            os.fsencode(source_name),
            destination_parent_descriptor,
            os.fsencode(destination_name),
            1,  # RENAME_NOREPLACE
        )
    elif sys.platform == "darwin":
        renameatx_np = getattr(libc, "renameatx_np", None)
        if renameatx_np is None:
            raise TerminalPublisherV2Error("renameatx_np is unavailable")
        renameatx_np.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameatx_np.restype = ctypes.c_int
        result = renameatx_np(
            source_parent_descriptor,
            os.fsencode(source_name),
            destination_parent_descriptor,
            os.fsencode(destination_name),
            0x00000004,  # RENAME_EXCL
        )
    else:
        raise TerminalPublisherV2Error(
            "platform lacks a proven atomic no-replace directory rename"
        )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(destination_name)
        if error == errno.EXDEV:
            raise OSError(error, "cross-filesystem rename")
        raise OSError(error, os.strerror(error))


def _open_or_create_relative_directory(
    root_descriptor: int, parts: tuple[str, ...]
) -> int:
    current = os.dup(root_descriptor)
    try:
        for part in parts:
            try:
                os.mkdir(part, 0o700, dir_fd=current)
                os.fsync(current)
            except FileExistsError:
                pass
            child = _open_directory_at(
                current, part, label=f"publish directory {part}"
            )
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _copy_held_inventory(
    held: HeldSealedArtifactV2,
    destination_descriptor: int,
) -> _ScannedInventory:
    held.revalidate()
    for evidence in sorted(
        held.inventory.directories,
        key=lambda item: (len(_safe_relative_path(item.relative_path).parts), item.relative_path),
    ):
        relative = _safe_relative_path(evidence.relative_path)
        descriptor = _open_or_create_relative_directory(
            destination_descriptor, tuple(relative.parts)
        )
        os.close(descriptor)
    held_by_path = {item.evidence.relative_path: item for item in held.files}
    for evidence in held.inventory.files:
        relative = _safe_relative_path(evidence.relative_path)
        parent = _open_or_create_relative_directory(
            destination_descriptor,
            tuple(relative.parent.parts) if str(relative.parent) != "." else (),
        )
        try:
            source = held_by_path[evidence.relative_path]
            source.revalidate()
            output = os.open(
                relative.name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent,
            )
            try:
                os.lseek(source.descriptor, 0, os.SEEK_SET)
                digest = hashlib.sha256()
                while True:
                    block = os.read(source.descriptor, 1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                    view = memoryview(block)
                    while view:
                        count = os.write(output, view)
                        if count <= 0:
                            raise TerminalPublisherV2Error(
                                "cross-filesystem copy was short"
                            )
                        view = view[count:]
                os.fsync(output)
                if digest.hexdigest() != evidence.sha256:
                    raise TerminalPublisherV2Error(
                        f"cross-filesystem source changed: {relative}"
                    )
            finally:
                os.close(output)
            os.fsync(parent)
        finally:
            os.close(parent)
    # SEALED is intentionally outside its own inventory; copy it separately.
    _write_bytes_exclusive_at(
        destination_descriptor,
        _ROOT_SEAL_NAME,
        held.seal_document.data,
    )
    copied = _scan_directory(
        destination_descriptor,
        relative=None,
        attempt_id=held.sealed.attempt_id,
        generation_token=held.sealed.generation_token,
        skip_root_seal=True,
    )
    source_hashes = {
        item.relative_path: item.sha256 for item in held.inventory.files
    }
    copied_hashes = {item.relative_path: item.sha256 for item in copied.files}
    if source_hashes != copied_hashes:
        raise TerminalPublisherV2Error(
            "cross-filesystem copy hash inventory differs from SEALED"
        )
    return copied


def _write_verifier_outputs(
    *,
    root_descriptor: int,
    held: HeldSealedArtifactV2,
    verification: Mapping[str, Any],
    published_inventory: _ScannedInventory,
    publish_mode: str,
) -> tuple[str, str]:
    if verification.get("status") != "PASS":
        raise TerminalPublisherV2Error(
            "independent scientific verification did not PASS"
        )
    verification_payload = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS",
        "independent_verifier": True,
        "attempt_id": held.sealed.attempt_id,
        "generation_token": held.sealed.generation_token,
        "sealed_sha256": held.sealed.seal_sha256,
        "source_inventory_sha256": held.sealed.inventory_sha256,
        "published_inventory_sha256": published_inventory.sha256,
        "published_inventory": published_inventory.payload(),
        "publish_mode": publish_mode,
        "verified_at": utc_now(),
        "verification": dict(verification),
    }
    verification_data = canonical_json_bytes(verification_payload)
    verification_sha = hashlib.sha256(verification_data).hexdigest()
    gate_payload = {
        "schema_version": GATE_SCHEMA,
        "status": "PASS",
        "independent_verifier": True,
        "attempt_id": held.sealed.attempt_id,
        "generation_token": held.sealed.generation_token,
        "verification_sha256": verification_sha,
        "sealed_sha256": held.sealed.seal_sha256,
        "published_inventory_sha256": published_inventory.sha256,
        "science_adopted": True,
        "downstream_released": True,
        "auto_terminate_uncontrolled_children": False,
    }
    gate_data = canonical_json_bytes(gate_payload)
    gate_sha = hashlib.sha256(gate_data).hexdigest()
    _write_bytes_exclusive_at(
        root_descriptor, "verification.json", verification_data
    )
    _write_bytes_exclusive_at(root_descriptor, "gate.json", gate_data)
    # PASS is the verifier's final write inside the still-private staging root.
    _write_bytes_exclusive_at(
        root_descriptor,
        PASS_MARKER,
        b"[MANAGED_EXECUTION_V2_PASS]\n",
    )
    os.fsync(root_descriptor)
    return verification_sha, gate_sha


@dataclass(frozen=True, slots=True)
class TerminalPublicationV2:
    schema_version: str
    final_path: Path
    attempt_id: str
    generation_token: str
    sealed_sha256: str
    source_inventory_sha256: str
    published_inventory_sha256: str
    verification_sha256: str
    gate_sha256: str
    publish_mode: str


def _destination_is_exact_root(
    *,
    held_descriptor: int,
    destination_parent_descriptor: int,
    destination_name: str,
) -> bool:
    try:
        held = os.fstat(held_descriptor)
        destination = os.stat(
            destination_name,
            dir_fd=destination_parent_descriptor,
            follow_symlinks=False,
        )
        return stat.S_ISDIR(destination.st_mode) and _same_physical(
            held, destination
        )
    except BaseException:
        return False


def verify_and_publish_sealed_attempt(
    held: HeldSealedArtifactV2,
    *,
    final_path: str | Path,
    verification: Mapping[str, Any],
    force_cross_filesystem: bool = False,
) -> TerminalPublicationV2:
    """Independent verifier PASS and atomic final-directory publication."""

    held.revalidate()
    destination = Path(final_path)
    if not destination.is_absolute():
        destination = Path(os.path.abspath(destination))
    parent_path = destination.parent
    try:
        resolved_parent = parent_path.resolve(strict=True)
    except OSError as exc:
        raise TerminalPublisherV2Error("terminal parent is unavailable") from exc
    if resolved_parent != parent_path:
        raise TerminalPublisherV2Error("terminal parent contains a symlink")
    destination_parent = _open_directory(parent_path, label="terminal parent")
    source_root_descriptor = held.root_descriptor
    publish_root_descriptor = source_root_descriptor
    source_parent_descriptor = held.root_parent_descriptor
    source_name = held.root_name
    publish_mode = "SAME_FILESYSTEM_ATOMIC_RENAME"
    copied_inventory = held.inventory
    temporary_parent_descriptor = -1
    temporary_name: str | None = None
    try:
        source_device = os.fstat(source_root_descriptor).st_dev
        destination_device = os.fstat(destination_parent).st_dev
        cross = force_cross_filesystem or source_device != destination_device
        if cross:
            publish_mode = "CROSS_FILESYSTEM_COPY_REHASH_ATOMIC_RENAME"
            temporary_name = (
                f".{destination.name}.publish-{uuid.uuid4()}"
            )
            os.mkdir(temporary_name, 0o700, dir_fd=destination_parent)
            os.fsync(destination_parent)
            temporary_parent_descriptor = destination_parent
            publish_root_descriptor = _open_directory_at(
                destination_parent,
                temporary_name,
                label="cross-filesystem publish staging",
            )
            copied_inventory = _copy_held_inventory(
                held, publish_root_descriptor
            )
            source_parent_descriptor = temporary_parent_descriptor
            source_name = temporary_name
        verification_sha, gate_sha = _write_verifier_outputs(
            root_descriptor=publish_root_descriptor,
            held=held,
            verification=verification,
            published_inventory=copied_inventory,
            publish_mode=publish_mode,
        )
        os.fsync(publish_root_descriptor)
        publication = TerminalPublicationV2(
            schema_version=TERMINAL_PUBLICATION_SCHEMA,
            final_path=destination,
            attempt_id=held.sealed.attempt_id,
            generation_token=held.sealed.generation_token,
            sealed_sha256=held.sealed.seal_sha256,
            source_inventory_sha256=held.sealed.inventory_sha256,
            published_inventory_sha256=copied_inventory.sha256,
            verification_sha256=verification_sha,
            gate_sha256=gate_sha,
            publish_mode=publish_mode,
        )
        try:
            _atomic_rename_noreplace(
                source_parent_descriptor=source_parent_descriptor,
                source_name=source_name,
                destination_parent_descriptor=destination_parent,
                destination_name=destination.name,
            )
            return publication
        except BaseException:
            # Recover only an exact successful rename; this check remains in
            # the verifier while its root descriptor is retained.
            if _destination_is_exact_root(
                held_descriptor=publish_root_descriptor,
                destination_parent_descriptor=destination_parent,
                destination_name=destination.name,
            ):
                return publication
            raise
    finally:
        # This finally runs only before an ordinary return completes.  Close is
        # best effort so post-visible descriptor cleanup cannot reverse PASS.
        if publish_root_descriptor != source_root_descriptor:
            try:
                os.close(publish_root_descriptor)
            except BaseException:
                pass
        try:
            os.close(destination_parent)
        except BaseException:
            pass


__all__ = [
    "DIRECTORY_EVIDENCE_SCHEMA",
    "DirectoryEvidenceV2",
    "FILE_EVIDENCE_SCHEMA",
    "FileEvidenceV2",
    "HeldSealedArtifactV2",
    "SealedWorkerArtifactV2",
    "TERMINAL_PUBLICATION_SCHEMA",
    "TerminalPublicationV2",
    "TerminalPublisherV2Error",
    "open_sealed_worker_artifact",
    "seal_worker_staging",
    "verify_and_publish_sealed_attempt",
]
