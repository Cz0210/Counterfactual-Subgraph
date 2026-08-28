"""Typed attempt and worker-staging authorities for managed execution v2.

The protocol gives every attempt, checkpoint, and worker staging directory an
unreused UUIDv4 plus an immutable generation token.  Worker-facing APIs can
write only raw evidence, ``SEALED.json``, and ``worker_exit.json`` metadata;
the independent verifier in :mod:`src.utils.terminal_publisher_v2` is the sole
writer of verification, gate, and terminal marker files.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import stat
from typing import Any, Mapping
import uuid

from src.utils.process_identity_v2 import (
    ProcessIdentityV2Error,
    canonical_json_bytes,
    linux_boot_id,
    require_auto_termination_disabled,
    require_uuid4,
)


ATTEMPT_MANIFEST_SCHEMA = "managed_attempt_manifest_v2"
GENERATION_TOKEN_SCHEMA = "managed_generation_token_v2"
WORKER_RAW_EVIDENCE_SCHEMA = "managed_worker_raw_evidence_v2"
WORKER_EXIT_SCHEMA = "managed_worker_exit_v2"
WORKER_SEALED_SCHEMA = "managed_worker_sealed_v2"
VERIFICATION_SCHEMA = "managed_verification_v2"
GATE_SCHEMA = "managed_gate_v2"
PASS_MARKER = "PASS"
FAILED_MARKER = "FAILED"

_WORKER_METADATA_NAMES = frozenset(
    {"raw_evidence.json", "SEALED.json", "worker_exit.json"}
)
_FORBIDDEN_WORKER_NAMES = frozenset(
    {
        PASS_MARKER,
        FAILED_MARKER,
        "gate.json",
        "verification.json",
        "adoption_receipt.json",
    }
)


class ManagedExecutionV2Error(RuntimeError):
    """Raised when an immutable managed-v2 authority cannot be proved."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ManagedExecutionV2Error(f"{label} is not lowercase SHA256")
    return value


def _absolute_existing_directory(path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(os.path.abspath(candidate))
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ManagedExecutionV2Error(f"{label} is unavailable") from exc
    if resolved != candidate:
        raise ManagedExecutionV2Error(f"{label} contains a symlink or alias")
    return candidate


def _open_directory(path: Path, *, label: str) -> int:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    observed = os.fstat(descriptor)
    named = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or observed.st_dev != named.st_dev
        or observed.st_ino != named.st_ino
        or observed.st_uid != os.getuid()
    ):
        os.close(descriptor)
        raise ManagedExecutionV2Error(f"{label} authority changed")
    return descriptor


def _open_directory_at(parent_descriptor: int, name: str, *, label: str) -> int:
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_descriptor,
    )
    observed = os.fstat(descriptor)
    named = os.stat(
        name, dir_fd=parent_descriptor, follow_symlinks=False
    )
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or observed.st_dev != named.st_dev
        or observed.st_ino != named.st_ino
        or observed.st_uid != os.getuid()
    ):
        os.close(descriptor)
        raise ManagedExecutionV2Error(f"{label} authority changed")
    return descriptor


def _directory_identity(descriptor: int) -> dict[str, int]:
    info = os.fstat(descriptor)
    return {
        "st_dev": int(info.st_dev),
        "st_ino": int(info.st_ino),
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "mode": int(info.st_mode),
        "uid": int(info.st_uid),
    }


def _physical_identity(info: os.stat_result) -> dict[str, int]:
    return {
        "st_dev": int(info.st_dev),
        "st_ino": int(info.st_ino),
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "mode": int(info.st_mode),
        "uid": int(info.st_uid),
        "nlink": int(info.st_nlink),
    }


def _same_physical(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _read_descriptor(descriptor: int, *, maximum_bytes: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    total = 0
    while True:
        block = os.read(descriptor, min(64 * 1024, maximum_bytes - total + 1))
        if not block:
            return b"".join(chunks)
        chunks.append(block)
        total += len(block)
        if total > maximum_bytes:
            raise ManagedExecutionV2Error("managed metadata exceeds read bound")


@dataclass(slots=True)
class HeldJSONV2:
    path: Path
    parent_descriptor: int
    name: str
    descriptor: int
    identity: Mapping[str, int]
    sha256: str
    payload: Mapping[str, Any]
    data: bytes
    _closed: bool = False

    @classmethod
    def open_at(
        cls,
        parent_descriptor: int,
        parent_path: Path,
        name: str,
        *,
        label: str,
        maximum_bytes: int = 4 * 1024 * 1024,
    ) -> "HeldJSONV2":
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        try:
            before = os.fstat(descriptor)
            named = os.stat(
                name, dir_fd=parent_descriptor, follow_symlinks=False
            )
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(named.st_mode)
                or not _same_physical(before, named)
                or before.st_uid != os.getuid()
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != 0o600
            ):
                raise ManagedExecutionV2Error(f"{label} is not immutable owner evidence")
            data = _read_descriptor(descriptor, maximum_bytes=maximum_bytes)
            after = os.fstat(descriptor)
            named_after = os.stat(
                name, dir_fd=parent_descriptor, follow_symlinks=False
            )
            if (
                _physical_identity(before) != _physical_identity(after)
                or not _same_physical(after, named_after)
            ):
                raise ManagedExecutionV2Error(f"{label} changed while opened")
            try:
                payload = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ManagedExecutionV2Error(f"{label} is not JSON") from exc
            if not isinstance(payload, dict) or canonical_json_bytes(payload) != data:
                raise ManagedExecutionV2Error(f"{label} is not canonical JSON")
            return cls(
                path=parent_path / name,
                parent_descriptor=parent_descriptor,
                name=name,
                descriptor=descriptor,
                identity=_physical_identity(after),
                sha256=hashlib.sha256(data).hexdigest(),
                payload=payload,
                data=data,
            )
        except BaseException:
            os.close(descriptor)
            raise

    def revalidate(self) -> Mapping[str, Any]:
        if self._closed:
            raise ManagedExecutionV2Error("held JSON authority is closed")
        held = os.fstat(self.descriptor)
        named = os.stat(
            self.name,
            dir_fd=self.parent_descriptor,
            follow_symlinks=False,
        )
        if (
            _physical_identity(held) != dict(self.identity)
            or not _same_physical(held, named)
            or _read_descriptor(self.descriptor, maximum_bytes=len(self.data) + 1)
            != self.data
        ):
            raise ManagedExecutionV2Error("held JSON authority changed")
        return self.payload

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            os.close(self.descriptor)
        except BaseException:
            pass
        self.descriptor = -1


def _write_bytes_exclusive_at(
    parent_descriptor: int,
    name: str,
    data: bytes,
    *,
    mode: int = 0o600,
) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        mode,
        dir_fd=parent_descriptor,
    )
    try:
        view = memoryview(data)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise ManagedExecutionV2Error("managed metadata write was short")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(parent_descriptor)


def _write_json_exclusive_at(
    parent_descriptor: int, name: str, payload: Mapping[str, Any]
) -> None:
    _write_bytes_exclusive_at(
        parent_descriptor, name, canonical_json_bytes(payload)
    )


def _new_uuid4() -> str:
    return str(uuid.uuid4())


def _generation_payload(*, attempt_id: str, generation_token: str, kind: str) -> dict[str, Any]:
    return {
        "schema_version": GENERATION_TOKEN_SCHEMA,
        "attempt_id": require_uuid4(attempt_id, label="attempt_id"),
        "generation_token": require_uuid4(
            generation_token, label="generation_token"
        ),
        "kind": kind,
    }


def _ensure_child_directory(parent_descriptor: int, name: str) -> int:
    try:
        os.mkdir(name, 0o700, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    except FileExistsError:
        pass
    return _open_directory_at(parent_descriptor, name, label=name)


@dataclass(slots=True)
class HeldManagedAttemptV2:
    stage_root: Path
    attempts_path: Path
    attempt_path: Path
    attempt_id: str
    generation_token: str
    stage_descriptor: int
    attempts_descriptor: int
    attempt_descriptor: int
    attempt_identity: Mapping[str, int]
    token: HeldJSONV2
    manifest: HeldJSONV2
    _closed: bool = False

    def revalidate(self) -> Mapping[str, Any]:
        if self._closed:
            raise ManagedExecutionV2Error("managed attempt authority is closed")
        held = os.fstat(self.attempt_descriptor)
        named = os.stat(
            self.attempt_id,
            dir_fd=self.attempts_descriptor,
            follow_symlinks=False,
        )
        if (
            not _same_physical(held, named)
            or held.st_dev != self.attempt_identity["st_dev"]
            or held.st_ino != self.attempt_identity["st_ino"]
        ):
            raise ManagedExecutionV2Error("managed attempt directory suffered ABA")
        token = self.token.revalidate()
        manifest = self.manifest.revalidate()
        if (
            token.get("attempt_id") != self.attempt_id
            or token.get("generation_token") != self.generation_token
            or manifest.get("attempt_id") != self.attempt_id
            or manifest.get("generation_token") != self.generation_token
            or manifest.get("attempt_path") != str(self.attempt_path)
        ):
            raise ManagedExecutionV2Error("managed attempt generation binding changed")
        return manifest

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.manifest.close()
        self.token.close()
        for descriptor_name in (
            "attempt_descriptor",
            "attempts_descriptor",
            "stage_descriptor",
        ):
            descriptor = getattr(self, descriptor_name)
            setattr(self, descriptor_name, -1)
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except BaseException:
                    pass

    def __enter__(self) -> "HeldManagedAttemptV2":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def create_managed_attempt(
    *,
    stage_root: str | Path,
    controller_id: str,
    task_id: str,
    git_commit: str,
    config_hash: str,
    input_hashes: Mapping[str, str],
    attempt_id: str | None = None,
    created_at: str | None = None,
    hostname: str | None = None,
    boot_id: str | None = None,
) -> HeldManagedAttemptV2:
    """Create one permanent, never-reused UUIDv4 attempt directory."""

    require_auto_termination_disabled()
    root = _absolute_existing_directory(stage_root, label="stage_root")
    selected_attempt = require_uuid4(
        _new_uuid4() if attempt_id is None else attempt_id,
        label="attempt_id",
    )
    if not controller_id or not task_id:
        raise ManagedExecutionV2Error("controller_id/task_id is absent")
    if not isinstance(git_commit, str) or not git_commit:
        raise ManagedExecutionV2Error("git_commit is absent")
    _require_sha256(config_hash, label="config_hash")
    normalized_inputs = {
        str(name): _require_sha256(value, label=f"input_hashes[{name!r}]")
        for name, value in sorted(input_hashes.items())
    }
    generation_token = _new_uuid4()
    if boot_id is None:
        try:
            selected_boot_id = linux_boot_id()
        except ProcessIdentityV2Error:
            selected_boot_id = f"non-linux-review-{socket.gethostname()}"
    else:
        selected_boot_id = boot_id
    stage_descriptor = attempts_descriptor = attempt_descriptor = -1
    try:
        stage_descriptor = _open_directory(root, label="stage_root")
        attempts_descriptor = _ensure_child_directory(stage_descriptor, "attempts")
        try:
            os.mkdir(selected_attempt, 0o700, dir_fd=attempts_descriptor)
        except FileExistsError as exc:
            raise ManagedExecutionV2Error(
                "attempt UUID/path already exists and may never be reused"
            ) from exc
        os.fsync(attempts_descriptor)
        attempt_descriptor = _open_directory_at(
            attempts_descriptor, selected_attempt, label="managed attempt"
        )
        attempt_path = root / "attempts" / selected_attempt
        _write_json_exclusive_at(
            attempt_descriptor,
            ".generation_token.json",
            _generation_payload(
                attempt_id=selected_attempt,
                generation_token=generation_token,
                kind="ATTEMPT",
            ),
        )
        manifest_payload = {
            "schema_version": ATTEMPT_MANIFEST_SCHEMA,
            "status": "ACTIVE",
            "attempt_id": selected_attempt,
            "controller_id": controller_id,
            "task_id": task_id,
            "git_commit": git_commit,
            "config_hash": config_hash,
            "input_hashes": normalized_inputs,
            "created_at": created_at or utc_now(),
            "hostname": hostname or socket.gethostname(),
            "boot_id": selected_boot_id,
            "attempt_path": str(attempt_path),
            "generation_token": generation_token,
            "auto_terminate_uncontrolled_children": False,
        }
        _write_json_exclusive_at(
            attempt_descriptor, "attempt_manifest.json", manifest_payload
        )
        token = HeldJSONV2.open_at(
            attempt_descriptor,
            attempt_path,
            ".generation_token.json",
            label="attempt generation token",
        )
        manifest = HeldJSONV2.open_at(
            attempt_descriptor,
            attempt_path,
            "attempt_manifest.json",
            label="attempt manifest",
        )
        held = HeldManagedAttemptV2(
            stage_root=root,
            attempts_path=root / "attempts",
            attempt_path=attempt_path,
            attempt_id=selected_attempt,
            generation_token=generation_token,
            stage_descriptor=stage_descriptor,
            attempts_descriptor=attempts_descriptor,
            attempt_descriptor=attempt_descriptor,
            attempt_identity=_directory_identity(attempt_descriptor),
            token=token,
            manifest=manifest,
        )
        held.revalidate()
        return held
    except BaseException:
        for descriptor in (
            attempt_descriptor,
            attempts_descriptor,
            stage_descriptor,
        ):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except BaseException:
                    pass
        # An attempt path is intentionally never removed, even after partial
        # creation.  Its UUID is burned and a later invocation must choose a
        # fresh UUID rather than creating an ABA-equivalent directory.
        raise


@dataclass(slots=True)
class HeldCheckpointV2:
    path: Path
    checkpoint_id: str
    generation_token: str
    parent_descriptor: int
    descriptor: int
    identity: Mapping[str, int]
    token: HeldJSONV2
    _closed: bool = False

    def revalidate(self) -> None:
        if self._closed:
            raise ManagedExecutionV2Error("checkpoint authority is closed")
        held = os.fstat(self.descriptor)
        named = os.stat(
            self.checkpoint_id,
            dir_fd=self.parent_descriptor,
            follow_symlinks=False,
        )
        payload = self.token.revalidate()
        if (
            held.st_dev != self.identity["st_dev"]
            or held.st_ino != self.identity["st_ino"]
            or not _same_physical(held, named)
            or payload.get("generation_token") != self.generation_token
        ):
            raise ManagedExecutionV2Error("checkpoint directory suffered ABA")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.token.close()
        for descriptor in (self.descriptor, self.parent_descriptor):
            try:
                os.close(descriptor)
            except BaseException:
                pass
        self.descriptor = self.parent_descriptor = -1


def create_checkpoint_directory(
    attempt: HeldManagedAttemptV2,
    *,
    checkpoint_id: str | None = None,
) -> HeldCheckpointV2:
    attempt.revalidate()
    selected = require_uuid4(
        _new_uuid4() if checkpoint_id is None else checkpoint_id,
        label="checkpoint_id",
    )
    checkpoints_descriptor = _ensure_child_directory(
        attempt.attempt_descriptor, "checkpoints"
    )
    checkpoint_descriptor = -1
    try:
        try:
            os.mkdir(selected, 0o700, dir_fd=checkpoints_descriptor)
        except FileExistsError as exc:
            raise ManagedExecutionV2Error(
                "checkpoint UUID/path already exists and may never be reused"
            ) from exc
        os.fsync(checkpoints_descriptor)
        checkpoint_descriptor = _open_directory_at(
            checkpoints_descriptor, selected, label="managed checkpoint"
        )
        generation_token = _new_uuid4()
        _write_json_exclusive_at(
            checkpoint_descriptor,
            ".generation_token.json",
            _generation_payload(
                attempt_id=attempt.attempt_id,
                generation_token=generation_token,
                kind="CHECKPOINT",
            ),
        )
        token = HeldJSONV2.open_at(
            checkpoint_descriptor,
            attempt.attempt_path / "checkpoints" / selected,
            ".generation_token.json",
            label="checkpoint generation token",
        )
        held = HeldCheckpointV2(
            path=attempt.attempt_path / "checkpoints" / selected,
            checkpoint_id=selected,
            generation_token=generation_token,
            parent_descriptor=checkpoints_descriptor,
            descriptor=checkpoint_descriptor,
            identity=_directory_identity(checkpoint_descriptor),
            token=token,
        )
        held.revalidate()
        return held
    except BaseException:
        for descriptor in (checkpoint_descriptor, checkpoints_descriptor):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except BaseException:
                    pass
        raise


@dataclass(slots=True)
class HeldWorkerStagingV2:
    attempt: HeldManagedAttemptV2
    staging_id: str
    generation_token: str
    path: Path
    artifact_root: Path
    staging_parent_descriptor: int
    descriptor: int
    artifact_descriptor: int
    identity: Mapping[str, int]
    artifact_identity: Mapping[str, int]
    token: HeldJSONV2
    artifact_token: HeldJSONV2
    _closed: bool = False

    def revalidate(self) -> None:
        if self._closed:
            raise ManagedExecutionV2Error("worker staging authority is closed")
        self.attempt.revalidate()
        held = os.fstat(self.descriptor)
        named = os.stat(
            self.staging_id,
            dir_fd=self.staging_parent_descriptor,
            follow_symlinks=False,
        )
        artifact = os.fstat(self.artifact_descriptor)
        named_artifact = os.stat(
            "artifacts", dir_fd=self.descriptor, follow_symlinks=False
        )
        token = self.token.revalidate()
        artifact_token = self.artifact_token.revalidate()
        if (
            held.st_dev != self.identity["st_dev"]
            or held.st_ino != self.identity["st_ino"]
            or artifact.st_dev != self.artifact_identity["st_dev"]
            or artifact.st_ino != self.artifact_identity["st_ino"]
            or not _same_physical(held, named)
            or not _same_physical(artifact, named_artifact)
            or token.get("generation_token") != self.generation_token
            or artifact_token.get("generation_token") != self.generation_token
            or token.get("attempt_id") != self.attempt.attempt_id
        ):
            raise ManagedExecutionV2Error("worker staging directory suffered ABA")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.artifact_token.close()
        self.token.close()
        for descriptor_name in (
            "artifact_descriptor",
            "descriptor",
            "staging_parent_descriptor",
        ):
            descriptor = getattr(self, descriptor_name)
            setattr(self, descriptor_name, -1)
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except BaseException:
                    pass

    def __enter__(self) -> "HeldWorkerStagingV2":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def create_worker_staging(
    attempt: HeldManagedAttemptV2,
    *,
    staging_id: str | None = None,
) -> HeldWorkerStagingV2:
    """Create one unique worker staging root and its artifact subdirectory."""

    attempt.revalidate()
    selected = require_uuid4(
        _new_uuid4() if staging_id is None else staging_id,
        label="staging_id",
    )
    generation_token = _new_uuid4()
    parent_descriptor = _ensure_child_directory(
        attempt.attempt_descriptor, "worker_staging"
    )
    descriptor = artifact_descriptor = -1
    try:
        try:
            os.mkdir(selected, 0o700, dir_fd=parent_descriptor)
        except FileExistsError as exc:
            raise ManagedExecutionV2Error(
                "worker staging UUID/path already exists and may never be reused"
            ) from exc
        os.fsync(parent_descriptor)
        descriptor = _open_directory_at(
            parent_descriptor, selected, label="worker staging"
        )
        _write_json_exclusive_at(
            descriptor,
            ".generation_token.json",
            _generation_payload(
                attempt_id=attempt.attempt_id,
                generation_token=generation_token,
                kind="WORKER_STAGING",
            ),
        )
        os.mkdir("artifacts", 0o700, dir_fd=descriptor)
        os.fsync(descriptor)
        artifact_descriptor = _open_directory_at(
            descriptor, "artifacts", label="worker artifact root"
        )
        _write_json_exclusive_at(
            artifact_descriptor,
            ".generation_token.json",
            _generation_payload(
                attempt_id=attempt.attempt_id,
                generation_token=generation_token,
                kind="WORKER_ARTIFACT_ROOT",
            ),
        )
        path = attempt.attempt_path / "worker_staging" / selected
        token = HeldJSONV2.open_at(
            descriptor,
            path,
            ".generation_token.json",
            label="worker staging generation token",
        )
        artifact_token = HeldJSONV2.open_at(
            artifact_descriptor,
            path / "artifacts",
            ".generation_token.json",
            label="worker artifact generation token",
        )
        held = HeldWorkerStagingV2(
            attempt=attempt,
            staging_id=selected,
            generation_token=generation_token,
            path=path,
            artifact_root=path / "artifacts",
            staging_parent_descriptor=parent_descriptor,
            descriptor=descriptor,
            artifact_descriptor=artifact_descriptor,
            identity=_directory_identity(descriptor),
            artifact_identity=_directory_identity(artifact_descriptor),
            token=token,
            artifact_token=artifact_token,
        )
        held.revalidate()
        return held
    except BaseException:
        for item in (artifact_descriptor, descriptor, parent_descriptor):
            if item >= 0:
                try:
                    os.close(item)
                except BaseException:
                    pass
        raise


def write_worker_metadata(
    staging: HeldWorkerStagingV2,
    *,
    name: str,
    payload: Mapping[str, Any],
) -> HeldJSONV2:
    """Write one worker-authorized metadata file; proof markers are rejected."""

    staging.revalidate()
    if name in _FORBIDDEN_WORKER_NAMES or name not in _WORKER_METADATA_NAMES:
        raise ManagedExecutionV2Error(
            "worker may write only raw_evidence.json, SEALED.json, and worker_exit.json"
        )
    _write_json_exclusive_at(staging.descriptor, name, payload)
    return HeldJSONV2.open_at(
        staging.descriptor,
        staging.path,
        name,
        label=f"worker metadata {name}",
    )


def write_worker_raw_evidence(
    staging: HeldWorkerStagingV2, payload: Mapping[str, Any]
) -> HeldJSONV2:
    body = {
        "schema_version": WORKER_RAW_EVIDENCE_SCHEMA,
        "attempt_id": staging.attempt.attempt_id,
        "generation_token": staging.generation_token,
        "recorded_at": utc_now(),
        "evidence": dict(payload),
    }
    return write_worker_metadata(
        staging, name="raw_evidence.json", payload=body
    )


def write_worker_exit(
    staging: HeldWorkerStagingV2, payload: Mapping[str, Any]
) -> HeldJSONV2:
    body = {
        "schema_version": WORKER_EXIT_SCHEMA,
        "attempt_id": staging.attempt.attempt_id,
        "generation_token": staging.generation_token,
        "recorded_at": utc_now(),
        "exit": dict(payload),
    }
    return write_worker_metadata(
        staging, name="worker_exit.json", payload=body
    )


def load_verified_gate(final_root: str | Path) -> Mapping[str, Any]:
    """Controller consumer: trust only an independent verifier's final root."""

    root = _absolute_existing_directory(final_root, label="verified final root")
    descriptor = _open_directory(root, label="verified final root")
    gate: HeldJSONV2 | None = None
    verification: HeldJSONV2 | None = None
    generation: HeldJSONV2 | None = None
    try:
        gate = HeldJSONV2.open_at(
            descriptor, root, "gate.json", label="independent verifier gate"
        )
        raw = gate.revalidate()
        if (
            raw.get("schema_version") != GATE_SCHEMA
            or raw.get("status") != "PASS"
            or raw.get("independent_verifier") is not True
            or raw.get("downstream_released") is not True
        ):
            raise ManagedExecutionV2Error("independent verifier gate is not PASS")
        verification = HeldJSONV2.open_at(
            descriptor,
            root,
            "verification.json",
            label="independent verification evidence",
        )
        generation = HeldJSONV2.open_at(
            descriptor,
            root,
            ".generation_token.json",
            label="published generation token",
        )
        verification_payload = verification.revalidate()
        generation_payload = generation.revalidate()
        if (
            verification.sha256 != raw.get("verification_sha256")
            or verification_payload.get("schema_version") != VERIFICATION_SCHEMA
            or verification_payload.get("status") != "PASS"
            or verification_payload.get("independent_verifier") is not True
            or verification_payload.get("attempt_id") != raw.get("attempt_id")
            or verification_payload.get("generation_token")
            != raw.get("generation_token")
            or generation_payload.get("attempt_id") != raw.get("attempt_id")
            or generation_payload.get("generation_token")
            != raw.get("generation_token")
        ):
            raise ManagedExecutionV2Error(
                "gate/verification/generation cross-binding is invalid"
            )
        pass_descriptor = os.open(
            PASS_MARKER,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=descriptor,
        )
        try:
            info = os.fstat(pass_descriptor)
            named = os.stat(
                PASS_MARKER, dir_fd=descriptor, follow_symlinks=False
            )
            data = _read_descriptor(pass_descriptor, maximum_bytes=256)
            if (
                not stat.S_ISREG(info.st_mode)
                or not _same_physical(info, named)
                or data != b"[MANAGED_EXECUTION_V2_PASS]\n"
            ):
                raise ManagedExecutionV2Error("verifier PASS marker is invalid")
        finally:
            os.close(pass_descriptor)
        gate.revalidate()
        verification.revalidate()
        generation.revalidate()
        return dict(raw)
    finally:
        if generation is not None:
            generation.close()
        if verification is not None:
            verification.close()
        if gate is not None:
            gate.close()
        os.close(descriptor)


__all__ = [
    "ATTEMPT_MANIFEST_SCHEMA",
    "FAILED_MARKER",
    "GATE_SCHEMA",
    "GENERATION_TOKEN_SCHEMA",
    "HeldCheckpointV2",
    "HeldJSONV2",
    "HeldManagedAttemptV2",
    "HeldWorkerStagingV2",
    "ManagedExecutionV2Error",
    "PASS_MARKER",
    "VERIFICATION_SCHEMA",
    "WORKER_EXIT_SCHEMA",
    "WORKER_RAW_EVIDENCE_SCHEMA",
    "WORKER_SEALED_SCHEMA",
    "create_checkpoint_directory",
    "create_managed_attempt",
    "create_worker_staging",
    "load_verified_gate",
    "utc_now",
    "write_worker_exit",
    "write_worker_metadata",
    "write_worker_raw_evidence",
]
