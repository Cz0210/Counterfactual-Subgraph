"""Fail-closed owner for the TasteMolNet T14 step-12,500 checkpoint audit.

This module deliberately stops before scientific deserialization.  It proves
the immutable JSON envelope and the large ``generation_state.pt`` hash using
bounded reads, inspects only the PyTorch ZIP central directory, and measures
live cgroup headroom.  It never opens the checkpoint SQLite snapshot or any
WAL/SHM sidecar and never calls :func:`torch.load`.

The current T14 checkpoint can enter a restore/save/reload canary only when its
large scientific payload is in tensor-storage entries that can be memory
mapped.  A large ``data.pkl`` is a monolithic Python pickle and therefore
cannot be represented as a safe streaming restore merely by passing
``mmap=True`` to ``torch.load``.  Such a checkpoint is closed with the typed
``BLOCKED_LOW_MEMORY_CANARY_UNAVAILABLE`` terminal instead of risking a cgroup
OOM while pretending that the load is streaming.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import time
from typing import Any, Callable, Mapping
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # Support the project's isolated ``python -I -B /absolute/path.py``
    # execution convention without trusting PYTHONPATH or the caller's cwd.
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.tastemolnet_t14_resume import (
    SOURCE_STEP,
    T14ResumeError,
    inspect_process_identity,
    load_resume_spec,
    read_cgroup_counter,
)
from src.baselines.comrecgc.contracts import stable_json_sha256


OWNER_SCHEMA = "tastemolnet_t14_checkpoint12500_audit_owner_v1"
AUDIT_SCHEMA = "tastemolnet_t14_checkpoint12500_no_sqlite_audit_v1"
ARCHIVE_SCHEMA = "tastemolnet_t14_checkpoint_archive_layout_v1"
BLOCKED_EXIT = 75
ERROR_EXIT = 70
STATE_FILENAME = "generation_state.pt"
SQLITE_FILENAME = "authoritative_graph_store.sqlite3"
MANIFEST_FILENAME = "checkpoint_manifest.json"
COMPLETE_FILENAME = "_CHECKPOINT_COMPLETE.json"
LATEST_FILENAME = "LATEST"
CHECKPOINT_SCHEMA = "comrecgc_generation_checkpoint_v2"
CHECKPOINT_BOUNDARY = "after_fully_completed_step_v1"
STATE_SCHEMA = "comrecgc_generation_state_v2"
LATEST_SCHEMA = "comrecgc_generation_checkpoint_latest_v1"
HASH_BLOCK_BYTES = 8 * 1024 * 1024
HASH_HEARTBEAT_BYTES = 512 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
# Unpickling this object graph is eager even when tensor storages are mmapped.
# Keep the admissible metadata envelope small enough that it cannot itself be
# the hundreds-of-GiB allocation observed in the historical T14 process.
MAX_STREAMABLE_PICKLE_BYTES = 64 * 1024 * 1024


class T14CheckpointAuditError(T14ResumeError):
    """The immutable step-12,500 checkpoint envelope failed verification."""


@dataclass(frozen=True, slots=True)
class ArchiveLayout:
    schema_version: str
    archive_format: str
    file_bytes: int
    entry_count: int
    compression_types: tuple[int, ...]
    pickle_entry: str | None
    pickle_bytes: int
    storage_entry_count: int
    storage_bytes: int
    non_storage_bytes: int
    metadata_limit_bytes: int
    mmap_tensor_storage_present: bool
    stream_restore_proven: bool
    block_reason: str | None
    torch_load_invoked: bool


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


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(
        _canonical_bytes({key: item for key, item in value.items() if key != field})
    ).hexdigest()


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


def _physical(path: str | Path, *, label: str, kind: str) -> Path:
    value = Path(path)
    if not value.is_absolute() or value.is_symlink():
        raise T14CheckpointAuditError(
            f"T14 {label} must be one absolute physical {kind}"
        )
    try:
        resolved = value.resolve(strict=True)
    except OSError as exc:
        raise T14CheckpointAuditError(f"T14 {label} is unavailable: {value}") from exc
    predicate = resolved.is_file if kind == "file" else resolved.is_dir
    if resolved != value or not predicate():
        raise T14CheckpointAuditError(
            f"T14 {label} contains an alias or is not a {kind}: {value}"
        )
    return value


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    physical = _physical(path, label=label, kind="file")
    size = physical.stat().st_size
    if size <= 0 or size > MAX_JSON_BYTES:
        raise T14CheckpointAuditError(
            f"T14 {label} size is outside the bounded JSON envelope: {size}"
        )
    try:
        value = json.loads(physical.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T14CheckpointAuditError(f"T14 {label} is unreadable") from exc
    if type(value) is not dict:
        raise T14CheckpointAuditError(f"T14 {label} must contain one JSON object")
    return value


def _sha256_file(
    path: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> str:
    """Hash a non-SQLite file with a fixed-size buffer and optional heartbeat."""

    if path.name == SQLITE_FILENAME or path.name.endswith(("-wal", "-shm")):
        raise T14CheckpointAuditError("T14 audit is forbidden from reading SQLite/WAL")
    physical = _physical(path, label=path.name, kind="file")
    total = physical.stat().st_size
    processed = 0
    next_report = HASH_HEARTBEAT_BYTES
    digest = hashlib.sha256()
    with physical.open("rb", buffering=0) as handle:
        while True:
            block = handle.read(HASH_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
            processed += len(block)
            if progress is not None and processed >= next_report:
                progress(processed, total)
                next_report = processed + HASH_HEARTBEAT_BYTES
    if processed != total:
        raise T14CheckpointAuditError("T14 checkpoint changed while being hashed")
    if progress is not None:
        progress(processed, total)
    return digest.hexdigest()


def inspect_archive_layout(state_path: str | Path) -> ArchiveLayout:
    """Inspect archive metadata without reading or unpickling ``data.pkl``."""

    path = _physical(state_path, label="generation state", kind="file")
    if not zipfile.is_zipfile(path):
        return ArchiveLayout(
            schema_version=ARCHIVE_SCHEMA,
            archive_format="LEGACY_OR_UNKNOWN_PICKLE",
            file_bytes=path.stat().st_size,
            entry_count=0,
            compression_types=(),
            pickle_entry=None,
            pickle_bytes=-1,
            storage_entry_count=0,
            storage_bytes=0,
            non_storage_bytes=path.stat().st_size,
            metadata_limit_bytes=MAX_STREAMABLE_PICKLE_BYTES,
            mmap_tensor_storage_present=False,
            stream_restore_proven=False,
            block_reason="STATE_IS_NOT_A_PYTORCH_ZIP_ARCHIVE",
            torch_load_invoked=False,
        )

    seen: set[str] = set()
    pickle_entries: list[tuple[str, int]] = []
    compression: set[int] = set()
    entry_count = 0
    storage_entry_count = 0
    storage_bytes = 0
    non_storage_bytes = 0
    with zipfile.ZipFile(path, mode="r") as archive:
        # ``ZipFile`` already owns one central-directory inventory.  Aggregate
        # in one pass so this audit does not retain another 421k ZipInfo list.
        for info in archive.infolist():
            entry_count += 1
            name = PurePosixPath(info.filename)
            normalized_name = str(name)
            if normalized_name in seen:
                raise T14CheckpointAuditError(
                    "T14 generation-state archive has duplicate names"
                )
            seen.add(normalized_name)
            if name.is_absolute() or ".." in name.parts:
                raise T14CheckpointAuditError(
                    "T14 generation-state archive has unsafe names"
                )
            compression.add(int(info.compress_type))
            is_storage = len(name.parts) >= 2 and name.parts[-2] == "data"
            if is_storage:
                storage_entry_count += 1
                storage_bytes += int(info.file_size)
            else:
                non_storage_bytes += int(info.file_size)
            if name.name == "data.pkl":
                pickle_entries.append((info.filename, int(info.file_size)))
    compression_types = tuple(sorted(compression))
    pickle_bytes = pickle_entries[0][1] if len(pickle_entries) == 1 else -1
    mmap_storage = storage_entry_count > 0 and compression_types == (
        zipfile.ZIP_STORED,
    )
    stream_restore_proven = (
        len(pickle_entries) == 1
        and 0 < pickle_bytes <= MAX_STREAMABLE_PICKLE_BYTES
        and mmap_storage
    )
    if len(pickle_entries) != 1:
        reason = "ARCHIVE_DATA_PICKLE_COUNT_INVALID"
    elif pickle_bytes > MAX_STREAMABLE_PICKLE_BYTES:
        reason = "MONOLITHIC_DATA_PICKLE_EXCEEDS_STREAMING_LIMIT"
    elif not mmap_storage:
        reason = "TENSOR_STORAGE_IS_NOT_MMAP_ELIGIBLE"
    else:
        reason = None
    return ArchiveLayout(
        schema_version=ARCHIVE_SCHEMA,
        archive_format="PYTORCH_ZIP",
        file_bytes=path.stat().st_size,
        entry_count=entry_count,
        compression_types=compression_types,
        pickle_entry=pickle_entries[0][0] if len(pickle_entries) == 1 else None,
        pickle_bytes=pickle_bytes,
        storage_entry_count=storage_entry_count,
        storage_bytes=storage_bytes,
        non_storage_bytes=non_storage_bytes,
        metadata_limit_bytes=MAX_STREAMABLE_PICKLE_BYTES,
        mmap_tensor_storage_present=mmap_storage,
        stream_restore_proven=stream_restore_proven,
        block_reason=reason,
        torch_load_invoked=False,
    )


def audit_checkpoint_without_sqlite(
    resume_spec_path: str | Path,
    *,
    hash_progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify all safe envelope evidence while never opening the SQLite file."""

    spec_path = _physical(resume_spec_path, label="resume spec", kind="file")
    spec = load_resume_spec(spec_path)
    checkpoint = _physical(spec["checkpoint_dir"], label="checkpoint", kind="directory")
    if checkpoint.name != "step-000000012500" or checkpoint.parent != Path(
        str(spec["checkpoint_root"])
    ):
        raise T14CheckpointAuditError("T14 checkpoint path is not the frozen 12,500 root")

    manifest_path = checkpoint / MANIFEST_FILENAME
    complete_path = checkpoint / COMPLETE_FILENAME
    latest_path = checkpoint.parent / LATEST_FILENAME
    identity_path = Path(str(spec["checkpoint_identity_path"]))
    manifest_sha = _sha256_file(manifest_path)
    identity_sha = _sha256_file(identity_path)
    if manifest_sha != spec.get("checkpoint_manifest_sha256"):
        raise T14CheckpointAuditError("T14 checkpoint manifest SHA256 changed")
    if identity_sha != spec.get("checkpoint_identity_sha256"):
        raise T14CheckpointAuditError("T14 checkpoint identity SHA256 changed")

    manifest = _json_object(manifest_path, label="checkpoint manifest")
    expected_header = (
        manifest.get("schema_version") == CHECKPOINT_SCHEMA
        and manifest.get("atomic_complete") is True
        and manifest.get("boundary") == CHECKPOINT_BOUNDARY
        and manifest.get("state_schema_version") == STATE_SCHEMA
        and manifest.get("checkpoint_dir") == checkpoint.name
        and manifest.get("completed_step") == SOURCE_STEP
        and manifest.get("next_step") == SOURCE_STEP + 1
        and manifest.get("total_steps") == 25_000
        and manifest.get("file_digest_algorithm") == "sha256"
        and manifest.get("checkpoint_digest_scheme") == "stable_json_sha256_v1"
    )
    if not expected_header:
        raise T14CheckpointAuditError("T14 checkpoint manifest contract changed")
    digest_payload = {
        key: item for key, item in manifest.items() if key != "checkpoint_digest"
    }
    if (
        manifest.get("checkpoint_digest") != stable_json_sha256(digest_payload)
        or manifest.get("checkpoint_digest") != spec.get("checkpoint_digest")
    ):
        raise T14CheckpointAuditError("T14 checkpoint digest changed")

    files = manifest.get("files")
    if type(files) is not dict or set(files) != {STATE_FILENAME, SQLITE_FILENAME}:
        raise T14CheckpointAuditError("T14 checkpoint file inventory changed")
    state_identity = files[STATE_FILENAME]
    sqlite_identity = files[SQLITE_FILENAME]
    if type(state_identity) is not dict or type(sqlite_identity) is not dict:
        raise T14CheckpointAuditError("T14 checkpoint file identities are malformed")
    state_path = _physical(checkpoint / STATE_FILENAME, label="generation state", kind="file")
    sqlite_path = _physical(
        checkpoint / SQLITE_FILENAME, label="SQLite snapshot", kind="file"
    )
    if any(Path(f"{sqlite_path}{suffix}").exists() for suffix in ("-wal", "-shm")):
        raise T14CheckpointAuditError("T14 sealed SQLite snapshot has a WAL/SHM sidecar")
    if state_path.stat().st_size != int(state_identity.get("bytes", -1)):
        raise T14CheckpointAuditError("T14 generation-state byte count changed")
    if sqlite_path.stat().st_size != int(sqlite_identity.get("bytes", -1)):
        raise T14CheckpointAuditError("T14 SQLite snapshot byte count changed")
    if sqlite_identity.get("sha256") != spec.get("sqlite_snapshot_sha256"):
        raise T14CheckpointAuditError("T14 recorded SQLite identity changed")
    state_sha = _sha256_file(state_path, progress=hash_progress)
    if (
        state_sha != state_identity.get("sha256")
        or state_sha != spec.get("generation_state_sha256")
    ):
        raise T14CheckpointAuditError("T14 generation-state SHA256 changed")

    complete = _json_object(complete_path, label="checkpoint completion marker")
    if complete != {
        "checkpoint_digest": spec["checkpoint_digest"],
        "manifest_sha256": manifest_sha,
        "schema_version": CHECKPOINT_SCHEMA,
    }:
        raise T14CheckpointAuditError("T14 checkpoint completion marker changed")
    latest = _json_object(latest_path, label="checkpoint latest pointer")
    if latest != {
        "checkpoint_digest": spec["checkpoint_digest"],
        "checkpoint_dir": checkpoint.name,
        "completed_step": SOURCE_STEP,
        "schema_version": LATEST_SCHEMA,
    }:
        raise T14CheckpointAuditError("T14 checkpoint latest pointer changed")

    audit = {
        "schema_version": AUDIT_SCHEMA,
        "status": "PASS_WITH_SQLITE_PAYLOAD_UNREAD_BY_POLICY",
        "resume_spec": str(spec_path),
        "resume_spec_file_sha256": _sha256_file(spec_path),
        "resume_spec_sha256": spec["spec_sha256"],
        "checkpoint_dir": str(checkpoint),
        "checkpoint_digest": spec["checkpoint_digest"],
        "checkpoint_manifest_sha256": manifest_sha,
        "checkpoint_identity_sha256": identity_sha,
        "generation_state_sha256": state_sha,
        "generation_state_bytes": state_path.stat().st_size,
        "sqlite_snapshot_recorded_sha256": sqlite_identity["sha256"],
        "sqlite_snapshot_bytes": sqlite_path.stat().st_size,
        "sqlite_payload_opened": False,
        "sqlite_payload_hash_recomputed": False,
        "wal_or_shm_opened": False,
        "torch_load_invoked": False,
        "manifest_and_state_hashes_pass": True,
        "written_at": _utc_now(),
    }
    audit["audit_sha256"] = _self_hash(audit, "audit_sha256")
    return spec, audit


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-spec", type=_absolute, required=True)
    parser.add_argument("--owner-root", type=_absolute, required=True)
    parser.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    return parser.parse_args(argv)


def _prepare_owner_root(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise T14CheckpointAuditError("T14 owner root must be an absolute physical path")
    path.mkdir(parents=True, exist_ok=True)
    if path.resolve(strict=True) != path:
        raise T14CheckpointAuditError("T14 owner root contains an alias")
    if (path / "terminal.json").exists():
        raise T14CheckpointAuditError("T14 owner root already has a terminal")
    return path


def _terminal(
    owner_root: Path,
    *,
    owner_pid: int,
    owner_start_ticks: int,
    status: str,
    reason_code: str,
    detail: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": OWNER_SCHEMA,
        "status": status,
        "reason_code": reason_code,
        "detail": detail,
        "owner_pid": owner_pid,
        "owner_start_ticks": owner_start_ticks,
        "science_started": False,
        "science_pid": None,
        "canary_steps_executed": 0,
        "forced_checkpoint_written": False,
        "checkpoint_reload_executed": False,
        "sqlite_payload_opened": False,
        "wal_or_shm_opened": False,
        "torch_load_invoked": False,
        "scientifically_safe_to_launch": False,
        "evidence": dict(evidence),
        "written_at": _utc_now(),
    }
    value["terminal_sha256"] = _self_hash(value, "terminal_sha256")
    _atomic_json(owner_root / "terminal.json", value)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    owner_root = _prepare_owner_root(args.owner_root)
    identity = inspect_process_identity(os.getpid(), proc_root=args.proc_root)
    if not identity.live or identity.start_ticks <= 0:
        raise T14CheckpointAuditError("T14 owner process identity is unavailable")

    heartbeat_base = {
        "schema_version": OWNER_SCHEMA,
        "owner_pid": identity.pid,
        "owner_start_ticks": identity.start_ticks,
        "science_started": False,
        "science_pid": None,
        "sqlite_payload_opened": False,
        "wal_or_shm_opened": False,
        "torch_load_invoked": False,
    }

    def heartbeat(state: str, **fields: Any) -> None:
        _atomic_json(
            owner_root / "heartbeat.json",
            {**heartbeat_base, "status": state, **fields, "written_at": _utc_now()},
        )

    heartbeat("AUDITING_IMMUTABLE_CHECKPOINT")
    try:
        last_heartbeat = 0.0

        def hash_progress(processed: int, total: int) -> None:
            nonlocal last_heartbeat
            now = time.monotonic()
            if processed == total or now - last_heartbeat >= 10.0:
                heartbeat(
                    "HASHING_GENERATION_STATE",
                    bytes_hashed=processed,
                    bytes_total=total,
                )
                last_heartbeat = now

        spec, checkpoint_audit = audit_checkpoint_without_sqlite(
            args.resume_spec,
            hash_progress=hash_progress,
        )
        _atomic_json(owner_root / "checkpoint_audit.json", checkpoint_audit)
        heartbeat("READING_CGROUP_HEADROOM")
        memory = spec.get("memory")
        if type(memory) is not dict:
            raise T14CheckpointAuditError("T14 resume spec has no memory contract")
        limit = read_cgroup_counter(memory["cgroup_limit_path"], allow_max=True)
        current = read_cgroup_counter(memory["cgroup_current_path"])
        failcnt = read_cgroup_counter(memory["cgroup_failcnt_path"])
        if current > limit:
            raise T14CheckpointAuditError("T14 cgroup current usage exceeds its limit")
        headroom = limit - current
        safety_margin = int(memory.get("safety_margin_bytes", -1))
        cgroup = {
            "cgroup_limit_path": memory["cgroup_limit_path"],
            "cgroup_current_path": memory["cgroup_current_path"],
            "cgroup_limit_bytes": limit,
            "cgroup_current_bytes": current,
            "cgroup_failcnt": failcnt,
            "cgroup_headroom_bytes": headroom,
            "required_safety_margin_bytes": safety_margin,
            "safety_margin_preserved_before_canary": (
                safety_margin > 0 and headroom >= safety_margin
            ),
        }
        _atomic_json(owner_root / "cgroup_headroom.json", cgroup)
        heartbeat("INSPECTING_CHECKPOINT_ARCHIVE", **cgroup)
        archive = inspect_archive_layout(Path(str(spec["checkpoint_dir"])) / STATE_FILENAME)
        archive_payload = asdict(archive)
        _atomic_json(owner_root / "archive_layout.json", archive_payload)
        evidence = {
            "checkpoint_audit": checkpoint_audit,
            "cgroup": cgroup,
            "archive_layout": archive_payload,
        }
        if not cgroup["safety_margin_preserved_before_canary"]:
            _terminal(
                owner_root,
                owner_pid=identity.pid,
                owner_start_ticks=identity.start_ticks,
                status="BLOCKED",
                reason_code="BLOCKED_T14_CGROUP_HEADROOM",
                detail="Live cgroup headroom does not preserve the frozen 64 GiB margin.",
                evidence=evidence,
            )
            return BLOCKED_EXIT
        if not archive.stream_restore_proven:
            _terminal(
                owner_root,
                owner_pid=identity.pid,
                owner_start_ticks=identity.start_ticks,
                status="BLOCKED",
                reason_code="BLOCKED_LOW_MEMORY_CANARY_UNAVAILABLE",
                detail=(
                    "The checkpoint has no safely streamable restore layout; "
                    f"archive_reason={archive.block_reason}. No torch.load or "
                    "scientific step was attempted."
                ),
                evidence=evidence,
            )
            return BLOCKED_EXIT

        # A small mmap-eligible pickle is necessary but not sufficient.  This
        # standalone owner has no reviewed T14 <=50-step streaming driver and
        # therefore must still refuse to infer safety from container layout.
        _terminal(
            owner_root,
            owner_pid=identity.pid,
            owner_start_ticks=identity.start_ticks,
            status="BLOCKED",
            reason_code="BLOCKED_LOW_MEMORY_CANARY_UNAVAILABLE",
            detail=(
                "Archive layout is mmap-eligible, but no hash-bound reviewed "
                "streaming restore/save/reload driver is present."
            ),
            evidence=evidence,
        )
        return BLOCKED_EXIT
    except Exception as exc:
        _terminal(
            owner_root,
            owner_pid=identity.pid,
            owner_start_ticks=identity.start_ticks,
            status="FAILED",
            reason_code="FAILED_T14_CHECKPOINT_AUDIT",
            detail=f"{type(exc).__name__}: {exc}",
            evidence={},
        )
        return ERROR_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
