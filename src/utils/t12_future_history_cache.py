"""Future-only local copies of closed T12 journal/first-embedding segments.

This is not a production resume/promotion gate. It never touches a running
reader, rewrites a source snapshot, or changes the authoritative record codec.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from typing import Any

from src.utils.main_ready_task_specs import stable_sha256
from src.utils.t8_hpc_t13_successor_v1 import atomic_json_no_replace
from src.utils.tastemolnet_t12_diagnostic_reconcile_v1 import validate_no_live_writer


BUFFER_BYTES = 1024 * 1024
SCHEMA = "t12_future_immutable_history_cache_v1"


def _identity(path: Path) -> dict[str, int]:
    if not path.is_absolute() or path.resolve(strict=True) != path or path.is_symlink():
        raise ValueError("T12_CACHE_PHYSICAL_FILE_REQUIRED")
    s = path.stat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError("T12_CACHE_REGULAR_FILE_REQUIRED")
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, mode=s.st_mode)


def _segments(snapshot: dict[str, Any]):
    root = Path(snapshot["history_root"])
    for row in snapshot["segments"]:
        yield root, row, Path(row["segment_file"])
    first = snapshot.get("first_seen_embedding_store")
    if first is not None:
        first_root = Path(first["store_root"])
        if first_root != root / "first-seen-embeddings":
            raise ValueError("T12_CACHE_FIRST_EMBEDDING_ROOT_DRIFT")
        for row in first["segments"]:
            yield first_root, row, Path("first-seen-embeddings") / row["segment_file"]


@dataclass(frozen=True)
class T12HistoryReadCache:
    """Metadata-bound read routing; the original codec still verifies records."""

    snapshot_sha256: str
    entries: dict[str, dict[str, Any]]

    def require_snapshot(self, snapshot: dict[str, Any]) -> None:
        if stable_sha256(snapshot) != self.snapshot_sha256:
            raise ValueError("T12_CACHE_SNAPSHOT_CHANGED")

    def open(self, source: Path):
        row = self.entries.get(str(source))
        if row is None:
            raise ValueError("T12_CACHE_UNBOUND_SOURCE")
        local = Path(row["cache_path"])
        if _identity(local) != row["cache_identity"]:
            raise ValueError("T12_CACHE_LOCAL_IDENTITY_CHANGED")
        # No full file hash on each hot reopen. The unchanged codec validates
        # chain/record semantics while rebuilding its disposable index.
        return local.open("rb", buffering=BUFFER_BYTES)


def load_cache(cache_root: Path, *, expected_manifest_sha256: str) -> T12HistoryReadCache:
    manifest_path = cache_root / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if stable_sha256(manifest) != expected_manifest_sha256 or manifest.get("schema_version") != SCHEMA:
        raise ValueError("T12_CACHE_MANIFEST_CHANGED")
    if manifest.get("future_only") is not True or manifest.get("source_modified") is not False:
        raise ValueError("T12_CACHE_FUTURE_ONLY_REQUIRED")
    entries = {row["source"]: row for row in manifest["files"]}
    if len(entries) != len(manifest["files"]):
        raise ValueError("T12_CACHE_DUPLICATE_SOURCE")
    for row in entries.values():
        local = Path(row["cache_path"])
        if not local.is_relative_to(cache_root) or _identity(local) != row["cache_identity"]:
            raise ValueError("T12_CACHE_LOCAL_IDENTITY_CHANGED")
        if local.stat().st_mode & 0o222:
            raise ValueError("T12_CACHE_FILE_NOT_READONLY")
    return T12HistoryReadCache(manifest["snapshot_sha256"], entries)


def stage_closed_snapshot(
    snapshot: dict[str, Any], *, expected_snapshot_sha256: str,
    cache_root: Path, producer_pid: int, producer_start_ticks: int,
    proc_root: Path = Path("/proc"), min_free_bytes: int,
    min_free_inodes: int,
) -> dict[str, Any]:
    """Copy each sealed file once, hash during copy and verify local bytes once.

    Caller supplies the snapshot identity from its committed checkpoint/receipt.
    A live/reused former producer, writer, incomplete segment, uncommitted tail,
    or inadequate local reserve blocks before creating a cache directory.
    """
    if stable_sha256(snapshot) != expected_snapshot_sha256:
        raise ValueError("T12_CACHE_SNAPSHOT_BINDING_REQUIRED")
    if min_free_bytes < 0 or min_free_inodes < 0:
        raise ValueError("T12_CACHE_RESOURCE_RESERVE_INVALID")
    if cache_root.exists() or cache_root.is_symlink():
        raise ValueError("T12_CACHE_FRESH_ROOT_REQUIRED")
    parent = cache_root.parent
    if not parent.is_absolute() or parent.resolve(strict=True) != parent:
        raise ValueError("T12_CACHE_PHYSICAL_PARENT_REQUIRED")
    source_root = Path(snapshot["history_root"])
    if cache_root.is_relative_to(source_root) or source_root.is_relative_to(cache_root):
        raise ValueError("T12_CACHE_MUST_NOT_MODIFY_SOURCE_ROOT")
    writer_before = validate_no_live_writer(
        output_root=source_root, expected_dead_pid=producer_pid,
        expected_start_ticks=producer_start_ticks, proc_root=proc_root)
    inventory = []
    for root, segment, relative in _segments(snapshot):
        name = segment["segment_file"]
        if type(name) is not str or Path(name).name != name:
            raise ValueError("T12_CACHE_SEGMENT_PATH_INVALID")
        source = root / name
        identity = _identity(source)
        if identity["bytes"] != segment["committed_bytes"]:
            raise ValueError("T12_CACHE_SOURCE_NOT_FULLY_SEALED")
        inventory.append((source, relative, segment, identity))
    required = sum(row[3]["bytes"] for row in inventory)
    fs = os.statvfs(parent)
    available_bytes = fs.f_bavail * fs.f_frsize
    if available_bytes - required < min_free_bytes or fs.f_favail - (len(inventory) + 3) < min_free_inodes:
        raise ValueError("T12_CACHE_LOCAL_RESOURCE_ADMISSION_FAILED")
    started = time.monotonic()
    cache_root.mkdir(mode=0o700)
    files = []
    for source, relative, segment, source_identity in inventory:
        target = cache_root / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        part = target.with_name(target.name + ".partial")
        digest = hashlib.sha256()
        copied = 0
        with source.open("rb", buffering=BUFFER_BYTES) as reader, part.open("xb", buffering=BUFFER_BYTES) as writer:
            for block in iter(lambda: reader.read(BUFFER_BYTES), b""):
                digest.update(block)
                copied += len(block)
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        expected = segment["committed_prefix_sha256"]
        if copied != source_identity["bytes"] or digest.hexdigest() != expected or _identity(source) != source_identity:
            raise ValueError("T12_CACHE_SOURCE_COPY_CHANGED")
        verified = hashlib.sha256()
        with part.open("rb", buffering=BUFFER_BYTES) as reader:
            for block in iter(lambda: reader.read(BUFFER_BYTES), b""):
                verified.update(block)
        if verified.hexdigest() != expected:
            raise ValueError("T12_CACHE_LOCAL_COPY_CHANGED")
        part.chmod(0o444)
        os.rename(part, target)  # target is inside our fresh private directory.
        fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        files.append(dict(source=str(source), source_identity=source_identity,
                          cache_path=str(target), cache_identity=_identity(target),
                          bytes=copied, sha256=expected, source_full_reads=1,
                          local_verification_full_reads=1))
    writer_after = validate_no_live_writer(
        output_root=source_root, expected_dead_pid=producer_pid,
        expected_start_ticks=producer_start_ticks, proc_root=proc_root)
    if any(_identity(Path(row["source"])) != row["source_identity"] for row in files):
        raise ValueError("T12_CACHE_SOURCE_CHANGED_BEFORE_SEAL")
    manifest = dict(schema_version=SCHEMA, future_only=True, source_modified=False,
        active_reader_replaced=False, diagnostic_checkpoint_promoted=False,
        snapshot_sha256=expected_snapshot_sha256, files=files,
        source_writer_before=writer_before, source_writer_after=writer_after,
        buffer_bytes=BUFFER_BYTES, copied_bytes=required,
        initial_copy_and_verify_seconds=time.monotonic()-started,
        kernel_page_cache_state="UNCONTROLLED_NO_DROP_CACHES",
        subsequent_local_decode_timing_seconds=None, min_free_bytes_retained=min_free_bytes,
        min_free_inodes_retained=min_free_inodes)
    atomic_json_no_replace(cache_root / "cache_manifest.json", manifest)
    return {"cache_root": str(cache_root), "cache_manifest_sha256": stable_sha256(manifest),
            "state": "FUTURE_READ_CACHE_SEALED_NOT_SCIENCE_PASS", "manifest": manifest}
