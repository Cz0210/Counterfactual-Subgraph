"""Bounded staging of the static CM-CReM DB in a real Slurm job directory.

No directory is guessed, no persistent DB is used as a fallback, and no source
is modified. Returned BLOCKED records are infrastructure failures, never zeros.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import time
from typing import Any

from src.baselines.cm_crem_runtime import atomic_json, digest, utc_now

HPC_SCOPE = Path("/share/home/u20526/czx")
OFFICIAL_DATABASE_URL = "https://www.qsar4u.com/files/cremdb/chembl22_sa2.db.gz"
LOCAL_FILESYSTEMS = {"ext2", "ext3", "ext4", "xfs", "btrfs", "zfs"}
SCHEMA = "cm_crem_job_local_static_database_v1"
_SQLITE_HEADER = b"SQLite format 3\x00"


class _Blocked(RuntimeError):
    def __init__(self, reason: str, **details: Any) -> None:
        super().__init__(reason)
        self.reason, self.details = reason, details


def _snapshot(path: Path) -> dict[str, int]:
    s = path.lstat()
    return {"device": s.st_dev, "inode": s.st_ino, "size": s.st_size,
            "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns,
            "uid": s.st_uid, "mode": stat.S_IMODE(s.st_mode)}


def _regular(path: Path, role: str) -> dict[str, int]:
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid():
        raise _Blocked("NONREGULAR_OR_NOT_OWNED_FILE", role=role, path=str(path))
    return _snapshot(path)


def _no_symlinks(path: Path) -> None:
    if not path.is_absolute():
        raise _Blocked("PATH_NOT_ABSOLUTE", path=str(path))
    for part in (path, *path.parents):
        if part.is_symlink():
            raise _Blocked("SYMLINK_PATH_REJECTED", path=str(part))


def _filesystem_identity(path: Path) -> dict[str, str]:
    """Read actual Linux mount metadata; never infer locality from /tmp name."""
    mounts = Path("/proc/self/mountinfo")
    if not mounts.is_file():
        raise _Blocked("LOCAL_MOUNT_METADATA_UNAVAILABLE")
    found = []
    for line in mounts.read_text().splitlines():
        before, after = line.split(" - ", 1)
        fields, tail = before.split(), after.split()
        mount = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4]))
        if path == mount or path.is_relative_to(mount):
            found.append((len(mount.parts), {"mount": str(mount), "source": tail[1],
                                            "filesystem_type": tail[0], "device": fields[2]}))
    if not found:
        raise _Blocked("SCRATCH_MOUNT_UNRESOLVED", path=str(path))
    found.sort(key=lambda item: item[0])
    result = found[-1][1]
    if result["filesystem_type"] not in LOCAL_FILESYSTEMS:
        raise _Blocked("NOT_VERIFIED_NODE_LOCAL_DISK", **result)
    return result


def _job_scratch() -> tuple[Path, dict[str, Any]]:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", job_id):
        raise _Blocked("MISSING_REAL_SLURM_JOB_ID")
    env_name = "SLURM_TMPDIR" if os.environ.get("SLURM_TMPDIR") else "TMPDIR"
    raw = os.environ.get(env_name)
    if not raw:
        raise _Blocked("NO_SCHEDULER_SCRATCH_ENVIRONMENT")
    path = Path(raw)
    _no_symlinks(path)
    if not path.is_dir():
        raise _Blocked("SCHEDULER_SCRATCH_NOT_EXISTING_DIRECTORY", path=str(path))
    if (path == HPC_SCOPE or path.is_relative_to(HPC_SCOPE) or
            any(path == p or path.is_relative_to(p) for p in
                (Path("/autodl-fs"), Path("/root/autodl-tmp"), Path("/dev/shm")))):
        raise _Blocked("PERSISTENT_AUTODL_OR_TMPFS_NOT_JOB_LOCAL_SCRATCH", path=str(path))
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o022:
        raise _Blocked("SCRATCH_NOT_EXCLUSIVELY_USER_WRITABLE", path=str(path))
    # A generic TMPDIR=/tmp (or a user-wide scratch directory) cannot establish
    # job ownership. If the site uses another naming contract, block for its
    # explicit audit rather than manufacture a job path ourselves.
    if not re.search(r"(?<![0-9])" + re.escape(job_id) + r"(?![0-9])", str(path)):
        raise _Blocked("JOB_PRIVATE_SCRATCH_BINDING_UNPROVEN", path=str(path), job_id=job_id)
    fs = _filesystem_identity(path)
    return path, {"job_id": job_id, "hostname": socket.gethostname(),
                  "environment_variable": env_name, "root": str(path),
                  "root_device": info.st_dev, "root_inode": info.st_ino,
                  "root_uid": info.st_uid, "filesystem": fs}


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stage_static_database(source_path: str | Path, receipt_path: str | Path,
                          *, reserve_bytes: int,
                          expected_url: str = OFFICIAL_DATABASE_URL) -> dict[str, Any]:
    """Stage once per actual Slurm job; reuse by immutable receipt/stat binding.

    Call once before launching generation workers, then pass `database_path` to
    all workers. A second call in the same job verifies cheap source/target stat
    and receipt identities, not another full DB hash. If a concurrent first
    stager or an incomplete attempt exists, return an explicit blocked state;
    this function never deletes/restarts that writer or falls back to source.
    """
    try:
        return _stage(source_path, receipt_path, reserve_bytes=reserve_bytes,
                      expected_url=expected_url)
    except _Blocked as exc:
        return {"schema": SCHEMA, "status": "BLOCKED_LOCAL_DATABASE_STAGING",
                "reason": exc.reason, "details": exc.details, "created_at": utc_now(),
                "persistent_database_fallback": False}
    except (OSError, json.JSONDecodeError) as exc:
        return {"schema": SCHEMA, "status": "BLOCKED_LOCAL_DATABASE_STAGING",
                "reason": "STORAGE_IO_OR_RECEIPT_ERROR", "error": str(exc),
                "errno": getattr(exc, "errno", None), "created_at": utc_now(),
                "persistent_database_fallback": False}


def _stage(source_path: str | Path, receipt_path: str | Path,
           *, reserve_bytes: int, expected_url: str) -> dict[str, Any]:
    if type(reserve_bytes) is not int or reserve_bytes < 1:
        raise _Blocked("POSITIVE_EXPLICIT_CAPACITY_RESERVE_REQUIRED")
    scratch, job = _job_scratch()
    source, authority = Path(source_path), Path(receipt_path)
    for path in (source, authority):
        _no_symlinks(path)
        if not path.is_relative_to(HPC_SCOPE):
            raise _Blocked("SOURCE_OUTSIDE_AUTHORIZED_HPC_SCOPE", path=str(path))
    source_before = _regular(source, "static_database")
    _regular(authority, "static_copy_receipt")
    if source_before["size"] < len(_SQLITE_HEADER):
        raise _Blocked("EMPTY_OR_TRUNCATED_STATIC_DATABASE")
    if source_before["mode"] & 0o022:
        raise _Blocked("SOURCE_DATABASE_GROUP_OR_WORLD_WRITABLE")
    # Static author DB only. Never copy an active SQLite/WAL dataset.
    companions = [str(Path(str(source) + suffix)) for suffix in ("-wal", "-shm", "-journal")
                  if Path(str(source) + suffix).exists()]
    if companions:
        raise _Blocked("DATABASE_NOT_SEALED_STATIC_COPY", companions=companions)
    raw_receipt = authority.read_bytes()
    receipt_sha = hashlib.sha256(raw_receipt).hexdigest()
    receipt = json.loads(raw_receipt)
    expected_sha = receipt.get("uncompressed_sha256")
    if (receipt.get("status") != "VERIFIED_STATIC_COPY" or receipt.get("url") != expected_url or
            not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)):
        raise _Blocked("STATIC_COPY_RECEIPT_INVALID")
    expected_bytes = receipt.get("uncompressed_bytes")
    if expected_bytes is not None and expected_bytes != source_before["size"]:
        raise _Blocked("STATIC_COPY_SIZE_CONFLICT")
    stage = scratch / ("cm-crem-static-db-" + expected_sha[:20])
    local, manifest = stage / "chembl22_sa2.db", stage / "staging.json"
    binding = {"source_path": str(source), "source_receipt_path": str(authority),
               "source_receipt_sha256": receipt_sha, "source_stat": source_before,
               "expected_sha256": expected_sha, "job_scratch": job}
    if stage.exists():
        _no_symlinks(stage)
        s = stage.stat()
        if not stage.is_dir() or s.st_uid != os.getuid() or stat.S_IMODE(s.st_mode) != 0o700:
            raise _Blocked("EXISTING_STAGE_IDENTITY_CONFLICT", path=str(stage))
        if not manifest.is_file():
            raise _Blocked("STAGING_IN_PROGRESS_OR_INCOMPLETE", path=str(stage))
        existing = json.loads(manifest.read_text())
        if existing.get("binding") != binding or existing.get("status") != "LOCAL_DATABASE_READY":
            raise _Blocked("EXISTING_STAGE_BINDING_CONFLICT", path=str(stage))
        current = _regular(local, "staged_database")
        if current != existing.get("local_stat") or current["mode"] != 0o444:
            raise _Blocked("STAGED_DATABASE_CHANGED", path=str(local))
        return {**existing, "reused": True, "full_database_hashes_this_call": 0}
    vfs = os.statvfs(scratch)
    available = vfs.f_bavail * vfs.f_frsize
    required = source_before["size"] + reserve_bytes + 65536
    if available < required:
        raise _Blocked("LOCAL_CAPACITY_RESERVE_NOT_MET", available_bytes=available,
                       required_bytes=required, reserve_bytes=reserve_bytes)
    if 0 <= vfs.f_favail < 16:
        raise _Blocked("LOCAL_FILE_SLOTS_INSUFFICIENT", available_file_slots=vfs.f_favail)
    try:
        stage.mkdir(mode=0o700)
    except FileExistsError:
        raise _Blocked("STAGING_IN_PROGRESS_OR_INCOMPLETE", path=str(stage))
    started = time.monotonic()
    temporary = stage / "chembl22_sa2.db.tmp"
    copied_hash, copied_bytes = hashlib.sha256(), 0
    read_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        fd_stat = os.fstat(read_fd)
        if (fd_stat.st_dev, fd_stat.st_ino, fd_stat.st_size) != (
                source_before["device"], source_before["inode"], source_before["size"]):
            raise _Blocked("SOURCE_CHANGED_BEFORE_COPY")
        write_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(write_fd, "wb") as target:
            with os.fdopen(os.dup(read_fd), "rb") as origin:
                first = True
                for block in iter(lambda: origin.read(1024 * 1024), b""):
                    if first and not block.startswith(_SQLITE_HEADER):
                        raise _Blocked("SOURCE_NOT_SQLITE_DATABASE")
                    first = False
                    target.write(block)
                    copied_hash.update(block)
                    copied_bytes += len(block)
            target.flush()
            os.fsync(target.fileno())
            os.fchmod(target.fileno(), 0o444)
            os.fsync(target.fileno())
        if _snapshot(source) != source_before:
            raise _Blocked("SOURCE_CHANGED_DURING_COPY", preserved_partial=str(temporary))
        if copied_bytes != source_before["size"] or copied_hash.hexdigest() != expected_sha:
            raise _Blocked("SOURCE_CONTENT_SHA_CONFLICT", preserved_partial=str(temporary))
        if _file_digest(temporary) != expected_sha:
            raise _Blocked("STAGED_CONTENT_SHA_CONFLICT", preserved_partial=str(temporary))
        if _snapshot(source) != source_before:
            raise _Blocked("SOURCE_CHANGED_DURING_TARGET_VERIFICATION", preserved_partial=str(temporary))
        os.rename(temporary, local)
        parent_fd = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        result = {"schema": SCHEMA, "status": "LOCAL_DATABASE_READY", "binding": binding,
                  "binding_sha256": digest(binding), "database_path": str(local),
                  "local_stat": _snapshot(local), "sha256": expected_sha,
                  "copied_bytes": copied_bytes, "copy_seconds": time.monotonic() - started,
                  "reserve_bytes": reserve_bytes, "available_bytes_before": available,
                  "full_database_hashes_this_call": 2, "source_unchanged": True,
                  "reused": False, "created_at": utc_now(), "manifest_path": str(manifest),
                  "persistent_database_fallback": False}
        atomic_json(manifest, result, immutable=True)
        return result
    finally:
        os.close(read_fd)
