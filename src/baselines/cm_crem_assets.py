"""Bounded staging of the static CM-CReM DB in a real Slurm job directory.

The legacy scheduler directory contract stays strict. An explicitly requested
project mktemp path may instead be created under a verified node-local base.
No persistent DB is used as a fallback. BLOCKED is infrastructure, never zero.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import tempfile
import time
from typing import Any
from urllib.parse import urlsplit

from src.baselines.cm_crem_runtime import atomic_json, digest, utc_now

HPC_SCOPE = Path("/share/home/u20526/czx")
OFFICIAL_DATABASE_URL = "https://www.qsar4u.com/files/cremdb/chembl22_sa2.db.gz"
LOCAL_FILESYSTEMS = {"ext2", "ext3", "ext4", "xfs", "btrfs", "zfs"}
SCHEMA = "cm_crem_job_local_static_database_v1"
_SQLITE_HEADER = b"SQLite format 3\x00"
SCRATCH_SCHEMA = "cm_crem_prepared_job_scratch_v1"
ZENODO_DOI = "10.5281/zenodo.16909329"
ZENODO_RECORD_URL = "https://zenodo.org/records/16909329"
ZENODO_FILENAME = "chembl22_sa2.db.gz"
ZENODO_COMPRESSED_BYTES = 350212897
ZENODO_COMPRESSED_MD5 = "91ce6b3d61270e927910162eeb63db43"
ZENODO_COMPRESSED_SHA256 = "6fe7f9534ae705fc508fa9be1d0c6a1baac988d5c1e67c8314e6524f4545c8fb"
_PROJECT_SCRATCH_FALLBACK = Path("/tmp")


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


def validate_database_source(spec: dict[str, Any], receipt: dict[str, Any],
                             require_compatibility: bool = True,
                             require_content: bool = True) -> dict[str, Any]:
    """Validate small static-asset receipts, without changing scientific pins.

    The author-authorized Zenodo source is an explicit provenance overlay, not
    a claim that bytes were compared with the unavailable historical URL.
    Compatibility means real read-only radius1 queries and a native public
    fixture replacement, not filename/license metadata or a bare PASS string.
    No database is opened or hashed here. Invalid binding raises ValueError.
    """
    def require(condition: bool, reason: str) -> None:
        if not condition:
            raise ValueError(reason)

    sha = receipt.get("uncompressed_sha256")
    require(not require_compatibility or require_content, "DATABASE_COMPATIBILITY_REQUIRES_CONTENT_BINDING")
    if require_content:
        require(receipt.get("status") == "VERIFIED_STATIC_COPY", "DATABASE_STATIC_COPY_NOT_VERIFIED")
        require(isinstance(sha, str) and bool(re.fullmatch(r"[0-9a-f]{64}", sha)), "DATABASE_CONTENT_SHA_MISSING")
        require(type(receipt.get("uncompressed_bytes")) is int and receipt["uncompressed_bytes"] > 16,
                "DATABASE_CONTENT_SIZE_MISSING")
    else:
        require(receipt.get("status") in {"VERIFIED_COMPRESSED_COPY", "VERIFIED_STATIC_COPY"},
                "DATABASE_COMPRESSED_COPY_NOT_VERIFIED")
    upstream_url = spec.get("upstream", {}).get("database", {}).get("url")
    require(upstream_url == OFFICIAL_DATABASE_URL, "ORIGINAL_REQUESTED_DATABASE_URL_CHANGED")
    overlay = spec.get("asset_source_overlay")
    source_mode = "HISTORICAL_AUTHOR_URL"
    overlay_sha = None
    url = upstream_url
    if overlay is not None:
        require(isinstance(overlay, dict), "DATABASE_SOURCE_OVERLAY_NOT_OBJECT")
        pins = {"record_doi": ZENODO_DOI, "record_url": ZENODO_RECORD_URL,
                "filename": ZENODO_FILENAME, "published_md5": ZENODO_COMPRESSED_MD5,
                "compressed_bytes": ZENODO_COMPRESSED_BYTES,
                "compressed_sha256": ZENODO_COMPRESSED_SHA256,
                "historical_url": OFFICIAL_DATABASE_URL, "old_file_bytes_compared": False}
        for key, expected in pins.items():
            require(key in overlay and overlay[key] == expected, "DATABASE_SOURCE_OVERLAY_PIN_CONFLICT:" + key)
        url = overlay.get("download_url")
        require(isinstance(url, str), "DATABASE_SOURCE_DOWNLOAD_URL_MISSING")
        parsed = urlsplit(url)
        require(parsed.scheme == "https" and parsed.netloc == "zenodo.org" and
                parsed.path == "/records/16909329/files/chembl22_sa2.db.gz" and
                parsed.query in ("", "download=1") and not parsed.fragment,
                "DATABASE_SOURCE_DOWNLOAD_URL_UNAUTHORIZED")
        license_info = overlay.get("license", {})
        approved_rights = {"https://creativecommons.org/licenses/by/4.0/",
                           "https://creativecommons.org/licenses/by/4.0/legalcode"}
        require(license_info.get("identifier") == "CC-BY-4.0" and
                license_info.get("rights_uri") in approved_rights and
                license_info.get("scope") == "database", "DATABASE_LICENSE_BINDING_MISSING")
        metadata_path = Path(license_info.get("metadata_path", ""))
        metadata_sha = license_info.get("metadata_sha256")
        require(metadata_path.is_absolute() and isinstance(metadata_sha, str) and
                bool(re.fullmatch(r"[0-9a-f]{64}", metadata_sha)), "DATABASE_LICENSE_METADATA_BINDING_MISSING")
        _no_symlinks(metadata_path)
        require(metadata_path.is_file() and metadata_path.stat().st_size <= 4 * 1024 ** 2,
                "DATABASE_LICENSE_METADATA_NOT_SMALL_REGULAR_FILE")
        metadata_bytes = metadata_path.read_bytes()
        require(hashlib.sha256(metadata_bytes).hexdigest() == metadata_sha, "DATABASE_LICENSE_METADATA_SHA_CONFLICT")
        # DataCite record may be saved as its response envelope or attributes.
        metadata = json.loads(metadata_bytes)
        attributes = metadata.get("data", {}).get("attributes", metadata)
        creators = attributes.get("creators", [])
        creator_names = [str(c.get("name", c.get("creatorName", ""))).lower() for c in creators]
        require(str(attributes.get("doi", "")).lower() == ZENODO_DOI and
                any("polishchuk" in name and "pavel" in name for name in creator_names),
                "DATABASE_AUTHOR_RECORD_IDENTITY_CONFLICT")
        rights = attributes.get("rightsList", [])
        require(any(r.get("rightsUri") in approved_rights and
                    str(r.get("rightsIdentifier", "")).lower() == "cc-by-4.0" for r in rights),
                "DATABASE_AUTHOR_LICENSE_NOT_CONFIRMED")
        for field, value in (("compressed_bytes", ZENODO_COMPRESSED_BYTES),
                             ("compressed_sha256", ZENODO_COMPRESSED_SHA256)):
            require(receipt.get(field) == value, "DATABASE_STATIC_COPY_COMPRESSED_PIN_CONFLICT:" + field)
        require(receipt.get("compressed_md5") == ZENODO_COMPRESSED_MD5,
                "DATABASE_STATIC_COPY_COMPRESSED_PIN_CONFLICT:compressed_md5")
        overlay_sha = digest(overlay)
        source_mode = "USER_AUTHORIZED_AUTHOR_ZENODO_SOURCE"
    require(receipt.get("url") == url, "DATABASE_STATIC_COPY_SOURCE_URL_CONFLICT")
    compatibility = receipt.get("compatibility")
    if require_compatibility:
        require(isinstance(compatibility, dict), "DATABASE_NATIVE_COMPATIBILITY_MISSING")
        require(receipt.get("compatibility_database_sha256") == sha, "DATABASE_COMPATIBILITY_CONTENT_BINDING_CONFLICT")
        expected_versions = {"crem": "0.2.14", "rdkit": "2023.9.6", "numpy": "1.26.4", "python": "3.11.5"}
        require(compatibility.get("schema") == "cm_crem_static_database_compatibility_v1" and
                compatibility.get("status") == "STATIC_DATABASE_COMPATIBILITY_PASS" and
                compatibility.get("versions") == expected_versions and compatibility.get("radius") == 1,
                "DATABASE_NATIVE_COMPATIBILITY_CONTRACT_CONFLICT")
        require(compatibility.get("connection_mode") == "mode=ro&immutable=1" and
                compatibility.get("query_only") is True and
                compatibility.get("journal_mode_changed") is False and
                compatibility.get("durability_changed") is False and
                compatibility.get("unchanged_static_source") is True and
                compatibility.get("stat_before") == compatibility.get("stat_after") and
                isinstance(compatibility.get("stat_before"), dict) and
                compatibility["stat_before"].get("bytes") == receipt["uncompressed_bytes"] and
                compatibility.get("sidecars_before") == compatibility.get("sidecars_after") == [],
                "DATABASE_COMPATIBILITY_STATIC_READ_CONTRACT_CONFLICT")
        required_columns = {"env", "freq", "core_num_atoms", "core_smi", "core_sma"}
        require(required_columns.issubset({c.get("name") for c in compatibility.get("columns", [])}) and
                compatibility.get("first_row_fields_valid") is True and
                compatibility.get("radius1_rowid_supported") is True,
                "DATABASE_NATIVE_RADIUS1_SCHEMA_EVIDENCE_MISSING")
        products = compatibility.get("public_fixture_products", [])
        require(compatibility.get("fixture_smiles") == "CCO" and
                compatibility.get("fixture_settings") == {"radius": 1, "min_inc": 0, "max_inc": 0,
                    "max_replacements": 4, "replace_ids": [0], "ncores": 1, "symmetry_fixes": True} and
                compatibility.get("fixture_mutate_calls") == 1 and
                type(compatibility.get("fixture_select_count")) is int and compatibility["fixture_select_count"] >= 1 and
                type(compatibility.get("actual_select_count")) is int and compatibility["actual_select_count"] >= compatibility["fixture_select_count"] and
                isinstance(products, list) and 1 <= len(products) <= 4 and
                all(isinstance(p, str) and p for p in products) and len(set(products)) == len(products) and
                compatibility.get("public_fixture_product_count") == len(products) and
                compatibility.get("experiment_generation_performed") is False and compatibility.get("oracle_calls") == 0,
                "DATABASE_NATIVE_QUERY_OR_REPLACEMENT_EVIDENCE_MISSING")
    return {"actual_source_url": url, "expected_url": url, "source_mode": source_mode,
            "source_overlay_sha256": overlay_sha, "original_requested_url": upstream_url,
            "uncompressed_sha256": sha if require_content else None,
            "content_validated": require_content, "compatibility_required": require_compatibility,
            "compatibility_validated": require_compatibility,
            "old_file_bytes_compared": False if overlay is not None else None}


def _compute_identity() -> dict[str, Any]:
    """Require batch-node identity too, not a job number copied onto a login."""
    job = os.environ.get("SLURM_JOB_ID", "")
    node = os.environ.get("SLURMD_NODENAME", "")
    host = socket.gethostname()
    if not re.fullmatch(r"[0-9]+", job):
        raise _Blocked("MISSING_REAL_SLURM_JOB_ID")
    if not node or node.split(".")[0] != host.split(".")[0]:
        raise _Blocked("COMPUTE_NODE_IDENTITY_UNPROVEN", hostname=host, slurmd_nodename=node)
    return {"job_id": job, "hostname": host, "slurmd_nodename": node,
            "raw_environment": {k: os.environ.get(k) for k in
                ("SLURM_JOB_ID", "SLURMD_NODENAME", "SLURM_STEP_ID", "SLURM_TMPDIR", "TMPDIR")}}


def _cm_run_root(run_root: str | Path) -> Path:
    root = Path(run_root)
    _no_symlinks(root)
    scopes = [HPC_SCOPE / ("counterfactual-subgraph-hpc-runtime/baselines/"+name)
              for name in ("cm_crem_global_v1", "cm_crem_global_v2")]
    if not any(root != scope and root.is_relative_to(scope) for scope in scopes) or not root.is_dir():
        raise _Blocked("CM_RUN_ROOT_NOT_EXISTING_AUTHORIZED_CHILD", path=str(root))
    info = root.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o022:
        raise _Blocked("CM_RUN_ROOT_OWNERSHIP_UNSAFE", path=str(root))
    return root


def _scratch_base(base: Path) -> dict[str, Any]:
    _no_symlinks(base)
    if not base.is_dir():
        raise _Blocked("LOCAL_SCRATCH_BASE_MISSING", path=str(base))
    forbidden = (HPC_SCOPE, Path("/share"), Path("/ssdfs"), Path("/autodl-fs"),
                 Path("/root/autodl-tmp"), Path("/dev/shm"))
    if any(base == p or base.is_relative_to(p) for p in forbidden):
        raise _Blocked("PERSISTENT_AUTODL_OR_TMPFS_NOT_JOB_LOCAL_SCRATCH", path=str(base))
    info = base.stat()
    mode = stat.S_IMODE(info.st_mode)
    private = info.st_uid == os.getuid() and not mode & 0o022
    trusted_sticky = info.st_uid == 0 and bool(info.st_mode & stat.S_ISVTX)
    if not (private or trusted_sticky) or not os.access(base, os.W_OK | os.X_OK):
        raise _Blocked("LOCAL_SCRATCH_BASE_OWNERSHIP_UNSAFE", path=str(base), uid=info.st_uid, mode=mode)
    return {"path": str(base), "device": info.st_dev, "inode": info.st_ino,
            "uid": info.st_uid, "mode": mode, "filesystem": _filesystem_identity(base)}


def _prepared_scratch(receipt: dict[str, Any] | str | Path) -> tuple[Path, dict[str, Any]]:
    supplied = receipt if isinstance(receipt, dict) else None
    receipt_path = Path(receipt["receipt_path"] if supplied is not None else receipt)
    _no_symlinks(receipt_path)
    _regular(receipt_path, "job_scratch_receipt")
    data = json.loads(receipt_path.read_text())
    if supplied is not None and supplied != data:
        raise _Blocked("SCRATCH_RECEIPT_MEMORY_DISK_CONFLICT")
    body = {k: v for k, v in data.items() if k != "receipt_sha256"}
    if (data.get("schema") != SCRATCH_SCHEMA or data.get("status") != "JOB_SCRATCH_READY" or
            data.get("receipt_sha256") != digest(body) or data.get("receipt_path") != str(receipt_path)):
        raise _Blocked("PREPARED_SCRATCH_RECEIPT_INVALID")
    runtime = _compute_identity()
    if any(data.get(k) != v for k, v in runtime.items()):
        raise _Blocked("PREPARED_SCRATCH_JOB_OR_ENVIRONMENT_CONFLICT")
    run_root = _cm_run_root(data["run_root"])
    if receipt_path.parent != run_root / "scratch_receipts":
        raise _Blocked("PREPARED_SCRATCH_RECEIPT_OUTSIDE_RUN")
    base = Path(data["base"]["path"])
    if _scratch_base(base) != data["base"]:
        raise _Blocked("PREPARED_SCRATCH_BASE_CHANGED")
    path = Path(data["root"])
    _no_symlinks(path)
    if path.parent != base or not path.name.startswith("cm-crem-job-" + runtime["job_id"] + "-"):
        raise _Blocked("PREPARED_SCRATCH_PATH_BINDING_INVALID")
    info = path.stat()
    current = {"device": info.st_dev, "inode": info.st_ino,
               "uid": info.st_uid, "mode": stat.S_IMODE(info.st_mode)}
    if not path.is_dir() or current != data["root_identity"] or current["uid"] != os.getuid() or current["mode"] != 0o700:
        raise _Blocked("PREPARED_SCRATCH_ROOT_CHANGED")
    if _filesystem_identity(path) != data["base"]["filesystem"]:
        raise _Blocked("PREPARED_SCRATCH_MOUNT_CHANGED")
    return path, {"job_id": runtime["job_id"], "hostname": runtime["hostname"],
                  "environment_variable": data["base_selection"], "root": str(path),
                  "root_device": info.st_dev, "root_inode": info.st_ino, "root_uid": info.st_uid,
                  "filesystem": data["base"]["filesystem"], "prepared_receipt_path": str(receipt_path),
                  "prepared_receipt_sha256": data["receipt_sha256"], "raw_environment": data["raw_environment"],
                  "construction": "PROJECT_MKTEMP_UNDER_VERIFIED_LOCAL_BASE"}


def prepare_job_scratch(run_root: str | Path, *, required_bytes: int,
                        reserve_bytes: int = 1024 ** 3) -> dict[str, Any]:
    """Explicitly create one 0700 CM directory per real job, recording raw env.

    This does not manufacture SLURM_TMPDIR/TMPDIR. Selection is actual
    SLURM_TMPDIR, else actual TMPDIR, else the user-authorized /tmp convention;
    a populated but invalid earlier choice blocks, never silently falls back.
    Returned receipts can be passed directly to stage_static_database.
    """
    try:
        if any(type(n) is not int or n < 1 for n in (required_bytes, reserve_bytes)):
            raise _Blocked("POSITIVE_EXPLICIT_CAPACITY_REQUIREMENT_REQUIRED")
        runtime = _compute_identity()
        root = _cm_run_root(run_root)
        directory = root / "scratch_receipts"
        _no_symlinks(directory)
        directory.mkdir(mode=0o700, exist_ok=True)
        if directory.stat().st_uid != os.getuid() or stat.S_IMODE(directory.stat().st_mode) != 0o700:
            raise _Blocked("SCRATCH_RECEIPT_DIRECTORY_UNSAFE")
        receipt_path = directory / ("job-" + runtime["job_id"] + "-" + runtime["hostname"] + ".json")
        if receipt_path.exists():
            _prepared_scratch(receipt_path)
            return json.loads(receipt_path.read_text())
        selection = "SLURM_TMPDIR" if os.environ.get("SLURM_TMPDIR") else (
            "TMPDIR" if os.environ.get("TMPDIR") else "AUTHORIZED_PROJECT_TMP_FALLBACK")
        raw = os.environ.get(selection) if selection != "AUTHORIZED_PROJECT_TMP_FALLBACK" else str(_PROJECT_SCRATCH_FALLBACK)
        base = Path(raw)
        identity = _scratch_base(base)
        vfs = os.statvfs(base)
        available = vfs.f_bavail * vfs.f_frsize
        required = required_bytes + reserve_bytes + 65536
        if available < required:
            raise _Blocked("LOCAL_CAPACITY_RESERVE_NOT_MET", available_bytes=available, required_bytes=required)
        if 0 <= vfs.f_favail < 16:
            raise _Blocked("LOCAL_FILE_SLOTS_INSUFFICIENT", available_file_slots=vfs.f_favail)
        path = Path(tempfile.mkdtemp(prefix="cm-crem-job-" + runtime["job_id"] + "-", dir=base))
        path.chmod(0o700)
        info = path.stat()
        data = {"schema": SCRATCH_SCHEMA, "status": "JOB_SCRATCH_READY", **runtime,
                "run_root": str(root), "root": str(path), "receipt_path": str(receipt_path),
                "base_selection": selection, "base": identity,
                "root_identity": {"device": info.st_dev, "inode": info.st_ino,
                                  "uid": info.st_uid, "mode": stat.S_IMODE(info.st_mode)},
                "available_bytes_before": available, "required_bytes": required_bytes,
                "reserve_bytes": reserve_bytes, "available_file_slots_before": vfs.f_favail,
                "created_at": utc_now()}
        data["receipt_sha256"] = digest(data)
        atomic_json(receipt_path, data, immutable=True)
        _prepared_scratch(receipt_path)
        return data
    except _Blocked as exc:
        return {"schema": SCRATCH_SCHEMA, "status": "BLOCKED_LOCAL_DATABASE_STAGING",
                "reason": exc.reason, "details": exc.details, "created_at": utc_now(),
                "persistent_database_fallback": False}
    except (OSError, ValueError, KeyError) as exc:
        return {"schema": SCRATCH_SCHEMA, "status": "BLOCKED_LOCAL_DATABASE_STAGING",
                "reason": "SCRATCH_PREPARATION_IO_OR_RECEIPT_ERROR", "error": str(exc),
                "errno": getattr(exc, "errno", None), "created_at": utc_now(),
                "persistent_database_fallback": False}


def stage_static_database(source_path: str | Path, receipt_path: str | Path,
                          *, reserve_bytes: int,
                          expected_url: str = OFFICIAL_DATABASE_URL,
                          scratch_receipt: dict[str, Any] | str | Path | None = None) -> dict[str, Any]:
    """Stage once per actual Slurm job; reuse by immutable receipt/stat binding.

    Call once before launching generation workers, then pass `database_path` to
    all workers. A second call in the same job verifies cheap source/target stat
    and receipt identities, not another full DB hash. If a concurrent first
    stager or an incomplete attempt exists, return an explicit blocked state;
    this function never deletes/restarts that writer or falls back to source.
    """
    try:
        return _stage(source_path, receipt_path, reserve_bytes=reserve_bytes,
                      expected_url=expected_url, scratch_receipt=scratch_receipt)
    except _Blocked as exc:
        return {"schema": SCHEMA, "status": "BLOCKED_LOCAL_DATABASE_STAGING",
                "reason": exc.reason, "details": exc.details, "created_at": utc_now(),
                "persistent_database_fallback": False}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"schema": SCHEMA, "status": "BLOCKED_LOCAL_DATABASE_STAGING",
                "reason": "STORAGE_IO_OR_RECEIPT_ERROR", "error": str(exc),
                "errno": getattr(exc, "errno", None), "created_at": utc_now(),
                "persistent_database_fallback": False}


def _stage(source_path: str | Path, receipt_path: str | Path,
           *, reserve_bytes: int, expected_url: str,
           scratch_receipt: dict[str, Any] | str | Path | None = None) -> dict[str, Any]:
    if type(reserve_bytes) is not int or reserve_bytes < 1:
        raise _Blocked("POSITIVE_EXPLICIT_CAPACITY_RESERVE_REQUIRED")
    scratch, job = _job_scratch() if scratch_receipt is None else _prepared_scratch(scratch_receipt)
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
