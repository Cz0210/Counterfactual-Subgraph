"""Tiny transport fixtures only; never an official or usable science database."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.baselines import cm_crem_assets as assets


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    # Resolve macOS temporary-directory symlinks. Production does not relax
    # symlink checks, and real /proc mount metadata is not mocked by the helper.
    base = tmp_path.resolve()
    source_root = base / "persistent"
    source_root.mkdir()
    scratch = base / "site-slurm-job-12345"
    scratch.mkdir(mode=0o700)
    source = source_root / "chembl22_sa2.db"
    body = b"SQLite format 3\x00" + b"UNIT_TEST_BYTES_NOT_A_SCIENCE_DB" * 300
    source.write_bytes(body)
    source.chmod(0o444)
    receipt = source_root / "database_receipt.json"
    receipt.write_text(json.dumps({"status": "VERIFIED_STATIC_COPY", "url": assets.OFFICIAL_DATABASE_URL,
        "uncompressed_sha256": hashlib.sha256(body).hexdigest(), "uncompressed_bytes": len(body)}))
    monkeypatch.setattr(assets, "HPC_SCOPE", source_root)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_TMPDIR", str(scratch))
    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.setattr(assets, "_filesystem_identity", lambda path: {
        "mount": str(base), "source": "fixture-device", "filesystem_type": "ext4", "device": "1:2"})
    return source, receipt, scratch


def test_single_copy_readonly_verified_and_reused_without_rehash(sandbox, monkeypatch):
    source, receipt, scratch = sandbox
    before = source.read_bytes()
    first = assets.stage_static_database(source, receipt, reserve_bytes=1024)
    assert first["status"] == "LOCAL_DATABASE_READY"
    local = Path(first["database_path"])
    assert local.parent.parent == scratch
    assert local.read_bytes() == before and local.stat().st_mode & 0o777 == 0o444
    assert first["copied_bytes"] == len(before)
    assert first["source_unchanged"] is True and first["full_database_hashes_this_call"] == 2
    def no_rehash(path):
        raise AssertionError("Same job reuse must not rehash the DB")
    monkeypatch.setattr(assets, "_file_digest", no_rehash)
    second = assets.stage_static_database(source, receipt, reserve_bytes=1024)
    assert second["reused"] is True and second["full_database_hashes_this_call"] == 0
    assert second["database_path"] == str(local) and source.read_bytes() == before


def test_reuse_rejects_target_metadata_change(sandbox):
    source, receipt, _ = sandbox
    staged = assets.stage_static_database(source, receipt, reserve_bytes=1024)
    Path(staged["database_path"]).chmod(0o644)
    result = assets.stage_static_database(source, receipt, reserve_bytes=1024)
    assert result["reason"] == "STAGED_DATABASE_CHANGED"


@pytest.mark.parametrize("change,reason", [
    ("no_job", "MISSING_REAL_SLURM_JOB_ID"),
    ("no_root", "NO_SCHEDULER_SCRATCH_ENVIRONMENT"),
    ("generic_tmp", "JOB_PRIVATE_SCRATCH_BINDING_UNPROVEN"),
    ("relative", "PATH_NOT_ABSOLUTE"),
    ("writable_shared", "SCRATCH_NOT_EXCLUSIVELY_USER_WRITABLE"),
    ("wrong_job", "JOB_PRIVATE_SCRATCH_BINDING_UNPROVEN"),
])
def test_no_guessed_or_shared_scratch(sandbox, monkeypatch, change, reason):
    source, receipt, scratch = sandbox
    if change == "no_job":
        monkeypatch.delenv("SLURM_JOB_ID")
    elif change == "no_root":
        monkeypatch.delenv("SLURM_TMPDIR")
    elif change == "generic_tmp":
        generic = scratch.parent / "generic-temp"
        generic.mkdir(mode=0o700)
        monkeypatch.setenv("SLURM_TMPDIR", str(generic))
    elif change == "relative":
        monkeypatch.setenv("SLURM_TMPDIR", "slurm-12345")
    elif change == "writable_shared":
        scratch.chmod(0o777)
    else:
        monkeypatch.setenv("SLURM_JOB_ID", "1234")
    result = assets.stage_static_database(source, receipt, reserve_bytes=1024)
    assert result["status"] == "BLOCKED_LOCAL_DATABASE_STAGING"
    assert result["reason"] == reason
    assert result["persistent_database_fallback"] is False
    assert not list(scratch.glob("cm-crem-*"))


def test_symlink_scratch_rejected(sandbox, monkeypatch):
    source, receipt, scratch = sandbox
    link = scratch.parent / "symlink-12345"
    link.symlink_to(scratch)
    monkeypatch.setenv("SLURM_TMPDIR", str(link))
    assert assets.stage_static_database(source, receipt, reserve_bytes=1024)["reason"] == "SYMLINK_PATH_REJECTED"


def test_existing_active_database_companions_rejected_without_read(sandbox):
    source, receipt, scratch = sandbox
    Path(str(source) + "-wal").write_bytes(b"preserve")
    result = assets.stage_static_database(source, receipt, reserve_bytes=1024)
    assert result["reason"] == "DATABASE_NOT_SEALED_STATIC_COPY"
    assert Path(str(source) + "-wal").read_bytes() == b"preserve"
    assert not list(scratch.iterdir())


def test_capacity_reserve_not_lowered(sandbox, monkeypatch):
    source, receipt, scratch = sandbox
    monkeypatch.setattr(assets.os, "statvfs", lambda p: SimpleNamespace(f_bavail=8, f_frsize=1024, f_favail=100))
    result = assets.stage_static_database(source, receipt, reserve_bytes=1024)
    assert result["reason"] == "LOCAL_CAPACITY_RESERVE_NOT_MET"
    assert not list(scratch.iterdir())


def test_wrong_content_preserves_partial_and_never_marks_ready(sandbox):
    source, receipt, scratch = sandbox
    content = json.loads(receipt.read_text())
    content["uncompressed_sha256"] = "0" * 64
    receipt.write_text(json.dumps(content))
    result = assets.stage_static_database(source, receipt, reserve_bytes=1024)
    assert result["reason"] == "SOURCE_CONTENT_SHA_CONFLICT"
    assert list(scratch.glob("*/chembl22_sa2.db.tmp"))
    assert not list(scratch.glob("*/staging.json"))
    assert source.exists()


def test_source_change_during_target_verification_is_blocked(sandbox, monkeypatch):
    source, receipt, _ = sandbox
    original_hash = assets._file_digest
    def mutate_after_copy(path):
        value = original_hash(path)
        source.chmod(0o644)
        source.write_bytes(source.read_bytes() + b"changed")
        return value
    monkeypatch.setattr(assets, "_file_digest", mutate_after_copy)
    result = assets.stage_static_database(source, receipt, reserve_bytes=1024)
    assert result["reason"] == "SOURCE_CHANGED_DURING_TARGET_VERIFICATION"


def test_incomplete_stage_is_not_overwritten_or_restarted(sandbox):
    source, receipt, scratch = sandbox
    pin = json.loads(receipt.read_text())["uncompressed_sha256"]
    stage = scratch / ("cm-crem-static-db-" + pin[:20])
    stage.mkdir(mode=0o700)
    evidence = stage / "chembl22_sa2.db.tmp"
    evidence.write_bytes(b"partial evidence")
    result = assets.stage_static_database(source, receipt, reserve_bytes=1024)
    assert result["reason"] == "STAGING_IN_PROGRESS_OR_INCOMPLETE"
    assert evidence.read_bytes() == b"partial evidence"


def test_eio_explicitly_blocks_no_persistent_fallback(sandbox, monkeypatch):
    source, receipt, _ = sandbox
    def eio(path):
        raise OSError(5, "Input/output error")
    monkeypatch.setattr(assets, "_file_digest", eio)
    result = assets.stage_static_database(source, receipt, reserve_bytes=1024)
    assert result["status"] == "BLOCKED_LOCAL_DATABASE_STAGING"
    assert result["errno"] == 5


def test_tmpdir_only_requires_same_job_private_checks(sandbox, monkeypatch):
    source, receipt, scratch = sandbox
    monkeypatch.delenv("SLURM_TMPDIR")
    monkeypatch.setenv("TMPDIR", str(scratch))
    result = assets.stage_static_database(source, receipt, reserve_bytes=1024)
    assert result["status"] == "LOCAL_DATABASE_READY"
    assert result["binding"]["job_scratch"]["environment_variable"] == "TMPDIR"


@pytest.mark.parametrize("fs_type", ["nfs", "nfs4", "lustre", "fuse", "tmpfs", "overlay"])
def test_actual_mount_parser_does_not_confuse_shared_or_memory_storage_with_local_disk(monkeypatch, fs_type):
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(Path, "read_text", lambda self: (
        "20 1 8:1 / / rw - ext4 /dev/sda1 rw\n"
        f"31 20 0:51 / /site rw - {fs_type} remote-source rw\n"))
    with pytest.raises(assets._Blocked, match="NOT_VERIFIED_NODE_LOCAL_DISK"):
        assets._filesystem_identity(Path("/site/job-12345"))


def test_actual_mount_parser_picks_deepest_local_mount(monkeypatch):
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(Path, "read_text", lambda self: (
        "20 1 0:1 / / rw - nfs server:/root rw\n"
        "31 20 8:5 / /site rw - xfs /dev/nvme0n1 rw\n"))
    result = assets._filesystem_identity(Path("/site/job-12345"))
    assert result["source"] == "/dev/nvme0n1" and result["filesystem_type"] == "xfs"
