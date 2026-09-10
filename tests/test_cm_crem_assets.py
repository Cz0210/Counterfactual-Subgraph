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


def prepared_fixture(sandbox, monkeypatch):
    source, receipt, scratch = sandbox
    run = assets.HPC_SCOPE / "counterfactual-subgraph-hpc-runtime/baselines/cm_crem_global_v1/test-run"
    run.mkdir(parents=True)
    monkeypatch.setenv("SLURMD_NODENAME", assets.socket.gethostname())
    return source, receipt, scratch, run


def test_explicit_preparation_records_real_env_and_stages_once(sandbox, monkeypatch):
    source, receipt, base, run = prepared_fixture(sandbox, monkeypatch)
    before_env = dict(os.environ)
    prepared = assets.prepare_job_scratch(run, required_bytes=source.stat().st_size, reserve_bytes=1024)
    assert prepared["status"] == "JOB_SCRATCH_READY"
    assert prepared["base_selection"] == "SLURM_TMPDIR"
    assert prepared["raw_environment"]["SLURM_TMPDIR"] == str(base)
    assert Path(prepared["root"]).parent == base
    assert Path(prepared["root"]).stat().st_mode & 0o777 == 0o700
    assert dict(os.environ) == before_env
    assert assets.prepare_job_scratch(run, required_bytes=100, reserve_bytes=1024) == prepared
    staged = assets.stage_static_database(source, receipt, reserve_bytes=1024, scratch_receipt=prepared)
    assert staged["status"] == "LOCAL_DATABASE_READY"
    assert assets.stage_static_database(source, receipt, reserve_bytes=1024,
        scratch_receipt=prepared["receipt_path"])["reused"] is True


def test_authorized_tmp_fallback_is_explicit_not_fake_site_env(sandbox, monkeypatch):
    source, receipt, base, run = prepared_fixture(sandbox, monkeypatch)
    monkeypatch.delenv("SLURM_TMPDIR")
    monkeypatch.setattr(assets, "_PROJECT_SCRATCH_FALLBACK", base)
    prepared = assets.prepare_job_scratch(run, required_bytes=100, reserve_bytes=1024)
    assert prepared["status"] == "JOB_SCRATCH_READY"
    assert prepared["base_selection"] == "AUTHORIZED_PROJECT_TMP_FALLBACK"
    assert prepared["raw_environment"]["SLURM_TMPDIR"] is None
    assert assets.stage_static_database(source, receipt, reserve_bytes=1024)["reason"] == "NO_SCHEDULER_SCRATCH_ENVIRONMENT"
    assert assets.stage_static_database(source, receipt, reserve_bytes=1024, scratch_receipt=prepared)["status"] == "LOCAL_DATABASE_READY"


def test_prepared_scratch_rejects_login_and_changed_environment(sandbox, monkeypatch):
    source, receipt, base, run = prepared_fixture(sandbox, monkeypatch)
    monkeypatch.setenv("SLURMD_NODENAME", "unrelated-compute-node")
    assert assets.prepare_job_scratch(run, required_bytes=100)["reason"] == "COMPUTE_NODE_IDENTITY_UNPROVEN"
    monkeypatch.setenv("SLURMD_NODENAME", assets.socket.gethostname())
    prepared = assets.prepare_job_scratch(run, required_bytes=100)
    monkeypatch.setenv("SLURM_JOB_ID", "12346")
    assert assets.stage_static_database(source, receipt, reserve_bytes=1024, scratch_receipt=prepared)["reason"] == "PREPARED_SCRATCH_JOB_OR_ENVIRONMENT_CONFLICT"


def test_prepared_receipt_cannot_be_forged_or_root_changed(sandbox, monkeypatch):
    source, receipt, base, run = prepared_fixture(sandbox, monkeypatch)
    prepared = assets.prepare_job_scratch(run, required_bytes=100)
    assert assets.stage_static_database(source, receipt, reserve_bytes=1024,
        scratch_receipt={**prepared, "root": str(base)})["reason"] == "SCRATCH_RECEIPT_MEMORY_DISK_CONFLICT"
    Path(prepared["root"]).chmod(0o777)
    assert assets.stage_static_database(source, receipt, reserve_bytes=1024,
        scratch_receipt=prepared)["reason"] == "PREPARED_SCRATCH_ROOT_CHANGED"


def test_invalid_explicit_base_is_not_replaced_by_fallback(sandbox, monkeypatch):
    _, _, base, run = prepared_fixture(sandbox, monkeypatch)
    monkeypatch.setenv("SLURM_TMPDIR", "/share/not-local")
    monkeypatch.setattr(assets, "_PROJECT_SCRATCH_FALLBACK", base)
    result = assets.prepare_job_scratch(run, required_bytes=100)
    assert result["status"] == "BLOCKED_LOCAL_DATABASE_STAGING"
    assert not list(base.glob("cm-crem-job-*"))


def test_preparation_capacity_is_checked_before_mkdir(sandbox, monkeypatch):
    _, _, base, run = prepared_fixture(sandbox, monkeypatch)
    monkeypatch.setattr(assets.os, "statvfs", lambda p: SimpleNamespace(f_bavail=10, f_frsize=1024, f_favail=100))
    assert assets.prepare_job_scratch(run, required_bytes=100, reserve_bytes=100000)["reason"] == "LOCAL_CAPACITY_RESERVE_NOT_MET"
    assert not list(base.glob("cm-crem-job-*"))


def official_source_fixture(tmp_path):
    # Metadata/compatibility structure fixtures, not actual official DB proof.
    metadata = tmp_path.resolve() / "datacite.json"
    metadata.write_text(json.dumps({"data": {"attributes": {"doi": assets.ZENODO_DOI,
        "creators": [{"name": "Polishchuk, Pavel"}],
        "rightsList": [{"rightsUri": "https://creativecommons.org/licenses/by/4.0/", "rightsIdentifier": "cc-by-4.0"}]}}}))
    url = "https://zenodo.org/records/16909329/files/chembl22_sa2.db.gz?download=1"
    overlay = {"record_doi": assets.ZENODO_DOI, "record_url": assets.ZENODO_RECORD_URL,
        "download_url": url, "filename": assets.ZENODO_FILENAME,
        "published_md5": assets.ZENODO_COMPRESSED_MD5, "compressed_bytes": assets.ZENODO_COMPRESSED_BYTES,
        "compressed_sha256": assets.ZENODO_COMPRESSED_SHA256, "historical_url": assets.OFFICIAL_DATABASE_URL,
        "old_file_bytes_compared": False, "license": {"identifier": "CC-BY-4.0",
            "rights_uri": "https://creativecommons.org/licenses/by/4.0/", "scope": "database",
            "metadata_path": str(metadata), "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest()}}
    spec = {"upstream": {"database": {"url": assets.OFFICIAL_DATABASE_URL}}, "asset_source_overlay": overlay}
    body_stat = {"bytes": 12345, "device": 1, "inode": 2, "mtime_ns": 3, "ctime_ns": 4}
    compatibility = {"schema": "cm_crem_static_database_compatibility_v1", "status": "STATIC_DATABASE_COMPATIBILITY_PASS",
        "versions": {"crem": "0.2.14", "rdkit": "2023.9.6", "numpy": "1.26.4", "python": "3.11.5"},
        "radius": 1, "connection_mode": "mode=ro&immutable=1", "query_only": True,
        "journal_mode_changed": False, "durability_changed": False, "unchanged_static_source": True,
        "stat_before": body_stat, "stat_after": body_stat, "sidecars_before": [], "sidecars_after": [],
        "columns": [{"name": n} for n in ("env", "freq", "core_num_atoms", "core_smi", "core_sma")],
        "first_row_fields_valid": True, "radius1_rowid_supported": True, "fixture_smiles": "CCO",
        "fixture_settings": {"radius": 1, "min_inc": 0, "max_inc": 0, "max_replacements": 4,
            "replace_ids": [0], "ncores": 1, "symmetry_fixes": True},
        "fixture_mutate_calls": 1, "fixture_select_count": 1, "actual_select_count": 3,
        "public_fixture_products": ["NCO"], "public_fixture_product_count": 1,
        "experiment_generation_performed": False, "oracle_calls": 0}
    receipt = {"status": "VERIFIED_STATIC_COPY", "url": url, "uncompressed_sha256": "a" * 64,
        "uncompressed_bytes": 12345, "compressed_bytes": assets.ZENODO_COMPRESSED_BYTES,
        "compressed_md5": assets.ZENODO_COMPRESSED_MD5, "compressed_sha256": assets.ZENODO_COMPRESSED_SHA256,
        "compatibility": compatibility, "compatibility_database_sha256": "a" * 64}
    return spec, receipt


def test_author_source_overlay_preserves_original_science_identity(tmp_path):
    spec, receipt = official_source_fixture(tmp_path)
    before = json.dumps(spec, sort_keys=True)
    result = assets.validate_database_source(spec, receipt)
    assert result["actual_source_url"] == spec["asset_source_overlay"]["download_url"]
    assert result["original_requested_url"] == assets.OFFICIAL_DATABASE_URL
    assert result["old_file_bytes_compared"] is False
    assert before == json.dumps(spec, sort_keys=True)


@pytest.mark.parametrize("field,value", [("record_doi", "10.5281/zenodo.1"),
    ("compressed_sha256", "b" * 64), ("published_md5", "0" * 32),
    ("old_file_bytes_compared", True), ("download_url", "https://other.org/chembl22_sa2.db.gz")])
def test_unapproved_author_overlay_rejected(tmp_path, field, value):
    spec, receipt = official_source_fixture(tmp_path)
    spec["asset_source_overlay"][field] = value
    with pytest.raises(ValueError):
        assets.validate_database_source(spec, receipt)


def test_license_metadata_requires_actual_author_record_and_hash(tmp_path):
    spec, receipt = official_source_fixture(tmp_path)
    metadata = Path(spec["asset_source_overlay"]["license"]["metadata_path"])
    metadata.write_text("{}")
    with pytest.raises(ValueError, match="DATABASE_LICENSE_METADATA_SHA_CONFLICT"):
        assets.validate_database_source(spec, receipt)


def test_missing_native_query_not_hidden_by_metadata_pass(tmp_path):
    spec, receipt = official_source_fixture(tmp_path)
    receipt["compatibility"]["fixture_select_count"] = 0
    with pytest.raises(ValueError, match="DATABASE_NATIVE_QUERY_OR_REPLACEMENT_EVIDENCE_MISSING"):
        assets.validate_database_source(spec, receipt)
    assert assets.validate_database_source(spec, receipt, require_compatibility=False)["compatibility_validated"] is False


def test_compatibility_must_bind_same_uncompressed_content(tmp_path):
    spec, receipt = official_source_fixture(tmp_path)
    receipt["compatibility_database_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="DATABASE_COMPATIBILITY_CONTENT_BINDING_CONFLICT"):
        assets.validate_database_source(spec, receipt)


def test_legacy_source_is_explicit_not_silently_rebound(tmp_path):
    spec, receipt = official_source_fixture(tmp_path)
    spec.pop("asset_source_overlay")
    with pytest.raises(ValueError, match="DATABASE_STATIC_COPY_SOURCE_URL_CONFLICT"):
        assets.validate_database_source(spec, receipt)
    receipt["url"] = assets.OFFICIAL_DATABASE_URL
    assert assets.validate_database_source(spec, receipt)["source_mode"] == "HISTORICAL_AUTHOR_URL"


def test_actual_datacite_legalcode_uri_and_predecompression_source_binding(tmp_path):
    spec, receipt = official_source_fixture(tmp_path)
    license_info = spec["asset_source_overlay"]["license"]
    license_info["rights_uri"] = "https://creativecommons.org/licenses/by/4.0/legalcode"
    metadata = Path(license_info["metadata_path"])
    data = json.loads(metadata.read_text())
    data["data"]["attributes"]["rightsList"][0]["rightsUri"] = license_info["rights_uri"]
    metadata.write_text(json.dumps(data))
    license_info["metadata_sha256"] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    receipt["status"] = "VERIFIED_COMPRESSED_COPY"
    for key in ("uncompressed_sha256", "uncompressed_bytes", "compatibility", "compatibility_database_sha256"):
        receipt.pop(key)
    result = assets.validate_database_source(spec, receipt, require_compatibility=False, require_content=False)
    assert result["content_validated"] is False and result["uncompressed_sha256"] is None
    assert result["compatibility_validated"] is False
    with pytest.raises(ValueError, match="DATABASE_STATIC_COPY_NOT_VERIFIED"):
        assets.validate_database_source(spec, receipt)
    with pytest.raises(ValueError, match="DATABASE_COMPATIBILITY_REQUIRES_CONTENT_BINDING"):
        assets.validate_database_source(spec, receipt, require_content=False)
