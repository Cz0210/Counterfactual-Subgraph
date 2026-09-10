"""Synthetic transport/gate fixtures only; never publish a production result."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from src.baselines import cm_crem_release as release


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = data if isinstance(data, bytes) else (json.dumps(data, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def run(tmp_path, monkeypatch):
    base = tmp_path.resolve() / "cm_crem_global_v1"
    root = base / "fixture-run"
    root.mkdir(parents=True)
    monkeypatch.setattr(release, "ALLOWED_ROOTS", (base,))
    for name in release.CORE_FILES:
        _write(root / name, {"fixture_data": "no production science"})
    science = "a" * 64
    _write(root / "spec.json", {"method_id": release.METHOD, "science_hash": science})
    _write(root / "bindings.json", {"fixture_bindings": True})
    csv_pins = {name: _write(root / "results/source_csv" / name, b"fixture_column\nfixture_only\n")
                for name in release.RESULT_FILES}
    # False here exercises the real gate shape on isolated synthetic inputs.
    # The test's run path and body remain explicitly fixture-only.
    _write(root / "results/export_manifest.json", {"fixture": False, "dataset": "bace", "oracle": "gine",
                                                 "source_files": csv_pins})
    spot = {"status": "CM_CREM_INDEPENDENT_SPOTCHECK_PASS", "independent": True,
            "science_hash": science, "implementation_commit": "b" * 40, "implementation_sha256": "c" * 64,
            "checked_records_count": 2, "scientific_pass_claimed": False,
            "source_test_evaluation_sha256": release._sha(root / "test_evaluation.json"),
            "source_selection_freeze_sha256": release._sha(root / "selection_freeze.json")}
    spot_sha = _write(root / "audit/independent_spotcheck.json", spot)
    _write(root / "audit/final_audit.json", {"status": "BACE_CM_CREM_FINAL_AUDIT_PASS",
        "scientific_pass_claimed": True, "main_matrix_written": False, "science_hash": science,
        "independent_spotcheck": {"path": "audit/independent_spotcheck.json", "sha256": spot_sha}})
    _write(root / "test/parents/fixture-parent.json", {"fixture": True, "distance": 0.0})
    _write(root / "models/never-package.pt", b"large-model-excluded")
    _write(root / "logs/never-package.log", b"cold-log-excluded")
    _write(root / "assets/chembl22_sa2.db", b"database-excluded")
    return root


def test_gate_package_import_and_no_model_database_log_copy(run):
    record = release.package_run(run)
    manifest = json.loads(Path(record["manifest_path"]).read_text())
    assert record["status"] == "CM_RESULT_PACKAGE_SEALED"
    assert not any(name.startswith(("models/", "logs/", "assets/")) for name in manifest["files"])
    destination = run.parent / "fixture-import"
    imported = release.verify_import(record["package_path"], record["manifest_path"], destination)
    assert imported["status"] == "CM_RESULT_IMPORT_VERIFIED"
    assert imported["full_package_hash_count"] == 1
    assert imported["main_matrix_written"] is False and imported["registry_created"] is False
    assert (destination / "test/parents/fixture-parent.json").read_bytes() == (run / "test/parents/fixture-parent.json").read_bytes()
    assert (destination / "cm_run_publication.json").is_file()
    with pytest.raises(release.ReleaseRejected, match="fresh"):
        release.verify_import(record["package_path"], record["manifest_path"], destination)
    with pytest.raises(release.ReleaseRejected, match="already exists"):
        release.package_run(run)


@pytest.mark.parametrize("field,value", [("status", "RECORD_RECONCILIATION_COMPLETE"),
    ("scientific_pass_claimed", False), ("main_matrix_written", True)])
def test_final_audit_flags_not_generic_pass(run, field, value):
    path = run / "audit/final_audit.json"
    data = json.loads(path.read_text())
    data[field] = value
    _write(path, data)
    with pytest.raises(release.ReleaseRejected, match="not accepted"):
        release.package_run(run)
    assert not (run / "release").exists()


@pytest.mark.parametrize("field,value", [("independent", False), ("checked_records_count", 0),
    ("implementation_commit", None), ("source_test_evaluation_sha256", "d" * 64)])
def test_spotcheck_actual_identity_and_source_bindings_required(run, field, value):
    spot_path = run / "audit/independent_spotcheck.json"
    data = json.loads(spot_path.read_text())
    data[field] = value
    sha = _write(spot_path, data)
    audit = json.loads((run / "audit/final_audit.json").read_text())
    audit["independent_spotcheck"]["sha256"] = sha
    _write(run / "audit/final_audit.json", audit)
    with pytest.raises(release.ReleaseRejected):
        release.package_run(run)


def test_synthetic_export_and_source_csv_conflict_rejected(run):
    _write(run / "results/source_csv/table2_k10.csv", b"changed\n")
    with pytest.raises(release.ReleaseRejected, match="CSV source binding"):
        release.package_run(run)


def test_source_symlink_never_followed(run):
    (run / "test/parents/linked.json").symlink_to(run / "spec.json")
    with pytest.raises(release.ReleaseRejected, match="Nonregular"):
        release.package_run(run)


def test_non_cm_destination_rejected_before_writes(run, tmp_path):
    packaged = release.package_run(run)
    outside = tmp_path / "matrix_authority"
    with pytest.raises(release.ReleaseRejected, match="outside fresh CM"):
        release.verify_import(packaged["package_path"], packaged["manifest_path"], outside)
    assert not outside.exists()


def _malicious_archive(run, kind):
    good = release.package_run(run)
    directory = run.parent / ("transport-fixture-" + kind)
    directory.mkdir()
    archive_path = directory / "bad.tar.gz"
    with tarfile.open(good["package_path"], "r:gz") as src, tarfile.open(archive_path, "w:gz") as dest:
        members = src.getmembers()
        first = members[0]
        if kind == "duplicate":
            dest.addfile(first, io.BytesIO(src.extractfile(first).read()))
        for member in members:
            data = src.extractfile(member).read()
            dest.addfile(member, io.BytesIO(data))
        if kind != "duplicate":
            name = {"traversal": "../escaped", "absolute": "/escaped", "unknown": "not_whitelisted.json"}.get(kind, "audit/evil")
            evil = tarfile.TarInfo(name)
            if kind == "symlink":
                evil.type, evil.linkname = tarfile.SYMTYPE, "../../escaped"
            elif kind == "hardlink":
                evil.type, evil.linkname = tarfile.LNKTYPE, "spec.json"
            elif kind == "directory":
                evil.type = tarfile.DIRTYPE
            elif kind == "fifo":
                evil.type = tarfile.FIFOTYPE
            dest.addfile(evil, io.BytesIO(b""))
    outer = json.loads(Path(good["manifest_path"]).read_text())
    outer["package_sha256"], outer["package_bytes"] = release._sha(archive_path), archive_path.stat().st_size
    manifest_path = directory / "manifest.json"
    _write(manifest_path, outer)
    return archive_path, manifest_path


@pytest.mark.parametrize("kind", ["duplicate", "traversal", "absolute", "unknown", "symlink", "hardlink", "directory", "fifo"])
def test_malicious_tar_members_cannot_finalize_or_escape(run, kind):
    package, manifest = _malicious_archive(run, kind)
    destination = run.parent / ("import-" + kind)
    with pytest.raises(release.ReleaseRejected):
        release.verify_import(package, manifest, destination)
    assert not destination.exists()
    assert not (run.parent / "escaped").exists()


def test_full_new_package_hash_is_verified_before_extract(run):
    package = release.package_run(run)
    with Path(package["package_path"]).open("ab") as stream:
        stream.write(b"new corruption")
    destination = run.parent / "bad-hash"
    with pytest.raises(release.ReleaseRejected, match="bytes/SHA"):
        release.verify_import(package["package_path"], package["manifest_path"], destination)
    assert not destination.exists()


def test_no_replace_finalization_preserves_even_empty_existing_destination(run):
    source, destination = run / "stage-fixture", run / "existing-fixture"
    source.mkdir()
    destination.mkdir()
    with pytest.raises((FileExistsError, OSError)):
        release._atomic_directory(source, destination)
    assert source.is_dir() and destination.is_dir()


def test_source_tree_change_during_package_preserves_evidence_not_final_release(run, monkeypatch):
    original = release._HashReader.read
    changed = False
    def change_source(reader, count=-1):
        nonlocal changed
        result = original(reader, count)
        if not changed:
            changed = True
            _write(run / "test/parents/fixture-parent.json", {"fixture": "changed by concurrent writer"})
        return result
    monkeypatch.setattr(release._HashReader, "read", change_source)
    with pytest.raises(release.ReleaseRejected, match="Source changed|source tree changed"):
        release.package_run(run)
    assert not (run / "release").exists()
    assert list(run.glob("release.tmp-*"))


def test_members_streamed_sha_rejects_repacked_changed_content(run):
    good = release.package_run(run)
    transport = run.parent / "repacked"
    transport.mkdir()
    package = transport / "changed.tar.gz"
    with tarfile.open(good["package_path"], "r:gz") as src, tarfile.open(package, "w:gz") as dst:
        for member in src.getmembers():
            body = src.extractfile(member).read()
            if member.name == "test/parents/fixture-parent.json":
                body = b"x" * len(body)
            dst.addfile(member, io.BytesIO(body))
    manifest = json.loads(Path(good["manifest_path"]).read_text())
    manifest.update(package_sha256=release._sha(package), package_bytes=package.stat().st_size)
    manifest_path = transport / "manifest.json"
    _write(manifest_path, manifest)
    destination = run.parent / "bad-member"
    with pytest.raises(release.ReleaseRejected, match="Member content SHA"):
        release.verify_import(package, manifest_path, destination)
    assert not destination.exists()


def test_prepared_archive_finalize_without_repacking(run, monkeypatch):
    def unsupported(source, destination):
        raise OSError(22, 'filesystem does not support renameat2 flags')
    monkeypatch.setattr(release, '_atomic_directory', unsupported)
    with pytest.raises(OSError):
        release.package_run(run)
    staging, = run.glob('release.tmp-*')
    package = staging/'cm_crem_results.tar.gz'
    original = package.read_bytes()
    monkeypatch.setattr(release, '_atomic_directory', release._receipt_directory)
    result = release.finalize_prepared_package(run, staging)
    assert Path(result['package_path']).read_bytes() == original
    assert not staging.exists()
    assert (run/'release/package_receipt.json').exists()


def test_receipt_fallback_never_overwrites_or_accepts_partial(run, monkeypatch):
    source, destination = run/'pending', run/'existing'
    source.mkdir(); destination.mkdir()
    _write(source/'package_receipt.json', {'fixture': True})
    with pytest.raises(FileExistsError):
        release._receipt_directory(source, destination)
    assert not (destination/'package_receipt.json').exists()
    fresh=run/'fresh'
    _write(source/'data.json', {'fixture': 'body'})
    def fail_sync(root):
        raise OSError(5, 'fixture fsync failure')
    monkeypatch.setattr(release, '_sync_directories', fail_sync)
    with pytest.raises(OSError):
        release._receipt_directory(source, fresh)
    assert not (fresh/'package_receipt.json').exists()
    assert (source/'package_receipt.json').exists()
