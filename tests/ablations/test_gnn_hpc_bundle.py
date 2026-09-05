import hashlib
import io
import json
import tarfile
import pytest
from src.ablations.gnn.hpc_bundle import atomic_json, checked_file, unpack_bundle, verify_bundle
from src.ablations.contracts import canonical_json_sha256


def fixture_bundle(root):
    root.mkdir()
    (root / "train.csv").write_text("id,smiles,label\n1,CCO,1\n")
    body = {"schema_version": "bace_gnn_cpu_bundle_v1", "main_matrix_write_allowed": False,
            "files": {"train.csv": checked_file(root / "train.csv")}}
    body["manifest_sha256"] = canonical_json_sha256(body)
    atomic_json(root / "bundle_manifest.json", body)
    return body


def test_bundle_roundtrip(tmp_path):
    root = tmp_path / "source"
    expected = fixture_bundle(root)
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in root.iterdir():
            tar.add(path, arcname=path.name)
    unpack_bundle(archive, checked_file(archive)["sha256"], tmp_path / "imported")
    assert verify_bundle(tmp_path / "imported") == expected
    with pytest.raises(ValueError, match="already exists"):
        unpack_bundle(archive, checked_file(archive)["sha256"], tmp_path / "imported")


def test_bundle_drift_and_extra_file_fail(tmp_path):
    root = tmp_path / "source"
    fixture_bundle(root)
    (root / "secret").write_text("no")
    with pytest.raises(ValueError, match="inventory"):
        verify_bundle(root)
    (root / "secret").unlink()
    (root / "train.csv").write_text("changed")
    with pytest.raises(ValueError, match="SHA mismatch"):
        verify_bundle(root)


@pytest.mark.parametrize("name,kind", [("../escape", "file"), ("/absolute", "file"), ("link", "symlink")])
def test_bundle_rejects_escape_and_links(tmp_path, name, kind):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(name)
        info.size = 1
        if kind == "symlink":
            info.type = tarfile.SYMTYPE
            info.linkname = "/tmp/elsewhere"
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        unpack_bundle(archive, checked_file(archive)["sha256"], tmp_path / "dest")


def test_ablation_cannot_write_main_matrix(tmp_path):
    root = tmp_path / "source"
    body = fixture_bundle(root)
    body["main_matrix_write_allowed"] = True
    body.pop("manifest_sha256")
    body["manifest_sha256"] = canonical_json_sha256(body)
    atomic_json(root / "bundle_manifest.json", body)
    with pytest.raises(ValueError, match="contract"):
        verify_bundle(root)
