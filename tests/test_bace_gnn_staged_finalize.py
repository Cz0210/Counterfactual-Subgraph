"""Publication failure/recovery over a genuine tiny trained BACE bundle."""
from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest

from test_bace_gnn_cpu_training import _bundle
from src.ablations.gnn.cpu_training import file_sha256, load_bundle, run_cpu_training
from src.ablations.gnn import staged_finalize as publication
from src.train import molecular_gnn_resume as resume


@pytest.fixture
def staged(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("rdkit")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    scope = tmp_path / "gnn"
    scope.mkdir()
    monkeypatch.setattr(publication, "HPC_GNN_SCOPE", scope)
    bundle = _bundle(tmp_path)
    attempt = scope / "campaign/gin/attempt"
    def unsupported(*args, **kwargs):
        raise OSError(errno.EINVAL, "renameat2 unsupported on HPC filesystem")
    monkeypatch.setattr(resume, "atomic_rename_directory_noreplace", unsupported)
    with pytest.raises(OSError, match="renameat2 unsupported"):
        run_cpu_training(bundle_root=bundle, backbone="gin", phase="train",
            output_root=attempt, config_path=bundle / "configs/gin.yaml", cpu_threads=1)
    envelope = json.loads((attempt / "training_state/training_contract.json").read_text())
    prefix = f".classifier.finalizing-{envelope['contract_sha256']}"
    stage = attempt / prefix
    assert json.loads((attempt / (prefix + ".complete.json")).read_text())["status"] == "BUNDLE_COMPLETE"
    assert len(list(stage.iterdir())) == 13
    return bundle, attempt, stage, envelope


def publish(staged):
    bundle, attempt, _, _ = staged
    root, manifest = load_bundle(bundle)
    return publication.finalize_one(bundle_root=root, manifest=manifest, backbone="gin",
                                    classifier_root=attempt / "classifier")


def test_real_staged_publish_preserves_bytes_inode_and_checkpoint_without_training(staged, monkeypatch):
    from scripts import train_molecular_gnn as trainer
    bundle, attempt, stage, envelope = staged
    before = {p.name: file_sha256(p) for p in stage.iterdir()}
    stage_inode = stage.stat().st_ino
    science_before = {p.name: file_sha256(p) for p in (attempt / "training_state").iterdir() if p.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError("Publication must never retrain/refit")
    monkeypatch.setattr(trainer, "main", forbidden)
    result = publish(staged)
    output = attempt / "classifier"
    assert result["status"] == "TRAINING_PASS"
    assert result["science_rerun"] is False and result["temperature_refit"] is False
    assert result["elapsed_seconds"] is None
    assert result["completed_epoch"] == 3
    assert output.stat().st_ino == stage_inode and not stage.exists()
    assert {p.name: file_sha256(p) for p in output.iterdir()} == before
    for name, digest in science_before.items():
        assert file_sha256(attempt / "training_state" / name) == digest
    assert (attempt / "training_state/training_complete.json").exists()
    terminal_before = (attempt / "training_terminal.json").read_bytes()
    again = publish(staged)
    assert again["status"] == "TRAINING_PASS"
    assert again["rename_performed_this_invocation"] is False
    assert (attempt / "training_terminal.json").read_bytes() == terminal_before


def test_incomplete_ready_is_not_cleaned_or_resealed(staged, monkeypatch):
    _, attempt, stage, envelope = staged
    ready = attempt / (stage.name + ".complete.json")
    ready.unlink()
    before = {p.name: file_sha256(p) for p in stage.iterdir()}
    def forbidden(*args, **kwargs):
        raise AssertionError("Missing ready may not enter destructive prepare branch")
    monkeypatch.setattr(publication.FinalizationWorkspace, "prepare", forbidden)
    with pytest.raises(FileNotFoundError):
        publish(staged)
    assert {p.name: file_sha256(p) for p in stage.iterdir()} == before
    assert not (attempt / "training_state/finalization_cleanup.json").exists()


def test_active_original_parent_lock_blocks_publication(staged):
    _, attempt, stage, envelope = staged
    authority = resume.OutputParentAuthority(attempt / "classifier",
        contract_sha256=envelope["contract_sha256"], resume=True)
    authority.open()
    try:
        with pytest.raises(resume.MolecularGNNResumeError, match="another writer"):
            publish(staged)
        assert stage.exists() and not (attempt / "classifier").exists()
    finally:
        authority.close()


def test_active_original_training_writer_blocks_publication(staged):
    import torch
    _, attempt, stage, envelope = staged
    store = resume.MolecularGNNResumeStore(attempt / "training_state", resume=True,
        contract=envelope["contract"], torch_module=torch)
    store.open()
    try:
        with pytest.raises(resume.MolecularGNNResumeError, match="writer"):
            publish(staged)
        assert stage.exists() and not (attempt / "classifier").exists()
    finally:
        store.close()


def test_conflicting_destination_is_never_overwritten(staged):
    _, attempt, stage, _ = staged
    output = attempt / "classifier"
    output.mkdir()
    marker = output / "unrelated.txt"
    marker.write_text("preserve me")
    with pytest.raises(resume.MolecularGNNResumeError, match="staging sibling"):
        publish(staged)
    assert marker.read_text() == "preserve me" and stage.exists()


def test_tampered_staged_inventory_blocks_without_retraining(staged):
    _, attempt, stage, _ = staged
    with (stage / "model.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(resume.MolecularGNNResumeError, match="inventory changed"):
        publish(staged)
    assert stage.exists() and not (attempt / "classifier").exists()


def test_scoped_path_rejects_non_gnn_and_symlink_paths(tmp_path, monkeypatch):
    scope = tmp_path / "gnn"
    scope.mkdir()
    monkeypatch.setattr(publication, "HPC_GNN_SCOPE", scope)
    with pytest.raises(ValueError, match="dedicated HPC GNN"):
        publication.scoped_path(tmp_path / "other")
    link = scope / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        publication.scoped_path(link / "classifier")
