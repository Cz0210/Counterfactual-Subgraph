from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import pytest

from src.train import molecular_gnn_resume as resume_module
from src.train.molecular_gnn_resume import (
    COMPLETE_NAME,
    FinalizationWorkspace,
    LATEST_NAME,
    MolecularGNNResumeError,
    MolecularGNNResumeStore,
    MolecularGNNStateReadAuthority,
    OutputParentAuthority,
    atomic_rename_directory_noreplace,
    assert_no_symlink_components,
    canonical_sha256,
)


torch = pytest.importorskip("torch")


def _model_optimizer() -> tuple[object, object]:
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.ReLU(), torch.nn.Linear(5, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def _contract(output: Path) -> dict[str, object]:
    return {
        "schema_version": "fixture_contract_v1",
        "dataset": "tastemolnet",
        "output_dir": str(output),
        "seed": 7,
    }


def _advance(model: object, optimizer: object) -> None:
    optimizer.zero_grad(set_to_none=True)
    values = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10.0
    loss = model(values).square().mean()
    loss.backward()
    optimizer.step()


def _save_epoch(
    store: MolecularGNNResumeStore,
    *,
    epoch: int,
    model: object,
    optimizer: object,
) -> dict[str, object]:
    return store.save(
        completed_epoch=epoch,
        model=model,
        optimizer=optimizer,
        best_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
        best_epoch=epoch,
        best_primary=0.5 + epoch / 100.0,
        best_tiebreak=0.4 + epoch / 100.0,
        epochs_without_improvement=0,
        history=[{"epoch": index} for index in range(1, epoch + 1)],
        metrics={"selection": 0.5 + epoch / 100.0},
    )


def test_epoch_checkpoint_roundtrip_restores_model_optimizer_rng_and_history(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    output = tmp_path / "output"
    torch.manual_seed(19)
    model, optimizer = _model_optimizer()
    _advance(model, optimizer)
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    with MolecularGNNResumeStore(
        root, resume=False, contract=_contract(output), torch_module=torch
    ) as store:
        manifest = _save_epoch(store, epoch=1, model=model, optimizer=optimizer)
        assert manifest["next_epoch"] == 2

    resumed_model, resumed_optimizer = _model_optimizer()
    with MolecularGNNResumeStore(
        root, resume=True, contract=_contract(output), torch_module=torch
    ) as store:
        snapshot = store.load(model=resumed_model, optimizer=resumed_optimizer)
        assert snapshot is not None
        assert snapshot.completed_epoch == 1
        assert snapshot.next_epoch == 2
        assert snapshot.history == [{"epoch": 1}]
        for key, value in resumed_model.state_dict().items():
            assert torch.equal(value, expected[key])


def test_interrupted_checkpoint_publication_is_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "state"
    output = tmp_path / "output"
    model, optimizer = _model_optimizer()
    store = MolecularGNNResumeStore(
        root, resume=False, contract=_contract(output), torch_module=torch
    )
    store.open()
    real_publish = resume_module._atomic_json_replace

    def interrupt(path: Path, payload: dict[str, object]) -> None:
        if path.name == LATEST_NAME:
            raise RuntimeError("simulated crash after checkpoint fsync")
        real_publish(path, payload)

    monkeypatch.setattr(resume_module, "_atomic_json_replace", interrupt)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _save_epoch(store, epoch=1, model=model, optimizer=optimizer)
    store.close()
    monkeypatch.setattr(resume_module, "_atomic_json_replace", real_publish)

    resumed_model, resumed_optimizer = _model_optimizer()
    with MolecularGNNResumeStore(
        root, resume=True, contract=_contract(output), torch_module=torch
    ) as resumed:
        snapshot = resumed.load(model=resumed_model, optimizer=resumed_optimizer)
        assert snapshot is not None and snapshot.completed_epoch == 1
        latest = json.loads((root / LATEST_NAME).read_text(encoding="utf-8"))
        assert latest["reconciled_after_interrupted_publication"] is True


def test_checkpoint_cleanup_keeps_current_and_previous_with_audit(tmp_path: Path) -> None:
    root = tmp_path / "state"
    model, optimizer = _model_optimizer()
    with MolecularGNNResumeStore(
        root,
        resume=False,
        contract=_contract(tmp_path / "output"),
        torch_module=torch,
    ) as store:
        for epoch in range(1, 4):
            _advance(model, optimizer)
            _save_epoch(store, epoch=epoch, model=model, optimizer=optimizer)
    assert sorted(path.name for path in root.glob("checkpoint-*.pt")) == [
        "checkpoint-000002.pt",
        "checkpoint-000003.pt",
    ]
    cleanup = json.loads((root / "checkpoint_cleanup.json").read_text(encoding="utf-8"))
    assert cleanup["removed"][0]["path"] == "checkpoint-000001.pt"
    assert cleanup["removed"][0]["reconstructable"] is True


def test_contract_tamper_and_concurrent_writer_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "state"
    output = tmp_path / "output"
    first = MolecularGNNResumeStore(
        root, resume=False, contract=_contract(output), torch_module=torch
    )
    first.open()
    with pytest.raises(MolecularGNNResumeError, match="another molecular-GNN"):
        MolecularGNNResumeStore(
            root, resume=True, contract=_contract(output), torch_module=torch
        ).open()
    first.close()
    contract_path = root / "training_contract.json"
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["contract"]["seed"] = 8
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    for _ in range(2):
        failed = MolecularGNNResumeStore(
            root, resume=True, contract=_contract(output), torch_module=torch
        )
        with pytest.raises(MolecularGNNResumeError, match="contract changed"):
            failed.open()


def test_live_writer_recomputes_contract_canonical_sha_and_physical_binding(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    output = tmp_path / "output"
    store = MolecularGNNResumeStore(
        root, resume=False, contract=_contract(output), torch_module=torch
    )
    store.open()
    contract_path = root / "training_contract.json"
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["contract"]["runtime_identity"] = {"torch": "drifted"}
    # Deliberately retain the old self-declared contract_sha256: the held
    # physical/canonical verifier must not trust that declaration.
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        MolecularGNNResumeError,
        match="physical|canonical|changed",
    ):
        store.verify_writer_authority()
    store.close()


def test_train_validation_cache_contract_drift_is_rejected_across_resume(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    output = tmp_path / "output"
    contract = _contract(output)
    contract["training_input"] = {
        "cache_contract": {
            "splits": {
                "train": {"path": "/private/train.pt", "sha256": "a" * 64},
                "validation": {
                    "path": "/private/validation.pt",
                    "sha256": "b" * 64,
                },
            },
            "test_loaded": False,
        }
    }
    with MolecularGNNResumeStore(
        root, resume=False, contract=contract, torch_module=torch
    ):
        pass
    drifted = copy.deepcopy(contract)
    drifted["training_input"]["cache_contract"]["splits"]["train"][
        "sha256"
    ] = "c" * 64
    with pytest.raises(MolecularGNNResumeError, match="resume contract changed"):
        MolecularGNNResumeStore(
            root, resume=True, contract=drifted, torch_module=torch
        ).open()


def test_named_lock_replacement_is_detected(tmp_path: Path) -> None:
    root = tmp_path / "state"
    with MolecularGNNResumeStore(
        root,
        resume=False,
        contract=_contract(tmp_path / "output"),
        torch_module=torch,
    ) as store:
        lock = root / ".writer.lock"
        lock.unlink()
        lock.write_bytes(b"replacement")
        with pytest.raises(MolecularGNNResumeError, match="lock inode changed"):
            store.verify_writer_authority()


def test_training_state_rejects_symlinked_parent(tmp_path: Path) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(physical, target_is_directory=True)
    with pytest.raises(MolecularGNNResumeError, match="symlink components"):
        MolecularGNNResumeStore(
            alias / "state",
            resume=False,
            contract=_contract(tmp_path / "output"),
            torch_module=torch,
        )


def test_completion_is_idempotent_and_contract_bound(tmp_path: Path) -> None:
    root = tmp_path / "state"
    output = tmp_path / "output"
    output.mkdir()
    identity = {"checkpoint_id": "a" * 64, "sha256s_sha256": "b" * 64}
    with MolecularGNNResumeStore(
        root, resume=False, contract=_contract(output), torch_module=torch
    ) as store:
        first = store.mark_complete(output_dir=output, output_identity=identity)
        second = store.mark_complete(output_dir=output, output_identity=identity)
        assert first == second
        assert store.completion() == first
    assert (root / COMPLETE_NAME).is_file()


def _finalization_authorities(
    tmp_path: Path, *, resume: bool = False
) -> tuple[Path, Path, OutputParentAuthority, FinalizationWorkspace]:
    output = tmp_path / "published"
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    contract_sha = canonical_sha256(_contract(output))
    parent = OutputParentAuthority(
        output, contract_sha256=contract_sha, resume=resume
    )
    parent.open()
    workspace = FinalizationWorkspace(
        output,
        contract_sha256=contract_sha,
        resume=resume,
        parent_authority=parent,
        training_state_root=state,
    )
    return output, state, parent, workspace


def test_atomic_directory_publish_never_replaces_existing_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (target / "owner").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        atomic_rename_directory_noreplace(source, target)
    assert source.is_dir()
    assert (target / "owner").read_text(encoding="utf-8") == "keep"


def test_linux_publish_uses_renameat2_noreplace_relative_to_held_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    calls: list[tuple[object, ...]] = []

    class FakeRenameAt2:
        argtypes = None
        restype = None

        def __call__(self, *arguments: object) -> int:
            calls.append(arguments)
            return 0

    class FakeLibrary:
        renameat2 = FakeRenameAt2()

    monkeypatch.setattr(resume_module.sys, "platform", "linux")
    monkeypatch.setattr(
        resume_module.ctypes, "CDLL", lambda *args, **kwargs: FakeLibrary()
    )
    parent_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        atomic_rename_directory_noreplace(
            source, target, directory_fd=parent_fd
        )
    finally:
        os.close(parent_fd)
    assert calls == [(parent_fd, b"source", parent_fd, b"target", 1)]


def test_output_parent_recovers_sentinel_before_claim_crash_window(
    tmp_path: Path,
) -> None:
    output = tmp_path / "published"
    contract_sha = canonical_sha256(_contract(output))
    interrupted = OutputParentAuthority(
        output, contract_sha256=contract_sha, resume=False
    )
    interrupted.sentinel_path.write_bytes(b"interrupted-sentinel")
    interrupted.sentinel_path.chmod(0o600)

    resumed = OutputParentAuthority(
        output, contract_sha256=contract_sha, resume=True
    )
    resumed.open()
    assert resumed.claim_path.is_file()
    resumed.verify()
    resumed.close()


def test_output_parent_detects_same_byte_parent_root_replacement(
    tmp_path: Path,
) -> None:
    parent_root = tmp_path / "publication-parent"
    parent_root.mkdir()
    output = parent_root / "published"
    authority = OutputParentAuthority(
        output,
        contract_sha256=canonical_sha256(_contract(output)),
        resume=False,
    )
    authority.open()
    displaced = tmp_path / "displaced-publication-parent"
    parent_root.rename(displaced)
    shutil.copytree(displaced, parent_root, copy_function=shutil.copy2)
    with pytest.raises(MolecularGNNResumeError, match="physical identity changed"):
        authority.verify()
    authority.close()


def test_finalization_recovers_empty_mkdir_before_claim_window(tmp_path: Path) -> None:
    output, _, parent, workspace = _finalization_authorities(tmp_path)
    workspace.staging.mkdir(mode=0o700)
    parent.close()

    _, _, resumed_parent, resumed = _finalization_authorities(tmp_path, resume=True)
    staging, ready = resumed.prepare()
    assert staging == resumed.staging
    assert ready is False
    assert resumed.claim_path.is_file()
    resumed.close()
    resumed_parent.close()
    assert not output.exists()


def test_finalization_rejects_nonempty_unclaimed_orphan(tmp_path: Path) -> None:
    _, _, parent, workspace = _finalization_authorities(tmp_path)
    workspace.staging.mkdir(mode=0o700)
    (workspace.staging / "unknown.bin").write_bytes(b"not-owned")
    parent.close()
    _, _, resumed_parent, resumed = _finalization_authorities(tmp_path, resume=True)
    with pytest.raises(MolecularGNNResumeError, match="unclaimed.*not empty"):
        resumed.prepare()
    resumed_parent.close()


def test_finalization_detects_same_byte_staging_root_replacement(tmp_path: Path) -> None:
    _, _, parent, workspace = _finalization_authorities(tmp_path)
    staging, _ = workspace.prepare()
    (staging / "model.pt").write_bytes(b"same-bytes")
    displaced = tmp_path / "displaced"
    staging.rename(displaced)
    staging.mkdir()
    (staging / "model.pt").write_bytes(b"same-bytes")
    with pytest.raises(MolecularGNNResumeError, match="directory identity changed"):
        workspace.mark_ready()
    workspace.close()
    parent.close()


def test_partial_finalization_cleanup_is_state_receipted_and_bounded(tmp_path: Path) -> None:
    _, state, parent, workspace = _finalization_authorities(tmp_path)
    staging, _ = workspace.prepare()
    (staging / "partial.bin").write_bytes(b"partial")
    workspace.close()
    parent.close()

    _, _, resumed_parent, resumed = _finalization_authorities(tmp_path, resume=True)
    staging, ready = resumed.prepare()
    assert ready is False
    assert list(staging.iterdir()) == []
    cleanup = json.loads((state / "finalization_cleanup.json").read_text(encoding="utf-8"))
    assert cleanup["status"] == "CLEANUP_COMPLETE"
    assert cleanup["inventory"][0]["path"] == "partial.bin"
    resumed.close()
    resumed_parent.close()


def test_finalization_cleanup_prepared_crash_preserves_first_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, state, parent, workspace = _finalization_authorities(tmp_path)
    staging, _ = workspace.prepare()
    (staging / "first.bin").write_bytes(b"first")
    (staging / "second.bin").write_bytes(b"second")
    workspace.close()
    parent.close()

    _, _, resumed_parent, interrupted = _finalization_authorities(
        tmp_path, resume=True
    )
    real_clear = resume_module._clear_finalization_directory_fd

    def partial_clear(root_fd: int, *, expected_inventory) -> None:
        os.unlink("first.bin", dir_fd=root_fd)
        os.fsync(root_fd)
        raise RuntimeError("simulated crash after first cleanup unlink")

    monkeypatch.setattr(
        resume_module, "_clear_finalization_directory_fd", partial_clear
    )
    with pytest.raises(RuntimeError, match="first cleanup unlink"):
        interrupted.prepare()
    prepared = json.loads(
        (state / "finalization_cleanup.json").read_text(encoding="utf-8")
    )
    assert prepared["status"] == "CLEANUP_PREPARED"
    assert [row["path"] for row in prepared["inventory"]] == [
        "first.bin",
        "second.bin",
    ]
    interrupted.close()
    resumed_parent.close()

    monkeypatch.setattr(
        resume_module, "_clear_finalization_directory_fd", real_clear
    )
    _, _, final_parent, recovered = _finalization_authorities(tmp_path, resume=True)
    recovered_staging, ready = recovered.prepare()
    assert ready is False
    assert list(recovered_staging.iterdir()) == []
    complete = json.loads(
        (state / "finalization_cleanup.json").read_text(encoding="utf-8")
    )
    assert complete["status"] == "CLEANUP_COMPLETE"
    assert complete["inventory"] == prepared["inventory"]
    assert complete["inventory_sha256"] == prepared["inventory_sha256"]
    recovered.close()
    final_parent.close()


def test_finalization_cleanup_uses_held_dirfd_not_replaced_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, parent, workspace = _finalization_authorities(tmp_path)
    staging, _ = workspace.prepare()
    (staging / "owned.bin").write_bytes(b"owned")
    workspace.close()
    parent.close()

    _, _, resumed_parent, resumed = _finalization_authorities(tmp_path, resume=True)
    real_clear = resume_module._clear_finalization_directory_fd
    displaced = tmp_path / "held-original-staging"

    def replace_named_path(root_fd: int, *, expected_inventory) -> None:
        resumed.staging.rename(displaced)
        resumed.staging.mkdir(mode=0o700)
        (resumed.staging / "foreign.bin").write_bytes(b"must-survive")
        real_clear(root_fd, expected_inventory=expected_inventory)

    monkeypatch.setattr(
        resume_module, "_clear_finalization_directory_fd", replace_named_path
    )
    with pytest.raises(MolecularGNNResumeError, match="directory identity changed"):
        resumed.prepare()
    assert (resumed.staging / "foreign.bin").read_bytes() == b"must-survive"
    assert list(displaced.iterdir()) == []
    resumed.close()
    resumed_parent.close()


def test_output_and_state_paths_reject_symlinked_parent_before_resolution(
    tmp_path: Path,
) -> None:
    physical = tmp_path / "physical-output-parent"
    physical.mkdir()
    alias = tmp_path / "output-alias"
    alias.symlink_to(physical, target_is_directory=True)
    with pytest.raises(MolecularGNNResumeError, match="symlink components"):
        assert_no_symlink_components(alias / "future-output", label="Taste output")


def test_terminal_read_authority_rejects_same_byte_state_root_replacement(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    output = tmp_path / "output"
    contract = _contract(output)
    with MolecularGNNResumeStore(
        state, resume=False, contract=contract, torch_module=torch
    ):
        pass
    authority = MolecularGNNStateReadAuthority(
        state, contract_sha256=canonical_sha256(contract)
    )
    authority.open()
    displaced = tmp_path / "displaced-state"
    state.rename(displaced)
    shutil.copytree(displaced, state, copy_function=shutil.copy2)
    with pytest.raises(MolecularGNNResumeError, match="root inode changed"):
        authority.verify()
    authority.close()


def test_terminal_read_authority_rejects_named_lock_replacement(tmp_path: Path) -> None:
    state = tmp_path / "state"
    output = tmp_path / "output"
    contract = _contract(output)
    with MolecularGNNResumeStore(
        state, resume=False, contract=contract, torch_module=torch
    ):
        pass
    authority = MolecularGNNStateReadAuthority(
        state, contract_sha256=canonical_sha256(contract)
    )
    authority.open()
    lock = state / ".writer.lock"
    lock.unlink()
    lock.write_bytes(b"replacement")
    with pytest.raises(MolecularGNNResumeError, match="lock inode changed"):
        authority.verify()
    authority.close()


def test_uninterrupted_and_crash_resume_trajectories_are_exactly_equal(
    tmp_path: Path,
) -> None:
    def trajectory(root: Path, *, crash_after_first: bool):
        random.seed(113)
        np.random.seed(113)
        torch.manual_seed(113)
        model, optimizer = _model_optimizer()
        contract = _contract(tmp_path / "trajectory-output")
        store = MolecularGNNResumeStore(
            root, resume=False, contract=contract, torch_module=torch
        )
        store.open()
        _advance(model, optimizer)
        _save_epoch(store, epoch=1, model=model, optimizer=optimizer)
        if crash_after_first:
            store.close()
            model, optimizer = _model_optimizer()
            store = MolecularGNNResumeStore(
                root, resume=True, contract=contract, torch_module=torch
            )
            store.open()
            snapshot = store.load(model=model, optimizer=optimizer)
            assert snapshot is not None and snapshot.next_epoch == 2
        _advance(model, optimizer)
        _save_epoch(store, epoch=2, model=model, optimizer=optimizer)
        result = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        random_probe = random.random()
        numpy_probe = float(np.random.random())
        torch_probe = torch.rand(4)
        store.close()
        return result, random_probe, numpy_probe, torch_probe

    uninterrupted = trajectory(tmp_path / "uninterrupted", crash_after_first=False)
    resumed = trajectory(tmp_path / "resumed", crash_after_first=True)
    for key in uninterrupted[0]:
        assert torch.equal(uninterrupted[0][key], resumed[0][key])
    assert uninterrupted[1] == resumed[1]
    assert uninterrupted[2] == resumed[2]
    assert torch.equal(uninterrupted[3], resumed[3])
