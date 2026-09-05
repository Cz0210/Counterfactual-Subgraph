"""Publish already-complete BACE CPU bundles on the scoped HPC filesystem.

This is a publication-only repair for renameat2(RENAME_NOREPLACE) EINVAL.
It neither invokes the trainer nor evaluates/refits a classifier.  The original
parent and training-store locks, staging claim, sealed inventory and inode are
retained.  POSIX rename is used only while both existing authorities are held
and the destination is absent; competing cooperating writers cannot enter.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from src.ablations.gnn.cpu_training import (
    TRAINED_BACKBONES, bundle_file, file_sha256, load_bundle,
)
from src.train.molecular_gnn_resume import (
    FinalizationWorkspace, MolecularGNNResumeStore, OutputParentAuthority,
    _atomic_json_noclobber, canonical_sha256,
)


HPC_GNN_SCOPE = Path("/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/gnn")


def scoped_path(value: str | Path, *, must_exist: bool = False) -> Path:
    path = Path(os.path.abspath(value))
    if path == HPC_GNN_SCOPE or not path.is_relative_to(HPC_GNN_SCOPE):
        raise ValueError("Publication writes are restricted to dedicated HPC GNN roots")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError("Publication path may not contain symlinks")
    if must_exist and not path.is_dir():
        raise ValueError("Existing dedicated GNN directory required")
    return path


def read_json(path: Path) -> dict[str, Any]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Publication evidence must be one physical regular file")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Publication evidence must be a JSON object")
    return payload


def _sealed_checkpoint(attempt: Path, store: MolecularGNNResumeStore, torch: Any) -> tuple[dict, dict]:
    latest = read_json(attempt / "training_state/latest_checkpoint.json")
    name = latest.get("checkpoint_file")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Invalid sealed checkpoint filename")
    path = attempt / "training_state" / name
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or latest.get("status") != "CHECKPOINT_COMPLETE"
            or latest.get("contract_sha256") != store.contract_sha256
            or latest.get("training_contract_evidence") != store.contract_evidence
            or latest.get("checkpoint_bytes") != info.st_size
            or latest.get("checkpoint_sha256") != file_sha256(path)):
        raise ValueError("Sealed final checkpoint identity mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    epoch = state.get("completed_epoch")
    if (state.get("contract_sha256") != store.contract_sha256
            or state.get("training_contract_evidence") != store.contract_evidence
            or epoch != latest.get("completed_epoch") or state.get("next_epoch") != epoch + 1
            or len(state.get("history", [])) != epoch or not state.get("best_state")):
        raise ValueError("Final checkpoint scientific state is incomplete")
    training = store.contract["training"]
    if (epoch < training["max_epochs"]
            and state["epochs_without_improvement"] < training["early_stopping_patience"]):
        raise ValueError("Science is not terminal: neither epoch ceiling nor validation early stop")
    return latest, state


def _verify_science(directory: Path, backbone: str, store: MolecularGNNResumeStore,
                    state: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    from src.oracles.gnn_oracle import load_gnn_checkpoint_bundle, verify_checkpoint_bundle
    audit = verify_checkpoint_bundle(directory)
    card = audit["model_card"]
    expected = {"dataset": "bace", "backbone": backbone, "seed": 7, "num_classes": 2,
                "source_label": 1, "selection_split": "validation",
                "training_resume_contract_sha256": store.contract_sha256,
                "calibration_used_for_model_fit_or_selection": False,
                "test_used_for_model_fit_or_selection": False,
                "test_loaded_during_training": False, "test_evaluated_during_training": False}
    if any(card.get(key) != value for key, value in expected.items()):
        raise ValueError("Staged classifier disagrees with BACE training/selection contract")
    metrics = read_json(directory / "training_metrics.json")
    if (metrics.get("best_epoch") != state["best_epoch"]
            or metrics.get("epochs_completed") != state["completed_epoch"]
            or metrics.get("history") != state["history"]
            or metrics.get("selection_metric") != store.contract["training"]["selection_metric"]):
        raise ValueError("Staged metrics disagree with terminal validation-selected checkpoint")
    split = read_json(directory / "split_manifest.json")
    if split.get("files") != store.contract["splits"] or any(
        split.get(key) is not False for key in (
            "calibration_loaded_for_training", "test_loaded_for_training",
            "test_evaluated_during_training", "test_used_for_checkpoint_selection")):
        raise ValueError("Staged split isolation does not close")
    model, metadata = load_gnn_checkpoint_bundle(directory, device="cpu")
    if (model.config.to_dict() != store.contract["model_config"]
            or metadata["feature_schema"].to_dict() != store.contract["feature_schema"]):
        raise ValueError("Staged architecture/feature schema differs from original training")
    weights = model.state_dict()
    if set(weights) != set(state["best_state"]) or any(
        not torch.equal(value, state["best_state"][name])
        or (value.is_floating_point() and not torch.isfinite(value).all())
        for name, value in weights.items()
    ):
        raise ValueError("Staged model does not equal the validation-selected best weights")
    temperature = float(metadata["temperature_scaling"]["temperature"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Previously fitted temperature is invalid")
    return audit


def finalize_one(*, bundle_root: Path, manifest: Mapping[str, Any], backbone: str,
                 classifier_root: str | Path) -> dict[str, Any]:
    """Adopt a complete stage or idempotently verify an already-published one."""
    import torch
    if backbone not in TRAINED_BACKBONES:
        raise ValueError("Reference GINE must never pass through staged publication")
    output = scoped_path(classifier_root)
    if output.name != "classifier":
        raise ValueError("Only the existing dedicated attempt/classifier output is allowed")
    attempt = scoped_path(output.parent, must_exist=True)
    training_root = attempt / "training_state"
    envelope = read_json(training_root / "training_contract.json")
    contract = envelope["contract"]
    cpu = read_json(attempt / "cpu_contract.json")
    if (envelope["contract_sha256"] != canonical_sha256(contract)
            or contract.get("dataset") != "bace" or contract.get("output_dir") != str(output)
            or contract["training"].get("seed") != 7
            or contract["model_config"].get("backbone") != backbone
            or cpu.get("backbone") != backbone or cpu.get("device") != "cpu"
            or cpu.get("bundle_manifest_sha256") != file_sha256(bundle_root / "bundle_manifest.json")
            or cpu.get("effective_config_sha256") != file_sha256(attempt / "effective_config.yaml")):
        raise ValueError("Original BACE CPU training contract binding differs")
    for name, relative in manifest["splits"].items():
        path = bundle_file(bundle_root, manifest, relative)
        if contract["splits"].get(name) != {"path": str(path), "sha256": file_sha256(path)}:
            raise ValueError("Original split input binding differs")
    parent = OutputParentAuthority(output, contract_sha256=envelope["contract_sha256"], resume=True)
    store = MolecularGNNResumeStore(training_root, resume=True, contract=contract, torch_module=torch)
    workspace = FinalizationWorkspace(output, contract_sha256=store.contract_sha256,
        resume=True, parent_authority=parent, training_state_root=training_root)
    try:
        # Original lock ordering. A healthy training/finalization owner blocks us.
        parent.open()
        store.open()
        latest, state = _sealed_checkpoint(attempt, store, torch)
        # Never let prepare() enter its incomplete-stage cleanup/rebuild branch.
        ready = read_json(workspace.ready_path)
        read_json(workspace.claim_path)
        if ready.get("status") != "BUNDLE_COMPLETE":
            raise ValueError("Only a pre-existing BUNDLE_COMPLETE stage can be adopted")
        renamed = False
        if output.exists():
            closure = workspace.verify_published()
            audit = _verify_science(output, backbone, store, state, torch)
        else:
            if not workspace.staging.is_dir():
                raise ValueError("No complete staged or published bundle exists")
            stage, complete = workspace.prepare()
            if not complete:
                raise ValueError("Incomplete stage may not be rebuilt by publication repair")
            audit = _verify_science(stage, backbone, store, state, torch)
            workspace.mark_ready()  # Verify existing inventory; no resealing/rewrite.
            parent.verify()
            store.verify_writer_authority()
            try:
                os.stat(output.name, dir_fd=parent.directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError("Immutable destination already exists; never overwrite")
            # HPC /share lacks renameat2 NOREPLACE. Both original exclusive locks
            # prevent a cooperating publisher racing this exact absent destination.
            os.rename(stage.name, output.name, src_dir_fd=parent.directory_fd,
                      dst_dir_fd=parent.directory_fd)
            os.fsync(parent.directory_fd)
            renamed = True
            closure = workspace.verify_published()  # Original inode+hash closure.
            from src.oracles.gnn_oracle import verify_checkpoint_bundle
            audit = verify_checkpoint_bundle(output)
        identity = {
            "model_sha256": file_sha256(output / "model.pt"),
            "model_card_sha256": file_sha256(output / "model_card.json"),
            "sha256s_sha256": file_sha256(output / "sha256sums.txt"),
            "checkpoint_id": audit["model_card"]["checkpoint_id"],
            "training_resume_contract_sha256": store.contract_sha256,
            "finalization_claim_sha256": closure["claim_sha256"],
            "finalization_completion_sha256": closure["completion_sha256"],
        }
        completion = store.mark_complete(output_dir=output, output_identity=identity)
        receipt = {
            "schema_version": "bace_gnn_staged_publication_v1", "status": "TRAINING_PASS",
            "backbone": backbone, "classifier_root": str(output),
            "training_contract_sha256": store.contract_sha256,
            "scientific_engine_commit": contract["source_identity"]["commit"],
            "publication_implementation_sha256": file_sha256(Path(__file__)),
            "input_bundle_manifest_sha256": cpu["bundle_manifest_sha256"],
            "completed_epoch": state["completed_epoch"], "best_epoch": state["best_epoch"],
            "final_checkpoint_sha256": latest["checkpoint_sha256"],
            "model_sha256": identity["model_sha256"], "finalization_closure": closure,
            "training_complete_status": completion["status"],
            "publication_only": True, "science_rerun": False, "temperature_refit": False,
            "sealed_bundle_bytes_modified": False, "staging_inode_preserved": True,
            "rename_fallback": "POSIX_SAME_PARENT_FD_UNDER_ORIGINAL_EXCLUSIVE_LOCKS",
            "publication_repair_reason": "HPC_RENAMEAT2_RENAME_NOREPLACE_EINVAL",
            "rename_performed_this_invocation": renamed,
            "elapsed_seconds": None, "cpu_seconds": None,
            "training_resource_measurement": "UNKNOWN_NOT_INFERRED_FROM_PUBLICATION_TIME",
            "calibration_split_loaded": False, "test_split_loaded": False,
            "main_matrix_write": False,
        }
        terminal = attempt / "training_terminal.json"
        if terminal.exists():
            previous = read_json(terminal)
            if (previous.get("status") != "TRAINING_PASS"
                    or previous.get("backbone") != backbone):
                raise ValueError("Conflicting pre-existing training terminal; preserved")
        else:
            _atomic_json_noclobber(terminal, receipt)
        return receipt
    finally:
        workspace.close()
        store.close()
        parent.close()


def finalize_models(*, bundle_root: str | Path, model_roots: Mapping[str, str],
                    output_root: str | Path) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}:
        raise ValueError("CPU-only publication must not receive a visible GPU")
    root, manifest = load_bundle(bundle_root)
    if set(model_roots) != {"gine", *TRAINED_BACKBONES}:
        raise ValueError("Exactly five BACE classifier roots are required")
    if Path(model_roots["gine"]) != root / manifest["gine_reference_root"]:
        raise ValueError("GINE reference root cannot be replaced")
    output = scoped_path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    results, errors = {}, {}
    for name in TRAINED_BACKBONES:
        try:
            result = finalize_one(bundle_root=root, manifest=manifest, backbone=name,
                                  classifier_root=model_roots[name])
            results[name] = result
            _atomic_json_noclobber(output / f"{name}.json", result)
        except Exception as exc:
            errors[name] = {"exception": type(exc).__name__, "message": str(exc)}
            _atomic_json_noclobber(output / f"{name}.failed.json", errors[name])
    summary = {"schema_version": "bace_gnn_staged_publication_campaign_v1",
        "status": "PASS" if not errors else "BLOCKED", "results": results, "errors": errors,
        "model_roots": dict(model_roots), "science_rerun": False, "main_matrix_write": False}
    _atomic_json_noclobber(output / "finalization_receipt.json", summary)
    return summary
