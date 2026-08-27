from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save as save_safetensors

import src.train.tastemolnet_clean_policy_init as t5
from src.train.tastemolnet_clean_policy import (
    hold_clean_policy_load_authority as stable_hold_clean_policy_load_authority,
    hold_clean_policy_output as stable_hold_clean_policy_output,
    validate_clean_policy_output as stable_validate_clean_policy_output,
)
from src.utils.autodl_tastemolnet_main_v1 import _queue
from src.utils.tastemolnet_research_policy import (
    NO_REDISTRIBUTION_MARKER,
    POLICY_V2_AUDIT_MARKER,
    load_tastemolnet_research_policy,
)


REPOSITORY = Path(__file__).resolve().parents[1]
POLICY = REPOSITORY / "configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml"


def _json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _tree_snapshot(path: Path) -> dict[str, str]:
    return {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.iterdir())
        if item.is_file()
    }


def _tree_sha(path: Path) -> str:
    return t5._canonical_sha256(  # noqa: SLF001
        {"schema_version": "test_taste_stage_root_v1", "files": _tree_snapshot(path)}
    )


class _FakeHeldStage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.descriptor = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        self.directory = SimpleNamespace(descriptor=self.descriptor)
        self.snapshot = _tree_snapshot(self.root)
        gate = json.loads((self.root / "gate.json").read_text(encoding="utf-8"))
        self.evidence = {
            "stage": gate["stage"],
            "gate_sha256": hashlib.sha256((self.root / "gate.json").read_bytes()).hexdigest(),
            "root_inventory_sha256": _tree_sha(self.root),
            "checkpoint_dir": gate["checkpoint_dir"],
            "checkpoint_id": gate["checkpoint_id"],
            "checkpoint_inventory_sha256": gate["checkpoint_inventory_sha256"],
            "checkpoint_stat_inventory_sha256": gate["checkpoint_stat_inventory_sha256"],
            "checkpoint_sha256s_sha256": gate["checkpoint_sha256s_sha256"],
            "t2_adoption_gate_sha256": gate["t2_adoption_binding"]["gate_sha256"],
            "t2_adoption_receipt_sha256": gate["t2_adoption_binding"]["receipt_sha256"],
            "t2_adoption_binding_sha256": t5._canonical_sha256(  # noqa: SLF001
                gate["t2_adoption_binding"]
            ),
        }

    def revalidate(self) -> dict[str, str]:
        if self.descriptor < 0 or _tree_snapshot(self.root) != self.snapshot:
            raise RuntimeError("fake held stage changed")
        return dict(self.evidence)

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


class _FakeHeldCheckpoint:
    def __init__(self, root: Path, evidence: dict[str, str]) -> None:
        self.root = root.resolve()
        self.evidence = dict(evidence)
        self.snapshot = _tree_snapshot(self.root)

    def revalidate(self) -> dict[str, str]:
        if _tree_snapshot(self.root) != self.snapshot:
            raise RuntimeError("fake held checkpoint changed")
        return dict(self.evidence)

    def read_frozen_gine_payload(self, name: str) -> bytes:
        self.revalidate()
        return (self.root / name).read_bytes()

    def close(self) -> None:
        pass


class _FakeHeldT2Adoption:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.binding = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )

    def revalidate(self) -> dict[str, object]:
        return json.loads(json.dumps(self.binding))

    def close(self) -> None:
        pass


def _patch_gnn_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def hold_stage(root: str | Path) -> _FakeHeldStage:
        return _FakeHeldStage(Path(root))

    def hold_checkpoint(
        path: str | Path, *, expected_stage_evidence: dict[str, str]
    ) -> _FakeHeldCheckpoint:
        assert str(Path(path).resolve()) == expected_stage_evidence["checkpoint_dir"]
        return _FakeHeldCheckpoint(Path(path), expected_stage_evidence)

    def hold_t2(
        root: str | Path,
        *,
        expected_gate_sha256: str,
        expected_receipt_sha256: str,
        expected_source_evidence_sha256: str,
    ) -> _FakeHeldT2Adoption:
        held = _FakeHeldT2Adoption(Path(root))
        assert held.binding["gate_sha256"] == expected_gate_sha256
        assert held.binding["receipt_sha256"] == expected_receipt_sha256
        assert (
            held.binding["source_evidence_sha256"]
            == expected_source_evidence_sha256
        )
        return held

    monkeypatch.setattr(t5, "_load_taste_gnn_stage_api", lambda: (hold_stage, hold_checkpoint))
    monkeypatch.setattr(t5, "hold_t2_gine_pass_adoption", hold_t2)


def _valid_lora_tensors() -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for index, target in enumerate(("wqkv", "wo", "w1", "w2", "w3"), start=1):
        prefix = f"base_model.model.layers.0.{target}"
        tensors[f"{prefix}.lora_A.weight"] = torch.full(
            (8, 4 + index), float(index), dtype=torch.float32
        )
        tensors[f"{prefix}.lora_B.weight"] = torch.zeros(
            (6 + index, 8), dtype=torch.float32
        )
    return tensors


def _fake_materialize_adapter(**kwargs: object) -> dict[str, object]:
    adapter_fd = int(kwargs["adapter_fd"])
    source = kwargs["source_authority"]
    config = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 8,
        "lora_alpha": 16, "lora_dropout": 0.05, "bias": "none",
        "target_modules": ["wqkv", "wo", "w1", "w2", "w3"],
        "inference_mode": True, "init_lora_weights": True,
        "modules_to_save": None, "rank_pattern": {}, "alpha_pattern": {},
        "use_dora": False, "use_rslora": False, "fan_in_fan_out": False,
        "layers_to_transform": None, "layers_pattern": None,
        "layer_replication": None, "lora_bias": False,
        "target_parameters": None,
        "base_model_name_or_path": str(source.source_model_dir),
    }
    t5._write_new_at(  # noqa: SLF001
        adapter_fd,
        "adapter_config.json",
        (json.dumps(config, indent=2, sort_keys=True) + "\n").encode(),
    )
    t5._write_new_at(  # noqa: SLF001
        adapter_fd, "adapter_model.safetensors", save_safetensors(_valid_lora_tensors())
    )
    inventory, digest, tensor_identity = t5._validate_adapter_directory_fd(  # noqa: SLF001
        adapter_fd, expected_source_model_path=source.source_model_dir
    )
    return {
        "adapter_inventory": inventory,
        "adapter_inventory_sha256": digest,
        "adapter_tensor_identity": tensor_identity,
        "peft_reload_verified": True,
    }


def _gpu_identity(
    authority: t5.TasteCleanPolicyReleaseAuthority,
) -> dict[str, object]:
    return {
        "physical_gpu_index": 2,
        "gpu_uuid": authority.gpu_uuid,
        "cuda_visible_devices": "2",
        "controller_binding_state": "controller_declared_only",
        "gpu_lock_authority_present": False,
        "execution_receipt_present": False,
        "controller_id": authority.controller_id,
        "controller_task_id": authority.controller_task_id,
    }


def _patch_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        t5,
        "_validate_t5_gpu",
        _gpu_identity,
    )


def _patch_publish_for_host(
    monkeypatch: pytest.MonkeyPatch, output_parent: Path
) -> None:
    if sys.platform.startswith("linux"):
        return

    def portable_test_publish(
        _parent_fd: int,
        source: str,
        target: str,
        *,
        fsync_parent: bool = True,
    ) -> None:
        del fsync_parent
        if source == ".PASS.prepared":
            candidates = [
                child
                for child in output_parent.iterdir()
                if child.is_dir() and (child / source).is_file()
            ]
            if len(candidates) != 1:
                raise AssertionError("portable marker publication is ambiguous")
            directory = candidates[0]
        else:
            directory = output_parent
        if (directory / target).exists():
            raise FileExistsError(target)
        os.rename(directory / source, directory / target)

    monkeypatch.setattr(t5, "_renameat2_noreplace", portable_test_publish)


def _patch_unreleased_producer_for_test(
    monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """Exercise the producer internals without weakening the public release gate."""

    config = t5.load_clean_policy_config(config_path)
    enabled = replace(config, tracked_release_enabled=True)
    monkeypatch.setattr(t5, "load_clean_policy_config", lambda _path: enabled)


def _config(path: Path, output_parent: Path) -> None:
    _json(
        path,
        {
            "schema_version": t5.CONFIG_SCHEMA,
            "dataset": "tastemolnet",
            "stage": t5.STAGE,
            "initializer_mode": t5.INITIALIZER_MODE,
            "tracked_release_enabled": False,
            "tracked_release_state": (
                "RELEASE_DISABLED_PENDING_FINAL_T3_T4_SOURCE_EXECUTION_RECEIPT"
            ),
            "external_sha_pinned_release_authority_required": True,
            "output_parent": str(output_parent),
            "fresh_output_required": True,
            "private_output": True,
            "initializer_data_split_used": "none",
            "taste_split_access_max": "train_only",
            "train_only_fallback_implemented": False,
            "rf_reference_count": 0,
            "gnn_reward_used": False,
            "validation_loaded": False,
            "calibration_loaded": False,
            "test_loaded": False,
            "data_redistribution_allowed": False,
            "hpc_execution_allowed": False,
            "required_pass_marker": t5.PASS_MARKER,
        },
    )


def _policy_receipt(root: Path) -> tuple[Path, str]:
    policy = load_tastemolnet_research_policy(POLICY)
    private = {
        "schema_version": "tastemolnet_existing_private_data_authority_v1",
        "prepared_root": str(root / "private" / "prepared"),
        "graph_cache_root": str(root / "private" / "cache"),
        "provenance_manifest_sha256": "1" * 64,
        "prepared_output_manifest_sha256": "2" * 64,
        "split_manifest_sha256": "3" * 64,
        "graph_cache_manifest_sha256": "4" * 64,
        "source_csv_sha256": "5" * 64,
        "prepared_rows": 13421,
        "split_rows": {
            "train": 9437,
            "validation": 1328,
            "calibration": 1328,
            "test": 1328,
        },
        "graph_cache_rows": 13421,
        "data_reprepared": False,
        "graph_cache_rebuilt": False,
        "cache_payloads_deserialized_by_audit": False,
        "test_rows_deserialized_by_audit": False,
    }
    receipt_root = root / "receipt"
    receipt_root.mkdir()
    receipt = receipt_root / "receipt.json"
    receipt_hash = _json(
        receipt,
        {
            "schema_version": "tastemolnet_research_reporting_policy_receipt_v2",
            "created_at": "2026-08-28T00:00:00Z",
            "dataset": "tastemolnet",
            "status": policy.status,
            "authorization_state": policy.authorization_state,
            "authorization_status": policy.authorization_status,
            "policy": policy.evidence(),
            "private_data_authority": private,
            "run_tastemolnet": 1,
            "heavy_route_authorized": True,
            "paper_reporting_authorized": True,
            "dataset_redistribution_authorized": False,
            "upstream_terms_status": "NOT_EXPLICITLY_STATED",
            "license_conclusion": "NOT_GRANTED_OR_INFERRED",
            "hpc_execution_authorized": False,
            "data_reprepared": False,
            "graph_cache_rebuilt": False,
            "terminal_marker": POLICY_V2_AUDIT_MARKER,
            "no_redistribution_marker": NO_REDISTRIBUTION_MARKER,
        },
    )
    (receipt_root / POLICY_V2_AUDIT_MARKER).write_text(
        POLICY_V2_AUDIT_MARKER + "\n", encoding="utf-8"
    )
    (receipt_root / NO_REDISTRIBUTION_MARKER).write_text(
        NO_REDISTRIBUTION_MARKER + "\n", encoding="utf-8"
    )
    (receipt_root / "tastemolnet_policy_audit.md").write_text(
        "aggregate-only policy receipt\n", encoding="utf-8"
    )
    return receipt, receipt_hash


def _release_fixture(tmp_path: Path) -> dict[str, object]:
    output_parent = tmp_path / "runtime" / "outputs" / "autodl" / "tastemolnet" / "clean-policy-initializer"
    output_parent.mkdir(parents=True, mode=0o700)
    config = tmp_path / "t5-config.json"
    _config(config, output_parent)

    model = tmp_path / "ChemLLM-7B-Chat"
    _json(model / "config.json", {"model_type": "internlm2", "_name_or_path": "ChemLLM-7B-Chat"})
    (model / "model.safetensors").write_bytes(b"generic-base-model")
    source = t5.inspect_generic_chemllm_base(model)
    receipt, receipt_hash = _policy_receipt(tmp_path)
    checkpoint = tmp_path / "gnn" / "seed7"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.pt").write_bytes(b"frozen-three-class-gine")
    (checkpoint / "last.pt").write_bytes(b"terminal-three-class-gine")
    (checkpoint / "config.yaml").write_text("backbone: gine\n", encoding="utf-8")
    _json(checkpoint / "feature_schema.json", {"schema_version": "taste-feature-v1"})
    _json(checkpoint / "label_map.json", t5.LABEL_MAP)
    _json(checkpoint / "temperature_scaling.json", {"temperature": 1.25})
    model_sha = hashlib.sha256((checkpoint / "model.pt").read_bytes()).hexdigest()
    last_sha = hashlib.sha256((checkpoint / "last.pt").read_bytes()).hexdigest()
    config_sha = hashlib.sha256((checkpoint / "config.yaml").read_bytes()).hexdigest()
    feature_sha = hashlib.sha256((checkpoint / "feature_schema.json").read_bytes()).hexdigest()
    label_map_sha = hashlib.sha256((checkpoint / "label_map.json").read_bytes()).hexdigest()
    temperature_sha = hashlib.sha256(
        (checkpoint / "temperature_scaling.json").read_bytes()
    ).hexdigest()
    inventory_sha = "6" * 64
    stat_inventory_sha = "7" * 64
    sha_inventory_sha = "8" * 64
    downstream_policy_sha = "9" * 64
    adoption_root = (
        tmp_path
        / "runtime"
        / "control"
        / "tastemolnet-t2-gine-pass-adoption-v1"
        / "tastemolnet-gine-v2-20260827T160838Z-274631"
    )
    adoption_root.mkdir(parents=True, mode=0o700)
    formal_inventory = [{"path": "model.pt", "sha256": model_sha}]
    t2_binding = {
        "schema_version": t5.DOWNSTREAM_BINDING_SCHEMA,
        "stage": "T2_GINE_FULL",
        "status": "PASS",
        "state": t5.ADOPTION_MARKER,
        "source_cid": "tastemolnet-gine-v2-20260827T160838Z-274631",
        "source_run_id": "tastemolnet-t2-gine-full-20260827T161802Z-698faeec",
        "adoption_root": str(adoption_root.resolve()),
        "adoption_root_inventory_sha256": "0" * 64,
        "gate_path": str((adoption_root / "gate.json").resolve()),
        "gate_sha256": "1" * 64,
        "receipt_path": str((adoption_root / "manifest.json").resolve()),
        "receipt_sha256": "2" * 64,
        "source_evidence_sha256": "3" * 64,
        "formal_bundle_root": str(checkpoint.resolve()),
        "formal_bundle_inventory": formal_inventory,
        "formal_bundle_inventory_sha256": t5._canonical_sha256(  # noqa: SLF001
            formal_inventory
        ),
        "formal_bundle_model_sha256": model_sha,
        "formal_bundle_sha256s_sha256": sha_inventory_sha,
    }
    _json(adoption_root / "gate.json", {"status": "PASS"})
    _json(adoption_root / "manifest.json", t2_binding)

    t3 = tmp_path / "gates" / "t3"
    t4 = tmp_path / "gates" / "t4"
    t3.mkdir(parents=True)
    t4.mkdir(parents=True)
    t3_gate = {
        "schema_version": "tastemolnet_main_stage_gate_v1",
        "stage": "T3_GINE_CALIBRATED",
        "status": "PASS",
        "marker": "TASTE_GINE_CALIBRATION_PASS",
        "depends_on": [t5.ADOPTION_MARKER],
        "t2_science_bundle_verified": True,
        "t2_adoption_binding": t2_binding,
        "checkpoint_dir": str(checkpoint.resolve()),
        "checkpoint_id": model_sha,
        "checkpoint_inventory_sha256": inventory_sha,
        "checkpoint_stat_inventory_sha256": stat_inventory_sha,
        "checkpoint_sha256s_sha256": sha_inventory_sha,
        "downstream_policy_sha256": downstream_policy_sha,
        "existing_fit_adopted": True,
        "temperature_refit_performed": False,
        "test_loaded": False,
    }
    t3_hash = _json(t3 / "gate.json", t3_gate)
    _json(
        t3 / "oracle_reference.json",
        {
            "schema_version": "tastemolnet_t3_oracle_reference_v1",
            "dataset": "tastemolnet",
            "checkpoint_id": model_sha,
            "selected_inference_asset": "model.pt",
            "model_sha256": model_sha,
            "last_checkpoint_terminal_only": True,
            "last_sha256": last_sha,
            "temperature_scaling_sha256": temperature_sha,
            "config_sha256": config_sha,
            "feature_schema_sha256": feature_sha,
            "label_map_sha256": label_map_sha,
            "num_classes": 3,
            "source_label": 1,
            "rf_oracle_used": False,
            "t2_adoption_binding": t2_binding,
        },
    )
    t4_gate = {
        "schema_version": "tastemolnet_main_stage_gate_v1",
        "stage": "T4_ORACLE_SMOKE",
        "status": "PASS",
        "marker": "TASTE_MULTICLASS_ORACLE_PASS",
        "depends_on": ["T3_GINE_CALIBRATED"],
        "t3_gate_sha256": t3_hash,
        "checkpoint_dir": str(checkpoint.resolve()),
        "checkpoint_id": model_sha,
        "checkpoint_inventory_sha256": inventory_sha,
        "checkpoint_stat_inventory_sha256": stat_inventory_sha,
        "checkpoint_sha256s_sha256": sha_inventory_sha,
        "physical_gpu_index": 1,
        "gpu_uuid": "GPU-22222222-2222-2222-2222-222222222222",
        "visible_device": "cuda:0",
        "cuda_visible_devices": "1",
        "downstream_policy_sha256": downstream_policy_sha,
        "selected_count": 16,
        "calibration_payload_loaded": True,
        "test_loaded": False,
        "per_example_predictions_written": False,
        "t2_adoption_binding": t2_binding,
    }
    t4_hash = _json(t4 / "gate.json", t4_gate)
    _json(
        t4 / "oracle_provenance.json",
        {
            "schema_version": "tastemolnet_t4_oracle_provenance_v1",
            "dataset": "tastemolnet",
            "checkpoint_dir": str(checkpoint.resolve()),
            "checkpoint_id": model_sha,
            "checkpoint_inventory_sha256": inventory_sha,
            "checkpoint_stat_inventory_sha256": stat_inventory_sha,
            "checkpoint_sha256s_sha256": sha_inventory_sha,
            "checkpoint_payload_files_opened": list(t5._T4_CHECKPOINT_PAYLOAD_FILES),  # noqa: SLF001
            "checkpoint_csv_payload_opened": False,
            "selected_inference_asset": "model.pt",
            "model_sha256": model_sha,
            "temperature_scaling_sha256": temperature_sha,
            "config_sha256": config_sha,
            "feature_schema_sha256": feature_sha,
            "physical_gpu_index": 1,
            "gpu_uuid": "GPU-22222222-2222-2222-2222-222222222222",
            "visible_device": "cuda:0",
            "cuda_visible_devices": "1",
            "checkpoint_load_count": 1,
            "rf_oracle_used": False,
            "test_loaded": False,
            "t2_adoption_binding": t2_binding,
        },
    )
    policy = load_tastemolnet_research_policy(POLICY)
    commit = "a" * 40
    tree = "b" * 40
    authority = tmp_path / "release-authority.json"
    authority_hash = _json(
        authority,
        {
            "schema_version": t5.RELEASE_AUTHORITY_SCHEMA,
            "authority_id": "taste-t5-test-authority",
            "created_at": "2026-08-28T00:00:00Z",
            "dataset": "tastemolnet",
            "stage": t5.STAGE,
            "status": "PASS",
            "release_enabled": False,
            "initializer_mode": t5.INITIALIZER_MODE,
            "initializer_data_split_used": "none",
            "taste_split_access_max": "train_only",
            "train_only_fallback_authorized": False,
            "policy_path": str(POLICY),
            "policy_file_sha256": policy.file_sha256,
            "policy_canonical_sha256": policy.canonical_sha256,
            "policy_receipt_path": str(receipt),
            "policy_receipt_sha256": receipt_hash,
            "source_model_path": str(model.resolve()),
            "source_model_inventory_sha256": source["source_model_inventory_sha256"],
            "source_model_classification": "CLEAN_CHEMLLM_BASE",
            "source_model_dataset_specific": False,
            "source_adapter_required": False,
            "source_adapter_path": None,
            "source_adapter_sha256": None,
            "project_root": str(REPOSITORY),
            "implementation_commit": commit,
            "implementation_tree": tree,
            "controller_id": "tastemolnet-main-v1-test",
            "controller_task_id": "T5_CLEAN_POLICY_READY-test",
            "physical_gpu_index": 2,
            "gpu_uuid": "GPU-11111111-1111-1111-1111-111111111111",
            "cuda_visible_devices": "2",
            "controller_binding_state": "controller_declared_only",
            "gpu_lock_authority_present": False,
            "execution_receipt_present": False,
            "frozen_oracle": {
                "dataset": "tastemolnet",
                "backbone": "gine",
                "num_classes": 3,
                "label_map": t5.LABEL_MAP,
                "source_label": 1,
                "strict_flip": "pred_before == 1 and pred_after != 1",
                "rf_oracle_used": False,
                "checkpoint_dir": str(checkpoint.resolve()),
                "checkpoint_id": model_sha,
                "checkpoint_sha256": model_sha,
                "checkpoint_inventory_sha256": inventory_sha,
                "checkpoint_stat_inventory_sha256": stat_inventory_sha,
                "checkpoint_sha256s_sha256": sha_inventory_sha,
                "feature_schema_sha256": feature_sha,
                "temperature_calibration_sha256": temperature_sha,
                "downstream_policy_sha256": downstream_policy_sha,
                "t2_adoption_binding": t2_binding,
                "t3_output_root": str(t3.resolve()),
                "t3_gate_sha256": t3_hash,
                "t3_root_inventory_sha256": _tree_sha(t3),
                "t4_output_root": str(t4.resolve()),
                "t4_gate_sha256": t4_hash,
                "t4_root_inventory_sha256": _tree_sha(t4),
            },
            "rf_reference_count": 0,
            "gnn_reward_used": False,
            "validation_loaded": False,
            "calibration_loaded": False,
            "test_loaded": False,
            "data_redistribution_allowed": False,
            "hpc_execution_allowed": False,
        },
    )
    return {
        "config": config,
        "output_parent": output_parent,
        "model": model,
        "receipt": receipt,
        "authority": authority,
        "authority_hash": authority_hash,
        "commit": commit,
        "tree": tree,
        "t3": t3,
        "t4": t4,
        "checkpoint": checkpoint,
    }


def test_tracked_contract_is_explicitly_disabled_and_train_is_only_a_maximum(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "private-output"
    parent.mkdir(mode=0o700)
    config_path = tmp_path / "config.json"
    _config(config_path, parent)
    config = t5.load_clean_policy_config(config_path)
    assert config.tracked_release_enabled is False
    assert config.tracked_release_state == (
        "RELEASE_DISABLED_PENDING_FINAL_T3_T4_SOURCE_EXECUTION_RECEIPT"
    )

    # The existing main-controller queue previously and incorrectly advertised
    # validation/calibration access for the T5 lane.
    queue = _queue(SimpleNamespace(controller_id="taste-main-test"), t2_status="PASS")  # type: ignore[arg-type]
    lane = queue["resource_lanes"]["gpu2_classifier_independent_precompute"]
    assert lane["allowed_splits"] == ["train"]
    assert lane["initializer_data_split_used"] == "none"
    assert lane["taste_split_access_max"] == "train_only"
    assert lane["t5_release_enabled"] is False
    assert lane["t5_release_state"] == (
        "RELEASE_DISABLED_PENDING_FINAL_T3_T4_SOURCE_EXECUTION_RECEIPT"
    )


def test_public_builder_stops_before_authority_or_model_load_while_release_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "private-output"
    parent.mkdir(mode=0o700)
    config_path = tmp_path / "config.json"
    _config(config_path, parent)

    def forbidden_authority_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("release authority/model path must not open")

    monkeypatch.setattr(t5, "load_release_authority", forbidden_authority_load)
    with pytest.raises(
        t5.TasteCleanPolicyReleaseDisabled,
        match="physical execution receipt is required",
    ):
        t5.build_clean_policy_initializer(
            config_path=config_path,
            release_authority_path=tmp_path / "absent-authority.json",
            expected_release_authority_sha256="0" * 64,
            policy_path=POLICY,
            policy_receipt_path=tmp_path / "absent-receipt.json",
            source_model_path=tmp_path / "absent-model",
            project_root=REPOSITORY,
            output_root=parent / "20260828T000000Z",
        )


def test_generic_base_rejects_bace_rf_and_symlink_sources(tmp_path: Path) -> None:
    base = tmp_path / "ChemLLM-7B-Chat"
    _json(base / "config.json", {"model_type": "internlm2", "_name_or_path": "ChemLLM"})
    (base / "model.safetensors").write_bytes(b"base")
    evidence = t5.inspect_generic_chemllm_base(base)
    assert evidence["initializer_data_split_used"] == "none"
    assert evidence["source_adapter_present"] is False

    (base / "bace_rf_ranked.json").write_text("{}", encoding="utf-8")
    with pytest.raises(t5.TasteCleanPolicyError, match="dataset/RF"):
        t5.inspect_generic_chemllm_base(base)

    target = tmp_path / "physical-ChemLLM"
    target.mkdir()
    linked = tmp_path / "ChemLLM-linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(t5.TasteCleanPolicyError, match="symlink"):
        t5.inspect_generic_chemllm_base(linked)


def test_real_safetensors_tensor_audit_closes_key_shape_dtype_finite_and_zero_step() -> None:
    tensors = _valid_lora_tensors()
    payload = save_safetensors(tensors)
    decoded = t5._load_safetensors_bytes(  # noqa: SLF001
        payload, label="positive real safetensors fixture"
    )
    identity = t5._lora_tensor_identity(  # noqa: SLF001
        decoded, label="positive real safetensors fixture"
    )
    assert identity["tensor_count"] == 10
    assert identity["rank"] == 8
    assert identity["target_modules"] == ["w1", "w2", "w3", "wo", "wqkv"]
    assert identity["all_finite"] is True
    assert identity["all_lora_b_zero"] is True
    assert len(identity["parameter_sha256"]) == 64

    with pytest.raises(t5.TasteCleanPolicyError, match="valid safetensors"):
        t5._load_safetensors_bytes(b"not-safetensors", label="malformed")  # noqa: SLF001

    missing = {name: tensor.clone() for name, tensor in tensors.items()}
    missing.pop("base_model.model.layers.0.wqkv.lora_B.weight")
    with pytest.raises(t5.TasteCleanPolicyError, match="incomplete LoRA A/B pair"):
        t5._lora_tensor_identity(missing, label="missing pair")  # noqa: SLF001

    wrong_key = {name: tensor.clone() for name, tensor in tensors.items()}
    value = wrong_key.pop("base_model.model.layers.0.wqkv.lora_A.weight")
    wrong_key["base_model.model.layers.0.q_proj.lora_A.weight"] = value
    with pytest.raises(t5.TasteCleanPolicyError, match="unreviewed module"):
        t5._lora_tensor_identity(wrong_key, label="wrong key")  # noqa: SLF001

    wrong_rank = {name: tensor.clone() for name, tensor in tensors.items()}
    wrong_rank["base_model.model.layers.0.wqkv.lora_A.weight"] = torch.ones(7, 5)
    with pytest.raises(t5.TasteCleanPolicyError, match="rank differs from 8"):
        t5._lora_tensor_identity(wrong_rank, label="wrong rank")  # noqa: SLF001

    non_finite = {name: tensor.clone() for name, tensor in tensors.items()}
    non_finite["base_model.model.layers.0.wqkv.lora_A.weight"][0, 0] = float("nan")
    with pytest.raises(t5.TasteCleanPolicyError, match="non-finite"):
        t5._lora_tensor_identity(non_finite, label="non-finite")  # noqa: SLF001

    trained = {name: tensor.clone() for name, tensor in tensors.items()}
    trained["base_model.model.layers.0.wqkv.lora_B.weight"][0, 0] = 1.0
    with pytest.raises(t5.TasteCleanPolicyError, match="LoRA B is non-zero"):
        t5._lora_tensor_identity(trained, label="trained adapter")  # noqa: SLF001

    mixed_dtype = {name: tensor.clone() for name, tensor in tensors.items()}
    mixed_dtype["base_model.model.layers.0.wqkv.lora_B.weight"] = mixed_dtype[
        "base_model.model.layers.0.wqkv.lora_B.weight"
    ].to(torch.float64)
    with pytest.raises(t5.TasteCleanPolicyError, match="mixes LoRA tensor dtypes"):
        t5._lora_tensor_identity(mixed_dtype, label="mixed dtype")  # noqa: SLF001


def test_materializer_performs_real_peft_save_and_fresh_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from peft import PeftModel, get_peft_model_state_dict
    from torch import nn
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutput

    import scripts.train_ppo as train_ppo

    class TinyConfig(PretrainedConfig):
        model_type = "tiny_taste_t5"

        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.vocab_size = 11
            self.hidden_size = 4

    class TinyCausalLM(PreTrainedModel):
        config_class = TinyConfig

        def __init__(self, config: TinyConfig) -> None:
            super().__init__(config)
            self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
            self.wqkv = nn.Linear(4, 4, bias=False)
            self.wo = nn.Linear(4, 4, bias=False)
            self.w1 = nn.Linear(4, 4, bias=False)
            self.w2 = nn.Linear(4, 4, bias=False)
            self.w3 = nn.Linear(4, 4, bias=False)
            self.lm_head = nn.Linear(4, config.vocab_size, bias=False)

        def forward(self, input_ids: torch.Tensor | None = None, **_kwargs: object) -> CausalLMOutput:
            assert input_ids is not None
            hidden = self.embed(input_ids)
            hidden = self.w3(self.w2(self.w1(self.wo(self.wqkv(hidden)))))
            return CausalLMOutput(logits=self.lm_head(hidden))

        def prepare_inputs_for_generation(
            self, input_ids: torch.Tensor, **_kwargs: object
        ) -> dict[str, torch.Tensor]:
            return {"input_ids": input_ids}

    class TinyTokenizer:
        def save_pretrained(self, _path: str) -> None:
            return None

    source_root = tmp_path / "ChemLLM-tiny-local"
    _json(
        source_root / "config.json",
        {"model_type": "internlm2", "_name_or_path": "ChemLLM-tiny-local"},
    )
    (source_root / "model.safetensors").write_bytes(b"local-test-base")
    source_evidence = t5.inspect_generic_chemllm_base(source_root)
    adapter_root = tmp_path / "adapter"
    adapter_root.mkdir(mode=0o700)
    adapter_fd = os.open(
        adapter_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    source = t5.hold_source_model_for_clean_policy(
        source_root, source_evidence["source_model_inventory_sha256"]
    )
    try:
        real_from_pretrained = PeftModel.from_pretrained
        reload_calls = 0

        def counted_from_pretrained(
            _class: type[PeftModel], model: nn.Module, *args: object, **kwargs: object
        ) -> PeftModel:
            nonlocal reload_calls
            reload_calls += 1
            return real_from_pretrained(model, *args, **kwargs)

        monkeypatch.setattr(
            PeftModel, "from_pretrained", classmethod(counted_from_pretrained)
        )
        monkeypatch.setattr(
            train_ppo,
            "import_training_dependencies",
            lambda: {"set_seed": torch.manual_seed, "torch": torch},
        )
        monkeypatch.setattr(
            train_ppo,
            "build_tokenizer",
            lambda *_args, **_kwargs: TinyTokenizer(),
        )
        monkeypatch.setattr(
            train_ppo,
            "build_quantized_base_model",
            lambda *_args, **_kwargs: TinyCausalLM(TinyConfig()),
        )
        if not sys.platform.startswith("linux"):
            identities = {
                tuple(os.stat(source_root)[field] for field in (2, 1)): source_root,
                tuple(os.stat(adapter_root)[field] for field in (2, 1)): adapter_root,
            }

            def portable_fd_path(descriptor: int, *, label: str) -> Path:
                info = os.fstat(descriptor)
                try:
                    return identities[(info.st_dev, info.st_ino)]
                except KeyError as exc:
                    raise t5.TasteCleanPolicyError(
                        f"unknown descriptor in {label}"
                    ) from exc

            monkeypatch.setattr(t5, "_fd_directory_path", portable_fd_path)

        materialized = t5._materialize_zero_step_lora(  # noqa: SLF001
            source_authority=source,
            adapter_fd=adapter_fd,
            seed=7,
            rank=8,
            alpha=16,
            dropout=0.05,
        )
        assert materialized["peft_reload_verified"] is True
        assert materialized["adapter_tensor_identity"]["tensor_count"] == 10
        assert materialized["adapter_tensor_identity"]["all_lora_b_zero"] is True
        assert reload_calls == 1
        assert (adapter_root / "adapter_model.safetensors").is_file()
        # The actual loader was exercised above; this import guards against a
        # future replacement with a local name-only double.
        assert callable(get_peft_model_state_dict)
    finally:
        source.close()
        os.close(adapter_fd)


def test_release_authority_is_hash_pinned_and_native_type_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _release_fixture(tmp_path)
    _patch_gnn_api(monkeypatch)
    monkeypatch.setattr(t5, "_git_identity", lambda _root: (fixture["commit"], fixture["tree"]))
    with pytest.raises(t5.TasteCleanPolicyReleaseDisabled, match="SHA-256"):
        t5.load_release_authority(
            fixture["authority"],
            expected_sha256="0" * 64,
            policy_path=POLICY,
            policy_receipt_path=fixture["receipt"],
            source_model_path=fixture["model"],
            project_root=REPOSITORY,
        )

    payload = json.loads(Path(fixture["authority"]).read_text(encoding="utf-8"))
    payload["rf_reference_count"] = False  # bool must not equal integer zero.
    bad_hash = _json(Path(fixture["authority"]), payload)
    with pytest.raises(t5.TasteCleanPolicyReleaseDisabled, match="rf_reference_count"):
        t5.load_release_authority(
            fixture["authority"],
            expected_sha256=bad_hash,
            policy_path=POLICY,
            policy_receipt_path=fixture["receipt"],
            source_model_path=fixture["model"],
            project_root=REPOSITORY,
        )


def test_release_authority_never_promotes_controller_declaration_to_gpu_lock_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _release_fixture(tmp_path)
    _patch_gnn_api(monkeypatch)
    monkeypatch.setattr(t5, "_git_identity", lambda _root: (fixture["commit"], fixture["tree"]))
    payload = json.loads(Path(fixture["authority"]).read_text(encoding="utf-8"))
    payload["controller_binding_state"] = "owned"
    payload["gpu_lock_authority_present"] = True
    authority_hash = _json(Path(fixture["authority"]), payload)
    with pytest.raises(
        t5.TasteCleanPolicyReleaseDisabled, match="controller_binding_state"
    ):
        t5.load_release_authority(
            fixture["authority"],
            expected_sha256=authority_hash,
            policy_path=POLICY,
            policy_receipt_path=fixture["receipt"],
            source_model_path=fixture["model"],
            project_root=REPOSITORY,
        )


def test_fresh_zero_step_initializer_publishes_five_file_closure_and_detects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _release_fixture(tmp_path)
    _patch_unreleased_producer_for_test(monkeypatch, Path(fixture["config"]))
    _patch_gnn_api(monkeypatch)
    monkeypatch.setattr(t5, "_git_identity", lambda _root: (fixture["commit"], fixture["tree"]))

    monkeypatch.setattr(t5, "_materialize_zero_step_lora", _fake_materialize_adapter)
    _patch_gpu(monkeypatch)
    _patch_publish_for_host(monkeypatch, Path(fixture["output_parent"]))
    output = Path(fixture["output_parent"]) / "20260828T010203Z"
    result = t5.build_clean_policy_initializer(
        config_path=fixture["config"],
        release_authority_path=fixture["authority"],
        expected_release_authority_sha256=str(fixture["authority_hash"]),
        policy_path=POLICY,
        policy_receipt_path=fixture["receipt"],
        source_model_path=fixture["model"],
        project_root=REPOSITORY,
        output_root=output,
    )
    assert set(result) == {
        "schema_version",
        "status",
        "stage",
        "output_root",
        "adapter_dir",
        "source_model_dir",
        "source_model_path",
        "policy_initializer_hash",
        "reference_model_hash",
        "reference_policy_hash",
        "source_model_inventory_sha256",
        "adapter_sha256",
        "manifest_sha256",
        "gate_sha256",
        "t5_gate_sha256",
        "pass_sha256",
        "t5_pass_sha256",
        "input_hashes_sha256",
        "output_hashes_sha256",
        "output_inventory_sha256",
        "root_inventory_sha256",
        "t5_output_inventory_sha256",
        "frozen_oracle_identity",
        "frozen_oracle_identity_sha256",
        "gpu_identity",
        "marker",
    }
    assert result["marker"] == t5.PASS_MARKER
    assert result["adapter_dir"] == str(output / "adapter")
    assert result["source_model_dir"] == str(Path(fixture["model"]).resolve())
    assert result["reference_model_hash"] != result["reference_policy_hash"]
    with t5.hold_clean_policy_output(output) as held:
        assert held.revalidate() == result
    assert stable_validate_clean_policy_output(output) == result
    with stable_hold_clean_policy_output(output) as held:
        assert held.revalidate() == result
    if not sys.platform.startswith("linux"):
        stable_paths = {
            (os.stat(result["source_model_dir"]).st_dev, os.stat(result["source_model_dir"]).st_ino): Path(
                result["source_model_dir"]
            ),
            (os.stat(result["adapter_dir"]).st_dev, os.stat(result["adapter_dir"]).st_ino): Path(
                result["adapter_dir"]
            ),
        }

        def portable_held_load_path(descriptor: int, *, label: str) -> Path:
            info = os.fstat(descriptor)
            try:
                return stable_paths[(info.st_dev, info.st_ino)]
            except KeyError as exc:
                raise t5.TasteCleanPolicyError(f"unknown held loader in {label}") from exc

        monkeypatch.setattr(t5, "_fd_directory_path", portable_held_load_path)
    with stable_hold_clean_policy_load_authority(output) as load_authority:
        token = load_authority.load_token()
        assert set(token.evidence()) == {
            "schema_version",
            "output_root",
            "source_model_load_path",
            "adapter_load_path",
            "source_model_inventory_sha256",
            "adapter_inventory_sha256",
            "adapter_parameter_sha256",
            "reference_policy_hash",
            "t5_output_inventory_sha256",
            "frozen_oracle_identity_sha256",
        }
        assert token.reference_policy_hash == result["reference_policy_hash"]
        assert token.reference_policy_hash == t5._reference_policy_hash(  # noqa: SLF001
            source_model_inventory_sha256=token.source_model_inventory_sha256,
            adapter_inventory_sha256=token.adapter_inventory_sha256,
            adapter_parameter_sha256=token.adapter_parameter_sha256,
        )
        assert token.source_model_load_path != Path(result["adapter_dir"])
        assert load_authority.revalidate_load_token(token) == token
    with t5.hold_clean_policy_output(output) as held_output:
        weights = output / "adapter" / "adapter_model.safetensors"
        held_copy = tmp_path / "held-adapter-weights"
        os.rename(weights, held_copy)
        os.rename(held_copy, weights)
        with pytest.raises(
            t5.TasteCleanPolicyError, match="physical stat inventory changed"
        ):
            held_output.revalidate()
    source = t5.validate_source_model_for_clean_policy(
        result["source_model_dir"], result["source_model_inventory_sha256"]
    )
    assert source["source_model_inventory_sha256"] == result["reference_model_hash"]
    with t5.hold_source_model_for_clean_policy(
        result["source_model_dir"], result["source_model_inventory_sha256"]
    ) as held_source:
        assert held_source.revalidate() == source
        unexpected = Path(result["source_model_dir"]) / "unexpected.bin"
        unexpected.write_bytes(b"source-drift")
        with pytest.raises(t5.TasteCleanPolicyError, match="physical stat inventory changed"):
            held_source.revalidate()
        unexpected.unlink()
        with pytest.raises(t5.TasteCleanPolicyError, match="physical stat inventory changed"):
            held_source.revalidate()
    assert (output / "PASS").read_text(encoding="utf-8") == t5.PASS_MARKER + "\n"
    for name in ("state.json", "manifest.json", "gate.json", "input_hashes.json", "output_hashes.json"):
        assert (output / name).is_file()
    provenance = json.loads((output / "policy_provenance.json").read_text(encoding="utf-8"))
    assert provenance["initializer_data_split_used"] == "none"
    assert provenance["taste_split_access_max"] == "train_only"
    assert provenance["optimizer_step_count"] == 0
    assert provenance["taste_splits_loaded"] == []

    # Recompute every affected top-level hash after changing strict_flip.  The
    # validator must reject the semantic drift, not merely notice a stale
    # digest in the evidence files.
    original_inputs = (output / "input_hashes.json").read_bytes()
    original_manifest = (output / "manifest.json").read_bytes()
    original_outputs = (output / "output_hashes.json").read_bytes()
    inputs = json.loads(original_inputs)
    inputs["frozen_oracle"]["strict_flip"] = "pred_before != pred_after"
    changed_input_sha = _json(output / "input_hashes.json", inputs)
    manifest = json.loads(original_manifest)
    manifest["input_hashes_sha256"] = changed_input_sha
    manifest["frozen_oracle_identity_sha256"] = t5._canonical_sha256(  # noqa: SLF001
        inputs["frozen_oracle"]
    )
    changed_manifest_sha = _json(output / "manifest.json", manifest)
    output_hashes = json.loads(original_outputs)
    output_hashes["input_hashes_sha256"] = changed_input_sha
    output_hashes["manifest_sha256"] = changed_manifest_sha
    _json(output / "output_hashes.json", output_hashes)
    with pytest.raises(t5.TasteCleanPolicyError, match="frozen-oracle semantics"):
        t5.validate_clean_policy_output(output)
    (output / "input_hashes.json").write_bytes(original_inputs)
    (output / "manifest.json").write_bytes(original_manifest)
    (output / "output_hashes.json").write_bytes(original_outputs)

    (output / "adapter" / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(t5.TasteCleanPolicyError, match="safetensors"):
        t5.validate_clean_policy_output(output)


def test_t3_gate_drift_and_existing_output_fail_closed_before_adapter_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _release_fixture(tmp_path)
    _patch_unreleased_producer_for_test(monkeypatch, Path(fixture["config"]))
    _patch_gnn_api(monkeypatch)
    monkeypatch.setattr(t5, "_git_identity", lambda _root: (fixture["commit"], fixture["tree"]))
    t3_gate_path = Path(fixture["t3"]) / "gate.json"
    t3_gate = json.loads(t3_gate_path.read_text(encoding="utf-8"))
    t3_gate["status"] = "FAILED"
    t3_gate_path.write_text(json.dumps(t3_gate), encoding="utf-8")
    output = Path(fixture["output_parent"]) / "20260828T020304Z"
    with pytest.raises(t5.TasteCleanPolicyError, match="T3 gate"):
        t5.build_clean_policy_initializer(
            config_path=fixture["config"],
            release_authority_path=fixture["authority"],
            expected_release_authority_sha256=str(fixture["authority_hash"]),
            policy_path=POLICY,
            policy_receipt_path=fixture["receipt"],
            source_model_path=fixture["model"],
            project_root=REPOSITORY,
            output_root=output,
        )
    assert not output.exists()


def test_t3_t4_self_consistent_root_rewrite_cannot_diverge_feature_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _release_fixture(tmp_path)
    t4_provenance_path = Path(fixture["t4"]) / "oracle_provenance.json"
    t4_provenance = json.loads(t4_provenance_path.read_text(encoding="utf-8"))
    t4_provenance["feature_schema_sha256"] = "d" * 64
    _json(t4_provenance_path, t4_provenance)

    authority_payload = json.loads(Path(fixture["authority"]).read_text(encoding="utf-8"))
    authority_payload["frozen_oracle"]["t4_root_inventory_sha256"] = _tree_sha(
        Path(fixture["t4"])
    )
    authority_hash = _json(Path(fixture["authority"]), authority_payload)
    _patch_gnn_api(monkeypatch)
    monkeypatch.setattr(t5, "_git_identity", lambda _root: (fixture["commit"], fixture["tree"]))
    with pytest.raises(
        t5.TasteCleanPolicyReleaseDisabled,
        match="T3/T4 model/feature/temperature differs",
    ):
        t5.load_release_authority(
            fixture["authority"],
            expected_sha256=authority_hash,
            policy_path=POLICY,
            policy_receipt_path=fixture["receipt"],
            source_model_path=fixture["model"],
            project_root=REPOSITORY,
        )


def test_publish_rejects_rename_copy_that_is_not_the_held_staging_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _release_fixture(tmp_path)
    _patch_unreleased_producer_for_test(monkeypatch, Path(fixture["config"]))
    _patch_gnn_api(monkeypatch)
    _patch_gpu(monkeypatch)
    monkeypatch.setattr(t5, "_git_identity", lambda _root: (fixture["commit"], fixture["tree"]))
    monkeypatch.setattr(t5, "_materialize_zero_step_lora", _fake_materialize_adapter)
    parent = Path(fixture["output_parent"])

    def replace_during_publish(_parent_fd: int, source: str, target: str) -> None:
        held_backup = parent / f"{source}.held-backup"
        os.rename(parent / source, held_backup)
        replacement = parent / f"{source}.replacement"
        replacement.mkdir(mode=0o700)
        os.rename(replacement, parent / target)

    monkeypatch.setattr(t5, "_renameat2_noreplace", replace_during_publish)
    output = parent / "20260828T030405Z"
    with pytest.raises(t5.TasteCleanPolicyError, match="held staging inode"):
        t5.build_clean_policy_initializer(
            config_path=fixture["config"],
            release_authority_path=fixture["authority"],
            expected_release_authority_sha256=str(fixture["authority_hash"]),
            policy_path=POLICY,
            policy_receipt_path=fixture["receipt"],
            source_model_path=fixture["model"],
            project_root=REPOSITORY,
            output_root=output,
        )
    assert output.is_dir()
    assert not (output / "PASS").exists()


def test_post_publish_external_revalidation_failure_never_writes_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _release_fixture(tmp_path)
    _patch_unreleased_producer_for_test(monkeypatch, Path(fixture["config"]))
    _patch_gnn_api(monkeypatch)
    monkeypatch.setattr(t5, "_git_identity", lambda _root: (fixture["commit"], fixture["tree"]))
    monkeypatch.setattr(t5, "_materialize_zero_step_lora", _fake_materialize_adapter)
    _patch_publish_for_host(monkeypatch, Path(fixture["output_parent"]))
    calls = 0

    def fail_after_publish(
        authority: t5.TasteCleanPolicyReleaseAuthority,
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise t5.TasteCleanPolicyReleaseDisabled(
                "synthetic post-publication external revalidation failure"
            )
        return _gpu_identity(authority)

    monkeypatch.setattr(t5, "_validate_t5_gpu", fail_after_publish)
    output = Path(fixture["output_parent"]) / "20260828T040506Z"
    with pytest.raises(
        t5.TasteCleanPolicyReleaseDisabled,
        match="post-publication external revalidation failure",
    ):
        t5.build_clean_policy_initializer(
            config_path=fixture["config"],
            release_authority_path=fixture["authority"],
            expected_release_authority_sha256=str(fixture["authority_hash"]),
            policy_path=POLICY,
            policy_receipt_path=fixture["receipt"],
            source_model_path=fixture["model"],
            project_root=REPOSITORY,
            output_root=output,
        )
    assert calls == 4
    assert output.is_dir()
    assert not (output / "PASS").exists()


def test_prepared_terminal_validation_failure_never_exposes_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _release_fixture(tmp_path)
    _patch_unreleased_producer_for_test(monkeypatch, Path(fixture["config"]))
    _patch_gnn_api(monkeypatch)
    monkeypatch.setattr(t5, "_git_identity", lambda _root: (fixture["commit"], fixture["tree"]))
    monkeypatch.setattr(t5, "_materialize_zero_step_lora", _fake_materialize_adapter)
    _patch_gpu(monkeypatch)
    _patch_publish_for_host(monkeypatch, Path(fixture["output_parent"]))
    real_validate = t5._validate_held_clean_policy_output  # noqa: SLF001

    def fail_prepared_validation(*args: object, **kwargs: object) -> dict[str, object]:
        if kwargs.get("marker_name") == ".PASS.prepared":
            raise t5.TasteCleanPolicyError("synthetic prepared terminal validation failure")
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(t5, "_validate_held_clean_policy_output", fail_prepared_validation)
    output = Path(fixture["output_parent"]) / "20260828T050607Z"
    with pytest.raises(
        t5.TasteCleanPolicyError,
        match="prepared terminal validation failure",
    ):
        t5.build_clean_policy_initializer(
            config_path=fixture["config"],
            release_authority_path=fixture["authority"],
            expected_release_authority_sha256=str(fixture["authority_hash"]),
            policy_path=POLICY,
            policy_receipt_path=fixture["receipt"],
            source_model_path=fixture["model"],
            project_root=REPOSITORY,
            output_root=output,
        )
    assert output.is_dir()
    assert not (output / "PASS").exists()
    assert (output / ".PASS.prepared").is_file()
    monkeypatch.setattr(t5, "_validate_held_clean_policy_output", real_validate)
    with pytest.raises(t5.TasteCleanPolicyError, match="top-level inventory"):
        t5.validate_clean_policy_output(output)
    with pytest.raises(t5.TasteCleanPolicyError, match="top-level inventory"):
        t5.hold_clean_policy_output(output)


def test_cli_and_paired_wrappers_expose_only_zero_step_autodl_route() -> None:
    help_run = subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts/build_tastemolnet_clean_policy_initializer.py"), "build", "--help"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert "oracle-neutral-sft" not in help_run.stdout
    autodl = (REPOSITORY / "scripts/autodl/run_tastemolnet_clean_policy_initializer.sh").read_text(encoding="utf-8")
    slurm = (REPOSITORY / "scripts/slurm/build_tastemolnet_clean_policy_initializer.sh").read_text(encoding="utf-8")
    assert "TASTEMOLNET_T5_RELEASE_AUTHORITY_SHA256" in autodl
    assert "REFUSING_HPC_EXECUTION" in slurm
    assert "exit 78" in slurm
    assert "sbatch" not in autodl.lower()
