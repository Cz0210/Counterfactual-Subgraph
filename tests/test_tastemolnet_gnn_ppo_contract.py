from __future__ import annotations

from copy import deepcopy
import io
import json
import logging
from pathlib import Path

import pytest

from scripts import train_tastemolnet_gnn_ppo as runner
from src.rewards.gnn_ppo_reward import TASTE_GNN_PPO_REWARD_SCHEMA
from src.train import tastemolnet_gnn_ppo as typed_ppo
from src.train.tastemolnet_gnn_ppo import (
    TASTE_PPO_CHECKPOINT_IDENTITY_SCHEMA,
    TASTE_PPO_MARKER,
    TastePPOObserver,
    build_taste_reward_manifest,
    build_taste_smoke_gate,
    validate_taste_adapter_checkpoint_reload,
    validate_taste_ppo_output,
)
from src.utils import retained_output_directory as retained
from src.utils.tastemolnet_gine_pass_adoption_v1 import (
    ADOPTION_MARKER,
    DOWNSTREAM_BINDING_SCHEMA,
    SOURCE_CID,
    SOURCE_RUN_ID,
)


def test_t6_runtime_receipt_fields_support_real_managed_t2(tmp_path: Path) -> None:
    managed_root = (tmp_path / "managed-t2").resolve()
    managed_root.mkdir()
    root, digest = runner._t2_runtime_receipt_fields(  # noqa: SLF001
        {
            "schema_version": runner.T6_MANAGED_T2_BINDING_SCHEMA,
            "root": str(managed_root),
            "receipt_inventory_sha256": "a" * 64,
        }
    )
    assert root == managed_root
    assert digest == "a" * 64


def test_t6_terminal_validator_accepts_real_managed_t2_binding(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "managed-t2").resolve()
    root.mkdir()
    files = {
        name: str(index) * 64
        for index, name in enumerate(
            sorted(typed_ppo._TASTE_PPO_MANAGED_T2_RECEIPT_FILES), start=1
        )
    }
    artifacts = {
        "model.pt": "8" * 64,
        "feature_schema.json": "9" * 64,
        "split_manifest.json": "a" * 64,
    }
    binding = {
        "schema_version": typed_ppo.TASTE_PPO_MANAGED_T2_BINDING_SCHEMA,
        "status": "PASS",
        "root": str(root),
        "receipt_id": root.name,
        "gate_sha256": files["gate.json"],
        "source_evidence_file_sha256": files["source_evidence.json"],
        "source_evidence_sha256": "b" * 64,
        "receipt_inventory_sha256": typed_ppo._canonical_sha256(files),  # noqa: SLF001
        "file_sha256": files,
        "source_artifact_hashes": artifacts,
        "model_sha256": artifacts["model.pt"],
        "feature_schema_file_sha256": artifacts["feature_schema.json"],
        "split_manifest_file_sha256": artifacts["split_manifest.json"],
    }
    validated = typed_ppo._validate_t2_adoption_binding(binding)  # noqa: SLF001
    assert validated == binding
    assert typed_ppo._t2_adoption_receipt_sha256(validated) == binding[  # noqa: SLF001
        "receipt_inventory_sha256"
    ]


def _row(destination: int) -> dict[str, object]:
    after = [0.90, 0.05, 0.05] if destination == 0 else [0.05, 0.05, 0.90]
    return {
        "schema_version": TASTE_GNN_PPO_REWARD_SCHEMA,
        "dataset": "tastemolnet",
        "num_classes": 3,
        "parent_id": f"parent-{destination}",
        "parent_smiles": "CCCO",
        "source_label": 1,
        "raw_fragment": "O",
        "core_fragment": "O",
        "final_fragment": "O",
        "parse_ok": True,
        "connected": True,
        "direct_substructure": True,
        "projection_used": False,
        "deletion_valid": True,
        "gnn_scored_deletion": True,
        "pred_before": 1,
        "pred_after": destination,
        "destination_label": destination,
        "p_before_all_classes": [0.05, 0.90, 0.05],
        "p_after_all_classes": after,
        "logits_before_all_classes": [-3.0, -0.1, -3.0],
        "logits_after_all_classes": [-0.1, -3.0, -3.0]
        if destination == 0
        else [-3.0, -3.0, -0.1],
        "source_probability_before": 0.90,
        "source_probability_after": 0.05,
        "cf_drop": 0.85,
        "margin_before": 0.85,
        "margin_after": -0.85,
        "margin_drop": 1.70,
        "cf_flip": True,
        "reward_valid": 0.25,
        "reward_substructure": 1.0,
        "reward_cf": 4.55,
        "reward_size": 0.25,
        "reward_projection": 0.0,
        "reward_kl": 0.0,
        "reward_total": 6.05,
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "rf_oracle_used": False,
        "calibration_loaded": False,
        "calibration_dataset_loaded": False,
        "frozen_temperature_calibration_loaded": True,
        "test_loaded": False,
        "oracle_checkpoint_hash": "a" * 64,
        "temperature_calibration_hash": "b" * 64,
        "feature_schema_hash": "c" * 64,
        "policy_initializer_hash": "d" * 64,
        "reference_policy_hash": "e" * 64,
    }


def _oracle() -> dict[str, object]:
    return {
        "dataset": "tastemolnet",
        "schema_version": "tastemolnet_gnn_ppo_oracle_provenance_v1",
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "rf_oracle_used": False,
        "num_classes": 3,
        "source_label": 1,
        "calibration_loaded": False,
        "calibration_dataset_loaded": False,
        "frozen_temperature_calibration_loaded": True,
        "test_loaded": False,
        "oracle_load_count": 1,
        "backbone": "gine",
        "temperature": 1.5,
        "checkpoint_id": "a" * 64,
        "temperature_calibration_hash": "b" * 64,
        "feature_schema_hash": "c" * 64,
        "policy_initializer_hash": "d" * 64,
        "reference_policy_hash": "e" * 64,
    }


def _observer() -> TastePPOObserver:
    observer = TastePPOObserver()
    for step in range(1, 6):
        observer.on_update(
            step_index=step,
            batch_ids=[f"batch-{step}"],
            reward_logs=[{"reward_total": 1.0, "step_index": step}],
            metrics={
                "global_step": step,
                "rollout_batch_size": 1,
                "reward_mean": 1.0,
                "reward_min": 1.0,
                "reward_max": 1.0,
                "ppo_reward_mean": 1.0,
                "policy_loss": 0.1,
                "value_loss": 0.1,
                "total_loss": 0.2,
                "approx_kl": 0.01,
                "parse_ok_rate": 1.0,
                "valid_rate": 1.0,
                "direct_substructure_rate": 1.0,
                "final_substructure_rate": 1.0,
                "projection_used_rate": 0.0,
                "oracle_ok_rate": 1.0,
                "cf_flip_rate": 1.0,
                "cf_drop_mean": 0.5,
            },
        )
    observer.on_checkpoint(
        step_index=5,
        checkpoint_dir=Path("/private/fresh-output/checkpoint-5"),
        checkpoint_kind="periodic",
    )
    observer.on_finish(output_dir="/private/fresh-output")
    return observer


def test_observer_captures_real_reward_rows_before_candidate_publication() -> None:
    observer = TastePPOObserver()
    source = _row(0)
    source["step_index"] = 1
    observer.on_update(
        step_index=1,
        batch_ids=["batch-1"],
        reward_logs=[source],
        metrics={
            "global_step": 1,
            "rollout_batch_size": 1,
            "reward_mean": 1.0,
            "reward_min": 1.0,
            "reward_max": 1.0,
            "ppo_reward_mean": 1.0,
            "policy_loss": 0.1,
            "value_loss": 0.1,
            "total_loss": 0.2,
            "approx_kl": 0.01,
            "parse_ok_rate": 1.0,
            "valid_rate": 1.0,
            "direct_substructure_rate": 1.0,
            "final_substructure_rate": 1.0,
            "projection_used_rate": 0.0,
            "oracle_ok_rate": 1.0,
            "cf_flip_rate": 1.0,
            "cf_drop_mean": 0.5,
        },
    )
    source["reward_total"] = 999.0
    rows = observer.captured_candidate_rows()
    assert rows[0]["step_index"] == 1
    assert rows[0]["reward_total"] == 6.05


def test_observer_finish_normalizes_stable_validation_dataclass() -> None:
    from scripts.train_ppo_stable import StableValidationState

    observer = TastePPOObserver()
    observer.on_finish(
        final_output_dir=Path("/private/reviewed/output"),
        candidate_pool_path=Path("/private/reviewed/output/candidate_pool.jsonl"),
        candidate_count=4,
        global_step=5,
        validation_state=StableValidationState(),
        early_stop_reason=None,
    )
    payload = observer.state_dict()["finish"]
    assert payload == {
        "final_output_dir": "/private/reviewed/output",
        "candidate_pool_path": "/private/reviewed/output/candidate_pool.jsonl",
        "candidate_count": 4,
        "global_step": 5,
        "validation_state": {
            "best_val_score": None,
            "best_step": None,
            "stale_eval_count": 0,
        },
        "early_stop_reason": None,
    }
    json.dumps(payload, allow_nan=False)


def _checkpoint_reload() -> dict[str, object]:
    return {
        "checkpoint_reload_pass": True,
        "policy_checkpoint_hash_schema": "tastemolnet_lora_checkpoint_identity_v1",
        "policy_checkpoint_hash": "7" * 64,
        "adapter_parameter_sha256": "2" * 64,
        "value_head_parameter_sha256": "8" * 64,
    }


def _canonical_sha256(payload: object) -> str:
    return __import__("hashlib").sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _t2_adoption_binding() -> dict[str, object]:
    adoption_root = Path("/private/reviewed/t2-adoption") / SOURCE_CID
    formal_inventory = [
        {
            "path": "model.pt",
            "kind": "file",
            "identity": {
                "device": 1,
                "inode": 2,
                "mode": 0o100600,
                "uid": 3,
                "gid": 4,
                "nlink": 1,
                "size": 5,
                "blocks": 8,
                "mtime_ns": 6,
                "ctime_ns": 7,
            },
            "sha256": "d" * 64,
        }
    ]
    return {
        "schema_version": DOWNSTREAM_BINDING_SCHEMA,
        "stage": "T2_GINE_FULL",
        "status": "PASS",
        "state": ADOPTION_MARKER,
        "source_cid": SOURCE_CID,
        "source_run_id": SOURCE_RUN_ID,
        "adoption_root": str(adoption_root),
        "adoption_root_inventory_sha256": "9" * 64,
        "gate_path": str(adoption_root / "gate.json"),
        "gate_sha256": "a" * 64,
        "receipt_path": str(adoption_root / "manifest.json"),
        "receipt_sha256": "b" * 64,
        "source_evidence_sha256": "c" * 64,
        "formal_bundle_root": "/private/reviewed/t2-bundle",
        "formal_bundle_inventory": formal_inventory,
        "formal_bundle_inventory_sha256": _canonical_sha256(formal_inventory),
        "formal_bundle_model_sha256": "d" * 64,
        "formal_bundle_sha256s_sha256": "e" * 64,
    }


def _t2_gate_hashes(
    binding: dict[str, object] | None = None,
) -> dict[str, str]:
    reviewed = _t2_adoption_binding() if binding is None else binding
    return {
        "t2_adoption_gate_sha256": str(reviewed["gate_sha256"]),
        "t2_adoption_receipt_sha256": str(reviewed["receipt_sha256"]),
        "t2_adoption_binding_sha256": _canonical_sha256(reviewed),
    }


def test_taste_reward_manifest_and_five_step_gate_pass() -> None:
    oracle = _oracle()
    manifest = build_taste_reward_manifest(
        [_row(0), _row(2)], oracle_provenance=oracle
    )
    assert manifest["strict_flip_count"] == 2
    assert manifest["destination_0_count"] == 1
    assert manifest["destination_2_count"] == 1
    gate = build_taste_smoke_gate(
        policy_parameter_hash_before="1" * 64,
        policy_parameter_hash_after="2" * 64,
        reference_parameter_hash_before="3" * 64,
        reference_parameter_hash_after="3" * 64,
        policy_t5_identity_before="e" * 64,
        reference_t5_identity_before="e" * 64,
        reference_t5_identity_after="e" * 64,
        expected_t5_reference_policy_hash="e" * 64,
        observer=_observer(),
        checkpoint_reload=_checkpoint_reload(),
        periodic_checkpoint_reload=_checkpoint_reload(),
        reward_manifest=manifest,
        oracle_provenance=oracle,
        policy_initializer_hash="d" * 64,
        **_t2_gate_hashes(),
        t3_gate_sha256="5" * 64,
        t4_gate_sha256="6" * 64,
        value_head_parameter_sha256="8" * 64,
    )
    assert gate["status"] == "PASS"
    assert gate["marker"] == TASTE_PPO_MARKER
    assert gate["optimizer_step_count"] == 5
    assert gate["strict_flip_observed"] is True
    assert gate["destination_contract_pass"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("num_classes", 3.0, "authority drift"),
        ("source_label", True, "authority drift"),
        ("destination_label", 1.0, "destination authority"),
        ("p_after_all_classes", [0.1, 0.9], "finite three-class"),
        ("p_after_all_classes", [0.8, 0.8, -0.6], "probability distribution"),
        ("pred_after", 2, "destination authority"),
        ("margin_drop", 1.0, "margin_drop drifted"),
        ("cf_flip", False, "strict-flip semantics"),
    ),
)
def test_taste_reward_manifest_rejects_multiclass_authority_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    row = _row(0)
    row[field] = value
    with pytest.raises(ValueError, match=message):
        build_taste_reward_manifest([row], oracle_provenance=_oracle())


@pytest.mark.parametrize(
    "field",
    (
        "oracle_checkpoint_hash",
        "temperature_calibration_hash",
        "feature_schema_hash",
        "policy_initializer_hash",
        "reference_policy_hash",
    ),
)
def test_taste_reward_manifest_rejects_valid_hex_hash_not_bound_to_oracle(
    field: str,
) -> None:
    row = _row(0)
    row[field] = "9" * 64
    with pytest.raises(ValueError, match="oracle/policy hash authority drifted"):
        build_taste_reward_manifest([row], oracle_provenance=_oracle())


@pytest.mark.parametrize(
    ("final_adapter", "periodic_adapter", "expected_failure"),
    (
        ("9" * 64, "9" * 64, "final_checkpoint_matches_trained_policy"),
        ("2" * 64, "9" * 64, "periodic_checkpoint_matches_trained_policy"),
        ("9" * 64, "2" * 64, "final_checkpoint_matches_trained_policy"),
    ),
)
def test_taste_gate_rejects_checkpoint_tensor_identity_not_bound_to_policy(
    final_adapter: str,
    periodic_adapter: str,
    expected_failure: str,
) -> None:
    oracle = _oracle()
    manifest = build_taste_reward_manifest([_row(0)], oracle_provenance=oracle)
    final_reload = {**_checkpoint_reload(), "adapter_parameter_sha256": final_adapter}
    periodic_reload = {
        **_checkpoint_reload(),
        "adapter_parameter_sha256": periodic_adapter,
    }
    gate = build_taste_smoke_gate(
        policy_parameter_hash_before="1" * 64,
        policy_parameter_hash_after="2" * 64,
        reference_parameter_hash_before="3" * 64,
        reference_parameter_hash_after="3" * 64,
        policy_t5_identity_before="e" * 64,
        reference_t5_identity_before="e" * 64,
        reference_t5_identity_after="e" * 64,
        expected_t5_reference_policy_hash="e" * 64,
        observer=_observer(),
        checkpoint_reload=final_reload,
        periodic_checkpoint_reload=periodic_reload,
        reward_manifest=manifest,
        oracle_provenance=oracle,
        policy_initializer_hash="d" * 64,
        **_t2_gate_hashes(),
        t3_gate_sha256="5" * 64,
        t4_gate_sha256="6" * 64,
        value_head_parameter_sha256="8" * 64,
    )
    assert gate["status"] == "FAIL"
    assert expected_failure in gate["failures"]


@pytest.mark.parametrize(
    "field",
    (
        "policy_t5_identity_before",
        "reference_t5_identity_before",
        "reference_t5_identity_after",
    ),
)
def test_taste_gate_rejects_loaded_policy_or_reference_not_bound_to_t5(
    field: str,
) -> None:
    oracle = _oracle()
    manifest = build_taste_reward_manifest([_row(0)], oracle_provenance=oracle)
    identities = {
        "policy_t5_identity_before": "e" * 64,
        "reference_t5_identity_before": "e" * 64,
        "reference_t5_identity_after": "e" * 64,
    }
    identities[field] = "9" * 64
    gate = build_taste_smoke_gate(
        policy_parameter_hash_before="1" * 64,
        policy_parameter_hash_after="2" * 64,
        reference_parameter_hash_before="3" * 64,
        reference_parameter_hash_after="3" * 64,
        expected_t5_reference_policy_hash="e" * 64,
        observer=_observer(),
        checkpoint_reload=_checkpoint_reload(),
        periodic_checkpoint_reload=_checkpoint_reload(),
        reward_manifest=manifest,
        oracle_provenance=oracle,
        policy_initializer_hash="d" * 64,
        **_t2_gate_hashes(),
        t3_gate_sha256="5" * 64,
        t4_gate_sha256="6" * 64,
        value_head_parameter_sha256="8" * 64,
        **identities,
    )
    assert gate["status"] == "FAIL"
    expected = (
        "policy_initialized_from_exact_t5"
        if field == "policy_t5_identity_before"
        else "reference_remains_exact_t5"
    )
    assert expected in gate["failures"]


def test_true_sweet_but_non_source_prediction_is_not_laundered_into_a_flip() -> None:
    row = _row(2)
    row["pred_before"] = 0
    row["p_before_all_classes"] = [0.90, 0.05, 0.05]
    row["logits_before_all_classes"] = [-0.1, -3.0, -3.0]
    row["source_probability_before"] = 0.05
    row["cf_drop"] = 0.0
    row["margin_before"] = -0.85
    row["margin_after"] = -0.85
    row["margin_drop"] = 0.0
    row["cf_flip"] = False
    manifest = build_taste_reward_manifest([row], oracle_provenance=_oracle())
    assert manifest["gnn_scored_deletion_count"] == 1
    assert manifest["strict_flip_count"] == 0
    assert manifest["destination_labels"] == []


def test_taste_smoke_gate_fails_without_real_update_and_flip() -> None:
    oracle = _oracle()
    row = deepcopy(_row(0))
    row["pred_after"] = 1
    row["destination_label"] = 1
    row["p_after_all_classes"] = [0.05, 0.90, 0.05]
    row["logits_after_all_classes"] = [-3.0, -0.1, -3.0]
    row["source_probability_after"] = 0.90
    row["cf_drop"] = 0.0
    row["margin_after"] = 0.85
    row["margin_drop"] = 0.0
    row["cf_flip"] = False
    manifest = build_taste_reward_manifest([row], oracle_provenance=oracle)
    observer = TastePPOObserver()
    observer.on_finish(output_dir="/private/fresh-output")
    gate = build_taste_smoke_gate(
        policy_parameter_hash_before="1" * 64,
        policy_parameter_hash_after="1" * 64,
        reference_parameter_hash_before="3" * 64,
        reference_parameter_hash_after="3" * 64,
        policy_t5_identity_before="e" * 64,
        reference_t5_identity_before="e" * 64,
        reference_t5_identity_after="e" * 64,
        expected_t5_reference_policy_hash="e" * 64,
        observer=observer,
        checkpoint_reload=_checkpoint_reload(),
        periodic_checkpoint_reload=_checkpoint_reload(),
        reward_manifest=manifest,
        oracle_provenance=oracle,
        policy_initializer_hash="4" * 64,
        **_t2_gate_hashes(),
        t3_gate_sha256="5" * 64,
        t4_gate_sha256="6" * 64,
        value_head_parameter_sha256="8" * 64,
    )
    assert gate["status"] == "FAIL"
    assert "minimum_update_count_met" in gate["failures"]
    assert "policy_parameters_changed" in gate["failures"]
    assert "strict_flip_observed" in gate["failures"]
    assert "destination_contract_pass" in gate["failures"]


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"metrics": {}}, "metrics are incomplete"),
        ({"metrics": {"global_step": 1.0}}, "metrics are incomplete"),
        ({"batch_ids": []}, "batch IDs are malformed"),
        ({"reward_logs": []}, "reward rows differ"),
    ),
)
def test_taste_observer_rejects_synthetic_update_evidence(
    change: dict[str, object],
    message: str,
) -> None:
    observer = TastePPOObserver()
    kwargs: dict[str, object] = {
        "step_index": 1,
        "batch_ids": ["batch-1"],
        "reward_logs": [{"reward_total": 1.0, "step_index": 1}],
        "metrics": {
            "global_step": 1,
            "rollout_batch_size": 1,
            "reward_mean": 1.0,
            "reward_min": 1.0,
            "reward_max": 1.0,
            "ppo_reward_mean": 1.0,
            "policy_loss": 0.1,
            "value_loss": 0.1,
            "total_loss": 0.2,
            "approx_kl": 0.01,
            "parse_ok_rate": 1.0,
            "valid_rate": 1.0,
            "direct_substructure_rate": 1.0,
            "final_substructure_rate": 1.0,
            "projection_used_rate": 0.0,
            "oracle_ok_rate": 1.0,
            "cf_flip_rate": 1.0,
            "cf_drop_mean": 0.5,
        },
    }
    kwargs.update(change)
    with pytest.raises(ValueError, match=message):
        observer.on_update(**kwargs)  # type: ignore[arg-type]


def test_taste_checkpoint_reload_builds_taste_identity_from_held_bytes(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = checkpoint / "adapter_config.json"
    weights = checkpoint / "adapter_model.safetensors"
    config_payload = {
        "peft_type": "LORA",
        "base_model_name_or_path": str(tmp_path / "source-model"),
    }
    config.write_text(
        json.dumps(config_payload)
        + "\n",
        encoding="utf-8",
    )
    import torch
    from safetensors.torch import save_file

    save_file(
        {
            "base_model.model.layer.lora_A.weight": torch.ones((2, 3)),
            "base_model.model.layer.lora_B.weight": torch.zeros((4, 2)),
        },
        str(weights),
    )
    torch.save(
        {"summary.weight": torch.ones((1, 4)), "summary.bias": torch.zeros(1)},
        checkpoint / "decoded_chem_value_head.pt",
    )
    result = validate_taste_adapter_checkpoint_reload(
        checkpoint,
        expected_base_model_path=tmp_path / "source-model",
        expected_adapter_config=config_payload,
    )
    assert result["checkpoint_reload_pass"] is True
    assert result["policy_checkpoint_hash_schema"] == TASTE_PPO_CHECKPOINT_IDENTITY_SCHEMA
    assert result["policy_checkpoint_hash_payload"]["schema_version"] == (
        TASTE_PPO_CHECKPOINT_IDENTITY_SCHEMA
    )
    assert len(result["adapter_parameter_sha256"]) == 64
    assert result["adapter_config"]["sha256"] == runner.sha256_file(config)
    assert result["adapter_weights"]["sha256"] == runner.sha256_file(weights)
    assert result["adapter_weights"]["physical_identity"]["inode"] == (
        weights.stat().st_ino
    )
    assert result["checkpoint_directory_identity"]["inode"] == (
        checkpoint.stat().st_ino
    )
    assert "bace" not in json.dumps(result).lower()


def test_taste_checkpoint_reload_rejects_equal_byte_weights_replacement_during_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config_payload = {
        "peft_type": "LORA",
        "base_model_name_or_path": str(tmp_path / "source-model"),
    }
    (checkpoint / "adapter_config.json").write_text(
        json.dumps(config_payload)
        + "\n",
        encoding="utf-8",
    )
    import torch
    import safetensors.torch
    from safetensors.torch import save_file

    weights = checkpoint / "adapter_model.safetensors"
    save_file(
        {
            "base_model.model.layer.lora_A.weight": torch.ones((2, 3)),
            "base_model.model.layer.lora_B.weight": torch.zeros((4, 2)),
        },
        str(weights),
    )
    torch.save(
        {"summary.weight": torch.ones((1, 4)), "summary.bias": torch.zeros(1)},
        checkpoint / "decoded_chem_value_head.pt",
    )
    original_load = safetensors.torch.load

    def replace_after_read(data: bytes):
        tensors = original_load(data)
        parked = checkpoint / "weights.parked"
        weights.rename(parked)
        weights.write_bytes(parked.read_bytes())
        return tensors

    monkeypatch.setattr(safetensors.torch, "load", replace_after_read)
    with pytest.raises(ValueError, match="leaf identity drifted"):
        validate_taste_adapter_checkpoint_reload(
            checkpoint,
            expected_base_model_path=tmp_path / "source-model",
            expected_adapter_config=config_payload,
        )


def test_checkpoint_serialization_rebinds_held_paths_to_reviewed_lexical_model(
    tmp_path: Path,
) -> None:
    class Tokenizer:
        name_or_path = "/proc/self/fd/41"
        init_kwargs = {"name_or_path": "/proc/self/fd/41"}

    class PeftConfig:
        base_model_name_or_path = "/proc/self/fd/42"

    class Policy:
        peft_config = {"default": PeftConfig()}

        class BaseModel:
            name_or_path = "/proc/self/fd/42"

        base_model = BaseModel()

        class Config:
            _name_or_path = "/proc/self/fd/43"

        config = Config()

    model = tmp_path / "source-model"
    tokenizer = Tokenizer()
    policy = Policy()
    runner._bind_checkpoint_serialization_paths(
        tokenizer=tokenizer,
        policy_model=policy,
        requested_model=model,
    )
    assert tokenizer.name_or_path == str(model)
    assert tokenizer.init_kwargs["name_or_path"] == str(model)
    assert policy.peft_config["default"].base_model_name_or_path == str(model)
    assert policy.config._name_or_path == str(model)


def test_real_peft_checkpoint_metadata_contains_only_reviewed_lexical_model(
    tmp_path: Path,
) -> None:
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")

    base = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            n_layer=1,
            n_head=1,
            n_embd=8,
            n_positions=16,
            vocab_size=32,
        )
    )
    base.config._name_or_path = "/proc/self/fd/777/source-model"
    policy = peft.get_peft_model(
        base,
        peft.LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["c_attn"],
            task_type="CAUSAL_LM",
            base_model_name_or_path="/proc/self/fd/777/source-model",
        ),
    )

    class Tokenizer:
        name_or_path = "/proc/self/fd/777/source-model"
        init_kwargs = {"name_or_path": "/proc/self/fd/777/source-model"}

    requested = tmp_path / "reviewed-source-model"
    runner._bind_checkpoint_serialization_paths(
        tokenizer=Tokenizer(),
        policy_model=policy,
        requested_model=requested,
    )
    expected_config = runner._expected_saved_peft_config(
        policy_model=policy,
        requested_model=requested,
    )
    checkpoint = tmp_path / "checkpoint"
    policy.save_pretrained(checkpoint)
    assert json.loads((checkpoint / "adapter_config.json").read_text()) == (
        expected_config
    )
    for path in checkpoint.rglob("*"):
        if path.is_file() and path.suffix.lower() in {
            ".json",
            ".jsonl",
            ".txt",
            ".md",
            ".yaml",
            ".yml",
        }:
            assert b"/proc/self/fd/" not in path.read_bytes(), path


def test_t6_uses_shared_checkpoint_saver_without_unbound_tokenizer_artifacts() -> None:
    project = Path(__file__).parents[1]
    stable = (project / "scripts/train_ppo_stable.py").read_text(encoding="utf-8")
    trainer = (project / "scripts/train_tastemolnet_gnn_ppo.py").read_text(
        encoding="utf-8"
    )
    assert "save_tokenizer=not bool(" in stable
    assert "args.skip_tokenizer_checkpoint = True" in trainer
    assert '"tokenizer_checkpoint_saved": False' in trainer


def test_t6_terminal_layout_rejects_an_extra_empty_checkpoint_directory() -> None:
    periodic = "checkpoint-5"
    files = {
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "decoded_chem_value_head.pt",
        f"{periodic}/README.md",
        f"{periodic}/adapter_config.json",
        f"{periodic}/adapter_model.safetensors",
        f"{periodic}/decoded_chem_value_head.pt",
        "candidate_pool.jsonl",
        "policy_provenance.json",
        "downstream_policy_binding.json",
        "parent_selection.json",
        "run_manifest.json",
        "observer_state.json",
        "oracle_provenance.json",
        "reward_manifest.json",
        "gate.json",
        "ppo_gate.json",
        "input_hashes.json",
        "manifest.json",
        "ppo_smoke_manifest.json",
        "state.json",
        "logs/T6_OURS_SMOKE.log",
    }
    runner._assert_terminal_layout(
        {
            "files": {name: {} for name in files},
            "directories": {"logs": {}, periodic: {}},
        },
        updates=5,
    )
    with pytest.raises(RuntimeError, match="directory layout changed"):
        runner._assert_terminal_layout(
            {
                "files": {name: {} for name in files},
                "directories": {"logs": {}, periodic: {}, "checkpoint-999": {}},
            },
            updates=5,
        )


def test_checkpoint_reload_rejects_fd_backed_or_wrong_base_model_metadata(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    hostile_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": "/proc/self/fd/99",
    }
    (checkpoint / "adapter_config.json").write_text(
        json.dumps(hostile_config)
        + "\n",
        encoding="utf-8",
    )
    import torch
    from safetensors.torch import save_file

    save_file(
        {
            "base_model.model.layer.lora_A.weight": torch.ones((2, 3)),
            "base_model.model.layer.lora_B.weight": torch.zeros((4, 2)),
        },
        str(checkpoint / "adapter_model.safetensors"),
    )
    torch.save(
        {"summary.weight": torch.ones((1, 4)), "summary.bias": torch.zeros(1)},
        checkpoint / "decoded_chem_value_head.pt",
    )
    with pytest.raises(ValueError, match="base-model authority drifted"):
        validate_taste_adapter_checkpoint_reload(
            checkpoint,
            expected_base_model_path=tmp_path / "source-model",
            expected_adapter_config={
                **hostile_config,
                "base_model_name_or_path": str(tmp_path / "source-model"),
            },
        )


@pytest.mark.parametrize(
    ("gate_field", "gate_value"),
    (
        (None, None),
        ("strict_flip_count", 0),
        ("three_class_oracle", False),
        ("policy_parameters_changed", False),
        ("input_t2_adoption_binding_sha256", "0" * 64),
    ),
)
def test_published_t6_consumer_reopens_exact_gate_inventory_and_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_field: str | None,
    gate_value: object,
) -> None:
    import torch
    from safetensors.torch import save_file

    output = retained.FreshOutputDirectory.create(tmp_path / "t6-output")
    root = output.path
    periodic = root / "checkpoint-5"
    periodic.mkdir(mode=0o700)
    model = tmp_path / "source-model"
    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": str(model),
    }
    tensor_state = {
        "base_model.model.layer.lora_A.weight": torch.ones((2, 3)),
        "base_model.model.layer.lora_B.weight": torch.zeros((4, 2)),
    }
    value_state = {
        "summary.weight": torch.ones((1, 4)),
        "summary.bias": torch.zeros(1),
    }
    for checkpoint in (root, periodic):
        (checkpoint / "README.md").write_text("---\nbase_model: reviewed\n---\n")
        (checkpoint / "adapter_config.json").write_text(
            json.dumps(adapter_config) + "\n",
            encoding="utf-8",
        )
        save_file(tensor_state, str(checkpoint / "adapter_model.safetensors"))
        torch.save(value_state, checkpoint / "decoded_chem_value_head.pt")
    final_reload = validate_taste_adapter_checkpoint_reload(
        root,
        expected_base_model_path=model,
        expected_adapter_config=adapter_config,
    )
    periodic_reload = validate_taste_adapter_checkpoint_reload(
        periodic,
        expected_base_model_path=model,
        expected_adapter_config=adapter_config,
    )
    rows = [
        {**_row(0 if step % 2 else 2), "step_index": step}
        for step in range(1, 6)
    ]
    candidate_bytes = runner._jsonl_document_bytes(rows)
    (root / "candidate_pool.jsonl").write_bytes(candidate_bytes)
    oracle = _oracle()
    reward = build_taste_reward_manifest(rows, oracle_provenance=oracle)
    observer = _observer()
    observer.checkpoints = [
        {
            "step_index": 5,
            "checkpoint_dir": str(periodic),
            "checkpoint_kind": "periodic",
        }
    ]
    observer.on_finish(
        final_output_dir=root,
        candidate_pool_path=root / "candidate_pool.jsonl",
        candidate_count=len(rows),
        global_step=5,
        validation_state={
            "best_val_score": None,
            "best_step": None,
            "stale_eval_count": 0,
        },
        early_stop_reason=None,
    )
    policy_after = final_reload["adapter_parameter_sha256"]
    value_after = final_reload["value_head_parameter_sha256"]
    t2_binding = _t2_adoption_binding()
    t2_hashes = _t2_gate_hashes(t2_binding)
    gate = build_taste_smoke_gate(
        policy_parameter_hash_before="1" * 64,
        policy_parameter_hash_after=policy_after,
        reference_parameter_hash_before="3" * 64,
        reference_parameter_hash_after="3" * 64,
        policy_t5_identity_before="e" * 64,
        reference_t5_identity_before="e" * 64,
        reference_t5_identity_after="e" * 64,
        expected_t5_reference_policy_hash="e" * 64,
        observer=observer,
        checkpoint_reload=final_reload,
        periodic_checkpoint_reload=periodic_reload,
        reward_manifest=reward,
        oracle_provenance=oracle,
        policy_initializer_hash="d" * 64,
        **t2_hashes,
        t3_gate_sha256="5" * 64,
        t4_gate_sha256="6" * 64,
        value_head_parameter_sha256=value_after,
    )
    assert gate["status"] == "PASS"
    if gate_field is not None and gate_field != "input_t2_adoption_binding_sha256":
        gate = {**gate, gate_field: gate_value}
    input_t2_hashes = dict(t2_hashes)
    if gate_field == "input_t2_adoption_binding_sha256":
        input_t2_hashes["t2_adoption_binding_sha256"] = str(gate_value)
    documents = {
        "policy_provenance.json": {
            "status": "PASS",
            "frozen_oracle_identity": {
                "t2_adoption_binding": t2_binding,
            },
        },
        "downstream_policy_binding.json": {"status": "PASS"},
        "parent_selection.json": {"status": "PASS"},
        "run_manifest.json": {
            "schema_version": "tastemolnet_ours_ppo_run_v1",
            "stage": "T6_OURS_SMOKE",
            "model_path": str(model),
            "adapter_config_authority": adapter_config,
            "num_classes": 3,
            "source_label": 1,
            "rf_oracle_used": False,
            "validation_loaded": False,
            "calibration_loaded": False,
            "test_loaded": False,
            "tokenizer_checkpoint_saved": False,
        },
        "observer_state.json": observer.state_dict(),
        "oracle_provenance.json": oracle,
        "reward_manifest.json": reward,
        "gate.json": gate,
        "ppo_gate.json": gate,
        "input_hashes.json": {
            "status": "PASS",
            "t2_adoption_binding": t2_binding,
            **input_t2_hashes,
        },
    }
    for name, payload in documents.items():
        (root / name).write_bytes(runner._json_document_bytes(payload))
    manifest = {
        **gate,
        "schema_version": "tastemolnet_ours_ppo_smoke_manifest_v1",
        "output_root": str(root),
        "no_dataset_redistribution": True,
        "candidate_pool_sha256": runner.hashlib.sha256(candidate_bytes).hexdigest(),
        "reward_manifest_sha256": runner.sha256_file(root / "reward_manifest.json"),
        "oracle_provenance_sha256": runner.sha256_file(root / "oracle_provenance.json"),
    }
    for name in ("manifest.json", "ppo_smoke_manifest.json"):
        (root / name).write_bytes(runner._json_document_bytes(manifest))
    state = {
        "stage": "T6_OURS_SMOKE",
        "state": "PASS",
        "status": "PASS",
        "output_root": str(root),
    }
    (root / "state.json").write_bytes(runner._json_document_bytes(state))
    logs = root / "logs"
    logs.mkdir(mode=0o700)
    (logs / "T6_OURS_SMOKE.log").write_text("done\n", encoding="utf-8")
    prepared = retained.prepare_terminal_output(
        output,
        marker_name="PASS",
        marker_payload=b"[TASTE_T6_OURS_PPO_SMOKE_PASS]\n",
    )
    monkeypatch.setattr(
        retained,
        "_link_held_noreplace",
        lambda directory_fd, _source_fd, target: __import__("os").link(
            ".PASS.prepared",
            target,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        ),
    )
    prepared.commit(retained_input_closure=lambda: None)
    prepared.close()
    if gate_field is None:
        evidence = validate_taste_ppo_output(root)
        assert evidence["status"] == "PASS"
        assert (
            evidence["policy_checkpoint_hash"]
            == final_reload["policy_checkpoint_hash"]
        )
    elif gate_field == "input_t2_adoption_binding_sha256":
        with pytest.raises(ValueError, match="T2 adoption"):
            validate_taste_ppo_output(root)
    else:
        with pytest.raises(ValueError, match="cannot be independently derived"):
            validate_taste_ppo_output(root)


def test_reported_output_filter_removes_descriptor_paths_from_logs() -> None:
    stream = io.StringIO()
    logger = logging.getLogger("test_taste_t6_reported_output_filter")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    logger.addFilter(
        runner._ReportedOutputPathFilter(
            Path("/proc/self/fd/47"),
            Path("/private/reviewed/t6-output"),
        )
    )
    logger.info(
        "output=%s nested=%s",
        "/proc/self/fd/47/checkpoint-5",
        [Path("/proc/self/fd/47/candidate_pool.jsonl")],
    )
    handler.flush()
    rendered = stream.getvalue()
    assert "/proc/self/fd/" not in rendered
    assert "/private/reviewed/t6-output/checkpoint-5" in rendered
    assert "/private/reviewed/t6-output/candidate_pool.jsonl" in rendered


def test_formal_t6_entrypoints_freeze_real_stable_loop_and_no_test_access() -> None:
    project = Path(__file__).parents[1]
    trainer = (project / "scripts/train_tastemolnet_gnn_ppo.py").read_text(
        encoding="utf-8"
    )
    wrapper = (
        project / "scripts/autodl/run_tastemolnet_ours_ppo_smoke.sh"
    ).read_text(encoding="utf-8")
    slurm = (
        project / "scripts/slurm/train_tastemolnet_gnn_ppo.sh"
    ).read_text(encoding="utf-8")
    assert "run_stable_decoded_chem_ppo_loop(" in trainer
    assert "args.val_dataset_path = None" in trainer
    assert '"validation_loaded": False' in trainer
    assert '"calibration_loaded": False' in trainer
    assert '"test_loaded": False' in trainer
    assert "BatchedGNNPPORewardAdapter.from_payloads(" in trainer
    assert "BatchedGNNPPORewardAdapter.from_checkpoint(" not in trainer
    assert "hold_taste_managed_evidence_binding_v2(" in trainer
    assert "managed_evidence_binding_v2" in trainer
    assert "_build_taste_managed_binding_v2(" in trainer
    assert "read_frozen_gine_payload(name)" in trainer
    assert "hold_readonly_file(train_csv" in trainer
    assert "FreshOutputDirectory" in trainer
    assert "prepare_terminal_output(" in trainer
    assert "args.candidate_pool_path = None" in trainer
    assert "checkpoint_path_is_retained=True" in trainer
    assert "load_tastemolnet_train_prompts(" in trainer
    assert "load_stable_prompt_examples(" not in trainer
    assert "load_tastemolnet_downstream_policy(" in trainer
    assert '"frozen_train_csv"' in trainer
    assert "prompt_pool_bound = max(256, int(args.parent_count) * 16)" in trainer
    assert 'dataset="tastemolnet"' in trainer
    assert "num_classes=3" in trainer
    assert "source_label=1" in trainer
    release = json.loads(runner.T6_RELEASE_CONFIG_PATH.read_text(encoding="utf-8"))
    if release["release_enabled"] is False:
        assert "TASTE_T6_WRAPPER_RELEASED=0" in wrapper
        assert "TASTE_T6_WRAPPER_NOT_RELEASED" in wrapper
        assert wrapper.index("TASTE_T6_WRAPPER_NOT_RELEASED") < wrapper.index("GPU_JSON=")
    else:
        assert release["release_enabled"] is True
        assert "TASTE_T6_WRAPPER_RELEASED=1" in wrapper
        assert "--gpu-index 0" in wrapper
    assert "TASTEMOLNET_T3_CHECKPOINT" in wrapper
    assert '--gnn-checkpoint "$TASTEMOLNET_T3_CHECKPOINT"' in wrapper
    assert "physical GPU0" in wrapper
    assert "--gpu-lock-mode exclusive" in wrapper
    assert '--updates "${TASTEMOLNET_T6_UPDATES:-5}"' in wrapper
    assert '--downstream-policy "$DOWNSTREAM_POLICY"' in wrapper
    assert '--base-policy "$BASE_POLICY"' in wrapper
    assert "--input-manifest \"$TASTEMOLNET_T5_OUTPUT/verification.json\"" in wrapper
    assert "--required-output-file downstream_policy_binding.json" in wrapper
    assert "--required-output-file adapter_model.safetensors" in wrapper
    assert "adapter_model.safetensors|adapter_model.bin" not in wrapper
    assert TASTE_PPO_MARKER == "[TASTE_T6_OURS_PPO_SMOKE_PASS]"
    assert f"--required-log-marker '{TASTE_PPO_MARKER}'" in wrapper
    assert "exit 64" in slurm
    assert "--config configs/hpc.yaml" in slurm
    assert "inference.fallback_to_heuristic=false" in slurm


def test_t6_checkpoint_contract_accepts_only_frozen_train_and_three_class_gine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    train = tmp_path / "train.csv"
    validation = tmp_path / "validation.csv"
    calibration = tmp_path / "calibration.csv"
    test = tmp_path / "test.csv"
    for path in (train, validation, calibration, test):
        path.write_text("molecule_id,parent_smiles,label\n1,CCO,1\n", encoding="utf-8")
    (checkpoint / "model.pt").write_bytes(b"model")
    (checkpoint / "feature_schema.json").write_text(
        json.dumps({"schema_sha256": "e" * 64}) + "\n", encoding="utf-8"
    )
    (checkpoint / "temperature_scaling.json").write_text(
        json.dumps(
            {
                "status": "fit",
                "temperature": 1.5,
                "selection_split": "validation",
                "test_used_for_fit": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (checkpoint / "label_map.json").write_text(
        json.dumps({"0": "Bitter", "1": "Sweet", "2": "Tasteless"}) + "\n",
        encoding="utf-8",
    )
    model_hash = runner.sha256_file(checkpoint / "model.pt")
    (checkpoint / "model_card.json").write_text(
        json.dumps(
            {
                "dataset": "tastemolnet",
                "oracle_backend": "gnn",
                "rf_oracle_used": False,
                "backbone": "gine",
                "num_classes": 3,
                "source_label": 1,
                "profile": "full",
                "checkpoint_id": model_hash,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    train_manifest = {
        "schema_version": "molecular_graph_dataset_v1",
        "num_records": 3,
        "num_classes": 3,
        "label_counts": {"0": 1, "1": 1, "2": 1},
        "split_counts": {"train": 3},
        "source_path": str(train),
        "source_sha256": runner.sha256_file(train),
        "dataset_fingerprint": "d" * 64,
        "feature_schema_sha256": "e" * 64,
    }
    split = {
        "schema_version": "molecular_gnn_split_manifest_v1",
        "dataset": "tastemolnet",
        "roles": {
            "train": "model_fitting",
            "validation": "checkpoint_selection_and_temperature_calibration",
            "calibration": "reserved_for_threshold_and_selector_only",
            "test": "frozen_model_final_quality_evaluation",
        },
        "files": {
            name: {"path": str(path), "sha256": runner.sha256_file(path)}
            for name, path in {
                "train": train,
                "validation": validation,
                "calibration": calibration,
                "test": test,
            }.items()
        },
        "train_manifest": train_manifest,
        "validation_manifest": {
            "schema_version": "molecular_graph_dataset_v1"
        },
        "calibration_loaded_for_training": False,
        "test_loaded_for_training": False,
        "test_evaluated_during_training": False,
        "test_used_for_checkpoint_selection": False,
    }
    (checkpoint / "split_manifest.json").write_text(
        json.dumps(split) + "\n", encoding="utf-8"
    )
    test_status = {
        "schema_version": "molecular_gnn_test_evaluation_status_v1",
        "status": "NOT_EVALUATED",
        "test_loaded": False,
        "reason": "held_out_until_frozen_final_evaluation",
        "path": str(test),
        "sha256": runner.sha256_file(test),
    }
    (checkpoint / "test_evaluation_status.json").write_text(
        json.dumps(test_status) + "\n", encoding="utf-8"
    )
    frozen = {
        "checkpoint_sha256": model_hash,
        "feature_schema_sha256": "e" * 64,
        "temperature_calibration_sha256": runner.sha256_file(
            checkpoint / "temperature_scaling.json"
        ),
    }
    assert frozen["feature_schema_sha256"] != runner.sha256_file(
        checkpoint / "feature_schema.json"
    )
    payloads = {
        name: (checkpoint / name).read_bytes()
        for name in (
            "model.pt",
            "model_card.json",
            "feature_schema.json",
            "label_map.json",
            "split_manifest.json",
            "test_evaluation_status.json",
            "temperature_scaling.json",
        )
    }
    checkpoint_evidence = {
        "checkpoint_inventory_sha256": "1" * 64,
        "checkpoint_stat_inventory_sha256": "2" * 64,
        "checkpoint_sha256s_sha256": "3" * 64,
    }
    contract, expected_train, train_sha, count, counts = (
        runner._checkpoint_and_train_contract(
            checkpoint,
            frozen_oracle=frozen,
            checkpoint_evidence=checkpoint_evidence,
            payloads=payloads,
        )
    )
    assert expected_train == train
    assert train_sha == runner.sha256_file(train)
    assert count == 3
    assert counts == {"0": 1, "1": 1, "2": 1}
    assert contract["train_loaded"] is True
    assert contract["validation_loaded"] is False
    assert contract["calibration_loaded"] is False
    assert contract["test_loaded"] is False
    split["train_manifest"] = {**train_manifest, "source_path": str(calibration)}
    payloads["split_manifest.json"] = (json.dumps(split) + "\n").encode("utf-8")
    with pytest.raises(ValueError, match="train manifest authority changed"):
        runner._checkpoint_and_train_contract(
            checkpoint,
            frozen_oracle=frozen,
            checkpoint_evidence=checkpoint_evidence,
            payloads=payloads,
        )


def test_t6_output_rejects_input_ancestor_or_descendant_before_creation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="overlaps immutable input"):
        runner._fresh_output(checkpoint / "child", inputs=[checkpoint])
    assert not (checkpoint / "child").exists()
    parent = tmp_path / "fresh-parent"
    parent.mkdir()
    with pytest.raises(ValueError, match="overlaps immutable input"):
        runner._fresh_output(parent, inputs=[parent / "future-input"])


def test_t6_execution_release_is_either_disabled_or_fully_pinned() -> None:
    release = json.loads(runner.T6_RELEASE_CONFIG_PATH.read_text(encoding="utf-8"))
    assert release["gpu_index"] == 0
    pinned = {
        key: value
        for key, value in release.items()
        if key not in {"schema_version", "release_enabled", "release_state", "gpu_index"}
    }
    if release["release_enabled"] is False:
        assert release["release_state"] == (
            "RELEASE_DISABLED_PENDING_INTEGRATION_COMMIT_AND_EXTERNAL_AUTHORITY"
        )
        assert all(value is None for value in pinned.values())
        with pytest.raises(RuntimeError, match="TASTE_T6_EXECUTION_NOT_RELEASED"):
            runner._assert_execution_released()
    else:
        assert release["release_enabled"] is True
        assert release["release_state"] == "RELEASED_BY_EXTERNAL_EXECUTION_AUTHORITY"
        assert all(value is not None for value in pinned.values())
        assert runner._assert_execution_released() == release


def test_t6_main_rejects_release_before_config_or_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Parser:
        def parse_args(self, _argv: object) -> object:
            return object()

    monkeypatch.setattr(runner, "build_parser", lambda: Parser())
    monkeypatch.setattr(
        runner,
        "apply_config_overrides",
        lambda *_args, **_kwargs: calls.append("config"),
    )
    monkeypatch.setattr(runner, "run", lambda _args: calls.append("run"))
    monkeypatch.setattr(
        runner,
        "_assert_execution_released",
        lambda: (_ for _ in ()).throw(
            RuntimeError("TASTE_T6_EXECUTION_NOT_RELEASED")
        ),
    )
    with pytest.raises(RuntimeError, match="TASTE_T6_EXECUTION_NOT_RELEASED"):
        runner.main([])
    assert calls == []
