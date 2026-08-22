from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

from src.train.bace_gnn_ppo import (
    B6_V2_SCHEMA,
    BacePPOObserver,
    REWARD_PROVENANCE_FIELDS,
    build_ppo_gate,
    build_reward_manifest,
    model_parameter_hash,
    validate_adapter_checkpoint_reload,
    validate_b6_v2_predecessor,
)


class _OneLoRAParameter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.tensor([0.5]))

    def forward(self) -> torch.Tensor:
        return self.lora_weight.square().sum()


def _reward_row() -> dict:
    row = {field: None for field in REWARD_PROVENANCE_FIELDS}
    row.update(
        {
            "dataset": "bace",
            "parent_id": "p0",
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
            "pred_before": 1,
            "pred_after": 0,
            "p_before": 0.9,
            "p_after": 0.1,
            "cf_drop": 0.8,
            "cf_flip": True,
            "reward_valid": 0.25,
            "reward_substructure": 1.0,
            "reward_cf": 4.4,
            "reward_size": 0.25,
            "reward_projection": 0.0,
            "reward_kl": 0.0,
            "reward_total": 5.9,
            "oracle_backend": "gnn",
            "rf_oracle_used": False,
            "calibration_loaded": False,
            "calibration_dataset_loaded": False,
            "frozen_temperature_calibration_loaded": True,
            "test_loaded": False,
            "oracle_checkpoint_hash": "gine-sha",
            "temperature": 1.2,
            "policy_initializer_hash": "clean-init-sha",
            "reference_policy_hash": "reference-sha",
            "gnn_scored_deletion": True,
        }
    )
    return row


def _metrics(reward_mean: float = 1.0) -> dict:
    return {
        "reward_mean": reward_mean,
        "reward_min": reward_mean,
        "reward_max": reward_mean,
        "policy_loss": -0.1,
        "value_loss": 0.1,
        "total_loss": -0.05,
        "approx_kl": 0.01,
        "parse_ok_rate": 1.0,
    }


def _oracle_provenance() -> dict:
    return {
        "dataset": "bace",
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "source_label": 1,
        "num_classes": 2,
        "rf_oracle_used": False,
        "calibration_loaded": False,
        "calibration_dataset_loaded": False,
        "frozen_temperature_calibration_loaded": True,
        "test_loaded": False,
    }


def _checkpoint_artifact(checkpoint_hash: str = "a" * 64) -> dict:
    return {
        "checkpoint_reload_pass": True,
        "policy_checkpoint_hash": checkpoint_hash,
        "policy_checkpoint_hash_schema": "bace_lora_checkpoint_identity_v1",
        "policy_checkpoint_hash_payload": {
            "schema_version": "bace_lora_checkpoint_identity_v1"
        },
        "adapter_config": {
            "path": "/checkpoint/adapter_config.json",
            "size": 10,
            "sha256": "b" * 64,
        },
        "adapter_weights": {
            "path": "/checkpoint/adapter_model.safetensors",
            "size": 20,
            "sha256": "c" * 64,
        },
    }


def _canary_preflight(*, real_gnn_inference: bool = True) -> dict:
    delta = 1 if real_gnn_inference else 0
    return {
        "schema_version": "bace_gnn_ppo_canary_connected_deletion_preflight_v1",
        "stage": "BACE_GNN_PPO_ADAPTER_CANARY",
        "status": "PASS" if real_gnn_inference else "FAIL",
        "dataset": "bace",
        "source_split": "train",
        "source_parent_count": 8,
        "train_csv_sha256": "d" * 64,
        "checkpoint_split_manifest_sha256": "e" * 64,
        "source_parent_ids_sha256": "f" * 64,
        "frozen_train_contract_pass": True,
        "parents": [
            {
                "parent_id": f"train-{index}",
                "source_split": "train",
                "source_label": 1,
                "parent_smiles_sha256": "a" * 64,
            }
            for index in range(8)
        ],
        "adapter_instance_reused": True,
        "oracle_load_count_before": 1,
        "oracle_load_count_after": 1,
        "oracle_prediction_batch_delta": delta,
        "scored_deletion_count_delta": delta,
        "gnn_scored_deletion_count": delta,
        "real_gnn_inference_observed": real_gnn_inference,
        "train_only_contract_pass": True,
        "oracle_backend": "gnn",
        "rf_oracle_used": False,
        "calibration_loaded": False,
        "calibration_dataset_loaded": False,
        "frozen_temperature_calibration_loaded": True,
        "test_loaded": False,
        "formal_b6_v2": False,
        "releases_b7": False,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materialize_passing_b6_root(root: Path) -> tuple[Path, dict[str, str]]:
    gnn_checkpoint = root / "frozen-gine"
    gnn_checkpoint.mkdir(parents=True)
    periodic = root / "checkpoint-5"
    periodic.mkdir(parents=True)
    adapter_config = {"peft_type": "LORA", "r": 8}
    _write_json(root / "adapter_config.json", adapter_config)
    _write_json(periodic / "adapter_config.json", adapter_config)
    weights = {"lora_weight": torch.tensor([0.25])}
    save_file(weights, root / "adapter_model.safetensors")
    save_file(weights, periodic / "adapter_model.safetensors")
    final_reload = validate_adapter_checkpoint_reload(root)
    periodic_reload = validate_adapter_checkpoint_reload(periodic)

    observer = BacePPOObserver()
    for step in range(1, 6):
        observer.on_update(
            step_index=step,
            batch_ids=["p0"],
            reward_logs=[_reward_row()],
            metrics=_metrics(),
        )
    observer.on_checkpoint(
        step_index=5,
        checkpoint_dir=periodic,
        checkpoint_kind="periodic",
    )
    observer.on_finish(final_output_dir=root, global_step=5)
    checkpoint_id = "gine-sha"
    policy_initializer_hash = "4" * 64
    reference_policy_hash = "3" * 64
    oracle = {
        **_oracle_provenance(),
        "schema_version": "bace_gnn_ppo_oracle_provenance_v1",
        "checkpoint_dir": str(gnn_checkpoint),
        "checkpoint_id": checkpoint_id,
        "backbone": "gine",
        "temperature": 1.2,
        "temperature_calibration_hash": "temperature-sha",
        "feature_schema_hash": "feature-sha",
        "oracle_load_count": 1,
        "gnn_scored_deletion_count": 1,
    }
    row = {
        **_reward_row(),
        "oracle_checkpoint_hash": checkpoint_id,
        "policy_initializer_hash": policy_initializer_hash,
        "reference_policy_hash": reference_policy_hash,
    }
    candidate_path = root / "candidate_pool.jsonl"
    candidate_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    oracle_path = root / "oracle_provenance.json"
    _write_json(oracle_path, oracle)
    reward = build_reward_manifest([row], oracle_provenance=oracle)
    reward_path = root / "reward_manifest.json"
    _write_json(reward_path, reward)
    gate = build_ppo_gate(
        stage="B6_PPO_SMOKE_V2",
        policy_parameter_hash_before="1" * 64,
        policy_parameter_hash_after="2" * 64,
        reference_parameter_hash_before=reference_policy_hash,
        reference_parameter_hash_after=reference_policy_hash,
        observer=observer,
        checkpoint_reload=final_reload,
        periodic_checkpoint_reload=periodic_reload,
        reward_manifest=reward,
        oracle_provenance=oracle,
        expected_checkpoints=(5,),
    )
    assert gate["schema_version"] == B6_V2_SCHEMA
    assert gate["status"] == "PASS"
    gate_path = root / "ppo_gate.json"
    _write_json(gate_path, gate)
    identities = {
        "checkpoint_id": checkpoint_id,
        "gnn_checkpoint": str(gnn_checkpoint),
        "policy_initializer_hash": policy_initializer_hash,
        "git_commit": "5" * 40,
    }
    manifest = {
        **gate,
        **identities,
        "reference_policy_hash": reference_policy_hash,
        "stable_loop": "scripts.train_ppo_stable.run_stable_decoded_chem_ppo_loop",
        "shared_algorithm_reimplemented": False,
        "policy_checkpoint_hash": final_reload["policy_checkpoint_hash"],
        "policy_checkpoint_hash_payload": final_reload[
            "policy_checkpoint_hash_payload"
        ],
        "final_adapter_config_identity": final_reload["adapter_config"],
        "final_adapter_weights_identity": final_reload["adapter_weights"],
        "last_periodic_checkpoint_step": 5,
        "last_periodic_policy_checkpoint_hash": periodic_reload[
            "policy_checkpoint_hash"
        ],
        "last_periodic_policy_checkpoint_hash_payload": periodic_reload[
            "policy_checkpoint_hash_payload"
        ],
        "last_periodic_adapter_config_identity": periodic_reload["adapter_config"],
        "last_periodic_adapter_weights_identity": periodic_reload["adapter_weights"],
        "candidate_pool": str(candidate_path),
        "candidate_pool_sha256": _sha256(candidate_path),
        "reward_manifest": str(reward_path),
        "reward_manifest_sha256": _sha256(reward_path),
        "oracle_provenance": str(oracle_path),
        "oracle_provenance_sha256": _sha256(oracle_path),
        "ppo_gate_sha256": _sha256(gate_path),
        "output_root": str(root),
    }
    manifest_path = root / "ppo_smoke_manifest.json"
    _write_json(manifest_path, manifest)
    (root / "PASS").write_text("[BACE_B6_V2_PASS]\n", encoding="utf-8")
    return manifest_path, identities


def test_b6_v2_requires_real_updates_parameter_change_and_reload(
    tmp_path: Path,
) -> None:
    policy = _OneLoRAParameter()
    reference = _OneLoRAParameter()
    reference.load_state_dict(policy.state_dict())
    before = model_parameter_hash(policy, trainable_only=True)
    reference_before = model_parameter_hash(reference, adapter_only=True)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1)
    observer = BacePPOObserver()
    for step in range(1, 6):
        optimizer.zero_grad(set_to_none=True)
        policy().backward()
        optimizer.step()
        observer.on_update(
            step_index=step,
            batch_ids=["p0"],
            reward_logs=[_reward_row()],
            metrics=_metrics(),
        )
    observer.on_checkpoint(
        step_index=5, checkpoint_dir=tmp_path / "checkpoint-5", checkpoint_kind="periodic"
    )
    observer.on_finish(final_output_dir=tmp_path, global_step=5)
    after = model_parameter_hash(policy, trainable_only=True)
    reference_after = model_parameter_hash(reference, adapter_only=True)

    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 8}), encoding="utf-8"
    )
    save_file({"lora_weight": policy.lora_weight.detach()}, tmp_path / "adapter_model.safetensors")
    reload_gate = validate_adapter_checkpoint_reload(tmp_path)
    assert reload_gate["policy_checkpoint_hash"] == hashlib.sha256(
        json.dumps(
            reload_gate["policy_checkpoint_hash_payload"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    reward_manifest = build_reward_manifest(
        [_reward_row()], oracle_provenance=_oracle_provenance()
    )
    gate = build_ppo_gate(
        stage="B6_PPO_SMOKE_V2",
        policy_parameter_hash_before=before,
        policy_parameter_hash_after=after,
        reference_parameter_hash_before=reference_before,
        reference_parameter_hash_after=reference_after,
        observer=observer,
        checkpoint_reload=reload_gate,
        periodic_checkpoint_reload=reload_gate,
        reward_manifest=reward_manifest,
        oracle_provenance=_oracle_provenance(),
        expected_checkpoints=(5,),
    )
    assert before != after
    assert reference_before == reference_after
    assert gate["ppo_update_count"] == 5
    assert gate["optimizer_step_count"] == 5
    assert gate["checkpoint_reload_pass"] is True
    assert gate["checkpoint_artifact_bound"] is True
    assert gate["final_matches_last_periodic_checkpoint"] is True
    assert len(gate["policy_checkpoint_hash"]) == 64
    assert gate["status"] == "PASS"


def test_scoring_without_five_optimizer_updates_cannot_pass_b6() -> None:
    observer = BacePPOObserver()
    observer.on_finish(final_output_dir=Path("unused"), global_step=0)
    gate = build_ppo_gate(
        stage="B6_PPO_SMOKE_V2",
        policy_parameter_hash_before="same",
        policy_parameter_hash_after="same",
        reference_parameter_hash_before="ref",
        reference_parameter_hash_after="ref",
        observer=observer,
        checkpoint_reload=_checkpoint_artifact(),
        periodic_checkpoint_reload=_checkpoint_artifact(),
        reward_manifest=build_reward_manifest(
            [_reward_row()], oracle_provenance=_oracle_provenance()
        ),
        oracle_provenance=_oracle_provenance(),
    )
    assert gate["status"] == "FAIL"
    assert "minimum_update_count_met" in gate["failures"]
    assert "policy_parameters_changed" in gate["failures"]


def test_canary_is_independent_non_formal_and_cannot_release_b7() -> None:
    observer = BacePPOObserver()
    observer.on_update(
        step_index=1,
        batch_ids=["p0"],
        reward_logs=[_reward_row()],
        metrics=_metrics(),
    )
    observer.on_finish(final_output_dir=Path("canary"), global_step=1)
    ppo_row = _reward_row()
    ppo_row.update(
        {
            "deletion_valid": False,
            "gnn_scored_deletion": False,
            "cf_flip": False,
        }
    )
    gate = build_ppo_gate(
        stage="BACE_GNN_PPO_ADAPTER_CANARY",
        policy_parameter_hash_before="policy-before",
        policy_parameter_hash_after="policy-after",
        reference_parameter_hash_before="reference",
        reference_parameter_hash_after="reference",
        observer=observer,
        checkpoint_reload=_checkpoint_artifact(),
        periodic_checkpoint_reload=_checkpoint_artifact(),
        reward_manifest=build_reward_manifest([ppo_row], oracle_provenance=_oracle_provenance()),
        oracle_provenance=_oracle_provenance(),
        canary_preflight=_canary_preflight(),
    )
    assert gate["status"] == "PASS"
    assert gate["stage"] == "BACE_GNN_PPO_ADAPTER_CANARY"
    assert gate["formal_b6_v2"] is False
    assert gate["releases_b7"] is False
    assert gate["ppo_generated_gnn_scored_deletion_count"] == 0
    assert gate["canary_preflight_real_gnn_inference"] is True


def test_canary_cannot_pass_without_real_same_adapter_gnn_inference() -> None:
    observer = BacePPOObserver()
    observer.on_update(
        step_index=1,
        batch_ids=["p0"],
        reward_logs=[_reward_row()],
        metrics=_metrics(),
    )
    observer.on_finish(final_output_dir=Path("canary"), global_step=1)
    gate = build_ppo_gate(
        stage="BACE_GNN_PPO_ADAPTER_CANARY",
        policy_parameter_hash_before="policy-before",
        policy_parameter_hash_after="policy-after",
        reference_parameter_hash_before="reference",
        reference_parameter_hash_after="reference",
        observer=observer,
        checkpoint_reload=_checkpoint_artifact(),
        periodic_checkpoint_reload=_checkpoint_artifact(),
        reward_manifest=build_reward_manifest(
            [_reward_row()], oracle_provenance=_oracle_provenance()
        ),
        oracle_provenance=_oracle_provenance(),
        canary_preflight=_canary_preflight(real_gnn_inference=False),
    )
    assert gate["status"] == "FAIL"
    assert "canary_preflight_real_gnn_inference" in gate["failures"]


def test_b6_generated_candidate_gnn_gate_ignores_canary_preflight() -> None:
    observer = BacePPOObserver()
    no_gnn_row = _reward_row()
    no_gnn_row.update(
        {
            "deletion_valid": False,
            "gnn_scored_deletion": False,
            "cf_flip": False,
        }
    )
    for step in range(1, 6):
        observer.on_update(
            step_index=step,
            batch_ids=["p0"],
            reward_logs=[no_gnn_row],
            metrics=_metrics(),
        )
    observer.on_checkpoint(
        step_index=5,
        checkpoint_dir=Path("checkpoint-5"),
        checkpoint_kind="periodic",
    )
    observer.on_finish(final_output_dir=Path("b6"), global_step=5)
    gate = build_ppo_gate(
        stage="B6_PPO_SMOKE_V2",
        policy_parameter_hash_before="policy-before",
        policy_parameter_hash_after="policy-after",
        reference_parameter_hash_before="reference",
        reference_parameter_hash_after="reference",
        observer=observer,
        checkpoint_reload=_checkpoint_artifact(),
        periodic_checkpoint_reload=_checkpoint_artifact(),
        reward_manifest=build_reward_manifest(
            [no_gnn_row], oracle_provenance=_oracle_provenance()
        ),
        oracle_provenance=_oracle_provenance(),
        canary_preflight=_canary_preflight(),
        expected_checkpoints=(5,),
    )
    assert gate["status"] == "FAIL"
    assert gate["at_least_one_gnn_scored_deletion"] is False
    assert "at_least_one_gnn_scored_deletion" in gate["failures"]


def test_b7_contract_requires_300_updates_and_all_six_periodic_checkpoints() -> None:
    observer = BacePPOObserver()
    for step in range(1, 301):
        observer.on_update(
            step_index=step,
            batch_ids=["p0"],
            reward_logs=[_reward_row()],
            metrics=_metrics(reward_mean=1.0),
        )
    for step in (50, 100, 150, 200, 250, 300):
        observer.on_checkpoint(
            step_index=step,
            checkpoint_dir=Path(f"checkpoint-{step}"),
            checkpoint_kind="periodic",
        )
    observer.on_finish(final_output_dir=Path("final"), global_step=300)
    gate = build_ppo_gate(
        stage="B7_PPO_FULL",
        policy_parameter_hash_before="policy-before",
        policy_parameter_hash_after="policy-after",
        reference_parameter_hash_before="reference",
        reference_parameter_hash_after="reference",
        observer=observer,
        checkpoint_reload=_checkpoint_artifact(),
        periodic_checkpoint_reload=_checkpoint_artifact(),
        reward_manifest=build_reward_manifest(
            [_reward_row()], oracle_provenance=_oracle_provenance()
        ),
        oracle_provenance=_oracle_provenance(),
        expected_checkpoints=(50, 100, 150, 200, 250, 300),
    )
    assert gate["exact_update_count_met"] is True
    assert gate["expected_checkpoints_saved"] is True
    assert gate["last_50_reward_collapse"] is False
    assert gate["status"] == "PASS"


def test_b7_deeply_revalidates_b6_manifest_and_physical_adapter_bytes(
    tmp_path: Path,
) -> None:
    manifest_path, identities = _materialize_passing_b6_root(tmp_path)
    result = validate_b6_v2_predecessor(
        manifest_path,
        checkpoint_id=identities["checkpoint_id"],
        gnn_checkpoint=identities["gnn_checkpoint"],
        policy_initializer_hash=identities["policy_initializer_hash"],
        git_commit=identities["git_commit"],
    )
    assert result["manifest"]["ppo_update_count"] == 5
    assert result["validated_policy_checkpoint"]["checkpoint_reload_pass"] is True

    # Keep the adapter valid and reloadable while changing its physical bytes;
    # a shallow manifest-only check would miss this mutation.
    save_file(
        {"lora_weight": torch.tensor([0.75])},
        tmp_path / "adapter_model.safetensors",
    )
    with pytest.raises(ValueError, match="mutated B6-v2 final adapter bytes"):
        validate_b6_v2_predecessor(
            manifest_path,
            checkpoint_id=identities["checkpoint_id"],
            gnn_checkpoint=identities["gnn_checkpoint"],
            policy_initializer_hash=identities["policy_initializer_hash"],
            git_commit=identities["git_commit"],
        )


def test_b7_rejects_b6_candidate_bytes_drift(tmp_path: Path) -> None:
    manifest_path, identities = _materialize_passing_b6_root(tmp_path)
    candidate_path = tmp_path / "candidate_pool.jsonl"
    original = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate_path.write_text(
        json.dumps({**original, "reward_total": 4.2}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="artifact hash drift"):
        validate_b6_v2_predecessor(
            manifest_path,
            checkpoint_id=identities["checkpoint_id"],
            gnn_checkpoint=identities["gnn_checkpoint"],
            policy_initializer_hash=identities["policy_initializer_hash"],
            git_commit=identities["git_commit"],
        )


def test_official_runner_calls_shared_stable_loop_and_has_no_private_optimizer() -> None:
    source = (Path(__file__).parents[1] / "scripts" / "train_bace_gnn_ppo.py").read_text(
        encoding="utf-8"
    )
    assert "run_stable_decoded_chem_ppo_loop(" in source
    assert "torch.optim" not in source
    assert "build_policy_model(" in source
    assert "build_value_model(" in source
    assert "reference_model = build_policy_model(" in source
    assert '"final_adapter_config_identity"' in source
    assert '"final_adapter_weights_identity"' in source
    assert '"policy_checkpoint_hash"' in source


def test_foreground_wrapper_canary_argv_reaches_independent_runner(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[1]
    data_root = tmp_path / "persistent"
    runtime_root = data_root / "counterfactual-subgraph-runtime"
    output = runtime_root / "outputs" / "canary"
    data_root.mkdir()
    environment = {
        **os.environ,
        "AUTODL_PYTHON": sys.executable,
        "AUTODL_DATA_ROOT": str(data_root),
        "AUTODL_RUNTIME_ROOT": str(runtime_root),
        "OUTPUT_ROOT": str(output),
        "CHEMLLM_MODEL_PATH": str(tmp_path / "missing-base"),
        "BACE_GNN_CHECKPOINT": str(tmp_path / "missing-gine"),
        "BACE_TRAIN_CSV": str(tmp_path / "missing-train.csv"),
        "BACE_POLICY_INITIALIZER": str(tmp_path / "missing-adapter"),
        "BACE_POLICY_PROVENANCE_MANIFEST": str(tmp_path / "missing-policy.json"),
    }
    completed = subprocess.run(
        [
            "bash",
            str(repository / "scripts" / "autodl" / "run_bace_gnn_ppo_stage.sh"),
            "BACE_GNN_PPO_ADAPTER_CANARY",
        ],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "invalid choice" not in completed.stderr
    failure_path = output / "FAIL.json"
    assert failure_path.is_file(), (completed.stdout, completed.stderr)
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["stage"] == "BACE_GNN_PPO_ADAPTER_CANARY"
