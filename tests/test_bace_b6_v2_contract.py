from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from safetensors.torch import save_file

from src.train.bace_gnn_ppo import (
    BacePPOObserver,
    REWARD_PROVENANCE_FIELDS,
    build_ppo_gate,
    build_reward_manifest,
    model_parameter_hash,
    validate_adapter_checkpoint_reload,
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
    )
    assert gate["status"] == "PASS"
    assert gate["stage"] == "BACE_GNN_PPO_ADAPTER_CANARY"
    assert gate["formal_b6_v2"] is False
    assert gate["releases_b7"] is False


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
