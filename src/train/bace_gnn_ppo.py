"""Artifact and gate helpers for real BACE GNN-backed stable PPO runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


B6_V2_SCHEMA = "bace_b6_ppo_smoke_v2"
B7_SCHEMA = "bace_b7_ppo_full_v1"
REWARD_PROVENANCE_FIELDS = (
    "dataset",
    "parent_id",
    "parent_smiles",
    "source_label",
    "raw_fragment",
    "core_fragment",
    "final_fragment",
    "parse_ok",
    "connected",
    "direct_substructure",
    "projection_used",
    "deletion_valid",
    "pred_before",
    "pred_after",
    "p_before",
    "p_after",
    "cf_drop",
    "cf_flip",
    "reward_valid",
    "reward_substructure",
    "reward_cf",
    "reward_size",
    "reward_projection",
    "reward_kl",
    "reward_total",
    "oracle_backend",
    "rf_oracle_used",
    "calibration_loaded",
    "calibration_dataset_loaded",
    "frozen_temperature_calibration_loaded",
    "test_loaded",
    "oracle_checkpoint_hash",
    "temperature",
    "policy_initializer_hash",
    "reference_policy_hash",
)


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                dict(payload),
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
                default=lambda value: str(value) if isinstance(value, Path) else value,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return target


def model_parameter_hash(
    model: Any,
    *,
    trainable_only: bool = False,
    adapter_only: bool = False,
) -> str:
    """Hash one explicit parameter subset without serializing a checkpoint."""

    import torch

    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if trainable_only and not bool(parameter.requires_grad):
            continue
        if adapter_only and "lora_" not in str(name).lower():
            continue
        tensor = parameter.detach().contiguous().view(-1)
        digest.update(str(name).encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(str(parameter.dtype).encode("ascii"))
        # Viewing as bytes works for bf16 and quantized dtypes unsupported by NumPy.
        byte_view = tensor.view(torch.uint8)
        digest.update(byte_view.cpu().numpy().tobytes())
        count += 1
    if count == 0:
        scope = "trainable" if trainable_only else "adapter" if adapter_only else "all"
        raise ValueError(f"No parameters found for {scope} hash")
    digest.update(str(count).encode("ascii"))
    return digest.hexdigest()


def validate_adapter_checkpoint_reload(checkpoint: str | Path) -> dict[str, Any]:
    """Deserialize saved adapter tensors and prove the checkpoint is finite."""

    root = Path(checkpoint).expanduser().resolve(strict=True)
    config = root / "adapter_config.json"
    if not config.is_file():
        raise ValueError(f"Adapter config is missing: {root}")
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict) or not config_payload:
        raise ValueError("Adapter config is empty or malformed")
    safe = root / "adapter_model.safetensors"
    binary = root / "adapter_model.bin"
    tensors: Mapping[str, Any]
    if safe.is_file():
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - AutoDL dependency
            raise RuntimeError("Checkpoint reload requires safetensors") from exc
        tensors = load_file(str(safe), device="cpu")
        weights_path = safe
    elif binary.is_file():
        import torch

        try:
            tensors = torch.load(binary, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - old torch
            tensors = torch.load(binary, map_location="cpu")
        weights_path = binary
    else:
        raise ValueError(f"Adapter weights are missing: {root}")
    if not isinstance(tensors, Mapping) or not tensors:
        raise ValueError("Reloaded adapter state is empty")
    import torch

    nonfinite = [
        str(name)
        for name, tensor in tensors.items()
        if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor.float()).all())
    ]
    if nonfinite:
        raise ValueError(f"Reloaded adapter contains invalid tensors: {nonfinite[:5]}")
    config_sha256 = _sha256_file(config)
    weights_sha256 = _sha256_file(weights_path)
    artifact_payload = {
        "schema_version": "bace_lora_checkpoint_identity_v1",
        "adapter_config_name": config.name,
        "adapter_config_sha256": config_sha256,
        "adapter_config_size": config.stat().st_size,
        "adapter_weights_name": weights_path.name,
        "adapter_weights_sha256": weights_sha256,
        "adapter_weights_size": weights_path.stat().st_size,
    }
    return {
        "checkpoint_reload_pass": True,
        "checkpoint": str(root),
        "weights_file": str(weights_path),
        "tensor_count": len(tensors),
        "all_tensors_finite": True,
        "adapter_config": {
            "path": str(config),
            "sha256": config_sha256,
            "size": config.stat().st_size,
        },
        "adapter_weights": {
            "path": str(weights_path),
            "sha256": weights_sha256,
            "size": weights_path.stat().st_size,
        },
        "policy_checkpoint_hash": hashlib.sha256(
            json.dumps(
                artifact_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "policy_checkpoint_hash_schema": "bace_lora_checkpoint_identity_v1",
        "policy_checkpoint_hash_payload": artifact_payload,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BacePPOObserver:
    """Collect the existing stable loop's real post-optimizer callbacks."""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.checkpoints: list[dict[str, Any]] = []
        self.finish: dict[str, Any] | None = None

    def on_update(
        self,
        *,
        step_index: int,
        batch_ids: Sequence[str],
        reward_logs: Sequence[Mapping[str, Any]],
        metrics: Mapping[str, Any],
    ) -> None:
        self.updates.append(
            {
                "step_index": int(step_index),
                "batch_ids": list(batch_ids),
                "reward_row_count": len(reward_logs),
                "metrics": dict(metrics),
            }
        )

    def on_checkpoint(
        self,
        *,
        step_index: int,
        checkpoint_dir: Path,
        checkpoint_kind: str,
    ) -> None:
        self.checkpoints.append(
            {
                "step_index": int(step_index),
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint_kind": str(checkpoint_kind),
            }
        )

    def on_finish(self, **kwargs: Any) -> None:
        self.finish = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in kwargs.items()
        }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def build_reward_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    oracle_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    missing = sorted(
        {
            field
            for row in rows
            for field in REWARD_PROVENANCE_FIELDS
            if field not in row
        }
    )
    reward_values = [row.get("reward_total") for row in rows]
    return {
        "schema_version": "bace_gnn_ppo_reward_manifest_v1",
        "candidate_count": len(rows),
        "required_fields": list(REWARD_PROVENANCE_FIELDS),
        "missing_required_fields": missing,
        "all_reward_values_finite": bool(reward_values)
        and all(_finite(value) for value in reward_values),
        "valid_candidate_count": sum(
            1 for row in rows if bool(row.get("parse_ok") and row.get("connected"))
        ),
        "gnn_scored_deletion_count": sum(
            1 for row in rows if bool(row.get("gnn_scored_deletion"))
        ),
        "strict_flip_count": sum(1 for row in rows if bool(row.get("cf_flip"))),
        "oracle_backend": oracle_provenance.get("oracle_backend"),
        "rf_oracle_used": oracle_provenance.get("rf_oracle_used"),
        "calibration_loaded": oracle_provenance.get("calibration_loaded"),
        "calibration_dataset_loaded": oracle_provenance.get(
            "calibration_dataset_loaded"
        ),
        "frozen_temperature_calibration_loaded": oracle_provenance.get(
            "frozen_temperature_calibration_loaded"
        ),
        "test_loaded": oracle_provenance.get("test_loaded"),
        "candidate_oracle_contract_pass": bool(rows)
        and all(
            row.get("oracle_backend") == "gnn"
            and row.get("rf_oracle_used") is False
            and row.get("calibration_loaded") is False
            and row.get("calibration_dataset_loaded") is False
            and row.get("frozen_temperature_calibration_loaded") is True
            and row.get("test_loaded") is False
            for row in rows
        ),
    }


def build_ppo_gate(
    *,
    stage: str,
    policy_parameter_hash_before: str,
    policy_parameter_hash_after: str,
    reference_parameter_hash_before: str,
    reference_parameter_hash_after: str,
    observer: BacePPOObserver,
    checkpoint_reload: Mapping[str, Any],
    periodic_checkpoint_reload: Mapping[str, Any],
    reward_manifest: Mapping[str, Any],
    oracle_provenance: Mapping[str, Any],
    expected_checkpoints: Sequence[int] = (),
    hard_kl: float = 0.8,
) -> dict[str, Any]:
    updates = list(observer.updates)
    ppo_update_count = len(updates)
    optimizer_steps = ppo_update_count  # fixed ppo_epochs=1 for B6-v2/B7
    observed_checkpoint_steps = sorted(
        {
            int(row["step_index"])
            for row in observer.checkpoints
            if row.get("checkpoint_kind") == "periodic"
        }
    )
    all_metrics_finite = all(
        _finite(value)
        for update in updates
        for key, value in update["metrics"].items()
        if key
        in {
            "reward_mean",
            "reward_min",
            "reward_max",
            "policy_loss",
            "value_loss",
            "total_loss",
            "approx_kl",
        }
    )
    kl_within_hard_limit = bool(updates) and all(
        _finite(update["metrics"].get("approx_kl"))
        and float(update["metrics"]["approx_kl"]) <= float(hard_kl)
        for update in updates
    )
    common = {
        "ppo_training_performed": ppo_update_count > 0,
        "ppo_update_count": ppo_update_count,
        "optimizer_step_count": optimizer_steps,
        "policy_parameter_hash_before": policy_parameter_hash_before,
        "policy_parameter_hash_after": policy_parameter_hash_after,
        "policy_parameters_changed": policy_parameter_hash_before
        != policy_parameter_hash_after,
        "reference_parameter_hash_before": reference_parameter_hash_before,
        "reference_parameter_hash_after": reference_parameter_hash_after,
        "reference_parameters_unchanged": reference_parameter_hash_before
        == reference_parameter_hash_after,
        "checkpoint_saved": bool(observer.finish),
        "checkpoint_reload_pass": checkpoint_reload.get("checkpoint_reload_pass") is True,
        "periodic_checkpoint_reload_pass": periodic_checkpoint_reload.get(
            "checkpoint_reload_pass"
        )
        is True,
        "policy_checkpoint_hash": checkpoint_reload.get("policy_checkpoint_hash"),
        "policy_checkpoint_hash_schema": checkpoint_reload.get(
            "policy_checkpoint_hash_schema"
        ),
        "policy_checkpoint_hash_payload": checkpoint_reload.get(
            "policy_checkpoint_hash_payload"
        ),
        "periodic_policy_checkpoint_hash": periodic_checkpoint_reload.get(
            "policy_checkpoint_hash"
        ),
        "periodic_policy_checkpoint_hash_schema": periodic_checkpoint_reload.get(
            "policy_checkpoint_hash_schema"
        ),
        "checkpoint_artifact_bound": bool(
            checkpoint_reload.get("adapter_config")
            and checkpoint_reload.get("adapter_weights")
            and checkpoint_reload.get("policy_checkpoint_hash")
            and checkpoint_reload.get("policy_checkpoint_hash_schema")
            == "bace_lora_checkpoint_identity_v1"
        ),
        "periodic_checkpoint_artifact_bound": bool(
            periodic_checkpoint_reload.get("adapter_config")
            and periodic_checkpoint_reload.get("adapter_weights")
            and periodic_checkpoint_reload.get("policy_checkpoint_hash")
            and periodic_checkpoint_reload.get("policy_checkpoint_hash_schema")
            == "bace_lora_checkpoint_identity_v1"
        ),
        "final_matches_last_periodic_checkpoint": bool(
            checkpoint_reload.get("policy_checkpoint_hash")
            and checkpoint_reload.get("policy_checkpoint_hash")
            == periodic_checkpoint_reload.get("policy_checkpoint_hash")
        ),
        "final_adapter_config": checkpoint_reload.get("adapter_config"),
        "final_adapter_weights": checkpoint_reload.get("adapter_weights"),
        "last_periodic_adapter_config": periodic_checkpoint_reload.get(
            "adapter_config"
        ),
        "last_periodic_adapter_weights": periodic_checkpoint_reload.get(
            "adapter_weights"
        ),
        "candidate_pool_saved": int(reward_manifest.get("candidate_count", 0)) > 0,
        "reward_manifest_saved": True,
        "oracle_backend": oracle_provenance.get("oracle_backend"),
        "dataset": oracle_provenance.get("dataset"),
        "classifier_type": oracle_provenance.get("classifier_type"),
        "source_label": oracle_provenance.get("source_label"),
        "num_classes": oracle_provenance.get("num_classes"),
        "rf_oracle_used": oracle_provenance.get("rf_oracle_used"),
        "calibration_loaded": oracle_provenance.get("calibration_loaded"),
        "calibration_dataset_loaded": oracle_provenance.get(
            "calibration_dataset_loaded"
        ),
        "frozen_temperature_calibration_loaded": oracle_provenance.get(
            "frozen_temperature_calibration_loaded"
        ),
        "test_loaded": oracle_provenance.get("test_loaded"),
        "gnn_oracle_backend": oracle_provenance.get("oracle_backend") == "gnn",
        "bace_classifier_contract_pass": bool(
            oracle_provenance.get("dataset") == "bace"
            and oracle_provenance.get("classifier_type") == "gnn"
            and oracle_provenance.get("source_label") == 1
            and oracle_provenance.get("num_classes") == 2
        ),
        "rf_guard_pass": oracle_provenance.get("rf_oracle_used") is False,
        "calibration_not_loaded": oracle_provenance.get("calibration_loaded") is False,
        "calibration_dataset_not_loaded": oracle_provenance.get(
            "calibration_dataset_loaded"
        )
        is False,
        "frozen_temperature_calibration_present": oracle_provenance.get(
            "frozen_temperature_calibration_loaded"
        )
        is True,
        "test_not_loaded": oracle_provenance.get("test_loaded") is False,
        "candidate_oracle_contract_pass": reward_manifest.get(
            "candidate_oracle_contract_pass"
        )
        is True,
        "all_reward_values_finite": reward_manifest.get("all_reward_values_finite") is True,
        "all_update_metrics_finite": all_metrics_finite,
        "at_least_one_valid_candidate": int(reward_manifest.get("valid_candidate_count", 0)) > 0,
        "at_least_one_gnn_scored_deletion": int(
            reward_manifest.get("gnn_scored_deletion_count", 0)
        )
        > 0,
        "reward_provenance_complete": not reward_manifest.get("missing_required_fields"),
        "kl_within_hard_limit": kl_within_hard_limit,
        "observed_checkpoint_steps": observed_checkpoint_steps,
        "expected_checkpoint_steps": list(expected_checkpoints),
        "expected_checkpoints_saved": set(expected_checkpoints).issubset(observed_checkpoint_steps),
    }
    if stage == "BACE_GNN_PPO_ADAPTER_CANARY":
        stage_requirements = {
            **common,
            "minimum_update_count_met": ppo_update_count >= 1,
            "bounded_update_count_met": ppo_update_count <= 2,
            "formal_b6_v2": False,
            "releases_b7": False,
        }
        required = (
            "ppo_training_performed",
            "minimum_update_count_met",
            "bounded_update_count_met",
            "policy_parameters_changed",
            "reference_parameters_unchanged",
            "checkpoint_saved",
            "checkpoint_reload_pass",
            "periodic_checkpoint_reload_pass",
            "checkpoint_artifact_bound",
            "periodic_checkpoint_artifact_bound",
            "final_matches_last_periodic_checkpoint",
            "candidate_pool_saved",
            "reward_manifest_saved",
            "all_reward_values_finite",
            "all_update_metrics_finite",
            "at_least_one_valid_candidate",
            "at_least_one_gnn_scored_deletion",
            "reward_provenance_complete",
            "gnn_oracle_backend",
            "bace_classifier_contract_pass",
            "rf_guard_pass",
            "calibration_not_loaded",
            "calibration_dataset_not_loaded",
            "frozen_temperature_calibration_present",
            "test_not_loaded",
            "candidate_oracle_contract_pass",
            "kl_within_hard_limit",
        )
        schema = "bace_gnn_ppo_adapter_canary_v1"
    elif stage == "B6_PPO_SMOKE_V2":
        stage_requirements = {
            **common,
            "minimum_update_count_met": ppo_update_count >= 5,
            "bounded_update_count_met": ppo_update_count <= 10,
        }
        required = (
            "ppo_training_performed",
            "minimum_update_count_met",
            "bounded_update_count_met",
            "policy_parameters_changed",
            "reference_parameters_unchanged",
            "checkpoint_saved",
            "checkpoint_reload_pass",
            "periodic_checkpoint_reload_pass",
            "checkpoint_artifact_bound",
            "periodic_checkpoint_artifact_bound",
            "final_matches_last_periodic_checkpoint",
            "candidate_pool_saved",
            "reward_manifest_saved",
            "all_reward_values_finite",
            "all_update_metrics_finite",
            "at_least_one_valid_candidate",
            "at_least_one_gnn_scored_deletion",
            "reward_provenance_complete",
            "gnn_oracle_backend",
            "bace_classifier_contract_pass",
            "rf_guard_pass",
            "calibration_not_loaded",
            "calibration_dataset_not_loaded",
            "frozen_temperature_calibration_present",
            "test_not_loaded",
            "candidate_oracle_contract_pass",
            "kl_within_hard_limit",
        )
        schema = B6_V2_SCHEMA
    elif stage == "B7_PPO_FULL":
        recent = updates[-50:]
        first = updates[:50]
        recent_mean = (
            sum(float(row["metrics"]["reward_mean"]) for row in recent) / len(recent)
            if recent
            else float("nan")
        )
        first_mean = (
            sum(float(row["metrics"]["reward_mean"]) for row in first) / len(first)
            if first
            else float("nan")
        )
        stage_requirements = {
            **common,
            "exact_update_count_met": ppo_update_count == 300,
            "candidate_syntax_not_collapsed": bool(recent)
            and any(float(row["metrics"].get("parse_ok_rate", 0.0)) > 0.0 for row in recent),
            "last_50_reward_collapse": not (
                _finite(recent_mean)
                and _finite(first_mean)
                and recent_mean >= first_mean - 2.0
            ),
            "last_50_reward_mean": recent_mean,
            "first_50_reward_mean": first_mean,
        }
        required = (
            "ppo_training_performed",
            "exact_update_count_met",
            "policy_parameters_changed",
            "reference_parameters_unchanged",
            "checkpoint_saved",
            "checkpoint_reload_pass",
            "periodic_checkpoint_reload_pass",
            "checkpoint_artifact_bound",
            "periodic_checkpoint_artifact_bound",
            "final_matches_last_periodic_checkpoint",
            "candidate_pool_saved",
            "reward_manifest_saved",
            "all_reward_values_finite",
            "all_update_metrics_finite",
            "at_least_one_valid_candidate",
            "at_least_one_gnn_scored_deletion",
            "reward_provenance_complete",
            "gnn_oracle_backend",
            "bace_classifier_contract_pass",
            "rf_guard_pass",
            "calibration_not_loaded",
            "calibration_dataset_not_loaded",
            "frozen_temperature_calibration_present",
            "test_not_loaded",
            "candidate_oracle_contract_pass",
            "kl_within_hard_limit",
            "expected_checkpoints_saved",
            "candidate_syntax_not_collapsed",
        )
        schema = B7_SCHEMA
    else:
        raise ValueError(f"Unsupported BACE PPO gate stage: {stage}")
    failures = [name for name in required if stage_requirements.get(name) is not True]
    if stage_requirements.get("last_50_reward_collapse") is True:
        failures.append("last_50_reward_collapse")
    return {
        "schema_version": schema,
        "stage": stage,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        **stage_requirements,
    }


__all__ = [
    "B6_V2_SCHEMA",
    "B7_SCHEMA",
    "BacePPOObserver",
    "REWARD_PROVENANCE_FIELDS",
    "atomic_json",
    "build_ppo_gate",
    "build_reward_manifest",
    "model_parameter_hash",
    "validate_adapter_checkpoint_reload",
]
