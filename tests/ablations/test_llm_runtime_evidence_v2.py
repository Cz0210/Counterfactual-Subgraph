from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.ablations.llm.contracts import LLMAblationContractError, canonical_json_sha256
from src.ablations.llm.model_scale_registry import load_model_scale_registry
from src.ablations.llm.runtime_evidence import (
    evaluate_runtime_model_evidence,
    load_bace_reference_v2,
    sha256_file,
    validate_stage_config_against_reference,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, payload: dict) -> tuple[str, str]:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return str(path), sha256_file(path)


def _reference(tmp_path: Path):
    payload = {
        "schema_version": "bace_ours_llm_reference_v2",
        "status": "PASS",
        "dataset": "bace",
        "method": "ours",
        "source_label": 1,
        "scientific_values_inferred": False,
        "main_policy_scientific_name": "CHEMLLM_7B_OFF_THE_SHELF_PLUS_FRESH_LORA_PPO",
        "main_policy_must_not_be_named": "CHEMLLM_7B_PROJECT_SFT_PPO",
        "base_model": {
            "repository_id": "AI4Chem/ChemLLM-7B-Chat",
            "revision": "b8b2ea19e48f53d190fe8dced94572717f8e89a2",
            "path": "/models/chemllm-7b",
            "directory_contract_sha256": "a" * 64,
            "parameters": {
                "total_parameters": 7_737_708_544,
                "dtype_parameter_counts": {"BF16": 7_737_708_544},
            },
        },
        "ppo": {
            "parameters": {
                "total_parameters": 18_874_368,
                "dtype_parameter_counts": {"F32": 18_874_368},
            },
            "trainable_parameters": 18_874_368,
            "loaded_total_parameters_base_plus_adapter": 7_756_582_912,
            "optimizer_updates": 300,
        },
        "stage_variants": {
            "A0_BRICS_FIXED": {"model": None, "status": "CPU_FRAMEWORK_AVAILABLE"},
            "A1_CHEMLLM_7B_OFF_THE_SHELF": {
                "repository_id": "AI4Chem/ChemLLM-7B-Chat",
                "revision": "b8b2ea19e48f53d190fe8dced94572717f8e89a2",
                "project_adapter_loaded": False,
                "status": "AVAILABLE",
            },
            "A2_CHEMLLM_7B_PROJECT_SFT": {
                "status": "BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT"
            },
            "A3_CHEMLLM_7B_PROJECT_SFT_PPO": {
                "actual_main_policy_available_as": "CHEMLLM_7B_FRESH_LORA_PPO",
                "status": "BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT",
            },
        },
    }
    payload["reference_contract_sha256"] = canonical_json_sha256(payload)
    path, sha = _write(tmp_path / "reference.json", payload)
    return load_bace_reference_v2(path, sha)


def test_runtime_reference_is_physical_self_hashed_and_crosses_stage_config(
    tmp_path: Path,
) -> None:
    reference = _reference(tmp_path)
    stage = yaml.safe_load(
        (REPO_ROOT / "configs/ablations/llm/bace_ours_stage_ablation_v2.yaml")
        .read_text()
    )
    validate_stage_config_against_reference(stage, reference)
    stage["design"]["variants"][1]["reference_status"] = "BLOCKED"
    try:
        validate_stage_config_against_reference(stage, reference)
    except Exception as exc:
        assert "disagrees" in str(exc)
    else:  # pragma: no cover - fail-closed assertion
        raise AssertionError("stage/reference drift was accepted")


def test_header_snapshot_does_not_become_actual_loaded_model_evidence(
    tmp_path: Path,
) -> None:
    registry = load_model_scale_registry(
        yaml.safe_load(
            (REPO_ROOT / "configs/ablations/llm/chemllm_model_scale_registry_v2.yaml")
            .read_text()
        )
    )
    snapshot = {
        "schema_version": "chemllm_snapshot_manifest_v1",
        "status": "PASS",
        "repository_id": "AI4Chem/CHEMLLM-2b-1_5",
        "revision": "215c0dbc89417a06bbc3bae43a3ad61e58f0a56e",
        "weights_downloaded": True,
        "parameters": {
            "count_source": "downloaded_safetensors_tensor_headers",
            "total_parameters": 1_889_110_016,
        },
    }
    snapshot_pin = _write(tmp_path / "snapshot.json", snapshot)
    decision = evaluate_runtime_model_evidence(
        registry,
        two_b_snapshot=snapshot_pin,
        two_b_parameter_report=None,
        seven_b_parameter_report=None,
        twenty_b_metadata=None,
    )
    assert decision["states"]["chemllm_2b_1_5"] == (
        "SNAPSHOT_READY_SCIENCE_LOAD_BLOCKED"
    )
    assert decision["runtime_science_ready"] is False


def test_actual_parameter_report_self_hash_is_required(tmp_path: Path) -> None:
    registry = load_model_scale_registry(
        yaml.safe_load(
            (REPO_ROOT / "configs/ablations/llm/chemllm_model_scale_registry_v2.yaml")
            .read_text()
        )
    )
    report = {
        "schema_version": "actual_parameter_count_report_v1",
        "source": "ACTUAL_LOADED_WEIGHTS",
        "total_parameters": 7_756_582_912,
        "trainable_parameters": 18_874_368,
        "embedding_parameters": 758_120_448,
        "non_embedding_parameters": 6_998_462_464,
        "lora_trainable_parameters": 18_874_368,
        "trainable_fraction": 18_874_368 / 7_756_582_912,
        "dtype": ["torch.bfloat16", "torch.float32"],
        "weight_bytes": 1,
        "config_hidden_size": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 92544,
    }
    report["parameter_report_sha256"] = canonical_json_sha256(report)
    report_pin = _write(tmp_path / "report.json", report)
    decision = evaluate_runtime_model_evidence(
        registry,
        two_b_snapshot=None,
        two_b_parameter_report=None,
        seven_b_parameter_report=report_pin,
        twenty_b_metadata=None,
    )
    assert decision["states"]["chemllm_7b_main"] == "RUNTIME_EVIDENCE_PASS"
    assert decision["runtime_science_ready"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("embedding_parameters", 1),
        ("trainable_fraction", 0.5),
        ("weight_bytes", 0),
        ("config_hidden_size", None),
        ("num_layers", 0),
        ("num_attention_heads", None),
        ("vocab_size", 0),
    ),
)
def test_actual_parameter_report_requires_full_numeric_closure(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    registry = load_model_scale_registry(
        yaml.safe_load(
            (REPO_ROOT / "configs/ablations/llm/chemllm_model_scale_registry_v2.yaml")
            .read_text()
        )
    )
    report = {
        "schema_version": "actual_parameter_count_report_v1",
        "source": "ACTUAL_LOADED_WEIGHTS",
        "total_parameters": 7_756_582_912,
        "trainable_parameters": 18_874_368,
        "embedding_parameters": 758_120_448,
        "non_embedding_parameters": 6_998_462_464,
        "lora_trainable_parameters": 18_874_368,
        "trainable_fraction": 18_874_368 / 7_756_582_912,
        "dtype": ["torch.bfloat16", "torch.float32"],
        "weight_bytes": 15_550_000_000,
        "config_hidden_size": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 92544,
    }
    report[field] = value
    report["parameter_report_sha256"] = canonical_json_sha256(report)
    report_pin = _write(tmp_path / f"report-{field}.json", report)
    with pytest.raises(LLMAblationContractError):
        evaluate_runtime_model_evidence(
            registry,
            two_b_snapshot=None,
            two_b_parameter_report=None,
            seven_b_parameter_report=report_pin,
            twenty_b_metadata=None,
        )
