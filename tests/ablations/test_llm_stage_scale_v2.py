from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from src.ablations.llm.comparability import (
    PROPOSAL_ONLY_SCALE_COMPARABLE,
    ModelComparabilityInput,
    compare_model_scale_inputs,
)
from src.ablations.llm.contracts import LLMAblationContractError
from src.ablations.llm.model_scale_registry import (
    FileIdentity,
    ModelSnapshotManifest,
    NOT_MEASURED_METADATA_ONLY,
    READY,
    VERIFIED_SAFETENSORS_HEADER_EXACT,
    load_model_scale_registry,
    require_exact_revision,
)
from src.ablations.llm.stage_scale import (
    LLMScaleFallbackVariant,
    LLMScaleVariant,
    LLMStageVariant,
    MatchedAdaptationPlan,
    ScaleFallbackAssetTopology,
    StageAssetTopology,
    assert_matched_adaptation,
    validate_non_factorial_design,
)
from src.ablations.output_schema import llm_v2_output_inventory


SHA = "a" * 64
REPO_ROOT = Path(__file__).resolve().parents[2]


def _sft() -> dict[str, object]:
    return {
        "dataset_sha256": SHA,
        "ordered_examples_sha256": "b" * 64,
        "optimizer_family": "adamw",
        "optimizer_updates": 500,
        "max_sequence_length": 512,
        "validation_policy_sha256": "c" * 64,
        "checkpoint_selection_policy_sha256": "d" * 64,
        "seed": 7,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_roles": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }


def _ppo() -> dict[str, object]:
    return {
        "parent_manifest_sha256": "e" * 64,
        "rollout_budget": 2400,
        "reward_config_sha256": "f" * 64,
        "reward_weights_sha256": "1" * 64,
        "kl_config_sha256": "2" * 64,
        "clip_config_sha256": "3" * 64,
        "optimizer_updates": 300,
        "sampling_policy_sha256": "4" * 64,
        "validation_policy_sha256": "5" * 64,
        "seed": 7,
        "global_effective_batch": 64,
    }


def test_stage_assets_enforce_off_the_shelf_sft_and_ppo_topology() -> None:
    StageAssetTopology(LLMStageVariant.BRICS_FIXED, None, None, None, None)
    StageAssetTopology(
        LLMStageVariant.CHEMLLM_7B_OFF_THE_SHELF, SHA, "b" * 64, None, None
    )
    StageAssetTopology(
        LLMStageVariant.CHEMLLM_7B_PROJECT_SFT, SHA, "b" * 64, "c" * 64, None
    )
    StageAssetTopology(
        LLMStageVariant.CHEMLLM_7B_PROJECT_SFT_PPO,
        SHA,
        "b" * 64,
        "c" * 64,
        "d" * 64,
    )
    with pytest.raises(LLMAblationContractError):
        StageAssetTopology(
            LLMStageVariant.CHEMLLM_7B_OFF_THE_SHELF,
            SHA,
            "b" * 64,
            "c" * 64,
            None,
        )


def test_matched_2b_7b_plan_only_changes_model_key() -> None:
    small = MatchedAdaptationPlan("chemllm_2b_1_5", _sft(), _ppo())
    reference = MatchedAdaptationPlan("chemllm_7b_main", _sft(), _ppo())
    assert_matched_adaptation(small, reference)
    changed = dict(_ppo())
    changed["optimizer_updates"] = 301
    with pytest.raises(LLMAblationContractError, match="ppo.optimizer_updates"):
        assert_matched_adaptation(
            MatchedAdaptationPlan("chemllm_2b_1_5", _sft(), changed), reference
        )


def test_no_scale_stage_full_factorial_and_outputs_are_separate() -> None:
    validate_non_factorial_design(
        stage_variants=[item.value for item in LLMStageVariant],
        scale_variants=[item.value for item in LLMScaleVariant],
        scale_stage_full_factorial=False,
    )
    with pytest.raises(LLMAblationContractError):
        validate_non_factorial_design(
            stage_variants=[item.value for item in LLMStageVariant],
            scale_variants=[item.value for item in LLMScaleVariant],
            scale_stage_full_factorial=True,
        )
    assert set(llm_v2_output_inventory("stage")).isdisjoint(
        llm_v2_output_inventory("scale")
    )


def test_registry_requires_exact_revision_and_keeps_20b_disabled() -> None:
    payload = yaml.safe_load(
        (REPO_ROOT / "configs/ablations/llm/chemllm_model_scale_registry_v2.yaml")
        .read_text(encoding="utf-8")
    )
    registry = load_model_scale_registry(payload)
    small = registry["chemllm_2b_1_5"]
    assert small.status == READY
    assert small.exact_revision == "215c0dbc89417a06bbc3bae43a3ad61e58f0a56e"
    assert small.total_parameters is None
    assert small.header_total_parameters == 1_889_110_016
    assert small.parameter_count_state == VERIFIED_SAFETENSORS_HEADER_EXACT
    assert small.actual_loaded_dtype is None
    assert small.header_dtype == "bfloat16"
    assert small.runtime_load_state == "BLOCKED_REMOTE_CODE_ISOLATED_IMPORT"
    main = registry["chemllm_7b_main"]
    assert main.exact_revision == "b8b2ea19e48f53d190fe8dced94572717f8e89a2"
    assert main.header_base_parameters == 7_737_708_544
    assert main.header_lora_parameters == 18_874_368
    assert main.header_total_parameters == 7_756_582_912
    assert main.executed_stage == "FRESH_LORA_PPO"
    assert main.actual_loaded_dtype is None
    assert main.header_dtype == "bfloat16"
    assert main.header_adapter_dtype == "float32"
    upper = registry["chemllm_20b_sft"]
    assert upper.exact_revision == "e8d0f503e00f143f6787263765ff6ee5f3fe3998"
    assert upper.metadata_only is True
    assert upper.run_enabled is False
    assert upper.download_weights is False
    assert upper.total_parameters is None
    assert upper.actual_loaded_dtype is None
    assert upper.metadata_index_parameter_estimate == 19_861_149_696
    assert upper.parameter_count_state == NOT_MEASURED_METADATA_ONLY
    with pytest.raises(LLMAblationContractError):
        require_exact_revision("main")


def test_registry_rejects_a_different_well_formed_revision() -> None:
    payload = yaml.safe_load(
        (REPO_ROOT / "configs/ablations/llm/chemllm_model_scale_registry_v2.yaml")
        .read_text(encoding="utf-8")
    )
    payload["models"]["chemllm_2b_1_5"]["exact_revision"] = "f" * 40
    with pytest.raises(LLMAblationContractError, match="drifted from audited pin"):
        load_model_scale_registry(payload)


def test_remote_code_requires_pinned_inventory_and_isolated_import() -> None:
    identity = FileIdentity("config.json", SHA, 1)
    tokenizer = (FileIdentity("tokenizer.json", "b" * 64, 2),)
    with pytest.raises(LLMAblationContractError, match="trust_remote_code"):
        ModelSnapshotManifest(
            repository_id="AI4Chem/CHEMLLM-2b-1_5",
            exact_revision="c" * 40,
            commit_date="2026-01-01T00:00:00Z",
            model_card=FileIdentity("README.md", "d" * 64, 3),
            config=identity,
            tokenizer_files=tokenizer,
            remote_code_files=(),
            weight_files=(FileIdentity("model.safetensors", "e" * 64, 4),),
            license_metadata={"license": "FROM_MODEL_CARD"},
            weights_downloaded=True,
            metadata_only=False,
            trust_remote_code=True,
            isolated_import_pass=False,
        )


def test_sft_ppo_plan_rejects_calibration_or_test() -> None:
    with pytest.raises(LLMAblationContractError, match="calibration or test"):
        MatchedAdaptationPlan(
            "chemllm_2b_1_5", _sft(), _ppo(), calibration_loaded=True
        )


def _comparability_input(key: str, *, sft: bool, ppo: bool, stage: str):
    return ModelComparabilityInput(
        registry_key=key,
        architecture_family="internlm2",
        tokenizer_family="internlm2",
        tokenizer_sha256=SHA,
        special_tokens_sha256="6" * 64,
        chat_template_sha256="b" * 64,
        prompt_rendering_sha256="c" * 64,
        molecular_token_handling="character_plus_bpe",
        output_decoding_sha256="d" * 64,
        max_context=4096,
        dtype="bfloat16",
        lora_target_roles=("q_proj", "k_proj", "v_proj", "o_proj"),
        trl_ppo_compatible=True,
        matched_project_sft_available=sft,
        matched_project_ppo_available=ppo,
        executed_stage=stage,
    )


def test_real_bace_fresh_lora_ppo_lineage_forces_proposal_only_fallback() -> None:
    small = _comparability_input(
        "chemllm_2b_1_5", sft=False, ppo=False, stage="OFF_THE_SHELF"
    )
    reference = _comparability_input(
        "chemllm_7b_main", sft=False, ppo=True, stage="FRESH_LORA_PPO"
    )
    report = compare_model_scale_inputs(small, reference)
    assert report.classification == PROPOSAL_ONLY_SCALE_COMPARABLE
    assert report.full_method_scale_claim_allowed is False
    assert report.effective_comparison == "2B_OFF_THE_SHELF_vs_7B_OFF_THE_SHELF"
    assert any("REQUESTED_LABEL_MISMATCH" in item for item in report.blockers)


def test_full_scale_compatibility_requires_exact_adaptation_plan_comparison() -> None:
    small = _comparability_input(
        "chemllm_2b_1_5", sft=True, ppo=True, stage="PROJECT_SFT_PPO"
    )
    reference = _comparability_input(
        "chemllm_7b_main", sft=True, ppo=True, stage="PROJECT_SFT_PPO"
    )
    without_plans = compare_model_scale_inputs(small, reference)
    assert without_plans.full_method_scale_claim_allowed is False
    assert "MATCHED_ADAPTATION_PLANS_REQUIRED" in without_plans.blockers
    with_plans = compare_model_scale_inputs(
        small,
        reference,
        small_adaptation=MatchedAdaptationPlan("chemllm_2b_1_5", _sft(), _ppo()),
        reference_adaptation=MatchedAdaptationPlan("chemllm_7b_main", _sft(), _ppo()),
    )
    assert with_plans.full_method_scale_claim_allowed is True


def test_proposal_only_scale_fallback_has_constructible_no_adapter_rows() -> None:
    for variant, key in (
        (LLMScaleFallbackVariant.CHEMLLM_2B_OFF_THE_SHELF, "chemllm_2b_1_5"),
        (LLMScaleFallbackVariant.CHEMLLM_7B_OFF_THE_SHELF, "chemllm_7b_main"),
    ):
        topology = ScaleFallbackAssetTopology(variant, key, SHA, "b" * 64)
        assert topology.project_sft_sha256 is None
        assert topology.project_ppo_sha256 is None
    with pytest.raises(LLMAblationContractError, match="may not load project"):
        ScaleFallbackAssetTopology(
            LLMScaleFallbackVariant.CHEMLLM_2B_OFF_THE_SHELF,
            "chemllm_2b_1_5",
            SHA,
            "b" * 64,
            project_sft_sha256="c" * 64,
        )


def test_v2_configs_fail_closed_on_absent_project_sft() -> None:
    stage = yaml.safe_load(
        (REPO_ROOT / "configs/ablations/llm/bace_ours_stage_ablation_v2.yaml")
        .read_text(encoding="utf-8")
    )
    variants = {row["id"]: row for row in stage["design"]["variants"]}
    assert variants["A1"]["display_name"] == "OFF_THE_SHELF_CHEMLLM"
    assert variants["A1"]["exact_base_revision"] == (
        "b8b2ea19e48f53d190fe8dced94572717f8e89a2"
    )
    assert variants["A2"]["availability"] == "BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT"
    assert variants["A3"]["observed_main_stage"] == "FRESH_LORA_PPO"
    assert variants["A3"]["reuse_main_result"] is False
    scale = yaml.safe_load(
        (REPO_ROOT / "configs/ablations/llm/bace_ours_scale_ablation_v2.yaml")
        .read_text(encoding="utf-8")
    )
    assert scale["scale_stage_full_factorial"] is False
    assert scale["resolved_model_pins"]["chemllm_2b_1_5"][
        "safetensors_header_total_parameters"
    ] == 1_889_110_016
    assert scale["resolved_model_pins"]["chemllm_20b_sft"][
        "actual_total_parameters"
    ] is None
    assert scale["primary_comparison"]["state"] == "BLOCKED_REFERENCE_MISSING_MATCHED_SFT"
    assert scale["fallback_comparison"]["claim"] == "MODEL_SCALE_PROPOSAL_SENSITIVITY"
