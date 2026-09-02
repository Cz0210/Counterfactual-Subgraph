"""Classify which model-scale claim is supported by frozen provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .contracts import LLMAblationContractError, canonical_json_sha256, require_sha256


MATCHED_PROJECT_ADAPTATION_COMPATIBLE = "MATCHED_PROJECT_ADAPTATION_COMPATIBLE"
PROPOSAL_ONLY_SCALE_COMPARABLE = "PROPOSAL_ONLY_SCALE_COMPARABLE"
NOT_SCALE_COMPARABLE = "NOT_SCALE_COMPARABLE"


@dataclass(frozen=True, slots=True)
class ModelComparabilityInput:
    registry_key: str
    architecture_family: str
    tokenizer_family: str
    tokenizer_sha256: str
    chat_template_sha256: str
    prompt_rendering_sha256: str
    molecular_token_handling: str
    output_decoding_sha256: str
    max_context: int
    dtype: str
    lora_target_roles: tuple[str, ...]
    trl_ppo_compatible: bool
    matched_project_sft_available: bool
    matched_project_ppo_available: bool
    executed_stage: str

    def __post_init__(self) -> None:
        for field in (
            "registry_key",
            "architecture_family",
            "tokenizer_family",
            "molecular_token_handling",
            "dtype",
            "executed_stage",
        ):
            if not str(getattr(self, field)).strip():
                raise LLMAblationContractError(f"{field} is required")
        for field in (
            "tokenizer_sha256",
            "chat_template_sha256",
            "prompt_rendering_sha256",
            "output_decoding_sha256",
        ):
            object.__setattr__(self, field, require_sha256(getattr(self, field), field=field))
        object.__setattr__(self, "lora_target_roles", tuple(self.lora_target_roles))
        if self.max_context <= 0 or not self.lora_target_roles:
            raise LLMAblationContractError("context and LoRA roles must be declared")


@dataclass(frozen=True, slots=True)
class ModelComparabilityReport:
    classification: str
    full_method_scale_claim_allowed: bool
    proposal_sensitivity_claim_allowed: bool
    matched_fields: tuple[str, ...]
    mismatched_fields: tuple[str, ...]
    blockers: tuple[str, ...]
    requested_primary_comparison: str
    effective_comparison: str | None
    schema_version: str = "chemllm_model_comparability_report_v1"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for field in ("matched_fields", "mismatched_fields", "blockers"):
            payload[field] = list(payload[field])
        payload["comparability_report_sha256"] = canonical_json_sha256(payload)
        return payload


def compare_model_scale_inputs(
    small: ModelComparabilityInput,
    reference: ModelComparabilityInput,
) -> ModelComparabilityReport:
    if small.registry_key == reference.registry_key:
        raise LLMAblationContractError("scale comparison needs distinct model entries")
    proposal_fields = (
        "architecture_family",
        "tokenizer_family",
        "chat_template_sha256",
        "prompt_rendering_sha256",
        "molecular_token_handling",
        "output_decoding_sha256",
    )
    matched = tuple(field for field in proposal_fields if getattr(small, field) == getattr(reference, field))
    mismatched = tuple(
        field for field in proposal_fields if getattr(small, field) != getattr(reference, field)
    )
    blockers: list[str] = []
    matched_adaptation = all(
        (
            small.matched_project_sft_available,
            small.matched_project_ppo_available,
            reference.matched_project_sft_available,
            reference.matched_project_ppo_available,
            small.trl_ppo_compatible,
            reference.trl_ppo_compatible,
        )
    )
    if not reference.matched_project_sft_available:
        blockers.append("REFERENCE_BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT")
    if reference.executed_stage != "PROJECT_SFT_PPO":
        blockers.append(
            "REFERENCE_REQUESTED_LABEL_MISMATCH:executed_stage=" + reference.executed_stage
        )
    if not small.matched_project_sft_available:
        blockers.append("SMALL_MODEL_MATCHED_SFT_NOT_AVAILABLE")
    if not small.matched_project_ppo_available:
        blockers.append("SMALL_MODEL_MATCHED_PPO_NOT_AVAILABLE")

    if not mismatched and matched_adaptation and not blockers:
        classification = MATCHED_PROJECT_ADAPTATION_COMPATIBLE
        full_allowed = True
        proposal_allowed = True
        effective = "2B_PROJECT_SFT_PPO_vs_7B_PROJECT_SFT_PPO"
    elif not mismatched:
        classification = PROPOSAL_ONLY_SCALE_COMPARABLE
        full_allowed = False
        proposal_allowed = True
        effective = "2B_OFF_THE_SHELF_vs_7B_OFF_THE_SHELF"
    else:
        classification = NOT_SCALE_COMPARABLE
        full_allowed = False
        proposal_allowed = False
        effective = None
        blockers.append("PROPOSAL_CONTRACT_MISMATCH:" + ",".join(mismatched))
    return ModelComparabilityReport(
        classification=classification,
        full_method_scale_claim_allowed=full_allowed,
        proposal_sensitivity_claim_allowed=proposal_allowed,
        matched_fields=matched,
        mismatched_fields=mismatched,
        blockers=tuple(blockers),
        requested_primary_comparison="2B_PROJECT_SFT_PPO_vs_7B_PROJECT_SFT_PPO",
        effective_comparison=effective,
    )


__all__ = [
    "MATCHED_PROJECT_ADAPTATION_COMPATIBLE",
    "ModelComparabilityInput",
    "ModelComparabilityReport",
    "NOT_SCALE_COMPARABLE",
    "PROPOSAL_ONLY_SCALE_COMPARABLE",
    "compare_model_scale_inputs",
]
