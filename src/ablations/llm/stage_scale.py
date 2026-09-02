"""Orthogonal ChemLLM training-stage and model-scale ablation contracts.

The objects in this module are deliberately model-runtime agnostic.  They bind
the proposal budget, decoding contract, project adaptation lineage, and output
schema without loading weights or opening calibration/test data.  Runtime
entrypoints may implement :class:`ProposalGenerator` only after all referenced
artifacts have been independently hash-verified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

from .contracts import LLMAblationContractError, canonical_json_sha256, require_sha256


class LLMStageVariant(str, Enum):
    """The four fixed stage-ablation rows."""

    BRICS_FIXED = "BRICS_FIXED"
    CHEMLLM_7B_OFF_THE_SHELF = "CHEMLLM_7B_OFF_THE_SHELF"
    CHEMLLM_7B_PROJECT_SFT = "CHEMLLM_7B_PROJECT_SFT"
    CHEMLLM_7B_PROJECT_SFT_PPO = "CHEMLLM_7B_PROJECT_SFT_PPO"


class LLMScaleVariant(str, Enum):
    """The non-factorial model-scale rows."""

    CHEMLLM_2B_PROJECT_SFT_PPO = "CHEMLLM_2B_PROJECT_SFT_PPO"
    CHEMLLM_7B_PROJECT_SFT_PPO = "CHEMLLM_7B_PROJECT_SFT_PPO"


@dataclass(frozen=True, slots=True)
class ProposalParent:
    parent_id: str
    smiles: str
    source_label: int

    def __post_init__(self) -> None:
        if not self.parent_id.strip() or not self.smiles.strip():
            raise LLMAblationContractError("proposal parent ID/SMILES must be non-empty")
        if isinstance(self.source_label, bool) or not isinstance(self.source_label, int):
            raise LLMAblationContractError("source_label must be an integer")


@dataclass(frozen=True, slots=True)
class ProposalBudget:
    parent_ids_sha256: str
    attempts_per_parent: int
    primary: str = "ATTEMPT_MATCHED"
    secondary: str = "VALID_CANDIDATE_MATCHED_DIAGNOSTIC_ONLY"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parent_ids_sha256",
            require_sha256(self.parent_ids_sha256, field="parent_ids_sha256"),
        )
        if self.primary != "ATTEMPT_MATCHED":
            raise LLMAblationContractError("primary proposal budget must be ATTEMPT_MATCHED")
        if self.secondary != "VALID_CANDIDATE_MATCHED_DIAGNOSTIC_ONLY":
            raise LLMAblationContractError("valid-candidate matching is diagnostic only")
        if isinstance(self.attempts_per_parent, bool) or self.attempts_per_parent <= 0:
            raise LLMAblationContractError("attempts_per_parent must be positive")


@dataclass(frozen=True, slots=True)
class DecodingRegime:
    name: str
    attempts_per_parent: int
    temperature: float
    top_p: float
    max_new_tokens: int

    def __post_init__(self) -> None:
        if not self.name.strip() or self.attempts_per_parent <= 0:
            raise LLMAblationContractError("decoding regime name/count is invalid")
        if self.temperature < 0 or not 0 < self.top_p <= 1 or self.max_new_tokens <= 0:
            raise LLMAblationContractError("decoding regime parameters are invalid")


@dataclass(frozen=True, slots=True)
class DecodingConfig:
    prompt_semantics_sha256: str
    regimes: tuple[DecodingRegime, ...]
    parser_config_sha256: str
    projection_config_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "prompt_semantics_sha256",
            "parser_config_sha256",
            "projection_config_sha256",
        ):
            object.__setattr__(self, field, require_sha256(getattr(self, field), field=field))
        object.__setattr__(self, "regimes", tuple(self.regimes))
        if not self.regimes or len({item.name for item in self.regimes}) != len(self.regimes):
            raise LLMAblationContractError("decoding regimes must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class SeedManifest:
    seed_manifest_sha256: str
    ordered_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "seed_manifest_sha256",
            require_sha256(self.seed_manifest_sha256, field="seed_manifest_sha256"),
        )
        object.__setattr__(self, "ordered_seeds", tuple(self.ordered_seeds))
        if not self.ordered_seeds or any(
            isinstance(seed, bool) or seed < 0 for seed in self.ordered_seeds
        ):
            raise LLMAblationContractError("seed manifest must contain non-negative seeds")


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    parent_id: str
    attempt_index: int
    raw_output: str
    fragment_smiles: str | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.parent_id.strip() or self.attempt_index < 0:
            raise LLMAblationContractError("candidate record identity is invalid")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class CandidatePool:
    variant: str
    records: tuple[CandidateRecord, ...]
    proposal_attempts: int
    proposal_shortfall: int
    contract_sha256: str

    def __post_init__(self) -> None:
        if not self.variant.strip():
            raise LLMAblationContractError("candidate pool variant is required")
        object.__setattr__(self, "records", tuple(self.records))
        if self.proposal_attempts < 0 or self.proposal_shortfall < 0:
            raise LLMAblationContractError("candidate pool counts must be non-negative")
        if len(self.records) + self.proposal_shortfall != self.proposal_attempts:
            raise LLMAblationContractError("candidate pool attempts do not close")
        object.__setattr__(
            self,
            "contract_sha256",
            require_sha256(self.contract_sha256, field="contract_sha256"),
        )


class ProposalGenerator(Protocol):
    """Batch-level interface shared by BRICS and all ChemLLM policy states."""

    variant: str

    def generate(
        self,
        parents: Sequence[ProposalParent],
        proposal_budget: ProposalBudget,
        decoding_config: DecodingConfig,
        seed_manifest: SeedManifest,
    ) -> CandidatePool:
        """Generate one attempt-matched candidate pool without selecting by oracle."""


@dataclass(frozen=True, slots=True)
class StageAssetTopology:
    """Exact adapter topology for one stage row.

    Paths are intentionally represented by their artifact digests here.  The
    existing :class:`~src.ablations.llm.contracts.ArtifactPin` performs the
    physical-path reopen at runtime.
    """

    variant: LLMStageVariant
    base_model_sha256: str | None
    tokenizer_sha256: str | None
    project_sft_sha256: str | None
    project_ppo_sha256: str | None

    def __post_init__(self) -> None:
        variant = self.variant
        if not isinstance(variant, LLMStageVariant):
            variant = LLMStageVariant(str(variant))
            object.__setattr__(self, "variant", variant)
        values = (
            self.base_model_sha256,
            self.tokenizer_sha256,
            self.project_sft_sha256,
            self.project_ppo_sha256,
        )
        if variant is LLMStageVariant.BRICS_FIXED:
            expected = (False, False, False, False)
        elif variant is LLMStageVariant.CHEMLLM_7B_OFF_THE_SHELF:
            expected = (True, True, False, False)
        elif variant is LLMStageVariant.CHEMLLM_7B_PROJECT_SFT:
            expected = (True, True, True, False)
        else:
            expected = (True, True, True, True)
        if tuple(value is not None for value in values) != expected:
            raise LLMAblationContractError(
                f"{variant.value} adapter topology must be {expected}"
            )
        for index, value in enumerate(values):
            if value is not None:
                require_sha256(value, field=f"stage_asset[{index}]")


MATCHED_SFT_FIELDS = (
    "dataset_sha256",
    "ordered_examples_sha256",
    "optimizer_family",
    "optimizer_updates",
    "max_sequence_length",
    "validation_policy_sha256",
    "checkpoint_selection_policy_sha256",
    "seed",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_target_roles",
)

MATCHED_PPO_FIELDS = (
    "parent_manifest_sha256",
    "rollout_budget",
    "reward_config_sha256",
    "reward_weights_sha256",
    "kl_config_sha256",
    "clip_config_sha256",
    "optimizer_updates",
    "sampling_policy_sha256",
    "validation_policy_sha256",
    "seed",
    "global_effective_batch",
)


@dataclass(frozen=True, slots=True)
class MatchedAdaptationPlan:
    """Config-only SFT+PPO plan used to compare 2B and 7B fairly."""

    model_registry_key: str
    sft: Mapping[str, Any]
    ppo: Mapping[str, Any]
    calibration_loaded: bool = False
    test_loaded: bool = False

    def __post_init__(self) -> None:
        if not self.model_registry_key.strip():
            raise LLMAblationContractError("model_registry_key is required")
        object.__setattr__(self, "sft", dict(self.sft))
        object.__setattr__(self, "ppo", dict(self.ppo))
        missing_sft = sorted(set(MATCHED_SFT_FIELDS) - set(self.sft))
        missing_ppo = sorted(set(MATCHED_PPO_FIELDS) - set(self.ppo))
        if missing_sft or missing_ppo:
            raise LLMAblationContractError(
                f"matched adaptation fields missing: sft={missing_sft}, ppo={missing_ppo}"
            )
        if self.calibration_loaded or self.test_loaded:
            raise LLMAblationContractError("SFT/PPO plans may not load calibration or test")
        for mapping, fields in ((self.sft, MATCHED_SFT_FIELDS), (self.ppo, MATCHED_PPO_FIELDS)):
            for field in fields:
                if field.endswith("sha256"):
                    require_sha256(mapping[field], field=field)
        if int(self.sft["optimizer_updates"]) <= 0 or int(self.ppo["optimizer_updates"]) <= 0:
            raise LLMAblationContractError("SFT/PPO optimizer updates must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["plan_sha256"] = canonical_json_sha256(payload)
        return payload


def assert_matched_adaptation(
    small: MatchedAdaptationPlan,
    reference: MatchedAdaptationPlan,
) -> None:
    """Require all adaptation variables to match except the base-model key."""

    if small.model_registry_key == reference.model_registry_key:
        raise LLMAblationContractError("scale rows must use distinct base-model entries")
    changed: list[str] = []
    for stage, fields in (("sft", MATCHED_SFT_FIELDS), ("ppo", MATCHED_PPO_FIELDS)):
        left = getattr(small, stage)
        right = getattr(reference, stage)
        changed.extend(f"{stage}.{field}" for field in fields if left[field] != right[field])
    if changed:
        raise LLMAblationContractError(
            "2B/7B matched adaptation changed non-scale fields: " + ", ".join(changed)
        )


def validate_non_factorial_design(
    *,
    stage_variants: Sequence[str],
    scale_variants: Sequence[str],
    scale_stage_full_factorial: bool,
) -> None:
    if scale_stage_full_factorial:
        raise LLMAblationContractError("scale x stage full factorial is forbidden")
    if tuple(stage_variants) != tuple(item.value for item in LLMStageVariant):
        raise LLMAblationContractError("stage design must contain exactly A0/A1/A2/A3")
    if tuple(scale_variants) != tuple(item.value for item in LLMScaleVariant):
        raise LLMAblationContractError("scale design must contain exactly S0/S1")


__all__ = [
    "CandidatePool",
    "CandidateRecord",
    "DecodingConfig",
    "DecodingRegime",
    "LLMScaleVariant",
    "LLMStageVariant",
    "MATCHED_PPO_FIELDS",
    "MATCHED_SFT_FIELDS",
    "MatchedAdaptationPlan",
    "ProposalBudget",
    "ProposalGenerator",
    "ProposalParent",
    "SeedManifest",
    "StageAssetTopology",
    "assert_matched_adaptation",
    "validate_non_factorial_design",
]
