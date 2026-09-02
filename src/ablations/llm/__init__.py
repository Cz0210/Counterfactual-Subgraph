"""BACE LLM proposer, stage, and scale ablation contracts."""

from .brics import (
    BRICSFixedGenerator,
    BRICSFragmentRecord,
    BRICSVocabulary,
    TrainingMolecule,
    build_train_only_brics_vocabulary,
    training_molecules_from_mappings,
)
from .budget import AttemptRegime, MatchedAttemptBudget, ParentInput, validate_attempt_matched_schedules
from .comparability import (
    MATCHED_PROJECT_ADAPTATION_COMPATIBLE,
    NOT_SCALE_COMPARABLE,
    PROPOSAL_ONLY_SCALE_COMPARABLE,
    ModelComparabilityInput,
    ModelComparabilityReport,
    compare_model_scale_inputs,
)
from .contracts import (
    AVAILABLE,
    BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT,
    ArtifactPin,
    GeneratorAssets,
    LLMAblationContractError,
    LLMProposerVariant,
    ProposalRequest,
    ProposalResult,
    artifact_sha256,
)
from .downstream import CommonDownstreamPlan, build_common_downstream_plan
from .early_launch_gate import (
    EarlyLaunchDecision,
    EarlyLaunchSnapshot,
    EarlyRunAuthorizationReceipt,
    evaluate_early_launch_gate,
    main_priority_runtime_action,
)
from .generators import ChemLLMGeneratorAdapter, ProposalGenerator, RuntimeGeneratorIdentity
from .model_scale_registry import (
    FileIdentity,
    ModelScaleEntry,
    ModelSnapshotManifest,
    load_model_scale_registry,
    require_exact_revision,
)
from .parameter_count import ParameterCountReport, count_actual_loaded_parameters
from .schema import (
    build_proposal_record,
    proposal_output_template,
    run_manifest_template,
    summarize_novelty,
    validate_proposal_record,
)
from .stage_scale import (
    CandidatePool,
    CandidateRecord,
    DecodingConfig,
    DecodingRegime,
    LLMScaleVariant,
    LLMStageVariant,
    MatchedAdaptationPlan,
    ProposalBudget,
    ProposalGenerator as StageScaleProposalGenerator,
    ProposalParent,
    SeedManifest,
    StageAssetTopology,
    assert_matched_adaptation,
    validate_non_factorial_design,
)

__all__ = [
    "AVAILABLE", "ArtifactPin", "AttemptRegime",
    "BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT", "BRICSFixedGenerator",
    "BRICSFragmentRecord", "BRICSVocabulary", "CandidatePool", "CandidateRecord",
    "ChemLLMGeneratorAdapter", "CommonDownstreamPlan", "DecodingConfig",
    "DecodingRegime", "EarlyLaunchDecision", "EarlyLaunchSnapshot",
    "EarlyRunAuthorizationReceipt", "FileIdentity", "GeneratorAssets",
    "LLMAblationContractError", "LLMProposerVariant", "LLMScaleVariant",
    "LLMStageVariant", "MATCHED_PROJECT_ADAPTATION_COMPATIBLE",
    "MatchedAdaptationPlan", "MatchedAttemptBudget", "ModelComparabilityInput",
    "ModelComparabilityReport", "ModelScaleEntry", "ModelSnapshotManifest",
    "NOT_SCALE_COMPARABLE", "PROPOSAL_ONLY_SCALE_COMPARABLE",
    "ParameterCountReport", "ParentInput", "ProposalBudget", "ProposalGenerator",
    "ProposalParent", "ProposalRequest", "ProposalResult", "RuntimeGeneratorIdentity",
    "SeedManifest", "StageAssetTopology", "StageScaleProposalGenerator",
    "TrainingMolecule", "artifact_sha256", "assert_matched_adaptation",
    "build_common_downstream_plan", "build_proposal_record",
    "build_train_only_brics_vocabulary", "compare_model_scale_inputs",
    "count_actual_loaded_parameters", "evaluate_early_launch_gate",
    "load_model_scale_registry", "main_priority_runtime_action", "proposal_output_template", "require_exact_revision",
    "run_manifest_template", "summarize_novelty", "training_molecules_from_mappings",
    "validate_attempt_matched_schedules", "validate_non_factorial_design",
    "validate_proposal_record",
]
