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
from .core_execution import (
    CORE_VARIANT_ORDER,
    MAIN_ADAPTATION_PATH,
    SFT_AUXILIARY_REASON,
    SFT_AUXILIARY_STATE,
    CoreLLMVariant,
    CoreRunSpec,
    derive_core_reference,
    load_authorized_launch_decision,
    load_core_run_spec,
    run_core_variant,
    status_core_run,
    validate_variant_artifact_bindings,
)
from .downstream import CommonDownstreamPlan, build_common_downstream_plan
from .early_launch_gate import (
    EarlyLaunchDecision,
    EarlyLaunchSnapshot,
    EarlyRunAuthorizationReceipt,
    evaluate_early_launch_gate,
    main_priority_runtime_action,
)
from .final16_owner_evidence import (
    Final16OwnerCoverage,
    assert_snapshot_matches_owner_coverage,
    evaluate_final16_owner_coverage,
)
from .generators import ChemLLMGeneratorAdapter, ProposalGenerator, RuntimeGeneratorIdentity
from .isolated_chemllm_load import (
    CHEMLLM_2B_REPOSITORY_ID,
    CHEMLLM_2B_REVISION,
    CHEMLLM_2B_TOTAL_PARAMETERS,
    ChemLLM2BSnapshotPin,
    audit_remote_code,
    build_isolated_child_command,
    build_isolated_child_environment,
    pin_chemllm_2b_snapshot,
    prepare_fresh_output_root,
    run_isolated_child_probe,
    validate_isolated_load_receipt,
)
from .model_scale_registry import (
    FileIdentity,
    ModelScaleEntry,
    ModelSnapshotManifest,
    load_model_scale_registry,
    require_exact_revision,
)
from .parameter_count import ParameterCountReport, count_actual_loaded_parameters
from .runtime_evidence import (
    BACEReferenceEvidence,
    evaluate_runtime_model_evidence,
    load_bace_reference_v2,
    runtime_run_contract_sha256,
    validate_off_the_shelf_7b_parameter_report,
    validate_stage_config_against_reference,
)
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
    LLMScaleFallbackVariant,
    LLMStageVariant,
    MatchedAdaptationPlan,
    ProposalBudget,
    ProposalGenerator as StageScaleProposalGenerator,
    ProposalParent,
    SeedManifest,
    ScaleFallbackAssetTopology,
    StageAssetTopology,
    assert_matched_adaptation,
    validate_non_factorial_design,
)

__all__ = [
    "AVAILABLE", "ArtifactPin", "AttemptRegime",
    "BACEReferenceEvidence", "BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT", "BRICSFixedGenerator",
    "BRICSFragmentRecord", "BRICSVocabulary", "CandidatePool", "CandidateRecord",
    "CHEMLLM_2B_REPOSITORY_ID", "CHEMLLM_2B_REVISION",
    "CHEMLLM_2B_TOTAL_PARAMETERS", "ChemLLM2BSnapshotPin",
    "ChemLLMGeneratorAdapter", "CommonDownstreamPlan", "DecodingConfig",
    "CORE_VARIANT_ORDER", "CoreLLMVariant", "CoreRunSpec",
    "DecodingRegime", "EarlyLaunchDecision", "EarlyLaunchSnapshot",
    "EarlyRunAuthorizationReceipt", "FileIdentity", "Final16OwnerCoverage", "GeneratorAssets",
    "LLMAblationContractError", "LLMProposerVariant", "LLMScaleVariant",
    "LLMScaleFallbackVariant", "LLMStageVariant", "MATCHED_PROJECT_ADAPTATION_COMPATIBLE",
    "MatchedAdaptationPlan", "MatchedAttemptBudget", "ModelComparabilityInput",
    "ModelComparabilityReport", "ModelScaleEntry", "ModelSnapshotManifest",
    "NOT_SCALE_COMPARABLE", "PROPOSAL_ONLY_SCALE_COMPARABLE",
    "MAIN_ADAPTATION_PATH",
    "ParameterCountReport", "ParentInput", "ProposalBudget", "ProposalGenerator",
    "ProposalParent", "ProposalRequest", "ProposalResult", "RuntimeGeneratorIdentity",
    "ScaleFallbackAssetTopology", "SeedManifest", "StageAssetTopology", "StageScaleProposalGenerator",
    "SFT_AUXILIARY_REASON", "SFT_AUXILIARY_STATE",
    "TrainingMolecule", "artifact_sha256", "assert_matched_adaptation",
    "audit_remote_code", "build_common_downstream_plan", "build_isolated_child_command",
    "build_isolated_child_environment", "build_proposal_record",
    "build_train_only_brics_vocabulary", "compare_model_scale_inputs",
    "count_actual_loaded_parameters", "evaluate_early_launch_gate",
    "evaluate_final16_owner_coverage", "assert_snapshot_matches_owner_coverage",
    "derive_core_reference", "load_authorized_launch_decision", "load_core_run_spec",
    "evaluate_runtime_model_evidence", "load_bace_reference_v2",
    "load_model_scale_registry", "main_priority_runtime_action", "pin_chemllm_2b_snapshot",
    "prepare_fresh_output_root", "proposal_output_template", "require_exact_revision",
    "run_core_variant", "run_isolated_child_probe", "run_manifest_template", "runtime_run_contract_sha256", "status_core_run", "summarize_novelty", "training_molecules_from_mappings",
    "validate_attempt_matched_schedules", "validate_non_factorial_design",
    "validate_isolated_load_receipt", "validate_off_the_shelf_7b_parameter_report", "validate_proposal_record", "validate_stage_config_against_reference", "validate_variant_artifact_bindings",
]
