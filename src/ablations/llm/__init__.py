"""BACE LLM-proposer ablation contracts (framework build only)."""

from .brics import (
    BRICSFixedGenerator,
    BRICSFragmentRecord,
    BRICSVocabulary,
    TrainingMolecule,
    build_train_only_brics_vocabulary,
    training_molecules_from_mappings,
)
from .budget import (
    AttemptRegime,
    MatchedAttemptBudget,
    ParentInput,
    validate_attempt_matched_schedules,
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
from .generators import (
    ChemLLMGeneratorAdapter,
    ProposalGenerator,
    RuntimeGeneratorIdentity,
)
from .schema import (
    build_proposal_record,
    proposal_output_template,
    run_manifest_template,
    summarize_novelty,
    validate_proposal_record,
)

__all__ = [
    "ArtifactPin",
    "AttemptRegime",
    "AVAILABLE",
    "BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT",
    "BRICSFixedGenerator",
    "BRICSFragmentRecord",
    "BRICSVocabulary",
    "ChemLLMGeneratorAdapter",
    "CommonDownstreamPlan",
    "GeneratorAssets",
    "LLMAblationContractError",
    "LLMProposerVariant",
    "MatchedAttemptBudget",
    "ParentInput",
    "ProposalGenerator",
    "ProposalRequest",
    "ProposalResult",
    "RuntimeGeneratorIdentity",
    "TrainingMolecule",
    "build_common_downstream_plan",
    "build_proposal_record",
    "build_train_only_brics_vocabulary",
    "artifact_sha256",
    "proposal_output_template",
    "run_manifest_template",
    "summarize_novelty",
    "training_molecules_from_mappings",
    "validate_attempt_matched_schedules",
    "validate_proposal_record",
]
