"""Model-facing interfaces for fragment generation backends."""

from src.models.interfaces import FragmentGenerator, GenerationRequest, GenerationResult
from src.models.llm_generator import ChemLLMGenerator, clean_generated_smiles
from src.models.local_loader import LocalArtifacts, load_local_hf_artifacts, resolve_local_artifact_paths
from src.models.prompt_builder import (
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
    FewShotExample,
    build_chemllm_messages,
    build_chemllm_prompt,
    build_counterfactual_system_prompt,
)

__all__ = [
    "ChemLLMGenerator",
    "FEW_SHOT_EXAMPLES",
    "FragmentGenerator",
    "FewShotExample",
    "GenerationRequest",
    "GenerationResult",
    "LocalArtifacts",
    "SYSTEM_PROMPT",
    "build_chemllm_messages",
    "build_chemllm_prompt",
    "build_counterfactual_system_prompt",
    "clean_generated_smiles",
    "load_local_hf_artifacts",
    "resolve_local_artifact_paths",
]
