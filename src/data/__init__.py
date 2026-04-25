"""Data schemas, prompt construction, and dataset adapters."""

from src.data.collators import CounterfactualPromptCollator, PromptBatch
from src.data.dataset import JsonlMoleculeDataset
from src.data.prompts import (
    build_counterfactual_prompt,
    build_exact_parent_substructure_prompt,
)
from src.data.schemas import FragmentExample, MoleculeRecord, normalize_molecule_record

__all__ = [
    "CounterfactualPromptCollator",
    "FragmentExample",
    "JsonlMoleculeDataset",
    "MoleculeRecord",
    "PromptBatch",
    "build_counterfactual_prompt",
    "build_exact_parent_substructure_prompt",
    "normalize_molecule_record",
]

try:  # pragma: no cover - optional compatibility layer
    from src.data.aids import AIDSHIVCsvDataset, AIDSHIVRecord, sample_random_aids_hiv_record
except ImportError:  # pragma: no cover - current repo may not include legacy module
    AIDSHIVCsvDataset = None
    AIDSHIVRecord = None
    sample_random_aids_hiv_record = None
else:
    __all__.extend(
        [
            "AIDSHIVCsvDataset",
            "AIDSHIVRecord",
            "sample_random_aids_hiv_record",
        ]
    )

try:  # pragma: no cover - optional compatibility layer
    from src.data.sft_preparation import (
        BalancedSamplingResult,
        PreparationSummary,
        PreparedSFTExample,
        build_balanced_candidate_pool,
        build_sft_instruction,
        filter_valid_hiv_records,
        generate_capped_fragment,
        label_ratio,
        load_hiv_dataframe,
        prepare_balanced_sft_examples,
        save_sft_jsonl,
        split_examples,
    )
except ImportError:  # pragma: no cover - current repo may not include legacy module
    BalancedSamplingResult = None
    PreparationSummary = None
    PreparedSFTExample = None
    build_balanced_candidate_pool = None
    build_sft_instruction = None
    filter_valid_hiv_records = None
    generate_capped_fragment = None
    label_ratio = None
    load_hiv_dataframe = None
    prepare_balanced_sft_examples = None
    save_sft_jsonl = None
    split_examples = None
else:
    __all__.extend(
        [
            "BalancedSamplingResult",
            "PreparationSummary",
            "PreparedSFTExample",
            "build_balanced_candidate_pool",
            "build_sft_instruction",
            "filter_valid_hiv_records",
            "generate_capped_fragment",
            "label_ratio",
            "load_hiv_dataframe",
            "prepare_balanced_sft_examples",
            "save_sft_jsonl",
            "split_examples",
        ]
    )
