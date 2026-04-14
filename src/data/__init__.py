"""Data schemas, prompt construction, and dataset adapters."""

from src.data.aids import AIDSHIVCsvDataset, AIDSHIVRecord, sample_random_aids_hiv_record
from src.data.collators import CounterfactualPromptCollator, PromptBatch
from src.data.dataset import JsonlMoleculeDataset
from src.data.prompts import build_counterfactual_prompt
from src.data.schemas import FragmentExample, MoleculeRecord, normalize_molecule_record
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

__all__ = [
    "AIDSHIVCsvDataset",
    "AIDSHIVRecord",
    "BalancedSamplingResult",
    "CounterfactualPromptCollator",
    "FragmentExample",
    "JsonlMoleculeDataset",
    "MoleculeRecord",
    "PreparationSummary",
    "PreparedSFTExample",
    "PromptBatch",
    "build_balanced_candidate_pool",
    "build_counterfactual_prompt",
    "build_sft_instruction",
    "filter_valid_hiv_records",
    "generate_capped_fragment",
    "label_ratio",
    "load_hiv_dataframe",
    "normalize_molecule_record",
    "prepare_balanced_sft_examples",
    "save_sft_jsonl",
    "sample_random_aids_hiv_record",
    "split_examples",
]
