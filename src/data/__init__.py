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

try:  # pragma: no cover - optional compatibility layer
    from src.data.hiv_dataset_utils import (
        HIVParentRecord,
        label_counts,
        load_hiv_dataframe as load_hiv_dataframe_v2,
        normalize_hiv_records,
        resolve_hiv_column_names,
        sample_records_by_strata,
        stratified_round_robin_order,
    )
    from src.data.sft_v3_builder import (
        SFTV3BuildArtifacts,
        SFTV3BuilderConfig,
        SFTV3Example,
        SFTV3ReferenceCandidate,
        build_and_write_sft_v3_dataset,
        select_reference_candidate_for_parent,
        split_examples_scaffold_aware,
    )
except ImportError:  # pragma: no cover - optional build dependency path
    HIVParentRecord = None
    label_counts = None
    load_hiv_dataframe_v2 = None
    normalize_hiv_records = None
    resolve_hiv_column_names = None
    sample_records_by_strata = None
    stratified_round_robin_order = None
    SFTV3BuildArtifacts = None
    SFTV3BuilderConfig = None
    SFTV3Example = None
    SFTV3ReferenceCandidate = None
    build_and_write_sft_v3_dataset = None
    select_reference_candidate_for_parent = None
    split_examples_scaffold_aware = None
else:
    __all__.extend(
        [
            "HIVParentRecord",
            "label_counts",
            "load_hiv_dataframe_v2",
            "normalize_hiv_records",
            "resolve_hiv_column_names",
            "sample_records_by_strata",
            "stratified_round_robin_order",
            "SFTV3BuildArtifacts",
            "SFTV3BuilderConfig",
            "SFTV3Example",
            "SFTV3ReferenceCandidate",
            "build_and_write_sft_v3_dataset",
            "select_reference_candidate_for_parent",
            "split_examples_scaffold_aware",
        ]
    )
