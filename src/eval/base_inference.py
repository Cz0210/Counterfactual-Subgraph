"""Batch base-model inference helpers for SFT-stage comparisons."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from rdkit import Chem
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    raise SystemExit(
        "RDKit is required for base-model inference. "
        "Please run inside the smiles_pip118 environment."
    ) from exc

from src.models.llm_generator import clean_generated_smiles
from src.utils.io import read_jsonl, write_jsonl


PROMPT_TEMPLATE = (
    "[System]\n"
    "Generate a valid, chemically capped subgraph for the following parent molecule. "
    "Output only the fragment SMILES.\n\n"
    "[Input]\n"
    "PARENT_SMILES: {parent_smiles}\n\n"
    "[Output]\n"
)
PARENT_SMILES_PATTERN = re.compile(
    r"PARENT_SMILES:\s*(?P<smiles>.+?)\n\n\[Output\]\n?\s*$",
    flags=re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class InferenceExample:
    """One sampled validation example for base-model inference."""

    index: int
    parent_smiles: str
    prompt: str
    reference_fragment: str


@dataclass(frozen=True, slots=True)
class BaseInferenceResult:
    """One generated result and its quick chemistry checks."""

    index: int
    parent_smiles: str
    prompt: str
    reference_fragment: str
    raw_generation: str
    prediction: str
    contains_dummy_atom: bool
    is_valid_smiles: bool
    error: str | None = None

    def to_json(self) -> dict[str, object]:
        """Return a JSON-serializable record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BaseInferenceSummary:
    """Top-line aggregate metrics for one inference run."""

    total_tested: int
    valid_count: int
    capped_count: int
    valid_and_capped_count: int
    error_count: int

    @property
    def validity_rate(self) -> float:
        if self.total_tested == 0:
            return 0.0
        return 100.0 * self.valid_count / self.total_tested

    @property
    def capping_rate(self) -> float:
        if self.total_tested == 0:
            return 0.0
        return 100.0 * self.capped_count / self.total_tested


def extract_parent_smiles(instruction: str) -> str:
    """Parse the parent SMILES back out of the stored instruction field."""

    match = PARENT_SMILES_PATTERN.search(str(instruction))
    if not match:
        raise ValueError("Could not extract parent SMILES from instruction.")
    return match.group("smiles").strip()


def load_inference_examples(
    val_file: Path,
    *,
    sample_size: int,
    seed: int,
) -> list[InferenceExample]:
    """Read the validation JSONL and sample a deterministic subset."""

    rows = read_jsonl(val_file)
    if not rows:
        raise ValueError(f"No rows were found in validation file: {val_file}")

    indexed_examples: list[InferenceExample] = []
    for index, row in enumerate(rows):
        instruction = str(row.get("instruction", ""))
        output = str(row.get("output", "")).strip()
        if not instruction or not output:
            continue
        parent_smiles = extract_parent_smiles(instruction)
        indexed_examples.append(
            InferenceExample(
                index=index,
                parent_smiles=parent_smiles,
                prompt=PROMPT_TEMPLATE.format(parent_smiles=parent_smiles),
                reference_fragment=output,
            )
        )

    if not indexed_examples:
        raise ValueError(f"No usable validation examples were found in {val_file}")

    import random

    rng = random.Random(seed)
    actual_sample_size = min(sample_size, len(indexed_examples))
    return rng.sample(indexed_examples, actual_sample_size)


def build_tokenizer(base_model_path: Path):
    """Load the base tokenizer for inference."""

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path),
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_base_model(base_model_path: Path):
    """Load the 4-bit base model for deterministic inference."""

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        trust_remote_code=True,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
    model.eval()
    return model


def validate_generated_smiles(smiles: str) -> tuple[bool, bool]:
    """Return whether the prediction contains '*' and is RDKit-parseable."""

    contains_dummy = "*" in str(smiles)
    try:
        molecule = Chem.MolFromSmiles(str(smiles).strip())
    except Exception:
        molecule = None
    return contains_dummy, molecule is not None


def generate_one(
    model,
    tokenizer,
    example: InferenceExample,
) -> BaseInferenceResult:
    """Run one forward generation and return one structured record."""

    try:
        encoded = tokenizer(example.prompt, return_tensors="pt")
        model_device = next(model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=64,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )

        prompt_length = encoded["input_ids"].shape[-1]
        generated_tokens = generated[0][prompt_length:]
        raw_generation = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        prediction = clean_generated_smiles(raw_generation)
        contains_dummy_atom, is_valid_smiles = validate_generated_smiles(prediction)
        return BaseInferenceResult(
            index=example.index,
            parent_smiles=example.parent_smiles,
            prompt=example.prompt,
            reference_fragment=example.reference_fragment,
            raw_generation=raw_generation,
            prediction=prediction,
            contains_dummy_atom=contains_dummy_atom,
            is_valid_smiles=is_valid_smiles,
        )
    except Exception as exc:
        return BaseInferenceResult(
            index=example.index,
            parent_smiles=example.parent_smiles,
            prompt=example.prompt,
            reference_fragment=example.reference_fragment,
            raw_generation="",
            prediction="",
            contains_dummy_atom=False,
            is_valid_smiles=False,
            error=str(exc),
        )


def write_base_inference_jsonl(log_file: Path, results: list[BaseInferenceResult]) -> None:
    """Persist the base inference results as JSONL."""

    write_jsonl(log_file, (result.to_json() for result in results))


def summarize_results(results: list[BaseInferenceResult]) -> BaseInferenceSummary:
    """Aggregate top-line counts over all inference results."""

    valid_count = sum(result.is_valid_smiles for result in results)
    capped_count = sum(result.contains_dummy_atom for result in results)
    both_count = sum(result.is_valid_smiles and result.contains_dummy_atom for result in results)
    error_count = sum(result.error is not None for result in results)
    return BaseInferenceSummary(
        total_tested=len(results),
        valid_count=valid_count,
        capped_count=capped_count,
        valid_and_capped_count=both_count,
        error_count=error_count,
    )
