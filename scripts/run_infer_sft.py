#!/usr/bin/env python3
"""Run SFT LoRA inference on a random validation subset and log chemistry stats."""


from __future__ import annotations

import sys
import os
# 将项目根目录加入环境变量
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from rdkit import Chem
except ImportError as exc:  # pragma: no cover - depends on HPC runtime
    raise SystemExit(
        "RDKit is required for scripts/run_infer_sft.py. "
        "Please run this script inside the smiles_pip118 conda environment."
    ) from exc

from src.models.llm_generator import clean_generated_smiles


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = REPO_ROOT / "pretrained_models" / "ChemLLM-7B-Chat"
DEFAULT_LORA_ROOT = REPO_ROOT / "ckpt" / "sft_v3_core_lora"
DEFAULT_VAL_JSONL = REPO_ROOT / "data" / "sft_v3_core_val.jsonl"
DEFAULT_LOG_PATH = REPO_ROOT / "outputs" / "sft_v3_core_eval" / "sft_infer_results.txt"
PROMPT_TEMPLATE = (
    "[System]\n"
    "You are a chemistry assistant. Output ONLY one valid connected substructure "
    "SMILES of the input molecule. Do not output dummy atoms such as '*'. "
    "No extra words, no explanations, no quotes.\n\n"
    "[User]\n"
    "SMILES: {parent_smiles}\n"
    "Return ONE connected substructure as a valid SMILES fragment. "
    "Do not use dummy atom '*'.\n\n"
    "[Assistant]\n"
)
PARENT_SMILES_PATTERN = re.compile(
    r"(?:PARENT_SMILES|SMILES):\s*(?P<smiles>.+?)(?:\n\n\[Assistant\]\n?\s*$|\n\n\[Output\]\n?\s*$)",
    flags=re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class InferenceExample:
    """One validation example for SFT inference."""

    index: int
    parent_smiles: str
    prompt: str
    reference_fragment: str


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """One generated result plus chemistry validation fields."""

    index: int
    parent_smiles: str
    prompt: str
    reference_fragment: str
    raw_generation: str
    generated_fragment: str
    contains_dummy_atom: bool
    is_valid_smiles: bool
    error: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        default=str(DEFAULT_BASE_MODEL),
        help="Path to the local ChemLLM-7B-Chat base model.",
    )
    parser.add_argument(
        "--lora-root",
        default=str(DEFAULT_LORA_ROOT),
        help="Directory containing LoRA checkpoints such as checkpoint-500.",
    )
    parser.add_argument(
        "--val-file",
        default=str(DEFAULT_VAL_JSONL),
        help="Path to the sft_val.jsonl file.",
    )
    parser.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG_PATH),
        help="Path to the detailed inference result log file.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of validation examples to sample for inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for validation sampling.",
    )
    return parser


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Load JSONL rows into memory."""

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected JSON object at line {line_number}, got {type(payload).__name__}."
                )
            rows.append(payload)
    return rows


def extract_parent_smiles(instruction: str) -> str:
    """Parse the parent SMILES back out of the stored instruction string."""

    match = PARENT_SMILES_PATTERN.search(str(instruction))
    if not match:
        raise ValueError("Could not extract parent SMILES from instruction.")
    return match.group("smiles").strip()


def load_inference_examples(val_file: Path, *, sample_size: int, seed: int) -> list[InferenceExample]:
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
        prompt = PROMPT_TEMPLATE.format(parent_smiles=parent_smiles)
        indexed_examples.append(
            InferenceExample(
                index=index,
                parent_smiles=parent_smiles,
                prompt=prompt,
                reference_fragment=output,
            )
        )

    if not indexed_examples:
        raise ValueError(f"No usable validation examples were found in {val_file}")

    rng = random.Random(seed)
    actual_sample_size = min(sample_size, len(indexed_examples))
    return rng.sample(indexed_examples, actual_sample_size)


def find_latest_lora_checkpoint(lora_root: Path) -> Path:
    """Return the numerically latest checkpoint under the LoRA root."""

    if not lora_root.exists():
        raise FileNotFoundError(f"LoRA checkpoint root does not exist: {lora_root}")

    checkpoint_dirs: list[tuple[int, Path]] = []
    for child in lora_root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("checkpoint-"):
            continue
        suffix = child.name.split("checkpoint-", maxsplit=1)[-1]
        if not suffix.isdigit():
            continue
        checkpoint_dirs.append((int(suffix), child))

    if checkpoint_dirs:
        checkpoint_dirs.sort(key=lambda item: item[0])
        return checkpoint_dirs[-1][1]

    adapter_config = lora_root / "adapter_config.json"
    if adapter_config.exists():
        return lora_root

    raise FileNotFoundError(
        "No LoRA checkpoint directory was found under "
        f"{lora_root}. Expected folders such as checkpoint-500."
    )


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


def build_lora_model(base_model_path: Path, checkpoint_path: Path):
    """Load the 4-bit base model and attach the latest LoRA checkpoint."""

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        trust_remote_code=True,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    base_model.config.use_cache = False
    if getattr(base_model, "generation_config", None) is not None:
        base_model.generation_config.use_cache = False

    model = PeftModel.from_pretrained(
        base_model,
        str(checkpoint_path),
        is_trainable=False,
    )
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
    return model


def validate_generated_smiles(smiles: str) -> tuple[bool, bool]:
    """Return whether the candidate contains '*' and is RDKit-parseable."""

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
) -> InferenceResult:
    """Run one forward generation and return cleaned validation metadata."""

    try:
        encoded = tokenizer(example.prompt, return_tensors="pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }
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
        generated_fragment = clean_generated_smiles(raw_generation)
        contains_dummy_atom, is_valid_smiles = validate_generated_smiles(generated_fragment)
        return InferenceResult(
            index=example.index,
            parent_smiles=example.parent_smiles,
            prompt=example.prompt,
            reference_fragment=example.reference_fragment,
            raw_generation=raw_generation,
            generated_fragment=generated_fragment,
            contains_dummy_atom=contains_dummy_atom,
            is_valid_smiles=is_valid_smiles,
        )
    except Exception as exc:
        return InferenceResult(
            index=example.index,
            parent_smiles=example.parent_smiles,
            prompt=example.prompt,
            reference_fragment=example.reference_fragment,
            raw_generation="",
            generated_fragment="",
            contains_dummy_atom=False,
            is_valid_smiles=False,
            error=str(exc),
        )


def write_results_log(
    log_file: Path,
    *,
    base_model_path: Path,
    checkpoint_path: Path,
    results: list[InferenceResult],
) -> None:
    """Write detailed per-example inference results to a text log."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    valid_count = sum(result.is_valid_smiles for result in results)
    dummy_count = sum(result.contains_dummy_atom for result in results)
    both_count = sum(result.is_valid_smiles and result.contains_dummy_atom for result in results)

    with log_file.open("w", encoding="utf-8") as handle:
        handle.write("SFT Inference Results\n")
        handle.write("=====================\n")
        handle.write(f"Base model: {base_model_path}\n")
        handle.write(f"LoRA checkpoint: {checkpoint_path}\n")
        handle.write(f"Total tested: {len(results)}\n")
        handle.write(f"Valid RDKit SMILES: {valid_count}\n")
        handle.write(f"Contains '*': {dummy_count}\n")
        handle.write(f"Valid and contains '*': {both_count}\n\n")

        for result in results:
            handle.write(f"[Sample {result.index}]\n")
            handle.write(f"Parent SMILES: {result.parent_smiles}\n")
            handle.write(f"Prompt: {result.prompt}\n")
            handle.write(f"Reference Fragment: {result.reference_fragment}\n")
            handle.write(f"Raw Generation: {result.raw_generation}\n")
            handle.write(f"Generated Fragment: {result.generated_fragment}\n")
            handle.write(f"Contains '*': {result.contains_dummy_atom}\n")
            handle.write(f"RDKit Valid: {result.is_valid_smiles}\n")
            if result.error:
                handle.write(f"Error: {result.error}\n")
            handle.write("\n")


def main() -> None:
    args = build_parser().parse_args()
    base_model_path = Path(args.base_model).expanduser().resolve()
    lora_root = Path(args.lora_root).expanduser().resolve()
    val_file = Path(args.val_file).expanduser().resolve()
    log_file = Path(args.log_file).expanduser().resolve()

    checkpoint_path = find_latest_lora_checkpoint(lora_root)
    tokenizer = build_tokenizer(base_model_path)
    model = build_lora_model(base_model_path, checkpoint_path)
    sampled_examples = load_inference_examples(
        val_file,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    results: list[InferenceResult] = []
    iterator = tqdm(sampled_examples, total=len(sampled_examples), desc="Running SFT inference")
    for example in iterator:
        result = generate_one(model, tokenizer, example)
        results.append(result)

    valid_count = sum(result.is_valid_smiles for result in results)
    dummy_count = sum(result.contains_dummy_atom for result in results)
    both_count = sum(result.is_valid_smiles and result.contains_dummy_atom for result in results)
    error_count = sum(result.error is not None for result in results)

    write_results_log(
        log_file,
        base_model_path=base_model_path,
        checkpoint_path=checkpoint_path,
        results=results,
    )

    print("SFT inference completed.")
    print(f"Base model: {base_model_path}")
    print(f"LoRA checkpoint: {checkpoint_path}")
    print(f"Validation file: {val_file}")
    print(f"Detailed log: {log_file}")
    print(f"Total tested: {len(results)}")
    print(f"Valid RDKit SMILES: {valid_count}")
    print(f"Contains '*': {dummy_count}")
    print(f"Valid and contains '*': {both_count}")
    print(f"Generation errors: {error_count}")


if __name__ == "__main__":
    main()
