#!/usr/bin/env python3
"""Run batch inference with the base ChemLLM model on a validation subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.base_inference import (
    build_base_model,
    build_tokenizer,
    generate_one,
    load_inference_examples,
    summarize_results,
    write_base_inference_jsonl,
)


DEFAULT_BASE_MODEL = REPO_ROOT / "pretrained_models" / "ChemLLM-7B-Chat"
DEFAULT_VAL_JSONL = REPO_ROOT / "data" / "sft_val.jsonl"
DEFAULT_LOG_PATH = REPO_ROOT / "outputs" / "hpc" / "logs" / "base_infer_results.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        default=str(DEFAULT_BASE_MODEL),
        help="Path to the local ChemLLM-7B-Chat base model.",
    )
    parser.add_argument(
        "--val-file",
        default=str(DEFAULT_VAL_JSONL),
        help="Path to the sft_val.jsonl file.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(DEFAULT_LOG_PATH),
        help="Path to the JSONL file where detailed base inference results will be saved.",
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


def main() -> None:
    args = build_parser().parse_args()
    base_model_path = Path(args.base_model).expanduser().resolve()
    val_file = Path(args.val_file).expanduser().resolve()
    output_jsonl = Path(args.output_jsonl).expanduser().resolve()

    tokenizer = build_tokenizer(base_model_path)
    model = build_base_model(base_model_path)
    sampled_examples = load_inference_examples(
        val_file,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    results = []
    iterator = tqdm(sampled_examples, total=len(sampled_examples), desc="Running base-model inference")
    for example in iterator:
        results.append(generate_one(model, tokenizer, example))

    write_base_inference_jsonl(output_jsonl, results)
    summary = summarize_results(results)

    print("Base-model inference completed.")
    print(f"Base model: {base_model_path}")
    print(f"Validation file: {val_file}")
    print(f"Detailed JSONL: {output_jsonl}")
    print(f"Total tested: {summary.total_tested}")
    print(f"Valid RDKit SMILES: {summary.valid_count}")
    print(f"Contains '*': {summary.capped_count}")
    print(f"Valid and contains '*': {summary.valid_and_capped_count}")
    print(f"Generation errors: {summary.error_count}")
    print(f"Base Validity: {summary.validity_rate:.2f}%")
    print(f"Base Capping Rate: {summary.capping_rate:.2f}%")


if __name__ == "__main__":
    main()
