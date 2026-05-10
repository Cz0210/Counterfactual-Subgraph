#!/usr/bin/env python3
"""QLoRA SFT training for ChemLLM-7B-Chat."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.sft_column_compat import (
    REQUIRED_SFT_COLUMNS,
    SUPPORTED_SFT_FORMATS_MESSAGE,
    build_missing_sft_fields_error,
    normalize_sft_example,
    validate_required_sft_fields,
)
DEFAULT_MODEL_PATH = (
    REPO_ROOT / "pretrained_models" / "ChemLLM-7B-Chat"
)
DEFAULT_TRAIN_FILE = REPO_ROOT / "data" / "sft_v3_core_train.jsonl"
DEFAULT_VAL_FILE = REPO_ROOT / "data" / "sft_v3_core_val.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "ckpt" / "sft_v3_core_lora"
LORA_TARGET_MODULES = (
    "wqkv",
    "wo",
    "w1",
    "w2",
    "w3",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Accepted for HPC wrapper compatibility. The current SFT trainer uses explicit CLI overrides only.",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the local ChemLLM-7B-Chat checkpoint.",
    )
    parser.add_argument(
        "--train-file",
        default=str(DEFAULT_TRAIN_FILE),
        help="Path to the SFT training JSONL.",
    )
    parser.add_argument(
        "--val-file",
        default=str(DEFAULT_VAL_FILE),
        help="Path to the SFT validation JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where checkpoints and tokenizer files will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for training.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Total optimization steps.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="How often to emit trainer logs.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="How often to save checkpoints.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="How often to run validation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate for QLoRA training.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints kept on disk.",
    )
    parser.add_argument(
        "--report-to",
        default="none",
        help=(
            "Trainer reporting backends. Use 'none' to disable integrations, "
            "'wandb', 'tensorboard', or a comma-separated list."
        ),
    )
    return parser


def parse_report_to(report_to: str) -> list[str]:
    """Normalize CLI report_to input into the Trainer-compatible list form."""

    normalized = str(report_to or "none").strip()
    if not normalized or normalized.lower() == "none":
        return []
    return [item.strip() for item in normalized.split(",") if item.strip()]


def load_local_dataset(train_file: Path, val_file: Path):
    """Load the JSONL train/validation splits."""

    data_files = {
        "train": str(train_file),
        "validation": str(val_file),
    }
    return load_dataset("json", data_files=data_files)


def _preview_text(value: object, limit: int) -> str:
    text = "<missing>" if value is None else str(value)
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def normalize_sft_dataset_split(dataset_split, *, split_name: str):
    """Materialize prompt/completion columns before handing data to TRL."""

    original_columns = list(dataset_split.column_names)

    def normalize_and_validate(example: dict[str, object], index: int) -> dict[str, object]:
        normalized = normalize_sft_example(example)
        validate_required_sft_fields(
            normalized,
            available_columns=original_columns,
            split_name=split_name,
            row_index=index,
        )
        return normalized

    normalized_split = dataset_split.map(
        normalize_and_validate,
        with_indices=True,
        desc=f"Normalizing {split_name} prompt/completion columns",
    )

    missing_columns = sorted(set(REQUIRED_SFT_COLUMNS) - set(normalized_split.column_names))
    if missing_columns:
        raise ValueError(
            build_missing_sft_fields_error(
                available_columns=normalized_split.column_names,
                missing_fields=missing_columns,
                split_name=split_name,
            )
        )
    if len(normalized_split) == 0:
        raise ValueError(
            f"{split_name} dataset is empty after normalization; "
            f"available columns={list(normalized_split.column_names)}. "
            f"{SUPPORTED_SFT_FORMATS_MESSAGE}"
        )

    first_example = normalized_split[0]
    print(f"{split_name} dataset column names: {list(normalized_split.column_names)}")
    print(
        f"{split_name} first prompt preview (300 chars): "
        f"{_preview_text(first_example.get('prompt'), 300)}"
    )
    print(
        f"{split_name} first completion preview (100 chars): "
        f"{_preview_text(first_example.get('completion'), 100)}"
    )
    return normalized_split


def build_tokenizer(model_path: Path):
    """Load tokenizer and make sure EOS and PAD are usable."""

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer does not define eos_token, but EOS forcing is required.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_quantized_model(model_path: Path):
    """Load ChemLLM in 4-bit mode and prepare it for QLoRA training."""

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def build_peft_config() -> LoraConfig:
    """Return the fixed LoRA configuration requested for ChemLLM SFT."""

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=list(LORA_TARGET_MODULES),
    )


def build_training_args(output_dir: Path, args: argparse.Namespace) -> TrainingArguments:
    """Construct the fixed training arguments for a stable Trainer setup."""

    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to=parse_report_to(args.report_to),
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
    )


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    model_path = Path(args.model_path).expanduser().resolve()
    train_file = Path(args.train_file).expanduser().resolve()
    val_file = Path(args.val_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    dataset = load_local_dataset(train_file, val_file)
    train_dataset = normalize_sft_dataset_split(dataset["train"], split_name="train")
    eval_dataset = normalize_sft_dataset_split(dataset["validation"], split_name="eval")
    tokenizer = build_tokenizer(model_path)
    model = build_quantized_model(model_path)
    peft_config = build_peft_config()
    training_args = build_training_args(output_dir, args)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = dict(train_result.metrics)
    metrics["train_samples"] = len(train_dataset)
    metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
