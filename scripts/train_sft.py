#!/usr/bin/env python3
"""QLoRA SFT training for ChemLLM-7B-Chat with forced EOS alignment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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


def build_text_column_mapper(tokenizer):
    """Create a dataset mapper that materializes one stable text column."""

    eos_token = tokenizer.eos_token

    def add_text_column(example: dict[str, str]) -> dict[str, str]:
        instruction_text = str(example["instruction"])
        output_text = str(example["output"]).strip()
        example["text"] = f"{instruction_text}{output_text}{eos_token}"
        return example

    return add_text_column


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
    tokenizer = build_tokenizer(model_path)
    add_text_column = build_text_column_mapper(tokenizer)
    train_dataset = dataset["train"].map(add_text_column)
    eval_dataset = dataset["validation"].map(add_text_column)
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
