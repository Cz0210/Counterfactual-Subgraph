#!/usr/bin/env python3
"""Continue the AIDS SFT-v3 LoRA adapter on fixed Mutagenicity SFT data."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.mutagenicity_continued_sft import (  # noqa: E402
    CompletionOnlyDataCollator,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_VAL_ROWS,
    GENERATION_FIELDS,
    ParentCoverageTracker,
    SupervisedTokenDataset,
    build_checkpoint_manifest,
    dataset_manifest,
    deterministic_smoke_sample,
    ensure_new_output_root,
    load_continued_sft_records,
    score_generated_fragment,
    tokenize_records,
    validate_peft_checkpoint,
    validate_train_val_isolation,
    write_csv_atomic,
    write_json_atomic,
)
from src.utils.env import load_and_merge_config_files  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "train" / "mutagenicity_continued_sft.yaml"
DEFAULT_FINAL_DATA_ROOT = Path("outputs/hpc/mutagenicity/final/sft_ppo_data_v1")
DEFAULT_FALLBACK_DATA_ROOT = Path("outputs/hpc/mutagenicity/sft_ppo_data_v1")
DEFAULT_BASE_MODEL = Path("pretrained_models/ChemLLM-7B-Chat")
DEFAULT_BASE_CHECKPOINT = Path(
    "outputs/hpc/sft_checkpoints/"
    "sft_v3_hiv_20260508_resplit_lr2e4_seed7_fix_columns/checkpoint-500"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--fallback-data-root", type=Path, default=None)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument(
        "--base-model-path",
        "--model-name-or-path",
        dest="base_model_path",
        type=Path,
        default=None,
    )
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--expected-train-rows", type=int, default=None)
    parser.add_argument("--expected-val-rows", type=int, default=None)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--lr-scheduler-type", default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--report-to", default=None)
    parser.add_argument("--generation-samples", type=int, default=None)
    parser.add_argument("--generation-max-new-tokens", type=int, default=None)
    return parser


def _nested(payload: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    dotted_key = ".".join(keys)
    if dotted_key in payload:
        value = payload[dotted_key]
        return default if value is None else value

    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return default if current is None else current


def _choice(cli_value: Any, config_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def _mode_choice(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    name: str,
    *,
    section: str,
    default: Any,
) -> Any:
    cli_value = getattr(args, name)
    mode_value = _nested(config, args.mode, name, default=None)
    common_value = _nested(config, section, name, default=None)
    return _choice(cli_value, mode_value, _choice(common_value, None, default))


def _resolve(path: Path) -> Path:
    expanded = path.expanduser()
    return (REPO_ROOT / expanded).resolve() if not expanded.is_absolute() else expanded.resolve()


def _resolve_data_root(preferred: Path, fallback: Path) -> Path:
    preferred = _resolve(preferred)
    fallback = _resolve(fallback)
    if preferred.is_dir():
        return preferred
    if fallback.is_dir():
        print(
            f"[MUTAGENICITY_SFT_DATA_FALLBACK] preferred={preferred} fallback={fallback}",
            flush=True,
        )
        return fallback
    raise FileNotFoundError(
        f"Neither Mutagenicity SFT data root exists: preferred={preferred} fallback={fallback}"
    )


def _jsonable_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            output[key] = None
        else:
            output[key] = value
    return output


def _generate_samples(
    *,
    model: Any,
    tokenizer: Any,
    records: Sequence[Any],
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - HPC runtime dependency
        raise RuntimeError("Generation sanity evaluation requires PyTorch") from exc

    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    rows: list[dict[str, Any]] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for record in records:
        encoded = tokenizer(record.prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_length = int(encoded["input_ids"].shape[-1])
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
        generated_ids = generated[0][prompt_length:]
        generated_text = tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        rows.append(
            score_generated_fragment(
                record,
                generated_text=generated_text,
                generation_length=int(generated_ids.shape[-1]),
            )
        )
    tokenizer.padding_side = previous_padding_side
    total = len(rows)
    summary = {
        "num_generation_samples": total,
        "parse_rate": (
            sum(bool(row["parse_ok"]) for row in rows) / total if total else 0.0
        ),
        "dummy_rate": (
            sum(bool(row["contains_dummy"]) for row in rows) / total if total else 0.0
        ),
        "empty_rate": (
            sum(bool(row["empty_output"]) for row in rows) / total if total else 0.0
        ),
        "exact_match_rate": (
            sum(bool(row["exact_match"]) for row in rows) / total if total else 0.0
        ),
        "mean_generation_length": (
            sum(int(row["generation_length"]) for row in rows) / total if total else 0.0
        ),
    }
    return rows, summary


def _training_report(
    resolved: Mapping[str, Any],
    manifest: Mapping[str, Any],
    coverage: Mapping[str, Any],
    train_metrics: Mapping[str, Any],
    eval_metrics: Mapping[str, Any],
    checkpoint_manifest: Mapping[str, Any],
) -> str:
    lines = [
        "# Mutagenicity Continued SFT Report",
        "",
        "## Initialization",
        "",
        f"- Base ChemLLM: `{resolved['base_model_path']}`",
        f"- Initial AIDS SFT-v3 adapter: `{resolved['base_checkpoint']}`",
        f"- Tokenizer: `{resolved['tokenizer_path']}`",
        "- Continued training starts a new optimizer/scheduler state; it does not resume the AIDS global step.",
        "",
        "## Data",
        "",
        f"- Full train rows: {manifest['num_train_rows_full']}",
        f"- Full validation rows: {manifest['num_val_rows_full']}",
        f"- Selected train rows: {manifest['num_train_rows_selected']}",
        f"- Selected validation rows: {manifest['num_val_rows_selected']}",
        "- Calibration/test loaded: false",
        "",
        "## Hyperparameters",
        "",
        f"- max_steps: {resolved['max_steps']}",
        f"- max_sequence_length: {resolved['max_sequence_length']}",
        f"- learning_rate: {resolved['learning_rate']}",
        f"- per_device_train_batch_size: {resolved['per_device_train_batch_size']}",
        f"- gradient_accumulation_steps: {resolved['gradient_accumulation_steps']}",
        f"- effective_batch_size: {coverage['effective_batch_size']}",
        f"- scheduler: {resolved['lr_scheduler_type']}",
        f"- warmup_ratio: {resolved['warmup_ratio']}",
        "",
        "## Coverage",
        "",
        f"- num_train_examples_seen: {coverage['num_train_examples_seen']}",
        f"- num_unique_train_parents_seen: {coverage['num_unique_train_parents_seen']}",
        f"- unique_train_parent_coverage: {coverage['unique_train_parent_coverage']:.6f}",
        f"- epochs_equivalent: {coverage['epochs_equivalent']:.6f}",
        f"- global_step: {coverage['global_step']}",
        "",
        "## Metrics",
        "",
        f"- train_loss: {train_metrics.get('train_loss')}",
        f"- eval_loss: {eval_metrics.get('eval_loss')}",
        f"- generation_parse_rate: {eval_metrics.get('generation_parse_rate')}",
        "",
        "## Checkpoints",
        "",
        f"- saved checkpoints: {checkpoint_manifest['num_training_checkpoints']}",
        f"- best checkpoint: `{checkpoint_manifest['best_checkpoint']}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_paths = [Path(value).expanduser().resolve() for value in args.config]
    if not config_paths:
        config_paths = [DEFAULT_CONFIG.resolve()]
    config = load_and_merge_config_files(config_paths)

    preferred_root = Path(
        _choice(
            args.data_root,
            _nested(config, "data", "preferred_root", default=None),
            DEFAULT_FINAL_DATA_ROOT,
        )
    )
    fallback_root = Path(
        _choice(
            args.fallback_data_root,
            _nested(config, "data", "fallback_root", default=None),
            DEFAULT_FALLBACK_DATA_ROOT,
        )
    )
    data_root = _resolve_data_root(preferred_root, fallback_root)
    train_csv = _resolve(
        Path(_choice(args.train_csv, None, data_root / "mutagenicity_sft_train.csv"))
    )
    val_csv = _resolve(
        Path(_choice(args.val_csv, None, data_root / "mutagenicity_sft_val.csv"))
    )
    base_model_path = _resolve(
        Path(
            _choice(
                args.base_model_path,
                _nested(config, "model", "base_model_path", default=None),
                DEFAULT_BASE_MODEL,
            )
        )
    )
    base_checkpoint = _resolve(
        Path(
            _choice(
                args.base_checkpoint,
                _nested(config, "model", "base_checkpoint", default=None),
                DEFAULT_BASE_CHECKPOINT,
            )
        )
    )
    tokenizer_path = _resolve(
        Path(
            _choice(
                args.tokenizer_path,
                _nested(config, "model", "tokenizer_path", default=None),
                base_model_path,
            )
        )
    )
    output_default = (
        Path("outputs/hpc/mutagenicity/sft_continued_v1_smoke")
        if args.mode == "smoke"
        else Path("outputs/hpc/mutagenicity/sft_continued_v1")
    )
    output_root = _resolve(
        Path(
            _choice(
                args.output_root,
                _nested(config, args.mode, "output_root", default=None),
                output_default,
            )
        )
    )

    resolved = {
        "dataset": "Mutagenicity",
        "mode": args.mode,
        "config_paths": [str(path) for path in config_paths],
        "data_root": str(data_root),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "base_model_path": str(base_model_path),
        "base_checkpoint": str(base_checkpoint),
        "tokenizer_path": str(tokenizer_path),
        "output_root": str(output_root),
        "expected_train_rows": int(
            _choice(
                args.expected_train_rows,
                _nested(config, "data", "expected_train_rows", default=None),
                EXPECTED_TRAIN_ROWS,
            )
        ),
        "expected_val_rows": int(
            _choice(
                args.expected_val_rows,
                _nested(config, "data", "expected_val_rows", default=None),
                EXPECTED_VAL_ROWS,
            )
        ),
        "max_train_rows": int(
            _mode_choice(args, config, "max_train_rows", section="training", default=0)
        ),
        "max_val_rows": int(
            _mode_choice(args, config, "max_val_rows", section="training", default=0)
        ),
        "seed": int(_mode_choice(args, config, "seed", section="training", default=7)),
        "max_sequence_length": int(
            _mode_choice(
                args,
                config,
                "max_sequence_length",
                section="training",
                default=DEFAULT_MAX_SEQUENCE_LENGTH,
            )
        ),
        "max_steps": int(
            _mode_choice(args, config, "max_steps", section="training", default=500)
        ),
        "per_device_train_batch_size": int(
            _mode_choice(
                args,
                config,
                "per_device_train_batch_size",
                section="training",
                default=4,
            )
        ),
        "per_device_eval_batch_size": int(
            _mode_choice(
                args,
                config,
                "per_device_eval_batch_size",
                section="training",
                default=4,
            )
        ),
        "gradient_accumulation_steps": int(
            _mode_choice(
                args,
                config,
                "gradient_accumulation_steps",
                section="training",
                default=4,
            )
        ),
        "learning_rate": float(
            _mode_choice(
                args, config, "learning_rate", section="training", default=2e-4
            )
        ),
        "logging_steps": int(
            _mode_choice(
                args, config, "logging_steps", section="training", default=10
            )
        ),
        "save_steps": int(
            _mode_choice(args, config, "save_steps", section="training", default=100)
        ),
        "eval_steps": int(
            _mode_choice(args, config, "eval_steps", section="training", default=100)
        ),
        "save_total_limit": int(
            _mode_choice(
                args, config, "save_total_limit", section="training", default=3
            )
        ),
        "warmup_ratio": float(
            _mode_choice(
                args, config, "warmup_ratio", section="training", default=0.03
            )
        ),
        "lr_scheduler_type": str(
            _mode_choice(
                args, config, "lr_scheduler_type", section="training", default="cosine"
            )
        ),
        "bf16": bool(
            _mode_choice(args, config, "bf16", section="training", default=True)
        ),
        "fp16": bool(
            _mode_choice(args, config, "fp16", section="training", default=False)
        ),
        "report_to": str(
            _mode_choice(args, config, "report_to", section="training", default="none")
        ),
        "generation_samples": int(
            _mode_choice(
                args,
                config,
                "generation_samples",
                section="generation",
                default=32 if args.mode == "smoke" else EXPECTED_VAL_ROWS,
            )
        ),
        "generation_max_new_tokens": int(
            _mode_choice(
                args,
                config,
                "generation_max_new_tokens",
                section="generation",
                default=64,
            )
        ),
        "source_label": 1,
        "target_label": 0,
        "calibration_or_test_loaded": False,
        "initialization_semantics": "base_chemlm_plus_trainable_aids_sft_v3_peft_adapter",
        "optimizer_state_resumed": False,
    }
    if resolved["save_steps"] != resolved["eval_steps"]:
        raise ValueError(
            "save_steps must equal eval_steps so eval-loss best-checkpoint selection "
            "is well-defined"
        )
    if resolved["max_steps"] <= 0:
        raise ValueError("max_steps must be positive")
    if resolved["save_steps"] <= 0 or resolved["save_steps"] > resolved["max_steps"]:
        raise ValueError(
            "save_steps must be positive and no greater than max_steps so at least "
            "one training checkpoint is produced"
        )
    if not base_model_path.is_dir():
        raise FileNotFoundError(f"ChemLLM base model directory does not exist: {base_model_path}")
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Tokenizer directory does not exist: {tokenizer_path}")
    base_checkpoint_audit = validate_peft_checkpoint(base_checkpoint)

    train_all = load_continued_sft_records(
        train_csv,
        expected_split="train",
        expected_count=resolved["expected_train_rows"],
    )
    val_all = load_continued_sft_records(
        val_csv,
        expected_split="val",
        expected_count=resolved["expected_val_rows"],
    )
    isolation = validate_train_val_isolation(train_all, val_all)
    train_records = deterministic_smoke_sample(
        train_all,
        max_rows=resolved["max_train_rows"],
        seed=resolved["seed"],
    )
    val_records = deterministic_smoke_sample(
        val_all,
        max_rows=resolved["max_val_rows"],
        seed=resolved["seed"] + 1,
    )

    try:
        import torch
        from peft import PeftModel
        from transformers import Trainer, TrainingArguments, set_seed
        from scripts.train_sft import (
            build_quantized_model,
            build_tokenizer,
            parse_report_to,
        )
    except ImportError as exc:  # pragma: no cover - HPC runtime dependency
        raise RuntimeError(
            "Mutagenicity continued SFT requires torch, transformers, peft, "
            "bitsandbytes, datasets, and trl in smiles_pip118"
        ) from exc

    set_seed(resolved["seed"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = build_tokenizer(tokenizer_path)
    train_tokens, train_token_audit = tokenize_records(
        tokenizer,
        train_records,
        max_sequence_length=resolved["max_sequence_length"],
    )
    val_tokens, val_token_audit = tokenize_records(
        tokenizer,
        val_records,
        max_sequence_length=resolved["max_sequence_length"],
    )
    output_root = ensure_new_output_root(output_root)

    manifest = dataset_manifest(
        train_path=train_csv,
        val_path=val_csv,
        train_all=train_all,
        val_all=val_all,
        train_selected=train_records,
        val_selected=val_records,
        isolation_audit=isolation,
        seed=resolved["seed"],
    )
    tokenization_audit = {
        "masking": "prompt labels are -100; completion and retained EOS tokens are supervised",
        "train": train_token_audit,
        "val": val_token_audit,
    }
    resolved["base_checkpoint_audit"] = base_checkpoint_audit
    write_json_atomic(output_root / "resolved_config.json", resolved)
    write_json_atomic(output_root / "dataset_manifest.json", manifest)
    write_json_atomic(output_root / "tokenization_audit.json", tokenization_audit)

    print("[MUTAGENICITY_CONTINUED_SFT_CONFIG]", flush=True)
    print(json.dumps(resolved, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    print(f"train_rows={len(train_records)} val_rows={len(val_records)}", flush=True)

    model = build_quantized_model(base_model_path)
    model = PeftModel.from_pretrained(
        model,
        str(base_checkpoint),
        is_trainable=True,
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameters <= 0:
        raise ValueError("Continued SFT loaded no trainable PEFT parameters")
    resolved["trainable_parameters"] = int(trainable_parameters)
    write_json_atomic(output_root / "resolved_config.json", resolved)

    coverage_tracker = ParentCoverageTracker(
        [record.molecule_id for record in train_records]
    )
    collator = CompletionOnlyDataCollator(
        pad_token_id=int(tokenizer.pad_token_id),
        coverage_tracker=coverage_tracker,
    )
    training_args = TrainingArguments(
        output_dir=str(output_root),
        per_device_train_batch_size=resolved["per_device_train_batch_size"],
        per_device_eval_batch_size=resolved["per_device_eval_batch_size"],
        gradient_accumulation_steps=resolved["gradient_accumulation_steps"],
        learning_rate=resolved["learning_rate"],
        max_steps=resolved["max_steps"],
        bf16=resolved["bf16"],
        fp16=resolved["fp16"],
        logging_steps=resolved["logging_steps"],
        save_steps=resolved["save_steps"],
        eval_steps=resolved["eval_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=resolved["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        lr_scheduler_type=resolved["lr_scheduler_type"],
        warmup_ratio=resolved["warmup_ratio"],
        report_to=parse_report_to(resolved["report_to"]),
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        seed=resolved["seed"],
        data_seed=resolved["seed"],
    )
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": SupervisedTokenDataset(train_tokens, track_coverage=True),
        "eval_dataset": SupervisedTokenDataset(val_tokens, track_coverage=False),
        "data_collator": collator,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:  # pragma: no cover - compatibility with older HPC transformers
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(
        **trainer_kwargs,
    )
    train_result = trainer.train()
    train_metrics = _jsonable_metrics(train_result.metrics)
    eval_metrics = _jsonable_metrics(trainer.evaluate())
    trainer.save_model(str(output_root))
    tokenizer.save_pretrained(str(output_root))
    trainer.save_state()

    generation_records = deterministic_smoke_sample(
        val_records,
        max_rows=min(resolved["generation_samples"], len(val_records)),
        seed=resolved["seed"] + 2,
    )
    generation_rows, generation_summary = _generate_samples(
        model=trainer.model,
        tokenizer=tokenizer,
        records=generation_records,
        max_new_tokens=resolved["generation_max_new_tokens"],
    )
    eval_metrics.update(
        {f"generation_{key}": value for key, value in generation_summary.items()}
    )
    trainer.save_metrics("train", train_metrics)
    trainer.save_metrics("eval", eval_metrics)
    write_json_atomic(output_root / "train_metrics.json", train_metrics)
    write_json_atomic(output_root / "eval_metrics.json", eval_metrics)
    write_csv_atomic(
        output_root / "generation_samples.csv", generation_rows, GENERATION_FIELDS
    )

    coverage = coverage_tracker.summary(
        global_step=int(trainer.state.global_step),
        per_device_batch_size=resolved["per_device_train_batch_size"],
        gradient_accumulation_steps=resolved["gradient_accumulation_steps"],
        world_size=int(getattr(training_args, "world_size", 1)),
        current_epoch=(
            float(trainer.state.epoch) if trainer.state.epoch is not None else None
        ),
    )
    coverage["full_coverage_threshold"] = 0.99
    coverage["full_coverage_validation_passed"] = (
        args.mode != "full" or coverage["unique_train_parent_coverage"] >= 0.99
    )
    write_json_atomic(output_root / "training_coverage.json", coverage)
    best_checkpoint = trainer.state.best_model_checkpoint
    best_payload = {
        "selection_split": "val",
        "selection_metric": "eval_loss",
        "best_checkpoint": best_checkpoint,
        "best_metric": trainer.state.best_metric,
        "calibration_or_test_used": False,
    }
    write_json_atomic(output_root / "best_checkpoint.json", best_payload)
    checkpoint_manifest = build_checkpoint_manifest(
        output_root,
        initialization_checkpoint=base_checkpoint,
        best_checkpoint=best_checkpoint,
    )
    write_json_atomic(output_root / "checkpoint_manifest.json", checkpoint_manifest)
    (output_root / "training_report.md").write_text(
        _training_report(
            resolved,
            manifest,
            coverage,
            train_metrics,
            eval_metrics,
            checkpoint_manifest,
        ),
        encoding="utf-8",
    )
    if not coverage["full_coverage_validation_passed"]:
        raise RuntimeError(
            "Full continued SFT did not cover enough unique train parents: "
            f"coverage={coverage['unique_train_parent_coverage']:.6f}"
        )
    completion = {
        "status": "complete",
        "mode": args.mode,
        "global_step": int(trainer.state.global_step),
        "best_checkpoint": best_checkpoint,
        "unique_train_parent_coverage": coverage["unique_train_parent_coverage"],
    }
    write_json_atomic(output_root / "_RUN_COMPLETE.json", completion)
    marker = (
        "[MUTAGENICITY_CONTINUED_SFT_SMOKE_OK]"
        if args.mode == "smoke"
        else "[MUTAGENICITY_CONTINUED_SFT_FULL_OK]"
    )
    print(marker, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
