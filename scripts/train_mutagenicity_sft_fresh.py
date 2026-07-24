#!/usr/bin/env python3
"""Train one fresh Mutagenicity LoRA from the pure ChemLLM base model."""

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
    dataset_manifest,
    deterministic_smoke_sample,
    ensure_new_output_root,
    load_continued_sft_records,
    score_generated_fragment,
    tokenize_records,
    validate_train_val_isolation,
    write_csv_atomic,
)
from src.train.mutagenicity_fresh_sft import (  # noqa: E402
    DATASET_VARIANTS,
    FreshLoRAConfig,
    audit_fresh_lora_model,
    fresh_checkpoint_manifest,
    initialize_fresh_lora,
    resolve_variant_csvs,
    tokenizer_reuse_audit,
    write_json_atomic,
)
from src.utils.env import load_and_merge_config_files  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "train" / "mutagenicity_fresh_sft.yaml"
DEFAULT_BASE_MODEL = Path("pretrained_models/ChemLLM-7B-Chat")
DEFAULT_V1_DATA_ROOT = Path("outputs/hpc/mutagenicity/final/sft_ppo_data_v1")
DEFAULT_V2_DATA_ROOT = Path("outputs/hpc/mutagenicity/final/sft_ppo_data_v2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--dataset-variant", choices=tuple(DATASET_VARIANTS), default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--base-model-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--tokenizer-fallback-path", type=Path, default=None)
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
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--report-to", default=None)
    parser.add_argument("--generation-samples", type=int, default=None)
    parser.add_argument("--generation-max-new-tokens", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--lora-target-modules", default=None)
    return parser


def _nested(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    dotted = ".".join(keys)
    if dotted in config:
        return config[dotted]
    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return default if current is None else current


def _pick(cli_value: Any, config_value: Any, default: Any) -> Any:
    return cli_value if cli_value is not None else (
        config_value if config_value is not None else default
    )


def _mode_pick(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    name: str,
    *,
    section: str,
    default: Any,
) -> Any:
    return _pick(
        getattr(args, name),
        _nested(config, args.mode, name, default=None),
        _nested(config, section, name, default=default),
    )


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (REPO_ROOT / value).resolve() if not value.is_absolute() else value.resolve()


def _metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in payload.items():
        if hasattr(value, "item"):
            value = value.item()
        output[str(key)] = (
            None if isinstance(value, float) and not math.isfinite(value) else value
        )
    return output


def _generate_samples(
    *,
    model: Any,
    tokenizer: Any,
    records: Sequence[Any],
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    previous_padding = tokenizer.padding_side
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
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        rows.append(
            score_generated_fragment(
                record,
                generated_text=text,
                generation_length=int(generated_ids.shape[-1]),
            )
        )
    tokenizer.padding_side = previous_padding
    total = len(rows)
    return rows, {
        "num_generation_samples": total,
        "parse_rate": sum(bool(row["parse_ok"]) for row in rows) / total if total else 0.0,
        "dummy_rate": sum(bool(row["contains_dummy"]) for row in rows) / total if total else 0.0,
        "empty_rate": sum(bool(row["empty_output"]) for row in rows) / total if total else 0.0,
        "exact_match_rate": sum(bool(row["exact_match"]) for row in rows) / total if total else 0.0,
    }


def _load_tokenizer(
    *,
    base_path: Path,
    requested_path: Path,
    fallback_path: Path | None,
    builder: Any,
) -> tuple[Any, Path, dict[str, Any]]:
    try:
        tokenizer = builder(requested_path)
        source = requested_path
        fallback_reason = None
    except (FileNotFoundError, OSError, ValueError) as exc:
        if fallback_path is None:
            raise
        tokenizer = builder(fallback_path)
        source = fallback_path
        fallback_reason = f"{type(exc).__name__}: {exc}"
    audit = tokenizer_reuse_audit(
        tokenizer=tokenizer,
        base_model_path=base_path,
        tokenizer_path=source,
    )
    audit["fallback_reason"] = fallback_reason
    return tokenizer, source, audit


def _resolve_settings(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    variant = str(
        _pick(
            args.dataset_variant,
            _nested(config, "data", "dataset_variant", default=None),
            "strict_v2",
        )
    )
    default_root = DEFAULT_V1_DATA_ROOT if variant == "current_v1" else DEFAULT_V2_DATA_ROOT
    data_root = _resolve(
        _pick(args.data_root, _nested(config, "data", "root", default=None), default_root)
    )
    default_train, default_val = resolve_variant_csvs(data_root, variant)
    train_csv = _resolve(_pick(args.train_csv, None, default_train))
    val_csv = _resolve(_pick(args.val_csv, None, default_val))
    base_model = _resolve(
        _pick(
            args.base_model_path,
            _nested(config, "model", "base_model_path", default=None),
            DEFAULT_BASE_MODEL,
        )
    )
    tokenizer_path = _resolve(
        _pick(
            args.tokenizer_path,
            _nested(config, "model", "tokenizer_path", default=None),
            base_model,
        )
    )
    fallback_raw = _pick(
        args.tokenizer_fallback_path,
        _nested(config, "model", "tokenizer_fallback_path", default=None),
        None,
    )
    fallback = _resolve(fallback_raw) if fallback_raw else None
    output_default = Path(
        "outputs/hpc/mutagenicity/sft_fresh_strict_v2_smoke"
        if args.mode == "smoke"
        else "outputs/hpc/mutagenicity/sft_fresh_strict_v2"
    )
    output_root = _resolve(
        _pick(
            args.output_root,
            _nested(config, args.mode, "output_root", default=None),
            output_default,
        )
    )
    current_v1 = variant == "current_v1"
    return {
        "dataset": "Mutagenicity",
        "mode": args.mode,
        "dataset_variant": variant,
        "data_root": str(data_root),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "base_model_path": str(base_model),
        "tokenizer_path_requested": str(tokenizer_path),
        "tokenizer_fallback_path": str(fallback) if fallback else None,
        "output_root": str(output_root),
        "expected_train_rows": int(
            _pick(
                args.expected_train_rows,
                _nested(config, "data", "expected_train_rows", default=None),
                EXPECTED_TRAIN_ROWS if current_v1 else 0,
            )
        ),
        "expected_val_rows": int(
            _pick(
                args.expected_val_rows,
                _nested(config, "data", "expected_val_rows", default=None),
                EXPECTED_VAL_ROWS if current_v1 else 0,
            )
        ),
        "max_train_rows": int(_mode_pick(args, config, "max_train_rows", section="training", default=0)),
        "max_val_rows": int(_mode_pick(args, config, "max_val_rows", section="training", default=0)),
        "seed": int(_mode_pick(args, config, "seed", section="training", default=7)),
        "max_sequence_length": int(_mode_pick(args, config, "max_sequence_length", section="training", default=DEFAULT_MAX_SEQUENCE_LENGTH)),
        "max_steps": int(_mode_pick(args, config, "max_steps", section="training", default=300)),
        "per_device_train_batch_size": int(_mode_pick(args, config, "per_device_train_batch_size", section="training", default=4)),
        "per_device_eval_batch_size": int(_mode_pick(args, config, "per_device_eval_batch_size", section="training", default=4)),
        "gradient_accumulation_steps": int(_mode_pick(args, config, "gradient_accumulation_steps", section="training", default=4)),
        "learning_rate": float(_mode_pick(args, config, "learning_rate", section="training", default=2e-4)),
        "logging_steps": int(_mode_pick(args, config, "logging_steps", section="training", default=10)),
        "save_steps": int(_mode_pick(args, config, "save_steps", section="training", default=50)),
        "eval_steps": int(_mode_pick(args, config, "eval_steps", section="training", default=50)),
        "save_total_limit": int(_mode_pick(args, config, "save_total_limit", section="training", default=8)),
        "warmup_ratio": float(_mode_pick(args, config, "warmup_ratio", section="training", default=0.03)),
        "lr_scheduler_type": str(_mode_pick(args, config, "lr_scheduler_type", section="training", default="cosine")),
        "early_stopping_patience": int(_mode_pick(args, config, "early_stopping_patience", section="training", default=2)),
        "bf16": bool(_mode_pick(args, config, "bf16", section="training", default=True)),
        "fp16": bool(_mode_pick(args, config, "fp16", section="training", default=False)),
        "report_to": str(_mode_pick(args, config, "report_to", section="training", default="none")),
        "generation_samples": int(_mode_pick(args, config, "generation_samples", section="generation", default=32)),
        "generation_max_new_tokens": int(_mode_pick(args, config, "generation_max_new_tokens", section="generation", default=64)),
        "lora_rank": int(_pick(args.lora_rank, _nested(config, "lora", "rank", default=None), 8)),
        "lora_alpha": int(_pick(args.lora_alpha, _nested(config, "lora", "alpha", default=None), 16)),
        "lora_dropout": float(_pick(args.lora_dropout, _nested(config, "lora", "dropout", default=None), 0.05)),
        "lora_target_modules": str(_pick(args.lora_target_modules, _nested(config, "lora", "target_modules", default=None), "wqkv,wo,w1,w2,w3")),
        "source_label": 1,
        "target_label": 0,
        "calibration_or_test_loaded": False,
        "source_adapter_checkpoint": None,
        "aids_adapter_weights_loaded": False,
        "initialization_semantics": "pure_chemlm_plus_random_fresh_lora",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_paths = [Path(value).expanduser().resolve() for value in args.config]
    if not config_paths:
        config_paths = [DEFAULT_CONFIG.resolve()]
    config = load_and_merge_config_files(config_paths)
    resolved = _resolve_settings(args, config)
    resolved["config_paths"] = [str(path) for path in config_paths]

    if resolved["save_steps"] != resolved["eval_steps"]:
        raise ValueError("Fresh SFT save_steps must equal eval_steps")
    if resolved["max_steps"] <= 0 or not 0 < resolved["save_steps"] <= resolved["max_steps"]:
        raise ValueError("Fresh SFT requires 0 < save_steps <= max_steps")
    base_model = Path(resolved["base_model_path"])
    tokenizer_requested = Path(resolved["tokenizer_path_requested"])
    tokenizer_fallback = (
        Path(resolved["tokenizer_fallback_path"])
        if resolved["tokenizer_fallback_path"]
        else None
    )
    for required in (Path(resolved["train_csv"]), Path(resolved["val_csv"]), base_model):
        if not required.exists():
            raise FileNotFoundError(f"Required Fresh SFT input does not exist: {required}")

    allow_duplicates = resolved["dataset_variant"] == "strict_multitarget_v2"
    train_all = load_continued_sft_records(
        resolved["train_csv"],
        expected_split="train",
        expected_count=resolved["expected_train_rows"] or None,
        allow_duplicate_parents=allow_duplicates,
    )
    val_all = load_continued_sft_records(
        resolved["val_csv"],
        expected_split="val",
        expected_count=resolved["expected_val_rows"] or None,
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
        from peft import LoraConfig, TaskType
        from transformers import (
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )
        from scripts.train_sft import (
            build_quantized_model,
            build_tokenizer,
            parse_report_to,
        )
    except ImportError as exc:  # pragma: no cover - HPC dependency
        raise RuntimeError(
            "Fresh Mutagenicity SFT requires torch, transformers, peft, "
            "bitsandbytes, and RDKit in smiles_pip118"
        ) from exc

    set_seed(resolved["seed"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer, tokenizer_source, tokenizer_audit = _load_tokenizer(
        base_path=base_model,
        requested_path=tokenizer_requested,
        fallback_path=tokenizer_fallback,
        builder=build_tokenizer,
    )
    resolved["tokenizer_path_resolved"] = str(tokenizer_source)
    resolved["tokenizer_audit"] = tokenizer_audit
    train_tokens, train_token_audit = tokenize_records(
        tokenizer, train_records, max_sequence_length=resolved["max_sequence_length"]
    )
    val_tokens, val_token_audit = tokenize_records(
        tokenizer, val_records, max_sequence_length=resolved["max_sequence_length"]
    )
    output_root = ensure_new_output_root(resolved["output_root"])
    manifest = dataset_manifest(
        train_path=resolved["train_csv"],
        val_path=resolved["val_csv"],
        train_all=train_all,
        val_all=val_all,
        train_selected=train_records,
        val_selected=val_records,
        isolation_audit=isolation,
        seed=resolved["seed"],
    )
    manifest["dataset_variant"] = resolved["dataset_variant"]
    write_json_atomic(output_root / "resolved_config.json", resolved)
    write_json_atomic(output_root / "dataset_manifest.json", manifest)
    write_json_atomic(
        output_root / "tokenization_audit.json",
        {
            "masking": "prompt=-100; completion and retained EOS are supervised",
            "train": train_token_audit,
            "val": val_token_audit,
        },
    )

    lora_settings = FreshLoRAConfig(
        rank=resolved["lora_rank"],
        alpha=resolved["lora_alpha"],
        dropout=resolved["lora_dropout"],
        target_modules=tuple(
            value.strip()
            for value in resolved["lora_target_modules"].split(",")
            if value.strip()
        ),
    )
    lora_settings.validate()
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_settings.rank,
        lora_alpha=lora_settings.alpha,
        lora_dropout=lora_settings.dropout,
        bias="none",
        target_modules=list(lora_settings.target_modules),
    )
    pure_model = build_quantized_model(base_model)
    model, loading_audit = initialize_fresh_lora(
        pure_model,
        lora_config=peft_config,
    )
    initialization_audit = audit_fresh_lora_model(
        model,
        base_model_path=base_model,
        loading_audit=loading_audit,
        lora_settings=lora_settings,
    )
    initialization_audit["tokenizer"] = tokenizer_audit
    write_json_atomic(
        output_root / "fresh_initialization_audit.json", initialization_audit
    )
    print("[MUTAGENICITY_FRESH_SFT_INITIALIZATION_AUDIT]", flush=True)
    for key in (
        "base_model_path",
        "adapter_initialized_from_scratch",
        "source_adapter_checkpoint",
        "aids_adapter_weights_loaded",
        "base_parameter_trainable_count",
        "adapter_trainable_parameter_count",
        "adapter_names",
        "active_adapters",
        "single_active_adapter",
        "initialization_audit_passed",
    ):
        print(f"{key}={initialization_audit[key]}", flush=True)
    print("[MUTAGENICITY_FRESH_SFT_INITIALIZATION_AUDIT_OK]", flush=True)

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
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": SupervisedTokenDataset(train_tokens, track_coverage=True),
        "eval_dataset": SupervisedTokenDataset(val_tokens, track_coverage=False),
        "data_collator": collator,
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=resolved["early_stopping_patience"]
            )
        ],
    }
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:  # pragma: no cover - older transformers
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    train_result = trainer.train()
    train_metrics = _metrics(train_result.metrics)
    eval_metrics = _metrics(trainer.evaluate())
    trainer.save_model(str(output_root))
    tokenizer.save_pretrained(str(output_root))
    trainer.save_state()

    generation_records = deterministic_smoke_sample(
        val_records,
        max_rows=min(resolved["generation_samples"], len(val_records)),
        seed=resolved["seed"] + 2,
    )
    generation_rows, generation_metrics = _generate_samples(
        model=trainer.model,
        tokenizer=tokenizer,
        records=generation_records,
        max_new_tokens=resolved["generation_max_new_tokens"],
    )
    eval_metrics.update(
        {f"generation_{key}": value for key, value in generation_metrics.items()}
    )
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
    write_json_atomic(output_root / "training_coverage.json", coverage)
    token_best = trainer.state.best_model_checkpoint
    write_json_atomic(
        output_root / "best_token_loss_checkpoint.json",
        {
            "checkpoint": token_best,
            "metric": "eval_loss",
            "value": trainer.state.best_metric,
            "selection_split": "val",
            "task_level_selection_required": True,
            "calibration_or_test_used": False,
        },
    )
    checkpoint_manifest = fresh_checkpoint_manifest(
        output_root,
        best_token_loss_checkpoint=token_best,
    )
    write_json_atomic(output_root / "checkpoint_manifest.json", checkpoint_manifest)
    report = [
        "# Mutagenicity Fresh SFT",
        "",
        f"- Variant: `{resolved['dataset_variant']}`",
        f"- Base model: `{base_model}`",
        "- Source adapter checkpoint: `null`",
        "- AIDS adapter weights loaded: `false`",
        f"- Fresh LoRA trainable parameters: {initialization_audit['adapter_trainable_parameter_count']}",
        f"- Training steps: {trainer.state.global_step}",
        f"- Token-loss best checkpoint: `{token_best}`",
        "- Task-level best checkpoint must be selected by `evaluate_mutagenicity_generator.py`.",
        f"- Unique train parent coverage: {coverage['unique_train_parent_coverage']:.6f}",
        "- Calibration/test loaded: false",
        "",
    ]
    (output_root / "training_report.md").write_text("\n".join(report), encoding="utf-8")
    write_json_atomic(
        output_root / "_RUN_COMPLETE.json",
        {
            "status": "complete",
            "mode": args.mode,
            "dataset_variant": resolved["dataset_variant"],
            "global_step": int(trainer.state.global_step),
            "best_token_loss_checkpoint": token_best,
            "task_level_selection_pending": True,
        },
    )
    print(
        "[MUTAGENICITY_FRESH_SFT_SMOKE_OK]"
        if args.mode == "smoke"
        else "[MUTAGENICITY_FRESH_SFT_FULL_OK]",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
