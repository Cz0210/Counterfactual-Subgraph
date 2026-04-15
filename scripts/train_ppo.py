#!/usr/bin/env python3
"""PPO training entrypoint for deletion-based counterfactual fragment generation.

This script is designed for a single high-memory GPU node such as A800.
It loads:

1. the ChemLLM base model in 4-bit mode,
2. the SFT LoRA adapter as the initial PPO policy,
3. the AIDS activity oracle for reward computation,
4. prompts from either the raw HIV CSV or a JSONL prompt file.

The reward logic is delegated to ``ChemRLRewarder`` and follows the v3
counterfactual objective: score the residual molecule after fragment deletion.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
from statistics import mean
from typing import Any, Sequence

from src.data.prompts import build_counterfactual_prompt
from src.data.schemas import MoleculeRecord
from src.models import clean_generated_smiles
from src.rewards import analyze_batch_collapse
from src.reward.reward_wrapper import ChemRLRewarder, RewardTrace
from src.utils.io import ensure_directory, read_jsonl
from src.utils.logging_utils import RunContext, configure_run_logger, write_runtime_manifest
from src.utils import apply_dotlist_overrides, load_and_merge_config_files


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = REPO_ROOT / "pretrained_models" / "ChemLLM-7B-Chat"
DEFAULT_SFT_ADAPTER = REPO_ROOT / "outputs" / "hpc" / "sft_checkpoints" / "checkpoint-500"
DEFAULT_ORACLE_PATH = REPO_ROOT / "outputs" / "hpc" / "oracle" / "aids_rf_model.pkl"
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "raw" / "AIDS" / "HIV.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "hpc" / "rl_checkpoints"

_SMILES_FIELD_PATTERNS = (
    re.compile(r"MOLECULE_SMILES:\s*(?P<smiles>[^\n\r]+)"),
    re.compile(r"PARENT_SMILES:\s*(?P<smiles>[^\n\r]+)"),
)
_LABEL_PATTERN = re.compile(r"ORIGINAL_LABEL:\s*(?P<label>[01])")


@dataclass(frozen=True, slots=True)
class PromptExample:
    """One PPO prompt example together with the parent molecule metadata."""

    index: int
    prompt: str
    parent_smiles: str
    original_label: int

    def to_dataset_row(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "query": self.prompt,
            "parent_smiles": self.parent_smiles,
            "original_label": int(self.original_label),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_BASE_MODEL),
        help="本地 ChemLLM-7B-Chat 基座路径。",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="附加配置文件，默认支持显式传入 configs/hpc.yaml。",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="对配置做 dotted-key 覆盖，例如 --set training.max_steps=1000。",
    )
    parser.add_argument(
        "--sft-adapter-path",
        default=str(DEFAULT_SFT_ADAPTER),
        help="SFT LoRA checkpoint 路径，例如 checkpoint-500。",
    )
    parser.add_argument(
        "--oracle-path",
        default=str(DEFAULT_ORACLE_PATH),
        help="AIDS Oracle bundle 路径。",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(DEFAULT_DATASET_PATH),
        help="训练数据路径，支持 HIV CSV 或 JSONL prompt 文件。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="PPO checkpoint 和日志输出目录。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="随机种子。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="PPO 优化步数上限。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="PPO batch_size。",
    )
    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=4,
        help="PPO mini_batch_size。",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=4,
        help="每个 PPO step 的内部优化轮数。",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.41e-5,
        help="PPO 学习率。",
    )
    parser.add_argument(
        "--init-kl-coef",
        type=float,
        default=0.1,
        help="初始 KL 惩罚系数。",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="梯度累积步数。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="每次 fragment 生成的最大新 token 数。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="采样温度。",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="nucleus sampling top-p。",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="每多少个 PPO step 保存一次 checkpoint。",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="每多少个 PPO step 输出一次摘要日志。",
    )
    parser.add_argument(
        "--max-prompt-examples",
        type=int,
        default=0,
        help="最多载入多少个 prompt，0 表示不截断。",
    )
    parser.add_argument(
        "--default-parent-label",
        type=int,
        default=1,
        choices=(0, 1),
        help="当数据里没有标签时默认使用的原始标签。AIDS 任务默认用 1 表示 Active。",
    )
    parser.add_argument(
        "--only-positive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="若数据带标签，则默认仅保留原始活性为 1 的分子做 PPO。",
    )
    parser.add_argument(
        "--include-label-in-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="从原始 CSV 构建 prompt 时是否显式写入 ORIGINAL_LABEL。",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否只从本地文件系统加载 Hugging Face 模型与 tokenizer。",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用 trust_remote_code，以兼容 InternLM2 架构。",
    )
    return parser


def apply_config_overrides(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Apply lightweight YAML config overrides to the PPO CLI args.

    The rule is:
    - explicit CLI values always win;
    - config values fill in fields that are still using parser defaults.
    """

    config_files = [REPO_ROOT / "configs" / "base.yaml"]
    config_files.extend(Path(path).expanduser().resolve() for path in args.config)
    if len(config_files) == 1:
        return args

    config = load_and_merge_config_files(config_files)
    if args.set:
        config = apply_dotlist_overrides(config, args.set)

    def _is_default(name: str) -> bool:
        return getattr(args, name) == parser.get_default(name)

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    runtime_cfg = config.get("runtime", {})
    paths_cfg = config.get("paths", {})

    if _is_default("model_path"):
        configured_model_path = model_cfg.get("model_path") or model_cfg.get("model_name_or_path")
        if configured_model_path:
            args.model_path = str(configured_model_path)

    if _is_default("dataset_path"):
        configured_dataset_path = data_cfg.get("dataset_path") or data_cfg.get("raw_hiv_csv")
        if configured_dataset_path:
            args.dataset_path = str(configured_dataset_path)

    if _is_default("output_dir"):
        configured_output_root = paths_cfg.get("output_root")
        if configured_output_root:
            args.output_dir = str(Path(str(configured_output_root)) / "rl_checkpoints")

    if _is_default("seed") and runtime_cfg.get("seed") is not None:
        args.seed = int(runtime_cfg["seed"])

    if _is_default("max_steps") and training_cfg.get("max_steps") is not None:
        args.max_steps = int(training_cfg["max_steps"])

    if _is_default("batch_size") and training_cfg.get("batch_size") is not None:
        args.batch_size = int(training_cfg["batch_size"])

    if (
        _is_default("gradient_accumulation_steps")
        and training_cfg.get("gradient_accumulation_steps") is not None
    ):
        args.gradient_accumulation_steps = int(training_cfg["gradient_accumulation_steps"])

    if _is_default("learning_rate") and training_cfg.get("learning_rate") is not None:
        args.learning_rate = float(training_cfg["learning_rate"])

    if _is_default("local_files_only") and runtime_cfg.get("local_files_only") is not None:
        args.local_files_only = bool(runtime_cfg["local_files_only"])

    if _is_default("trust_remote_code") and model_cfg.get("trust_remote_code") is not None:
        args.trust_remote_code = bool(model_cfg["trust_remote_code"])

    return args


def normalize_hiv_label(value: object) -> int | None:
    """Normalize HIV labels from either MoleculeNet or raw NCI formats."""

    if value is None:
        return None

    normalized = str(value).strip().upper()
    mapping = {
        "CI": 0,
        "CM": 1,
        "CA": 1,
        "0": 0,
        "1": 1,
    }
    if normalized in mapping:
        return mapping[normalized]

    try:
        integer_value = int(float(str(value).strip()))
    except Exception:
        return None
    return integer_value if integer_value in (0, 1) else None


def extract_parent_smiles_from_prompt(prompt: str) -> str:
    """Parse the parent molecule back out of one stored prompt string."""

    text = str(prompt or "")
    for pattern in _SMILES_FIELD_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group("smiles").strip()
    raise ValueError("Could not extract parent SMILES from the prompt text.")


def extract_label_from_prompt(prompt: str) -> int | None:
    """Parse ORIGINAL_LABEL back out of a prompt when present."""

    match = _LABEL_PATTERN.search(str(prompt or ""))
    if not match:
        return None
    return int(match.group("label"))


def build_prompt_example_from_json_row(
    row: dict[str, Any],
    *,
    index: int,
    default_parent_label: int,
    only_positive: bool,
) -> PromptExample | None:
    """Normalize one JSONL row into the PPO prompt contract."""

    prompt = str(
        row.get("prompt")
        or row.get("instruction")
        or row.get("query")
        or row.get("text")
        or ""
    ).strip()

    parent_smiles = str(row.get("parent_smiles") or row.get("smiles") or "").strip()
    if not parent_smiles and prompt:
        try:
            parent_smiles = extract_parent_smiles_from_prompt(prompt)
        except ValueError:
            parent_smiles = ""

    label_candidates = (
        row.get("original_label"),
        row.get("label"),
        row.get("HIV_active"),
        row.get("activity"),
    )
    original_label = next(
        (
            normalized
            for normalized in (normalize_hiv_label(candidate) for candidate in label_candidates)
            if normalized is not None
        ),
        None,
    )
    if original_label is None and prompt:
        original_label = extract_label_from_prompt(prompt)
    if original_label is None:
        original_label = int(default_parent_label)

    if only_positive and original_label != 1:
        return None

    if not prompt and parent_smiles:
        record = MoleculeRecord(record_id=index, smiles=parent_smiles, label=original_label)
        prompt = build_counterfactual_prompt(record, include_label=True)

    if not prompt or not parent_smiles:
        return None

    return PromptExample(
        index=index,
        prompt=prompt,
        parent_smiles=parent_smiles,
        original_label=int(original_label),
    )


def load_prompt_examples_from_jsonl(
    dataset_path: Path,
    *,
    default_parent_label: int,
    only_positive: bool,
) -> list[PromptExample]:
    """Load PPO prompts from a JSONL file."""

    examples: list[PromptExample] = []
    for index, row in enumerate(read_jsonl(dataset_path)):
        normalized = build_prompt_example_from_json_row(
            row,
            index=index,
            default_parent_label=default_parent_label,
            only_positive=only_positive,
        )
        if normalized is not None:
            examples.append(normalized)
    return examples


def load_prompt_examples_from_csv(
    dataset_path: Path,
    *,
    default_parent_label: int,
    only_positive: bool,
    include_label_in_prompt: bool,
) -> list[PromptExample]:
    """Load PPO prompts from the raw HIV CSV file using the canonical prompt builder."""

    examples: list[PromptExample] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            parent_smiles = str(row.get("smiles") or "").strip()
            if not parent_smiles:
                continue

            label_candidates = (
                row.get("HIV_active"),
                row.get("activity"),
            )
            original_label = next(
                (
                    normalized
                    for normalized in (normalize_hiv_label(candidate) for candidate in label_candidates)
                    if normalized is not None
                ),
                None,
            )
            if original_label is None:
                original_label = int(default_parent_label)

            if only_positive and original_label != 1:
                continue

            record = MoleculeRecord(
                record_id=index,
                smiles=parent_smiles,
                label=int(original_label),
            )
            prompt = build_counterfactual_prompt(
                record,
                include_label=bool(include_label_in_prompt),
            )
            examples.append(
                PromptExample(
                    index=index,
                    prompt=prompt,
                    parent_smiles=parent_smiles,
                    original_label=int(original_label),
                )
            )
    return examples


def load_prompt_examples(
    dataset_path: Path,
    *,
    default_parent_label: int,
    only_positive: bool,
    include_label_in_prompt: bool,
    max_prompt_examples: int,
) -> list[PromptExample]:
    """Load prompt examples from either JSONL or CSV."""

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix == ".jsonl":
        examples = load_prompt_examples_from_jsonl(
            dataset_path,
            default_parent_label=default_parent_label,
            only_positive=only_positive,
        )
    elif suffix == ".csv":
        examples = load_prompt_examples_from_csv(
            dataset_path,
            default_parent_label=default_parent_label,
            only_positive=only_positive,
            include_label_in_prompt=include_label_in_prompt,
        )
    else:
        raise ValueError(
            "Unsupported dataset format. Please provide either a .jsonl prompt file or a .csv HIV dataset."
        )

    if not examples:
        raise ValueError(f"No usable PPO prompt examples were found in {dataset_path}")

    if max_prompt_examples > 0:
        examples = examples[:max_prompt_examples]
    return examples


def import_training_dependencies() -> dict[str, Any]:
    """Import heavy PPO dependencies lazily so the script stays inspectable."""

    try:
        import torch
        from datasets import Dataset
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            set_seed,
        )
        try:
            from trl import PPOConfig, PPOTrainer
            from trl.experimental.ppo import AutoModelForCausalLMWithValueHead
        except ImportError:
            from trl import PPOConfig, PPOTrainer
            from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "PPO training requires torch, transformers, datasets, peft, and trl."
        ) from exc

    return {
        "torch": torch,
        "Dataset": Dataset,
        "PeftModel": PeftModel,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForCausalLMWithValueHead": AutoModelForCausalLMWithValueHead,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "PPOConfig": PPOConfig,
        "PPOTrainer": PPOTrainer,
        "set_seed": set_seed,
    }


def build_tokenizer(
    deps: dict[str, Any],
    *,
    model_path: Path,
    trust_remote_code: bool,
    local_files_only: bool,
) -> Any:
    """Load the tokenizer with PPO-safe left padding."""

    tokenizer = deps["AutoTokenizer"].from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        use_fast=False,
    )
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer must define eos_token for PPO generation.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_quantized_base_model(
    deps: dict[str, Any],
    *,
    model_path: Path,
    trust_remote_code: bool,
    local_files_only: bool,
) -> Any:
    """Load the 4-bit ChemLLM base model for QLoRA-style PPO."""

    torch = deps["torch"]
    quantization_config = deps["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = deps["AutoModelForCausalLM"].from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = deps["prepare_model_for_kbit_training"](
        model,
        use_gradient_checkpointing=True,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def build_value_head_model(
    deps: dict[str, Any],
    *,
    model_path: Path,
    adapter_path: Path,
    trust_remote_code: bool,
    local_files_only: bool,
    is_trainable: bool,
) -> Any:
    """Load base model + LoRA adapter, then wrap it with a value head."""

    base_model = build_quantized_base_model(
        deps,
        model_path=model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    peft_model = deps["PeftModel"].from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=is_trainable,
    )
    value_head_model = deps["AutoModelForCausalLMWithValueHead"].from_pretrained(peft_model)
    if hasattr(value_head_model, "pretrained_model"):
        pretrained_model = value_head_model.pretrained_model
        if hasattr(pretrained_model, "config"):
            pretrained_model.config.use_cache = False
        if hasattr(pretrained_model, "generation_config") and pretrained_model.generation_config:
            pretrained_model.generation_config.use_cache = False
    if not is_trainable:
        for parameter in value_head_model.parameters():
            parameter.requires_grad = False
        value_head_model.eval()
    return value_head_model


def build_hf_dataset(deps: dict[str, Any], tokenizer: Any, examples: Sequence[PromptExample]) -> Any:
    """Materialize a Hugging Face dataset with one tokenized query column."""

    dataset = deps["Dataset"].from_list([example.to_dataset_row() for example in examples])

    def tokenize_row(row: dict[str, Any]) -> dict[str, Any]:
        encoded = tokenizer(
            row["query"],
            truncation=True,
            padding=False,
        )
        row["input_ids"] = encoded["input_ids"]
        return row

    return dataset.map(tokenize_row)


def build_data_collator() -> Any:
    """Return a minimal collator compatible with classic TRL PPOTrainer."""

    def collate(features: list[dict[str, Any]]) -> dict[str, list[Any]]:
        return {key: [feature[key] for feature in features] for key in features[0]}

    return collate


def normalize_generation_tensors(torch: Any, generated: Any) -> list[Any]:
    """Normalize different TRL generate() return types into a list of tensors."""

    if torch.is_tensor(generated):
        if generated.dim() == 1:
            return [generated]
        return [generated[index] for index in range(generated.shape[0])]
    return list(generated)


def summarize_reward_traces(traces: Sequence[RewardTrace], responses: Sequence[str]) -> dict[str, float]:
    """Aggregate reward traces into one flat logging dictionary."""

    rewards = [float(trace.reward) for trace in traces]
    target_probabilities = [
        float(trace.target_probability)
        for trace in traces
        if trace.target_probability is not None
    ]
    collapse = analyze_batch_collapse(list(responses))

    return {
        "reward_mean": mean(rewards) if rewards else 0.0,
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "valid_rate": mean(1.0 if trace.valid_smiles else 0.0 for trace in traces) if traces else 0.0,
        "connected_rate": (
            mean(1.0 if trace.connected_fragment else 0.0 for trace in traces) if traces else 0.0
        ),
        "subgraph_rate": mean(1.0 if trace.is_subgraph else 0.0 for trace in traces) if traces else 0.0,
        "deletion_rate": mean(1.0 if trace.deletion_success else 0.0 for trace in traces) if traces else 0.0,
        "flip_rate": mean(1.0 if trace.flip_success else 0.0 for trace in traces) if traces else 0.0,
        "target_probability_mean": mean(target_probabilities) if target_probabilities else 0.0,
        "response_length_mean": mean(len(response) for response in responses) if responses else 0.0,
        "collapse_longest_run": float(collapse.longest_run),
        "collapse_dominant_fraction": float(collapse.dominant_character_fraction),
        "collapse_duplicate_fraction": float(collapse.duplicate_output_fraction),
    }


def scalarize_stats(payload: dict[str, Any]) -> dict[str, float | int | str | bool | None]:
    """Best-effort conversion of trainer stats to JSON-friendly scalars."""

    flattened: dict[str, float | int | str | bool | None] = {}
    for key, value in payload.items():
        if isinstance(value, (float, int, str, bool)) or value is None:
            flattened[str(key)] = value
            continue
        if hasattr(value, "item"):
            try:
                flattened[str(key)] = value.item()
                continue
            except Exception:
                pass
        flattened[str(key)] = str(value)
    return flattened


def append_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Append dictionaries to a JSONL file."""

    ensure_directory(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def save_checkpoint(
    trainer: Any,
    tokenizer: Any,
    output_dir: Path,
    *,
    step: int | str,
) -> Path:
    """Persist the PPO policy and tokenizer to one checkpoint directory."""

    checkpoint_dir = output_dir / f"checkpoint-{step}"
    ensure_directory(checkpoint_dir)
    if hasattr(trainer, "save_pretrained"):
        trainer.save_pretrained(str(checkpoint_dir))
    elif hasattr(trainer, "model") and hasattr(trainer.model, "save_pretrained"):
        trainer.model.save_pretrained(str(checkpoint_dir))
    else:  # pragma: no cover - depends on TRL runtime
        raise RuntimeError("Neither PPOTrainer nor its model exposes save_pretrained().")
    tokenizer.save_pretrained(str(checkpoint_dir))
    return checkpoint_dir


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args = apply_config_overrides(args, parser)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    sft_adapter_path = Path(args.sft_adapter_path).expanduser().resolve()
    oracle_path = Path(args.oracle_path).expanduser().resolve()

    ensure_directory(output_dir)
    run_name = f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = output_dir / "logs"
    logger = configure_run_logger(
        "train_ppo",
        context=RunContext(
            run_name=run_name,
            output_dir=output_dir,
            stage="ppo",
            seed=args.seed,
        ),
        log_dir=log_dir,
    )

    write_runtime_manifest(
        output_dir / "train_ppo_manifest.json",
        {
            "run_name": run_name,
            "config_files": [str(Path(path).expanduser().resolve()) for path in args.config],
            "model_path": str(model_path),
            "sft_adapter_path": str(sft_adapter_path),
            "oracle_path": str(oracle_path),
            "dataset_path": str(dataset_path),
            "output_dir": str(output_dir),
            "args": vars(args),
        },
    )

    examples = load_prompt_examples(
        dataset_path,
        default_parent_label=args.default_parent_label,
        only_positive=args.only_positive,
        include_label_in_prompt=args.include_label_in_prompt,
        max_prompt_examples=args.max_prompt_examples,
    )
    logger.info("Loaded %s PPO prompt examples from %s", len(examples), dataset_path)

    deps = import_training_dependencies()
    deps["set_seed"](args.seed)
    torch = deps["torch"]
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = build_tokenizer(
        deps,
        model_path=model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    dataset = build_hf_dataset(deps, tokenizer, examples)

    actual_batch_size = max(1, min(int(args.batch_size), len(dataset)))
    actual_mini_batch_size = max(1, min(int(args.mini_batch_size), actual_batch_size))
    logger.info(
        "Using PPO batch_size=%s mini_batch_size=%s on %s prompt examples",
        actual_batch_size,
        actual_mini_batch_size,
        len(dataset),
    )

    policy_model = build_value_head_model(
        deps,
        model_path=model_path,
        adapter_path=sft_adapter_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        is_trainable=True,
    )
    reference_model = build_value_head_model(
        deps,
        model_path=model_path,
        adapter_path=sft_adapter_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        is_trainable=False,
    )

    ppo_config = deps["PPOConfig"](
        learning_rate=float(args.learning_rate),
        batch_size=int(actual_batch_size),
        mini_batch_size=int(actual_mini_batch_size),
        init_kl_coef=float(args.init_kl_coef),
        seed=int(args.seed),
    )

    ppo_trainer = deps["PPOTrainer"](
        config=ppo_config,
        model=policy_model,
        ref_model=reference_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=build_data_collator(),
    )

    rewarder = ChemRLRewarder(
        oracle_path=oracle_path,
        default_parent_label=args.default_parent_label,
    )

    generation_kwargs = {
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": True,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": False,
    }

    stats_path = output_dir / "ppo_stats.jsonl"
    samples_path = output_dir / "ppo_samples.jsonl"
    global_step = 0

    logger.info("Starting PPO training loop with max_steps=%s", args.max_steps)

    while global_step < int(args.max_steps):
        for batch in ppo_trainer.dataloader:
            global_step += 1
            query_tensors = [
                torch.tensor(ids, dtype=torch.long, device=ppo_trainer.accelerator.device)
                for ids in batch["input_ids"]
            ]
            generated = ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                **generation_kwargs,
            )
            response_tensors = normalize_generation_tensors(torch, generated)
            raw_responses = [
                tokenizer.decode(response_tensor, skip_special_tokens=True).strip()
                for response_tensor in response_tensors
            ]
            cleaned_responses = [clean_generated_smiles(text) for text in raw_responses]
            parent_smiles_batch = [str(value) for value in batch["parent_smiles"]]
            original_label_batch = [int(value) for value in batch["original_label"]]

            traces = rewarder.calculate_reward_details_batch(
                parent_smiles_batch,
                cleaned_responses,
                parent_labels=original_label_batch,
            )
            rewards = [
                torch.tensor(trace.reward, dtype=torch.float32, device=ppo_trainer.accelerator.device)
                for trace in traces
            ]

            trainer_stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            summary = summarize_reward_traces(traces, cleaned_responses)
            merged_stats = {
                "step": global_step,
                **summary,
                **scalarize_stats(trainer_stats),
            }

            if ppo_trainer.accelerator.is_main_process:
                append_jsonl(stats_path, [merged_stats])
                sample_rows: list[dict[str, Any]] = []
                for local_index, trace in enumerate(traces[: min(3, len(traces))]):
                    sample_rows.append(
                        {
                            "step": global_step,
                            "index_in_batch": local_index,
                            "query": batch["query"][local_index],
                            "parent_smiles": parent_smiles_batch[local_index],
                            "original_label": original_label_batch[local_index],
                            "raw_response": raw_responses[local_index],
                            "cleaned_response": cleaned_responses[local_index],
                            "reward_trace": trace.to_dict(),
                        }
                    )
                append_jsonl(samples_path, sample_rows)

            if global_step == 1 or global_step % int(args.logging_steps) == 0:
                logger.info(
                    "step=%s reward_mean=%.4f valid_rate=%.3f subgraph_rate=%.3f flip_rate=%.3f "
                    "target_prob=%.4f dup=%.3f longest_run=%.0f",
                    global_step,
                    summary["reward_mean"],
                    summary["valid_rate"],
                    summary["subgraph_rate"],
                    summary["flip_rate"],
                    summary["target_probability_mean"],
                    summary["collapse_duplicate_fraction"],
                    summary["collapse_longest_run"],
                )

            try:
                ppo_trainer.log_stats(
                    trainer_stats,
                    {
                        "query": batch["query"],
                        "response": raw_responses,
                    },
                    rewards,
                )
            except Exception:
                logger.debug("ppo_trainer.log_stats failed; continuing.", exc_info=True)

            if global_step % int(args.save_steps) == 0 and ppo_trainer.accelerator.is_main_process:
                checkpoint_dir = save_checkpoint(
                    ppo_trainer,
                    tokenizer,
                    output_dir,
                    step=global_step,
                )
                logger.info("Saved PPO checkpoint to %s", checkpoint_dir)

            if global_step >= int(args.max_steps):
                break

    if ppo_trainer.accelerator.is_main_process:
        final_checkpoint = save_checkpoint(
            ppo_trainer,
            tokenizer,
            output_dir,
            step="final",
        )
        logger.info("Training finished. Final checkpoint saved to %s", final_checkpoint)


if __name__ == "__main__":
    main()
