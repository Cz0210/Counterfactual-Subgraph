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
from dataclasses import dataclass
from datetime import datetime
import inspect
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Sequence

# 开启离线模式：只记录不上传
os.environ["WANDB_MODE"] = "offline"
# 确保不会因为寻找 netrc 文件或登录信息而卡住
os.environ["WANDB_SILENT"] = "true"

from src.data.prompts import build_counterfactual_prompt
from src.data.schemas import MoleculeRecord
from src.models import clean_generated_smiles
from src.reward.reward_wrapper import ChemRLRewarder
from src.utils.io import ensure_directory, read_jsonl
from src.utils.logging_utils import RunContext, configure_run_logger, write_runtime_manifest
from src.utils import apply_dotlist_overrides, load_and_merge_config_files


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = REPO_ROOT / "pretrained_models" / "ChemLLM-7B-Chat"
DEFAULT_SFT_ADAPTER = REPO_ROOT / "outputs" / "hpc" / "sft_checkpoints" / "checkpoint-500"
DEFAULT_ORACLE_PATH = REPO_ROOT / "outputs" / "hpc" / "oracle" / "aids_rf_model.pkl"
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "raw" / "AIDS" / "HIV.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "hpc" / "rl_checkpoints"
DEFAULT_WANDB_RUN_NAME = "ppo_aids_rl_v1"

_SMILES_FIELD_PATTERNS = (
    re.compile(r"MOLECULE_SMILES:\s*(?P<smiles>[^\n\r]+)"),
    re.compile(r"PARENT_SMILES:\s*(?P<smiles>[^\n\r]+)"),
)
_LABEL_PATTERN = re.compile(r"ORIGINAL_LABEL:\s*(?P<label>[01])")
_MODEL_WRAPPER_ATTRS = ("pretrained_model", "base_model", "model")
_REWARD_BACKBONE_ATTRS = _MODEL_WRAPPER_ATTRS + ("lm_backbone", "backbone", "language_model")


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
        "--diagnose-reward-flow",
        action="store_true",
        help="输出额外的 reward-flow 调试日志，便于 smoke test 排查。",
    )
    parser.add_argument(
        "--skip-generate-completions",
        action="store_true",
        help="Skip TRL experimental PPO generate_completions(), useful when eval_dataloader is unavailable.",
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
            DataCollatorWithPadding,
            set_seed,
        )
        try:
            from trl.experimental.ppo import PPOConfig, PPOTrainer
        except ImportError:
            from trl import PPOConfig, PPOTrainer
        try:
            import trl.experimental.ppo.ppo_trainer as ppo_trainer_module
        except ImportError:
            ppo_trainer_module = None
        try:
            from trl import AutoModelForCausalLMWithValueHead
        except ImportError:
            from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "PPO training requires torch, transformers, datasets, peft, and trl."
        ) from exc

    # 终极类级别猴子补丁：只要 TRL 敢实例化这个外壳，默认就带这两个空方法，直接规避报错
    if ppo_trainer_module is not None and hasattr(ppo_trainer_module, "PolicyAndValueWrapper"):
        wrapper_cls = ppo_trainer_module.PolicyAndValueWrapper
        if not hasattr(wrapper_cls, "gradient_checkpointing_disable"):
            wrapper_cls.gradient_checkpointing_disable = lambda self: None
        if not hasattr(wrapper_cls, "gradient_checkpointing_enable"):
            wrapper_cls.gradient_checkpointing_enable = lambda self: None

    return {
        "torch": torch,
        "Dataset": Dataset,
        "PeftModel": PeftModel,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForCausalLMWithValueHead": AutoModelForCausalLMWithValueHead,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "DataCollatorWithPadding": DataCollatorWithPadding,
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
    prepare_for_training: bool,
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
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False
    if prepare_for_training and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if prepare_for_training:
        model = deps["prepare_model_for_kbit_training"](
            model,
            use_gradient_checkpointing=True,
        )
    if prepare_for_training and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def build_value_model(
    deps: dict[str, Any],
    *,
    model_path: Path,
    tokenizer: Any,
    trust_remote_code: bool,
    local_files_only: bool,
) -> Any:
    """Load a TRL value-head wrapper compatible with InternLM2-based ChemLLM.

    InternLM2 configs used by ChemLLM are not always registered for the Hugging
    Face sequence-classification auto-model mapping, so the PPO value model
    falls back to TRL's causal-LM value-head wrapper. We then
    monkey-patch ``base_model_prefix`` so newer experimental PPO trainers can
    treat the wrapper like a native transformers model.
    """

    base_model = build_quantized_base_model(
        deps,
        model_path=model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        prepare_for_training=False,
    )
    value_model = deps["AutoModelForCausalLMWithValueHead"].from_pretrained(base_model)

    # Monkey Patch: 欺骗 TRL experimental API，让它以为这是一个原生的分类模型
    if not hasattr(value_model, "base_model_prefix"):
        value_model.base_model_prefix = "pretrained_model"

    pretrained_model = getattr(value_model, "pretrained_model", None)
    if pretrained_model is not None:
        if hasattr(pretrained_model, "config"):
            pretrained_model.config.use_cache = False
            pretrained_model.config.pad_token_id = tokenizer.pad_token_id
        if (
            hasattr(pretrained_model, "generation_config")
            and pretrained_model.generation_config is not None
        ):
            pretrained_model.generation_config.use_cache = False

    if hasattr(value_model, "v_head"):
        value_model.v_head.requires_grad_(True)

    for parameter in value_model.parameters():
        parameter.requires_grad = False

    if hasattr(value_model, "v_head"):
        for parameter in value_model.v_head.parameters():
            parameter.requires_grad = True

    if pretrained_model is not None:
        pretrained_model.eval()
    value_model.train()

    if not any(parameter.requires_grad for parameter in value_model.parameters()):
        raise RuntimeError("Could not identify a trainable TRL value head for PPO.")

    return value_model


def _iter_named_module_candidates(
    model: Any,
    *,
    attr_names: Sequence[str],
) -> Sequence[tuple[str, Any]]:
    """Yield the root object and common wrapper layers for adapter discovery."""

    if model is None:
        return ()

    candidates: list[tuple[str, Any]] = []
    queue: list[tuple[str, Any]] = [("self", model)]
    visited: set[int] = set()

    while queue:
        path, current = queue.pop(0)
        if current is None:
            continue
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        candidates.append((path, current))

        for attr_name in attr_names:
            child = getattr(current, attr_name, None)
            if child is None:
                continue
            child_path = f"{path}.{attr_name}" if path != "self" else attr_name
            queue.append((child_path, child))

    return candidates


def _iter_score_adapter_candidates(model: Any) -> Sequence[tuple[str, Any]]:
    """Yield likely wrapper layers that may own a PPO-compatible value head."""

    return _iter_named_module_candidates(model, attr_names=_MODEL_WRAPPER_ATTRS)


def _resolve_hidden_size_from_model(model: Any) -> int | None:
    """Best-effort hidden-size inference for lightweight PPO score heads."""

    config = getattr(model, "config", None)
    for attr_name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, attr_name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def _infer_module_device_and_dtype(
    module: Any,
    *,
    torch: Any,
) -> tuple[Any, Any]:
    """Infer a stable device / floating dtype pair for a newly created head."""

    device = None
    dtype = None

    for tensor in list(module.parameters()) + list(module.buffers()):
        if device is None:
            device = tensor.device
        tensor_dtype = getattr(tensor, "dtype", None)
        if tensor_dtype is not None and getattr(tensor_dtype, "is_floating_point", False):
            dtype = tensor_dtype
            break

    if device is None:
        try:
            first_parameter = next(module.parameters())
            device = first_parameter.device
        except StopIteration:
            device = torch.device("cpu")

    if dtype is None:
        module_dtype = getattr(module, "dtype", None)
        if isinstance(module_dtype, torch.dtype) and module_dtype.is_floating_point:
            dtype = module_dtype

    if dtype is None:
        config_dtype = getattr(getattr(module, "config", None), "torch_dtype", None)
        if isinstance(config_dtype, torch.dtype) and config_dtype.is_floating_point:
            dtype = config_dtype

    if dtype is None:
        dtype = torch.float32

    return device, dtype


def _find_reward_backbone_on_model(model: Any) -> tuple[str | None, Any, str | None]:
    """Find a usable LM-like backbone exposed by a reward wrapper if present."""

    if model is None:
        return None, None, None

    for attr_name in _REWARD_BACKBONE_ATTRS:
        candidate = getattr(model, attr_name, None)
        if candidate is not None and callable(getattr(candidate, "forward", None)):
            return attr_name, candidate, f"self.{attr_name}"

    for path, candidate in _iter_named_module_candidates(model, attr_names=_REWARD_BACKBONE_ATTRS):
        if path == "self":
            continue
        if candidate is not None and callable(getattr(candidate, "forward", None)):
            return None, candidate, path

    return None, None, None


def _coerce_reward_backbone_candidate(model: Any) -> tuple[Any, str | None]:
    """Normalize a fallback model into one forwardable backbone object."""

    if model is None:
        return None, None

    if callable(getattr(model, "forward", None)):
        return model, f"fallback:{model.__class__.__name__}"

    _, candidate, source = _find_reward_backbone_on_model(model)
    return candidate, source


def ensure_score_head_for_experimental_ppo(model: Any, name: str = "model") -> Any:
    """Attach a local ``.score`` adapter when TRL PPO only finds ``.v_head``.

    Newer ``trl.experimental`` PPO utilities call ``model.score(hidden_states)``
    for critic-style value evaluation. ``AutoModelForCausalLMWithValueHead``
    exposes ``v_head`` instead, so we bridge that interface locally without
    touching site-packages or changing the policy/reference generation path.
    """

    logger = logging.getLogger("train_ppo")

    if model is None:
        logger.warning("%s is None; cannot attach experimental PPO score adapter.", name)
        return model

    if hasattr(model, "score"):
        logger.info("%s already has .score; no adapter needed.", name)
        return model

    for path, candidate in _iter_score_adapter_candidates(model):
        v_head = getattr(candidate, "v_head", None)
        if v_head is None:
            continue

        def _score(hidden_states: Any, *args: Any, _v_head: Any = v_head, **kwargs: Any) -> Any:
            return _v_head(hidden_states)

        setattr(model, "score", _score)
        logger.info(
            "Attached experimental PPO score adapter to %s using %s.v_head (type=%s); hasattr(score)=%s",
            name,
            path,
            candidate.__class__.__name__,
            hasattr(model, "score"),
        )
        return model

    logger.warning(
        "%s has no .score and no reachable .v_head; experimental PPO may fail. type=%s",
        name,
        type(model),
    )
    return model


def ensure_reward_model_for_experimental_ppo(
    reward_model: Any,
    *,
    fallback_lm_model: Any = None,
    deps: dict[str, Any] | None = None,
    name: str = "reward_model",
) -> Any:
    """Adapt custom reward components to TRL experimental PPO's reward-model API.

    ``ChemRewardModelWrapper`` computes chemistry-aware rewards from decoded
    strings, but newer ``trl.experimental`` PPO ``get_reward()`` expects a
    Hugging Face-style reward model with:

    - ``base_model_prefix`` pointing to a forwardable LM backbone;
    - ``getattr(model, model.base_model_prefix)`` returning that backbone;
    - ``score(hidden_states)`` mapping token hidden states to scalar logits.

    When the custom reward component does not already expose that contract, we
    build a lightweight compatibility adapter for smoke-test / interface
    validation only. This adapter is not equivalent to the repository's
    chemistry reward objective.
    """

    logger = logging.getLogger("train_ppo")

    if deps is not None:
        torch = deps["torch"]
    else:  # pragma: no cover - exercised in real training environments
        import torch  # type: ignore

    logger.info("reward_model before adapter: %s", type(reward_model))

    if reward_model is None:
        logger.warning("%s is None; cannot build an experimental PPO reward adapter.", name)
        return reward_model

    base_model_prefix = getattr(reward_model, "base_model_prefix", None)
    existing_backbone = None
    if isinstance(base_model_prefix, str) and base_model_prefix:
        existing_backbone = getattr(reward_model, base_model_prefix, None)

    if existing_backbone is not None and hasattr(reward_model, "score"):
        logger.info("%s already exposes base_model_prefix=%s and .score; no adapter needed.", name, base_model_prefix)
        logger.info("reward_model after adapter: %s", type(reward_model))
        logger.info("reward_model has base_model_prefix: %s", hasattr(reward_model, "base_model_prefix"))
        logger.info("reward_model base_model_prefix: %s", getattr(reward_model, "base_model_prefix", None))
        logger.info("reward_model has score: %s", hasattr(reward_model, "score"))
        return reward_model

    direct_attr_name, backbone, backbone_source = _find_reward_backbone_on_model(reward_model)
    if backbone is not None:
        exposed_attr_name = direct_attr_name or "pretrained_model"
        if getattr(reward_model, exposed_attr_name, None) is not backbone:
            setattr(reward_model, exposed_attr_name, backbone)
        reward_model.base_model_prefix = exposed_attr_name
        reward_model = ensure_score_head_for_experimental_ppo(reward_model, name=name)

        if not hasattr(reward_model, "score"):
            hidden_size = _resolve_hidden_size_from_model(backbone)
            if hidden_size is None:
                raise ValueError(
                    f"Cannot infer hidden_size for {name} score head from backbone source {backbone_source}."
                )
            device, dtype = _infer_module_device_and_dtype(backbone, torch=torch)
            score_head = torch.nn.Linear(hidden_size, 1, bias=False)
            score_head.to(device=device, dtype=dtype)
            with torch.no_grad():
                score_head.weight.zero_()
            for parameter in score_head.parameters():
                parameter.requires_grad = False
            setattr(reward_model, "score", score_head)
            logger.info(
                "Attached linear score head to %s using backbone source %s; hidden_size=%s device=%s dtype=%s",
                name,
                backbone_source,
                hidden_size,
                device,
                dtype,
            )

        logger.info("reward_model after adapter: %s", type(reward_model))
        logger.info("reward_model has base_model_prefix: %s", hasattr(reward_model, "base_model_prefix"))
        logger.info("reward_model base_model_prefix: %s", getattr(reward_model, "base_model_prefix", None))
        logger.info("reward_model has score: %s", hasattr(reward_model, "score"))
        return reward_model

    fallback_backbone, fallback_source = _coerce_reward_backbone_candidate(fallback_lm_model)
    if fallback_backbone is None:
        logger.warning(
            "%s does not expose a TRL-compatible backbone and no fallback LM backbone was available. type=%s",
            name,
            type(reward_model),
        )
        logger.info("reward_model after adapter: %s", type(reward_model))
        logger.info("reward_model has base_model_prefix: %s", hasattr(reward_model, "base_model_prefix"))
        logger.info("reward_model base_model_prefix: %s", getattr(reward_model, "base_model_prefix", None))
        logger.info("reward_model has score: %s", hasattr(reward_model, "score"))
        return reward_model

    hidden_size = _resolve_hidden_size_from_model(fallback_backbone)
    if hidden_size is None:
        raise ValueError(
            f"Cannot infer hidden_size for {name} fallback reward adapter from {fallback_source}."
        )
    device, dtype = _infer_module_device_and_dtype(fallback_backbone, torch=torch)

    class ExperimentalPPORewardModelAdapter(torch.nn.Module):
        base_model_prefix = "pretrained_model"

        def __init__(self, pretrained_model: Any, chemistry_reward_component: Any) -> None:
            super().__init__()
            self.pretrained_model = pretrained_model
            self.score = torch.nn.Linear(hidden_size, 1, bias=False)
            self.score.to(device=device, dtype=dtype)
            with torch.no_grad():
                self.score.weight.zero_()
            for parameter in self.score.parameters():
                parameter.requires_grad = False
            self.interface_compatibility_only = True
            self.adapter_note = (
                "Using experimental PPO-compatible reward adapter for smoke test / interface compatibility. "
                "ChemRewardModelWrapper remains the chemistry reward component and is not equivalent to TRL hidden-state reward head."
            )
            object.__setattr__(self, "chemistry_reward_component", chemistry_reward_component)

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("output_hidden_states", True)
            kwargs.setdefault("return_dict", True)
            return self.pretrained_model(*args, **kwargs)

    adapted_reward_model = ExperimentalPPORewardModelAdapter(
        pretrained_model=fallback_backbone,
        chemistry_reward_component=reward_model,
    )
    logger.info(
        "Using experimental PPO-compatible reward adapter for smoke test / interface compatibility. "
        "ChemRewardModelWrapper remains the chemistry reward component and is not equivalent to TRL hidden-state reward head."
    )
    logger.info(
        "Attached fallback reward adapter to %s using %s; hidden_size=%s device=%s dtype=%s",
        name,
        fallback_source,
        hidden_size,
        device,
        dtype,
    )
    logger.info("reward_model after adapter: %s", type(adapted_reward_model))
    logger.info("reward_model has base_model_prefix: %s", hasattr(adapted_reward_model, "base_model_prefix"))
    logger.info("reward_model base_model_prefix: %s", getattr(adapted_reward_model, "base_model_prefix", None))
    logger.info("reward_model has score: %s", hasattr(adapted_reward_model, "score"))
    return adapted_reward_model


def build_policy_model(
    deps: dict[str, Any],
    *,
    model_path: Path,
    adapter_path: Path,
    trust_remote_code: bool,
    local_files_only: bool,
    is_trainable: bool,
) -> Any:
    """Load base model + LoRA adapter as a native causal LM.

    Newer ``trl.experimental.ppo.PPOTrainer`` versions manage the policy/value
    wrapping internally, so we intentionally keep the external model as a
    plain causal LM (potentially PEFT-wrapped) instead of constructing an
    explicit value-head wrapper here.
    """

    base_model = build_quantized_base_model(
        deps,
        model_path=model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        prepare_for_training=is_trainable,
    )
    policy_model = deps["PeftModel"].from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=is_trainable,
    )
    if hasattr(policy_model, "config"):
        policy_model.config.use_cache = False
    if hasattr(policy_model, "generation_config") and policy_model.generation_config:
        policy_model.generation_config.use_cache = False
    if not is_trainable:
        for parameter in policy_model.parameters():
            parameter.requires_grad = False
        policy_model.eval()
    return policy_model


def _decode_text_batch(tokenizer: Any, payload: Any, *, torch: Any) -> list[str]:
    """Decode one batch-like payload into plain strings."""

    if payload is None:
        return []
    if isinstance(payload, str):
        return [payload]
    if torch.is_tensor(payload):
        if payload.dim() == 1:
            payload = payload.unsqueeze(0)
        return [
            str(text).strip()
            for text in tokenizer.batch_decode(payload.detach().cpu().tolist(), skip_special_tokens=True)
        ]
    if isinstance(payload, dict):
        for key in ("query", "prompt", "response", "completion", "text"):
            if key in payload:
                return _decode_text_batch(tokenizer, payload[key], torch=torch)
        return [str(payload)]

    values = list(payload) if isinstance(payload, Sequence) else [payload]
    if not values:
        return []
    if all(isinstance(value, str) for value in values):
        return [str(value).strip() for value in values]
    if all(torch.is_tensor(value) for value in values):
        batch = []
        for value in values:
            tensor = value.detach().cpu()
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)
            batch.append(tensor.tolist())
        return [str(text).strip() for text in tokenizer.batch_decode(batch, skip_special_tokens=True)]
    if all(isinstance(value, (list, tuple)) for value in values):
        return [str(text).strip() for text in tokenizer.batch_decode(values, skip_special_tokens=True)]
    return [str(value).strip() for value in values]


class _KeywordLookup:
    """Small helper to retrieve the first present non-null kwarg."""

    @staticmethod
    def first_present(mapping: dict[str, Any], keys: Sequence[str]) -> Any:
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
        return None


def _extract_fragment_from_text(combined_text: str, prompt_text: str | None) -> str:
    """Recover the generated fragment from a decoded prompt+completion string."""

    normalized_combined = str(combined_text or "").strip()
    normalized_prompt = str(prompt_text or "").strip()

    if normalized_prompt and normalized_combined.startswith(normalized_prompt):
        candidate = normalized_combined[len(normalized_prompt) :].strip()
        if candidate:
            cleaned = clean_generated_smiles(candidate)
            if cleaned:
                return cleaned

    if "FRAGMENT_SMILES:" in normalized_combined:
        _, _, suffix = normalized_combined.partition("FRAGMENT_SMILES:")
        cleaned = clean_generated_smiles(suffix)
        if cleaned:
            return cleaned

    return clean_generated_smiles(normalized_combined)


def build_reward_model_wrapper(
    deps: dict[str, Any],
    *,
    tokenizer: Any,
    rewarder: ChemRLRewarder,
) -> Any:
    """Create the repository's chemistry reward component.

    This wrapper computes deletion-based chemistry rewards from decoded text.
    It is *not* the same contract as the hidden-state reward model expected by
    newer ``trl.experimental`` PPO ``get_reward()`` paths, so the trainer-facing
    reward model may still need a separate interface adapter.
    """

    torch = deps["torch"]

    class ChemRewardModelWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tokenizer = tokenizer
            self.rewarder = rewarder
            self.reward_floor = float(rewarder.minimum_reward)

        def _recover_prompt_texts(
            self,
            kwargs: dict[str, Any],
            *,
            batch_size: int,
        ) -> list[str | None]:
            prompt_payload = _KeywordLookup.first_present(
                kwargs,
                ("prompts", "queries", "query", "input_texts", "texts"),
            )
            if prompt_payload is None:
                return [None] * batch_size

            prompt_texts = _decode_text_batch(self.tokenizer, prompt_payload, torch=torch)
            if not prompt_texts:
                return [None] * batch_size
            if len(prompt_texts) < batch_size:
                prompt_texts = prompt_texts + [prompt_texts[-1]] * (batch_size - len(prompt_texts))
            return prompt_texts[:batch_size]

        def forward(self, input_ids: Any, attention_mask: Any = None, **kwargs: Any) -> Any:
            decoded_texts = _decode_text_batch(self.tokenizer, input_ids, torch=torch)
            if not decoded_texts:
                if torch.is_tensor(input_ids):
                    return torch.zeros((0, 1), dtype=torch.float32, device=input_ids.device)
                return torch.zeros((0, 1), dtype=torch.float32)

            prompt_texts = self._recover_prompt_texts(kwargs, batch_size=len(decoded_texts))
            parent_smiles_batch: list[str] = []
            parent_labels: list[int] = []
            generated_smiles_batch: list[str] = []

            for decoded_text, prompt_text in zip(decoded_texts, prompt_texts, strict=True):
                context_text = prompt_text or decoded_text
                try:
                    parent_smiles = extract_parent_smiles_from_prompt(context_text)
                except Exception:
                    try:
                        parent_smiles = extract_parent_smiles_from_prompt(decoded_text)
                    except Exception:
                        parent_smiles = ""

                label = extract_label_from_prompt(context_text)
                if label is None:
                    label = extract_label_from_prompt(decoded_text)
                if label is None:
                    label = self.rewarder.default_parent_label

                parent_smiles_batch.append(parent_smiles)
                parent_labels.append(int(label))
                generated_smiles_batch.append(
                    _extract_fragment_from_text(decoded_text, prompt_text)
                )

            rewards = self.rewarder.calculate_rewards_with_labels(
                parent_smiles_batch,
                generated_smiles_batch,
                parent_labels,
            )

            device = input_ids.device if torch.is_tensor(input_ids) else None
            reward_tensor = torch.tensor(
                [float(score) for score in rewards],
                dtype=torch.float32,
                device=device,
            ).view(-1, 1)
            return reward_tensor

    return ChemRewardModelWrapper()


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

    dataset = dataset.map(tokenize_row)
    if "input_ids" not in dataset.column_names:
        raise RuntimeError("Tokenized PPO dataset is missing required 'input_ids' column.")
    # 将数据集的底层格式转为 PyTorch Tensor，同时保留 query / smiles 等原始列。
    dataset.set_format(
        type="torch",
        columns=["input_ids"],
        output_all_columns=True,
    )
    return dataset


def build_data_collator(deps: dict[str, Any], tokenizer: Any) -> Any:
    """Return a tensor-producing collator compatible with experimental PPOTrainer.

    We use Hugging Face's standard ``DataCollatorWithPadding`` for the model
    inputs, then re-attach non-tensor metadata fields so reward reconstruction
    still has access to the original prompt strings.
    """

    import torch

    standard_collator = deps["DataCollatorWithPadding"](
        tokenizer=tokenizer,
        return_tensors="pt",
    )

    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("PPO data_collator received an empty features list.")

        batch: dict[str, Any] = {}

        if "input_ids" not in features[0]:
            raise KeyError("PPO data_collator expected every feature to contain 'input_ids'.")

        token_features = []
        for feature in features:
            input_ids = feature["input_ids"]
            if torch.is_tensor(input_ids):
                input_ids = input_ids.detach().cpu().tolist()
            token_features.append({"input_ids": input_ids})
        batch.update(standard_collator(token_features))

        for key in features[0]:
            if key in {"input_ids", "attention_mask"}:
                continue
            values = [feature[key] for feature in features]
            if all(torch.is_tensor(value) for value in values):
                try:
                    batch[key] = torch.stack(values)
                    continue
                except Exception:
                    pass
            batch[key] = values

        if "input_ids" not in batch or not torch.is_tensor(batch["input_ids"]):
            raise RuntimeError("PPO data_collator failed to return tensor 'input_ids'.")
        return batch

    return collate


def build_ppo_config(
    deps: dict[str, Any],
    *,
    args: argparse.Namespace,
    actual_batch_size: int,
    actual_mini_batch_size: int,
    logger: Any,
) -> Any:
    """Build PPOConfig defensively across TRL versions.

    Different TRL releases expose different PPOConfig signatures. Instead of
    hardcoding one version's arguments, inspect the runtime signature and keep
    only the kwargs that are actually supported.
    """

    ppo_config_cls = deps["PPOConfig"]
    try:
        supported_keys = set(inspect.signature(ppo_config_cls).parameters.keys())
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        supported_keys = set()

    desired_kwargs: dict[str, Any] = {
        "batch_size": int(actual_batch_size),
        "mini_batch_size": int(actual_mini_batch_size),
        "gradient_checkpointing": False,
        "learning_rate": float(args.learning_rate),
        "max_steps": int(args.max_steps),
        "report_to": "wandb",
        "run_name": DEFAULT_WANDB_RUN_NAME,
        "seed": int(args.seed),
    }

    generation_kwargs = {
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": True,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "use_cache": False,
    }

    if "init_kl_coef" in supported_keys:
        desired_kwargs["init_kl_coef"] = float(args.init_kl_coef)
    elif "kl_coef" in supported_keys:
        desired_kwargs["kl_coef"] = float(args.init_kl_coef)

    if "log_with" in supported_keys:
        desired_kwargs["log_with"] = "wandb"

    if "generate_kwargs" in supported_keys:
        desired_kwargs["generate_kwargs"] = generation_kwargs
    elif "generation_kwargs" in supported_keys:
        desired_kwargs["generation_kwargs"] = generation_kwargs
    elif "response_length" in supported_keys:
        desired_kwargs["response_length"] = int(args.max_new_tokens)
    elif "max_new_tokens" in supported_keys:
        desired_kwargs["max_new_tokens"] = int(args.max_new_tokens)

    filtered_kwargs = {
        key: value
        for key, value in desired_kwargs.items()
        if not supported_keys or key in supported_keys
    }
    logger.info("Filtered PPOConfig kwargs for current TRL version: %s", sorted(filtered_kwargs))
    return ppo_config_cls(**filtered_kwargs)


def build_ppo_trainer(
    deps: dict[str, Any],
    *,
    ppo_config: Any,
    policy_model: Any,
    reference_model: Any,
    value_model: Any,
    tokenizer: Any,
    dataset: Any,
    reward_model: Any,
    logger: Any,
) -> Any:
    """Build PPOTrainer defensively across TRL API variants."""
    trainer_cls = deps["PPOTrainer"]
    trainer_signature = inspect.signature(trainer_cls.__init__)
    supported_keys = set(trainer_signature.parameters.keys())

    trainer_kwargs: dict[str, Any] = {}

    if "args" in supported_keys:
        trainer_kwargs["args"] = ppo_config
    elif "config" in supported_keys:
        trainer_kwargs["config"] = ppo_config

    if "model" in supported_keys:
        trainer_kwargs["model"] = policy_model
    elif "policy" in supported_keys:
        trainer_kwargs["policy"] = policy_model
    elif "policy_model" in supported_keys:
        trainer_kwargs["policy_model"] = policy_model

    if "ref_model" in supported_keys:
        trainer_kwargs["ref_model"] = reference_model
    elif "ref_policy" in supported_keys:
        trainer_kwargs["ref_policy"] = reference_model
    elif "reference_model" in supported_keys:
        trainer_kwargs["reference_model"] = reference_model

    if "value_model" in supported_keys:
        trainer_kwargs["value_model"] = value_model
    elif "critic_model" in supported_keys:
        trainer_kwargs["critic_model"] = value_model

    if "processing_class" in supported_keys:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in supported_keys:
        trainer_kwargs["tokenizer"] = tokenizer

    if "train_dataset" in supported_keys:
        trainer_kwargs["train_dataset"] = dataset
    elif "dataset" in supported_keys:
        trainer_kwargs["dataset"] = dataset

    if "dataset_text_field" in supported_keys:
        trainer_kwargs["dataset_text_field"] = "query"

    if "data_collator" in supported_keys:
        trainer_kwargs["data_collator"] = build_data_collator(deps, tokenizer)

    if "reward_model" in supported_keys:
        trainer_kwargs["reward_model"] = reward_model
    elif "reward_funcs" in supported_keys:
        trainer_kwargs["reward_funcs"] = [reward_model]

    logger.info("Filtered PPOTrainer kwargs for current TRL version: %s", sorted(trainer_kwargs))
    return trainer_cls(**trainer_kwargs)


def diagnose_eval_dataloader_for_generate_completions(ppo_trainer: Any) -> str | None:
    """Return a concrete reason when trainer-side completions are unsafe to run."""

    eval_dataloader = getattr(ppo_trainer, "eval_dataloader", None)
    if eval_dataloader is None:
        return "ppo_trainer.eval_dataloader is None"

    eval_dataset = getattr(eval_dataloader, "dataset", None)
    if eval_dataset is None:
        return "ppo_trainer.eval_dataloader.dataset is None"

    try:
        len(eval_dataset)
    except Exception as exc:
        return f"len(ppo_trainer.eval_dataloader.dataset) failed: {exc}"

    sampler = getattr(eval_dataloader, "sampler", None)
    if sampler is None:
        return "ppo_trainer.eval_dataloader.sampler is None"

    data_source = getattr(sampler, "data_source", None)
    if data_source is None:
        return "ppo_trainer.eval_dataloader.sampler.data_source is None"

    try:
        len(data_source)
    except Exception as exc:
        return f"len(ppo_trainer.eval_dataloader.sampler.data_source) failed: {exc}"

    return None


def disable_generate_completions_if_needed(
    ppo_trainer: Any,
    logger: Any,
    *,
    reason: str = "",
) -> None:
    """Replace trainer-side completion preview generation with a no-op logger."""

    import types

    original = getattr(ppo_trainer, "generate_completions", None)

    def _skip_generate_completions(self: Any, *args: Any, **kwargs: Any) -> None:
        logger.warning(
            "[PPO_GENERATE_COMPLETIONS_SKIPPED] Skipping TRL generate_completions because eval_dataloader is unavailable. reason=%s",
            reason,
        )
        return None

    ppo_trainer.generate_completions = types.MethodType(_skip_generate_completions, ppo_trainer)
    logger.warning(
        "[PPO_GENERATE_COMPLETIONS_SKIPPED] Patched ppo_trainer.generate_completions. original=%s reason=%s",
        original,
        reason,
    )


def sync_generation_token_ids(
    model: Any,
    tokenizer: Any,
    *,
    visited: set[int] | None = None,
) -> None:
    """Synchronize generation token ids across wrapped PPO models."""

    if model is None:
        return

    if visited is None:
        visited = set()

    model_identity = id(model)
    if model_identity in visited:
        return
    visited.add(model_identity)

    config = getattr(model, "config", None)
    if config is not None:
        if hasattr(config, "pad_token_id"):
            config.pad_token_id = tokenizer.pad_token_id
        if hasattr(config, "eos_token_id"):
            config.eos_token_id = tokenizer.eos_token_id

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        if hasattr(generation_config, "pad_token_id"):
            generation_config.pad_token_id = tokenizer.pad_token_id
        if hasattr(generation_config, "eos_token_id"):
            generation_config.eos_token_id = tokenizer.eos_token_id

    for attribute_name in (
        "policy_model",
        "pretrained_model",
        "model",
        "base_model",
        "language_model",
    ):
        nested_model = getattr(model, attribute_name, None)
        if nested_model is not None:
            sync_generation_token_ids(
                nested_model,
                tokenizer,
                visited=visited,
            )


def patch_internlm_cache(trainer_model: Any) -> None:
    """Patch the deepest InternLM2 generation class to ignore cache inputs.

    Experimental TRL and newer transformers can pass DynamicCache-structured
    values that InternLM2's custom ``prepare_inputs_for_generation`` does not
    understand. We therefore walk through common wrapper layers, find the
    underlying causal LM class, and override its preparation method so every
    generation call behaves like cache is fully disabled.
    """

    import torch

    inner_model = trainer_model
    for _ in range(5):
        if hasattr(inner_model, "policy_model"):
            inner_model = inner_model.policy_model
        elif hasattr(inner_model, "base_model"):
            inner_model = inner_model.base_model
        elif hasattr(inner_model, "model") and not isinstance(inner_model.model, torch.nn.ModuleDict):
            inner_model = inner_model.model
        else:
            break

    model_cls = inner_model.__class__
    if getattr(model_cls, "_is_cache_patched", False):
        return

    original_prepare = getattr(model_cls, "prepare_inputs_for_generation", None)
    if original_prepare is None:
        return

    def patched_prepare(
        self: Any,
        input_ids: Any,
        past_key_values: Any = None,
        attention_mask: Any = None,
        inputs_embeds: Any = None,
        **kwargs: Any,
    ) -> Any:
        # 核心绝杀：无论 TRL 怎么传，强制把 past_key_values 设为 None
        # 这会彻底切断出 Bug 的代码分支，让模型使用全量计算（等效于关闭 cache）
        kwargs["use_cache"] = False
        return original_prepare(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    model_cls.prepare_inputs_for_generation = patched_prepare
    model_cls._is_cache_patched = True


def _safe_source_path(target: Any) -> str | None:
    """Best-effort source path lookup for runtime module introspection."""

    if target is None:
        return None
    try:
        source_path = inspect.getsourcefile(target) or inspect.getfile(target)
    except (OSError, TypeError):
        return None
    return str(source_path) if source_path else None


def _safe_git_commit() -> str | None:
    """Return the current repository commit if git metadata is available."""

    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def collect_runtime_environment_debug() -> dict[str, Any]:
    """Collect basic runtime environment fields for Slurm log diagnosis."""

    return {
        "python_executable": sys.executable,
        "cwd": str(Path.cwd()),
        "git_commit": _safe_git_commit(),
        "hf_home": os.environ.get("HF_HOME"),
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE"),
        "huggingface_hub_cache": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
    }


def collect_runtime_model_debug(model: Any) -> dict[str, Any]:
    """Collect the actual runtime module/file chain for wrapped PPO models."""

    debug_info: dict[str, Any] = {
        **collect_runtime_environment_debug(),
        "layers": [],
    }

    current_model = model
    visited: set[int] = set()
    for depth in range(8):
        if current_model is None:
            break
        identity = id(current_model)
        if identity in visited:
            break
        visited.add(identity)

        class_module_name = getattr(current_model.__class__, "__module__", None)
        class_module = sys.modules.get(class_module_name)
        prepare_method = getattr(current_model.__class__, "prepare_inputs_for_generation", None)
        layer_info: dict[str, Any] = {
            "depth": depth,
            "class_name": current_model.__class__.__name__,
            "class_module": class_module_name,
            "class_source_file": _safe_source_path(current_model.__class__),
            "module_file": getattr(class_module, "__file__", None),
            "prepare_inputs_source_file": _safe_source_path(prepare_method),
        }
        debug_info["layers"].append(layer_info)

        next_model = None
        for attribute_name in (
            "policy_model",
            "pretrained_model",
            "model",
            "base_model",
            "lm_backbone",
            "backbone",
            "language_model",
        ):
            candidate = getattr(current_model, attribute_name, None)
            if candidate is not None and id(candidate) not in visited:
                layer_info["next_attr"] = attribute_name
                next_model = candidate
                break
        current_model = next_model

    return debug_info


def log_runtime_model_debug(
    logger: Any,
    *,
    label: str,
    model: Any,
) -> None:
    """Emit runtime import-path diagnostics into the training log."""

    logger.info("Runtime model debug [%s]: %s", label, collect_runtime_model_debug(model))


def save_final_model(
    trainer: Any,
    tokenizer: Any,
    output_dir: Path,
) -> Path:
    """Persist the final PPO artifacts after trainer-managed training."""

    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None and not bool(getattr(accelerator, "is_main_process", True)):
        return output_dir

    ensure_directory(output_dir)
    save_model = getattr(trainer, "save_model", None)
    if callable(save_model):
        save_model(str(output_dir))
    elif hasattr(trainer, "save_pretrained"):
        trainer.save_pretrained(str(output_dir))
    elif hasattr(trainer, "model") and hasattr(trainer.model, "save_pretrained"):
        trainer.model.save_pretrained(str(output_dir))
    else:  # pragma: no cover - depends on TRL runtime
        raise RuntimeError("Neither PPOTrainer nor its model exposes a save method.")
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


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
    run_name = f"{DEFAULT_WANDB_RUN_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    logger.info("Runtime environment: %s", collect_runtime_environment_debug())

    write_runtime_manifest(
        output_dir / "train_ppo_manifest.json",
        {
            "run_name": run_name,
            "git_commit": _safe_git_commit(),
            "config_files": [str(Path(path).expanduser().resolve()) for path in args.config],
            "model_path": str(model_path),
            "sft_adapter_path": str(sft_adapter_path),
            "oracle_path": str(oracle_path),
            "dataset_path": str(dataset_path),
            "output_dir": str(output_dir),
            "wandb_run_name": DEFAULT_WANDB_RUN_NAME,
            "wandb_mode": os.environ.get("WANDB_MODE"),
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

    policy_model = build_policy_model(
        deps,
        model_path=model_path,
        adapter_path=sft_adapter_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        is_trainable=True,
    )
    reference_model = build_policy_model(
        deps,
        model_path=model_path,
        adapter_path=sft_adapter_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        is_trainable=False,
    )
    value_model = build_value_model(
        deps,
        model_path=model_path,
        tokenizer=tokenizer,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    value_model = ensure_score_head_for_experimental_ppo(value_model, "value_model")
    logger.info("value_model has v_head: %s", hasattr(value_model, "v_head"))
    logger.info("value_model has score after adapter: %s", hasattr(value_model, "score"))

    ppo_config = build_ppo_config(
        deps,
        args=args,
        actual_batch_size=int(actual_batch_size),
        actual_mini_batch_size=int(actual_mini_batch_size),
        logger=logger,
    )

    rewarder = ChemRLRewarder(
        oracle_path=oracle_path,
        default_parent_label=args.default_parent_label,
    )
    chem_reward_model = build_reward_model_wrapper(
        deps,
        tokenizer=tokenizer,
        rewarder=rewarder,
    )
    logger.info("chem_reward_model component type: %s", type(chem_reward_model))
    reward_model = ensure_reward_model_for_experimental_ppo(
        chem_reward_model,
        fallback_lm_model=(
            getattr(value_model, "pretrained_model", None)
            or getattr(reference_model, "pretrained_model", None)
            or reference_model
            or policy_model
        ),
        deps=deps,
        name="reward_model",
    )
    if args.diagnose_reward_flow:
        logger.info(
            "Reward-flow diagnostics enabled: chem_reward_model=%s trainer_reward_model=%s",
            type(chem_reward_model),
            type(reward_model),
        )
        logger.info(
            "Reward-flow diagnostics: trainer reward_model base_model_prefix=%s has_score=%s interface_compatibility_only=%s",
            getattr(reward_model, "base_model_prefix", None),
            hasattr(reward_model, "score"),
            getattr(reward_model, "interface_compatibility_only", False),
        )

    ppo_trainer = build_ppo_trainer(
        deps,
        ppo_config=ppo_config,
        policy_model=policy_model,
        reference_model=reference_model,
        value_model=value_model,
        tokenizer=tokenizer,
        dataset=dataset,
        reward_model=reward_model,
        logger=logger,
    )
    log_runtime_model_debug(logger, label="policy_model", model=policy_model)
    log_runtime_model_debug(logger, label="reference_model", model=reference_model)
    log_runtime_model_debug(logger, label="value_model", model=value_model)
    log_runtime_model_debug(logger, label="reward_model", model=reward_model)
    log_runtime_model_debug(logger, label="ppo_trainer.model_before_patch", model=getattr(ppo_trainer, "model", None))
    sync_generation_token_ids(getattr(ppo_trainer, "model", None), tokenizer)
    patch_internlm_cache(ppo_trainer.model)
    log_runtime_model_debug(logger, label="ppo_trainer.model_after_patch", model=getattr(ppo_trainer, "model", None))

    generate_completions_skip_reason = None
    if args.skip_generate_completions:
        generate_completions_skip_reason = "skip flag enabled"
    else:
        generate_completions_skip_reason = diagnose_eval_dataloader_for_generate_completions(ppo_trainer)
    if generate_completions_skip_reason is not None:
        disable_generate_completions_if_needed(
            ppo_trainer,
            logger,
            reason=generate_completions_skip_reason,
        )

    logger.info("Starting PPO training loop with max_steps=%s", args.max_steps)
    ppo_trainer.train()
    final_output_dir = save_final_model(
        ppo_trainer,
        tokenizer,
        output_dir,
    )
    logger.info("Training finished. Final checkpoint saved to %s", final_output_dir)


if __name__ == "__main__":
    main()
