"""Fresh-LoRA initialization and audit helpers for Mutagenicity SFT."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


FRESH_INITIALIZATION_SCHEMA = "mutagenicity_fresh_lora_v1"
DATASET_VARIANTS = {
    "current_v1": (
        "mutagenicity_sft_train.csv",
        "mutagenicity_sft_val.csv",
    ),
    "strict_v2": (
        "mutagenicity_sft_train_strict_v2.csv",
        "mutagenicity_sft_val_strict_v2.csv",
    ),
    "fallback_v2": (
        "mutagenicity_sft_train_fallback_v2.csv",
        "mutagenicity_sft_val_fallback_v2.csv",
    ),
    "strict_multitarget_v2": (
        "mutagenicity_sft_train_strict_multitarget_v2.csv",
        "mutagenicity_sft_val_strict_v2.csv",
    ),
}


@dataclass(frozen=True, slots=True)
class FreshLoRAConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("wqkv", "wo", "w1", "w2", "w3")
    adapter_name: str = "default"

    def validate(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not self.target_modules:
            raise ValueError("At least one LoRA target module is required")
        if self.adapter_name != "default":
            raise ValueError("Fresh Mutagenicity SFT currently requires adapter_name=default")


def resolve_variant_csvs(
    data_root: str | Path,
    variant: str,
) -> tuple[Path, Path]:
    """Resolve the fixed train/validation files for one dataset variant."""

    if variant not in DATASET_VARIANTS:
        raise ValueError(
            f"Unsupported dataset variant {variant!r}; "
            f"expected one of {sorted(DATASET_VARIANTS)}"
        )
    root = Path(data_root).expanduser().resolve()
    train_name, val_name = DATASET_VARIANTS[variant]
    return root / train_name, root / val_name


def assert_pure_base_model(model: Any) -> dict[str, Any]:
    """Reject a base model that already contains PEFT/LoRA state."""

    peft_config = getattr(model, "peft_config", None)
    hf_peft_loaded = bool(getattr(model, "_hf_peft_config_loaded", False))
    lora_names = [
        str(name)
        for name, _parameter in model.named_parameters()
        if "lora_" in str(name).lower()
    ]
    if peft_config or hf_peft_loaded or lora_names:
        configured = (
            sorted(str(value) for value in peft_config)
            if isinstance(peft_config, Mapping)
            else []
        )
        raise ValueError(
            "Fresh SFT requires a pure ChemLLM base without adapters: "
            f"configured_adapters={configured} "
            f"hf_peft_loaded={hf_peft_loaded} "
            f"lora_parameter_examples={lora_names[:10]}"
        )
    if hasattr(model, "peft_config") and not peft_config:
        try:
            delattr(model, "peft_config")
        except AttributeError as exc:
            raise ValueError("Could not remove an empty base-model peft_config marker") from exc
    return {
        "pure_base_model_verified": True,
        "preexisting_adapter_names": [],
        "preexisting_lora_parameter_count": 0,
    }


def initialize_fresh_lora(
    base_model: Any,
    *,
    lora_config: Any,
    get_peft_model_fn: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Wrap one pure base exactly once with a randomly initialized LoRA."""

    if get_peft_model_fn is None:
        try:
            from peft import get_peft_model
        except ImportError as exc:  # pragma: no cover - HPC dependency
            raise RuntimeError("Fresh Mutagenicity SFT requires peft") from exc
        get_peft_model_fn = get_peft_model

    base_audit = assert_pure_base_model(base_model)
    model = get_peft_model_fn(base_model, lora_config)
    return model, {
        "loading_route": "pure_base_then_get_peft_model_once",
        "get_peft_model_call_count": 1,
        "source_adapter_checkpoint": None,
        "aids_adapter_weights_loaded": False,
        **base_audit,
    }


def _active_adapter_names(model: Any) -> list[str]:
    active = getattr(model, "active_adapters", None)
    if callable(active):
        active = active()
    if active is None:
        active = getattr(model, "active_adapter", None)
        if callable(active):
            active = active()
    if active is None:
        return []
    if isinstance(active, str):
        return [active]
    return [str(value) for value in active]


def audit_fresh_lora_model(
    model: Any,
    *,
    base_model_path: str | Path,
    loading_audit: Mapping[str, Any],
    lora_settings: FreshLoRAConfig,
) -> dict[str, Any]:
    """Prove that exactly one fresh LoRA is active and the base is frozen."""

    peft_config = getattr(model, "peft_config", None)
    if not isinstance(peft_config, Mapping):
        raise ValueError("Fresh SFT model has no mapping-valued peft_config")
    adapter_names = [str(value) for value in peft_config]
    active_adapters = _active_adapter_names(model)
    if adapter_names != [lora_settings.adapter_name]:
        raise ValueError(
            "Fresh SFT requires exactly the default adapter; "
            f"configured={adapter_names}"
        )
    if active_adapters != [lora_settings.adapter_name]:
        raise ValueError(
            "Fresh SFT requires exactly one active adapter; "
            f"active={active_adapters}"
        )

    named_parameters = list(model.named_parameters())
    total_count = sum(int(parameter.numel()) for _, parameter in named_parameters)
    trainable = [
        (str(name), parameter)
        for name, parameter in named_parameters
        if bool(parameter.requires_grad)
    ]
    trainable_lora = [
        (name, parameter) for name, parameter in trainable if "lora_" in name.lower()
    ]
    base_trainable = [
        (name, parameter) for name, parameter in trainable if "lora_" not in name.lower()
    ]
    adapter_count = sum(int(parameter.numel()) for _, parameter in trainable_lora)
    base_count = sum(int(parameter.numel()) for _, parameter in base_trainable)
    if adapter_count <= 0:
        raise ValueError("Fresh SFT has no trainable LoRA parameters")
    if base_count:
        raise ValueError(
            "Fresh SFT base parameters are unexpectedly trainable: "
            f"count={base_count} examples={[name for name, _ in base_trainable[:10]]}"
        )
    if total_count <= 0:
        raise ValueError("Fresh SFT model has no parameters")
    if loading_audit.get("source_adapter_checkpoint") is not None:
        raise ValueError("Fresh SFT loading audit unexpectedly names a source adapter")
    if bool(loading_audit.get("aids_adapter_weights_loaded")):
        raise ValueError("Fresh SFT loading audit reports AIDS adapter weights")

    return {
        "schema": FRESH_INITIALIZATION_SCHEMA,
        "base_model_path": str(Path(base_model_path).expanduser().resolve()),
        "adapter_initialized_from_scratch": True,
        "source_adapter_checkpoint": None,
        "aids_adapter_weights_loaded": False,
        "adapter_names": adapter_names,
        "active_adapters": active_adapters,
        "single_active_adapter": True,
        "base_parameter_trainable_count": base_count,
        "adapter_trainable_parameter_count": adapter_count,
        "total_parameter_count": total_count,
        "trainable_percent": 100.0 * adapter_count / total_count,
        "trainable_parameter_name_examples": [
            name for name, _ in trainable_lora[:20]
        ],
        "lora_rank": int(lora_settings.rank),
        "lora_alpha": int(lora_settings.alpha),
        "lora_dropout": float(lora_settings.dropout),
        "lora_target_modules": list(lora_settings.target_modules),
        "loading": dict(loading_audit),
        "initialization_audit_passed": True,
    }


def tokenizer_reuse_audit(
    *,
    tokenizer: Any,
    base_model_path: str | Path,
    tokenizer_path: str | Path,
) -> dict[str, Any]:
    base = Path(base_model_path).expanduser().resolve()
    source = Path(tokenizer_path).expanduser().resolve()
    return {
        "base_model_path": str(base),
        "tokenizer_path": str(source),
        "tokenizer_reused": source != base,
        "adapter_weights_reused": False,
        "vocab_size": int(len(tokenizer)),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
    }


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(dict(payload), handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, target)


def fresh_checkpoint_manifest(
    output_root: str | Path,
    *,
    best_token_loss_checkpoint: str | Path | None,
) -> dict[str, Any]:
    """List fresh checkpoints without pretending a prior adapter initialized them."""

    root = Path(output_root).expanduser().resolve()

    def adapter_record(path: Path, *, step: int | None) -> dict[str, Any]:
        config = path / "adapter_config.json"
        weights = next(
            (
                candidate
                for candidate in (
                    path / "adapter_model.safetensors",
                    path / "adapter_model.bin",
                )
                if candidate.is_file()
            ),
            None,
        )
        if not config.is_file() or weights is None:
            raise ValueError(f"Incomplete fresh LoRA adapter output: {path}")
        return {
            "checkpoint": str(path),
            "step": step,
            "adapter_config": str(config),
            "adapter_weights": str(weights),
            "adapter_weight_bytes": weights.stat().st_size,
        }

    def checkpoint_step(path: Path) -> int:
        suffix = path.name.removeprefix("checkpoint-")
        return int(suffix) if suffix.isdigit() else 10**18

    checkpoints: list[dict[str, Any]] = []
    for checkpoint in sorted(root.glob("checkpoint-*"), key=checkpoint_step):
        if not checkpoint.is_dir():
            continue
        checkpoints.append(adapter_record(checkpoint, step=checkpoint_step(checkpoint)))
    final_adapter = adapter_record(root, step=None)
    return {
        "output_root": str(root),
        "initialization": "pure_chemlm_plus_random_fresh_lora",
        "initialization_checkpoint": None,
        "source_adapter_checkpoint": None,
        "aids_adapter_weights_loaded": False,
        "num_training_checkpoints": len(checkpoints),
        "checkpoints": checkpoints,
        "final_adapter": final_adapter,
        "best_token_loss_checkpoint": (
            str(Path(best_token_loss_checkpoint).expanduser().resolve())
            if best_token_loss_checkpoint
            else None
        ),
    }


def assert_no_adapter_files_read(paths: Sequence[str | Path]) -> None:
    """Guard tokenizer fallback code from accepting adapter files explicitly."""

    forbidden = {"adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"}
    offenders = [
        str(Path(path))
        for path in paths
        if Path(path).name in forbidden
    ]
    if offenders:
        raise ValueError(f"Tokenizer fallback must not read adapter files: {offenders}")


__all__ = [
    "DATASET_VARIANTS",
    "FRESH_INITIALIZATION_SCHEMA",
    "FreshLoRAConfig",
    "assert_no_adapter_files_read",
    "assert_pure_base_model",
    "audit_fresh_lora_model",
    "fresh_checkpoint_manifest",
    "initialize_fresh_lora",
    "resolve_variant_csvs",
    "tokenizer_reuse_audit",
    "write_json_atomic",
]
