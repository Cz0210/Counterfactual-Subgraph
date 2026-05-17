"""Full candidate-pool generation helpers for SFT / PPO inference."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.chem import is_parent_substructure
from src.data.ppo_prompt_dataset import PPOPromptRecord, load_ppo_prompt_records
from src.models.llm_generator import clean_generated_smiles
from src.rewards import ChemRLRewarder, CounterfactualTeacherScorer, TeacherSemanticScorer
from src.utils.io import ensure_directory, write_jsonl


_ADAPTER_FILENAMES = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
_TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


@dataclass(frozen=True, slots=True)
class FullPoolGenerationConfig:
    """Runtime knobs for full candidate-pool inference."""

    label_col: str = "label"
    smiles_col: str = "parent_smiles"
    target_label: int = 1
    num_return_sequences: int = 4
    generation_temperature: float = 0.7
    generation_top_p: float = 0.9
    generation_do_sample: bool = True
    max_new_tokens: int = 96
    batch_size: int = 1
    seed: int = 13
    enable_parent_projection: bool = False
    enable_projected_cf_reward: bool = False
    enable_substructure_distance_reward: bool = False
    substructure_distance_reward_weight: float = 0.3
    projection_penalty: float = 1.0
    enable_minimal_syntax_repair: bool = True
    enable_component_salvage: bool = True
    limit: int = 0
    local_files_only: bool = True
    trust_remote_code: bool = True


@dataclass(frozen=True, slots=True)
class CheckpointInspection:
    """Lightweight description of one adapter checkpoint directory."""

    checkpoint_dir: str
    exists: bool
    root_has_adapter: bool
    root_adapter_files: tuple[str, ...]
    checkpoint_subdirs: tuple[str, ...]
    checkpoint_subdirs_with_adapter: tuple[str, ...]
    tokenizer_files: tuple[str, ...]
    candidate_pool_exists: bool
    trainer_state_exists: bool
    decoded_chem_value_head_exists: bool
    selected_load_path: str | None
    selected_load_mode: str | None


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"none", "null", "nan"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _safe_mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def set_global_generation_seed(seed: int | None) -> None:
    """Set one global generation seed without touching per-sample sampling."""

    if seed is None:
        return

    import numpy as np
    import torch
    from transformers import set_seed

    resolved_seed = int(seed)
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    torch.manual_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seed)
    set_seed(resolved_seed)


def inspect_checkpoint_directory(checkpoint_dir: str | Path) -> CheckpointInspection:
    """Inspect one saved SFT / PPO checkpoint directory for inference readiness."""

    path = Path(checkpoint_dir).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        return CheckpointInspection(
            checkpoint_dir=str(path),
            exists=False,
            root_has_adapter=False,
            root_adapter_files=(),
            checkpoint_subdirs=(),
            checkpoint_subdirs_with_adapter=(),
            tokenizer_files=(),
            candidate_pool_exists=False,
            trainer_state_exists=False,
            decoded_chem_value_head_exists=False,
            selected_load_path=None,
            selected_load_mode=None,
        )

    root_adapter_files = tuple(
        filename for filename in _ADAPTER_FILENAMES if (path / filename).exists()
    )
    root_has_adapter = bool((path / "adapter_config.json").exists())

    checkpoint_dirs: list[Path] = []
    for child in path.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("checkpoint-"):
            continue
        suffix = child.name.split("checkpoint-", maxsplit=1)[-1]
        if not suffix.isdigit():
            continue
        checkpoint_dirs.append(child)
    checkpoint_dirs.sort(key=lambda item: int(item.name.split("checkpoint-", maxsplit=1)[-1]))
    checkpoint_names = tuple(child.name for child in checkpoint_dirs)
    checkpoint_adapter_dirs = tuple(
        child.name for child in checkpoint_dirs if (child / "adapter_config.json").exists()
    )

    tokenizer_files = tuple(
        filename for filename in _TOKENIZER_FILENAMES if (path / filename).exists()
    )

    selected_load_path: str | None = None
    selected_load_mode: str | None = None
    if root_has_adapter:
        selected_load_path = str(path)
        selected_load_mode = "root_adapter"
    elif checkpoint_adapter_dirs:
        selected_load_path = str(path / checkpoint_adapter_dirs[-1])
        selected_load_mode = "latest_checkpoint_subdir"

    return CheckpointInspection(
        checkpoint_dir=str(path),
        exists=True,
        root_has_adapter=root_has_adapter,
        root_adapter_files=root_adapter_files,
        checkpoint_subdirs=checkpoint_names,
        checkpoint_subdirs_with_adapter=checkpoint_adapter_dirs,
        tokenizer_files=tokenizer_files,
        candidate_pool_exists=(path / "candidate_pool.jsonl").exists(),
        trainer_state_exists=(path / "trainer_state.json").exists(),
        decoded_chem_value_head_exists=(path / "decoded_chem_value_head.pt").exists(),
        selected_load_path=selected_load_path,
        selected_load_mode=selected_load_mode,
    )


def resolve_adapter_load_path(checkpoint_dir: str | Path) -> Path:
    """Resolve one checkpoint directory to the concrete adapter path to load."""

    inspection = inspect_checkpoint_directory(checkpoint_dir)
    if inspection.selected_load_path is None:
        path = Path(checkpoint_dir).expanduser().resolve()
        raise FileNotFoundError(
            "Could not resolve an adapter load path from "
            f"{path}. Expected adapter files at the root or in checkpoint-* subdirectories."
        )
    return Path(inspection.selected_load_path).expanduser().resolve()


def _build_tokenizer(
    *,
    base_model_path: Path,
    trust_remote_code: bool,
    local_files_only: bool,
) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        use_fast=False,
    )
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer must define eos_token for candidate-pool generation.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _build_lora_model(
    *,
    base_model_path: Path,
    adapter_path: Path,
    trust_remote_code: bool,
    local_files_only: bool,
) -> Any:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.config.use_cache = False
    if getattr(base_model, "generation_config", None) is not None:
        base_model.generation_config.use_cache = False

    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=False,
    )
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
    return model


def build_generation_kwargs(
    *,
    encoded: dict[str, Any],
    tokenizer: Any,
    config: FullPoolGenerationConfig,
) -> dict[str, Any]:
    """Build one generate() kwargs dict compatible with current PEFT wrappers."""

    generation_kwargs: dict[str, Any] = {
        **encoded,
        "max_new_tokens": int(config.max_new_tokens),
        "num_return_sequences": int(config.num_return_sequences),
        "do_sample": bool(config.generation_do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": False,
    }
    if config.generation_do_sample:
        generation_kwargs["temperature"] = float(config.generation_temperature)
        generation_kwargs["top_p"] = float(config.generation_top_p)
    generation_kwargs.pop("generator", None)
    return generation_kwargs


def generate_ids_with_sanitized_kwargs(
    model: Any,
    generation_kwargs: dict[str, Any],
    *,
    torch_module: Any,
) -> Any:
    """Call model.generate() after removing unsupported kwargs like generator."""

    sanitized_generation_kwargs = dict(generation_kwargs)
    sanitized_generation_kwargs.pop("generator", None)
    with torch_module.no_grad():
        return model.generate(**sanitized_generation_kwargs)


def _resolve_final_fragment(row: dict[str, Any]) -> str | None:
    explicit_final = _normalize_text(row.get("final_fragment"))
    if explicit_final:
        return explicit_final
    projected_fragment = _normalize_text(
        row.get("projected_fragment") or row.get("nearest_parent_subgraph_smiles")
    )
    projected_used = _as_bool(row.get("used_projected_subgraph_for_reward"))
    projection_success = _as_bool(row.get("projection_success"))
    if (projected_used or projection_success) and projected_fragment:
        return projected_fragment
    return _normalize_text(
        row.get("core_fragment") or row.get("raw_fragment") or row.get("fragment")
    )


def _resolve_projection_method(row: dict[str, Any]) -> str:
    direct_substructure = bool(
        _as_bool(row.get("direct_substructure"))
        or _as_bool(row.get("direct_substructure_success"))
    )
    if direct_substructure:
        return "direct_match"
    if bool(_as_bool(row.get("projection_success"))) or bool(
        _as_bool(row.get("used_projected_subgraph_for_reward"))
    ):
        return "nearest_parent_subgraph"
    return "none"


@lru_cache(maxsize=16384)
def _infer_final_substructure(parent_smiles: str, fragment_smiles: str) -> bool:
    try:
        return bool(is_parent_substructure(parent_smiles, fragment_smiles))
    except Exception:
        return False


def _enrich_reward_log(
    row: dict[str, Any],
    *,
    record: PPOPromptRecord,
    candidate_index: int,
    raw_response: str,
) -> dict[str, Any]:
    final_fragment = _resolve_final_fragment(row)
    direct_substructure = bool(
        _as_bool(row.get("direct_substructure"))
        or _as_bool(row.get("direct_substructure_success"))
    )
    projection_method = _resolve_projection_method(row)
    projection_used = projection_method == "nearest_parent_subgraph"
    final_substructure = (
        _infer_final_substructure(record.parent_smiles, final_fragment)
        if final_fragment is not None
        else False
    )
    substructure_similarity = _as_float(row.get("substructure_similarity"))
    if substructure_similarity is None:
        substructure_similarity = _as_float(row.get("subdist_similarity"))
    if substructure_similarity is None and direct_substructure:
        substructure_similarity = 1.0
    if substructure_similarity is None and projection_used:
        substructure_similarity = _as_float(row.get("projection_score"))
    connected = bool(_as_bool(row.get("connected")) or _as_bool(row.get("connected_ok")))

    enriched = dict(row)
    enriched.update(
        {
            "parent_index": int(record.parent_index),
            "candidate_index": int(candidate_index),
            "label": int(record.label),
            "prompt": record.prompt,
            "raw_response": raw_response,
            "raw_output": raw_response,
            "raw_fragment": _normalize_text(
                row.get("raw_fragment") or row.get("fragment") or row.get("raw_output")
            ),
            "core_fragment": _normalize_text(row.get("core_fragment")),
            "projected_fragment": _normalize_text(
                row.get("projected_fragment") or row.get("nearest_parent_subgraph_smiles")
            ),
            "final_fragment": final_fragment,
            "connected": connected,
            "final_substructure": final_substructure,
            "projection_used": projection_used,
            "projection_method": projection_method,
            "substructure_similarity": substructure_similarity,
            "substructure_distance_reward": _as_float(
                row.get("substructure_distance_reward") or row.get("subdist_reward")
            ),
            "projection_penalty_applied": _as_float(
                row.get("projection_penalty_applied") or row.get("projection_penalty")
            )
            or 0.0,
            "fragment_atom_count": row.get("fragment_atom_count"),
            "projection_atom_count": row.get("projection_atom_count"),
            "cf_oracle_called": bool(
                _as_bool(row.get("cf_oracle_called"))
                or _as_bool(row.get("counterfactual_called"))
                or _as_bool(row.get("counterfactual_teacher_called"))
            ),
            "cf_oracle_skipped": not bool(
                _as_bool(row.get("cf_oracle_called"))
                or _as_bool(row.get("counterfactual_called"))
                or _as_bool(row.get("counterfactual_teacher_called"))
            ),
        }
    )
    return enriched


def _build_generation_summary(
    rows: list[dict[str, Any]],
    *,
    dataset_metadata: dict[str, Any],
    config: FullPoolGenerationConfig,
    adapter_path: Path,
    load_mode: str,
    checkpoint_inspection: CheckpointInspection | None,
    sft_inspection: CheckpointInspection,
    ppo_checkpoint_path: str | None,
) -> dict[str, Any]:
    total = len(rows)
    parse_ok_count = sum(1 for row in rows if bool(_as_bool(row.get("parse_ok"))))
    valid_count = sum(1 for row in rows if bool(_as_bool(row.get("valid"))))
    direct_substructure_count = sum(
        1 for row in rows if bool(_as_bool(row.get("direct_substructure")))
    )
    final_substructure_count = sum(
        1 for row in rows if bool(_as_bool(row.get("final_substructure")))
    )
    projection_used_count = sum(
        1 for row in rows if bool(_as_bool(row.get("projection_used")))
    )
    cf_flip_count = sum(1 for row in rows if bool(_as_bool(row.get("cf_flip"))))
    cf_drop_values = [
        value for value in (_as_float(row.get("cf_drop")) for row in rows) if value is not None
    ]
    failure_counter = Counter(
        str(row.get("failure_tag"))
        for row in rows
        if _normalize_text(row.get("failure_tag")) is not None
    )

    return {
        "dataset": dataset_metadata,
        "generation": asdict(config),
        "model_load": {
            "load_mode": load_mode,
            "adapter_path": str(adapter_path),
            "ppo_checkpoint_path": (
                str(Path(ppo_checkpoint_path).expanduser().resolve())
                if ppo_checkpoint_path
                else None
            ),
            "sft_lora_inspection": asdict(sft_inspection),
            "ppo_checkpoint_inspection": (
                asdict(checkpoint_inspection) if checkpoint_inspection is not None else None
            ),
            "load_note": (
                "PPO inference uses the resolved PPO adapter directly. "
                "Decoded-chem PPO saves the final trainable LoRA weights at the checkpoint root, "
                "so loading SFT then stacking PPO is not required when adapter files are present."
                if load_mode == "ppo"
                else "SFT-only baseline uses the resolved SFT adapter path."
            ),
        },
        "outputs": {
            "num_rows": total,
            "num_unique_parents": len({row["parent_index"] for row in rows}),
            "avg_candidates_per_parent": (
                float(total) / float(len({row["parent_index"] for row in rows}))
                if rows
                else 0.0
            ),
            "parse_ok_rate": _safe_rate(parse_ok_count, total),
            "valid_rate": _safe_rate(valid_count, total),
            "direct_substructure_rate": _safe_rate(direct_substructure_count, total),
            "final_substructure_rate": _safe_rate(final_substructure_count, total),
            "projection_used_rate": _safe_rate(projection_used_count, total),
            "cf_flip_rate": _safe_rate(cf_flip_count, total),
            "cf_drop_mean": _safe_mean(cf_drop_values),
            "failure_tag_distribution": dict(sorted(failure_counter.items())),
        },
    }


def generate_full_candidate_pool(
    *,
    dataset_path: str | Path,
    base_model_path: str | Path,
    sft_lora_path: str | Path,
    teacher_path: str | Path,
    out_jsonl: str | Path,
    out_summary_json: str | Path,
    config: FullPoolGenerationConfig,
    ppo_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run full-dataset inference and write one candidate_pool JSONL."""

    import torch

    set_global_generation_seed(config.seed)

    dataset_records, dataset_metadata = load_ppo_prompt_records(
        dataset_path,
        label_col=config.label_col,
        smiles_col=config.smiles_col,
        target_label=config.target_label,
        limit=config.limit,
    )
    if not dataset_records:
        raise ValueError(f"No usable prompt records were found in {dataset_path}")

    base_model = Path(base_model_path).expanduser().resolve()
    sft_dir = Path(sft_lora_path).expanduser().resolve()
    teacher_bundle = Path(teacher_path).expanduser().resolve()
    pool_path = Path(out_jsonl).expanduser().resolve()
    summary_path = Path(out_summary_json).expanduser().resolve()
    ensure_directory(pool_path.parent)
    ensure_directory(summary_path.parent)

    sft_inspection = inspect_checkpoint_directory(sft_dir)
    ppo_inspection = (
        inspect_checkpoint_directory(ppo_checkpoint_path)
        if ppo_checkpoint_path not in (None, "")
        else None
    )

    if ppo_checkpoint_path not in (None, ""):
        adapter_path = resolve_adapter_load_path(ppo_checkpoint_path)
        load_mode = "ppo"
    else:
        adapter_path = resolve_adapter_load_path(sft_dir)
        load_mode = "sft_only"

    tokenizer = _build_tokenizer(
        base_model_path=base_model,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    model = _build_lora_model(
        base_model_path=base_model,
        adapter_path=adapter_path,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )

    teacher_scorer = TeacherSemanticScorer(
        teacher_path=teacher_bundle,
        device="cpu",
    )
    counterfactual_teacher_scorer = CounterfactualTeacherScorer(
        teacher_path=teacher_bundle,
        device="cpu",
        teacher_scorer=teacher_scorer,
    )
    rewarder = ChemRLRewarder(
        oracle_path=teacher_bundle,
        default_parent_label=int(config.target_label),
        teacher_scorer=teacher_scorer,
        counterfactual_teacher_scorer=counterfactual_teacher_scorer,
        teacher_sem_scale=1.0,
        teacher_sem_missing_penalty=-5.0,
        enable_parent_projection=config.enable_parent_projection,
        projection_penalty=config.projection_penalty,
        enable_substructure_distance_reward=config.enable_substructure_distance_reward,
        substructure_distance_reward_weight=config.substructure_distance_reward_weight,
        enable_projected_cf_reward=config.enable_projected_cf_reward,
        enable_minimal_syntax_repair=config.enable_minimal_syntax_repair,
        enable_component_salvage=config.enable_component_salvage,
        max_generation_chars=max(96, int(config.max_new_tokens)),
        require_teacher_sem=True,
    )

    generated_rows: list[dict[str, Any]] = []
    batch_size = max(1, int(config.batch_size))
    model_device = next(model.parameters()).device

    for batch_start in range(0, len(dataset_records), batch_size):
        batch_records = dataset_records[batch_start : batch_start + batch_size]
        prompt_texts = [record.prompt for record in batch_records]
        encoded = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        encoded = {key: value.to(model_device) for key, value in encoded.items()}

        generation_kwargs = build_generation_kwargs(
            encoded=encoded,
            tokenizer=tokenizer,
            config=config,
        )
        generation_kwargs.pop("generator", None)
        generated_ids = generate_ids_with_sanitized_kwargs(
            model,
            generation_kwargs,
            torch_module=torch,
        )

        response_ids = generated_ids[:, encoded["input_ids"].shape[1] :]
        response_texts = tokenizer.batch_decode(
            response_ids.detach().cpu().tolist(),
            skip_special_tokens=True,
        )
        full_texts = tokenizer.batch_decode(
            generated_ids.detach().cpu().tolist(),
            skip_special_tokens=True,
        )

        parent_smiles_batch: list[str] = []
        fragments: list[str] = []
        labels: list[int] = []
        metas: list[dict[str, Any]] = []
        response_text_batch: list[str] = []
        record_repeats: list[PPOPromptRecord] = []
        candidate_indices: list[int] = []

        sequence_index = 0
        for record in batch_records:
            for candidate_index in range(int(config.num_return_sequences)):
                full_text = full_texts[sequence_index]
                response_text = response_texts[sequence_index]
                fragment = clean_generated_smiles(response_text)
                if full_text.startswith(record.prompt):
                    suffix = full_text[len(record.prompt) :].strip()
                    if suffix:
                        fragment = clean_generated_smiles(suffix)
                parent_smiles_batch.append(record.parent_smiles)
                fragments.append(fragment)
                labels.append(int(record.label))
                metas.append(
                    {
                        "id": record.parent_index,
                        "index": record.parent_index,
                        "prompt": record.prompt,
                        "parent_index": record.parent_index,
                        "candidate_index": candidate_index,
                    }
                )
                response_text_batch.append(response_text)
                record_repeats.append(record)
                candidate_indices.append(candidate_index)
                sequence_index += 1

        reward_tensor, reward_logs = rewarder.compute_rewards_from_decoded(
            parent_smiles=parent_smiles_batch,
            generated_fragments=fragments,
            raw_outputs=response_text_batch,
            labels=labels,
            metas=metas,
            device=model_device,
        )
        del reward_tensor

        for index, reward_log in enumerate(reward_logs):
            generated_rows.append(
                _enrich_reward_log(
                    reward_log,
                    record=record_repeats[index],
                    candidate_index=candidate_indices[index],
                    raw_response=response_text_batch[index],
                )
            )

    write_jsonl(pool_path, generated_rows)
    summary = _build_generation_summary(
        generated_rows,
        dataset_metadata=dataset_metadata,
        config=config,
        adapter_path=adapter_path,
        load_mode=load_mode,
        checkpoint_inspection=ppo_inspection,
        sft_inspection=sft_inspection,
        ppo_checkpoint_path=str(ppo_checkpoint_path) if ppo_checkpoint_path not in (None, "") else None,
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "CheckpointInspection",
    "FullPoolGenerationConfig",
    "build_generation_kwargs",
    "generate_full_candidate_pool",
    "generate_ids_with_sanitized_kwargs",
    "inspect_checkpoint_directory",
    "resolve_adapter_load_path",
    "set_global_generation_seed",
]
