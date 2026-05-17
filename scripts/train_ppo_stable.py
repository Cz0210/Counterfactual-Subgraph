#!/usr/bin/env python3
"""Parallel stable PPO training entrypoint for conservative decoded-chem runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path
from typing import Any, Sequence

from src.data.ppo_prompt_dataset import PPOPromptRecord, load_ppo_prompt_records
from src.eval.full_candidate_pool import _enrich_reward_log
from src.reward.reward_wrapper import ChemRLRewarder
from src.rewards.counterfactual_oracle import CounterfactualTeacherScorer
from src.rewards.reward_wrapper_stable import (
    StableChemRLRewardWrapper,
    StableTeacherConfidenceGateConfig,
)
from src.rewards.teacher_semantic import (
    TeacherSemanticScorer,
    require_teacher_semantic_scorer,
)
from src.utils.io import ensure_directory, write_jsonl
from src.utils.logging_utils import RunContext, configure_run_logger, write_runtime_manifest

from scripts.train_ppo import (
    DEFAULT_WANDB_RUN_NAME,
    _build_response_mask,
    _build_sequence_reward_assignments,
    _compute_response_logprobs,
    _compute_response_values,
    _decode_text_batch,
    _discounted_cumsum,
    _extract_fragment_from_text,
    _infer_single_training_device,
    _masked_mean,
    _masked_whiten,
    _safe_git_commit,
    _resolve_batch_list_field,
    apply_config_overrides,
    apply_decoded_chem_generation_defaults,
    build_data_collator,
    build_hf_dataset,
    build_parser as build_base_parser,
    build_policy_model,
    build_reward_model_wrapper,
    build_tokenizer,
    build_value_model,
    collect_runtime_environment_debug,
    ensure_score_head_for_experimental_ppo,
    extract_fragment_smiles,
    import_training_dependencies,
    load_prompt_examples,
    log_available_ppo_step_apis,
    log_runtime_model_debug,
    patch_internlm_cache,
    resolve_decoded_chem_generation_config,
    resolve_projected_cf_reward_enabled,
    resolve_sft_lora_path,
    resolve_substructure_distance_reward_config,
    save_decoded_chem_checkpoint,
    sync_generation_token_ids,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "hpc" / "rl_checkpoints"
DEFAULT_VAL_SCORE_ATOM_RATIO_TARGET = 0.35


def _first_non_empty_env_value(name: str, aliases: Sequence[str] = ()) -> str | None:
    for candidate_name in (name, *aliases):
        value = os.environ.get(candidate_name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _env_bool(name: str, default: bool, aliases: Sequence[str] = ()) -> bool:
    value = _first_non_empty_env_value(name, aliases)
    if value is None:
        return bool(default)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _env_int(name: str, default: int | None, aliases: Sequence[str] = ()) -> int | None:
    value = _first_non_empty_env_value(name, aliases)
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _env_float(
    name: str,
    default: float | None,
    aliases: Sequence[str] = (),
) -> float | None:
    value = _first_non_empty_env_value(name, aliases)
    if value is None:
        return default
    try:
        return float(str(value).strip())
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except Exception:
        return default
    if math.isnan(numeric):
        return default
    return numeric


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _safe_mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


@dataclass(frozen=True, slots=True)
class StablePPOConfig:
    ppo_learning_rate: float | None
    ppo_clip_range: float | None
    ppo_epochs: int | None
    max_grad_norm: float | None
    target_kl: float | None
    hard_kl: float | None
    enable_adaptive_kl: bool
    kl_penalty_init: float
    kl_penalty_multiplier: float
    reward_clip_min: float | None
    reward_clip_max: float | None
    normalize_reward: bool
    normalize_advantage: bool
    enable_teacher_confidence_gate: bool
    min_teacher_p_before: float
    low_conf_cf_weight: float
    enable_stable_early_stop: bool
    early_stop_patience: int
    early_stop_min_delta: float
    save_best_checkpoint: bool
    eval_every_steps: int
    val_dataset_path: str | None
    eval_num_samples: int


@dataclass(frozen=True, slots=True)
class StableValidationState:
    best_val_score: float | None = None
    best_step: int | None = None
    stale_eval_count: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser()
    parser.description = __doc__
    parser.add_argument(
        "--ppo-learning-rate",
        type=float,
        default=_env_float("PPO_LEARNING_RATE", None),
        help="Stable PPO optimizer learning rate override.",
    )
    parser.add_argument(
        "--ppo-clip-range",
        type=float,
        default=_env_float("PPO_CLIP_RANGE", None),
        help="Stable PPO clip range override.",
    )
    parser.add_argument(
        "--stable-ppo-epochs",
        type=int,
        default=_env_int("PPO_EPOCHS", None),
        help="Stable PPO epoch override for each update.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=_env_float("MAX_GRAD_NORM", None),
        help="Stable PPO gradient clipping max norm override.",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=_env_float("TARGET_KL", None),
        help="Soft KL target used for warnings and adaptive penalties.",
    )
    parser.add_argument(
        "--hard-kl",
        type=float,
        default=_env_float("HARD_KL", None),
        help="Hard KL limit that triggers stable early stop.",
    )
    parser.add_argument(
        "--enable-adaptive-kl",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("ENABLE_ADAPTIVE_KL", False),
        help="Enable adaptive KL penalty in stable PPO.",
    )
    parser.add_argument(
        "--kl-penalty-init",
        type=float,
        default=_env_float("KL_PENALTY_INIT", 0.05) or 0.05,
        help="Initial KL penalty when adaptive KL is enabled.",
    )
    parser.add_argument(
        "--kl-penalty-multiplier",
        type=float,
        default=_env_float("KL_PENALTY_MULTIPLIER", 1.5) or 1.5,
        help="Adaptive KL multiplier for penalty updates.",
    )
    parser.add_argument(
        "--reward-clip-min",
        type=float,
        default=_env_float("REWARD_CLIP_MIN", None),
        help="Optional minimum reward clip applied before PPO update.",
    )
    parser.add_argument(
        "--reward-clip-max",
        type=float,
        default=_env_float("REWARD_CLIP_MAX", None),
        help="Optional maximum reward clip applied before PPO update.",
    )
    parser.add_argument(
        "--normalize-reward",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("NORMALIZE_REWARD", False),
        help="Normalize rewards per PPO batch when feasible.",
    )
    parser.add_argument(
        "--normalize-advantage",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("NORMALIZE_ADVANTAGE", False),
        help="Normalize PPO advantages when feasible.",
    )
    parser.add_argument(
        "--enable-teacher-confidence-gate",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("ENABLE_TEACHER_CONFIDENCE_GATE", False),
        help="Enable stable-only teacher confidence gate for low-confidence parents.",
    )
    parser.add_argument(
        "--min-teacher-p-before",
        type=float,
        default=_env_float("MIN_TEACHER_P_BEFORE", 0.5) or 0.5,
        help="Teacher confidence gate threshold on p_before.",
    )
    parser.add_argument(
        "--low-conf-cf-weight",
        type=float,
        default=_env_float("LOW_CONF_CF_WEIGHT", 0.3) or 0.3,
        help="Weight applied to counterfactual reward under low parent confidence.",
    )
    parser.add_argument(
        "--enable-stable-early-stop",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("ENABLE_STABLE_EARLY_STOP", False),
        help="Enable stable PPO validation-based early stopping.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=_env_int("EARLY_STOP_PATIENCE", 3) or 3,
        help="How many eval windows may fail to improve before early stop.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=_env_float("EARLY_STOP_MIN_DELTA", 0.0) or 0.0,
        help="Minimum validation score improvement counted as progress.",
    )
    parser.add_argument(
        "--save-best-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("SAVE_BEST_CHECKPOINT", False),
        help="Save best validation checkpoint to best_val_checkpoint.",
    )
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=_env_int("EVAL_EVERY_STEPS", 0) or 0,
        help="Run validation every N PPO steps when VAL_DATASET_PATH is set.",
    )
    parser.add_argument(
        "--val-dataset-path",
        default=_first_non_empty_env_value("VAL_DATASET_PATH"),
        help="Optional fixed validation dataset path for stable PPO eval.",
    )
    parser.add_argument(
        "--eval-num-samples",
        type=int,
        default=_env_int("EVAL_NUM_SAMPLES", 0) or 0,
        help="Optional limit on validation prompt count. 0 means full validation set.",
    )
    return parser


def resolve_stable_config(args: argparse.Namespace) -> StablePPOConfig:
    return StablePPOConfig(
        ppo_learning_rate=args.ppo_learning_rate,
        ppo_clip_range=args.ppo_clip_range,
        ppo_epochs=args.stable_ppo_epochs,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        hard_kl=args.hard_kl,
        enable_adaptive_kl=bool(args.enable_adaptive_kl),
        kl_penalty_init=float(args.kl_penalty_init),
        kl_penalty_multiplier=float(args.kl_penalty_multiplier),
        reward_clip_min=args.reward_clip_min,
        reward_clip_max=args.reward_clip_max,
        normalize_reward=bool(args.normalize_reward),
        normalize_advantage=bool(args.normalize_advantage),
        enable_teacher_confidence_gate=bool(args.enable_teacher_confidence_gate),
        min_teacher_p_before=float(args.min_teacher_p_before),
        low_conf_cf_weight=float(args.low_conf_cf_weight),
        enable_stable_early_stop=bool(args.enable_stable_early_stop),
        early_stop_patience=max(1, int(args.early_stop_patience)),
        early_stop_min_delta=float(args.early_stop_min_delta),
        save_best_checkpoint=bool(args.save_best_checkpoint),
        eval_every_steps=max(0, int(args.eval_every_steps)),
        val_dataset_path=args.val_dataset_path,
        eval_num_samples=max(0, int(args.eval_num_samples)),
    )


def _resolve_final_fragment_for_metrics(row: dict[str, Any]) -> str | None:
    projected_fragment = row.get("projected_fragment") or row.get(
        "nearest_parent_subgraph_smiles"
    )
    projection_method = str(row.get("projection_method") or "").strip().lower()
    if projection_method == "nearest_parent_subgraph" and projected_fragment:
        return str(projected_fragment).strip()
    for key in ("core_fragment", "fragment", "raw_fragment"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _infer_final_substructure_rate_from_logs(
    reward_logs: Sequence[dict[str, Any]],
) -> float:
    if not reward_logs:
        return 0.0
    success_count = 0
    for reward_log in reward_logs:
        direct_substructure = bool(
            reward_log.get("direct_substructure")
            or reward_log.get("direct_substructure_success")
        )
        projection_method = str(reward_log.get("projection_method") or "").strip().lower()
        final_fragment = _resolve_final_fragment_for_metrics(reward_log)
        if direct_substructure:
            success_count += 1
            continue
        if projection_method == "nearest_parent_subgraph" and final_fragment:
            success_count += 1
    return _safe_rate(success_count, len(reward_logs))


def _summarize_step_metrics(reward_logs: Sequence[dict[str, Any]]) -> dict[str, float]:
    total = len(reward_logs)
    parse_ok_count = sum(1 for row in reward_logs if bool(row.get("parse_ok")))
    valid_count = sum(1 for row in reward_logs if bool(row.get("valid")))
    direct_substructure_count = sum(
        1
        for row in reward_logs
        if bool(row.get("direct_substructure") or row.get("direct_substructure_success"))
    )
    projection_used_count = sum(
        1
        for row in reward_logs
        if str(row.get("projection_method") or "").strip().lower()
        == "nearest_parent_subgraph"
    )
    oracle_ok_count = sum(1 for row in reward_logs if bool(row.get("oracle_ok")))
    cf_flip_count = sum(1 for row in reward_logs if bool(row.get("cf_flip")))
    core_unusable_count = sum(
        1
        for row in reward_logs
        if "core_unusable" in str(row.get("failure_tag") or "")
        or "core_unusable" in str(row.get("invalid_detail") or "")
    )
    parse_failed_count = sum(
        1
        for row in reward_logs
        if str(row.get("failure_tag") or "").startswith("parse_failed")
    )
    reward_values = [
        _safe_float(row.get("reward_total", row.get("total")))
        for row in reward_logs
    ]
    cf_drop_values = [
        _safe_float(row.get("cf_drop"))
        for row in reward_logs
        if row.get("cf_drop") is not None
    ]
    atom_ratio_values = [
        _safe_float(
            row.get("atom_ratio", row.get("final_fragment_atom_ratio")),
            default=float("nan"),
        )
        for row in reward_logs
    ]
    atom_ratio_values = [value for value in atom_ratio_values if not math.isnan(value)]
    return {
        "batch_size": float(total),
        "reward_mean": _safe_mean(reward_values) or 0.0,
        "parse_ok_rate": _safe_rate(parse_ok_count, total),
        "valid_rate": _safe_rate(valid_count, total),
        "direct_substructure_rate": _safe_rate(direct_substructure_count, total),
        "final_substructure_rate": _infer_final_substructure_rate_from_logs(reward_logs),
        "projection_used_rate": _safe_rate(projection_used_count, total),
        "oracle_ok_rate": _safe_rate(oracle_ok_count, total),
        "cf_flip_rate": _safe_rate(cf_flip_count, total),
        "cf_drop_mean": _safe_mean(cf_drop_values) or 0.0,
        "core_unusable_count": float(core_unusable_count),
        "parse_failed_count": float(parse_failed_count),
        "atom_ratio_mean": _safe_mean(atom_ratio_values) or 0.0,
    }


def _maybe_clip_and_normalize_rewards(
    reward_tensor: Any,
    *,
    reward_logs: list[dict[str, Any]],
    step_index: int,
    stable_config: StablePPOConfig,
    logger: Any,
    torch: Any,
) -> Any:
    processed = reward_tensor.clone().detach().to(dtype=torch.float32)

    if stable_config.reward_clip_min is not None or stable_config.reward_clip_max is not None:
        clip_min = (
            float(stable_config.reward_clip_min)
            if stable_config.reward_clip_min is not None
            else None
        )
        clip_max = (
            float(stable_config.reward_clip_max)
            if stable_config.reward_clip_max is not None
            else None
        )
        for index in range(processed.numel()):
            raw_reward = float(processed[index].item())
            clipped_reward = raw_reward
            if clip_min is not None:
                clipped_reward = max(clipped_reward, clip_min)
            if clip_max is not None:
                clipped_reward = min(clipped_reward, clip_max)
            processed[index] = float(clipped_reward)
            if index < len(reward_logs):
                reward_logs[index]["reward_total"] = float(clipped_reward)
                reward_logs[index]["total"] = float(clipped_reward)
            logger.info(
                "[STABLE_PPO_REWARD_CLIP] step=%s index=%s raw_reward=%s clipped_reward=%s clip_min=%s clip_max=%s",
                step_index,
                index,
                raw_reward,
                clipped_reward,
                clip_min,
                clip_max,
            )

    if stable_config.normalize_reward:
        if processed.numel() <= 1:
            logger.info(
                "[STABLE_PPO_REWARD_NORMALIZE] step=%s applied=False reason=batch_size_le_1",
                step_index,
            )
        else:
            reward_mean = processed.mean()
            reward_std = processed.std(unbiased=False)
            if float(reward_std.item()) <= 1e-8:
                logger.warning(
                    "[STABLE_PPO_REWARD_NORMALIZE] step=%s applied=False reason=std_too_small reward_std=%s",
                    step_index,
                    float(reward_std.item()),
                )
            else:
                processed = (processed - reward_mean) / reward_std
                logger.info(
                    "[STABLE_PPO_REWARD_NORMALIZE] step=%s applied=True reward_mean=%s reward_std=%s",
                    step_index,
                    float(reward_mean.item()),
                    float(reward_std.item()),
                )
                for index in range(processed.numel()):
                    if index < len(reward_logs):
                        reward_logs[index]["reward_total"] = float(processed[index].item())
                        reward_logs[index]["total"] = float(processed[index].item())

    return processed.to(device=reward_tensor.device, dtype=reward_tensor.dtype)


def _normalize_advantages_if_needed(
    advantages: Any,
    response_mask: Any,
    *,
    step_index: int,
    stable_config: StablePPOConfig,
    logger: Any,
    torch: Any,
) -> Any:
    if not stable_config.normalize_advantage:
        return advantages
    valid_count = int(response_mask.sum().detach().cpu().item())
    if valid_count <= 1:
        logger.warning(
            "[STABLE_PPO_ADVANTAGE_NORMALIZE] step=%s applied=False reason=insufficient_tokens valid_count=%s",
            step_index,
            valid_count,
        )
        return advantages
    normalized = _masked_whiten(advantages, response_mask, torch=torch)
    logger.info(
        "[STABLE_PPO_ADVANTAGE_NORMALIZE] step=%s applied=True valid_count=%s",
        step_index,
        valid_count,
    )
    return normalized


def _build_eval_summary(
    rows: Sequence[dict[str, Any]],
) -> dict[str, float]:
    total = len(rows)
    if total == 0:
        return {
            "parse_ok_rate": 0.0,
            "valid_rate": 0.0,
            "direct_substructure_rate": 0.0,
            "final_substructure_rate": 0.0,
            "projection_used_rate": 0.0,
            "cf_flip_rate": 0.0,
            "cf_drop_mean": 0.0,
            "reward_mean": 0.0,
            "core_unusable_rate": 0.0,
            "atom_ratio_mean": 0.0,
            "oracle_ok_rate": 0.0,
        }

    parse_ok_rate = _safe_rate(sum(1 for row in rows if bool(row.get("parse_ok"))), total)
    valid_rate = _safe_rate(sum(1 for row in rows if bool(row.get("valid"))), total)
    direct_substructure_rate = _safe_rate(
        sum(
            1
            for row in rows
            if bool(row.get("direct_substructure") or row.get("direct_substructure_success"))
        ),
        total,
    )
    final_substructure_rate = _safe_rate(
        sum(1 for row in rows if bool(row.get("final_substructure"))),
        total,
    )
    projection_used_rate = _safe_rate(
        sum(1 for row in rows if bool(row.get("projection_used"))),
        total,
    )
    cf_flip_rate = _safe_rate(sum(1 for row in rows if bool(row.get("cf_flip"))), total)
    reward_mean = _safe_mean(
        [_safe_float(row.get("reward_total", row.get("total"))) for row in rows]
    ) or 0.0
    cf_drop_mean = _safe_mean(
        [_safe_float(row.get("cf_drop")) for row in rows if row.get("cf_drop") is not None]
    ) or 0.0
    core_unusable_rate = _safe_rate(
        sum(
            1
            for row in rows
            if "core_unusable" in str(row.get("failure_tag") or "")
            or "core_unusable" in str(row.get("invalid_detail") or "")
        ),
        total,
    )
    atom_ratio_values = [
        _safe_float(row.get("atom_ratio", row.get("final_fragment_atom_ratio")), default=float("nan"))
        for row in rows
    ]
    atom_ratio_values = [value for value in atom_ratio_values if not math.isnan(value)]
    atom_ratio_mean = _safe_mean(atom_ratio_values) or 0.0
    oracle_ok_rate = _safe_rate(sum(1 for row in rows if bool(row.get("oracle_ok"))), total)
    return {
        "parse_ok_rate": parse_ok_rate,
        "valid_rate": valid_rate,
        "direct_substructure_rate": direct_substructure_rate,
        "final_substructure_rate": final_substructure_rate,
        "projection_used_rate": projection_used_rate,
        "cf_flip_rate": cf_flip_rate,
        "cf_drop_mean": cf_drop_mean,
        "reward_mean": reward_mean,
        "core_unusable_rate": core_unusable_rate,
        "atom_ratio_mean": atom_ratio_mean,
        "oracle_ok_rate": oracle_ok_rate,
    }


def _compute_val_score(summary: dict[str, float]) -> float:
    return (
        float(summary["cf_drop_mean"])
        + float(summary["cf_flip_rate"])
        + 0.8 * float(summary["direct_substructure_rate"])
        + 0.5 * float(summary["final_substructure_rate"])
        - 0.5 * float(summary["projection_used_rate"])
        - float(summary["core_unusable_rate"])
        - 0.3
        * abs(float(summary["atom_ratio_mean"]) - DEFAULT_VAL_SCORE_ATOM_RATIO_TARGET)
    )


def _evaluate_validation_set(
    *,
    deps: dict[str, Any],
    args: argparse.Namespace,
    stable_config: StablePPOConfig,
    policy_model: Any,
    tokenizer: Any,
    reward_wrapper: StableChemRLRewardWrapper,
    step_index: int,
    logger: Any,
) -> dict[str, float] | None:
    if not stable_config.val_dataset_path:
        return None

    val_records, _metadata = load_ppo_prompt_records(
        stable_config.val_dataset_path,
        label_col="label",
        smiles_col="parent_smiles",
        target_label=args.default_parent_label,
        limit=stable_config.eval_num_samples,
    )
    if not val_records:
        logger.warning(
            "[STABLE_PPO_EVAL_METRICS] step=%s skipped=True reason=no_usable_val_records val_dataset_path=%s",
            step_index,
            stable_config.val_dataset_path,
        )
        return None

    torch = deps["torch"]
    generation_config = resolve_decoded_chem_generation_config(args)
    model_device = next(policy_model.parameters()).device
    generator = torch.Generator(device=model_device)
    generator.manual_seed(int(args.seed))

    policy_model.eval()
    eval_rows: list[dict[str, Any]] = []
    batch_size = max(1, min(int(args.batch_size), len(val_records)))

    for batch_start in range(0, len(val_records), batch_size):
        batch_records = val_records[batch_start : batch_start + batch_size]
        encoded = tokenizer(
            [record.prompt for record in batch_records],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        generation_kwargs: dict[str, Any] = {
            **encoded,
            "max_new_tokens": generation_config.max_new_tokens,
            "do_sample": generation_config.do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": False,
            "generator": generator,
        }
        if generation_config.do_sample:
            generation_kwargs["temperature"] = generation_config.temperature
            generation_kwargs["top_p"] = generation_config.top_p

        with torch.no_grad():
            generated_ids = policy_model.generate(**generation_kwargs)

        response_ids = generated_ids[:, encoded["input_ids"].shape[1] :]
        response_texts = _decode_text_batch(
            tokenizer,
            response_ids.detach().cpu().tolist(),
            torch=torch,
        )
        full_texts = _decode_text_batch(
            tokenizer,
            generated_ids.detach().cpu().tolist(),
            torch=torch,
        )

        parent_smiles_batch: list[str] = []
        fragments: list[str] = []
        labels: list[int] = []
        metas: list[dict[str, Any]] = []
        for index, record in enumerate(batch_records):
            full_text = full_texts[index]
            response_text = response_texts[index]
            fragment = _extract_fragment_from_text(full_text, record.prompt)
            if not fragment:
                fragment = extract_fragment_smiles(response_text)
            parent_smiles_batch.append(record.parent_smiles)
            fragments.append(fragment)
            labels.append(int(record.label))
            metas.append(
                {
                    "id": record.parent_index,
                    "index": record.parent_index,
                    "prompt": record.prompt,
                }
            )

        reward_tensor, reward_logs = reward_wrapper.compute_rewards_from_decoded(
            parent_smiles=parent_smiles_batch,
            generated_fragments=fragments,
            raw_outputs=response_texts,
            labels=labels,
            metas=metas,
            device=model_device,
            step_index=step_index,
        )
        del reward_tensor

        for index, reward_log in enumerate(reward_logs):
            eval_rows.append(
                _enrich_reward_log(
                    reward_log,
                    record=batch_records[index],
                    candidate_index=0,
                    raw_response=response_texts[index],
                )
            )

    policy_model.train()
    summary = _build_eval_summary(eval_rows)
    summary["val_score"] = _compute_val_score(summary)
    logger.info(
        "[STABLE_PPO_EVAL_METRICS] step=%s parse_ok_rate=%.4f valid_rate=%.4f direct_substructure_rate=%.4f final_substructure_rate=%.4f projection_used_rate=%.4f cf_flip_rate=%.4f cf_drop_mean=%.4f reward_mean=%.4f core_unusable_rate=%.4f atom_ratio_mean=%.4f val_score=%.4f",
        step_index,
        summary["parse_ok_rate"],
        summary["valid_rate"],
        summary["direct_substructure_rate"],
        summary["final_substructure_rate"],
        summary["projection_used_rate"],
        summary["cf_flip_rate"],
        summary["cf_drop_mean"],
        summary["reward_mean"],
        summary["core_unusable_rate"],
        summary["atom_ratio_mean"],
        summary["val_score"],
    )
    return summary


def _maybe_save_best_checkpoint(
    *,
    deps: dict[str, Any],
    stable_config: StablePPOConfig,
    summary: dict[str, float] | None,
    validation_state: StableValidationState,
    step_index: int,
    policy_model: Any,
    value_model: Any,
    tokenizer: Any,
    output_dir: Path,
    logger: Any,
) -> tuple[StableValidationState, bool]:
    if summary is None:
        return validation_state, False
    current_score = float(summary["val_score"])
    best_score = validation_state.best_val_score
    improvement = (
        best_score is None
        or current_score > float(best_score) + float(stable_config.early_stop_min_delta)
    )
    if improvement:
        new_state = StableValidationState(
            best_val_score=current_score,
            best_step=step_index,
            stale_eval_count=0,
        )
        if stable_config.save_best_checkpoint:
            best_dir = output_dir / "best_val_checkpoint"
            ensure_directory(best_dir)
            save_decoded_chem_checkpoint(
                policy_model=policy_model,
                value_model=value_model,
                tokenizer=tokenizer,
                output_dir=best_dir,
                torch=deps["torch"],
            )
            logger.info(
                "[STABLE_PPO_BEST_CHECKPOINT] step=%s val_score=%.4f path=%s",
                step_index,
                current_score,
                best_dir,
            )
        return new_state, False

    stale_count = validation_state.stale_eval_count + 1
    new_state = StableValidationState(
        best_val_score=validation_state.best_val_score,
        best_step=validation_state.best_step,
        stale_eval_count=stale_count,
    )
    should_stop = (
        stable_config.enable_stable_early_stop
        and stale_count >= stable_config.early_stop_patience
    )
    if should_stop:
        logger.info(
            "[STABLE_PPO_EARLY_STOP] step=%s best_step=%s best_val_score=%s reason=patience_exhausted stale_eval_count=%s",
            step_index,
            validation_state.best_step,
            validation_state.best_val_score,
            stale_count,
        )
    return new_state, should_stop


def run_stable_decoded_chem_ppo_loop(
    *,
    deps: dict[str, Any],
    args: argparse.Namespace,
    stable_config: StablePPOConfig,
    actual_batch_size: int,
    policy_model: Any,
    reference_model: Any,
    value_model: Any,
    tokenizer: Any,
    train_dataset: Any,
    chem_reward_model: Any,
    stable_reward_wrapper: StableChemRLRewardWrapper,
    output_dir: Path,
    logger: Any,
) -> Path:
    torch = deps["torch"]
    del chem_reward_model

    log_available_ppo_step_apis(logger)
    logger.info(
        "[STABLE_PPO_CONFIG] learning_rate=%s clip_range=%s ppo_epochs=%s max_grad_norm=%s target_kl=%s hard_kl=%s adaptive_kl=%s kl_penalty_init=%s kl_penalty_multiplier=%s reward_clip_min=%s reward_clip_max=%s normalize_reward=%s normalize_advantage=%s enable_teacher_confidence_gate=%s min_teacher_p_before=%s low_conf_cf_weight=%s enable_stable_early_stop=%s early_stop_patience=%s early_stop_min_delta=%s save_best_checkpoint=%s eval_every_steps=%s val_dataset_path=%s eval_num_samples=%s",
        stable_config.ppo_learning_rate,
        stable_config.ppo_clip_range,
        stable_config.ppo_epochs,
        stable_config.max_grad_norm,
        stable_config.target_kl,
        stable_config.hard_kl,
        stable_config.enable_adaptive_kl,
        stable_config.kl_penalty_init,
        stable_config.kl_penalty_multiplier,
        stable_config.reward_clip_min,
        stable_config.reward_clip_max,
        stable_config.normalize_reward,
        stable_config.normalize_advantage,
        stable_config.enable_teacher_confidence_gate,
        stable_config.min_teacher_p_before,
        stable_config.low_conf_cf_weight,
        stable_config.enable_stable_early_stop,
        stable_config.early_stop_patience,
        stable_config.early_stop_min_delta,
        stable_config.save_best_checkpoint,
        stable_config.eval_every_steps,
        stable_config.val_dataset_path,
        stable_config.eval_num_samples,
    )
    logger.info(
        "[STABLE_PPO_CONFIG] ppo_loop=%s diagnose_reward_flow=%s require_chemistry_reward_path=%s candidate_pool_path=%s",
        args.ppo_loop,
        args.diagnose_reward_flow,
        args.require_chemistry_reward_path,
        args.candidate_pool_path or str(output_dir / "candidate_pool.jsonl"),
    )

    rollout_device = _infer_single_training_device(
        logger=logger,
        torch=torch,
        policy_model=policy_model,
        reference_model=reference_model,
        value_model=value_model,
    )
    policy_model.train()
    reference_model.eval()
    value_model.train()
    sync_generation_token_ids(policy_model, tokenizer)
    patch_internlm_cache(policy_model)
    sync_generation_token_ids(reference_model, tokenizer)
    sync_generation_token_ids(value_model, tokenizer)
    log_runtime_model_debug(logger, label="stable_decoded_chem.policy_model", model=policy_model)
    log_runtime_model_debug(logger, label="stable_decoded_chem.reference_model", model=reference_model)
    log_runtime_model_debug(logger, label="stable_decoded_chem.value_model", model=value_model)

    trainable_parameters = [
        parameter
        for parameter in list(policy_model.parameters()) + list(value_model.parameters())
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(stable_config.ppo_learning_rate or args.learning_rate),
    )
    data_collator = build_data_collator(deps, tokenizer)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(actual_batch_size),
        shuffle=False,
        collate_fn=data_collator,
    )
    dataloader_iterator = iter(dataloader)

    candidate_pool_rows: list[dict[str, Any]] = []
    candidate_pool_path = (
        Path(args.candidate_pool_path).expanduser().resolve()
        if args.candidate_pool_path
        else output_dir / "candidate_pool.jsonl"
    )
    generation_config = resolve_decoded_chem_generation_config(args)
    validation_state = StableValidationState()
    current_kl_penalty = float(stable_config.kl_penalty_init)
    should_stop = False
    early_stop_reason = None
    chemistry_reward_called = False
    chemistry_reward_update_called = False

    for step_index in range(1, int(args.max_steps) + 1):
        try:
            batch = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(dataloader)
            batch = next(dataloader_iterator)

        prompt_texts = [str(text) for text in _resolve_batch_list_field(batch, "query", "prompt", "text")]
        parent_smiles_batch = [
            str(smiles)
            for smiles in _resolve_batch_list_field(batch, "parent_smiles", "smiles")
        ]
        parent_labels = [int(label) for label in _resolve_batch_list_field(batch, "original_label", "label")]
        try:
            batch_ids = list(_resolve_batch_list_field(batch, "index", "id", "graph_id"))
        except KeyError:
            batch_ids = list(range(len(parent_smiles_batch)))

        input_ids = batch["input_ids"].to(rollout_device)
        attention_mask = batch["attention_mask"].to(rollout_device)
        with torch.no_grad():
            generation_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": generation_config.max_new_tokens,
                "do_sample": generation_config.do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": False,
            }
            if generation_config.do_sample:
                generation_kwargs["top_p"] = generation_config.top_p
                generation_kwargs["temperature"] = generation_config.temperature
            generated_ids = policy_model.generate(**generation_kwargs)

        response_ids = generated_ids[:, input_ids.shape[1] :]
        if response_ids.size(1) == 0:
            surrogate_token_id = tokenizer.eos_token_id
            if surrogate_token_id is None:
                surrogate_token_id = tokenizer.pad_token_id
            if surrogate_token_id is None:
                surrogate_token_id = 0
            response_ids = generated_ids.new_full(
                (generated_ids.size(0), 1),
                int(surrogate_token_id),
            )
        response_mask = _build_response_mask(
            response_ids,
            eos_token_id=tokenizer.eos_token_id,
            torch=torch,
        )
        response_texts = tokenizer.batch_decode(
            response_ids.detach().cpu().tolist(),
            skip_special_tokens=True,
        )
        full_texts = tokenizer.batch_decode(
            generated_ids.detach().cpu().tolist(),
            skip_special_tokens=True,
        )

        fragments: list[str] = []
        for full_text, response_text, prompt_text in zip(
            full_texts,
            response_texts,
            prompt_texts,
            strict=True,
        ):
            fragment = _extract_fragment_from_text(full_text, prompt_text)
            if not fragment:
                fragment = extract_fragment_smiles(response_text)
            fragments.append(fragment)

        reward_tensor, reward_logs = stable_reward_wrapper.compute_rewards_from_decoded(
            parent_smiles=parent_smiles_batch,
            generated_fragments=fragments,
            raw_outputs=response_texts,
            labels=parent_labels,
            metas=[
                {
                    "id": batch_ids[index],
                    "index": batch_ids[index],
                    "prompt": prompt_texts[index],
                }
                for index in range(len(parent_smiles_batch))
            ],
            device=rollout_device,
            step_index=step_index,
        )
        chemistry_reward_called = True
        reward_tensor = _maybe_clip_and_normalize_rewards(
            reward_tensor,
            reward_logs=reward_logs,
            step_index=step_index,
            stable_config=stable_config,
            logger=logger,
            torch=torch,
        )

        for reward_log in reward_logs:
            reward_log["step_index"] = step_index
        candidate_pool_rows.extend(reward_logs)

        step_metrics = _summarize_step_metrics(reward_logs)

        with torch.no_grad():
            old_logprobs = _compute_response_logprobs(
                policy_model,
                query_ids=input_ids,
                query_attention_mask=attention_mask,
                response_ids=response_ids,
                response_mask=response_mask,
                torch=torch,
            )
            ref_logprobs = _compute_response_logprobs(
                reference_model,
                query_ids=input_ids,
                query_attention_mask=attention_mask,
                response_ids=response_ids,
                response_mask=response_mask,
                torch=torch,
            )
            old_values = _compute_response_values(
                value_model,
                query_ids=input_ids,
                query_attention_mask=attention_mask,
                response_ids=response_ids,
                response_mask=response_mask,
                torch=torch,
            )

        token_rewards = _build_sequence_reward_assignments(
            sequence_rewards=reward_tensor.to(dtype=old_logprobs.dtype),
            policy_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            response_mask=response_mask,
            kl_coef=float(args.init_kl_coef),
            torch=torch,
        )
        returns = _discounted_cumsum(
            token_rewards,
            response_mask=response_mask,
            gamma=1.0,
            torch=torch,
        )
        advantages = returns - old_values
        advantages = advantages.detach()
        advantages = _normalize_advantages_if_needed(
            advantages,
            response_mask,
            step_index=step_index,
            stable_config=stable_config,
            logger=logger,
            torch=torch,
        )

        old_logprobs = old_logprobs.detach()
        ref_logprobs = ref_logprobs.detach()
        old_values = old_values.detach()
        returns = returns.detach()

        clip_range = float(
            stable_config.ppo_clip_range
            if stable_config.ppo_clip_range is not None
            else 0.2
        )
        value_clip_range = 0.2
        value_loss_coef = 0.5
        max_grad_norm = float(
            stable_config.max_grad_norm
            if stable_config.max_grad_norm is not None
            else 1.0
        )
        ppo_epochs = max(
            1,
            int(stable_config.ppo_epochs if stable_config.ppo_epochs is not None else args.ppo_epochs),
        )
        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_total_loss = 0.0
        last_approx_kl = 0.0
        last_approx_kl_tensor = None
        for _ppo_epoch in range(ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            current_logprobs = _compute_response_logprobs(
                policy_model,
                query_ids=input_ids,
                query_attention_mask=attention_mask,
                response_ids=response_ids,
                response_mask=response_mask,
                torch=torch,
            )
            current_values = _compute_response_values(
                value_model,
                query_ids=input_ids,
                query_attention_mask=attention_mask,
                response_ids=response_ids,
                response_mask=response_mask,
                torch=torch,
            )

            ratios = torch.exp(current_logprobs - old_logprobs)
            unclipped_objective = ratios * advantages
            clipped_objective = torch.clamp(
                ratios,
                1.0 - clip_range,
                1.0 + clip_range,
            ) * advantages
            policy_loss = -_masked_mean(
                torch.minimum(unclipped_objective, clipped_objective),
                response_mask,
            )

            clipped_values = old_values + torch.clamp(
                current_values - old_values,
                -value_clip_range,
                value_clip_range,
            )
            value_loss = 0.5 * _masked_mean(
                torch.maximum(
                    (current_values - returns) ** 2,
                    (clipped_values - returns) ** 2,
                ),
                response_mask,
            )

            approx_kl_tensor = _masked_mean(
                current_logprobs - ref_logprobs,
                response_mask,
            )
            total_loss = policy_loss + value_loss_coef * value_loss
            if stable_config.enable_adaptive_kl:
                total_loss = total_loss + float(current_kl_penalty) * approx_kl_tensor
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
            optimizer.step()

            last_policy_loss = float(policy_loss.detach().cpu().item())
            last_value_loss = float(value_loss.detach().cpu().item())
            last_total_loss = float(total_loss.detach().cpu().item())
            last_approx_kl = float(approx_kl_tensor.detach().cpu().item())
            last_approx_kl_tensor = approx_kl_tensor

        chemistry_reward_update_called = True
        if stable_config.enable_adaptive_kl and stable_config.target_kl is not None:
            if last_approx_kl > float(stable_config.target_kl):
                current_kl_penalty *= float(stable_config.kl_penalty_multiplier)
            elif last_approx_kl < float(stable_config.target_kl) / 2.0:
                current_kl_penalty = max(
                    0.0,
                    current_kl_penalty / float(stable_config.kl_penalty_multiplier),
                )
            logger.info(
                "[STABLE_PPO_ADAPTIVE_KL] step=%s approx_kl=%.4f kl_penalty=%.6f",
                step_index,
                last_approx_kl,
                current_kl_penalty,
            )
        elif stable_config.enable_adaptive_kl:
            logger.warning(
                "[STABLE_PPO_ADAPTIVE_KL] step=%s skipped=True reason=missing_target_kl kl_penalty=%.6f",
                step_index,
                current_kl_penalty,
            )

        if stable_config.target_kl is not None and last_approx_kl > float(stable_config.target_kl):
            logger.warning(
                "[STABLE_PPO_KL_WARNING] step=%s approx_kl=%.4f target_kl=%.4f",
                step_index,
                last_approx_kl,
                float(stable_config.target_kl),
            )
        if stable_config.hard_kl is not None and last_approx_kl > float(stable_config.hard_kl):
            logger.warning(
                "[STABLE_PPO_KL_WARNING] step=%s approx_kl=%.4f hard_kl=%.4f",
                step_index,
                last_approx_kl,
                float(stable_config.hard_kl),
            )
            should_stop = True
            early_stop_reason = "hard_kl"

        logger.info(
            "[STABLE_PPO_UPDATE] step=%s reward_mean=%.4f reward_min=%.4f reward_max=%.4f policy_loss=%.4f value_loss=%.4f total_loss=%.4f approx_kl=%.4f parse_ok_rate=%.4f valid_rate=%.4f direct_substructure_rate=%.4f final_substructure_rate=%.4f projection_used_rate=%.4f oracle_ok_rate=%.4f cf_flip_rate=%.4f cf_drop_mean=%.4f core_unusable_count=%s parse_failed_count=%s atom_ratio_mean=%.4f",
            step_index,
            float(reward_tensor.mean().detach().cpu().item()),
            float(reward_tensor.min().detach().cpu().item()),
            float(reward_tensor.max().detach().cpu().item()),
            last_policy_loss,
            last_value_loss,
            last_total_loss,
            last_approx_kl,
            step_metrics["parse_ok_rate"],
            step_metrics["valid_rate"],
            step_metrics["direct_substructure_rate"],
            step_metrics["final_substructure_rate"],
            step_metrics["projection_used_rate"],
            step_metrics["oracle_ok_rate"],
            step_metrics["cf_flip_rate"],
            step_metrics["cf_drop_mean"],
            int(step_metrics["core_unusable_count"]),
            int(step_metrics["parse_failed_count"]),
            step_metrics["atom_ratio_mean"],
        )

        if step_index % max(1, int(args.save_steps)) == 0:
            checkpoint_dir = output_dir / f"checkpoint-{step_index}"
            save_decoded_chem_checkpoint(
                policy_model=policy_model,
                value_model=value_model,
                tokenizer=tokenizer,
                output_dir=checkpoint_dir,
                torch=torch,
            )

        if (
            stable_config.val_dataset_path
            and stable_config.eval_every_steps > 0
            and step_index % stable_config.eval_every_steps == 0
        ):
            summary = _evaluate_validation_set(
                deps=deps,
                args=args,
                stable_config=stable_config,
                policy_model=policy_model,
                tokenizer=tokenizer,
                reward_wrapper=stable_reward_wrapper,
                step_index=step_index,
                logger=logger,
            )
            validation_state, stop_from_eval = _maybe_save_best_checkpoint(
                deps=deps,
                stable_config=stable_config,
                summary=summary,
                validation_state=validation_state,
                step_index=step_index,
                policy_model=policy_model,
                value_model=value_model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                logger=logger,
            )
            if stop_from_eval:
                should_stop = True
                early_stop_reason = "validation_patience"

        if should_stop:
            logger.info(
                "[STABLE_PPO_EARLY_STOP] step=%s best_step=%s best_val_score=%s reason=%s",
                step_index,
                validation_state.best_step,
                validation_state.best_val_score,
                early_stop_reason,
            )
            break

    if args.require_chemistry_reward_path and not chemistry_reward_called:
        raise RuntimeError("Stable decoded chemistry PPO did not call the chemistry reward path.")
    if args.require_chemistry_reward_path and not chemistry_reward_update_called:
        raise RuntimeError("Stable decoded chemistry PPO did not execute a PPO update.")
    write_jsonl(candidate_pool_path, candidate_pool_rows)
    logger.info(
        "Saved stable decoded chemistry candidate pool to %s (%s rows)",
        candidate_pool_path,
        len(candidate_pool_rows),
    )
    return save_decoded_chem_checkpoint(
        policy_model=policy_model,
        value_model=value_model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        torch=torch,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args = apply_config_overrides(args, parser)
    args = apply_decoded_chem_generation_defaults(args)
    (
        args.enable_substructure_distance_reward,
        args.substructure_distance_reward_weight,
    ) = resolve_substructure_distance_reward_config(args)
    projected_cf_reward_enabled = resolve_projected_cf_reward_enabled(args)
    args.projected_cf_reward_enabled = projected_cf_reward_enabled
    stable_config = resolve_stable_config(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    sft_lora_path, sft_lora_source = resolve_sft_lora_path(args)
    oracle_path = Path(args.oracle_path).expanduser().resolve()
    teacher_path = Path(args.teacher_path).expanduser().resolve()

    ensure_directory(output_dir)
    run_name = f"{DEFAULT_WANDB_RUN_NAME}_stable_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = configure_run_logger(
        "train_ppo_stable",
        context=RunContext(
            run_name=run_name,
            output_dir=output_dir,
            stage="ppo_stable",
            seed=args.seed,
        ),
        log_dir=output_dir / "logs",
    )
    logger.info("Runtime environment: %s", collect_runtime_environment_debug())

    write_runtime_manifest(
        output_dir / "train_ppo_stable_manifest.json",
        {
            "run_name": run_name,
            "git_commit": _safe_git_commit(),
            "model_path": str(model_path),
            "sft_lora_path": str(sft_lora_path),
            "sft_lora_source": sft_lora_source,
            "oracle_path": str(oracle_path),
            "teacher_path": str(teacher_path),
            "dataset_path": str(dataset_path),
            "output_dir": str(output_dir),
            "args": vars(args),
            "stable_config": stable_config.__dict__,
        },
    )

    examples = load_prompt_examples(
        dataset_path,
        default_parent_label=args.default_parent_label,
        only_positive=args.only_positive,
        include_label_in_prompt=args.include_label_in_prompt,
        max_prompt_examples=args.max_prompt_examples,
    )
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

    policy_model = build_policy_model(
        deps,
        model_path=model_path,
        adapter_path=sft_lora_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        is_trainable=True,
    )
    reference_model = build_policy_model(
        deps,
        model_path=model_path,
        adapter_path=sft_lora_path,
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
    value_model = ensure_score_head_for_experimental_ppo(value_model, "stable_value_model")

    teacher_scorer = TeacherSemanticScorer(
        teacher_path=teacher_path,
        device="cpu",
        logger=logger,
    )
    if args.require_teacher_sem:
        require_teacher_semantic_scorer(
            teacher_scorer,
            teacher_path=teacher_path,
        )
    counterfactual_teacher_scorer = None
    if not args.disable_counterfactual_teacher:
        counterfactual_teacher_scorer = CounterfactualTeacherScorer(
            teacher_path=teacher_path,
            device="cpu",
            logger=logger,
            flip_bonus=args.teacher_cf_flip_bonus,
            missing_penalty=args.teacher_sem_missing_penalty,
            teacher_scorer=teacher_scorer,
        )

    rewarder = ChemRLRewarder(
        oracle_path=oracle_path,
        default_parent_label=args.default_parent_label,
        teacher_scorer=teacher_scorer,
        counterfactual_teacher_scorer=counterfactual_teacher_scorer,
        teacher_sem_scale=args.teacher_sem_scale,
        teacher_sem_missing_penalty=args.teacher_sem_missing_penalty,
        full_parent_penalty=args.full_parent_penalty,
        empty_residual_penalty=args.empty_residual_penalty,
        enable_parent_aware_repair=args.enable_parent_aware_repair,
        repair_min_similarity=args.repair_min_similarity,
        repair_max_candidates=args.repair_max_candidates,
        enable_parent_projection=args.enable_parent_projection,
        projection_min_score=args.projection_min_score,
        projection_max_candidates=args.projection_max_candidates,
        projection_min_atoms=args.projection_min_atoms,
        projection_max_atom_ratio=args.projection_max_atom_ratio,
        projection_penalty=args.projection_penalty,
        projection_enable_khop3=args.projection_enable_khop3,
        projection_mcs_timeout=args.projection_mcs_timeout,
        enable_substructure_distance_reward=args.enable_substructure_distance_reward,
        substructure_distance_reward_weight=args.substructure_distance_reward_weight,
        substructure_distance_min_atom_ratio=args.substructure_distance_min_atom_ratio,
        substructure_distance_max_atom_ratio=args.substructure_distance_max_atom_ratio,
        substructure_distance_topk=args.substructure_distance_topk,
        substructure_distance_mcs_timeout=args.substructure_distance_mcs_timeout,
        substructure_distance_sim_threshold=args.substructure_distance_sim_threshold,
        enable_projected_cf_reward=projected_cf_reward_enabled,
        disable_projected_cf_reward=args.disable_projected_cf_reward,
        enable_minimal_syntax_repair=args.enable_minimal_syntax_repair,
        syntax_repair_max_edits=args.repair_max_edits,
        syntax_repair_min_atoms=args.repair_min_atoms,
        syntax_repair_allow_parentheses_fix=args.repair_allow_parentheses_fix,
        syntax_repair_allow_ring_fix=args.repair_allow_ring_fix,
        syntax_repair_allow_tail_trim=args.repair_allow_tail_trim,
        syntax_repair_allow_balanced_prefix_salvage=args.repair_allow_balanced_prefix_salvage,
        syntax_repair_prefer_prefix_salvage=args.repair_prefer_prefix_salvage,
        syntax_repair_max_suffix_trim=args.repair_max_suffix_trim,
        syntax_repair_max_added_closures=args.repair_max_added_closures,
        enable_component_salvage=args.enable_component_salvage,
        component_salvage_method=args.component_salvage_method,
        component_salvage_min_atoms=args.component_salvage_min_atoms,
        multi_dummy_hard_fail_threshold=args.multi_dummy_hard_fail_threshold,
        enable_light_dummy_salvage=args.enable_light_dummy_salvage,
        near_parent_hard_ratio=args.near_parent_hard_ratio,
        min_residual_atoms=args.min_residual_atoms,
        min_residual_ratio=args.min_residual_ratio,
        min_fragment_atoms=args.min_fragment_atoms,
        tiny_fragment_hard_fail_penalty=args.tiny_fragment_hard_fail_penalty,
        enable_size_window_reward=args.enable_size_window_reward,
        size_window_low=args.size_window_low,
        size_window_high=args.size_window_high,
        size_window_bonus=args.size_window_bonus,
        size_window_small_penalty=args.size_window_small_penalty,
        size_window_large_penalty=args.size_window_large_penalty,
        max_generation_chars=args.reward_max_fragment_chars,
        core_output_mode=args.core_output_mode,
        dummy_output_penalty=args.dummy_output_penalty,
        require_teacher_sem=args.require_teacher_sem,
        disable_counterfactual_teacher=args.disable_counterfactual_teacher,
    )
    stable_reward_wrapper = StableChemRLRewardWrapper(
        base_rewarder=rewarder,
        teacher_conf_gate=StableTeacherConfidenceGateConfig(
            enabled=stable_config.enable_teacher_confidence_gate,
            min_teacher_p_before=stable_config.min_teacher_p_before,
            low_conf_cf_weight=stable_config.low_conf_cf_weight,
        ),
        logger=logger,
    )
    chem_reward_model = build_reward_model_wrapper(
        deps,
        tokenizer=tokenizer,
        rewarder=rewarder,
    )

    final_output_dir = run_stable_decoded_chem_ppo_loop(
        deps=deps,
        args=args,
        stable_config=stable_config,
        actual_batch_size=actual_batch_size,
        policy_model=policy_model,
        reference_model=reference_model,
        value_model=value_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        chem_reward_model=chem_reward_model,
        stable_reward_wrapper=stable_reward_wrapper,
        output_dir=output_dir,
        logger=logger,
    )
    logger.info("Stable PPO training finished. Final checkpoint saved to %s", final_output_dir)


if __name__ == "__main__":
    main()
