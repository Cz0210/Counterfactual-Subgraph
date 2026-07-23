#!/usr/bin/env python3
"""Run the shared stable PPO core for Mutagenicity source 1 -> target 0."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_ppo import (  # noqa: E402
    _safe_git_commit,
    apply_config_overrides,
    apply_decoded_chem_generation_defaults,
    build_hf_dataset,
    build_quantized_base_model,
    build_tokenizer,
    build_value_model,
    collect_runtime_environment_debug,
    ensure_score_head_for_experimental_ppo,
    import_training_dependencies,
    resolve_projected_cf_reward_enabled,
    resolve_substructure_distance_reward_config,
)
from scripts.train_ppo_stable import (  # noqa: E402
    build_parser as build_stable_parser,
    build_stable_reward_wrapper,
    resolve_stable_config,
    run_stable_decoded_chem_ppo_loop,
)
from src.data.mutagenicity_continued_sft import (  # noqa: E402
    ensure_new_output_root,
    load_single_trainable_peft_adapter,
    write_csv_atomic,
    write_json_atomic,
)
from src.rewards.teacher_semantic import (  # noqa: E402
    TeacherSemanticScorer,
    require_teacher_semantic_scorer,
)
from src.train.mutagenicity_stable_ppo import (  # noqa: E402
    EXPECTED_TRAIN_ROWS,
    EXPECTED_VAL_ROWS,
    MutagenicityCounterfactualTeacherScorer,
    MutagenicityPPORunObserver,
    SOURCE_LABEL,
    TARGET_LABEL,
    audit_mutagenicity_ppo_models,
    build_parent_coverage_plan,
    deterministically_order_records,
    load_mutagenicity_ppo_records,
    validate_candidate_pool_schema,
    validate_policy_adapter_checkpoint,
    validate_train_val_isolation,
)
from src.utils.io import read_jsonl  # noqa: E402
from src.utils.logging_utils import (  # noqa: E402
    RunContext,
    configure_run_logger,
    write_runtime_manifest,
)


DEFAULT_DATA_ROOT = REPO_ROOT / "outputs/hpc/mutagenicity/final/sft_ppo_data_v1"
DEFAULT_TRAIN_CSV = DEFAULT_DATA_ROOT / "mutagenicity_ppo_prompts_train_label1.csv"
DEFAULT_VAL_CSV = DEFAULT_DATA_ROOT / "mutagenicity_ppo_prompts_val_label1.csv"
DEFAULT_BASE_MODEL = REPO_ROOT / "pretrained_models/ChemLLM-7B-Chat"
DEFAULT_POLICY_ADAPTER = (
    REPO_ROOT / "outputs/hpc/mutagenicity/final/sft_continued_v1_best"
)
DEFAULT_TOKENIZER_FALLBACK = (
    REPO_ROOT
    / "outputs/hpc/sft_checkpoints/"
    "sft_v3_hiv_20260508_resplit_lr2e4_seed7_fix_columns/checkpoint-500"
)
DEFAULT_TEACHER = (
    REPO_ROOT / "outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl"
)
DEFAULT_SMOKE_OUTPUT = REPO_ROOT / "outputs/hpc/mutagenicity/ppo_stable_v1_smoke"
DEFAULT_FULL_OUTPUT = REPO_ROOT / "outputs/hpc/mutagenicity/ppo_stable_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = build_stable_parser()
    parser.description = __doc__
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--base-model-path", type=Path, default=None)
    parser.add_argument("--policy-adapter-checkpoint", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument(
        "--tokenizer-fallback-path",
        type=Path,
        default=DEFAULT_TOKENIZER_FALLBACK,
    )
    parser.add_argument("--max-parents", type=int, default=0)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--rollout-batch-size", type=int, default=0)
    parser.add_argument("--expected-train-rows", type=int, default=EXPECTED_TRAIN_ROWS)
    parser.add_argument("--expected-val-rows", type=int, default=EXPECTED_VAL_ROWS)
    parser.add_argument("--source-label", type=int, default=SOURCE_LABEL)
    parser.add_argument("--target-label", type=int, default=TARGET_LABEL)
    parser.set_defaults(
        model_path=str(DEFAULT_BASE_MODEL),
        dataset_path=str(DEFAULT_TRAIN_CSV),
        val_dataset_path=str(DEFAULT_VAL_CSV),
        teacher_path=str(DEFAULT_TEACHER),
        oracle_path=str(DEFAULT_TEACHER),
        output_dir=str(DEFAULT_FULL_OUTPUT),
        default_parent_label=SOURCE_LABEL,
        only_positive=True,
        ppo_loop="decoded_chem",
        require_chemistry_reward_path=True,
        require_teacher_sem=True,
        log_unified_ppo_samples=True,
        stable_ppo_epochs=1,
        ppo_learning_rate=1e-6,
        ppo_clip_range=0.05,
        max_grad_norm=0.5,
        target_kl=0.30,
        hard_kl=0.80,
        enable_adaptive_kl=True,
        kl_penalty_init=0.05,
        kl_penalty_multiplier=1.5,
        reward_clip_min=-5.0,
        reward_clip_max=5.0,
        normalize_reward=True,
        normalize_advantage=True,
        enable_teacher_confidence_gate=True,
        min_teacher_p_before=0.5,
        low_conf_cf_weight=0.3,
        enable_stable_early_stop=False,
        save_best_checkpoint=True,
        enable_parent_projection=True,
        enable_projected_cf_reward=True,
        enable_substructure_distance_reward=True,
        substructure_distance_reward_weight=0.3,
        projection_penalty=1.0,
        enable_minimal_syntax_repair=True,
        enable_component_salvage=True,
        gen_temperature=0.5,
        gen_top_p=0.8,
        gen_do_sample=True,
    )
    return parser


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _has_tokenizer_assets(path: Path) -> bool:
    return any(
        (path / name).is_file()
        for name in (
            "tokenizer_config.json",
            "tokenizer.model",
            "tokenizer.json",
        )
    )


def resolve_tokenizer_path(
    *,
    explicit_path: Path | None,
    policy_adapter_checkpoint: Path,
    fallback_path: Path,
    base_model_path: Path,
) -> tuple[Path, str]:
    if explicit_path is not None:
        path = _resolve(explicit_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Tokenizer path does not exist: {path}")
        return path, "explicit"
    if _has_tokenizer_assets(policy_adapter_checkpoint):
        return policy_adapter_checkpoint, "policy_adapter_checkpoint"
    resolved_fallback = _resolve(fallback_path)
    if _has_tokenizer_assets(resolved_fallback):
        return resolved_fallback, "aids_checkpoint_tokenizer_fallback"
    if _has_tokenizer_assets(base_model_path):
        return base_model_path, "base_model"
    raise FileNotFoundError(
        "No tokenizer assets were found in the policy adapter, AIDS fallback, "
        f"or base model: {policy_adapter_checkpoint}, {resolved_fallback}, {base_model_path}"
    )


def _write_selected_validation_csv(
    path: Path,
    records: list[Any],
) -> None:
    rows = [
        {
            "molecule_id": record.molecule_id,
            "parent_smiles": record.parent_smiles,
            "label": record.label,
            "prompt": record.prompt,
        }
        for record in records
    ]
    write_csv_atomic(
        path,
        rows,
        ("molecule_id", "parent_smiles", "label", "prompt"),
    )


def _teacher_audit(
    *,
    teacher_scorer: TeacherSemanticScorer,
    teacher_path: Path,
    example_parent_smiles: str,
) -> dict[str, Any]:
    require_teacher_semantic_scorer(teacher_scorer, teacher_path=teacher_path)
    example = teacher_scorer.score_smiles(
        example_parent_smiles,
        label=SOURCE_LABEL,
        parent_smiles=example_parent_smiles,
    )
    if not example.get("teacher_result_ok"):
        raise ValueError(f"Mutagenicity RF teacher smoke score failed: {example}")
    if int(example.get("teacher_label")) != SOURCE_LABEL:
        raise ValueError(
            "Teacher-consistent PPO parent was not predicted as source label 1: "
            f"{example}"
        )
    return {
        "teacher_path": str(teacher_path),
        "teacher_available": bool(teacher_scorer.available),
        "teacher_format": teacher_scorer.teacher_format,
        "fingerprint_radius": teacher_scorer.fingerprint_radius,
        "fingerprint_bits": teacher_scorer.fingerprint_bits,
        "source_label": SOURCE_LABEL,
        "target_label": TARGET_LABEL,
        "strict_flip_definition": "pred_before==1_and_pred_after==0",
        "cf_drop_definition": "p1_before_minus_p1_after",
        "target_prob_gain_definition": "p0_after_minus_p0_before",
        "example_parent_pred": int(example["teacher_label"]),
        "example_parent_prob_1": float(example["teacher_prob"]),
        "teacher_audit_passed": True,
    }


def _load_single_adapter(
    *,
    deps: dict[str, Any],
    base_model_path: Path,
    adapter_checkpoint: Path,
    trust_remote_code: bool,
    local_files_only: bool,
    is_trainable: bool,
) -> tuple[Any, dict[str, Any]]:
    model, audit = load_single_trainable_peft_adapter(
        base_model_path=base_model_path,
        adapter_checkpoint=adapter_checkpoint,
        project_root=REPO_ROOT,
        base_model_loader=lambda resolved_base: build_quantized_base_model(
            deps,
            model_path=resolved_base,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            prepare_for_training=is_trainable,
        ),
        peft_model_class=deps["PeftModel"],
    )
    if not is_trainable:
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        audit = {
            **audit,
            "is_trainable": False,
            "frozen_after_single_adapter_load": True,
        }
    return model, audit


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args = apply_config_overrides(args, parser)
    args = apply_decoded_chem_generation_defaults(args)
    (
        args.enable_substructure_distance_reward,
        args.substructure_distance_reward_weight,
    ) = resolve_substructure_distance_reward_config(args)
    projected_cf_reward_enabled = resolve_projected_cf_reward_enabled(args)
    args.projected_cf_reward_enabled = projected_cf_reward_enabled

    if (int(args.source_label), int(args.target_label)) != (
        SOURCE_LABEL,
        TARGET_LABEL,
    ):
        raise ValueError("Mutagenicity stable PPO supports source=1, target=0 only")
    if args.ppo_loop != "decoded_chem":
        raise ValueError("Mutagenicity stable PPO requires --ppo-loop decoded_chem")

    mode = str(args.mode)
    train_csv = _resolve(args.train_csv or args.dataset_path)
    val_csv = _resolve(args.val_csv or args.val_dataset_path)
    base_model_path = _resolve(args.base_model_path or args.model_path)
    policy_adapter_checkpoint = _resolve(
        args.policy_adapter_checkpoint or DEFAULT_POLICY_ADAPTER
    )
    teacher_path = _resolve(args.teacher_path)
    output_root = _resolve(
        args.output_dir
        if args.output_dir != str(DEFAULT_FULL_OUTPUT) or mode == "full"
        else DEFAULT_SMOKE_OUTPUT
    )
    if mode == "smoke" and args.output_dir == str(DEFAULT_FULL_OUTPUT):
        output_root = DEFAULT_SMOKE_OUTPUT.resolve()

    validate_policy_adapter_checkpoint(policy_adapter_checkpoint)
    if not base_model_path.is_dir():
        raise FileNotFoundError(f"ChemLLM base model is missing: {base_model_path}")
    if not teacher_path.is_file():
        raise FileNotFoundError(f"Mutagenicity RF teacher is missing: {teacher_path}")
    output_root = ensure_new_output_root(output_root)

    train_all = load_mutagenicity_ppo_records(
        train_csv,
        expected_split="train",
        expected_count=int(args.expected_train_rows),
    )
    val_all = load_mutagenicity_ppo_records(
        val_csv,
        expected_split="val",
        expected_count=int(args.expected_val_rows),
    )
    isolation_audit = validate_train_val_isolation(train_all, val_all)

    if mode == "smoke":
        requested_parents = int(args.max_parents or 5)
        if requested_parents != 5:
            raise ValueError("Mutagenicity PPO smoke requires exactly 5 parents")
        train_selected = deterministically_order_records(
            train_all, seed=int(args.seed), limit=5
        )
        val_selected = deterministically_order_records(
            val_all, seed=int(args.seed) + 1, limit=min(32, len(val_all))
        )
        rollout_batch_size = int(args.rollout_batch_size or 1)
        if rollout_batch_size != 1:
            raise ValueError("Mutagenicity PPO smoke requires rollout_batch_size=1")
    else:
        requested_parents = int(args.max_parents or len(train_all))
        if requested_parents != EXPECTED_TRAIN_ROWS:
            raise ValueError(
                "First Mutagenicity PPO full run must use all 1448 train parents"
            )
        train_selected = deterministically_order_records(
            train_all, seed=int(args.seed)
        )
        val_selected = deterministically_order_records(
            val_all, seed=int(args.seed) + 1
        )
        rollout_batch_size = int(args.rollout_batch_size or args.batch_size or 64)

    coverage_plan = build_parent_coverage_plan(
        num_dataset_rows=len(train_selected),
        rollout_batch_size=rollout_batch_size,
        sampler_seed=int(args.seed),
        max_updates=(int(args.max_updates) if int(args.max_updates) > 0 else None),
    )
    if mode == "smoke" and coverage_plan.max_updates != 5:
        raise ValueError("Mutagenicity PPO smoke requires max_updates=5")
    if mode == "full" and coverage_plan.max_updates != coverage_plan.updates_per_epoch:
        raise ValueError(
            "First Mutagenicity PPO full run is fixed to one equivalent epoch: "
            f"max_updates={coverage_plan.max_updates} "
            f"updates_per_epoch={coverage_plan.updates_per_epoch}"
        )

    args.dataset_path = str(train_csv)
    args.output_dir = str(output_root)
    args.candidate_pool_path = str(output_root / "candidate_pool.jsonl")
    args.model_path = str(base_model_path)
    args.sft_lora_path = str(policy_adapter_checkpoint)
    args.oracle_path = str(teacher_path)
    args.teacher_path = str(teacher_path)
    args.default_parent_label = SOURCE_LABEL
    args.batch_size = coverage_plan.rollout_batch_size
    args.max_steps = coverage_plan.max_updates
    args.max_prompt_examples = 0

    selected_val_path = output_root / "validation_input.csv"
    _write_selected_validation_csv(selected_val_path, val_selected)
    args.val_dataset_path = str(selected_val_path)
    args.eval_num_samples = len(val_selected)
    if int(args.eval_every_steps) <= 0:
        args.eval_every_steps = 1 if mode == "smoke" else 5
    if int(args.save_steps) <= 0 or int(args.save_steps) > coverage_plan.max_updates:
        args.save_steps = 1 if mode == "smoke" else 5
    stable_config = resolve_stable_config(args)

    tokenizer_path, tokenizer_source = resolve_tokenizer_path(
        explicit_path=args.tokenizer_path,
        policy_adapter_checkpoint=policy_adapter_checkpoint,
        fallback_path=_resolve(args.tokenizer_fallback_path),
        base_model_path=base_model_path,
    )
    resolved_config = {
        "mode": mode,
        "git_commit": _safe_git_commit(),
        "base_model_path": str(base_model_path),
        "policy_adapter_checkpoint": str(policy_adapter_checkpoint),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_source": tokenizer_source,
        "teacher_path": str(teacher_path),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "validation_input_csv": str(selected_val_path),
        "output_root": str(output_root),
        "source_label": SOURCE_LABEL,
        "target_label": TARGET_LABEL,
        "strict_flip_definition": "pred_before==1_and_pred_after==0",
        "cf_drop_definition": "p1_before_minus_p1_after",
        "rollout_batch_size": coverage_plan.rollout_batch_size,
        "samples_per_update": coverage_plan.samples_per_update,
        "updates_per_epoch": coverage_plan.updates_per_epoch,
        "max_updates": coverage_plan.max_updates,
        "ppo_epochs": stable_config.ppo_epochs,
        "learning_rate": stable_config.ppo_learning_rate,
        "clip_range": stable_config.ppo_clip_range,
        "max_grad_norm": stable_config.max_grad_norm,
        "target_kl": stable_config.target_kl,
        "hard_kl": stable_config.hard_kl,
        "adaptive_kl": stable_config.enable_adaptive_kl,
        "reward_clip_min": stable_config.reward_clip_min,
        "reward_clip_max": stable_config.reward_clip_max,
        "eval_every_steps": stable_config.eval_every_steps,
        "save_steps": int(args.save_steps),
        "sampler_seed": int(args.seed),
        "shuffle_enabled": True,
        "sampling_with_replacement": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    dataset_manifest = {
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "num_train_rows_full": len(train_all),
        "num_val_rows_full": len(val_all),
        "num_train_rows_selected": len(train_selected),
        "num_val_rows_selected": len(val_selected),
        "num_unique_train_parents": len(
            {record.molecule_id for record in train_selected}
        ),
        "num_unique_val_parents": len(
            {record.molecule_id for record in val_selected}
        ),
        "train_parent_order": [record.molecule_id for record in train_selected],
        "sampling": "deterministic_sha256_order_without_replacement",
        "isolation_audit": isolation_audit,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    write_json_atomic(output_root / "resolved_config.json", resolved_config)
    write_json_atomic(output_root / "dataset_manifest.json", dataset_manifest)
    write_runtime_manifest(
        output_root / "runtime_manifest.json",
        {
            "run_name": f"mutagenicity_ppo_stable_{mode}",
            "resolved_config": resolved_config,
            "stable_config": asdict(stable_config),
        },
    )

    logger = configure_run_logger(
        "train_mutagenicity_ppo_stable",
        context=RunContext(
            run_name=f"mutagenicity_ppo_stable_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            output_dir=output_root,
            stage="mutagenicity_ppo_stable",
            seed=int(args.seed),
        ),
        log_dir=output_root / "logs",
    )
    logger.info("Runtime environment: %s", collect_runtime_environment_debug())
    logger.info(
        "[MUTAGENICITY_PPO_TEACHER_CONFIG] teacher_path=%s source_label=1 "
        "target_label=0 strict_flip_definition=pred_before==1_and_pred_after==0 "
        "cf_drop_definition=p1_before_minus_p1_after",
        teacher_path,
    )
    logger.info(
        "[MUTAGENICITY_PPO_COVERAGE_PLAN] num_dataset_rows=%s "
        "samples_per_update=%s updates_per_epoch=%s max_updates=%s "
        "shuffle_enabled=true sampling_with_replacement=false",
        coverage_plan.num_dataset_rows,
        coverage_plan.samples_per_update,
        coverage_plan.updates_per_epoch,
        coverage_plan.max_updates,
    )

    deps = import_training_dependencies()
    deps["set_seed"](int(args.seed))
    torch = deps["torch"]
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = build_tokenizer(
        deps,
        model_path=tokenizer_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    train_examples = [record.to_prompt_example() for record in train_selected]
    train_dataset = build_hf_dataset(deps, tokenizer, train_examples)

    policy_model, policy_load_audit = _load_single_adapter(
        deps=deps,
        base_model_path=base_model_path,
        adapter_checkpoint=policy_adapter_checkpoint,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        is_trainable=True,
    )
    reference_model, reference_load_audit = _load_single_adapter(
        deps=deps,
        base_model_path=base_model_path,
        adapter_checkpoint=policy_adapter_checkpoint,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        is_trainable=False,
    )
    value_model = build_value_model(
        deps,
        model_path=base_model_path,
        tokenizer=tokenizer,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    value_model = ensure_score_head_for_experimental_ppo(
        value_model, "mutagenicity_stable_value_model"
    )
    model_audit = audit_mutagenicity_ppo_models(
        policy_model=policy_model,
        reference_model=reference_model,
        value_model=value_model,
        base_model_path=base_model_path,
        policy_adapter_checkpoint=policy_adapter_checkpoint,
    )
    model_audit["policy_loading"] = policy_load_audit
    model_audit["reference_loading"] = reference_load_audit
    write_json_atomic(output_root / "model_audit.json", model_audit)
    logger.info(
        "[MUTAGENICITY_PPO_MODEL_AUDIT] base_model_path=%s "
        "policy_adapter_checkpoint=%s policy_adapter_names=%s "
        "active_adapters=%s policy_trainable_params=%s "
        "reference_trainable_params=%s base_params_trainable=%s "
        "value_head_trainable_params=%s model_audit_passed=true",
        base_model_path,
        policy_adapter_checkpoint,
        model_audit["policy_adapter_names"],
        model_audit["active_adapters"],
        model_audit["policy_trainable_params"],
        model_audit["reference_trainable_params"],
        model_audit["base_params_trainable"],
        model_audit["value_head_trainable_params"],
    )

    teacher_scorer = TeacherSemanticScorer(
        teacher_path=teacher_path,
        device="cpu",
        logger=logger,
    )
    teacher_audit = _teacher_audit(
        teacher_scorer=teacher_scorer,
        teacher_path=teacher_path,
        example_parent_smiles=train_selected[0].parent_smiles,
    )
    write_json_atomic(output_root / "teacher_audit.json", teacher_audit)
    counterfactual_teacher = MutagenicityCounterfactualTeacherScorer(
        teacher_path=teacher_path,
        device="cpu",
        logger=logger,
        flip_bonus=args.teacher_cf_flip_bonus,
        missing_penalty=args.teacher_sem_missing_penalty,
        teacher_scorer=teacher_scorer,
    )
    stable_reward_wrapper = build_stable_reward_wrapper(
        args=args,
        stable_config=stable_config,
        teacher_scorer=teacher_scorer,
        counterfactual_teacher_scorer=counterfactual_teacher,
        oracle_path=teacher_path,
        projected_cf_reward_enabled=projected_cf_reward_enabled,
        logger=logger,
    )
    observer = MutagenicityPPORunObserver(
        output_root=output_root,
        dataset_parent_ids=[record.molecule_id for record in train_selected],
        coverage_plan=coverage_plan,
        resolved_config=resolved_config,
        dataset_manifest=dataset_manifest,
        require_full_coverage=True,
    )
    final_output_dir = run_stable_decoded_chem_ppo_loop(
        deps=deps,
        args=args,
        stable_config=stable_config,
        actual_batch_size=coverage_plan.rollout_batch_size,
        policy_model=policy_model,
        reference_model=reference_model,
        value_model=value_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        stable_reward_wrapper=stable_reward_wrapper,
        output_dir=output_root,
        logger=logger,
        run_observer=observer,
        stop_after_dataset_exhausted=True,
    )

    candidate_rows = read_jsonl(output_root / "candidate_pool.jsonl")
    validate_candidate_pool_schema(candidate_rows)
    logger.info(
        "Mutagenicity stable PPO finished. Final checkpoint=%s candidates=%s",
        final_output_dir,
        len(candidate_rows),
    )
    marker = (
        "[MUTAGENICITY_STABLE_PPO_SMOKE_OK]"
        if mode == "smoke"
        else "[MUTAGENICITY_STABLE_PPO_FULL_OK]"
    )
    print(marker, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
