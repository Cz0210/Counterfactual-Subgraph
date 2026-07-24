#!/usr/bin/env python3
"""Evaluate Mutagenicity generators on the full teacher-correct validation cohort."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_mutagenicity_ppo_stable import (  # noqa: E402
    _load_single_adapter,
    build_parser as build_mutagenicity_ppo_parser,
)
from scripts.train_ppo import (  # noqa: E402
    _decode_text_batch,
    _extract_fragment_from_text,
    apply_decoded_chem_generation_defaults,
    build_quantized_base_model,
    build_tokenizer,
    extract_fragment_smiles,
    import_training_dependencies,
    resolve_projected_cf_reward_enabled,
)
from scripts.train_ppo_stable import (  # noqa: E402
    build_stable_reward_wrapper,
    isolated_generation_rng,
    resolve_stable_config,
)
from src.eval.mutagenicity_generator import (  # noqa: E402
    aggregate_generator_rows,
    best_task_checkpoint_payload,
    checkpoint_step,
    compare_parent_difficulty,
    fragment_frequency_strict_summary,
    rank_checkpoints,
    stable_json_hash,
    summarize_strategy_failures,
    validation_cohort_hash,
)
from src.rewards.teacher_semantic import TeacherSemanticScorer  # noqa: E402
from src.train.mutagenicity_stable_ppo import (  # noqa: E402
    MutagenicityCounterfactualTeacherScorer,
    enrich_mutagenicity_candidate_row,
    load_mutagenicity_ppo_records,
)
from src.train.mutagenicity_fresh_sft import write_json_atomic  # noqa: E402


DEFAULT_VAL = Path(
    "outputs/hpc/mutagenicity/final/sft_ppo_data_v2/"
    "mutagenicity_ppo_prompts_val_label1_v2.csv"
)
DEFAULT_TEACHER = Path(
    "outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl"
)
DEFAULT_BASE = Path("pretrained_models/ChemLLM-7B-Chat")
GENERATOR_DETAIL_FIELDS = (
    "model_name",
    "model_path",
    "model_kind",
    "candidate_index",
    "generation_seed",
    "molecule_id",
    "parent_smiles",
    "prompt",
    "generated_text",
    "raw_fragment",
    "core_fragment",
    "final_fragment",
    "residual_smiles",
    "parse_ok",
    "valid",
    "connected",
    "direct_substructure",
    "final_substructure",
    "projection_used",
    "projection_failed",
    "oracle_ok",
    "pred_before",
    "pred_after",
    "prob_before_0",
    "prob_before_1",
    "prob_after_0",
    "prob_after_1",
    "cf_drop",
    "cf_flip",
    "target_prob_gain",
    "atom_ratio",
    "reward_total",
    "failure_tag",
    "invalid_detail",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=PATH_OR_PURE_BASE",
        help="Repeat for every base/SFT/PPO checkpoint to compare.",
    )
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--teacher-path", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-val-parents", type=int, default=260)
    parser.add_argument(
        "--cohort-split",
        choices=("train", "val"),
        default="val",
        help="Validation is checkpoint-selection eligible; train is diagnostic only.",
    )
    parser.add_argument(
        "--expected-parent-count",
        type=int,
        default=None,
        help="Expected rows for the selected cohort (defaults to --expected-val-parents).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidate-counts", default="1,4,8")
    parser.add_argument("--seeds", default="7,17,27,37,47,57,67,77")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--projection-max-atom-ratio", type=float, default=0.70)
    parser.add_argument("--atom-ratio-target", type=float, default=0.35)
    parser.add_argument(
        "--difficulty-models",
        default=None,
        metavar="SFT_NAME,PPO_NAME",
        help="Optionally write parent difficulty analysis from two Hit@1 models.",
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _parse_models(values: Sequence[str]) -> list[tuple[str, str]]:
    models: list[tuple[str, str]] = []
    names: set[str] = set()
    for value in values:
        name, separator, path = str(value).partition("=")
        if not separator or not name.strip() or not path.strip():
            raise ValueError(f"Invalid --model {value!r}; expected NAME=PATH_OR_PURE_BASE")
        name = name.strip()
        if name in names:
            raise ValueError(f"Duplicate model name: {name}")
        names.add(name)
        models.append((name, path.strip()))
    return models


def _parse_positive_ints(raw: str, *, field: str) -> list[int]:
    values = sorted({int(token.strip()) for token in raw.split(",") if token.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{field} must contain positive integers")
    return values


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or sorted({key for row in rows for key in row}))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _reward_stack(
    *,
    teacher_path: Path,
    logger: logging.Logger,
    projection_max_atom_ratio: float,
) -> Any:
    parser = build_mutagenicity_ppo_parser()
    reward_args = parser.parse_args([])
    reward_args = apply_decoded_chem_generation_defaults(reward_args)
    reward_args.teacher_path = str(teacher_path)
    reward_args.oracle_path = str(teacher_path)
    reward_args.default_parent_label = 1
    reward_args.enable_parent_projection = True
    reward_args.enable_projected_cf_reward = True
    reward_args.enable_substructure_distance_reward = True
    reward_args.projection_max_atom_ratio = float(projection_max_atom_ratio)
    stable_config = resolve_stable_config(reward_args)
    teacher = TeacherSemanticScorer(
        teacher_path=teacher_path,
        device="cpu",
        logger=logger,
    )
    counterfactual = MutagenicityCounterfactualTeacherScorer(
        teacher_path=teacher_path,
        device="cpu",
        logger=logger,
        flip_bonus=reward_args.teacher_cf_flip_bonus,
        missing_penalty=reward_args.teacher_sem_missing_penalty,
        teacher_scorer=teacher,
    )
    return build_stable_reward_wrapper(
        args=reward_args,
        stable_config=stable_config,
        teacher_scorer=teacher,
        counterfactual_teacher_scorer=counterfactual,
        oracle_path=teacher_path,
        projected_cf_reward_enabled=resolve_projected_cf_reward_enabled(reward_args),
        logger=logger,
    )


def _load_model(
    *,
    model_spec: str,
    deps: Mapping[str, Any],
    base_model: Path,
    trust_remote_code: bool,
    local_files_only: bool,
) -> tuple[Any, str, str]:
    if model_spec.upper() == "PURE_BASE":
        model = build_quantized_base_model(
            dict(deps),
            model_path=base_model,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            prepare_for_training=False,
        )
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        return model, "pure_base", str(base_model)
    checkpoint = _resolve(model_spec)
    model, _audit = _load_single_adapter(
        deps=dict(deps),
        base_model_path=base_model,
        adapter_checkpoint=checkpoint,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        is_trainable=False,
    )
    return model, "adapter_checkpoint", str(checkpoint)


def _evaluate_model(
    *,
    model_name: str,
    model_path: str,
    model_kind: str,
    model: Any,
    tokenizer: Any,
    records: Sequence[Any],
    reward_wrapper: Any,
    deps: Mapping[str, Any],
    seeds: Sequence[int],
    batch_size: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    torch = deps["torch"]
    model_device = next(model.parameters()).device
    model.eval()
    rows: list[dict[str, Any]] = []
    for candidate_index, seed in enumerate(seeds, start=1):
        for batch_start in range(0, len(records), batch_size):
            batch = records[batch_start : batch_start + batch_size]
            encoded = tokenizer(
                [record.prompt for record in batch],
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            encoded = {key: value.to(model_device) for key, value in encoded.items()}
            kwargs: dict[str, Any] = {
                **encoded,
                "max_new_tokens": int(args.max_new_tokens),
                "do_sample": bool(args.do_sample),
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": False,
            }
            if args.do_sample:
                kwargs["temperature"] = float(args.temperature)
                kwargs["top_p"] = float(args.top_p)
                if int(args.top_k) > 0:
                    kwargs["top_k"] = int(args.top_k)
            batch_seed = int(seed) + batch_start * 9973
            with isolated_generation_rng(
                torch=torch,
                seed=batch_seed,
                model_device=model_device,
            ):
                with torch.no_grad():
                    generated_ids = model.generate(**kwargs)
            response_ids = generated_ids[:, encoded["input_ids"].shape[1] :]
            response_texts = _decode_text_batch(
                tokenizer, response_ids.detach().cpu().tolist(), torch=torch
            )
            full_texts = _decode_text_batch(
                tokenizer, generated_ids.detach().cpu().tolist(), torch=torch
            )
            fragments: list[str] = []
            for index, record in enumerate(batch):
                fragment = _extract_fragment_from_text(full_texts[index], record.prompt)
                fragments.append(fragment or extract_fragment_smiles(response_texts[index]))
            reward_tensor, reward_logs = reward_wrapper.compute_rewards_from_decoded(
                parent_smiles=[record.parent_smiles for record in batch],
                generated_fragments=fragments,
                raw_outputs=response_texts,
                labels=[1] * len(batch),
                metas=[
                    {
                        "id": record.molecule_id,
                        "index": record.row_index,
                        "prompt": record.prompt,
                    }
                    for record in batch
                ],
                device=model_device,
                step_index=0,
            )
            del reward_tensor
            for index, reward_log in enumerate(reward_logs):
                enriched = enrich_mutagenicity_candidate_row(
                    reward_log,
                    molecule_id=batch[index].molecule_id,
                    parent_smiles=batch[index].parent_smiles,
                    prompt=batch[index].prompt,
                    generated_text=response_texts[index],
                    generated_fragment=fragments[index],
                    global_step=0,
                )
                enriched.update(
                    {
                        "model_name": model_name,
                        "model_path": model_path,
                        "model_kind": model_kind,
                        "candidate_index": candidate_index,
                        "generation_seed": int(seed),
                    }
                )
                rows.append(enriched)
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = _parse_models(args.model)
    candidate_counts = _parse_positive_ints(args.candidate_counts, field="candidate-counts")
    seeds = _parse_positive_ints(args.seeds, field="seeds")
    max_candidates = max(candidate_counts)
    if len(seeds) < max_candidates:
        raise ValueError(
            f"At least {max_candidates} seeds are required for Hit@{max_candidates}"
        )
    val_csv = _resolve(args.val_csv)
    teacher_path = _resolve(args.teacher_path)
    base_model = _resolve(args.base_model_path)
    tokenizer_path = _resolve(args.tokenizer_path or base_model)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Generator eval output is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for required in (val_csv, teacher_path, base_model, tokenizer_path):
        if not required.exists():
            raise FileNotFoundError(f"Generator eval input does not exist: {required}")

    expected_parent_count = int(
        args.expected_parent_count
        if args.expected_parent_count is not None
        else args.expected_val_parents
    )
    records = load_mutagenicity_ppo_records(
        val_csv,
        expected_split=str(args.cohort_split),
        expected_count=expected_parent_count,
    )
    if (
        args.cohort_split == "val"
        and len(records) != 260
        and expected_parent_count == 260
    ):
        raise ValueError("Task checkpoint selection requires the complete 260-parent val cohort")
    cohort_hash = validation_cohort_hash(record.molecule_id for record in records)
    decoding = {
        "candidate_counts": candidate_counts,
        "seeds": seeds[:max_candidates],
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": bool(args.do_sample),
        "projection_enabled": True,
        "projection_max_atom_ratio": float(args.projection_max_atom_ratio),
        "teacher_path": str(teacher_path),
        "source_label": 1,
        "target_label": 0,
        "strict_flip_definition": "pred_before==1_and_pred_after==0",
        "cf_drop_definition": "p1_before_minus_p1_after",
    }
    decoding_hash = stable_json_hash(decoding)
    run_config = {
        "val_csv": str(val_csv),
        "cohort_split": str(args.cohort_split),
        "num_cohort_parents": len(records),
        "num_val_parents": len(records),
        "validation_cohort_hash": cohort_hash,
        "base_model_path": str(base_model),
        "tokenizer_path": str(tokenizer_path),
        "teacher_path": str(teacher_path),
        "models": [{"name": name, "spec": spec} for name, spec in models],
        "decoding": decoding,
        "decoding_config_hash": decoding_hash,
        "checkpoint_selection_eligible": args.cohort_split == "val",
        "calibration_or_test_loaded": False,
    }
    write_json_atomic(output_dir / "run_config.json", run_config)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("mutagenicity_generator_eval")
    deps = import_training_dependencies()
    tokenizer = build_tokenizer(
        deps,
        model_path=tokenizer_path,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )
    reward_wrapper = _reward_stack(
        teacher_path=teacher_path,
        logger=logger,
        projection_max_atom_ratio=args.projection_max_atom_ratio,
    )

    all_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    hit1_rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for model_name, model_spec in models:
        model, model_kind, model_path = _load_model(
            model_spec=model_spec,
            deps=deps,
            base_model=base_model,
            trust_remote_code=bool(args.trust_remote_code),
            local_files_only=bool(args.local_files_only),
        )
        model_rows = _evaluate_model(
            model_name=model_name,
            model_path=model_path,
            model_kind=model_kind,
            model=model,
            tokenizer=tokenizer,
            records=records,
            reward_wrapper=reward_wrapper,
            deps=deps,
            seeds=seeds[:max_candidates],
            batch_size=max(1, int(args.batch_size)),
            args=args,
        )
        all_rows.extend(model_rows)
        hit1_rows_by_model[model_name] = [
            row for row in model_rows if int(row["candidate_index"]) == 1
        ]
        for count in candidate_counts:
            prefix_rows = [
                row for row in model_rows if int(row["candidate_index"]) <= count
            ]
            metrics = aggregate_generator_rows(
                prefix_rows,
                num_parents=len(records),
                num_candidates_per_parent=count,
            )
            metrics.update(
                {
                    "model_name": model_name,
                    "model_path": model_path,
                    "model_kind": model_kind,
                    "checkpoint": model_path,
                    "step": checkpoint_step(model_path),
                    "validation_cohort_hash": cohort_hash,
                    "decoding_config_hash": decoding_hash,
                }
            )
            metric_rows.append(metrics)
        del model
        if deps["torch"].cuda.is_available():
            deps["torch"].cuda.empty_cache()

    _write_csv(output_dir / "generation_samples.csv", all_rows, GENERATOR_DETAIL_FIELDS)
    _write_csv(
        output_dir / "fragment_frequency_strict_summary.csv",
        fragment_frequency_strict_summary(all_rows),
    )
    _write_csv(output_dir / "generator_metrics.csv", metric_rows)
    write_json_atomic(
        output_dir / "generator_metrics.json",
        {"run_config": run_config, "metrics": metric_rows},
    )
    hit1_metrics = [
        row for row in metric_rows if int(row["num_candidates_per_parent"]) == 1
    ]
    ranked = rank_checkpoints(
        hit1_metrics,
        atom_ratio_target=float(args.atom_ratio_target),
    )
    _write_csv(output_dir / "checkpoint_ranking.csv", ranked)
    best: dict[str, Any] | None = None
    if args.cohort_split == "val":
        best = best_task_checkpoint_payload(
            ranked,
            cohort_hash=cohort_hash,
            decoding_config_hash=decoding_hash,
        )
        write_json_atomic(output_dir / "best_task_checkpoint.json", best)

    if args.difficulty_models:
        sft_name, separator, ppo_name = args.difficulty_models.partition(",")
        if not separator or sft_name not in hit1_rows_by_model or ppo_name not in hit1_rows_by_model:
            raise ValueError(
                "--difficulty-models must name two evaluated models as SFT_NAME,PPO_NAME"
            )
        difficulty_rows, difficulty_summary = compare_parent_difficulty(
            hit1_rows_by_model[sft_name],
            hit1_rows_by_model[ppo_name],
        )
        _write_csv(output_dir / "parent_difficulty.csv", difficulty_rows)
        write_json_atomic(
            output_dir / "hard_parent_summary.json", difficulty_summary
        )
        _write_csv(
            output_dir / "strategy_failure_summary.csv",
            summarize_strategy_failures(difficulty_rows),
        )

    report = [
        "# Mutagenicity Generator Evaluation",
        "",
        f"- Cohort split/parents: {args.cohort_split} / {len(records)}",
        f"- Cohort hash: `{cohort_hash}`",
        f"- Decoding hash: `{decoding_hash}`",
        "- Strict flip: `pred_before==1 and pred_after==0`",
        "- CFDrop: `p1_before-p1_after`",
        (
            f"- Task-best model: `{best['checkpoint']}`"
            if best is not None
            else "- Task-best model: not produced (train diagnostic cohort)"
        ),
        "- Calibration/test loaded: false",
        "",
        "Checkpoint ranking is lexicographic by strict flip, CFDrop, final "
        "substructure, parseability, atom-ratio deviation, and duplicate rate.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("[MUTAGENICITY_GENERATOR_EVAL_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
