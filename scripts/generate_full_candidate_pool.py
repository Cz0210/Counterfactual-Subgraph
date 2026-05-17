#!/usr/bin/env python3
"""Generate a full candidate_pool.jsonl from an SFT or PPO checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.full_candidate_pool import (  # noqa: E402
    FullPoolGenerationConfig,
    generate_full_candidate_pool,
)


DEFAULT_DATASET = (
    REPO_ROOT
    / "outputs"
    / "hpc"
    / "sft_v3_hiv_runs"
    / "sft_v3_hiv_20260508_resplit"
    / "dataset"
    / "sft_v3_hiv_ppo_prompts_train_label1.csv"
)
DEFAULT_BASE_MODEL = REPO_ROOT / "pretrained_models" / "ChemLLM-7B-Chat"
DEFAULT_SFT_LORA = (
    REPO_ROOT
    / "outputs"
    / "hpc"
    / "sft_checkpoints"
    / "sft_v3_hiv_20260508_resplit_lr2e4_seed7_fix_columns"
    / "checkpoint-500"
)
DEFAULT_SUMMARY = REPO_ROOT / "outputs" / "hpc" / "full_candidate_pools" / "generation_summary.json"


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Could not parse boolean value: {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config path for HPC wrapper parity. The script uses explicit CLI paths.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Ignored dotted overrides kept only for Slurm wrapper parity.",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(DEFAULT_DATASET),
        help="Full label=1 PPO prompt CSV / JSONL used for offline candidate generation.",
    )
    parser.add_argument(
        "--base-model-path",
        default=str(DEFAULT_BASE_MODEL),
        help="Local ChemLLM base model path.",
    )
    parser.add_argument(
        "--sft-lora-path",
        default=str(DEFAULT_SFT_LORA),
        help="SFT LoRA checkpoint path used for the SFT-only baseline and PPO fallback inspection.",
    )
    parser.add_argument(
        "--ppo-checkpoint-path",
        default="",
        help="Optional PPO checkpoint directory. When set, load the resolved PPO adapter for inference.",
    )
    parser.add_argument(
        "--teacher-path",
        required=True,
        help="Teacher / oracle bundle path. Current pipeline uses the same AIDS RF bundle for oracle and teacher scoring.",
    )
    parser.add_argument(
        "--out-jsonl",
        required=True,
        help="Path to the generated candidate_pool JSONL.",
    )
    parser.add_argument(
        "--out-summary-json",
        default=str(DEFAULT_SUMMARY),
        help="Path to the machine-readable generation summary JSON.",
    )
    parser.add_argument("--label-col", default="label", help="Preferred label column.")
    parser.add_argument("--smiles-col", default="parent_smiles", help="Preferred smiles column.")
    parser.add_argument("--target-label", type=int, default=1, help="Only keep this parent label.")
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=4,
        help="How many candidates to sample per parent molecule.",
    )
    parser.add_argument(
        "--generation-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--generation-top-p",
        type=float,
        default=0.9,
        help="Sampling top-p.",
    )
    parser.add_argument(
        "--generation-do-sample",
        type=_parse_bool,
        default=True,
        help="Whether to sample during generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Maximum generated fragment length in tokens.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Prompt batch size.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument(
        "--enable-parent-projection",
        action="store_true",
        help="Reuse parent projection in reward / evaluation.",
    )
    parser.add_argument(
        "--enable-projected-cf-reward",
        action="store_true",
        help="Allow projected fragments to receive counterfactual reward when valid.",
    )
    parser.add_argument(
        "--enable-substructure-distance-reward",
        action="store_true",
        help="Enable nearest-parent-subgraph distance reward.",
    )
    parser.add_argument(
        "--substructure-distance-reward-weight",
        type=float,
        default=0.3,
        help="Weight for nearest-parent-subgraph distance reward.",
    )
    parser.add_argument(
        "--projection-penalty",
        type=float,
        default=1.0,
        help="Penalty applied after successful non-direct projection.",
    )
    parser.add_argument(
        "--enable-minimal-syntax-repair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable syntax-only repair before hard parse failure.",
    )
    parser.add_argument(
        "--enable-component-salvage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable disconnected-component salvage.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional parent limit for debugging. 0 means full dataset.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = generate_full_candidate_pool(
        dataset_path=args.dataset_path,
        base_model_path=args.base_model_path,
        sft_lora_path=args.sft_lora_path,
        ppo_checkpoint_path=args.ppo_checkpoint_path or None,
        teacher_path=args.teacher_path,
        out_jsonl=args.out_jsonl,
        out_summary_json=args.out_summary_json,
        config=FullPoolGenerationConfig(
            label_col=str(args.label_col),
            smiles_col=str(args.smiles_col),
            target_label=int(args.target_label),
            num_return_sequences=int(args.num_return_sequences),
            generation_temperature=float(args.generation_temperature),
            generation_top_p=float(args.generation_top_p),
            generation_do_sample=bool(args.generation_do_sample),
            max_new_tokens=int(args.max_new_tokens),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            enable_parent_projection=bool(args.enable_parent_projection),
            enable_projected_cf_reward=bool(args.enable_projected_cf_reward),
            enable_substructure_distance_reward=bool(
                args.enable_substructure_distance_reward
            ),
            substructure_distance_reward_weight=float(
                args.substructure_distance_reward_weight
            ),
            projection_penalty=float(args.projection_penalty),
            enable_minimal_syntax_repair=bool(args.enable_minimal_syntax_repair),
            enable_component_salvage=bool(args.enable_component_salvage),
            limit=int(args.limit),
        ),
    )
    outputs = summary["outputs"]
    model_load = summary["model_load"]
    print(f"dataset_path: {Path(args.dataset_path).expanduser().resolve()}")
    print(f"out_jsonl: {Path(args.out_jsonl).expanduser().resolve()}")
    print(f"out_summary_json: {Path(args.out_summary_json).expanduser().resolve()}")
    print(f"load_mode: {model_load['load_mode']}")
    print(f"adapter_path: {model_load['adapter_path']}")
    print(f"num_rows: {outputs['num_rows']}")
    print(f"num_unique_parents: {outputs['num_unique_parents']}")
    print(f"parse_ok_rate: {outputs['parse_ok_rate']}")
    print(f"final_substructure_rate: {outputs['final_substructure_rate']}")
    print(f"cf_flip_rate: {outputs['cf_flip_rate']}")


if __name__ == "__main__":
    main()
