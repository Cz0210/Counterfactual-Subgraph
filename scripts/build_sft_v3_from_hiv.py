#!/usr/bin/env python3
"""Build a higher-quality SFT v3 dataset directly from raw HIV.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = REPO_ROOT / "data" / "raw" / "AIDS" / "HIV.csv"
DEFAULT_TRAIN_JSONL = REPO_ROOT / "data" / "sft_v3_hiv_train.jsonl"
DEFAULT_VAL_JSONL = REPO_ROOT / "data" / "sft_v3_hiv_val.jsonl"

from src.data.sft_v3_builder import SFTV3BuilderConfig, build_and_write_sft_v3_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Accepted for HPC wrapper compatibility. The current builder uses explicit CLI overrides only.",
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_CSV),
        help="Path to the raw HIV.csv file.",
    )
    parser.add_argument(
        "--train-output",
        default=str(DEFAULT_TRAIN_JSONL),
        help="Output JSONL path for the train split.",
    )
    parser.add_argument(
        "--val-output",
        default=str(DEFAULT_VAL_JSONL),
        help="Output JSONL path for the validation split.",
    )
    parser.add_argument(
        "--positive-label",
        default=1,
        help="Raw label value that should be treated as the positive class.",
    )
    parser.add_argument(
        "--neg-pos-ratio",
        type=float,
        default=2.0,
        help="Target negative:positive ratio for the sampled parent pool.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for sampling, candidate ordering, and split assignment.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio after candidate selection.",
    )
    parser.add_argument(
        "--max-parents",
        type=int,
        default=None,
        help="Optional cap on how many parent molecules are processed. Useful for smoke runs.",
    )
    parser.add_argument(
        "--min-atom-ratio",
        type=float,
        default=0.10,
        help="Minimum fragment atom_ratio retained in the SFT dataset.",
    )
    parser.add_argument(
        "--max-atom-ratio",
        type=float,
        default=0.55,
        help="Maximum fragment atom_ratio retained in the SFT dataset.",
    )
    parser.add_argument(
        "--min-frag-atoms",
        type=int,
        default=3,
        help="Minimum fragment atom count retained in the SFT dataset.",
    )
    parser.add_argument(
        "--max-frag-atoms",
        type=int,
        default=30,
        help="Maximum fragment atom count retained in the SFT dataset.",
    )
    parser.add_argument(
        "--oracle-path",
        default=None,
        help="Optional classifier/oracle bundle used for weak candidate ranking.",
    )
    parser.add_argument(
        "--use-oracle-ranking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If an oracle path is provided, rank filtered candidates with counterfactual weak supervision.",
    )
    parser.add_argument(
        "--max-candidates-per-parent",
        type=int,
        default=160,
        help="Upper bound on parent-derived candidate fragments enumerated per molecule.",
    )
    parser.add_argument(
        "--include-label-in-prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to include ORIGINAL_LABEL in the constructed instruction prompt.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SFTV3BuilderConfig(
        positive_label=args.positive_label,
        neg_pos_ratio=args.neg_pos_ratio,
        seed=args.seed,
        val_ratio=args.val_ratio,
        max_parents=args.max_parents,
        min_atom_ratio=args.min_atom_ratio,
        max_atom_ratio=args.max_atom_ratio,
        min_frag_atoms=args.min_frag_atoms,
        max_frag_atoms=args.max_frag_atoms,
        oracle_path=args.oracle_path,
        use_oracle_ranking=bool(args.use_oracle_ranking),
        max_candidates_per_parent=args.max_candidates_per_parent,
        include_label_in_prompt=bool(args.include_label_in_prompt),
    )
    artifacts = build_and_write_sft_v3_dataset(
        input_csv=args.input_csv,
        train_output=args.train_output,
        val_output=args.val_output,
        config=config,
    )

    print("SFT v3 HIV dataset build completed.")
    print(f"Train JSONL: {artifacts.train_output}")
    print(f"Val JSONL: {artifacts.val_output}")
    print(f"Train summary JSON: {artifacts.train_summary_path}")
    print(f"Val summary JSON: {artifacts.val_summary_path}")
    print(f"Report TXT: {artifacts.report_path}")
    print(f"Dropped summary JSON: {artifacts.dropped_summary_path}")
    print(
        "Counts: "
        f"train={artifacts.train_count} "
        f"val={artifacts.val_count} "
        f"total={artifacts.total_count}"
    )


if __name__ == "__main__":
    main()
