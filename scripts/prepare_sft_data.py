#!/usr/bin/env python3
"""Prepare balanced SFT data for capped or core-only fragment targets."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = REPO_ROOT / "data" / "raw" / "AIDS" / "HIV.csv"
DEFAULT_TRAIN_JSONL = REPO_ROOT / "data" / "sft_train.jsonl"
DEFAULT_VAL_JSONL = REPO_ROOT / "data" / "sft_val.jsonl"
DEFAULT_CORE_TRAIN_JSONL = REPO_ROOT / "data" / "sft_v3_core_train.jsonl"
DEFAULT_CORE_VAL_JSONL = REPO_ROOT / "data" / "sft_v3_core_val.jsonl"
DEFAULT_CORE_AUDIT_JSON = REPO_ROOT / "outputs" / "sft_v3_core_audit.json"

from src.data.sft_preparation import (
    build_core_sft_audit_payload,
    filter_valid_hiv_records,
    label_ratio,
    load_hiv_dataframe,
    prepare_balanced_sft_examples,
    save_audit_json,
    save_sft_jsonl,
    split_examples,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Accepted for HPC wrapper compatibility. The current data builder uses explicit CLI overrides only.",
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_CSV),
        help="Path to the raw HIV CSV file.",
    )
    parser.add_argument(
        "--train-output",
        default=str(DEFAULT_TRAIN_JSONL),
        help="Path to the train JSONL output.",
    )
    parser.add_argument(
        "--val-output",
        default=str(DEFAULT_VAL_JSONL),
        help="Path to the validation JSONL output.",
    )
    parser.add_argument(
        "--audit-output",
        default=str(DEFAULT_CORE_AUDIT_JSON),
        help="Audit JSON path used by core-only dataset builds.",
    )
    parser.add_argument(
        "--negative-sample-size",
        type=int,
        default=3700,
        help="Accepted for backward compatibility. The v3 builder now fills negatives to the requested total_examples automatically.",
    )
    parser.add_argument(
        "--target-examples",
        type=int,
        default=5000,
        help="Target number of successful SFT examples to build.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=4500,
        help="Number of records to save into the train JSONL.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=500,
        help="Number of records to save into the validation JSONL.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for sampling, shuffling, and fragment selection.",
    )
    parser.add_argument(
        "--min-real-atoms",
        type=int,
        default=4,
        help="Minimum number of non-dummy atoms required in a raw capped fragment before core normalization.",
    )
    parser.add_argument(
        "--max-cut-attempts",
        type=int,
        default=24,
        help="How many random cut attempts to try per molecule.",
    )
    parser.add_argument(
        "--target-format",
        choices=("capped", "core"),
        default="capped",
        help="Whether response/output should stay capped with dummy atoms or be converted to core_no_dummy targets.",
    )
    parser.add_argument(
        "--no-dummy-target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Alias for --target-format core.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.train_size + args.val_size != args.target_examples:
        raise SystemExit(
            "--train-size + --val-size must equal --target-examples "
            f"({args.train_size} + {args.val_size} != {args.target_examples})."
        )

    target_format = "core" if args.no_dummy_target else str(args.target_format)
    train_output = Path(args.train_output).expanduser().resolve()
    val_output = Path(args.val_output).expanduser().resolve()
    audit_output = Path(args.audit_output).expanduser().resolve()
    if target_format == "core":
        if train_output == DEFAULT_TRAIN_JSONL.resolve():
            train_output = DEFAULT_CORE_TRAIN_JSONL.resolve()
        if val_output == DEFAULT_VAL_JSONL.resolve():
            val_output = DEFAULT_CORE_VAL_JSONL.resolve()

    input_csv = Path(args.input_csv).expanduser().resolve()
    dataframe = load_hiv_dataframe(input_csv)
    valid_records = filter_valid_hiv_records(dataframe)
    if valid_records.empty:
        raise SystemExit("No valid molecules remained after RDKit filtering.")

    examples, summary = prepare_balanced_sft_examples(
        valid_records,
        total_examples=args.target_examples,
        seed=args.seed,
        target_format=target_format,
        min_real_atoms=args.min_real_atoms,
        max_cut_attempts=args.max_cut_attempts,
    )
    if len(examples) < args.target_examples:
        raise SystemExit(
            "Could not build enough successful SFT examples. "
            f"Built {len(examples)} examples, expected {args.target_examples}."
        )

    train_examples, val_examples = split_examples(
        examples,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    save_sft_jsonl(train_output, train_examples)
    save_sft_jsonl(val_output, val_examples)

    overall_ratio = label_ratio(examples)
    train_ratio = label_ratio(train_examples)
    val_ratio = label_ratio(val_examples)

    print("SFT data preparation completed.")
    print(f"Input CSV: {input_csv}")
    print(f"Target format: {target_format}")
    print(f"Valid molecules: {len(valid_records)}")
    print(
        "Generation stats: "
        f"successful={summary.successful_examples} "
        f"failed={summary.failed_fragment_records} "
        f"total_attempts={summary.total_generation_attempts} "
        f"raw_candidates={summary.total_raw_candidates} "
        f"refill_attempts={summary.refill_records_attempted}"
    )
    print(
        "Drop by failure_tag: "
        f"{summary.drop_by_failure_tag}"
    )
    print(
        "Overall label ratio: "
        f"negative={overall_ratio[0]:.4f} positive={overall_ratio[1]:.4f}"
    )
    print(
        "Train label ratio: "
        f"negative={train_ratio[0]:.4f} positive={train_ratio[1]:.4f}"
    )
    print(
        "Validation label ratio: "
        f"negative={val_ratio[0]:.4f} positive={val_ratio[1]:.4f}"
    )
    print(f"Saved train JSONL: {train_output}")
    print(f"Saved val JSONL: {val_output}")

    if target_format == "core":
        audit_payload = build_core_sft_audit_payload(
            examples=examples,
            summary=summary,
            train_output=train_output,
            val_output=val_output,
        )
        save_audit_json(audit_output, audit_payload)
        print(f"Saved audit JSON: {audit_output}")


if __name__ == "__main__":
    main()
