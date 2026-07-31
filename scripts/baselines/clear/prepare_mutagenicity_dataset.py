#!/usr/bin/env python3
"""Prepare strict train/validation Mutagenicity pickles for official CLEAR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.clear_mutagenicity_adapter import (  # noqa: E402
    DEFAULT_EXPECTED_COUNTS,
    build_clear_dataset_payload,
    load_phase_a_cohorts,
    write_clear_dataset,
)


DEFAULT_DATA_ROOT = (
    "outputs/hpc/datasets/mutagenicity_v1_teacher_consistent"
)
DEFAULT_SUMMARY = (
    "outputs/hpc/mutagenicity/baselines/clear/dataset_v1/"
    "dataset_summary.json"
)


def _default(root: str, filename: str) -> str:
    return str(Path(root) / filename)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--train-positive-csv")
    parser.add_argument("--train-negative-csv")
    parser.add_argument("--val-positive-csv")
    parser.add_argument("--val-negative-csv")
    parser.add_argument(
        "--official-dataset-dir",
        default="baselines/clear_official/dataset",
    )
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY)
    parser.add_argument("--num-split-repetitions", type=int, default=10)
    parser.add_argument(
        "--expected-train-positive",
        type=int,
        default=DEFAULT_EXPECTED_COUNTS["train_positive"],
    )
    parser.add_argument(
        "--expected-train-negative",
        type=int,
        default=DEFAULT_EXPECTED_COUNTS["train_negative"],
    )
    parser.add_argument(
        "--expected-val-positive",
        type=int,
        default=DEFAULT_EXPECTED_COUNTS["val_positive"],
    )
    parser.add_argument(
        "--expected-val-negative",
        type=int,
        default=DEFAULT_EXPECTED_COUNTS["val_negative"],
    )
    parser.add_argument(
        "--forbid-calibration-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _resolve_inputs(args: argparse.Namespace) -> dict[str, str]:
    root = args.data_root
    return {
        "train_positive_csv": args.train_positive_csv
        or _default(root, "train_source_label1_teacher_correct.csv"),
        "train_negative_csv": args.train_negative_csv
        or _default(root, "train_target_label0_teacher_correct.csv"),
        "val_positive_csv": args.val_positive_csv
        or _default(root, "val_source_label1_teacher_correct.csv"),
        "val_negative_csv": args.val_negative_csv
        or _default(root, "val_target_label0_teacher_correct.csv"),
    }


def _load_graph_data_class() -> type:
    clear_src = REPO_ROOT / "baselines" / "clear_official" / "src"
    if str(clear_src) not in sys.path:
        sys.path.insert(0, str(clear_src))
    from data_sampler import GraphData

    return GraphData


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.forbid_calibration_test:
        raise ValueError("Phase A requires --forbid-calibration-test.")
    paths = _resolve_inputs(args)
    cohorts = load_phase_a_cohorts(
        **paths,
        expected_counts={
            "train_positive": args.expected_train_positive,
            "train_negative": args.expected_train_negative,
            "val_positive": args.expected_val_positive,
            "val_negative": args.expected_val_negative,
        },
    )
    data, split_payload, summary = build_clear_dataset_payload(
        cohorts=cohorts,
        graph_data_class=_load_graph_data_class(),
        num_split_repetitions=args.num_split_repetitions,
    )
    summary["input_paths"] = {
        key: str(Path(value).expanduser().resolve())
        for key, value in paths.items()
    }
    result = write_clear_dataset(
        data=data,
        split_payload=split_payload,
        summary=summary,
        official_dataset_dir=args.official_dataset_dir,
        summary_path=args.summary_path,
    )
    print("[MUTAGENICITY_CLEAR_DATASET_PREP_OK]", flush=True)
    print(f"train_rows={result['train_rows']}", flush=True)
    print(f"val_rows={result['val_rows']}", flush=True)
    print(f"atom_vocabulary={result['atom_vocabulary']}", flush=True)
    print(
        f"formal_charge_vocabulary={result['formal_charge_vocabulary']}",
        flush=True,
    )
    print(f"max_num_nodes={result['max_num_nodes']}", flush=True)
    print(f"summary_path={result['summary_path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
