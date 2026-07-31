#!/usr/bin/env python3
"""Probe strict Mutagenicity CLEAR source graph chemistry round trips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.clear_mutagenicity_adapter import (  # noqa: E402
    ClearMutagenicityCodecError,
    DEFAULT_EXPECTED_COUNTS,
    load_phase_a_cohorts,
    run_codec_probe,
    write_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-positive-csv", required=True)
    parser.add_argument("--train-negative-csv", required=True)
    parser.add_argument("--val-positive-csv", required=True)
    parser.add_argument("--val-negative-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=64,
        help="Number of deterministic required-category-first probe rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Recorded reproducibility seed; category-first sampling is stable.",
    )
    parser.add_argument(
        "--forbid-calibration-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--expected-train-positive", type=int, default=DEFAULT_EXPECTED_COUNTS["train_positive"]
    )
    parser.add_argument(
        "--expected-train-negative", type=int, default=DEFAULT_EXPECTED_COUNTS["train_negative"]
    )
    parser.add_argument(
        "--expected-val-positive", type=int, default=DEFAULT_EXPECTED_COUNTS["val_positive"]
    )
    parser.add_argument(
        "--expected-val-negative", type=int, default=DEFAULT_EXPECTED_COUNTS["val_negative"]
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.forbid_calibration_test:
        raise ValueError("Codec probe requires --forbid-calibration-test.")
    cohorts = load_phase_a_cohorts(
        train_positive_csv=args.train_positive_csv,
        train_negative_csv=args.train_negative_csv,
        val_positive_csv=args.val_positive_csv,
        val_negative_csv=args.val_negative_csv,
        expected_counts={
            "train_positive": args.expected_train_positive,
            "train_negative": args.expected_train_negative,
            "val_positive": args.expected_val_positive,
            "val_negative": args.expected_val_negative,
        },
    )
    rows, summary = run_codec_probe(cohorts=cohorts, limit=args.limit)
    summary["seed"] = int(args.seed)
    summary["input_paths"] = {
        "train_positive_csv": str(Path(args.train_positive_csv).resolve()),
        "train_negative_csv": str(Path(args.train_negative_csv).resolve()),
        "val_positive_csv": str(Path(args.val_positive_csv).resolve()),
        "val_negative_csv": str(Path(args.val_negative_csv).resolve()),
    }
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "codec_probe_rows.jsonl", rows)
    (output / "codec_probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not summary["probe_passed"]:
        raise ClearMutagenicityCodecError(
            f"Source codec probe failed for {summary['probe_failed_rows']} rows."
        )
    print("[MUTAGENICITY_CLEAR_CODEC_PROBE_OK]", flush=True)
    print(f"probe_rows={summary['probe_rows']}", flush=True)
    print(f"failed_count={summary['failed_count']}", flush=True)
    print(
        "required_category_representatives="
        f"{summary['required_category_representatives']}",
        flush=True,
    )
    print(f"output_dir={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
