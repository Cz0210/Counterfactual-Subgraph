#!/usr/bin/env python3
"""Create deterministic label-aware Bemis-Murcko group splits for Mutagenicity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.mutagenicity import (
    DEFAULT_SPLIT_RATIOS,
    SPLIT_NAMES,
    STANDARD_SPLIT_FIELDS,
    build_scaffold_splits,
    read_csv_rows,
    write_csv,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--processed-dir", default="data/processed/Mutagenicity/v1")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--calibration-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--attempts", type=int, default=128)
    return parser


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Mutagenicity Split Report",
        "",
        "Split method: deterministic label-aware Bemis-Murcko scaffold grouping.",
        "The empty Murcko scaffold is treated as one explicit acyclic group.",
        "",
        f"- Total rows: {summary['total_rows']}",
        f"- Split counts: `{summary['split_counts']}`",
        f"- Actual ratios: `{summary['split_ratios_actual']}`",
        f"- Label counts: `{summary['split_label_counts']}`",
        f"- Unique scaffolds: {summary['num_unique_scaffolds']}",
        f"- Scaffold overlap: {summary['scaffold_overlap_count']}",
        f"- Canonical SMILES overlap: {summary['canonical_smiles_overlap_count']}",
        f"- Validation passed: **{summary['split_validation_passed']}**",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir) if args.output_dir else processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = read_csv_rows(processed_dir / "mutagenicity_benchmark_clean.csv")
    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "calibration": args.calibration_ratio,
        "test": args.test_ratio,
    }
    manifest, summary = build_scaffold_splits(
        benchmark,
        seed=args.seed,
        ratios=ratios,
        attempts=args.attempts,
    )
    write_csv(
        output_dir / "mutagenicity_split_manifest.csv", manifest, STANDARD_SPLIT_FIELDS
    )
    for split in SPLIT_NAMES:
        write_csv(
            output_dir / f"{split}.csv",
            (row for row in manifest if row["split"] == split),
            STANDARD_SPLIT_FIELDS,
        )
    write_json(output_dir / "split_summary.json", summary)
    (output_dir / "split_report.md").write_text(_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    if not summary["split_validation_passed"]:
        print("[MUTAGENICITY_SPLIT_FAILED]", flush=True)
        return 1
    print("[MUTAGENICITY_SPLIT_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
