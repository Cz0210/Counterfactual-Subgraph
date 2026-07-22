#!/usr/bin/env python3
"""Build the canonical clean Mutagenicity SMILES benchmark from curated CSV."""

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
    BENCHMARK_FIELDS,
    MASTER_FIELDS,
    load_csv_source,
    preprocess_curated_source,
    preprocessing_summary,
    write_csv,
    write_json,
)


DUPLICATE_FIELDS = (
    "canonical_isomeric_smiles",
    "molecule_id",
    "label",
    "source_row_ids",
    "duplicate_count",
    "representative_source_row_id",
)
CONFLICT_FIELDS = (
    "canonical_isomeric_smiles",
    "molecule_id",
    "labels",
    "source_row_ids",
    "conflict_count",
)
ID_MAP_FIELDS = (
    "source_dataset",
    "source_row_id",
    "curated_row_id",
    "smiles_original",
    "canonical_isomeric_smiles",
    "molecule_id",
    "label",
    "keep",
    "drop_reason",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument(
        "--curated-csv",
        default="data/raw/Mutagenicity/smiles/smiles_mutagenicity_curated.csv",
    )
    parser.add_argument(
        "--raw-csv", default="data/raw/Mutagenicity/smiles/smiles_mutagenicity_raw.csv"
    )
    parser.add_argument("--output-dir", default="data/processed/Mutagenicity/v1")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        (
            "# Mutagenicity Preprocessing Report",
            "",
            f"- Source curated rows: {summary['source_rows']}",
            f"- Selected rows: {summary['selected_rows']}",
            f"- Clean unique molecules: {summary['num_clean_molecules']}",
            f"- Dropped rows: {summary['num_dropped_rows']}",
            f"- Duplicate groups: {summary['num_duplicate_groups']}",
            f"- Conflict groups: {summary['num_conflict_groups']}",
            f"- Label counts: `{summary['clean_label_counts']}`",
            f"- Drop reasons: `{summary['drop_reason_counts']}`",
            "",
            "Canonicalization preserves stereochemistry. No neutralization or tautomer "
            "canonicalization is applied.",
            "",
            f"Preprocess passed: **{summary['preprocess_passed']}**",
        )
    ) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curated = load_csv_source(args.curated_csv)
    raw = load_csv_source(args.raw_csv) if args.raw_csv else None
    result = preprocess_curated_source(
        curated,
        raw_source=raw,
        max_rows=args.max_rows,
        seed=args.seed,
    )
    summary = preprocessing_summary(result, Path(args.curated_csv))
    write_csv(output_dir / "mutagenicity_master.csv", result["master"], MASTER_FIELDS)
    write_csv(
        output_dir / "mutagenicity_benchmark_clean.csv", result["clean"], BENCHMARK_FIELDS
    )
    write_csv(output_dir / "mutagenicity_dropped.csv", result["dropped"], MASTER_FIELDS)
    write_csv(
        output_dir / "mutagenicity_duplicates.csv", result["duplicates"], DUPLICATE_FIELDS
    )
    write_csv(output_dir / "mutagenicity_conflicts.csv", result["conflicts"], CONFLICT_FIELDS)
    write_csv(output_dir / "mutagenicity_id_map.csv", result["id_map"], ID_MAP_FIELDS)
    write_json(output_dir / "preprocess_summary.json", summary)
    (output_dir / "preprocess_report.md").write_text(_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    if not summary["preprocess_passed"]:
        print("[MUTAGENICITY_PREPROCESS_FAILED]", flush=True)
        return 1
    print("[MUTAGENICITY_PREPROCESS_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
