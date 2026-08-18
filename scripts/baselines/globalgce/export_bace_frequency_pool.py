#!/usr/bin/env python3
"""Convert frozen BACE GlobalGCE ranks into a fullgraph candidate CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_bace_action_adapter import (  # noqa: E402
    adapt_globalgce_fullgraph_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--selected-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.selected_csv).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    adapted = adapt_globalgce_fullgraph_rows(rows)
    if args.validate_only:
        print(
            "[BACE_GLOBALGCE_FULLGRAPH_ADAPTER_VALIDATE_OK] "
            f"rows={len(adapted)} native_output_type=full_counterfactual_graph"
        )
        return 0
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank", "candidate_id", "candidate_smiles", "canonical_smiles",
        "rf_strict_flip", "connected", "native_output_type", "action_adapter",
        "selection_mode", "source_split",
    ]
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in adapted:
            writer.writerow(
                {
                    "rank": candidate.rank,
                    "candidate_id": candidate.candidate_id,
                    "candidate_smiles": candidate.candidate_smiles,
                    "canonical_smiles": candidate.canonical_smiles,
                    "rf_strict_flip": candidate.source_row.get("rf_strict_flip", True),
                    "connected": True,
                    "native_output_type": candidate.native_output_type,
                    "action_adapter": candidate.action_adapter,
                    "selection_mode": candidate.source_row.get(
                        "selection_mode", "globalgce_frequency_top20_train_support_v1"
                    ),
                    "source_split": candidate.source_row.get("source_split", "train"),
                }
            )
    print(f"[BACE_GLOBALGCE_FULLGRAPH_ADAPTER_OK] rows={len(adapted)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
