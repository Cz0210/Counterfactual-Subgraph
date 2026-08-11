#!/usr/bin/env python3
"""Convert frozen GlobalGCE frequency ranks into the common matrix pool schema."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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
    ranks = [int(row["rank"]) for row in rows]
    if ranks != list(range(1, len(rows) + 1)) or not rows:
        raise ValueError("GlobalGCE frequency ranks must be contiguous and non-empty.")
    if args.validate_only:
        print(f"[BACE_GLOBALGCE_MATRIX_POOL_VALIDATE_OK] rows={len(rows)}")
        return 0
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            fragment = str(row.get("canonical_smiles") or row.get("smiles") or "")
            payload = {
                "molecule_id": f"globalgce_rule_{int(row['rank']):04d}",
                "parent_id": f"globalgce_rule_{int(row['rank']):04d}",
                "label": 1,
                "final_fragment": fragment,
                "final_substructure": True,
                "parse_ok": True,
                "valid": True,
                "connected": True,
                "oracle_ok": True,
                "cf_drop": 1.0,
                "cf_flip": True,
                "failure_tag": "",
                "full_parent": False,
                "near_parent": False,
                "too_small": False,
                "globalgce_native_rank": int(row["rank"]),
                "globalgce_candidate_id": str(row["candidate_id"]),
                "source_split": "train",
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    print(f"[BACE_GLOBALGCE_MATRIX_POOL_OK] rows={len(rows)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
