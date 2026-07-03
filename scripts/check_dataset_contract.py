#!/usr/bin/env python3
"""Check the repository AIDS/HIV dataset contract."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the canonical AIDS/HIV CSV path, schema, row count, and label distribution."
    )
    parser.add_argument("--config", default=None, help="Accepted for Slurm/config compatibility; not used.")
    parser.add_argument("--csv", default="data/raw/AIDS/HIV.csv", help="Canonical AIDS/HIV CSV path.")
    parser.add_argument("--smiles-column", default="smiles", help="SMILES column name.")
    parser.add_argument("--label-column", default="HIV_active", help="Label column name.")
    parser.add_argument("--expected-total", type=int, default=41127, help="Expected total row count.")
    parser.add_argument("--expected-label0", type=int, default=39684, help="Expected count for label 0.")
    parser.add_argument("--expected-label1", type=int, default=1443, help="Expected count for label 1.")
    parser.add_argument(
        "--out-json",
        default="outputs/hpc/diagnostics/dataset_contract_aids_hiv.json",
        help="Where to write the diagnostic JSON.",
    )
    return parser.parse_args()


def normalize_label(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    csv_path = Path(args.csv)
    result: dict[str, Any] = {
        "csv_path": str(csv_path),
        "exists": csv_path.exists(),
        "columns": [],
        "total_rows": 0,
        "smiles_nonempty_count": 0,
        "label_distribution": {},
        "pass": False,
        "error_messages": [],
    }

    errors: list[str] = result["error_messages"]
    if not csv_path.exists():
        errors.append(f"CSV does not exist: {csv_path}")
        return result

    label_counts: Counter[str] = Counter()
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            result["columns"] = columns

            if args.smiles_column not in columns:
                errors.append(f"Missing SMILES column: {args.smiles_column}")
            if args.label_column not in columns:
                errors.append(f"Missing label column: {args.label_column}")

            for row in reader:
                result["total_rows"] += 1
                if args.smiles_column in row and str(row.get(args.smiles_column, "")).strip():
                    result["smiles_nonempty_count"] += 1
                if args.label_column in row:
                    label_counts[normalize_label(row.get(args.label_column, ""))] += 1
    except Exception as exc:  # pragma: no cover - defensive diagnostic path
        errors.append(f"Failed to read CSV: {exc}")
        result["label_distribution"] = dict(sorted(label_counts.items()))
        return result

    result["label_distribution"] = dict(sorted(label_counts.items()))

    if result["total_rows"] != args.expected_total:
        errors.append(f"Expected {args.expected_total} rows, found {result['total_rows']}")
    if label_counts.get("0", 0) != args.expected_label0:
        errors.append(f"Expected label 0 count {args.expected_label0}, found {label_counts.get('0', 0)}")
    if label_counts.get("1", 0) != args.expected_label1:
        errors.append(f"Expected label 1 count {args.expected_label1}, found {label_counts.get('1', 0)}")

    result["pass"] = not errors
    return result


def write_json(path: str, result: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    result = build_result(args)
    write_json(args.out_json, result)

    if result["pass"]:
        print("[DATASET_CONTRACT_OK]")
        print(f"csv_path={result['csv_path']}")
        print(f"total_rows={result['total_rows']}")
        print(f"label_distribution={result['label_distribution']}")
        print(f"out_json={args.out_json}")
        return 0

    print("[DATASET_CONTRACT_FAIL]", file=sys.stderr)
    for error in result["error_messages"]:
        print(f"- {error}", file=sys.stderr)
    print(f"out_json={args.out_json}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
