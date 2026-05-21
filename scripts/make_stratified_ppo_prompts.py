#!/usr/bin/env python3
"""Stratify and shuffle PPO prompt rows to avoid long hard-example blocks."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - depends on runtime
    Chem = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Accepted for HPC wrapper parity.")
    parser.add_argument("--dataset-path", required=True, help="Input PPO prompt CSV path.")
    parser.add_argument("--out-csv", required=True, help="Output stratified shuffled CSV path.")
    parser.add_argument("--out-json", required=True, help="Output summary JSON path.")
    parser.add_argument("--seed", type=int, default=13, help="Shuffle seed.")
    parser.add_argument("--smiles-col", default="", help="Preferred SMILES column name.")
    parser.add_argument("--label-col", default="label", help="Optional label column name for metadata.")
    return parser


def _resolve_smiles_column(fieldnames: list[str], requested: str) -> str:
    lowered = {name.strip().lower(): name for name in fieldnames if str(name).strip()}
    candidates = [requested, "parent_smiles", "smiles"]
    for candidate in candidates:
        key = str(candidate or "").strip().lower()
        if key and key in lowered:
            return lowered[key]
    raise ValueError(
        f"Could not resolve SMILES column from requested={requested!r} and fieldnames={fieldnames!r}"
    )


def _atom_bucket(atom_count: int) -> str:
    if atom_count < 20:
        return "0-20"
    if atom_count < 40:
        return "20-40"
    if atom_count < 60:
        return "40-60"
    return "60+"


def _compute_smiles_features(smiles: str) -> dict[str, Any]:
    text = str(smiles or "").strip()
    has_dot_or_salt = "." in text
    component_count = max(1, len([chunk for chunk in text.split(".") if chunk]))
    sanitize_fail = False
    atom_count = 0

    if Chem is None or not text:
        return {
            "atom_count": atom_count,
            "has_dot_or_salt": has_dot_or_salt,
            "component_count": component_count,
            "sanitize_fail": True,
        }

    mol = Chem.MolFromSmiles(text, sanitize=False)
    if mol is None:
        return {
            "atom_count": atom_count,
            "has_dot_or_salt": has_dot_or_salt,
            "component_count": component_count,
            "sanitize_fail": True,
        }

    atom_count = int(mol.GetNumAtoms())
    try:
        component_count = int(len(Chem.GetMolFrags(mol)))
    except Exception:
        component_count = component_count

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        sanitize_fail = True

    return {
        "atom_count": atom_count,
        "has_dot_or_salt": has_dot_or_salt or component_count > 1,
        "component_count": component_count,
        "sanitize_fail": sanitize_fail,
    }


def _round_robin_interleave(groups: dict[tuple[str, bool], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    bucket_keys = sorted(groups.keys(), key=lambda item: (item[0], item[1]))
    output: list[dict[str, Any]] = []
    remaining = True
    while remaining:
        remaining = False
        for bucket_key in bucket_keys:
            bucket_rows = groups[bucket_key]
            if bucket_rows:
                output.append(bucket_rows.pop())
                remaining = True
    return output


def _build_block_summary(rows: list[dict[str, Any]], block_size: int = 50) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block_start in range(0, len(rows), block_size):
        block_rows = rows[block_start : block_start + block_size]
        atom_counts = [int(row["_atom_count"]) for row in block_rows]
        component_counts = [int(row["_component_count"]) for row in block_rows]
        sanitize_fail_count = sum(1 for row in block_rows if bool(row["_sanitize_fail"]))
        dot_salt_count = sum(1 for row in block_rows if bool(row["_has_dot_or_salt"]))
        blocks.append(
            {
                "block_start": block_start + 1,
                "block_end": block_start + len(block_rows),
                "atom_count_mean": (sum(atom_counts) / len(atom_counts)) if atom_counts else 0.0,
                "atom_count_max": max(atom_counts) if atom_counts else 0,
                "dot_salt_count": int(dot_salt_count),
                "component_count_mean": (
                    sum(component_counts) / len(component_counts)
                    if component_counts
                    else 0.0
                ),
                "sanitize_fail_count": int(sanitize_fail_count),
            }
        )
    return blocks


def stratify_and_shuffle_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    smiles_col: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(int(seed))
    buckets: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        features = _compute_smiles_features(str(row.get(smiles_col) or ""))
        enriched = dict(row)
        enriched["_atom_count"] = int(features["atom_count"])
        enriched["_has_dot_or_salt"] = bool(features["has_dot_or_salt"])
        enriched["_component_count"] = int(features["component_count"])
        enriched["_sanitize_fail"] = bool(features["sanitize_fail"])
        bucket_key = (
            _atom_bucket(int(features["atom_count"])),
            bool(features["has_dot_or_salt"]),
        )
        buckets[bucket_key].append(enriched)

    bucket_sizes = {
        f"{atom_bucket}|dot_or_salt={has_dot_or_salt}": len(bucket_rows)
        for (atom_bucket, has_dot_or_salt), bucket_rows in buckets.items()
    }
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)

    interleaved = _round_robin_interleave(buckets)
    summary = {
        "seed": int(seed),
        "num_rows": len(interleaved),
        "bucket_sizes": bucket_sizes,
        "blocks": _build_block_summary(interleaved, block_size=50),
    }
    return interleaved, summary


def main() -> None:
    args = build_parser().parse_args()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])

    smiles_col = _resolve_smiles_column(fieldnames, args.smiles_col)
    shuffled_rows, summary = stratify_and_shuffle_rows(
        rows,
        seed=args.seed,
        smiles_col=smiles_col,
    )

    output_fieldnames = list(fieldnames)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames)
        writer.writeheader()
        for row in shuffled_rows:
            writer.writerow({field: row.get(field, "") for field in output_fieldnames})

    summary.update(
        {
            "dataset_path": str(dataset_path),
            "resolved_smiles_col": smiles_col,
            "label_col": args.label_col,
            "out_csv": str(out_csv),
        }
    )
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
