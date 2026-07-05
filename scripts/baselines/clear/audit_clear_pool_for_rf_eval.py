#!/usr/bin/env python3
"""Audit whether a CLEAR full-graph pool can be evaluated by the RF oracle."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.clear_rf_adapter import (  # noqa: E402
    FULL_GRAPH_FIELDS,
    convert_clear_record_graphs,
    read_jsonl,
    record_has_full_graph_fields,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clear-pool",
        default="outputs/hpc/baselines/clear/aids/candidate_pool/clear_aids_candidate_pool.with_graphs.jsonl",
    )
    parser.add_argument("--dataset", default="aids")
    parser.add_argument(
        "--out-json",
        default="outputs/hpc/baselines/clear/aids/rf_unified/audit_clear_rf_feasibility.json",
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    return parser


def missing_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    required = (
        "original_adj",
        "cf_adj",
        "original_x",
        "cf_x",
        "original_label",
        "exp_id",
        "instance_index",
    )
    return {field: sum(1 for row in rows if row.get(field) in (None, "")) for field in required}


def main() -> int:
    args = build_parser().parse_args()
    clear_pool = Path(args.clear_pool).expanduser()
    if not clear_pool.exists():
        raise FileNotFoundError(f"CLEAR pool not found: {clear_pool}")
    rows = read_jsonl(clear_pool, max_records=args.max_records)
    reason_counts: Counter[str] = Counter()
    original_reason_counts: Counter[str] = Counter()
    cf_reason_counts: Counter[str] = Counter()
    has_original_smiles = 0
    has_cf_smiles = 0
    full_graph_count = 0
    original_ok = 0
    cf_ok = 0
    examples: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        if row.get("original_smiles"):
            has_original_smiles += 1
        if row.get("cf_smiles"):
            has_cf_smiles += 1
        if record_has_full_graph_fields(row):
            full_graph_count += 1
        converted = convert_clear_record_graphs(row)
        original = converted["original"]
        cf = converted["cf"]
        if original.ok:
            original_ok += 1
        else:
            original_reason_counts[original.reason or "unknown"] += 1
        if cf.ok:
            cf_ok += 1
        else:
            cf_reason_counts[cf.reason or "unknown"] += 1
            reason_counts[cf.reason or "unknown"] += 1
        if len(examples) < 5:
            examples.append(
                {
                    "record_index": index,
                    "candidate_id": row.get("candidate_id"),
                    "instance_index": row.get("instance_index"),
                    "original_ok": original.ok,
                    "original_smiles": original.smiles,
                    "original_reason": original.reason,
                    "cf_ok": cf.ok,
                    "cf_smiles": cf.smiles,
                    "cf_reason": cf.reason,
                }
            )

    num_records = len(rows)
    usable_count = has_cf_smiles + cf_ok if has_cf_smiles else cf_ok
    usable = usable_count > 0
    if not usable:
        reason = "no_cf_smiles_and_cf_graph_to_mol_failed"
        next_step = (
            "RF oracle evaluation is not currently usable. Keep clear_graphpred native diagnostics and inspect "
            "CLEAR graph feature semantics before reporting RF-unified metrics."
        )
    else:
        reason = None
        next_step = (
            "Run convert_clear_pool_to_rf_fullgraph_candidates.py, then evaluate CLEAR-RF-FullGraph with "
            "GREED-GED and MolCLR using the same parent set and RF oracle as Ours/GT-FullGraph."
        )

    summary = {
        "dataset": args.dataset,
        "clear_pool": str(clear_pool),
        "num_records": num_records,
        "max_records": args.max_records,
        "has_full_graph_fields_count": full_graph_count,
        "full_graph_fields": list(FULL_GRAPH_FIELDS),
        "missing_field_counts": missing_counts(rows),
        "has_original_smiles_count": has_original_smiles,
        "has_cf_smiles_count": has_cf_smiles,
        "cf_graph_to_mol_ok_count": cf_ok,
        "cf_graph_to_mol_ok_rate": (cf_ok / num_records) if num_records else 0.0,
        "original_graph_to_mol_ok_count": original_ok,
        "original_graph_to_mol_ok_rate": (original_ok / num_records) if num_records else 0.0,
        "invalid_reason_counts": dict(reason_counts),
        "original_invalid_reason_counts": dict(original_reason_counts),
        "cf_invalid_reason_counts": dict(cf_reason_counts),
        "rf_oracle_usable": usable,
        "reason": reason,
        "recommended_next_step": next_step,
        "examples": examples,
    }
    write_json(args.out_json, summary)
    print("[CLEAR_RF_AUDIT]")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if usable:
        print("[CLEAR_RF_AUDIT_OK]")
    else:
        print("[CLEAR_RF_AUDIT_NOT_USABLE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
