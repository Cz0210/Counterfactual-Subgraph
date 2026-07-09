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
    analyze_clear_record_schema,
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
    parser.add_argument("--adj-threshold", type=float, default=0.5)
    parser.add_argument("--min-valid-candidates", type=int, default=20)
    parser.add_argument("--min-valid-rate", type=float, default=0.001)
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


def merge_counter(target: Counter[str], values: dict[str, int]) -> None:
    for key, value in values.items():
        target[str(key)] += int(value)


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
    original_active = 0
    original_onehot = 0
    cf_active = 0
    cf_onehot = 0
    cf_continuous = 0
    original_argmax: Counter[str] = Counter()
    cf_argmax: Counter[str] = Counter()
    original_decode_modes: Counter[str] = Counter()
    cf_decode_modes: Counter[str] = Counter()
    original_decoded_atomic_nums: Counter[str] = Counter()
    cf_decoded_atomic_nums: Counter[str] = Counter()
    node_mask_sources: Counter[str] = Counter()
    cf_adj_mins: list[float] = []
    cf_adj_maxs: list[float] = []
    cf_adj_means: list[float] = []
    total_attempted_bonds = 0
    total_skipped_bonds_for_valence = 0

    for index, row in enumerate(rows):
        if row.get("original_smiles"):
            has_original_smiles += 1
        if row.get("cf_smiles"):
            has_cf_smiles += 1
        if record_has_full_graph_fields(row):
            full_graph_count += 1
        schema = analyze_clear_record_schema(row, adjacency_threshold=float(args.adj_threshold))
        node_mask_sources[str(schema.get("node_mask_source") or "unknown")] += 1
        original_schema = schema.get("original_x") or {}
        cf_schema = schema.get("cf_x") or {}
        original_active += int(original_schema.get("active_rows") or 0)
        original_onehot += int(original_schema.get("onehot_like_count") or 0)
        cf_active += int(cf_schema.get("active_rows") or 0)
        cf_onehot += int(cf_schema.get("onehot_like_count") or 0)
        cf_continuous += int(cf_schema.get("continuous_count") or 0)
        merge_counter(original_argmax, original_schema.get("argmax_distribution") or {})
        merge_counter(cf_argmax, cf_schema.get("argmax_distribution") or {})
        merge_counter(original_decode_modes, original_schema.get("decode_mode_counts") or {})
        merge_counter(cf_decode_modes, cf_schema.get("decode_mode_counts") or {})
        merge_counter(original_decoded_atomic_nums, original_schema.get("decoded_atomic_num_counts") or {})
        merge_counter(cf_decoded_atomic_nums, cf_schema.get("decoded_atomic_num_counts") or {})
        if isinstance(schema.get("cf_adj_min"), (int, float)):
            cf_adj_mins.append(float(schema["cf_adj_min"]))
        if isinstance(schema.get("cf_adj_max"), (int, float)):
            cf_adj_maxs.append(float(schema["cf_adj_max"]))
        if isinstance(schema.get("cf_adj_mean"), (int, float)):
            cf_adj_means.append(float(schema["cf_adj_mean"]))

        converted = convert_clear_record_graphs(row, adjacency_threshold=float(args.adj_threshold))
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
        total_attempted_bonds += int(cf.attempted_bonds or 0)
        total_skipped_bonds_for_valence += int(cf.skipped_bonds_for_valence or 0)
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
                    "node_mask_source": cf.node_mask_source,
                    "num_nodes_used": cf.num_nodes_used,
                    "atom_decode_source": cf.atom_decode_source,
                    "atom_decode_mode": cf.atom_decode_mode,
                    "attempted_bonds": cf.attempted_bonds,
                    "skipped_bonds_for_valence": cf.skipped_bonds_for_valence,
                }
            )

    num_records = len(rows)
    usable_count = cf_ok
    valid_rate = (usable_count / num_records) if num_records else 0.0
    quality_gate_pass = usable_count >= int(args.min_valid_candidates) and valid_rate >= float(args.min_valid_rate)
    usable = quality_gate_pass
    if not usable:
        reason = "quality_gate_failed_or_no_cf_smiles"
        next_step = (
            "RF oracle evaluation is not currently usable. Inspect CLEAR graph feature semantics or adjust "
            "the conservative graph-to-SMILES adapter before reporting RF-unified metrics."
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
        "adj_threshold": float(args.adj_threshold),
        "min_valid_candidates": int(args.min_valid_candidates),
        "min_valid_rate": float(args.min_valid_rate),
        "quality_gate_pass": bool(quality_gate_pass),
        "original_x_onehot_like_rate": (original_onehot / original_active) if original_active else 0.0,
        "cf_x_onehot_like_rate": (cf_onehot / cf_active) if cf_active else 0.0,
        "cf_x_continuous_rate": (cf_continuous / cf_active) if cf_active else 0.0,
        "original_x_argmax_distribution": dict(original_argmax),
        "cf_x_argmax_distribution": dict(cf_argmax),
        "atom_type_argmax_counts": dict(cf_argmax),
        "original_x_decode_mode_counts": dict(original_decode_modes),
        "cf_x_decode_mode_counts": dict(cf_decode_modes),
        "original_x_decoded_atomic_num_counts": dict(original_decoded_atomic_nums),
        "cf_x_decoded_atomic_num_counts": dict(cf_decoded_atomic_nums),
        "node_mask_source_counts": dict(node_mask_sources),
        "cf_adj_min": min(cf_adj_mins) if cf_adj_mins else None,
        "cf_adj_max": max(cf_adj_maxs) if cf_adj_maxs else None,
        "cf_adj_mean": (sum(cf_adj_means) / len(cf_adj_means)) if cf_adj_means else None,
        "total_attempted_bonds": total_attempted_bonds,
        "total_skipped_bonds_for_valence": total_skipped_bonds_for_valence,
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
