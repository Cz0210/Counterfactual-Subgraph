#!/usr/bin/env python3
"""Convert CLEAR full-graph JSONL records into RF/unified fullgraph candidates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.clear_rf_adapter import (  # noqa: E402
    as_bool,
    as_float,
    as_int,
    convert_clear_record_graphs,
    read_jsonl,
    to_jsonable,
    write_json,
    write_jsonl,
)


CSV_FIELDS = [
    "candidate_id",
    "source_method",
    "dataset",
    "candidate_smiles",
    "candidate_valid",
    "candidate_num_atoms",
    "source_exp_id",
    "source_instance_index",
    "source_record_index",
    "official_flip",
    "official_target_success",
    "clear_original_pred_label",
    "clear_cf_pred_label",
    "edge_added_count",
    "edge_deleted_count",
    "edge_changed_count",
    "feature_l1_cost",
    "total_action_cost",
    "source_original_smiles",
    "source_original_label",
    "invalid_reason",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clear-pool",
        default="outputs/hpc/baselines/clear/aids/candidate_pool/clear_aids_candidate_pool.with_graphs.jsonl",
    )
    parser.add_argument(
        "--out-csv",
        default="outputs/hpc/baselines/clear/aids/rf_unified/clear_aids_rf_fullgraph_candidates.csv",
    )
    parser.add_argument(
        "--out-jsonl",
        default="outputs/hpc/baselines/clear/aids/rf_unified/clear_aids_rf_fullgraph_candidates.jsonl",
    )
    parser.add_argument(
        "--out-summary",
        default="outputs/hpc/baselines/clear/aids/rf_unified/clear_aids_rf_fullgraph_candidates_summary.json",
    )
    parser.add_argument("--dataset", default="aids")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    return parser


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (list, dict, tuple))
                    else ("" if row.get(key) is None else row.get(key))
                    for key in CSV_FIELDS
                }
            )


def candidate_from_record(row: dict[str, Any], *, dataset: str, record_index: int) -> dict[str, Any]:
    converted = convert_clear_record_graphs(row)
    original = converted["original"]
    cf = converted["cf"]
    candidate_id = str(row.get("candidate_id") or f"CLEAR_RF_{dataset}_{record_index:06d}")
    valid = bool(cf.ok and cf.smiles)
    return {
        "candidate_id": candidate_id,
        "source_method": "CLEAR",
        "method": "CLEAR-RF-FullGraph",
        "dataset": dataset,
        "candidate_smiles": cf.smiles,
        "smiles": cf.smiles,
        "candidate_valid": valid,
        "candidate_num_atoms": cf.num_atoms,
        "candidate_num_bonds": cf.num_bonds,
        "source_exp_id": as_int(row.get("exp_id")),
        "source_instance_index": as_int(row.get("instance_index")),
        "source_record_index": record_index,
        "official_flip": as_bool(row.get("official_flip")),
        "official_target_success": as_bool(row.get("official_target_success")),
        "clear_original_pred_label": as_int(row.get("official_original_pred_label")),
        "clear_cf_pred_label": as_int(row.get("official_cf_pred_label")),
        "edge_added_count": as_int(row.get("num_edge_added")),
        "edge_deleted_count": as_int(row.get("num_edge_deleted")),
        "edge_changed_count": as_int(row.get("num_edge_changed")),
        "feature_l1_cost": as_float(row.get("feature_l1_cost")),
        "total_action_cost": as_float(row.get("total_cost")),
        "source_original_smiles": original.smiles,
        "source_original_label": as_int(row.get("original_label")),
        "source_original_valid": bool(original.ok),
        "invalid_reason": None if valid else (cf.reason or "unknown_conversion_failure"),
        "invalid_error": cf.error,
    }


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def main() -> int:
    args = build_parser().parse_args()
    pool_path = Path(args.clear_pool).expanduser()
    if not pool_path.exists():
        raise FileNotFoundError(f"CLEAR pool not found: {pool_path}")
    rows = read_jsonl(pool_path, max_records=args.max_records)
    candidates = [candidate_from_record(row, dataset=args.dataset, record_index=index) for index, row in enumerate(rows)]
    valid_rows = [row for row in candidates if row.get("candidate_valid") is True]
    csv_rows = candidates if args.include_invalid else valid_rows
    invalid_reasons = Counter(str(row.get("invalid_reason")) for row in candidates if row.get("candidate_valid") is not True)
    write_csv(args.out_csv, csv_rows)
    write_jsonl(args.out_jsonl, candidates)
    summary = {
        "dataset": args.dataset,
        "clear_pool": str(pool_path),
        "out_csv": args.out_csv,
        "out_jsonl": args.out_jsonl,
        "out_summary": args.out_summary,
        "num_input_records": len(rows),
        "num_detail_records": len(candidates),
        "num_valid_candidates": len(valid_rows),
        "num_csv_rows": len(csv_rows),
        "include_invalid": bool(args.include_invalid),
        "valid_rate": (len(valid_rows) / len(candidates)) if candidates else 0.0,
        "invalid_reason_counts": dict(invalid_reasons),
        "mean_edge_changed": mean([float(row["edge_changed_count"]) for row in candidates if row.get("edge_changed_count") is not None]),
        "mean_feature_l1_cost": mean([float(row["feature_l1_cost"]) for row in candidates if row.get("feature_l1_cost") is not None]),
        "mean_total_action_cost": mean([float(row["total_action_cost"]) for row in candidates if row.get("total_action_cost") is not None]),
        "source_method": "CLEAR",
        "method": "CLEAR-RF-FullGraph",
        "note": "CSV defaults to candidate_valid=true rows only. JSONL keeps invalid rows for audit.",
    }
    write_json(args.out_summary, summary)
    print("[CLEAR_RF_CONVERT_SUMMARY]")
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    print("[CLEAR_RF_CONVERT_PREVIEW]")
    for row in candidates[:3]:
        print(json.dumps({key: row.get(key) for key in CSV_FIELDS}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
