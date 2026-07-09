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
    analyze_clear_record_schema,
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
    "node_mask_source",
    "num_nodes_used",
    "atom_decode_source",
    "atom_decode_mode",
    "skipped_bonds_for_valence",
    "attempted_bonds",
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
    parser.add_argument("--min-valid-candidates", type=int, default=20)
    parser.add_argument("--min-valid-rate", type=float, default=0.001)
    parser.add_argument("--adj-threshold", type=float, default=0.5)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate an existing candidate CSV and exit without reading the CLEAR pool.",
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Candidate CSV to validate when --validate-only is set. Defaults to --out-csv.",
    )
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


def candidate_from_record(row: dict[str, Any], *, dataset: str, record_index: int, adj_threshold: float) -> dict[str, Any]:
    converted = convert_clear_record_graphs(row, adjacency_threshold=float(adj_threshold))
    original = converted["original"]
    cf = converted["cf"]
    candidate_id = str(row.get("candidate_id") or f"CLEAR_RF_{dataset}_{record_index:06d}")
    cf_valid = bool(getattr(cf, "valid", getattr(cf, "ok", False)))
    original_valid = bool(getattr(original, "valid", getattr(original, "ok", False)))
    cf_reason = getattr(cf, "invalid_reason", getattr(cf, "reason", None))
    valid = bool(cf_valid and getattr(cf, "smiles", None))
    return {
        "candidate_id": candidate_id,
        "source_method": "CLEAR",
        "method": "CLEAR-RF-FullGraph",
        "dataset": dataset,
        "candidate_smiles": getattr(cf, "smiles", None),
        "smiles": getattr(cf, "smiles", None),
        "candidate_valid": valid,
        "candidate_num_atoms": getattr(cf, "num_atoms", None),
        "candidate_num_bonds": getattr(cf, "num_bonds", None),
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
        "source_original_smiles": getattr(original, "smiles", None),
        "source_original_label": as_int(row.get("original_label")),
        "source_original_valid": original_valid,
        "invalid_reason": None if valid else (cf_reason or "unknown_conversion_failure"),
        "invalid_error": getattr(cf, "error", None),
        "node_mask_source": getattr(cf, "node_mask_source", "unknown"),
        "num_nodes_used": getattr(cf, "num_nodes_used", None),
        "atom_decode_source": getattr(cf, "atom_decode_source", "unknown"),
        "atom_decode_mode": getattr(cf, "atom_decode_mode", "unknown"),
        "skipped_bonds_for_valence": getattr(cf, "skipped_bonds_for_valence", 0),
        "attempted_bonds": getattr(cf, "attempted_bonds", 0),
    }


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def merge_counter(target: Counter[str], values: dict[str, int]) -> None:
    for key, value in values.items():
        target[str(key)] += int(value)


def summarize_schema(rows: list[dict[str, Any]], *, adj_threshold: float) -> dict[str, Any]:
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
    num_nodes_values: list[float] = []
    for row in rows:
        schema = analyze_clear_record_schema(row, adjacency_threshold=float(adj_threshold))
        node_mask_sources[str(schema.get("node_mask_source") or "unknown")] += 1
        if schema.get("num_nodes_used") is not None:
            num_nodes_values.append(float(schema["num_nodes_used"]))
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
    return {
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
        "num_nodes_used_mean": mean(num_nodes_values),
        "cf_adj_min": min(cf_adj_mins) if cf_adj_mins else None,
        "cf_adj_max": max(cf_adj_maxs) if cf_adj_maxs else None,
        "cf_adj_mean": mean(cf_adj_means),
        "cf_adj_threshold": float(adj_threshold),
    }


def validate_candidate_csv(path: str | Path, *, min_valid_candidates: int) -> dict[str, Any]:
    candidate_path = Path(path).expanduser()
    result: dict[str, Any] = {
        "candidate_csv": str(candidate_path),
        "candidate_csv_exists": candidate_path.exists(),
        "candidate_smiles_column_present": False,
        "num_rows": 0,
        "rdkit_valid_count": 0,
        "rdkit_valid_rate": 0.0,
        "atom_count_min": None,
        "atom_count_max": None,
        "validation_pass": False,
        "validation_error": None,
    }
    if not candidate_path.exists():
        result["validation_error"] = "candidate_csv_missing"
        return result
    try:
        from rdkit import Chem
    except Exception as exc:  # pragma: no cover - environment dependent
        result["validation_error"] = f"rdkit_unavailable:{exc}"
        return result
    atom_counts: list[int] = []
    with candidate_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        result["candidate_smiles_column_present"] = "candidate_smiles" in (reader.fieldnames or [])
        if not result["candidate_smiles_column_present"]:
            result["validation_error"] = "candidate_smiles_column_missing"
            return result
        for row in reader:
            result["num_rows"] += 1
            smiles = (row.get("candidate_smiles") or "").strip()
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue
            result["rdkit_valid_count"] += 1
            atom_counts.append(int(mol.GetNumAtoms()))
    result["rdkit_valid_rate"] = (result["rdkit_valid_count"] / result["num_rows"]) if result["num_rows"] else 0.0
    result["atom_count_min"] = min(atom_counts) if atom_counts else None
    result["atom_count_max"] = max(atom_counts) if atom_counts else None
    result["validation_pass"] = bool(
        result["candidate_smiles_column_present"]
        and result["rdkit_valid_count"] >= int(min_valid_candidates)
    )
    return result


def main() -> int:
    args = build_parser().parse_args()
    if args.validate_only:
        validation = validate_candidate_csv(
            args.input_csv or args.out_csv,
            min_valid_candidates=int(args.min_valid_candidates),
        )
        if args.out_summary:
            write_json(args.out_summary, {"csv_validation": validation})
        print("[CLEAR_RF_CSV_VALIDATION]")
        print(json.dumps(to_jsonable(validation), indent=2, sort_keys=True))
        return 0 if validation.get("validation_pass") else 2

    pool_path = Path(args.clear_pool).expanduser()
    if not pool_path.exists():
        raise FileNotFoundError(f"CLEAR pool not found: {pool_path}")
    rows = read_jsonl(pool_path, max_records=args.max_records)
    schema_summary = summarize_schema(rows, adj_threshold=float(args.adj_threshold))
    candidates = [
        candidate_from_record(row, dataset=args.dataset, record_index=index, adj_threshold=float(args.adj_threshold))
        for index, row in enumerate(rows)
    ]
    valid_rows = [row for row in candidates if row.get("candidate_valid") is True]
    csv_rows = candidates if args.include_invalid else valid_rows
    invalid_reasons = Counter(str(row.get("invalid_reason")) for row in candidates if row.get("candidate_valid") is not True)
    valid_rate = (len(valid_rows) / len(candidates)) if candidates else 0.0
    conversion_quality_gate_pass = (
        len(valid_rows) >= int(args.min_valid_candidates)
        and valid_rate >= float(args.min_valid_rate)
    )
    write_csv(args.out_csv, csv_rows)
    write_jsonl(args.out_jsonl, candidates)
    csv_validation = validate_candidate_csv(args.out_csv, min_valid_candidates=int(args.min_valid_candidates))
    validation_gate_pass = bool(csv_validation.get("validation_pass"))
    quality_gate_pass = bool(conversion_quality_gate_pass and validation_gate_pass)
    summary = {
        "dataset": args.dataset,
        "clear_pool": str(pool_path),
        "out_csv": args.out_csv,
        "out_jsonl": args.out_jsonl,
        "out_summary": args.out_summary,
        "num_input_records": len(rows),
        "num_detail_records": len(candidates),
        "num_valid_candidates": len(valid_rows),
        "num_invalid_candidates": len(candidates) - len(valid_rows),
        "num_csv_rows": len(csv_rows),
        "include_invalid": bool(args.include_invalid),
        "valid_rate": valid_rate,
        "invalid_reason_counts": dict(invalid_reasons),
        "min_valid_candidates": int(args.min_valid_candidates),
        "min_valid_rate": float(args.min_valid_rate),
        "conversion_quality_gate_pass": bool(conversion_quality_gate_pass),
        "quality_gate_pass": bool(quality_gate_pass),
        "csv_validation": csv_validation,
        "csv_validation_pass": validation_gate_pass,
        "adj_threshold": float(args.adj_threshold),
        **schema_summary,
        "mean_edge_changed": mean([float(row["edge_changed_count"]) for row in candidates if row.get("edge_changed_count") is not None]),
        "mean_feature_l1_cost": mean([float(row["feature_l1_cost"]) for row in candidates if row.get("feature_l1_cost") is not None]),
        "mean_total_action_cost": mean([float(row["total_action_cost"]) for row in candidates if row.get("total_action_cost") is not None]),
        "total_attempted_bonds": sum(int(row.get("attempted_bonds") or 0) for row in candidates),
        "total_skipped_bonds_for_valence": sum(int(row.get("skipped_bonds_for_valence") or 0) for row in candidates),
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
    if not (quality_gate_pass and validation_gate_pass):
        print("[CLEAR_RF_CONVERT_FAILED_QUALITY_GATE]")
        return 2
    print("[CLEAR_RF_CONVERT_QUALITY_GATE_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
