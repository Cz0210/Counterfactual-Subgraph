#!/usr/bin/env python3
"""Convert CLEAR per-instance exports into a unified candidate/action pool.

The CLEAR export contains paired original and counterfactual graph arrays.
This script keeps CLEAR's official prediction fields for diagnostics while
materializing action-level edge and feature diffs for later unified evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ADJ_ALIASES = ("original_adj", "adj", "adj_input")
CF_ADJ_ALIASES = ("cf_adj", "counterfactual_adj", "adj_reconst_binary")
X_ALIASES = ("original_x", "original_features", "features")
CF_X_ALIASES = ("cf_x", "cf_features_reconst", "counterfactual_x", "features_reconst")
INSTANCE_ALIASES = ("instance_index", "orin_index", "graph_id")
PROXIMITY_ALIASES = ("proximity", "Proximity")
PROXIMITY_X_ALIASES = ("proximity_x", "proximity_X", "Proximity_x", "Proximity_X")
PROXIMITY_A_ALIASES = ("proximity_a", "proximity_A", "Proximity_a", "Proximity_A")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert CLEAR export_test pickle files into a JSONL candidate/action "
            "pool for unified CCRCov/action-rule evaluation."
        )
    )
    parser.add_argument(
        "--export-dir",
        default="outputs/hpc/baselines/clear/ogbg_molhiv/test_exports",
        help="Directory containing clear_<dataset>_exp*_test_counterfactuals.pkl files.",
    )
    parser.add_argument("--dataset", default="ogbg_molhiv", help="CLEAR dataset name.")
    parser.add_argument(
        "--out-jsonl",
        default=(
            "outputs/hpc/baselines/clear/ogbg_molhiv/candidate_pool/"
            "clear_ogbg_molhiv_candidate_pool.jsonl"
        ),
        help="Output candidate pool JSONL path.",
    )
    parser.add_argument(
        "--out-summary",
        default=(
            "outputs/hpc/baselines/clear/ogbg_molhiv/candidate_pool/"
            "clear_ogbg_molhiv_candidate_pool_summary.json"
        ),
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--prefer-exp",
        default="all",
        help="Experiment ids to include: all, one id like 0, or comma-separated ids like 0,1.",
    )
    parser.add_argument(
        "--deduplicate-by",
        default="none",
        choices=("none", "instance_index"),
        help="Optional deduplication key. Default keeps all exp-level candidates.",
    )
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap on output candidates.")
    parser.add_argument(
        "--include-full-graphs",
        action="store_true",
        help="Include original/counterfactual adjacency and feature arrays in JSONL.",
    )
    parser.add_argument(
        "--filter-official-flip",
        action="store_true",
        help="Keep only candidates that flip under CLEAR's own predictor. Default keeps non-flips.",
    )
    parser.add_argument(
        "--feature-change-eps",
        type=float,
        default=1e-6,
        help="Tolerance for treating a node feature vector as changed.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Accepted for Slurm/config compatibility; not used by this conversion script.",
    )
    return parser.parse_args()


def get_first(record: dict[str, Any], aliases: Iterable[str], missing: Counter[str], logical_name: str) -> Any:
    for key in aliases:
        if key in record:
            return record[key]
    missing[logical_name] += 1
    return None


def to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    return arr


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    arr = to_numpy(value)
    if arr is None or arr.size == 0:
        return None
    try:
        return int(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return None


def to_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    arr = to_numpy(value)
    if arr is None:
        return None
    return [float(x) for x in arr.reshape(-1).tolist()]


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    arr = to_numpy(value)
    if arr is None or arr.size == 0:
        return None
    try:
        out = float(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def safe_metric(record: dict[str, Any], aliases: Iterable[str]) -> Any:
    for key in aliases:
        if key in record:
            return to_jsonable(record[key])
    return None


def infer_exp_id(path: Path, record: dict[str, Any]) -> int | None:
    value = to_int(record.get("exp_id"))
    if value is not None:
        return value
    match = re.search(r"_exp(\d+)_", path.name)
    if match:
        return int(match.group(1))
    return None


def parse_prefer_exp(value: str) -> set[int] | None:
    if value.strip().lower() == "all":
        return None
    exp_ids: set[int] = set()
    for part in value.split(","):
        part = part.strip().lower().removeprefix("exp")
        if not part:
            continue
        exp_ids.add(int(part))
    return exp_ids


def find_export_files(export_dir: Path, dataset: str, prefer_exp: str) -> list[Path]:
    selected_exp_ids = parse_prefer_exp(prefer_exp)
    files = sorted(export_dir.glob(f"clear_{dataset}_exp*_test_counterfactuals.pkl"))
    if selected_exp_ids is None:
        return files

    selected: list[Path] = []
    for path in files:
        match = re.search(r"_exp(\d+)_", path.name)
        if match and int(match.group(1)) in selected_exp_ids:
            selected.append(path)
    return selected


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported CLEAR export pickle payload in {path}: {type(payload)!r}")


def binary_adjacency(value: Any) -> np.ndarray | None:
    arr = to_numpy(value)
    if arr is None:
        return None
    if arr.ndim != 2:
        return None
    return (arr > 0.5).astype(np.int8)


def infer_node_count(record: dict[str, Any], original_adj: np.ndarray | None, original_x: np.ndarray | None) -> int | None:
    for key in ("original_num_nodes", "num_nodes", "num_node_real"):
        value = to_int(record.get(key))
        if value is not None and value > 0:
            return value
    max_num_nodes = to_int(record.get("max_num_nodes"))
    if max_num_nodes is not None and max_num_nodes > 0:
        return max_num_nodes
    if original_adj is not None:
        return int(original_adj.shape[0])
    if original_x is not None and original_x.ndim >= 1:
        return int(original_x.shape[0])
    return None


def compute_edge_diff(
    original_adj_value: Any,
    cf_adj_value: Any,
    num_nodes: int | None,
) -> tuple[dict[str, Any], str | None]:
    original_adj = binary_adjacency(original_adj_value)
    cf_adj = binary_adjacency(cf_adj_value)
    if original_adj is None or cf_adj is None:
        return {
            "num_edge_added": None,
            "num_edge_deleted": None,
            "num_edge_changed": None,
            "edge_cost": None,
            "action_edges_added": [],
            "action_edges_deleted": [],
        }, "missing_or_invalid_adjacency"

    n = min(original_adj.shape[0], original_adj.shape[1], cf_adj.shape[0], cf_adj.shape[1])
    if num_nodes is not None:
        n = min(n, max(0, int(num_nodes)))

    added: list[list[int]] = []
    deleted: list[list[int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            before = int(original_adj[i, j])
            after = int(cf_adj[i, j])
            if before == 0 and after == 1:
                added.append([i, j])
            elif before == 1 and after == 0:
                deleted.append([i, j])

    num_added = len(added)
    num_deleted = len(deleted)
    num_changed = num_added + num_deleted
    return {
        "num_edge_added": num_added,
        "num_edge_deleted": num_deleted,
        "num_edge_changed": num_changed,
        "edge_cost": num_changed,
        "action_edges_added": added,
        "action_edges_deleted": deleted,
    }, None


def compute_feature_diff(
    original_x_value: Any,
    cf_x_value: Any,
    num_nodes: int | None,
    eps: float,
) -> tuple[dict[str, Any], str | None]:
    original_x = to_numpy(original_x_value)
    cf_x = to_numpy(cf_x_value)
    if original_x is None or cf_x is None or original_x.ndim < 2 or cf_x.ndim < 2:
        return {
            "feature_l1_cost": None,
            "feature_l2_cost": None,
            "num_node_feature_changed": None,
            "changed_node_indices": [],
            "changed_node_feature_pairs": [],
        }, "missing_or_invalid_features"

    n = min(original_x.shape[0], cf_x.shape[0])
    if num_nodes is not None:
        n = min(n, max(0, int(num_nodes)))
    d = min(original_x.shape[1], cf_x.shape[1])

    original = original_x[:n, :d].astype(float, copy=False)
    cf = cf_x[:n, :d].astype(float, copy=False)
    diff = cf - original
    row_l1 = np.abs(diff).sum(axis=1) if n > 0 else np.asarray([])
    row_l2 = np.sqrt((diff * diff).sum(axis=1)) if n > 0 else np.asarray([])
    changed_mask = row_l1 > eps
    changed_indices = np.where(changed_mask)[0].astype(int).tolist()
    changed_pairs = [
        {
            "node_index": int(idx),
            "original_features": original[idx].tolist(),
            "cf_features": cf[idx].tolist(),
            "l1": float(row_l1[idx]),
            "l2": float(row_l2[idx]),
        }
        for idx in changed_indices
    ]

    return {
        "feature_l1_cost": float(np.abs(diff).sum()),
        "feature_l2_cost": float(np.sqrt((diff * diff).sum())),
        "num_node_feature_changed": len(changed_indices),
        "changed_node_indices": changed_indices,
        "changed_node_feature_pairs": changed_pairs,
    }, None


def build_candidate(
    *,
    record: dict[str, Any],
    path: Path,
    dataset: str,
    output_index: int,
    missing: Counter[str],
    include_full_graphs: bool,
    feature_eps: float,
) -> dict[str, Any]:
    exp_id = infer_exp_id(path, record)
    instance_index = get_first(record, INSTANCE_ALIASES, missing, "instance_index")
    original_label = to_int(record.get("original_label"))
    target_cf_label = to_int(record.get("target_cf_label"))
    original_pred_label = to_int(record.get("original_pred_label"))
    cf_pred_label = to_int(record.get("cf_pred_label"))
    original_pred_prob = to_float_list(record.get("original_pred_prob"))
    cf_pred_prob = to_float_list(record.get("cf_pred_prob"))
    original_adj = get_first(record, ADJ_ALIASES, missing, "original_adj")
    cf_adj = get_first(record, CF_ADJ_ALIASES, missing, "cf_adj")
    original_x = get_first(record, X_ALIASES, missing, "original_x")
    cf_x = get_first(record, CF_X_ALIASES, missing, "cf_x")
    original_adj_binary = binary_adjacency(original_adj)
    original_x_np = to_numpy(original_x)
    num_nodes = infer_node_count(record, original_adj_binary, original_x_np)

    edge_diff, edge_error = compute_edge_diff(original_adj, cf_adj, num_nodes)
    feature_diff, feature_error = compute_feature_diff(original_x, cf_x, num_nodes, feature_eps)
    edge_cost = edge_diff["edge_cost"]
    feature_l1_cost = feature_diff["feature_l1_cost"]
    total_cost = None
    if edge_cost is not None and feature_l1_cost is not None:
        total_cost = float(edge_cost) + float(feature_l1_cost)

    official_flip = (
        bool(original_pred_label != cf_pred_label)
        if original_pred_label is not None and cf_pred_label is not None
        else None
    )
    official_target_success = (
        bool(cf_pred_label == target_cf_label)
        if cf_pred_label is not None and target_cf_label is not None
        else None
    )
    official_original_correct = (
        bool(original_pred_label == original_label)
        if original_pred_label is not None and original_label is not None
        else None
    )

    instance_id = to_int(instance_index)
    if instance_id is None:
        instance_id = output_index
    candidate = {
        "candidate_id": f"CLEAR_{dataset}_exp{exp_id}_idx{instance_id}_cand{output_index:06d}",
        "source": "CLEAR",
        "dataset": record.get("dataset", dataset),
        "exp_id": exp_id,
        "split": record.get("split", "test"),
        "instance_index": instance_id,
        "orin_index": to_int(record.get("orin_index")),
        "original_label": original_label,
        "target_cf_label": target_cf_label,
        "official_original_pred_label": original_pred_label,
        "official_cf_pred_label": cf_pred_label,
        "official_flip": official_flip,
        "official_target_success": official_target_success,
        "official_original_correct": official_original_correct,
        "official_original_pred_prob": original_pred_prob,
        "official_cf_pred_prob": cf_pred_prob,
        **edge_diff,
        **feature_diff,
        "total_cost": total_cost,
        "causality": safe_metric(record, ("causality", "Causality")),
        "validity": safe_metric(record, ("validity", "Validity")),
        "proximity": safe_metric(record, PROXIMITY_ALIASES),
        "proximity_x": safe_metric(record, PROXIMITY_X_ALIASES),
        "proximity_a": safe_metric(record, PROXIMITY_A_ALIASES),
    }
    errors = [err for err in (edge_error, feature_error) if err]
    if errors:
        candidate["conversion_warnings"] = errors

    if include_full_graphs:
        candidate.update(
            {
                "original_adj": to_jsonable(original_adj),
                "cf_adj": to_jsonable(cf_adj),
                "original_x": to_jsonable(original_x),
                "cf_x": to_jsonable(cf_x),
            }
        )
    return to_jsonable(candidate)


def bool_count(values: Iterable[Any]) -> int:
    return sum(1 for value in values if value is True)


def mean_or_none(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return float(sum(numeric) / len(numeric))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_preview(candidates: list[dict[str, Any]]) -> None:
    print("[CLEAR_CONVERT_PREVIEW]")
    for candidate in candidates[:3]:
        compact = {
            "candidate_id": candidate.get("candidate_id"),
            "exp_id": candidate.get("exp_id"),
            "instance_index": candidate.get("instance_index"),
            "official_flip": candidate.get("official_flip"),
            "num_edge_added": candidate.get("num_edge_added"),
            "num_edge_deleted": candidate.get("num_edge_deleted"),
            "feature_l1_cost": candidate.get("feature_l1_cost"),
            "total_cost": candidate.get("total_cost"),
        }
        print(json.dumps(compact, sort_keys=True))


def main() -> int:
    args = parse_args()
    export_dir = Path(args.export_dir).expanduser()
    out_jsonl = Path(args.out_jsonl).expanduser()
    out_summary = Path(args.out_summary).expanduser()
    files = find_export_files(export_dir, args.dataset, args.prefer_exp)
    if not files:
        raise FileNotFoundError(
            f"No CLEAR export pickle files found in {export_dir} for dataset={args.dataset}. "
            f"Expected pattern: clear_{args.dataset}_exp*_test_counterfactuals.pkl"
        )

    print("[CLEAR_CONVERT_CONFIG]")
    print(f"export_dir={export_dir}")
    print(f"dataset={args.dataset}")
    print(f"out_jsonl={out_jsonl}")
    print(f"out_summary={out_summary}")
    print(f"prefer_exp={args.prefer_exp}")
    print(f"deduplicate_by={args.deduplicate_by}")
    print(f"max_records={args.max_records}")
    print(f"include_full_graphs={args.include_full_graphs}")
    print(f"filter_official_flip={args.filter_official_flip}")
    print(f"num_export_files={len(files)}")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    missing: Counter[str] = Counter()
    count_by_exp: Counter[str] = Counter()
    seen_instances: set[int] = set()
    duplicate_instance_count = 0
    num_input_records = 0
    candidates_written = 0
    preview: list[dict[str, Any]] = []
    summary_values: dict[str, list[Any]] = defaultdict(list)

    with out_jsonl.open("w", encoding="utf-8") as handle:
        for path in files:
            print(f"[CLEAR_CONVERT_LOAD] {path}")
            records = load_records(path)
            for record in records:
                num_input_records += 1
                candidate = build_candidate(
                    record=record,
                    path=path,
                    dataset=args.dataset,
                    output_index=candidates_written,
                    missing=missing,
                    include_full_graphs=args.include_full_graphs,
                    feature_eps=args.feature_change_eps,
                )
                if args.filter_official_flip and candidate.get("official_flip") is not True:
                    continue

                instance_index = candidate.get("instance_index")
                if args.deduplicate_by == "instance_index" and instance_index is not None:
                    if int(instance_index) in seen_instances:
                        duplicate_instance_count += 1
                        continue
                    seen_instances.add(int(instance_index))

                handle.write(json.dumps(candidate, sort_keys=True) + "\n")
                if len(preview) < 3:
                    preview.append(candidate)
                candidates_written += 1
                count_by_exp[str(candidate.get("exp_id"))] += 1

                for key in (
                    "official_flip",
                    "official_target_success",
                    "official_original_correct",
                    "num_edge_changed",
                    "num_edge_added",
                    "num_edge_deleted",
                    "feature_l1_cost",
                    "feature_l2_cost",
                    "total_cost",
                ):
                    summary_values[key].append(candidate.get(key))

                if args.max_records is not None and candidates_written >= args.max_records:
                    break
            if args.max_records is not None and candidates_written >= args.max_records:
                break

    official_flip_count = bool_count(summary_values["official_flip"])
    official_target_success_count = bool_count(summary_values["official_target_success"])
    official_original_correct_count = bool_count(summary_values["official_original_correct"])
    denom = candidates_written if candidates_written > 0 else 1
    summary = {
        "dataset": args.dataset,
        "source": "CLEAR",
        "export_dir": str(export_dir),
        "out_jsonl": str(out_jsonl),
        "num_input_records": num_input_records,
        "num_output_candidates": candidates_written,
        "count_by_exp": dict(sorted(count_by_exp.items())),
        "official_flip_count": official_flip_count,
        "official_flip_rate": official_flip_count / denom,
        "official_target_success_count": official_target_success_count,
        "official_target_success_rate": official_target_success_count / denom,
        "official_original_correct_count": official_original_correct_count,
        "official_original_correct_rate": official_original_correct_count / denom,
        "mean_edge_changed": mean_or_none(summary_values["num_edge_changed"]),
        "mean_edge_added": mean_or_none(summary_values["num_edge_added"]),
        "mean_edge_deleted": mean_or_none(summary_values["num_edge_deleted"]),
        "mean_feature_l1_cost": mean_or_none(summary_values["feature_l1_cost"]),
        "mean_feature_l2_cost": mean_or_none(summary_values["feature_l2_cost"]),
        "mean_total_cost": mean_or_none(summary_values["total_cost"]),
        "missing_field_counts": dict(sorted(missing.items())),
        "deduplicate_by": args.deduplicate_by,
        "duplicate_instance_count": duplicate_instance_count,
        "filter_official_flip": args.filter_official_flip,
        "include_full_graphs": args.include_full_graphs,
        "input_files": [str(path) for path in files],
    }
    write_json(out_summary, summary)
    print_preview(preview)
    print("[CLEAR_CONVERT_DONE]")
    print(f"num_input_records={num_input_records}")
    print(f"num_output_candidates={candidates_written}")
    print(f"out_jsonl={out_jsonl}")
    print(f"out_summary={out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
