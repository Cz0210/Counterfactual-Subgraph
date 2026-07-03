#!/usr/bin/env python3
"""Export top-K official GCFExplainer full counterfactual graph summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_torch() -> Any:
    try:
        import torch

        return torch
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("GCF official export requires PyTorch to read counterfactuals.pt") from exc


def _to_jsonable(value: Any, *, max_items: int = 20) -> Any:
    torch = sys.modules.get("torch")
    if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(value):
        if value.numel() <= max_items:
            return value.detach().cpu().tolist()
        return {
            "tensor_shape": list(value.shape),
            "tensor_dtype": str(value.dtype),
            "tensor_sum": float(value.detach().float().sum().cpu().item()),
        }
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v, max_items=max_items) for k, v in list(value.items())[:max_items]}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v, max_items=max_items) for v in list(value)[:max_items]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            "type": type(value).__name__,
            "attrs": _to_jsonable(vars(value), max_items=max_items),
        }
    return repr(value)


def _stable_hash(value: Any) -> str:
    payload = json.dumps(_to_jsonable(value), sort_keys=True, ensure_ascii=False, default=repr)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _indices_from_covering(value: Any) -> set[int]:
    if value is None:
        return set()
    torch = sys.modules.get("torch")
    try:
        if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(value):
            tensor = value
            if tensor.is_sparse:
                tensor = tensor.coalesce()
                if tensor._nnz() == 0:
                    return set()
                return {int(i) for i in tensor.indices()[0].detach().cpu().tolist()}
            if tensor.ndim == 0:
                return {0} if bool(tensor.item()) else set()
            return {int(i) for i in torch.where(tensor.detach().cpu().reshape(-1) > 0)[0].tolist()}
    except Exception:
        return set()
    if isinstance(value, dict):
        for key in ("indices", "covered_indices", "input_graph_indices"):
            if key in value:
                return _indices_from_covering(value[key])
        return {int(k) for k, v in value.items() if v}
    if isinstance(value, (list, tuple, set)):
        indices: set[int] = set()
        for item in value:
            try:
                indices.add(int(item))
            except Exception:
                continue
        return indices
    return set()


def _numeric_field(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except Exception:
            continue
        if math.isfinite(number):
            return number
    return None


def _importance_value(candidate: dict[str, Any]) -> float | None:
    parts = candidate.get("importance_parts")
    if isinstance(parts, (list, tuple)) and parts:
        try:
            return float(parts[0])
        except Exception:
            return None
    try:
        return float(parts)
    except Exception:
        return None


def _candidate_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("counterfactual_candidates")
    graph_map = payload.get("graph_map") if isinstance(payload.get("graph_map"), dict) else {}
    if not isinstance(candidates, list):
        return []
    records: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        graph_hash = str(candidate.get("graph_hash") or _stable_hash(candidate))
        graph = graph_map.get(candidate.get("graph_hash")) or graph_map.get(graph_hash)
        covering = _indices_from_covering(
            candidate.get("input_graphs_covering_list")
            or candidate.get("covered_indices")
            or candidate.get("covering")
        )
        cost = _numeric_field(candidate, ("cost", "distance", "dist", "ged", "ged_norm"))
        records.append(
            {
                "candidate_id": f"gcf_official_{index}",
                "source_index": index,
                "frequency": int(candidate.get("frequency") or 0),
                "graph_hash": graph_hash,
                "importance_value": _importance_value(candidate),
                "covered_indices": covering,
                "coverage_available": bool(covering),
                "cost": cost,
                "graph": graph,
                "raw_candidate": candidate,
            }
        )
    return records


def _greedy_select(records: list[dict[str, Any]], max_k: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    covered: set[int] = set()
    remaining = records[:]
    for rank in range(1, max_k + 1):
        if not remaining:
            break
        best_i = 0
        best_key: tuple[int, int, float] | None = None
        for idx, record in enumerate(remaining):
            gain = len(set(record["covered_indices"]) - covered)
            key = (gain, int(record["frequency"]), float(record.get("importance_value") or -1.0))
            if best_key is None or key > best_key:
                best_key = key
                best_i = idx
        chosen = remaining.pop(best_i)
        covered.update(chosen["covered_indices"])
        chosen = dict(chosen)
        chosen["rank"] = rank
        chosen["covered_count_at_rank"] = len(covered)
        selected.append(chosen)
    return selected


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_top_k(value: str) -> list[int]:
    return sorted({int(part.strip()) for part in value.split(",") if part.strip()})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--counterfactuals-path", default=None)
    parser.add_argument("--top-k-list", default="1,5,10,20,50,100")
    parser.add_argument("--eval-theta", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--train-theta", type=float, default=None)
    parser.add_argument("--out-dir", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    torch = _load_torch()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_dir / "summary_export"
    out_dir.mkdir(parents=True, exist_ok=True)
    counterfactuals_path = (
        Path(args.counterfactuals_path).expanduser().resolve()
        if args.counterfactuals_path
        else run_dir / "counterfactuals.pt"
    )
    run_config = _read_json(run_dir / "gcf_official_run_config.json")
    alpha = args.alpha if args.alpha is not None else run_config.get("alpha")
    train_theta = args.train_theta if args.train_theta is not None else run_config.get("train_theta")
    top_k_list = _parse_top_k(args.top_k_list)
    max_k = max(top_k_list) if top_k_list else 0

    payload = torch.load(counterfactuals_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected official counterfactuals.pt to contain dict, got {type(payload).__name__}")
    records = _candidate_records(payload)
    eligible = [
        record
        for record in records
        if record.get("importance_value") is None or float(record.get("importance_value") or 0.0) >= 0.5
    ]
    eligible.sort(key=lambda row: (int(row.get("frequency") or 0), float(row.get("importance_value") or -1.0)), reverse=True)
    selected = _greedy_select(eligible, max_k=max_k)
    selected_with_graph = [row for row in selected if row.get("graph") is not None]
    selected_graphs = [row["graph"] for row in selected_with_graph]
    selected_records_light = [
        {key: value for key, value in row.items() if key not in {"graph", "raw_candidate", "covered_indices"}}
        | {"covered_indices": sorted(row.get("covered_indices") or [])}
        for row in selected_with_graph
    ]

    selected_pt = out_dir / "selected_counterfactual_graphs.pt"
    torch.save(
        {
            "selected_graphs": selected_graphs,
            "selected_records": selected_records_light,
            "source_counterfactuals_path": str(counterfactuals_path),
            "selection_policy": "greedy_coverage_gain_with_frequency_fallback",
            "GCF_MODE": "official_native",
            "CF_MODE": "strict_flip",
        },
        selected_pt,
    )

    metadata_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    total_coverable = 0
    for row in records:
        total_coverable = max(total_coverable, *(row.get("covered_indices") or {0}))
    total_coverable = total_coverable + 1 if total_coverable else 0

    for k in top_k_list:
        subset = selected[:k]
        covered: set[int] = set()
        costs: list[float] = []
        for row in subset:
            covered.update(row.get("covered_indices") or set())
            if row.get("cost") is not None:
                costs.append(float(row["cost"]))
            metadata_rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "rank": row["rank"],
                    "frequency": row["frequency"],
                    "graph_hash": row["graph_hash"],
                    "alpha": alpha,
                    "train_theta": train_theta,
                    "eval_theta": args.eval_theta,
                    "k": k,
                    "covered_count": len(covered),
                    "coverage": (len(covered) / total_coverable) if total_coverable else "",
                    "median_cost": median(costs) if costs else "",
                    "avg_cost": (sum(costs) / len(costs)) if costs else "",
                }
            )
        summary_rows.append(
            {
                "alpha": alpha,
                "train_theta": train_theta,
                "eval_theta": args.eval_theta,
                "k": k,
                "covered_count": len(covered),
                "coverage": (len(covered) / total_coverable) if total_coverable else "",
                "median_cost": median(costs) if costs else "",
                "avg_cost": (sum(costs) / len(costs)) if costs else "",
                "selected_graphs_path": str(selected_pt),
                "run_dir": str(run_dir),
            }
        )

    _write_csv(
        out_dir / "selected_counterfactual_metadata.csv",
        metadata_rows,
        [
            "candidate_id",
            "rank",
            "frequency",
            "graph_hash",
            "alpha",
            "train_theta",
            "eval_theta",
            "k",
            "covered_count",
            "coverage",
            "median_cost",
            "avg_cost",
        ],
    )
    _write_csv(
        out_dir / "native_coverage_cost_by_k.csv",
        summary_rows,
        ["alpha", "train_theta", "eval_theta", "k", "covered_count", "coverage", "median_cost", "avg_cost", "run_dir", "selected_graphs_path"],
    )
    summary = {
        "GCF_MODE": "official_native",
        "CF_MODE": "strict_flip",
        "source_counterfactuals_path": str(counterfactuals_path),
        "num_raw_candidates": len(records),
        "num_eligible_candidates": len(eligible),
        "num_selected": len(selected),
        "num_selected_with_graph": len(selected_with_graph),
        "coverage_available_count": sum(1 for row in records if row.get("coverage_available")),
        "selected_counterfactual_graphs_pt": str(selected_pt),
        "top_k_list": top_k_list,
        "rows": summary_rows,
    }
    (out_dir / "selected_counterfactual_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("[GCF_OFFICIAL_EXPORT_DONE]", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
