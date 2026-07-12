#!/usr/bin/env python3
"""Export greedy top-K summaries for GCFExplainer-HIVCSV candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.gcf_hiv_csv_dataset import HIVCSVGraphDataset  # noqa: E402
from src.baselines.gcf_hiv_csv_model import torch_load  # noqa: E402
from scripts.gcf_hiv_csv_run_vrrw import graph_distance_proxy  # noqa: E402


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _indices_from_sparse(value: Any) -> set[int]:
    try:
        import torch

        if torch.is_tensor(value):
            if value.is_sparse:
                value = value.coalesce()
                return {int(v) for v in value.indices()[0].detach().cpu().tolist()}
            return {int(v) for v in torch.where(value.detach().cpu().reshape(-1) > 0)[0].tolist()}
    except Exception:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(v) for v in value}
    return set()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite_float_or_default(value: Any, default: float) -> float:
    if value is None or value == "":
        return float(default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _greedy(records: list[dict[str, Any]], max_k: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    covered: set[int] = set()
    remaining = records[:]
    for rank in range(1, max_k + 1):
        if not remaining:
            break
        best_i = 0
        best_key: tuple[int, int, float] | None = None
        for idx, row in enumerate(remaining):
            gain = len(set(row["covered_indices"]) - covered)
            key = (
                gain,
                int(row.get("frequency") or 0),
                -_finite_float_or_default(row.get("min_distance_seen"), 999.0),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_i = idx
        chosen = dict(remaining.pop(best_i))
        previous_covered = len(covered)
        covered.update(chosen["covered_indices"])
        chosen["selected_rank"] = rank
        chosen["marginal_coverage_gain"] = len(covered) - previous_covered
        chosen["covered_count_at_rank"] = len(covered)
        selected.append(chosen)
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-dir", default="outputs/hpc/gcfexplainer_hiv_csv/dataset")
    parser.add_argument("--gnn-dir", default="outputs/hpc/gcfexplainer_hiv_csv/gnn")
    parser.add_argument("--top-k-list", default="1,5,10")
    parser.add_argument("--eval-theta", type=float, default=0.1)
    parser.add_argument("--target-label", type=int, default=1)
    parser.add_argument("--out-dir", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    import torch

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_dir / "summary_export"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = torch_load(str(run_dir / "counterfactuals.pt"), map_location="cpu")
    dataset = HIVCSVGraphDataset(args.dataset_dir)
    parent_indices = list(payload.get("target_parent_indices") or [i for i, g in enumerate(dataset.graphs) if int(g.y.item()) == int(args.target_label)])
    candidates = payload.get("counterfactual_candidates") or []
    graph_map = payload.get("graph_map") if isinstance(payload.get("graph_map"), dict) else {}
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(candidates):
        if not isinstance(row, dict):
            continue
        graph_hash = str(row.get("graph_hash") or "")
        graph = graph_map.get(graph_hash)
        if graph is None:
            continue
        records.append(
            {
                "candidate_id": f"gcf_hiv_csv_{idx}",
                "graph_hash": graph_hash,
                "frequency": int(row.get("frequency") or 0),
                "covered_indices": _indices_from_sparse(row.get("input_graphs_covering_list")),
                "min_distance_seen": _finite_float_or_default(row.get("min_distance_seen"), 999.0),
                "candidate_pred": row.get("candidate_pred", ""),
                "candidate_score_label1": row.get("candidate_score_label1", ""),
                "graph": graph,
            }
        )
    top_k_list = _parse_ints(args.top_k_list)
    selected = _greedy(records, max(top_k_list) if top_k_list else len(records))
    selected_graphs = [row["graph"] for row in selected]
    selected_records = [{k: v for k, v in row.items() if k not in {"graph", "covered_indices"}} | {"covered_indices": sorted(row["covered_indices"])} for row in selected]
    selected_path = out_dir / "selected_counterfactual_graphs.pt"
    torch.save(
        {
            "selected_graphs": selected_graphs,
            "selected_records": selected_records,
            "GCF_MODE": "hiv_csv_adapted",
            "DATASET_SOURCE": "HIV_CSV",
            "CF_MODE": "strict_flip",
            "dataset_dir": str(Path(args.dataset_dir).expanduser().resolve()),
            "gnn_dir": str(Path(args.gnn_dir).expanduser().resolve()),
        },
        selected_path,
    )

    metadata_rows: list[dict[str, Any]] = []
    for row in selected_records:
        metadata_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "rank": row["selected_rank"],
                "graph_hash": row["graph_hash"],
                "frequency": row["frequency"],
                "candidate_pred": row["candidate_pred"],
                "min_distance_seen": row["min_distance_seen"],
                "covered_count": len(row["covered_indices"]),
                "GCF_MODE": "hiv_csv_adapted",
                "DATASET_SOURCE": "HIV_CSV",
                "CF_MODE": "strict_flip",
            }
        )
    _write_csv(
        out_dir / "selected_counterfactual_metadata.csv",
        metadata_rows,
        ["candidate_id", "rank", "graph_hash", "frequency", "candidate_pred", "min_distance_seen", "covered_count", "GCF_MODE", "DATASET_SOURCE", "CF_MODE"],
    )

    summary_rows: list[dict[str, Any]] = []
    parent_graphs = [dataset[i] for i in parent_indices]
    for k in top_k_list:
        subset = selected[:k]
        covered: set[int] = set()
        best_distances: list[float] = []
        for parent_pos, parent_graph in enumerate(parent_graphs):
            distances = [graph_distance_proxy(parent_graph, row["graph"]) for row in subset if int(row.get("candidate_pred") or -1) != int(args.target_label)]
            close = [dist for dist in distances if dist <= float(args.eval_theta)]
            if close:
                covered.add(parent_pos)
                best_distances.append(min(close))
        summary_rows.append(
            {
                "method": "GCFExplainer-HIVCSV",
                "k": k,
                "eval_theta": float(args.eval_theta),
                "covered_count": len(covered),
                "coverage": len(covered) / len(parent_graphs) if parent_graphs else 0.0,
                "median_cost": median(best_distances) if best_distances else "",
                "avg_cost": (sum(best_distances) / len(best_distances)) if best_distances else "",
                "selected_graphs_path": str(selected_path),
                "GCF_MODE": "hiv_csv_adapted",
                "DATASET_SOURCE": "HIV_CSV",
                "CF_MODE": "strict_flip",
            }
        )
    _write_csv(
        out_dir / "native_coverage_cost_by_k.csv",
        summary_rows,
        ["method", "k", "eval_theta", "covered_count", "coverage", "median_cost", "avg_cost", "selected_graphs_path", "GCF_MODE", "DATASET_SOURCE", "CF_MODE"],
    )
    summary = {
        "method": "GCFExplainer-HIVCSV",
        "GCF_MODE": "hiv_csv_adapted",
        "DATASET_SOURCE": "HIV_CSV",
        "CF_MODE": "strict_flip",
        "num_candidates_raw": len(records),
        "num_selected": len(selected),
        "top_k_list": top_k_list,
        "selected_counterfactual_graphs_pt": str(selected_path),
        "rows": summary_rows,
    }
    (out_dir / "selected_counterfactual_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[GCF_HIV_CSV_EXPORT_DONE]", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
