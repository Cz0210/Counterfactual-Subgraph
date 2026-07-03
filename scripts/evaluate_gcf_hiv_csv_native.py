#!/usr/bin/env python3
"""Native fullgraph close-CF evaluation for GCFExplainer-HIVCSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.gcf_hiv_csv_dataset import HIVCSVGraphDataset  # noqa: E402
from src.baselines.gcf_hiv_csv_model import torch_load  # noqa: E402
from scripts.gcf_hiv_csv_run_vrrw import _load_model, _predict_graphs, graph_distance_proxy  # noqa: E402


def _parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dataset-dir", default="outputs/hpc/gcfexplainer_hiv_csv/dataset")
    parser.add_argument("--gnn-dir", default="outputs/hpc/gcfexplainer_hiv_csv/gnn")
    parser.add_argument("--selected-graphs", required=True)
    parser.add_argument("--selected-metadata", default="")
    parser.add_argument("--theta-list", default="0.05,0.10,0.20")
    parser.add_argument("--target-label", type=int, default=1)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    import torch

    device = args.device if not (args.device == "cuda" and not torch.cuda.is_available()) else "cpu"
    dataset = HIVCSVGraphDataset(args.dataset_dir)
    model = _load_model(dataset, Path(args.gnn_dir).expanduser().resolve(), device)
    payload = torch_load(str(Path(args.selected_graphs).expanduser().resolve()), map_location="cpu")
    selected_graphs = list(payload.get("selected_graphs") or [])
    selected_records = [dict(row) for row in payload.get("selected_records") or [] if isinstance(row, dict)]
    candidate_preds, candidate_scores = _predict_graphs(model, selected_graphs, device=device)
    parent_indices = [idx for idx, graph in enumerate(dataset.graphs) if int(graph.y.item()) == int(args.target_label)]
    parent_graphs = [dataset[idx] for idx in parent_indices]

    details: list[dict[str, Any]] = []
    for parent_pos, parent_idx in enumerate(parent_indices):
        parent_graph = parent_graphs[parent_pos]
        for cand_idx, candidate in enumerate(selected_graphs):
            record = selected_records[cand_idx] if cand_idx < len(selected_records) else {}
            pred_after = candidate_preds[cand_idx] if cand_idx < len(candidate_preds) else None
            details.append(
                {
                    "method": "GCFExplainer-HIVCSV",
                    "parent_position": parent_pos,
                    "parent_index": parent_idx,
                    "target_label": int(args.target_label),
                    "candidate_id": record.get("candidate_id", f"gcf_hiv_csv_{cand_idx}"),
                    "candidate_rank": int(record.get("selected_rank") or record.get("rank") or cand_idx + 1),
                    "pred_after": pred_after,
                    "candidate_score_label1": candidate_scores[cand_idx] if cand_idx < len(candidate_scores) else "",
                    "distance": graph_distance_proxy(parent_graph, candidate),
                    "cf_flip": bool(pred_after is not None and int(pred_after) != int(args.target_label)),
                    "GCF_MODE": "hiv_csv_adapted",
                    "DATASET_SOURCE": "HIV_CSV",
                    "TEACHER_TYPE": "hiv_csv_gnn",
                    "CF_MODE": "strict_flip",
                }
            )

    theta_list = _parse_floats(args.theta_list)
    ranks = sorted({int(row["candidate_rank"]) for row in details})
    summary_rows: list[dict[str, Any]] = []
    for k in ranks:
        for theta in theta_list:
            covered: set[int] = set()
            best_distances: list[float] = []
            flip_values: list[float] = []
            for parent_pos in range(len(parent_indices)):
                rows = [
                    row
                    for row in details
                    if int(row["parent_position"]) == parent_pos
                    and int(row["candidate_rank"]) <= int(k)
                    and bool(row["cf_flip"])
                    and float(row["distance"]) <= float(theta)
                ]
                if not rows:
                    continue
                rows.sort(key=lambda row: float(row["distance"]))
                covered.add(parent_pos)
                best_distances.append(float(rows[0]["distance"]))
                flip_values.append(1.0)
            summary_rows.append(
                {
                    "method": "GCFExplainer-HIVCSV",
                    "GCF_MODE": "hiv_csv_adapted",
                    "DATASET_SOURCE": "HIV_CSV",
                    "TEACHER_TYPE": "hiv_csv_gnn",
                    "CF_MODE": "strict_flip",
                    "threshold": float(theta),
                    "k": int(k),
                    "num_parents": len(parent_indices),
                    "num_candidates": min(int(k), len(selected_graphs)),
                    "num_close_cf_covered": len(covered),
                    "close_cf_coverage": len(covered) / len(parent_indices) if parent_indices else 0.0,
                    "avg_best_distance": (sum(best_distances) / len(best_distances)) if best_distances else "",
                    "median_best_distance": median(best_distances) if best_distances else "",
                    "flip_rate_among_covered": (sum(flip_values) / len(flip_values)) if flip_values else "",
                }
            )

    out_dir = Path(args.out_dir).expanduser().resolve()
    _write_csv(
        out_dir / "native_ccrcov_details.csv",
        details,
        [
            "method",
            "parent_position",
            "parent_index",
            "target_label",
            "candidate_id",
            "candidate_rank",
            "pred_after",
            "candidate_score_label1",
            "distance",
            "cf_flip",
            "GCF_MODE",
            "DATASET_SOURCE",
            "TEACHER_TYPE",
            "CF_MODE",
        ],
    )
    _write_csv(
        out_dir / "native_ccrcov_summary.csv",
        summary_rows,
        [
            "method",
            "GCF_MODE",
            "DATASET_SOURCE",
            "TEACHER_TYPE",
            "CF_MODE",
            "threshold",
            "k",
            "num_parents",
            "num_candidates",
            "num_close_cf_covered",
            "close_cf_coverage",
            "avg_best_distance",
            "median_best_distance",
            "flip_rate_among_covered",
        ],
    )
    config = {
        "method": "GCFExplainer-HIVCSV",
        "GCF_MODE": "hiv_csv_adapted",
        "DATASET_SOURCE": "HIV_CSV",
        "TEACHER_TYPE": "hiv_csv_gnn",
        "CF_MODE": "strict_flip",
        "dataset_dir": str(Path(args.dataset_dir).expanduser().resolve()),
        "gnn_dir": str(Path(args.gnn_dir).expanduser().resolve()),
        "selected_graphs": str(Path(args.selected_graphs).expanduser().resolve()),
        "theta_list": theta_list,
        "target_label": int(args.target_label),
        "num_parents": len(parent_indices),
        "num_candidates": len(selected_graphs),
    }
    (out_dir / "native_eval_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("[GCF_HIV_CSV_NATIVE_EVAL_DONE]", flush=True)
    print(json.dumps(config, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

