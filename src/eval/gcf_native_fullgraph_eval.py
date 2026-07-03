"""Native official GCFExplainer fullgraph evaluation.

This evaluator intentionally stays in the official GCFExplainer graph space:
official GNN predictions determine strict counterfactual status, and official
NeuroSED normalized graph distance provides the native distance.  It does not
use NetworkX GED for fullgraph pairwise evaluation.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence


CF_MODE = "strict_flip"
GCF_MODE = "official_native"
TEACHER_TYPE = "official_gnn"
DISTANCE_TYPE = "official_native"


@contextmanager
def official_import_context(official_repo: str | Path) -> Iterable[Path]:
    repo = Path(official_repo).expanduser().resolve()
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    try:
        yield repo
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def _load_torch() -> Any:
    try:
        import torch

        return torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Official native GCF evaluation requires PyTorch.") from exc


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_rate(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _load_selected_graphs(path: str | Path) -> tuple[list[Any], list[dict[str, Any]]]:
    torch = _load_torch()
    payload = torch.load(Path(path).expanduser().resolve(), map_location="cpu")
    if isinstance(payload, dict):
        graphs = payload.get("selected_graphs") or payload.get("graphs") or []
        records = payload.get("selected_records") or []
        return list(graphs), [dict(row) for row in records if isinstance(row, dict)]
    if isinstance(payload, list):
        return payload, []
    raise TypeError(f"Unsupported selected graph payload type: {type(payload).__name__}")


def _predict_graphs(graphs: list[Any], model: Any, device: str, batch_size: int = 256) -> tuple[list[int], list[float]]:
    torch = _load_torch()
    from torch_geometric.data import DataLoader

    if not graphs:
        return [], []
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    labels: list[int] = []
    probs: list[float] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(device))[-1]
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            probabilities = torch.exp(logits) if float(logits.max().detach().cpu()) <= 0.0 else torch.softmax(logits, dim=-1)
            pred = torch.argmax(probabilities, dim=-1)
            labels.extend(int(v) for v in pred.detach().cpu().tolist())
            if probabilities.shape[-1] > 1:
                probs.extend(float(v) for v in probabilities[:, 1].detach().cpu().tolist())
            else:
                probs.extend(float(v) for v in probabilities.reshape(-1).detach().cpu().tolist())
    return labels, probs


def _normalise_distance_matrix(original_graphs: list[Any], candidates: list[Any], matrix: Any) -> Any:
    import util

    torch = _load_torch()
    original_counts = util.graph_element_counts(original_graphs)
    candidate_counts = util.graph_element_counts(candidates)
    denom = torch.cartesian_prod(candidate_counts, original_counts).sum(dim=1).view(
        len(candidate_counts), len(original_counts)
    )
    normalized = matrix / denom
    return normalized.T


def evaluate_native_fullgraph(
    *,
    official_repo: str | Path,
    selected_graphs_path: str | Path,
    out_dir: str | Path,
    dataset: str = "aids",
    direction: str = "official_default_label0_to_label1",
    thresholds: Sequence[float] = (0.05, 0.1, 0.2),
    top_k_list: Sequence[int] = (1, 5, 10, 20, 50, 100),
    device: str = "cuda:0",
    batch_size: int = 256,
) -> dict[str, Any]:
    torch = _load_torch()
    output = Path(out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_graphs, selected_records = _load_selected_graphs(selected_graphs_path)
    max_k = max(top_k_list) if top_k_list else len(selected_graphs)
    selected_graphs = selected_graphs[:max_k]
    selected_records = selected_records[:max_k]

    with official_import_context(official_repo) as repo:
        from data import load_dataset
        import distance
        from gnn import load_trained_gnn, load_trained_prediction

        dataset_graphs = load_dataset(dataset)
        model = load_trained_gnn(dataset, device=device)
        preds = load_trained_prediction(dataset, device=device).detach().cpu()
        parent_label = 0 if direction == "official_default_label0_to_label1" else 1
        parent_indices = torch.where(preds == int(parent_label))[0].detach().cpu().tolist()
        parent_graphs = dataset_graphs[parent_indices]
        candidate_preds, candidate_prob1 = _predict_graphs(selected_graphs, model, device=device, batch_size=batch_size)
        neurosed = distance.load_neurosed(
            parent_graphs,
            neurosed_model_path=f"data/{dataset}/neurosed/best_model.pt",
            device=device,
        )
        distance_raw = neurosed.predict_outer_with_queries(selected_graphs, batch_size=1000).detach().cpu()
        distance_matrix = _normalise_distance_matrix(parent_graphs, selected_graphs, distance_raw).detach().cpu()

    details: list[dict[str, Any]] = []
    for parent_pos, original_index in enumerate(parent_indices):
        for cand_idx, graph in enumerate(selected_graphs):
            dist = float(distance_matrix[parent_pos, cand_idx].item())
            pred_after = candidate_preds[cand_idx] if cand_idx < len(candidate_preds) else None
            details.append(
                {
                    "parent_position": parent_pos,
                    "official_parent_index": original_index,
                    "parent_label": parent_label,
                    "candidate_id": selected_records[cand_idx].get("candidate_id", f"gcf_official_{cand_idx}") if cand_idx < len(selected_records) else f"gcf_official_{cand_idx}",
                    "candidate_rank": selected_records[cand_idx].get("rank", cand_idx + 1) if cand_idx < len(selected_records) else cand_idx + 1,
                    "candidate_pred": pred_after,
                    "candidate_prob_label1": candidate_prob1[cand_idx] if cand_idx < len(candidate_prob1) else "",
                    "distance": dist,
                    "cf_flip": bool(pred_after is not None and int(pred_after) != int(parent_label)),
                    "CF_MODE": CF_MODE,
                    "TEACHER_TYPE": TEACHER_TYPE,
                    "GCF_MODE": GCF_MODE,
                    "DISTANCE_TYPE": DISTANCE_TYPE,
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for k in top_k_list:
        for theta in thresholds:
            covered: set[int] = set()
            best_distances: list[float] = []
            for parent_pos in range(len(parent_indices)):
                rows = [
                    row
                    for row in details
                    if int(row["parent_position"]) == parent_pos and int(row["candidate_rank"]) <= int(k)
                ]
                eligible = [row for row in rows if bool(row["cf_flip"]) and float(row["distance"]) <= float(theta)]
                if not eligible:
                    continue
                eligible.sort(key=lambda row: float(row["distance"]))
                covered.add(parent_pos)
                best_distances.append(float(eligible[0]["distance"]))
            summary_rows.append(
                {
                    "method": "GCFExplainerOfficial",
                    "dataset": dataset,
                    "direction": direction,
                    "k": int(k),
                    "theta": float(theta),
                    "num_parents": len(parent_indices),
                    "num_candidates": min(int(k), len(selected_graphs)),
                    "covered_count": len(covered),
                    "coverage": _safe_rate(len(covered), len(parent_indices)),
                    "median_cost": median(best_distances) if best_distances else "",
                    "avg_cost": _mean(best_distances) if best_distances else "",
                    "CF_MODE": CF_MODE,
                    "TEACHER_TYPE": TEACHER_TYPE,
                    "GCF_MODE": GCF_MODE,
                    "DISTANCE_TYPE": DISTANCE_TYPE,
                    "selected_graphs_path": str(Path(selected_graphs_path).expanduser().resolve()),
                }
            )

    _write_csv(
        output / "native_ccrcov_details.csv",
        details,
        [
            "parent_position",
            "official_parent_index",
            "parent_label",
            "candidate_id",
            "candidate_rank",
            "candidate_pred",
            "candidate_prob_label1",
            "distance",
            "cf_flip",
            "CF_MODE",
            "TEACHER_TYPE",
            "GCF_MODE",
            "DISTANCE_TYPE",
        ],
    )
    _write_csv(
        output / "native_ccrcov_summary.csv",
        summary_rows,
        [
            "method",
            "dataset",
            "direction",
            "k",
            "theta",
            "num_parents",
            "num_candidates",
            "covered_count",
            "coverage",
            "median_cost",
            "avg_cost",
            "CF_MODE",
            "TEACHER_TYPE",
            "GCF_MODE",
            "DISTANCE_TYPE",
            "selected_graphs_path",
        ],
    )
    config = {
        "CF_MODE": CF_MODE,
        "TEACHER_TYPE": TEACHER_TYPE,
        "GCF_MODE": GCF_MODE,
        "DISTANCE_TYPE": DISTANCE_TYPE,
        "dataset": dataset,
        "direction": direction,
        "thresholds": list(thresholds),
        "top_k_list": list(top_k_list),
        "num_parents": len(parent_indices),
        "num_selected_candidates": len(selected_graphs),
        "selected_graphs_path": str(Path(selected_graphs_path).expanduser().resolve()),
        "official_repo": str(Path(official_repo).expanduser().resolve()),
    }
    (output / "native_eval_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return {"config": config, "summary_rows": summary_rows, "details_rows": len(details)}

