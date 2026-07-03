#!/usr/bin/env python3
"""Run a lightweight GCF-style VRRW on the adapted HIVCSV graph dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.gcf_hiv_csv_dataset import HIVCSVGraphDataset  # noqa: E402
from src.baselines.gcf_hiv_csv_model import build_gcf_style_gnn, load_torch_stack, torch_load  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _edge_pairs(graph: Any) -> set[tuple[int, int]]:
    edges = graph.edge_index.detach().cpu().t().tolist()
    pairs: set[tuple[int, int]] = set()
    for i, j in edges:
        if int(i) == int(j):
            continue
        pairs.add(tuple(sorted((int(i), int(j)))))
    return pairs


def _graph_hash(graph: Any) -> str:
    payload = {
        "x": graph.x.detach().cpu().numpy().tobytes().hex(),
        "edge_index": graph.edge_index.detach().cpu().numpy().tobytes().hex(),
        "num_nodes": int(graph.num_nodes),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _unique_edge_data(graph: Any) -> tuple[list[tuple[int, int]], dict[tuple[int, int], list[float]]]:
    edges = graph.edge_index.detach().cpu().t().tolist()
    attrs = graph.edge_attr.detach().cpu().tolist() if getattr(graph, "edge_attr", None) is not None else []
    seen: list[tuple[int, int]] = []
    attr_by_pair: dict[tuple[int, int], list[float]] = {}
    for idx, (src, dst) in enumerate(edges):
        pair = tuple(sorted((int(src), int(dst))))
        if pair[0] == pair[1] or pair in attr_by_pair:
            continue
        seen.append(pair)
        attr_by_pair[pair] = list(attrs[idx]) if idx < len(attrs) else [1.0, 0.0, 0.0, 0.0, 0.0]
    return seen, attr_by_pair


def _rebuild_edges(graph: Any, pairs: list[tuple[int, int]], attr_by_pair: dict[tuple[int, int], list[float]]) -> Any:
    torch, _F, _DataLoader, _GCNConv, _global_max_pool = load_torch_stack()
    edge_rows: list[list[int]] = []
    edge_attrs: list[list[float]] = []
    for a, b in pairs:
        attr = attr_by_pair.get((a, b), [1.0, 0.0, 0.0, 0.0, 0.0])
        edge_rows.append([a, b])
        edge_rows.append([b, a])
        edge_attrs.append(attr)
        edge_attrs.append(attr)
    graph.edge_index = torch.tensor(edge_rows, dtype=torch.long).t().contiguous() if edge_rows else torch.empty((2, 0), dtype=torch.long)
    bond_dim = int(graph.edge_attr.shape[1]) if getattr(graph, "edge_attr", None) is not None and graph.edge_attr.ndim == 2 else 5
    graph.edge_attr = torch.tensor(edge_attrs, dtype=torch.float32) if edge_attrs else torch.empty((0, bond_dim), dtype=torch.float32)
    graph.bond_types = ["SINGLE" for _ in pairs]
    return graph


def _mutate_graph(graph: Any, *, rng: random.Random, atom_symbols: list[str], max_tries: int = 20) -> tuple[Any, str]:
    torch, _F, _DataLoader, _GCNConv, _global_max_pool = load_torch_stack()
    num_nodes = int(graph.num_nodes)
    if num_nodes <= 0:
        return graph.clone(), "empty_noop"
    choices = ["node_label_change", "edge_add"]
    if len(_edge_pairs(graph)) > 0:
        choices.append("edge_delete")
    action = rng.choice(choices)
    candidate = graph.clone()
    if action == "node_label_change":
        node = rng.randrange(num_nodes)
        current = int(torch.argmax(candidate.x[node]).item())
        if candidate.x.shape[1] <= 1:
            return candidate, "node_label_noop"
        new_label = rng.randrange(int(candidate.x.shape[1]))
        if new_label == current:
            new_label = (new_label + 1) % int(candidate.x.shape[1])
        candidate.x[node] = 0
        candidate.x[node, new_label] = 1
        symbols = list(getattr(candidate, "atom_symbols", []))
        if len(symbols) == num_nodes and new_label < len(atom_symbols):
            symbols[node] = atom_symbols[new_label]
            candidate.atom_symbols = symbols
        return candidate, f"node_label_change:{node}:{current}->{new_label}"
    pairs, attr_by_pair = _unique_edge_data(candidate)
    if action == "edge_delete" and pairs:
        remove = rng.choice(pairs)
        pairs = [pair for pair in pairs if pair != remove]
        attr_by_pair.pop(remove, None)
        return _rebuild_edges(candidate, pairs, attr_by_pair), f"edge_delete:{remove[0]}-{remove[1]}"
    existing = set(pairs)
    for _ in range(max_tries):
        a = rng.randrange(num_nodes)
        b = rng.randrange(num_nodes)
        if a == b:
            continue
        pair = tuple(sorted((a, b)))
        if pair in existing:
            continue
        pairs.append(pair)
        attr_by_pair[pair] = [1.0, 0.0, 0.0, 0.0, 0.0]
        return _rebuild_edges(candidate, pairs, attr_by_pair), f"edge_add:{pair[0]}-{pair[1]}"
    return candidate, "edge_add_noop"


def graph_distance_proxy(a: Any, b: Any) -> float:
    torch, _F, _DataLoader, _GCNConv, _global_max_pool = load_torch_stack()
    edges_a = _edge_pairs(a)
    edges_b = _edge_pairs(b)
    node_diff = abs(int(a.num_nodes) - int(b.num_nodes))
    edge_diff = len(edges_a ^ edges_b)
    hist_a = a.x.detach().cpu().sum(dim=0)
    hist_b = b.x.detach().cpu().sum(dim=0)
    hist_dim = max(hist_a.numel(), hist_b.numel())
    if hist_a.numel() < hist_dim:
        hist_a = torch.cat([hist_a, torch.zeros(hist_dim - hist_a.numel(), dtype=hist_a.dtype)])
    if hist_b.numel() < hist_dim:
        hist_b = torch.cat([hist_b, torch.zeros(hist_dim - hist_b.numel(), dtype=hist_b.dtype)])
    hist_diff = float((hist_a - hist_b).abs().sum().item())
    denom = max(1.0, float(int(a.num_nodes) + int(b.num_nodes) + len(edges_a) + len(edges_b)))
    return float((node_diff + edge_diff + hist_diff) / denom)


def _predict_graphs(model: Any, graphs: list[Any], device: str, batch_size: int = 256) -> tuple[list[int], list[float]]:
    torch, F, DataLoader, _GCNConv, _global_max_pool = load_torch_stack()
    model.eval()
    preds: list[int] = []
    scores: list[float] = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            logits = model(batch.to(device))[-1]
            probs = F.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)
            preds.extend(int(v) for v in pred.detach().cpu().tolist())
            scores.extend(float(v) for v in probs[:, 1].detach().cpu().tolist())
    return preds, scores


def _load_model(dataset: HIVCSVGraphDataset, gnn_dir: Path, device: str) -> Any:
    summary = _read_json(gnn_dir / "gnn_train_summary.json")
    arch = summary.get("model_architecture") if isinstance(summary.get("model_architecture"), dict) else {}
    model = build_gcf_style_gnn(
        int(dataset.num_features),
        int(dataset.num_classes),
        num_layers=int(arch.get("num_layers", 3)),
        dim=int(arch.get("dim", 20)),
        dropout=float(arch.get("dropout", 0.0)),
        device=device,
    )
    state = torch_load(str(gnn_dir / "model_best.pth"), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


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
    parser.add_argument("--run-dir", default="outputs/hpc/gcfexplainer_hiv_csv/smoke/alpha_0.5_theta_0.05_steps_200")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--theta", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--teleport", type=float, default=0.1)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--target-label", type=int, default=1)
    parser.add_argument("--counterfactual-label", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    torch, _F, _DataLoader, _GCNConv, _global_max_pool = load_torch_stack()
    device = args.device if not (args.device == "cuda" and not torch.cuda.is_available()) else "cpu"
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    gnn_dir = Path(args.gnn_dir).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_summary = _read_json(dataset_dir / "dataset_summary.json")
    print("[GCF_HIV_CSV_CONFIG]", flush=True)
    print("DATASET_SOURCE=HIV_CSV", flush=True)
    print("GCF_MODE=hiv_csv_adapted", flush=True)
    print("CF_MODE=strict_flip", flush=True)
    print(f"CSV_PATH={dataset_summary.get('csv_path', '')}", flush=True)
    print(f"TARGET_LABEL={args.target_label}", flush=True)
    print(f"COUNTERFACTUAL_LABEL={args.counterfactual_label}", flush=True)
    rng = random.Random(int(args.seed))
    dataset = HIVCSVGraphDataset(dataset_dir)
    model = _load_model(dataset, gnn_dir, device)
    parent_indices = [idx for idx, graph in enumerate(dataset.graphs) if int(graph.y.item()) == int(args.target_label)]
    if not parent_indices:
        raise ValueError(f"No graphs with target_label={args.target_label}")
    atom_vocab = dataset_summary.get("atom_vocab") if isinstance(dataset_summary.get("atom_vocab"), dict) else {}
    atom_symbols = [symbol for symbol, _idx in sorted(atom_vocab.items(), key=lambda item: int(item[1]))]
    if not atom_symbols:
        atom_symbols = [str(i) for i in range(int(dataset.num_features))]

    graph_map: dict[str, Any] = {}
    candidates: dict[str, dict[str, Any]] = {}
    metadata_rows: list[dict[str, Any]] = []
    log_lines: list[str] = []
    started = time.time()
    for step in range(1, int(args.max_steps) + 1):
        parent_global = rng.choice(parent_indices)
        parent_graph = dataset[parent_global]
        best_for_step: tuple[Any, str, int, float, float] | None = None
        for _ in range(max(1, min(int(args.sample_size), 512))):
            candidate_graph, action = _mutate_graph(parent_graph, rng=rng, atom_symbols=atom_symbols)
            pred, score = _predict_graphs(model, [candidate_graph], device=device, batch_size=1)
            pred_label = int(pred[0])
            if pred_label == int(args.target_label):
                continue
            if args.counterfactual_label is not None and pred_label != int(args.counterfactual_label):
                continue
            distance = graph_distance_proxy(parent_graph, candidate_graph)
            if best_for_step is None or distance < best_for_step[4]:
                best_for_step = (candidate_graph, action, pred_label, float(score[0]), distance)
        if best_for_step is None:
            continue
        candidate_graph, action, pred_label, score_label1, distance = best_for_step
        graph_hash = _graph_hash(candidate_graph)
        graph_map[graph_hash] = candidate_graph
        record = candidates.setdefault(
            graph_hash,
            {
                "graph_hash": graph_hash,
                "frequency": 0,
                "importance_parts": (1.0, 1.0),
                "input_graphs_covering_list": set(),
                "first_action": action,
                "candidate_pred": pred_label,
                "candidate_score_label1": score_label1,
                "min_distance_seen": distance,
            },
        )
        record["frequency"] += 1
        record["min_distance_seen"] = min(float(record.get("min_distance_seen", distance)), distance)
        if distance <= float(args.theta):
            record["input_graphs_covering_list"].add(parent_indices.index(parent_global))
        if step % max(1, int(args.max_steps) // 10) == 0:
            line = f"[GCF_HIV_CSV_PROGRESS] step={step} candidates={len(candidates)} elapsed={time.time() - started:.1f}s"
            print(line, flush=True)
            log_lines.append(line)

    counterfactual_candidates: list[dict[str, Any]] = []
    for rank, (graph_hash, record) in enumerate(sorted(candidates.items(), key=lambda item: int(item[1]["frequency"]), reverse=True)):
        covering = torch.zeros(len(parent_indices), dtype=torch.float32)
        for local_idx in record["input_graphs_covering_list"]:
            covering[int(local_idx)] = 1.0
        out_record = dict(record)
        out_record["input_graphs_covering_list"] = covering.to_sparse()
        out_record["rank_by_frequency"] = rank + 1
        counterfactual_candidates.append(out_record)
        metadata_rows.append(
            {
                "candidate_id": f"gcf_hiv_csv_{rank}",
                "rank": rank + 1,
                "graph_hash": graph_hash,
                "frequency": int(record["frequency"]),
                "candidate_pred": record["candidate_pred"],
                "candidate_score_label1": record["candidate_score_label1"],
                "min_distance_seen": record["min_distance_seen"],
                "covered_count": len(record["input_graphs_covering_list"]),
                "first_action": record["first_action"],
                "GCF_MODE": "hiv_csv_adapted",
                "DATASET_SOURCE": "HIV_CSV",
                "CF_MODE": "strict_flip",
            }
        )
    torch.save(
        {
            "graph_map": graph_map,
            "counterfactual_candidates": counterfactual_candidates,
            "target_parent_indices": parent_indices,
            "dataset_dir": str(dataset_dir),
            "gnn_dir": str(gnn_dir),
            "GCF_MODE": "hiv_csv_adapted",
            "DATASET_SOURCE": "HIV_CSV",
            "CF_MODE": "strict_flip",
        },
        run_dir / "counterfactuals.pt",
    )
    _write_csv(
        run_dir / "candidate_metadata.csv",
        metadata_rows,
        [
            "candidate_id",
            "rank",
            "graph_hash",
            "frequency",
            "candidate_pred",
            "candidate_score_label1",
            "min_distance_seen",
            "covered_count",
            "first_action",
            "GCF_MODE",
            "DATASET_SOURCE",
            "CF_MODE",
        ],
    )
    config = {
        "GCF_MODE": "hiv_csv_adapted",
        "DATASET_SOURCE": "HIV_CSV",
        "CF_MODE": "strict_flip",
        "CSV_PATH": dataset_summary.get("csv_path", ""),
        "dataset_dir": str(dataset_dir),
        "gnn_dir": str(gnn_dir),
        "run_dir": str(run_dir),
        "alpha": float(args.alpha),
        "theta": float(args.theta),
        "max_steps": int(args.max_steps),
        "teleport": float(args.teleport),
        "sample_size": int(args.sample_size),
        "target_label": int(args.target_label),
        "counterfactual_label": args.counterfactual_label,
        "device": device,
        "num_target_parents": len(parent_indices),
        "num_candidates": len(counterfactual_candidates),
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (run_dir / "vrrw_command.json").write_text(json.dumps({"argv": sys.argv}, indent=2) + "\n", encoding="utf-8")
    (run_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    (run_dir / "vrrw_stdout.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    (run_dir / "vrrw_stderr.log").write_text("", encoding="utf-8")
    print("[GCF_HIV_CSV_DONE]", flush=True)
    print(json.dumps(config, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
