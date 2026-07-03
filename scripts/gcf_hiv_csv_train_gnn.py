#!/usr/bin/env python3
"""Train a GCF-style GNN teacher on the project HIV.csv graph dataset.

This adapted path is for the canonical project CSV
``data/raw/AIDS/HIV.csv`` converted to ``graphs.pt``.  It reads only the
project-prepared graph file and does not download external data.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.gcf_hiv_csv_dataset import HIVCSVGraphDataset  # noqa: E402


def _load_torch_stack() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import DataLoader
        from torch_geometric.nn import GCNConv, global_max_pool

        return torch, F, DataLoader, GCNConv, global_max_pool
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "gcf_hiv_csv_train_gnn.py requires torch and torch_geometric at runtime. "
            "Run it on HPC in the smiles_pip118 environment."
        ) from exc


def _counter(labels: Sequence[int]) -> dict[str, int]:
    counts = Counter(int(label) for label in labels)
    return {str(label): int(counts.get(label, 0)) for label in sorted(counts)}


def stratified_split_indices(
    labels: Sequence[int],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[int]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(int(label), []).append(index)
    splits = {"train": [], "val": [], "test": []}
    for label, indices in sorted(by_label.items()):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        if n_train + n_val > n_total:
            n_val = max(0, n_total - n_train)
        train = shuffled[:n_train]
        val = shuffled[n_train : n_train + n_val]
        test = shuffled[n_train + n_val :]
        if n_total >= 3:
            if not val:
                val = [train.pop()]
            if not test:
                test = [train.pop()]
        splits["train"].extend(train)
        splits["val"].extend(val)
        splits["test"].extend(test)
    for values in splits.values():
        rng.shuffle(values)
    return splits


def class_weights_from_labels(labels: Sequence[int], num_classes: int) -> list[float]:
    counts = Counter(int(label) for label in labels)
    total = len(labels)
    weights: list[float] = []
    for label in range(num_classes):
        count = counts.get(label, 0)
        weights.append(float(total / (num_classes * count)) if count else 0.0)
    return weights


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_score_pos: Sequence[float]) -> dict[str, Any]:
    labels = sorted(set(int(v) for v in y_true) | set(int(v) for v in y_pred) | {0, 1})
    per_class: dict[str, dict[str, float]] = {}
    recalls: list[float] = []
    f1s: list[float] = []
    correct = sum(1 for a, b in zip(y_true, y_pred) if int(a) == int(b))
    for label in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if int(a) == label and int(b) == label)
        fp = sum(1 for a, b in zip(y_true, y_pred) if int(a) != label and int(b) == label)
        fn = sum(1 for a, b in zip(y_true, y_pred) if int(a) == label and int(b) != label)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        per_class[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for value in y_true if int(value) == label),
        }
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": _safe_div(correct, len(y_true)),
        "per_class": per_class,
        "macro_f1": float(sum(f1s) / len(f1s)) if f1s else 0.0,
        "balanced_accuracy": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "y_true_counts": _counter(y_true),
        "y_pred_counts": _counter(y_pred),
        "positive_pred_rate": _safe_div(sum(1 for value in y_pred if int(value) == 1), len(y_pred)),
        "roc_auc": _binary_auc(y_true, y_score_pos),
    }


def _binary_auc(y_true: Sequence[int], y_score_pos: Sequence[float]) -> float | None:
    pairs = [(float(score), int(label)) for score, label in zip(y_score_pos, y_true)]
    positives = sum(1 for _, label in pairs if label == 1)
    negatives = sum(1 for _, label in pairs if label == 0)
    if positives == 0 or negatives == 0:
        return None
    pairs.sort(key=lambda item: item[0])
    rank_sum = 0.0
    rank = 1
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        rank += j - i
        i = j
    auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def build_model(num_features: int, num_classes: int, *, num_layers: int, dim: int, dropout: float, device: str) -> Any:
    torch, F, _DataLoader, GCNConv, global_max_pool = _load_torch_stack()

    class GCFStyleGNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.convs = torch.nn.ModuleList()
            self.bns = torch.nn.ModuleList()
            self.convs.append(GCNConv(num_features, dim))
            self.bns.append(torch.nn.BatchNorm1d(dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(dim, dim))
                self.bns.append(torch.nn.BatchNorm1d(dim))
            self.fc = torch.nn.Linear(dim, num_classes)

        def forward(self, data: Any) -> tuple[Any, Any, Any]:
            x = data.x
            edge_index = data.edge_index
            batch = data.batch
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=dropout, training=self.training)
            node_embeddings = x
            graph_embeddings = global_max_pool(node_embeddings, batch)
            logits = self.fc(graph_embeddings)
            return node_embeddings, graph_embeddings, logits

    return GCFStyleGNN().to(device)


def _subset(dataset: HIVCSVGraphDataset, indices: Sequence[int]) -> list[Any]:
    return [dataset[int(index)] for index in indices]


def train_one_epoch(model: Any, loader: Any, optimizer: Any, criterion: Any, device: str) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)[-1]
        loss = criterion(logits, batch.y.long())
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(batch.num_graphs)
        total_graphs += int(batch.num_graphs)
    return _safe_div(total_loss, total_graphs)


def evaluate(model: Any, loader: Any, criterion: Any, device: str) -> dict[str, Any]:
    torch, F, _DataLoader, _GCNConv, _global_max_pool = _load_torch_stack()
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score_pos: list[float] = []
    losses: list[float] = []
    graph_embeddings: list[Any] = []
    logits_all: list[Any] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _node_emb, graph_emb, logits = model(batch)
            loss = criterion(logits, batch.y.long())
            probs = F.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)
            y_true.extend(int(v) for v in batch.y.detach().cpu().tolist())
            y_pred.extend(int(v) for v in pred.detach().cpu().tolist())
            if probs.shape[-1] > 1:
                y_score_pos.extend(float(v) for v in probs[:, 1].detach().cpu().tolist())
            else:
                y_score_pos.extend(float(v) for v in probs.reshape(-1).detach().cpu().tolist())
            losses.append(float(loss.item()) * int(batch.num_graphs))
            graph_embeddings.append(graph_emb.detach().cpu())
            logits_all.append(logits.detach().cpu())
    metrics = classification_metrics(y_true, y_pred, y_score_pos)
    metrics["loss"] = _safe_div(sum(losses), len(y_true))
    metrics["graph_embeddings"] = torch.cat(graph_embeddings, dim=0) if graph_embeddings else torch.empty((0, 0))
    metrics["logits"] = torch.cat(logits_all, dim=0) if logits_all else torch.empty((0, 0))
    metrics["preds"] = torch.tensor(y_pred, dtype=torch.long)
    return metrics


def _selection_score(metrics: dict[str, Any], prefer: str) -> float:
    if prefer == "macro_f1":
        return float(metrics.get("macro_f1") or 0.0)
    if prefer == "balanced_accuracy":
        return float(metrics.get("balanced_accuracy") or 0.0)
    if prefer == "label1_recall":
        return float(metrics.get("per_class", {}).get("1", {}).get("recall") or 0.0)
    if prefer == "loss":
        return -float(metrics.get("loss") or math.inf)
    raise ValueError(f"unsupported checkpoint metric: {prefer}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dataset-dir", default="outputs/hpc/gcfexplainer_hiv_csv/dataset")
    parser.add_argument("--out-dir", default="outputs/hpc/gcfexplainer_hiv_csv/gnn")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--checkpoint-metric",
        choices=["macro_f1", "balanced_accuracy", "label1_recall", "loss"],
        default="macro_f1",
    )
    parser.add_argument("--label1-recall-warning-threshold", type=float, default=0.2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    torch, _F, DataLoader, _GCNConv, _global_max_pool = _load_torch_stack()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    dataset = HIVCSVGraphDataset(args.dataset_dir)
    labels = [int(graph.y.item()) for graph in dataset.graphs]
    splits = stratified_split_indices(
        labels,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )
    train_labels = [labels[index] for index in splits["train"]]
    weights = class_weights_from_labels(train_labels, int(dataset.num_classes))
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device) if args.use_class_weights else None
    print("[GCF_HIV_CSV_GNN_CONFIG]", flush=True)
    print("GCF_MODE=hiv_csv_adapted", flush=True)
    print("DATASET_SOURCE=HIV_CSV", flush=True)
    print("CF_MODE=strict_flip", flush=True)
    print("[GCF_HIV_CSV_CLASS_BALANCE]", flush=True)
    print(f"train_label_counts={_counter(train_labels)}", flush=True)
    print(f"class_weights={weights}", flush=True)

    model = build_model(
        int(dataset.num_features),
        int(dataset.num_classes),
        num_layers=int(args.num_layers),
        dim=int(args.dim),
        dropout=float(args.dropout),
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    loaders = {
        "train": DataLoader(_subset(dataset, splits["train"]), batch_size=int(args.batch_size), shuffle=True),
        "val": DataLoader(_subset(dataset, splits["val"]), batch_size=int(args.batch_size), shuffle=False),
        "test": DataLoader(_subset(dataset, splits["test"]), batch_size=int(args.batch_size), shuffle=False),
        "all": DataLoader(dataset.graphs, batch_size=int(args.batch_size), shuffle=False),
    }

    best_score = -math.inf
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        score = _selection_score(val_metrics, args.checkpoint_metric)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_label1_recall": val_metrics["per_class"].get("1", {}).get("recall"),
            "checkpoint_score": score,
        }
        history.append(row)
        print(
            "[GCF_HIV_CSV_GNN_EPOCH] "
            f"epoch={epoch} train_loss={train_loss:.6f} val_macro_f1={val_metrics['macro_f1']:.6f} "
            f"val_balanced_accuracy={val_metrics['balanced_accuracy']:.6f} val_label1_recall={row['val_label1_recall']}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model_best.pth")
    final_metrics = {split: evaluate(model, loaders[split], criterion, device) for split in ("train", "val", "test", "all")}
    torch.save(final_metrics["all"]["preds"], out_dir / "preds.pt")
    torch.save(final_metrics["all"]["logits"], out_dir / "logits.pt")
    torch.save(final_metrics["all"]["graph_embeddings"], out_dir / "graph_embeddings.pt")
    split_payload = {
        "seed": int(args.seed),
        "stratified": True,
        "train_indices": splits["train"],
        "val_indices": splits["val"],
        "test_indices": splits["test"],
        "train_label_counts": _counter([labels[i] for i in splits["train"]]),
        "val_label_counts": _counter([labels[i] for i in splits["val"]]),
        "test_label_counts": _counter([labels[i] for i in splits["test"]]),
    }
    (out_dir / "train_val_test_split.json").write_text(json.dumps(split_payload, indent=2) + "\n", encoding="utf-8")

    label1_recall = final_metrics["test"]["per_class"].get("1", {}).get("recall", 0.0)
    warnings: list[str] = []
    if label1_recall < float(args.label1_recall_warning_threshold):
        warnings.append(
            f"label1_recall_low:{label1_recall:.6f}<threshold:{float(args.label1_recall_warning_threshold):.6f}"
        )
    summary = {
        "GCF_MODE": "hiv_csv_adapted",
        "DATASET_SOURCE": "HIV_CSV",
        "TEACHER_TYPE": "hiv_csv_gnn",
        "CF_MODE": "strict_flip",
        "dataset_dir": str(Path(args.dataset_dir).expanduser().resolve()),
        "out_dir": str(out_dir),
        "num_graphs": len(dataset),
        "num_features": int(dataset.num_features),
        "num_classes": int(dataset.num_classes),
        "model_architecture": {
            "num_layers": int(args.num_layers),
            "dim": int(args.dim),
            "dropout": float(args.dropout),
        },
        "split": split_payload,
        "use_class_weights": bool(args.use_class_weights),
        "class_weights": weights,
        "checkpoint_metric": args.checkpoint_metric,
        "best_score": best_score,
        "history": history,
        "metrics": {
            split: {
                key: value
                for key, value in metrics.items()
                if key not in {"graph_embeddings", "logits", "preds"}
            }
            for split, metrics in final_metrics.items()
        },
        "warnings": warnings,
    }
    (out_dir / "gnn_train_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[GCF_HIV_CSV_GNN_DONE]", flush=True)
    print(json.dumps({"out_dir": str(out_dir), "test_metrics": summary["metrics"]["test"], "warnings": warnings}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
