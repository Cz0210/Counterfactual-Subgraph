"""GCF-style GNN utilities for the adapted HIVCSV baseline."""

from __future__ import annotations

from typing import Any


def load_torch_stack() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import DataLoader
        from torch_geometric.nn import GCNConv, global_max_pool

        return torch, F, DataLoader, GCNConv, global_max_pool
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "GCF HIVCSV utilities require torch and torch_geometric at runtime. "
            "Run on HPC in the smiles_pip118 environment."
        ) from exc


def build_gcf_style_gnn(
    num_features: int,
    num_classes: int,
    *,
    num_layers: int = 3,
    dim: int = 20,
    dropout: float = 0.0,
    device: str = "cpu",
) -> Any:
    torch, F, _DataLoader, GCNConv, global_max_pool = load_torch_stack()

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


def torch_load(path: str, *, map_location: str | None = None) -> Any:
    torch, _F, _DataLoader, _GCNConv, _global_max_pool = load_torch_stack()
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

