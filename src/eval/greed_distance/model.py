"""Small GREED-style Siamese GIN distance model.

This implementation is intentionally project-owned and lightweight. It follows
the GREED idea used by GCFExplainer as a neural normalized-GED approximator:
encode each graph independently, then predict distance with the L2 norm between
graph embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # pragma: no cover - dependency checked in runtime
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def _require_torch() -> Any:
    if torch is None or nn is None:
        raise RuntimeError("GREED distance training/inference requires PyTorch.")
    return torch


if nn is not None:

    class GreedGraphEncoder(nn.Module):
        """Padded-adjacency GIN encoder with sum pooling."""

        def __init__(
            self,
            *,
            num_layers: int = 8,
            hidden_dim: int = 64,
            max_atomic_num: int = 128,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.max_atomic_num = int(max_atomic_num)
            self.atom_embedding = nn.Embedding(self.max_atomic_num + 1, hidden_dim, padding_idx=0)
            self.mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                    )
                    for _ in range(int(num_layers))
                ]
            )
            self.eps = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(int(num_layers))])
            self.out = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, atom_ids: Any, adjacency: Any, mask: Any) -> Any:
            atom_ids = atom_ids.clamp(min=0, max=self.max_atomic_num)
            h = self.atom_embedding(atom_ids) * mask.unsqueeze(-1)
            for layer, eps in zip(self.mlps, self.eps):
                neigh = torch.bmm(adjacency, h)
                h = layer((1.0 + eps) * h + neigh) * mask.unsqueeze(-1)
            pooled = h.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
            return self.out(pooled)


    class GreedGEDModel(nn.Module):
        """Siamese GIN model predicting normalized GED by embedding distance."""

        def __init__(
            self,
            *,
            num_layers: int = 8,
            hidden_dim: int = 64,
            max_atomic_num: int = 128,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.encoder = GreedGraphEncoder(
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                max_atomic_num=max_atomic_num,
                dropout=dropout,
            )

        def embed(self, atom_ids: Any, adjacency: Any, mask: Any) -> Any:
            return self.encoder(atom_ids, adjacency, mask)

        def forward(self, batch: dict[str, Any]) -> Any:
            za = self.embed(batch["atom_a"], batch["adj_a"], batch["mask_a"])
            zb = self.embed(batch["atom_b"], batch["adj_b"], batch["mask_b"])
            return torch.linalg.vector_norm(za - zb, ord=2, dim=-1)

else:  # pragma: no cover

    class GreedGEDModel:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("GREED distance model requires PyTorch.")


def graph_record_to_tensors(graph: dict[str, Any]) -> tuple[list[int], list[tuple[int, int]]]:
    atom_ids = [int(node.get("atomic_num") or 0) for node in graph.get("nodes", [])]
    edges = [
        (int(edge.get("source")), int(edge.get("target")))
        for edge in graph.get("edges", [])
        if edge.get("source") is not None and edge.get("target") is not None
    ]
    return atom_ids, edges


def make_padded_graph_batch(graphs: list[dict[str, Any]], *, device: str | Any) -> tuple[Any, Any, Any]:
    torch_mod = _require_torch()
    max_nodes = max(1, max((len(graph.get("nodes", [])) for graph in graphs), default=1))
    atom = torch_mod.zeros((len(graphs), max_nodes), dtype=torch_mod.long, device=device)
    adj = torch_mod.zeros((len(graphs), max_nodes, max_nodes), dtype=torch_mod.float32, device=device)
    mask = torch_mod.zeros((len(graphs), max_nodes), dtype=torch_mod.float32, device=device)
    for row_index, graph in enumerate(graphs):
        atom_ids, edges = graph_record_to_tensors(graph)
        n = min(len(atom_ids), max_nodes)
        if n:
            atom[row_index, :n] = torch_mod.tensor(atom_ids[:n], dtype=torch_mod.long, device=device)
            mask[row_index, :n] = 1.0
        for src, dst in edges:
            if 0 <= src < n and 0 <= dst < n and src != dst:
                adj[row_index, src, dst] = 1.0
                adj[row_index, dst, src] = 1.0
    return atom, adj, mask


def make_pair_batch(
    graph_a: list[dict[str, Any]],
    graph_b: list[dict[str, Any]],
    *,
    device: str | Any,
) -> dict[str, Any]:
    atom_a, adj_a, mask_a = make_padded_graph_batch(graph_a, device=device)
    atom_b, adj_b, mask_b = make_padded_graph_batch(graph_b, device=device)
    return {
        "atom_a": atom_a,
        "adj_a": adj_a,
        "mask_a": mask_a,
        "atom_b": atom_b,
        "adj_b": adj_b,
        "mask_b": mask_b,
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: Any,
    model_config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> None:
    torch_mod = _require_torch()
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch_mod.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": dict(model_config),
            "metrics": dict(metrics or {}),
        },
        destination,
    )


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    torch_mod = _require_torch()
    source = Path(path).expanduser().resolve()
    payload = torch_mod.load(source, map_location=device)
    config = dict(payload.get("model_config") or {})
    model = GreedGEDModel(**config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload
