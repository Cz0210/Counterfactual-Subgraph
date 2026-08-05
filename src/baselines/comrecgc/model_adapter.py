"""Model and embedding adapters injected into unmodified COMRECGC."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .contracts import ContractError, sha256_file


def _runtime_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn.functional as functional
        from torch_geometric.data import Batch
    except Exception as exc:  # pragma: no cover - HPC runtime dependency
        raise RuntimeError("COMRECGC model adapters require torch and torch_geometric.") from exc
    return torch, functional, Batch


class InternalLabelGNN:
    """Expose project GNN outputs as COMRECGC source=0,target=1 log-probs."""

    def __init__(self, model: Any, *, swap_project_labels: bool) -> None:
        self.model = model
        self.swap_project_labels = bool(swap_project_labels)

    def eval(self) -> "InternalLabelGNN":
        self.model.eval()
        return self

    def to(self, device: str | Any) -> "InternalLabelGNN":
        self.model.to(device)
        return self

    def __call__(self, data: Any, edge_weight: Any = None) -> tuple[Any, Any, Any]:
        del edge_weight
        torch, functional, _Batch = _runtime_stack()
        node_embeddings, graph_embeddings, outputs = self.model(data)
        log_probs = functional.log_softmax(outputs, dim=-1)
        if self.swap_project_labels:
            log_probs = log_probs[:, torch.tensor([1, 0], device=log_probs.device)]
        return node_embeddings, graph_embeddings, log_probs


def load_aids_gnn(checkpoint: str | Path, *, num_features: int, device: str) -> tuple[Any, dict[str, Any]]:
    from src.baselines.gcf_hiv_csv_model import build_gcf_style_gnn, torch_load

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"AIDS project GNN checkpoint missing: {path}")
    model = build_gcf_style_gnn(num_features, 2, num_layers=3, dim=20, dropout=0.0, device=device)
    model.load_state_dict(torch_load(str(path), map_location=device))
    wrapped = InternalLabelGNN(model, swap_project_labels=True).eval()
    return wrapped, {
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256_file(path),
        "label_mapping": "project[1,0]_to_comrecgc[0,1]",
        "model_retrained": False,
    }


def load_mutagenicity_gnn(
    checkpoint: str | Path,
    *,
    num_features: int,
    official_gnn_class: Any,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    torch, _functional, _Batch = _runtime_stack()
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Mutagenicity project GNN checkpoint missing: {path}")
    model = official_gnn_class(
        num_features=num_features,
        num_classes=2,
        num_layers=3,
        dim=20,
        dropout=0.0,
    ).to(device)
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state)
    wrapped = InternalLabelGNN(model, swap_project_labels=False).eval()
    return wrapped, {
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256_file(path),
        "label_mapping": "prepared_gnn_label_source0_target1",
        "model_retrained": False,
    }


def _one_hot_atomic_numbers(graphs: Sequence[Any], atom_vocabulary: Sequence[str]) -> list[dict[str, Any]]:
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RDKit is required for the AIDS GREED adapter.") from exc
    atomic_numbers = [int(Chem.GetPeriodicTable().GetAtomicNumber(symbol)) for symbol in atom_vocabulary]
    converted: list[dict[str, Any]] = []
    for graph in graphs:
        x = graph.x.detach().cpu()
        if x.ndim != 2 or int(x.shape[1]) != len(atomic_numbers):
            raise ContractError("AIDS graph feature dimension differs from its frozen atom vocabulary.")
        indices = x.argmax(dim=1).tolist()
        nodes = [{"atomic_num": atomic_numbers[int(index)]} for index in indices]
        pairs: set[tuple[int, int]] = set()
        for source, target in graph.edge_index.detach().cpu().t().tolist():
            a, b = sorted((int(source), int(target)))
            if a != b:
                pairs.add((a, b))
        converted.append(
            {
                "nodes": nodes,
                "edges": [{"source": a, "target": b} for a, b in sorted(pairs)],
            }
        )
    return converted


class AIDSGreedEmbeddingAdapter:
    """Expose the frozen project GREED encoder through COMRECGC's API."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        atom_vocabulary: Sequence[str],
        device: str,
    ) -> None:
        from src.eval.greed_distance.model import load_checkpoint

        self.path = Path(checkpoint).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Project AIDS GREED checkpoint missing: {self.path}")
        self.model, self.payload = load_checkpoint(self.path, device=device)
        self.atom_vocabulary = tuple(str(value) for value in atom_vocabulary)
        self.device = device
        self._targets: Any = None

    def eval(self) -> "AIDSGreedEmbeddingAdapter":
        self.model.eval()
        return self

    def to(self, device: str | Any) -> "AIDSGreedEmbeddingAdapter":
        self.device = str(device)
        self.model.to(device)
        return self

    def embed_model(self, graphs: Any) -> Any:
        from src.eval.greed_distance.model import make_padded_graph_batch

        torch, _functional, _Batch = _runtime_stack()
        data_list = graphs.to_data_list() if hasattr(graphs, "to_data_list") else list(graphs)
        records = _one_hot_atomic_numbers(data_list, self.atom_vocabulary)
        atoms, adjacency, mask = make_padded_graph_batch(records, device=self.device)
        with torch.no_grad():
            return self.model.embed(atoms, adjacency, mask)

    def embed_targets(self, graphs: Any) -> None:
        self._targets = self.embed_model(graphs)

    def predict_outer_with_queries(self, graphs: Sequence[Any], batch_size: int = 128) -> Any:
        del batch_size
        torch, _functional, Batch = _runtime_stack()
        if self._targets is None:
            raise RuntimeError("embed_targets must be called before predict_outer_with_queries.")
        queries = self.embed_model(Batch.from_data_list(list(graphs)))
        return torch.cdist(queries, self._targets, p=2)

    def provenance(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.path),
            "checkpoint_sha256": sha256_file(self.path),
            "distance_model": "project_greed_hiv_ged",
            "checkpoint_retrained": False,
        }
