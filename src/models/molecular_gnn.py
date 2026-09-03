"""Task-specific molecular graph classifiers with interchangeable backbones."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from src.data.molecular_graph_featurizer import MolecularFeatureSchema
from src.models.gnn_backbone_registry import (
    get_gnn_backbone_spec,
    normalize_gnn_backbone,
)

try:  # Keep metadata/config imports usable on lightweight local machines.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only without PyTorch.
    torch = None
    nn = None


_ModuleBase = nn.Module if nn is not None else object


def _require_torch() -> tuple[Any, Any]:
    if torch is None or nn is None:
        raise RuntimeError(
            "Molecular GNN construction requires PyTorch. Activate the AutoDL "
            "classifier environment before building a model."
        )
    return torch, nn


@dataclass(frozen=True, slots=True)
class MolecularGNNConfig:
    backbone: str = "gine"
    num_classes: int = 2
    num_layers: int = 5
    hidden_dim: int = 256
    dropout: float = 0.2
    pooling: str = "mean"
    readout_layers: int = 2
    normalization: str = "batch_norm"
    residual: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "backbone", normalize_gnn_backbone(self.backbone))
        object.__setattr__(self, "pooling", str(self.pooling).strip().lower())
        object.__setattr__(
            self,
            "normalization",
            str(self.normalization).strip().lower().replace("-", "_"),
        )
        if int(self.num_classes) < 2:
            raise ValueError("Molecular GNN num_classes must be at least two.")
        if int(self.num_layers) < 1:
            raise ValueError("Molecular GNN num_layers must be positive.")
        if int(self.hidden_dim) < 4:
            raise ValueError("Molecular GNN hidden_dim must be at least four.")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("Molecular GNN dropout must be in [0, 1).")
        if self.pooling not in {"mean", "sum", "max"}:
            raise ValueError(f"Unsupported molecular graph pooling: {self.pooling}")
        if int(self.readout_layers) < 1:
            raise ValueError("Molecular GNN readout_layers must be positive.")
        if self.normalization not in {"batch_norm", "layer_norm", "none"}:
            raise ValueError(
                f"Unsupported molecular GNN normalization: {self.normalization}"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["edge_feature_mode"] = get_gnn_backbone_spec(
            self.backbone
        ).edge_feature_mode
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MolecularGNNConfig":
        source = payload.get("gnn", payload)
        fields = {
            "backbone",
            "num_classes",
            "num_layers",
            "hidden_dim",
            "dropout",
            "pooling",
            "readout_layers",
            "normalization",
            "residual",
        }
        return cls(**{key: source[key] for key in fields if key in source})


class CategoricalFeatureEncoder(_ModuleBase):
    """Sum one embedding table per categorical feature field."""

    def __init__(self, cardinalities: Sequence[int], hidden_dim: int) -> None:
        _torch, torch_nn = _require_torch()
        super().__init__()
        values = tuple(int(value) for value in cardinalities)
        if not values or any(value <= 1 for value in values):
            raise ValueError(f"Invalid categorical feature cardinalities: {values}")
        self.cardinalities = values
        self.hidden_dim = int(hidden_dim)
        self.embeddings = torch_nn.ModuleList(
            [torch_nn.Embedding(value, self.hidden_dim) for value in values]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch_mod, _torch_nn = _require_torch()
        for embedding in self.embeddings:
            torch_mod.nn.init.xavier_uniform_(embedding.weight)

    def forward(self, features: Any) -> Any:
        if features.ndim != 2 or int(features.shape[1]) != len(self.embeddings):
            raise ValueError(
                "Categorical graph feature tensor has the wrong number of columns: "
                f"expected={len(self.embeddings)}, observed={tuple(features.shape)}"
            )
        encoded = None
        for column, (embedding, cardinality) in enumerate(
            zip(self.embeddings, self.cardinalities, strict=True)
        ):
            indices = features[:, column].long()
            if indices.numel() and (
                int(indices.min().item()) < 0 or int(indices.max().item()) >= cardinality
            ):
                raise ValueError(
                    f"Categorical feature column {column} exceeds cardinality {cardinality}."
                )
            value = embedding(indices)
            encoded = value if encoded is None else encoded + value
        return encoded


def _mlp(hidden_dim: int) -> Any:
    _torch, torch_nn = _require_torch()
    return torch_nn.Sequential(
        torch_nn.Linear(hidden_dim, hidden_dim * 2),
        torch_nn.ReLU(),
        torch_nn.Linear(hidden_dim * 2, hidden_dim),
    )


class MolecularMessageLayer(_ModuleBase):
    """One pure-PyTorch message layer for a registered backbone."""

    def __init__(self, backbone: str, hidden_dim: int) -> None:
        _torch, torch_nn = _require_torch()
        super().__init__()
        self.backbone = normalize_gnn_backbone(backbone)
        self.hidden_dim = int(hidden_dim)
        if self.backbone in {"gine", "gin"}:
            self.eps = torch_nn.Parameter(_torch.zeros(1))
            self.update_mlp = _mlp(hidden_dim)
        elif self.backbone == "gcn":
            self.gcn_linear = torch_nn.Linear(hidden_dim, hidden_dim)
        elif self.backbone == "gatv2":
            self.attn_source = torch_nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.attn_target = torch_nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.attn_edge = torch_nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.attn_score = torch_nn.Linear(hidden_dim, 1, bias=False)
            self.attn_out = torch_nn.Linear(hidden_dim, hidden_dim)
        else:  # pragma: no cover - normalized registry already rejects this.
            raise AssertionError(self.backbone)

    @staticmethod
    def _aggregate_sum(messages: Any, destinations: Any, num_nodes: int) -> Any:
        torch_mod, _torch_nn = _require_torch()
        output = torch_mod.zeros(
            (int(num_nodes), int(messages.shape[-1])),
            dtype=messages.dtype,
            device=messages.device,
        )
        if messages.numel():
            output.index_add_(0, destinations, messages)
        return output

    @staticmethod
    def _segment_softmax(scores: Any, destinations: Any, num_nodes: int) -> Any:
        torch_mod, _torch_nn = _require_torch()
        if scores.numel() == 0:
            return scores
        maxima = torch_mod.full(
            (int(num_nodes),),
            -torch_mod.inf,
            dtype=scores.dtype,
            device=scores.device,
        )
        if hasattr(maxima, "scatter_reduce_"):
            maxima.scatter_reduce_(
                0, destinations, scores, reduce="amax", include_self=True
            )
        else:  # pragma: no cover - old torch compatibility fallback.
            for edge_index in range(int(scores.shape[0])):
                target = int(destinations[edge_index].item())
                maxima[target] = torch_mod.maximum(maxima[target], scores[edge_index])
        exponentials = torch_mod.exp(scores - maxima[destinations])
        denominators = torch_mod.zeros(
            (int(num_nodes),), dtype=scores.dtype, device=scores.device
        )
        denominators.index_add_(0, destinations, exponentials)
        return exponentials / denominators[destinations].clamp_min(1e-12)

    def forward(self, x: Any, edge_index: Any, edge_embedding: Any) -> Any:
        torch_mod, _torch_nn = _require_torch()
        if edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
            raise ValueError("Molecular edge_index must have shape [2, num_edges].")
        if edge_embedding.ndim != 2 or int(edge_embedding.shape[0]) != int(
            edge_index.shape[1]
        ):
            raise ValueError("Molecular edge embeddings do not align with edge_index.")
        sources = edge_index[0].long()
        destinations = edge_index[1].long()
        num_nodes = int(x.shape[0])

        if self.backbone in {"gine", "gin"}:
            messages = x[sources] + edge_embedding
            if self.backbone == "gine":
                messages = torch_mod.relu(messages)
            aggregated = self._aggregate_sum(messages, destinations, num_nodes)
            return self.update_mlp((1.0 + self.eps) * x + aggregated)

        if self.backbone == "gcn":
            degrees = torch_mod.ones(
                (num_nodes,), dtype=x.dtype, device=x.device
            )
            if destinations.numel():
                degrees.index_add_(
                    0,
                    destinations,
                    torch_mod.ones_like(destinations, dtype=x.dtype),
                )
            if sources.numel():
                norm = (degrees[sources] * degrees[destinations]).rsqrt()
                messages = (x[sources] + edge_embedding) * norm.unsqueeze(-1)
            else:
                messages = x.new_empty((0, self.hidden_dim))
            aggregated = self._aggregate_sum(messages, destinations, num_nodes)
            aggregated = aggregated + x / degrees.unsqueeze(-1)
            return self.gcn_linear(aggregated)

        source_hidden = self.attn_source(x[sources])
        target_hidden = self.attn_target(x[destinations])
        edge_hidden = self.attn_edge(edge_embedding)
        scores = self.attn_score(
            torch_mod.nn.functional.leaky_relu(
                source_hidden + target_hidden + edge_hidden, negative_slope=0.2
            )
        ).reshape(-1)
        attention = self._segment_softmax(scores, destinations, num_nodes)
        messages = (source_hidden + edge_hidden) * attention.unsqueeze(-1)
        aggregated = self._aggregate_sum(messages, destinations, num_nodes)
        # Isolated atoms retain a direct self representation.
        return self.attn_out(aggregated + self.attn_target(x))


def _normalization(name: str, hidden_dim: int) -> Any:
    _torch, torch_nn = _require_torch()
    if name == "batch_norm":
        return torch_nn.BatchNorm1d(hidden_dim)
    if name == "layer_norm":
        return torch_nn.LayerNorm(hidden_dim)
    return torch_nn.Identity()


class MolecularGNN(_ModuleBase):
    """Shared classifier head over GINE/GIN/GCN/GATv2 message backbones."""

    def __init__(
        self,
        config: MolecularGNNConfig,
        *,
        node_cardinalities: Sequence[int],
        edge_cardinalities: Sequence[int],
    ) -> None:
        _torch, torch_nn = _require_torch()
        super().__init__()
        self.config = config
        self.node_cardinalities = tuple(int(value) for value in node_cardinalities)
        self.edge_cardinalities = tuple(int(value) for value in edge_cardinalities)
        self.node_encoder = CategoricalFeatureEncoder(
            self.node_cardinalities, config.hidden_dim
        )
        self.edge_encoder = CategoricalFeatureEncoder(
            self.edge_cardinalities, config.hidden_dim
        )
        self.layers = torch_nn.ModuleList(
            [
                MolecularMessageLayer(config.backbone, config.hidden_dim)
                for _ in range(config.num_layers)
            ]
        )
        self.normalizations = torch_nn.ModuleList(
            [
                _normalization(config.normalization, config.hidden_dim)
                for _ in range(config.num_layers)
            ]
        )
        self.dropout = torch_nn.Dropout(config.dropout)
        readout: list[Any] = []
        for _index in range(config.readout_layers - 1):
            readout.extend(
                (
                    torch_nn.Linear(config.hidden_dim, config.hidden_dim),
                    torch_nn.ReLU(),
                    torch_nn.Dropout(config.dropout),
                )
            )
        readout.append(torch_nn.Linear(config.hidden_dim, config.num_classes))
        self.classifier = torch_nn.Sequential(*readout)

    def _unpack(
        self,
        data: Any | None,
        *,
        x: Any | None,
        edge_index: Any | None,
        edge_attr: Any | None,
        batch: Any | None,
    ) -> tuple[Any, Any, Any, Any]:
        torch_mod, _torch_nn = _require_torch()
        if data is not None:
            x = getattr(data, "x", x)
            edge_index = getattr(data, "edge_index", edge_index)
            edge_attr = getattr(data, "edge_attr", edge_attr)
            batch = getattr(data, "batch", batch)
        if x is None or edge_index is None or edge_attr is None:
            raise ValueError("MolecularGNN requires x, edge_index, and edge_attr.")
        if batch is None:
            batch = torch_mod.zeros((int(x.shape[0]),), dtype=torch_mod.long, device=x.device)
        return x.long(), edge_index.long(), edge_attr.long(), batch.long()

    def encode_graph(
        self,
        data: Any | None = None,
        *,
        x: Any | None = None,
        edge_index: Any | None = None,
        edge_attr: Any | None = None,
        batch: Any | None = None,
    ) -> Any:
        torch_mod, _torch_nn = _require_torch()
        x, edge_index, edge_attr, batch = self._unpack(
            data, x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch
        )
        node_hidden = self.node_encoder(x)
        edge_hidden = self.edge_encoder(edge_attr)
        for layer, normalization in zip(
            self.layers, self.normalizations, strict=True
        ):
            updated = layer(node_hidden, edge_index, edge_hidden)
            if self.config.residual and updated.shape == node_hidden.shape:
                updated = updated + node_hidden
            node_hidden = self.dropout(
                torch_mod.relu(normalization(updated))
            )
        return self._pool(node_hidden, batch)

    def _pool(self, node_hidden: Any, batch: Any) -> Any:
        torch_mod, _torch_nn = _require_torch()
        if batch.numel() == 0:
            raise ValueError("Cannot pool an empty molecular graph batch.")
        num_graphs = int(batch.max().item()) + 1
        if self.config.pooling in {"sum", "mean"}:
            pooled = torch_mod.zeros(
                (num_graphs, self.config.hidden_dim),
                dtype=node_hidden.dtype,
                device=node_hidden.device,
            )
            pooled.index_add_(0, batch, node_hidden)
            if self.config.pooling == "mean":
                counts = torch_mod.zeros(
                    (num_graphs,), dtype=node_hidden.dtype, device=node_hidden.device
                )
                counts.index_add_(
                    0, batch, torch_mod.ones_like(batch, dtype=node_hidden.dtype)
                )
                pooled = pooled / counts.clamp_min(1.0).unsqueeze(-1)
            return pooled
        pooled = torch_mod.full(
            (num_graphs, self.config.hidden_dim),
            -torch_mod.inf,
            dtype=node_hidden.dtype,
            device=node_hidden.device,
        )
        expanded = batch.reshape(-1, 1).expand(-1, self.config.hidden_dim)
        if hasattr(pooled, "scatter_reduce_"):
            pooled.scatter_reduce_(
                0, expanded, node_hidden, reduce="amax", include_self=True
            )
        else:  # pragma: no cover - old torch compatibility fallback.
            for node_index in range(int(node_hidden.shape[0])):
                graph_index = int(batch[node_index].item())
                pooled[graph_index] = torch_mod.maximum(
                    pooled[graph_index], node_hidden[node_index]
                )
        return pooled

    def forward(
        self,
        data: Any | None = None,
        *,
        x: Any | None = None,
        edge_index: Any | None = None,
        edge_attr: Any | None = None,
        batch: Any | None = None,
    ) -> Any:
        graph_hidden = self.encode_graph(
            data,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
        )
        return self.classifier(graph_hidden)


def _cardinalities(
    schema: MolecularFeatureSchema | Mapping[str, Any] | Sequence[int],
    *,
    kind: str,
) -> tuple[int, ...]:
    if isinstance(schema, MolecularFeatureSchema):
        return (
            schema.node_cardinalities if kind == "node" else schema.edge_cardinalities
        )
    if isinstance(schema, Mapping):
        if "node_feature_schema" in schema:
            feature = schema[f"{kind}_feature_schema"]
            return tuple(int(field["cardinality"]) for field in feature["fields"])
        if "fields" in schema:
            return tuple(int(field["cardinality"]) for field in schema["fields"])
        key = f"{kind}_cardinalities"
        if key in schema:
            return tuple(int(value) for value in schema[key])
    return tuple(int(value) for value in schema)


def build_molecular_gnn(
    *,
    backbone: str = "gine",
    num_classes: int,
    node_feature_schema: MolecularFeatureSchema | Mapping[str, Any] | Sequence[int],
    edge_feature_schema: MolecularFeatureSchema | Mapping[str, Any] | Sequence[int],
    num_layers: int = 5,
    hidden_dim: int = 256,
    dropout: float = 0.2,
    pooling: str = "mean",
    readout_layers: int = 2,
    normalization: str = "batch_norm",
    residual: bool = True,
    rwpe_walk_length: int = 16,
    attention_heads: int = 4,
    local_mpnn: str = "gine",
    global_attention: str = "multihead",
    backend: str = "auto",
    ffn: bool = True,
    rwpe_dim: int = 16,
    rwpe_raw_normalization: str = "batch_norm",
) -> Any:
    """Build a classifier through the only public backbone-selection API."""

    canonical = normalize_gnn_backbone(backbone)
    if canonical == "gps":
        from src.models.graphgps_backbone import build_graphgps_molecular_gnn

        return build_graphgps_molecular_gnn(
            num_classes=int(num_classes),
            node_feature_schema=node_feature_schema,
            edge_feature_schema=edge_feature_schema,
            num_layers=int(num_layers),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            pooling=pooling,
            readout_layers=int(readout_layers),
            normalization=normalization,
            residual=bool(residual),
            rwpe_walk_length=int(rwpe_walk_length),
            attention_heads=int(attention_heads),
            local_mpnn=local_mpnn,
            global_attention=global_attention,
            backend=backend,
        )

    if canonical == "gatedgcn_plus":
        from src.models.gatedgcn_plus_backbone import (
            build_gatedgcn_plus_molecular_gnn,
        )

        return build_gatedgcn_plus_molecular_gnn(
            num_classes=int(num_classes),
            node_feature_schema=node_feature_schema,
            edge_feature_schema=edge_feature_schema,
            num_layers=int(num_layers),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            pooling=pooling,
            readout_layers=int(readout_layers),
            normalization=normalization,
            residual=bool(residual),
            ffn=bool(ffn),
            rwpe_walk_length=int(rwpe_walk_length),
            rwpe_dim=int(rwpe_dim),
            rwpe_raw_normalization=rwpe_raw_normalization,
        )

    config = MolecularGNNConfig(
        backbone=canonical,
        num_classes=int(num_classes),
        num_layers=int(num_layers),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
        pooling=pooling,
        readout_layers=int(readout_layers),
        normalization=normalization,
        residual=bool(residual),
    )
    return MolecularGNN(
        config,
        node_cardinalities=_cardinalities(node_feature_schema, kind="node"),
        edge_cardinalities=_cardinalities(edge_feature_schema, kind="edge"),
    )


__all__ = [
    "CategoricalFeatureEncoder",
    "MolecularGNN",
    "MolecularGNNConfig",
    "MolecularMessageLayer",
    "build_molecular_gnn",
]
