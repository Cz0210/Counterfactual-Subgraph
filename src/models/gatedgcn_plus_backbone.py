"""GatedGCN+ molecular backbone adapted from pinned GNN+ components.

The message/update equations follow ``LUOyk1999/GNNPlus`` commit
``0e02ad9acc2f1e54b5ad71c051bf5dfb1fcb4f28``.  This module intentionally
adapts only the graph-level GatedGCN layer, residual feed-forward block, and
RWSE encoder to the project's existing categorical molecular schema.  It does
not import GraphGym or execute code from a moving upstream checkout.  The
project-specific depth, width, dropout, RWSE length, and readout settings are
parameter-matched ablation choices, not an upstream BACE recipe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

from src.models.gnn_backbone_registry import get_gnn_backbone_spec
from src.models.molecular_gnn import CategoricalFeatureEncoder

try:  # Keep parameter matching and metadata usable without torch.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - lightweight metadata environments.
    torch = None
    nn = None


_ModuleBase = nn.Module if nn is not None else object

GATEDGCN_PLUS_OFFICIAL_REPOSITORY = "LUOyk1999/GNNPlus"
GATEDGCN_PLUS_OFFICIAL_COMMIT = "0e02ad9acc2f1e54b5ad71c051bf5dfb1fcb4f28"
GATEDGCN_PLUS_LICENSE = "MIT"
GATEDGCN_PLUS_LICENSE_SHA256 = (
    "a09fa408be8fa4a095f18b6f241dae744af059771984a22dbfd6afa5d1765fd7"
)
GATEDGCN_PLUS_ALLOWED_HIDDEN_DIMS = (64, 96, 128, 160, 192, 256)
GATEDGCN_PLUS_RWPE_WALK_LENGTH = 16
GATEDGCN_PLUS_RWPE_DIM = 16
GATEDGCN_PLUS_NUM_LAYERS = 5
GATEDGCN_PLUS_DROPOUT = 0.2
GATEDGCN_PLUS_PARAMETER_MATCH_MAX_RELATIVE_DIFFERENCE = 0.15


def _require_torch() -> tuple[Any, Any]:
    if torch is None or nn is None:
        raise RuntimeError(
            "GatedGCN+ construction requires PyTorch; metadata-only parameter "
            "matching remains available."
        )
    return torch, nn


def _positive_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class GatedGCNPlusConfig:
    backbone: str = "gatedgcn_plus"
    num_classes: int = 2
    num_layers: int = GATEDGCN_PLUS_NUM_LAYERS
    hidden_dim: int = 160
    dropout: float = GATEDGCN_PLUS_DROPOUT
    pooling: str = "mean"
    readout_layers: int = 2
    normalization: str = "batch_norm"
    residual: bool = True
    ffn: bool = True
    rwpe_walk_length: int = GATEDGCN_PLUS_RWPE_WALK_LENGTH
    rwpe_dim: int = GATEDGCN_PLUS_RWPE_DIM
    rwpe_raw_normalization: str = "batch_norm"

    def __post_init__(self) -> None:
        from src.models.gnn_backbone_registry import normalize_gnn_backbone

        canonical = normalize_gnn_backbone(self.backbone)
        object.__setattr__(self, "backbone", canonical)
        object.__setattr__(self, "pooling", str(self.pooling).strip().lower())
        object.__setattr__(
            self,
            "normalization",
            str(self.normalization).strip().lower().replace("-", "_"),
        )
        object.__setattr__(
            self,
            "rwpe_raw_normalization",
            str(self.rwpe_raw_normalization).strip().lower().replace("-", "_"),
        )
        if canonical != "gatedgcn_plus":
            raise ValueError("GatedGCNPlusConfig requires backbone=gatedgcn_plus")
        if self.num_classes < 2 or self.num_layers < 1 or self.hidden_dim < 4:
            raise ValueError("GatedGCN+ class/layer/hidden dimensions are invalid")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("GatedGCN+ dropout must be in [0, 1)")
        if self.pooling not in {"mean", "sum", "max"}:
            raise ValueError(f"Unsupported GatedGCN+ pooling: {self.pooling}")
        if self.normalization != "batch_norm":
            raise ValueError("The pinned GatedGCN+ recipe requires batch_norm")
        if self.rwpe_raw_normalization != "batch_norm":
            raise ValueError("The pinned GatedGCN+ RWPE recipe requires batch_norm")
        if self.residual is not True or self.ffn is not True:
            raise ValueError("GatedGCN+ requires both residual and FFN blocks")
        if self.readout_layers != 2:
            raise ValueError("The matched GatedGCN+ recipe requires two readout layers")
        if self.rwpe_walk_length != GATEDGCN_PLUS_RWPE_WALK_LENGTH:
            raise ValueError("GatedGCN+ requires topology-only RWPE length 16")
        if self.rwpe_dim != GATEDGCN_PLUS_RWPE_DIM:
            raise ValueError("GatedGCN+ requires a 16-dimensional RWPE projection")
        if self.hidden_dim <= self.rwpe_dim:
            raise ValueError("GatedGCN+ hidden_dim must exceed rwpe_dim")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["edge_feature_mode"] = get_gnn_backbone_spec(
            self.backbone
        ).edge_feature_mode
        payload["official_repository"] = GATEDGCN_PLUS_OFFICIAL_REPOSITORY
        payload["official_commit"] = GATEDGCN_PLUS_OFFICIAL_COMMIT
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GatedGCNPlusConfig":
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
            "ffn",
            "rwpe_walk_length",
            "rwpe_dim",
            "rwpe_raw_normalization",
        }
        return cls(**{key: source[key] for key in fields if key in source})


class GatedGCNPlusLayer(_ModuleBase):
    """Edge-gated message passing followed by the pinned residual FFN."""

    def __init__(self, hidden_dim: int, *, dropout: float) -> None:
        torch_mod, torch_nn = _require_torch()
        del torch_mod
        super().__init__()
        hidden = int(hidden_dim)
        self.hidden_dim = hidden
        self.A = torch_nn.Linear(hidden, hidden, bias=True)
        self.B = torch_nn.Linear(hidden, hidden, bias=True)
        self.C = torch_nn.Linear(hidden, hidden, bias=True)
        self.D = torch_nn.Linear(hidden, hidden, bias=True)
        self.E = torch_nn.Linear(hidden, hidden, bias=True)
        self.node_norm = torch_nn.BatchNorm1d(hidden)
        self.edge_norm = torch_nn.BatchNorm1d(hidden)
        self.local_norm = torch_nn.BatchNorm1d(hidden)
        self.ff_linear1 = torch_nn.Linear(hidden, 2 * hidden)
        self.ff_linear2 = torch_nn.Linear(2 * hidden, hidden)
        self.ff_norm = torch_nn.BatchNorm1d(hidden)
        self.dropout = torch_nn.Dropout(float(dropout))
        self.ff_dropout1 = torch_nn.Dropout(float(dropout))
        self.ff_dropout2 = torch_nn.Dropout(float(dropout))

    @staticmethod
    def _sum(values: Any, destinations: Any, *, num_nodes: int) -> Any:
        torch_mod, _torch_nn = _require_torch()
        result = torch_mod.zeros(
            (int(num_nodes), int(values.shape[-1])),
            dtype=values.dtype,
            device=values.device,
        )
        if values.numel():
            result.index_add_(0, destinations, values)
        return result

    def forward(self, x: Any, edge_index: Any, edge_attr: Any) -> tuple[Any, Any]:
        torch_mod, _torch_nn = _require_torch()
        if edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
            raise ValueError("GatedGCN+ edge_index must have shape [2, num_edges]")
        if edge_attr.ndim != 2 or int(edge_attr.shape[0]) != int(
            edge_index.shape[1]
        ):
            raise ValueError("GatedGCN+ edge attributes do not align with edges")
        sources = edge_index[0].long()
        destinations = edge_index[1].long()
        x_input = x
        edge_input = edge_attr

        ax = self.A(x)
        bx = self.B(x)
        # PyG's source_to_target convention maps D(x_i) to the destination and
        # E(x_j) to the source in the pinned upstream implementation.
        edge_logits = (
            self.D(x)[destinations]
            + self.E(x)[sources]
            + self.C(edge_attr)
        )
        gates = torch_mod.sigmoid(edge_logits)
        numerator = self._sum(
            gates * bx[sources], destinations, num_nodes=int(x.shape[0])
        )
        denominator = self._sum(
            gates, destinations, num_nodes=int(x.shape[0])
        )
        updated_x = ax + numerator / (denominator + 1e-6)
        updated_edge = edge_logits

        updated_x = self.node_norm(updated_x)
        if int(updated_edge.shape[0]):
            updated_edge = self.edge_norm(updated_edge)
        updated_x = self.dropout(torch_mod.relu(updated_x))
        updated_edge = self.dropout(torch_mod.relu(updated_edge))
        updated_x = x_input + updated_x
        updated_edge = edge_input + updated_edge

        local = self.local_norm(updated_x)
        feed_forward = self.ff_dropout1(
            torch_mod.relu(self.ff_linear1(local))
        )
        feed_forward = self.ff_dropout2(self.ff_linear2(feed_forward))
        return self.ff_norm(local + feed_forward), updated_edge


class GatedGCNPlusMolecularGNN(_ModuleBase):
    """Graph-level GatedGCN+ over the shared molecular feature schema."""

    def __init__(
        self,
        config: GatedGCNPlusConfig,
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
            self.node_cardinalities, config.hidden_dim - config.rwpe_dim
        )
        self.rwpe_raw_norm = torch_nn.BatchNorm1d(config.rwpe_walk_length)
        self.rwpe_encoder = torch_nn.Linear(
            config.rwpe_walk_length, config.rwpe_dim
        )
        self.edge_encoder = CategoricalFeatureEncoder(
            self.edge_cardinalities, config.hidden_dim
        )
        self.layers = torch_nn.ModuleList(
            GatedGCNPlusLayer(config.hidden_dim, dropout=config.dropout)
            for _ in range(config.num_layers)
        )
        self.classifier = torch_nn.Sequential(
            torch_nn.Linear(config.hidden_dim, config.hidden_dim),
            torch_nn.ReLU(),
            torch_nn.Dropout(config.dropout),
            torch_nn.Linear(config.hidden_dim, config.num_classes),
        )

    def _pool(self, x: Any, batch: Any) -> Any:
        torch_mod, _torch_nn = _require_torch()
        if not batch.numel():
            raise ValueError("Cannot pool an empty GatedGCN+ graph batch")
        graphs = int(batch.max().item()) + 1
        if self.config.pooling in {"sum", "mean"}:
            pooled = torch_mod.zeros(
                (graphs, self.config.hidden_dim), dtype=x.dtype, device=x.device
            )
            pooled.index_add_(0, batch, x)
            if self.config.pooling == "mean":
                counts = torch_mod.zeros((graphs,), dtype=x.dtype, device=x.device)
                counts.index_add_(0, batch, torch_mod.ones_like(batch, dtype=x.dtype))
                pooled = pooled / counts.clamp_min(1.0).unsqueeze(-1)
            return pooled
        pooled = torch_mod.full(
            (graphs, self.config.hidden_dim),
            -torch_mod.inf,
            dtype=x.dtype,
            device=x.device,
        )
        expanded = batch.reshape(-1, 1).expand(-1, self.config.hidden_dim)
        if hasattr(pooled, "scatter_reduce_"):
            pooled.scatter_reduce_(0, expanded, x, reduce="amax", include_self=True)
        else:  # pragma: no cover - old torch compatibility.
            for index in range(int(x.shape[0])):
                graph = int(batch[index].item())
                pooled[graph] = torch_mod.maximum(pooled[graph], x[index])
        return pooled

    def forward(
        self,
        data: Any | None = None,
        *,
        x: Any | None = None,
        edge_index: Any | None = None,
        edge_attr: Any | None = None,
        batch: Any | None = None,
        random_walk_pe: Any | None = None,
    ) -> Any:
        torch_mod, _torch_nn = _require_torch()
        if data is not None:
            x = getattr(data, "x", x)
            edge_index = getattr(data, "edge_index", edge_index)
            edge_attr = getattr(data, "edge_attr", edge_attr)
            batch = getattr(data, "batch", batch)
            random_walk_pe = getattr(data, "random_walk_pe", random_walk_pe)
        if x is None or edge_index is None or edge_attr is None:
            raise ValueError("GatedGCN+ requires x, edge_index, and edge_attr")
        if batch is None:
            batch = torch_mod.zeros(
                (int(x.shape[0]),), dtype=torch_mod.long, device=x.device
            )
        if random_walk_pe is None:
            raise ValueError("GatedGCN+ requires topology-only random_walk_pe")
        if random_walk_pe.ndim != 2 or tuple(random_walk_pe.shape) != (
            int(x.shape[0]),
            self.config.rwpe_walk_length,
        ):
            raise ValueError(
                "GatedGCN+ random_walk_pe shape differs from [num_nodes, 16]"
            )
        atom = self.node_encoder(x.long())
        pe = self.rwpe_encoder(
            self.rwpe_raw_norm(random_walk_pe.to(dtype=torch_mod.float32))
        )
        hidden = torch_mod.cat((atom, pe), dim=1)
        edges = self.edge_encoder(edge_attr.long())
        for layer in self.layers:
            hidden, edges = layer(hidden, edge_index.long(), edges)
        return self.classifier(self._pool(hidden, batch.long()))


@dataclass(frozen=True, slots=True)
class GatedGCNPlusParameterCandidate:
    hidden_dim: int
    parameter_count: int
    absolute_difference: int
    relative_difference: float
    within_tolerance: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GatedGCNPlusParameterMatch:
    reference_parameter_count: int
    selected_hidden_dim: int
    selected_parameter_count: int
    selected_relative_difference: float
    max_relative_difference: float
    allowed_hidden_dims: tuple[int, ...]
    candidates: tuple[GatedGCNPlusParameterCandidate, ...]
    selection_inputs: tuple[str, ...] = (
        "reference_parameter_count",
        "allowed_hidden_dims",
        "model_architecture",
    )
    validation_metrics_loaded: bool = False
    test_metrics_loaded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "gatedgcn_plus_parameter_match_v1",
            **asdict(self),
            "allowed_hidden_dims": list(self.allowed_hidden_dims),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selection_inputs": list(self.selection_inputs),
        }


def estimate_gatedgcn_plus_parameter_count(
    hidden_dim: int,
    *,
    node_embedding_cardinality_sum: int = 173,
    edge_embedding_cardinality_sum: int = 19,
    rwpe_walk_length: int = GATEDGCN_PLUS_RWPE_WALK_LENGTH,
    rwpe_dim: int = GATEDGCN_PLUS_RWPE_DIM,
    num_layers: int = GATEDGCN_PLUS_NUM_LAYERS,
    num_classes: int = 2,
) -> int:
    """Count trainable parameters in the adapted GatedGCN+ architecture."""

    hidden = _positive_int(hidden_dim, field="hidden_dim")
    node_sum = _positive_int(
        node_embedding_cardinality_sum, field="node_embedding_cardinality_sum"
    )
    edge_sum = _positive_int(
        edge_embedding_cardinality_sum, field="edge_embedding_cardinality_sum"
    )
    walk = _positive_int(rwpe_walk_length, field="rwpe_walk_length")
    pe_dim = _positive_int(rwpe_dim, field="rwpe_dim")
    layers = _positive_int(num_layers, field="num_layers")
    classes = _positive_int(num_classes, field="num_classes")
    if hidden <= pe_dim:
        raise ValueError("hidden_dim must exceed rwpe_dim")
    categorical = node_sum * (hidden - pe_dim) + edge_sum * hidden
    rwpe = 2 * walk + walk * pe_dim + pe_dim
    # A/B/C/D/E, node+edge BN, and the two-BN H->2H->H residual FFN.
    layer = 9 * hidden * hidden + 16 * hidden
    readout = hidden * hidden + hidden + hidden * classes + classes
    return categorical + rwpe + layers * layer + readout


def match_gatedgcn_plus_hidden_dim(
    reference_parameter_count: int,
    *,
    allowed_hidden_dims: Sequence[int] = GATEDGCN_PLUS_ALLOWED_HIDDEN_DIMS,
    max_relative_difference: float = (
        GATEDGCN_PLUS_PARAMETER_MATCH_MAX_RELATIVE_DIFFERENCE
    ),
) -> GatedGCNPlusParameterMatch:
    """Choose an allowed width using parameter count only, never metrics."""

    reference = _positive_int(
        reference_parameter_count, field="reference_parameter_count"
    )
    dims = tuple(int(value) for value in allowed_hidden_dims)
    if dims != GATEDGCN_PLUS_ALLOWED_HIDDEN_DIMS:
        raise ValueError(
            "GatedGCN+ hidden dimensions must remain exactly "
            f"{GATEDGCN_PLUS_ALLOWED_HIDDEN_DIMS}"
        )
    threshold = float(max_relative_difference)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("max_relative_difference must be finite in (0, 1]")
    candidates: list[GatedGCNPlusParameterCandidate] = []
    for hidden in dims:
        count = estimate_gatedgcn_plus_parameter_count(hidden)
        difference = abs(count - reference)
        relative = difference / reference
        candidates.append(
            GatedGCNPlusParameterCandidate(
                hidden_dim=hidden,
                parameter_count=count,
                absolute_difference=difference,
                relative_difference=relative,
                within_tolerance=relative <= threshold,
            )
        )
    eligible = [candidate for candidate in candidates if candidate.within_tolerance]
    if not eligible:
        raise ValueError(
            "No allowed GatedGCN+ hidden dimension is within the parameter tolerance"
        )
    selected = min(
        eligible,
        key=lambda candidate: (candidate.relative_difference, candidate.hidden_dim),
    )
    return GatedGCNPlusParameterMatch(
        reference_parameter_count=reference,
        selected_hidden_dim=selected.hidden_dim,
        selected_parameter_count=selected.parameter_count,
        selected_relative_difference=selected.relative_difference,
        max_relative_difference=threshold,
        allowed_hidden_dims=dims,
        candidates=tuple(candidates),
    )


def build_gatedgcn_plus_molecular_gnn(
    *,
    num_classes: int,
    node_feature_schema: Any,
    edge_feature_schema: Any,
    **kwargs: Any,
) -> GatedGCNPlusMolecularGNN:
    from src.models.molecular_gnn import _cardinalities

    config = GatedGCNPlusConfig(
        backbone="gatedgcn_plus", num_classes=int(num_classes), **kwargs
    )
    return GatedGCNPlusMolecularGNN(
        config,
        node_cardinalities=_cardinalities(node_feature_schema, kind="node"),
        edge_cardinalities=_cardinalities(edge_feature_schema, kind="edge"),
    )


def gatedgcn_plus_runtime_capabilities() -> dict[str, Any]:
    """Run a CPU-only constructor/count check without opening any split."""

    result: dict[str, Any] = {
        "torch_available": torch is not None,
        "model_build_pass": False,
        "rwpe_available": torch is not None,
        "parameter_count_matches_receipt": False,
        "selected_hidden_dim": 160,
        "expected_parameter_count": estimate_gatedgcn_plus_parameter_count(160),
        "actual_parameter_count": None,
        "validation_metrics_loaded": False,
        "test_metrics_loaded": False,
    }
    if torch is None:
        return result
    try:
        from src.data.molecular_graph_featurizer import (
            default_molecular_feature_schema,
        )

        schema = default_molecular_feature_schema()
        model = build_gatedgcn_plus_molecular_gnn(
            num_classes=2,
            node_feature_schema=schema,
            edge_feature_schema=schema,
            hidden_dim=160,
        )
        actual = sum(parameter.numel() for parameter in model.parameters())
    except Exception as exc:  # pragma: no cover - environment capability report.
        result["error"] = f"{type(exc).__name__}:{exc}"
        return result
    result.update(
        {
            "model_build_pass": True,
            "actual_parameter_count": actual,
            "parameter_count_matches_receipt": (
                actual == result["expected_parameter_count"] == 1_219_138
            ),
        }
    )
    return result


__all__ = [
    "GATEDGCN_PLUS_ALLOWED_HIDDEN_DIMS",
    "GATEDGCN_PLUS_OFFICIAL_COMMIT",
    "GATEDGCN_PLUS_OFFICIAL_REPOSITORY",
    "GATEDGCN_PLUS_LICENSE_SHA256",
    "GATEDGCN_PLUS_RWPE_WALK_LENGTH",
    "GatedGCNPlusConfig",
    "GatedGCNPlusLayer",
    "GatedGCNPlusMolecularGNN",
    "build_gatedgcn_plus_molecular_gnn",
    "estimate_gatedgcn_plus_parameter_count",
    "gatedgcn_plus_runtime_capabilities",
    "match_gatedgcn_plus_hidden_dim",
]
