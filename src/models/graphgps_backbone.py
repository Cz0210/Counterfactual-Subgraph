"""GraphGPS molecular backbone and parameter-matching utilities.

The parameter matcher is deliberately independent of classifier metrics.  It
compares an audited GINE parameter count with the five pre-registered hidden
dimensions and fails closed when no GraphGPS candidate is within the frozen
relative-difference threshold.

Random-walk positional encodings are topology-only preprocessing artifacts.
The model never accepts labels while preparing them and requires their shape
to match the frozen walk length before a forward pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

from src.data.molecular_graph_featurizer import MolecularFeatureSchema
from src.models.gnn_backbone_registry import get_gnn_backbone_spec

try:  # Metadata and parameter matching stay usable without torch locally.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised on lightweight laptops.
    torch = None
    nn = None


GRAPHGPS_ALLOWED_HIDDEN_DIMS = (96, 128, 160, 192, 256)
GRAPHGPS_RWPE_WALK_LENGTH = 16
GRAPHGPS_NUM_LAYERS = 5
GRAPHGPS_ATTENTION_HEADS = 4
GRAPHGPS_DROPOUT = 0.2
GRAPHGPS_LOCAL_MPNN = "gine"
GRAPHGPS_GLOBAL_ATTENTION = "multihead"
GRAPHGPS_POOLING = "mean"
GRAPHGPS_PARAMETER_MATCH_MAX_RELATIVE_DIFFERENCE = 0.15
GRAPHGPS_PARAMETER_FORMULA_SCHEMA = "graphgps_parameter_formula_v1"


def _require_torch() -> tuple[Any, Any]:
    if torch is None or nn is None:
        raise RuntimeError(
            "GraphGPS construction requires PyTorch; activate the classifier "
            "environment. Metadata-only parameter matching remains available."
        )
    return torch, nn


def _positive_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class GraphGPSParameterCandidate:
    hidden_dim: int
    parameter_count: int
    absolute_difference: int
    relative_difference: float
    within_tolerance: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GraphGPSParameterMatch:
    reference_parameter_count: int
    selected_hidden_dim: int
    selected_parameter_count: int
    selected_relative_difference: float
    max_relative_difference: float
    allowed_hidden_dims: tuple[int, ...]
    candidates: tuple[GraphGPSParameterCandidate, ...]
    selection_inputs: tuple[str, ...] = (
        "reference_parameter_count",
        "allowed_hidden_dims",
        "model_architecture",
    )
    validation_metrics_loaded: bool = False
    test_metrics_loaded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "graphgps_parameter_match_v1",
            "reference_parameter_count": self.reference_parameter_count,
            "selected_hidden_dim": self.selected_hidden_dim,
            "selected_parameter_count": self.selected_parameter_count,
            "selected_relative_difference": self.selected_relative_difference,
            "max_relative_difference": self.max_relative_difference,
            "allowed_hidden_dims": list(self.allowed_hidden_dims),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selection_inputs": list(self.selection_inputs),
            "validation_metrics_loaded": self.validation_metrics_loaded,
            "test_metrics_loaded": self.test_metrics_loaded,
        }


def estimate_graphgps_parameter_count(
    hidden_dim: int,
    *,
    node_embedding_cardinality_sum: int = 173,
    edge_embedding_cardinality_sum: int = 19,
    rwpe_walk_length: int = GRAPHGPS_RWPE_WALK_LENGTH,
    num_layers: int = GRAPHGPS_NUM_LAYERS,
    num_classes: int = 2,
) -> int:
    """Return the audited PyG-GPSConv parameter count for one configuration.

    Each local GINE receives an already projected ``H``-dimensional bond
    embedding, hence its ``edge_dim`` is ``None``.  Adding another ``H -> H``
    edge projection would double-project bonds and invalidate the frozen
    parameter comparison.
    """

    hidden = _positive_int(hidden_dim, field="hidden_dim")
    node_sum = _positive_int(
        node_embedding_cardinality_sum,
        field="node_embedding_cardinality_sum",
    )
    edge_sum = _positive_int(
        edge_embedding_cardinality_sum,
        field="edge_embedding_cardinality_sum",
    )
    walk_length = _positive_int(rwpe_walk_length, field="rwpe_walk_length")
    layers = _positive_int(num_layers, field="num_layers")
    classes = _positive_int(num_classes, field="num_classes")

    categorical_embeddings = (node_sum + edge_sum) * hidden
    rwpe_projection = walk_length * hidden + hidden
    # PyG GPSConv with local GINE(MLP H->2H->H, train_eps), MHA, FFN,
    # and three affine BatchNorms.
    gps_layer = 12 * hidden * hidden + 16 * hidden + 1
    # Two-layer mean-pooled classifier H->H->C.
    readout = hidden * hidden + hidden + hidden * classes + classes
    return categorical_embeddings + rwpe_projection + layers * gps_layer + readout


def match_graphgps_hidden_dim(
    reference_parameter_count: int,
    *,
    allowed_hidden_dims: Sequence[int] = GRAPHGPS_ALLOWED_HIDDEN_DIMS,
    max_relative_difference: float = GRAPHGPS_PARAMETER_MATCH_MAX_RELATIVE_DIFFERENCE,
) -> GraphGPSParameterMatch:
    """Select the closest allowed GraphGPS size using parameter count only."""

    reference = _positive_int(
        reference_parameter_count, field="reference_parameter_count"
    )
    dims = tuple(int(value) for value in allowed_hidden_dims)
    if dims != GRAPHGPS_ALLOWED_HIDDEN_DIMS:
        raise ValueError(
            "GraphGPS hidden dimensions must remain exactly "
            f"{GRAPHGPS_ALLOWED_HIDDEN_DIMS}"
        )
    if (
        type(max_relative_difference) not in (int, float)
        or isinstance(max_relative_difference, bool)
        or not math.isfinite(float(max_relative_difference))
        or not 0.0 < float(max_relative_difference) <= 1.0
    ):
        raise ValueError("max_relative_difference must be finite in (0, 1]")
    threshold = float(max_relative_difference)
    candidates: list[GraphGPSParameterCandidate] = []
    for hidden_dim in dims:
        count = estimate_graphgps_parameter_count(hidden_dim)
        absolute = abs(count - reference)
        relative = absolute / reference
        candidates.append(
            GraphGPSParameterCandidate(
                hidden_dim=hidden_dim,
                parameter_count=count,
                absolute_difference=absolute,
                relative_difference=relative,
                within_tolerance=relative <= threshold,
            )
        )
    eligible = [candidate for candidate in candidates if candidate.within_tolerance]
    if not eligible:
        raise ValueError(
            "No allowed GraphGPS hidden dimension is within the parameter-count "
            f"tolerance {threshold:.6f}"
        )
    selected = min(
        eligible,
        key=lambda candidate: (candidate.relative_difference, candidate.hidden_dim),
    )
    return GraphGPSParameterMatch(
        reference_parameter_count=reference,
        selected_hidden_dim=selected.hidden_dim,
        selected_parameter_count=selected.parameter_count,
        selected_relative_difference=selected.relative_difference,
        max_relative_difference=threshold,
        allowed_hidden_dims=dims,
        candidates=tuple(candidates),
    )


def graphgps_runtime_capabilities() -> dict[str, Any]:
    """Inspect optional PyG primitives without importing them at module load."""

    result: dict[str, Any] = {
        "torch_available": torch is not None,
        "torch_geometric_available": False,
        "torch_geometric_version": None,
        "gpsconv_available": False,
        "add_random_walk_pe_available": False,
    }
    try:
        import torch_geometric
        from torch_geometric.nn import GPSConv  # noqa: F401
        from torch_geometric.transforms import AddRandomWalkPE  # noqa: F401
    except ImportError:
        return result
    result.update(
        {
            "torch_geometric_available": True,
            "torch_geometric_version": str(torch_geometric.__version__),
            "gpsconv_available": True,
            "add_random_walk_pe_available": True,
        }
    )
    return result


def compute_topology_only_random_walk_pe(
    edge_index: Any,
    *,
    num_nodes: int,
    walk_length: int = GRAPHGPS_RWPE_WALK_LENGTH,
) -> Any:
    """Compute diagonal random-walk landing probabilities from topology only.

    This fallback is intended for preprocessing small molecular graphs when the
    PyG transform is unavailable.  It has no label argument by design.
    """

    torch_mod, _torch_nn = _require_torch()
    nodes = _positive_int(num_nodes, field="num_nodes")
    length = _positive_int(walk_length, field="walk_length")
    if edge_index.ndim != 2 or tuple(edge_index.shape)[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    adjacency = torch_mod.zeros(
        (nodes, nodes), dtype=torch_mod.float64, device=edge_index.device
    )
    if int(edge_index.shape[1]):
        sources = edge_index[0].long()
        destinations = edge_index[1].long()
        adjacency.index_put_(
            (sources, destinations),
            torch_mod.ones_like(sources, dtype=adjacency.dtype),
            accumulate=True,
        )
    degrees = adjacency.sum(dim=1)
    transition = adjacency / degrees.clamp_min(1.0).unsqueeze(1)
    power = transition
    diagonals: list[Any] = []
    for _step in range(length):
        diagonals.append(torch_mod.diagonal(power))
        power = power @ transition
    return torch_mod.stack(diagonals, dim=1).to(dtype=torch_mod.float32)


def prepare_graphgps_random_walk_pe(
    data: Any,
    *,
    walk_length: int = GRAPHGPS_RWPE_WALK_LENGTH,
) -> Any:
    """Attach topology-only RWPE during preprocessing, never during selection."""

    length = _positive_int(walk_length, field="walk_length")
    if length != GRAPHGPS_RWPE_WALK_LENGTH:
        raise ValueError(
            f"GraphGPS RWPE walk_length must remain {GRAPHGPS_RWPE_WALK_LENGTH}"
        )
    edge_index = getattr(data, "edge_index", None)
    num_nodes = getattr(data, "num_nodes", None)
    if edge_index is None or num_nodes is None:
        raise ValueError("GraphGPS preprocessing requires data.edge_index/num_nodes")
    try:
        from torch_geometric.transforms import AddRandomWalkPE
    except ImportError:
        data.random_walk_pe = compute_topology_only_random_walk_pe(
            edge_index, num_nodes=int(num_nodes), walk_length=length
        )
        return data
    transformed = AddRandomWalkPE(
        walk_length=length, attr_name="random_walk_pe"
    )(data)
    return transformed


def validate_graphgps_random_walk_pe(
    positional_encoding: Any,
    *,
    num_nodes: int,
    walk_length: int = GRAPHGPS_RWPE_WALK_LENGTH,
) -> Any:
    if positional_encoding is None:
        raise ValueError(
            "GraphGPS requires preprocessing-time random_walk_pe; labels and "
            "test metrics may not be used to synthesize it at model runtime."
        )
    if positional_encoding.ndim != 2 or tuple(positional_encoding.shape) != (
        int(num_nodes),
        int(walk_length),
    ):
        raise ValueError(
            "random_walk_pe shape differs from [num_nodes, rwpe_walk_length]"
        )
    return positional_encoding


@dataclass(frozen=True, slots=True)
class GraphGPSMolecularConfig:
    backbone: str = "gps"
    num_classes: int = 2
    num_layers: int = GRAPHGPS_NUM_LAYERS
    hidden_dim: int = 160
    dropout: float = GRAPHGPS_DROPOUT
    pooling: str = GRAPHGPS_POOLING
    readout_layers: int = 2
    normalization: str = "batch_norm"
    residual: bool = True
    rwpe_walk_length: int = GRAPHGPS_RWPE_WALK_LENGTH
    attention_heads: int = GRAPHGPS_ATTENTION_HEADS
    local_mpnn: str = GRAPHGPS_LOCAL_MPNN
    global_attention: str = GRAPHGPS_GLOBAL_ATTENTION
    backend: str = "auto"

    def __post_init__(self) -> None:
        if self.backbone != "gps":
            raise ValueError("GraphGPS config backbone must be gps")
        if int(self.hidden_dim) not in GRAPHGPS_ALLOWED_HIDDEN_DIMS:
            raise ValueError("GraphGPS hidden_dim is outside the pre-registered set")
        fixed = {
            "num_layers": (int(self.num_layers), GRAPHGPS_NUM_LAYERS),
            "rwpe_walk_length": (
                int(self.rwpe_walk_length),
                GRAPHGPS_RWPE_WALK_LENGTH,
            ),
            "attention_heads": (
                int(self.attention_heads),
                GRAPHGPS_ATTENTION_HEADS,
            ),
            "local_mpnn": (str(self.local_mpnn), GRAPHGPS_LOCAL_MPNN),
            "global_attention": (
                str(self.global_attention),
                GRAPHGPS_GLOBAL_ATTENTION,
            ),
            "pooling": (str(self.pooling), GRAPHGPS_POOLING),
        }
        changed = [field for field, values in fixed.items() if values[0] != values[1]]
        if changed:
            raise ValueError(f"Frozen GraphGPS fields changed: {changed}")
        if int(self.hidden_dim) % int(self.attention_heads) != 0:
            raise ValueError("GraphGPS hidden_dim must be divisible by attention_heads")
        if int(self.num_classes) < 2 or int(self.readout_layers) != 2:
            raise ValueError("GraphGPS requires num_classes>=2 and two readout layers")
        if float(self.dropout) != GRAPHGPS_DROPOUT:
            raise ValueError("GraphGPS dropout must remain 0.2")
        if self.normalization != "batch_norm" or self.residual is not True:
            raise ValueError("GraphGPS requires residual affine batch normalization")
        if self.backend not in {"auto", "pyg_gpsconv", "project_fallback"}:
            raise ValueError("Unsupported GraphGPS backend")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["edge_feature_mode"] = get_gnn_backbone_spec(
            "gps"
        ).edge_feature_mode
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GraphGPSMolecularConfig":
        source = payload.get("gnn", payload)
        if not isinstance(source, Mapping):
            raise ValueError("GraphGPS config must be a mapping")
        fields = set(cls.__dataclass_fields__)
        unknown = sorted(set(source).difference(fields | {"edge_feature_mode"}))
        if unknown:
            raise ValueError(f"Unsupported GraphGPS configuration fields: {unknown}")
        return cls(**{key: source[key] for key in fields if key in source})


_ModuleBase = nn.Module if nn is not None else object


class _ProjectFallbackGPSLayer(_ModuleBase):
    """Minimal local-GINE + global-attention fallback for non-PyG runtimes."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        torch_mod, torch_nn = _require_torch()
        super().__init__()
        self.eps = torch_nn.Parameter(torch_mod.zeros(1))
        self.local_mlp = torch_nn.Sequential(
            torch_nn.Linear(hidden_dim, hidden_dim * 2),
            torch_nn.ReLU(),
            torch_nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.attention = torch_nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = torch_nn.Sequential(
            torch_nn.Linear(hidden_dim, hidden_dim * 2),
            torch_nn.ReLU(),
            torch_nn.Dropout(dropout),
            torch_nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm_local = torch_nn.BatchNorm1d(hidden_dim)
        self.norm_attention = torch_nn.BatchNorm1d(hidden_dim)
        self.norm_ff = torch_nn.BatchNorm1d(hidden_dim)
        self.dropout = torch_nn.Dropout(dropout)

    def _local(self, x: Any, edge_index: Any, edge_attr: Any) -> Any:
        torch_mod, _torch_nn = _require_torch()
        sources = edge_index[0].long()
        destinations = edge_index[1].long()
        messages = torch_mod.relu(x[sources] + edge_attr)
        aggregated = torch_mod.zeros_like(x)
        if messages.numel():
            aggregated.index_add_(0, destinations, messages)
        return self.local_mlp((1.0 + self.eps) * x + aggregated)

    def _global(self, x: Any, batch: Any) -> Any:
        torch_mod, _torch_nn = _require_torch()
        output = torch_mod.zeros_like(x)
        for graph_id in torch_mod.unique(batch, sorted=True):
            indices = torch_mod.nonzero(batch == graph_id, as_tuple=False).reshape(-1)
            sequence = x[indices].unsqueeze(0)
            attended, _weights = self.attention(
                sequence, sequence, sequence, need_weights=False
            )
            output[indices] = attended.squeeze(0)
        return output

    def forward(self, x: Any, edge_index: Any, batch: Any, edge_attr: Any) -> Any:
        local = self.norm_local(x + self.dropout(self._local(x, edge_index, edge_attr)))
        global_hidden = self.norm_attention(
            local + self.dropout(self._global(local, batch))
        )
        return self.norm_ff(
            global_hidden + self.dropout(self.feed_forward(global_hidden))
        )


class GraphGPSMolecularGNN(_ModuleBase):
    """Molecular GraphGPS classifier sharing categorical atom/bond features."""

    def __init__(
        self,
        config: GraphGPSMolecularConfig,
        *,
        node_cardinalities: Sequence[int],
        edge_cardinalities: Sequence[int],
    ) -> None:
        _torch, torch_nn = _require_torch()
        from src.models.molecular_gnn import CategoricalFeatureEncoder

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
        self.rwpe_encoder = torch_nn.Linear(
            config.rwpe_walk_length, config.hidden_dim
        )
        capabilities = graphgps_runtime_capabilities()
        use_pyg = capabilities["gpsconv_available"] and config.backend != "project_fallback"
        if config.backend == "pyg_gpsconv" and not use_pyg:
            raise RuntimeError("GraphGPS config requires PyG GPSConv, but it is unavailable")
        self.graphgps_backend = "pyg_gpsconv" if use_pyg else "project_fallback"
        if use_pyg:
            from torch_geometric.nn import GINEConv, GPSConv

            layers: list[Any] = []
            for _index in range(config.num_layers):
                local_mlp = torch_nn.Sequential(
                    torch_nn.Linear(config.hidden_dim, config.hidden_dim * 2),
                    torch_nn.ReLU(),
                    torch_nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                )
                local = GINEConv(local_mlp, train_eps=True, edge_dim=None)
                layers.append(
                    GPSConv(
                        channels=config.hidden_dim,
                        conv=local,
                        heads=config.attention_heads,
                        dropout=config.dropout,
                        act="relu",
                        norm="batch_norm",
                        attn_type="multihead",
                    )
                )
            self.layers = torch_nn.ModuleList(layers)
        else:
            self.layers = torch_nn.ModuleList(
                [
                    _ProjectFallbackGPSLayer(
                        config.hidden_dim,
                        config.attention_heads,
                        config.dropout,
                    )
                    for _index in range(config.num_layers)
                ]
            )
        self.classifier = torch_nn.Sequential(
            torch_nn.Linear(config.hidden_dim, config.hidden_dim),
            torch_nn.ReLU(),
            torch_nn.Dropout(config.dropout),
            torch_nn.Linear(config.hidden_dim, config.num_classes),
        )

    def _pool(self, node_hidden: Any, batch: Any) -> Any:
        torch_mod, _torch_nn = _require_torch()
        if batch.numel() == 0:
            raise ValueError("Cannot pool an empty GraphGPS graph batch")
        num_graphs = int(batch.max().item()) + 1
        pooled = torch_mod.zeros(
            (num_graphs, self.config.hidden_dim),
            dtype=node_hidden.dtype,
            device=node_hidden.device,
        )
        pooled.index_add_(0, batch, node_hidden)
        counts = torch_mod.zeros(
            (num_graphs,), dtype=node_hidden.dtype, device=node_hidden.device
        )
        counts.index_add_(0, batch, torch_mod.ones_like(batch, dtype=node_hidden.dtype))
        return pooled / counts.clamp_min(1.0).unsqueeze(-1)

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
            raise ValueError("GraphGPS requires x, edge_index, and edge_attr")
        if batch is None:
            batch = torch_mod.zeros((int(x.shape[0]),), dtype=torch_mod.long, device=x.device)
        random_walk_pe = validate_graphgps_random_walk_pe(
            random_walk_pe,
            num_nodes=int(x.shape[0]),
            walk_length=self.config.rwpe_walk_length,
        )
        node_hidden = self.node_encoder(x.long()) + self.rwpe_encoder(
            random_walk_pe.to(dtype=torch_mod.float32)
        )
        edge_hidden = self.edge_encoder(edge_attr.long())
        for layer in self.layers:
            if self.graphgps_backend == "pyg_gpsconv":
                node_hidden = layer(
                    node_hidden,
                    edge_index.long(),
                    batch=batch.long(),
                    edge_attr=edge_hidden,
                )
            else:
                node_hidden = layer(
                    node_hidden, edge_index.long(), batch.long(), edge_hidden
                )
        return self.classifier(self._pool(node_hidden, batch.long()))


def _cardinalities(
    schema: MolecularFeatureSchema | Mapping[str, Any] | Sequence[int],
    *,
    kind: str,
) -> tuple[int, ...]:
    if isinstance(schema, MolecularFeatureSchema):
        return schema.node_cardinalities if kind == "node" else schema.edge_cardinalities
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


def build_graphgps_molecular_gnn(
    *,
    num_classes: int,
    node_feature_schema: MolecularFeatureSchema | Mapping[str, Any] | Sequence[int],
    edge_feature_schema: MolecularFeatureSchema | Mapping[str, Any] | Sequence[int],
    num_layers: int = GRAPHGPS_NUM_LAYERS,
    hidden_dim: int = 160,
    dropout: float = GRAPHGPS_DROPOUT,
    pooling: str = GRAPHGPS_POOLING,
    readout_layers: int = 2,
    normalization: str = "batch_norm",
    residual: bool = True,
    rwpe_walk_length: int = GRAPHGPS_RWPE_WALK_LENGTH,
    attention_heads: int = GRAPHGPS_ATTENTION_HEADS,
    local_mpnn: str = GRAPHGPS_LOCAL_MPNN,
    global_attention: str = GRAPHGPS_GLOBAL_ATTENTION,
    backend: str = "auto",
) -> GraphGPSMolecularGNN:
    config = GraphGPSMolecularConfig(
        num_classes=int(num_classes),
        num_layers=int(num_layers),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
        pooling=str(pooling),
        readout_layers=int(readout_layers),
        normalization=str(normalization),
        residual=bool(residual),
        rwpe_walk_length=int(rwpe_walk_length),
        attention_heads=int(attention_heads),
        local_mpnn=str(local_mpnn),
        global_attention=str(global_attention),
        backend=str(backend),
    )
    return GraphGPSMolecularGNN(
        config,
        node_cardinalities=_cardinalities(node_feature_schema, kind="node"),
        edge_cardinalities=_cardinalities(edge_feature_schema, kind="edge"),
    )


__all__ = [
    "GRAPHGPS_ALLOWED_HIDDEN_DIMS",
    "GRAPHGPS_ATTENTION_HEADS",
    "GRAPHGPS_DROPOUT",
    "GRAPHGPS_GLOBAL_ATTENTION",
    "GRAPHGPS_LOCAL_MPNN",
    "GRAPHGPS_NUM_LAYERS",
    "GRAPHGPS_PARAMETER_MATCH_MAX_RELATIVE_DIFFERENCE",
    "GRAPHGPS_POOLING",
    "GRAPHGPS_RWPE_WALK_LENGTH",
    "GraphGPSMolecularConfig",
    "GraphGPSMolecularGNN",
    "GraphGPSParameterCandidate",
    "GraphGPSParameterMatch",
    "build_graphgps_molecular_gnn",
    "compute_topology_only_random_walk_pe",
    "estimate_graphgps_parameter_count",
    "graphgps_runtime_capabilities",
    "match_graphgps_hidden_dim",
    "prepare_graphgps_random_walk_pe",
    "validate_graphgps_random_walk_pe",
]
