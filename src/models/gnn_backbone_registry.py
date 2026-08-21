"""Registry of molecular message-passing backbones.

The registry is intentionally declarative: every backbone consumes the same
encoded atom and bond information, while the recorded ``edge_feature_mode``
documents how that information enters its message function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class GNNBackboneSpec:
    name: str
    display_name: str
    edge_feature_mode: str
    description: str
    aliases: tuple[str, ...] = ()


_BACKBONES: dict[str, GNNBackboneSpec] = {}
_ALIASES: dict[str, str] = {}


def register_gnn_backbone(spec: GNNBackboneSpec) -> None:
    name = str(spec.name).strip().lower()
    if not name or name != spec.name:
        raise ValueError("Backbone registry names must be normalized lowercase strings.")
    if name in _BACKBONES:
        raise ValueError(f"Molecular GNN backbone is already registered: {name}")
    aliases = tuple(str(alias).strip().lower() for alias in spec.aliases)
    collisions = [alias for alias in (name, *aliases) if alias in _ALIASES]
    if collisions:
        raise ValueError(f"Molecular GNN backbone aliases collide: {collisions}")
    _BACKBONES[name] = spec
    for alias in (name, *aliases):
        _ALIASES[alias] = name


def normalize_gnn_backbone(name: str) -> str:
    normalized = str(name or "").strip().lower().replace("-", "").replace("_", "")
    canonical = _ALIASES.get(normalized)
    if canonical is None:
        raise ValueError(
            f"Unknown molecular GNN backbone {name!r}; "
            f"available={','.join(available_gnn_backbones())}"
        )
    return canonical


def get_gnn_backbone_spec(name: str) -> GNNBackboneSpec:
    return _BACKBONES[normalize_gnn_backbone(name)]


def available_gnn_backbones() -> tuple[str, ...]:
    return tuple(sorted(_BACKBONES))


def iter_gnn_backbone_specs() -> Iterable[GNNBackboneSpec]:
    for name in available_gnn_backbones():
        yield _BACKBONES[name]


register_gnn_backbone(
    GNNBackboneSpec(
        name="gine",
        display_name="GINE",
        edge_feature_mode="native_edge_conditioned_message",
        description="GIN-style sum aggregation with learned bond-conditioned messages.",
        aliases=("gineconv",),
    )
)
register_gnn_backbone(
    GNNBackboneSpec(
        name="gin",
        display_name="GIN",
        edge_feature_mode="additive_edge_conditioned_message",
        description="GIN aggregation with the shared learned bond embedding added to messages.",
        aliases=("ginconv",),
    )
)
register_gnn_backbone(
    GNNBackboneSpec(
        name="gcn",
        display_name="GCN",
        edge_feature_mode="normalized_additive_edge_conditioned_message",
        description="Degree-normalized graph convolution retaining shared bond embeddings.",
        aliases=("gcnconv",),
    )
)
register_gnn_backbone(
    GNNBackboneSpec(
        name="gatv2",
        display_name="GATv2",
        edge_feature_mode="native_edge_conditioned_attention",
        description="Dynamic attention over atom pairs and the shared learned bond embedding.",
        aliases=("gat2", "gatv2conv"),
    )
)


__all__ = [
    "GNNBackboneSpec",
    "available_gnn_backbones",
    "get_gnn_backbone_spec",
    "iter_gnn_backbone_specs",
    "normalize_gnn_backbone",
    "register_gnn_backbone",
]
