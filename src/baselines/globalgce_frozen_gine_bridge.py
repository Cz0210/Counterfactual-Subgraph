"""Differentiable access to the exact frozen molecular GINE for GlobalGCE.

GlobalGCE decodes continuous dense node, adjacency, and bond tensors.  The
paper classifier instead consumes the project's categorical molecular graph
schema.  This module connects those boundaries without training a surrogate:

* the hard forward is the ordinary frozen :class:`MolecularGNN` computation;
* straight-through embedding expectations carry gradients only to GlobalGCE
  transformation tensors;
* every classifier parameter is frozen and the calibrated temperature is
  preserved;
* hard products still have to pass RDKit and the ordinary oracle before they
  can become final counterfactuals.

The relaxation is therefore a training bridge, not a replacement oracle and
not a new counterfactual action.  Native LHS->RHS attachment semantics remain
owned by :mod:`src.baselines.globalgce_bace_native_rules`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx

from src.data.molecular_graph_featurizer import (
    MolecularFeatureSchema,
    MolecularGraphFeaturizer,
)
from src.oracles.gnn_oracle import load_gnn_checkpoint_bundle, sha256_file


BRIDGE_SCHEMA_VERSION = "bace_globalgce_frozen_gine_st_bridge_v2"
EDGE_SCORE_RELAXATION = "pinned_official_affine_scores_softmax_dim_minus_one_v1"


class FrozenGINEBridgeError(ValueError):
    """The soft graph cannot be evaluated under the frozen-GINE contract."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - AutoDL runtime dependency.
        raise RuntimeError("The GlobalGCE frozen-GINE bridge requires PyTorch.") from exc
    return torch


def _rdkit() -> Any:
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - AutoDL runtime dependency.
        raise RuntimeError("The GlobalGCE frozen-GINE bridge requires RDKit.") from exc
    return Chem


def _normalize_distribution(value: Any, *, name: str) -> Any:
    torch = _torch()
    if value.ndim != 2 or int(value.shape[-1]) < 2:
        raise FrozenGINEBridgeError(f"{name} must be a rank-2 class tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise FrozenGINEBridgeError(f"{name} contains non-finite values")
    if bool((value < 0.0).any().item()):
        raise FrozenGINEBridgeError(f"{name} contains negative class weights")
    denominator = value.sum(dim=-1, keepdim=True)
    if bool((denominator <= 0.0).any().item()):
        raise FrozenGINEBridgeError(f"{name} has an empty class distribution")
    return value / denominator


def _categorical_score_distribution(value: Any, *, name: str) -> Any:
    """Map pinned-official categorical scores to a differentiable simplex.

    Pinned GlobalGCE commit ``157e65c`` does *not* apply a sigmoid to the
    reconstructed edge attributes.  Its apparent ``nn.Sigmoid()`` is passed as
    the third positional ``bias`` argument of the final ``nn.Linear`` instead
    of being appended to ``nn.Sequential``.  The resulting finite affine
    scores are therefore allowed to be negative and are hard-decoded with
    ``argmax`` by the official graph codec.

    Softmax is the order-preserving categorical relaxation of those scores:
    it accepts the full official score domain, preserves the exact hard
    ``argmax`` class, and carries gradients without clipping or changing the
    frozen classifier.  This function is deliberately separate from
    ``_normalize_distribution`` because node decoder values and native one-hot
    rows remain non-negative weights under the pinned implementation.
    """

    torch = _torch()
    if value.ndim != 2 or int(value.shape[-1]) < 2:
        raise FrozenGINEBridgeError(f"{name} must be a rank-2 class tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise FrozenGINEBridgeError(f"{name} contains non-finite values")
    return torch.softmax(value, dim=-1)


def _straight_through(hard: Any, soft: Any) -> Any:
    """Return ``hard`` numerically while retaining ``soft`` gradients."""

    if hard.shape != soft.shape:
        raise FrozenGINEBridgeError(
            f"Straight-through tensors differ: hard={hard.shape}, soft={soft.shape}"
        )
    return hard + soft - soft.detach()


def _edge_position(left: int, right: int) -> int:
    high, low = max(int(left), int(right)), min(int(left), int(right))
    if high == low:
        raise FrozenGINEBridgeError("Self loops have no GlobalGCE edge slot")
    return (high - 1) * high // 2 + low


def _schema_index(schema: MolecularFeatureSchema, *, kind: str, name: str) -> int:
    fields = schema.node_fields if kind == "node" else schema.edge_fields
    matches = [index for index, field in enumerate(fields) if field.name == name]
    if len(matches) != 1:
        raise FrozenGINEBridgeError(
            f"Frozen GINE schema must contain one {kind} field {name!r}"
        )
    return matches[0]


def _field_encode(
    schema: MolecularFeatureSchema, *, kind: str, name: str, value: Any
) -> int:
    fields = schema.node_fields if kind == "node" else schema.edge_fields
    return int(fields[_schema_index(schema, kind=kind, name=name)].encode(value))


def _bond_type(name: str) -> Any:
    Chem = _rdkit()
    normalized = str(name).strip().lower()
    mapping = {
        "single": Chem.BondType.SINGLE,
        "double": Chem.BondType.DOUBLE,
        "triple": Chem.BondType.TRIPLE,
        "aromatic": Chem.BondType.AROMATIC,
    }
    if normalized not in mapping:
        raise FrozenGINEBridgeError(f"Unsupported GlobalGCE bond class: {name!r}")
    return mapping[normalized]


@dataclass(frozen=True, slots=True)
class _HardGraph:
    x: Any
    active_native_indices: tuple[int, ...]
    native_to_active: Mapping[int, int]
    edge_features: Mapping[tuple[int, int], tuple[int, ...]]
    hard_edges: frozenset[tuple[int, int]]
    sanitized: bool
    failure_reason: str | None


def _fallback_node_row(
    schema: MolecularFeatureSchema,
    *,
    atomic_number: int,
    degree: int,
    in_ring: bool,
) -> tuple[int, ...]:
    values = {
        "atomic_num": atomic_number,
        "degree": degree,
        "formal_charge": 0,
        "chirality": "CHI_UNSPECIFIED",
        "total_hydrogens": 0,
        "hybridization": "UNSPECIFIED",
        "is_aromatic": 0,
        "is_in_ring": int(in_ring),
    }
    return tuple(field.encode(values[field.name]) for field in schema.node_fields)


def _fallback_edge_row(
    schema: MolecularFeatureSchema,
    *,
    bond_name: str,
    in_ring: bool,
) -> tuple[int, ...]:
    values = {
        "bond_type": str(bond_name).strip().upper(),
        "stereo": "STEREONONE",
        "is_conjugated": 0,
        "is_in_ring": int(in_ring),
    }
    return tuple(field.encode(values[field.name]) for field in schema.edge_fields)


def _hard_graph(
    *,
    features: Any,
    adjacency: Any,
    edge_attributes: Any,
    atom_symbols: Sequence[str],
    bond_names: Sequence[str],
    schema: MolecularFeatureSchema,
) -> _HardGraph:
    """Hard-decode one dense graph while retaining the native node order."""

    torch = _torch()
    Chem = _rdkit()
    node_labels = features.argmax(dim=-1)
    active_native = tuple(
        index
        for index in range(int(features.shape[0]))
        if int(node_labels[index].item()) > 0
    )
    if not active_native:
        raise FrozenGINEBridgeError("GlobalGCE hard graph contains no active atom")
    native_to_active = {native: index for index, native in enumerate(active_native)}
    graph = nx.Graph()
    graph.add_nodes_from(active_native)
    hard_edges: set[tuple[int, int]] = set()
    hard_bonds: dict[tuple[int, int], str] = {}
    for position, left in enumerate(active_native):
        for right in active_native[position + 1 :]:
            forward = float(adjacency[left, right].detach().item()) > 0.5
            reverse = float(adjacency[right, left].detach().item()) > 0.5
            if forward != reverse:
                raise FrozenGINEBridgeError(
                    f"GlobalGCE hard adjacency is asymmetric at ({left},{right})"
                )
            if not forward:
                continue
            edge_row = edge_attributes[_edge_position(left, right)]
            label = int(edge_row.argmax(dim=-1).detach().item())
            if label <= 0:
                # A soft decoder can transiently disagree between its dense
                # adjacency gate and explicit no-edge bond class.  The exact
                # hard graph uses their conjunction; final native-rule
                # validation still rejects inconsistent frozen rule tensors.
                continue
            if label >= len(bond_names):
                raise FrozenGINEBridgeError(
                    f"Active GlobalGCE edge has invalid bond label={label}"
                )
            pair = (native_to_active[left], native_to_active[right])
            hard_edges.add(pair)
            hard_bonds[pair] = str(bond_names[label])
            graph.add_edge(left, right)

    editable = Chem.RWMol()
    for native in active_native:
        label = int(node_labels[native].detach().item())
        if label <= 0 or label > len(atom_symbols):
            raise FrozenGINEBridgeError(
                f"GlobalGCE hard graph has invalid atom label={label}"
            )
        editable.AddAtom(Chem.Atom(str(atom_symbols[label - 1])))
    for (left, right), name in sorted(hard_bonds.items()):
        editable.AddBond(left, right, _bond_type(name))
        if str(name).strip().lower() == "aromatic":
            editable.GetAtomWithIdx(left).SetIsAromatic(True)
            editable.GetAtomWithIdx(right).SetIsAromatic(True)
    molecule = editable.GetMol()
    sanitized = True
    failure: str | None = None
    try:
        Chem.SanitizeMol(molecule)
        if len(Chem.GetMolFrags(molecule)) != 1:
            raise ValueError("hard graph is disconnected")
    except Exception as exc:
        sanitized = False
        failure = f"{type(exc).__name__}:{exc}"

    featurizer = MolecularGraphFeaturizer(schema)
    if sanitized:
        node_rows = tuple(featurizer._encode_atom(atom) for atom in molecule.GetAtoms())
        edge_rows: dict[tuple[int, int], tuple[int, ...]] = {}
        for bond in molecule.GetBonds():
            left, right = sorted(
                (int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx()))
            )
            edge_rows[(left, right)] = featurizer._encode_bond(bond)
    else:
        cycle_edges: set[tuple[int, int]] = set()
        for cycle in nx.cycle_basis(graph):
            for index, left_native in enumerate(cycle):
                right_native = cycle[(index + 1) % len(cycle)]
                pair = tuple(
                    sorted(
                        (
                            native_to_active[left_native],
                            native_to_active[right_native],
                        )
                    )
                )
                cycle_edges.add(pair)
        cycle_nodes = {node for pair in cycle_edges for node in pair}
        periodic = Chem.GetPeriodicTable()
        node_rows = tuple(
            _fallback_node_row(
                schema,
                atomic_number=int(
                    periodic.GetAtomicNumber(
                        str(atom_symbols[int(node_labels[native].item()) - 1])
                    )
                ),
                degree=int(graph.degree(native)),
                in_ring=native_to_active[native] in cycle_nodes,
            )
            for native in active_native
        )
        edge_rows = {
            pair: _fallback_edge_row(
                schema, bond_name=name, in_ring=pair in cycle_edges
            )
            for pair, name in hard_bonds.items()
        }
    return _HardGraph(
        x=torch.tensor(node_rows, dtype=torch.long, device=features.device),
        active_native_indices=active_native,
        native_to_active=native_to_active,
        edge_features=edge_rows,
        hard_edges=frozenset(hard_edges),
        sanitized=sanitized,
        failure_reason=failure,
    )


class FrozenGINEDifferentiableBridge:
    """Official-GlobalGCE compatible wrapper around one frozen calibrated GINE."""

    def __init__(
        self,
        model: Any,
        *,
        feature_schema: MolecularFeatureSchema,
        atom_symbols: Sequence[str],
        bond_names: Sequence[str],
        checkpoint_id: str,
        temperature: float,
        device: str | Any = "cpu",
        expected_num_classes: int = 2,
    ) -> None:
        torch = _torch()
        if str(getattr(model.config, "backbone", "")).lower() != "gine":
            raise FrozenGINEBridgeError("GlobalGCE bridge requires the frozen GINE")
        if type(expected_num_classes) is not int or expected_num_classes < 2:
            raise FrozenGINEBridgeError(
                "GlobalGCE expected_num_classes must be an exact integer >= 2"
            )
        observed_num_classes = getattr(model.config, "num_classes", None)
        if (
            type(observed_num_classes) is not int
            or observed_num_classes != expected_num_classes
        ):
            raise FrozenGINEBridgeError(
                "GlobalGCE bridge class count differs from its frozen contract"
            )
        if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
            raise FrozenGINEBridgeError("Frozen GINE temperature must be positive")
        symbols = tuple(str(value) for value in atom_symbols)
        bonds = tuple(str(value).strip().lower() for value in bond_names)
        if not symbols or not bonds or bonds[0] != "no_edge":
            raise FrozenGINEBridgeError("GlobalGCE atom/bond vocabularies are invalid")
        self.model = model.to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.feature_schema = feature_schema
        self.atom_symbols = symbols
        self.bond_names = bonds
        self.checkpoint_id = str(checkpoint_id)
        self.temperature = float(temperature)
        self.num_classes = expected_num_classes
        self.device = torch.device(device)
        self.last_audit: dict[str, Any] = {}
        self._atomic_field = _schema_index(
            feature_schema, kind="node", name="atomic_num"
        )
        self._bond_field = _schema_index(
            feature_schema, kind="edge", name="bond_type"
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        atom_symbols: Sequence[str],
        bond_names: Sequence[str],
        device: str | Any = "cpu",
        expected_num_classes: int = 2,
    ) -> "FrozenGINEDifferentiableBridge":
        root = Path(checkpoint_dir).expanduser().resolve(strict=True)
        model, metadata = load_gnn_checkpoint_bundle(root, device=device)
        temperature = float(
            metadata["temperature_scaling"].get("temperature", 1.0)
        )
        return cls(
            model,
            feature_schema=metadata["feature_schema"],
            atom_symbols=atom_symbols,
            bond_names=bond_names,
            checkpoint_id=sha256_file(root / "model.pt"),
            temperature=temperature,
            device=device,
            expected_num_classes=expected_num_classes,
        )

    def train(self, mode: bool = True) -> "FrozenGINEDifferentiableBridge":
        """Match ``nn.Module``'s interface while keeping the classifier in eval."""

        del mode
        self.model.eval()
        return self

    def eval(self) -> "FrozenGINEDifferentiableBridge":
        self.model.eval()
        return self

    def parameters(self, recurse: bool = True) -> Any:
        return self.model.parameters(recurse=recurse)

    def named_parameters(self, recurse: bool = True) -> Any:
        return self.model.named_parameters(recurse=recurse)

    def to(self, device: str | Any) -> "FrozenGINEDifferentiableBridge":
        torch = _torch()
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def _mapped_node_distribution(self, source: Any) -> Any:
        torch = _torch()
        probabilities = _normalize_distribution(source, name="node_features")
        active = probabilities[:, 1:]
        active = active / active.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        cardinality = self.feature_schema.node_fields[self._atomic_field].cardinality
        mapped = torch.zeros(
            (int(source.shape[0]), cardinality),
            dtype=source.dtype,
            device=source.device,
        )
        periodic = _rdkit().GetPeriodicTable()
        for native_label, symbol in enumerate(self.atom_symbols, start=1):
            target = _field_encode(
                self.feature_schema,
                kind="node",
                name="atomic_num",
                value=int(periodic.GetAtomicNumber(symbol)),
            )
            mapped[:, target] = mapped[:, target] + active[:, native_label - 1]
        return mapped

    def _mapped_edge_distribution(self, source: Any) -> tuple[Any, Any]:
        torch = _torch()
        probabilities = _categorical_score_distribution(
            source, name="edge_attributes"
        )
        presence = 1.0 - probabilities[:, 0]
        bonds = probabilities[:, 1:]
        bonds = bonds / bonds.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        cardinality = self.feature_schema.edge_fields[self._bond_field].cardinality
        mapped = torch.zeros(
            (int(source.shape[0]), cardinality),
            dtype=source.dtype,
            device=source.device,
        )
        for native_label, bond in enumerate(self.bond_names[1:], start=1):
            target = _field_encode(
                self.feature_schema,
                kind="edge",
                name="bond_type",
                value=str(bond).upper(),
            )
            mapped[:, target] = mapped[:, target] + bonds[:, native_label - 1]
        return mapped, presence

    def _one_graph(
        self, features: Any, adjacency: Any, edge_attributes: Any
    ) -> tuple[Any, dict[str, Any]]:
        torch = _torch()
        graph = _hard_graph(
            features=features,
            adjacency=adjacency,
            edge_attributes=edge_attributes,
            atom_symbols=self.atom_symbols,
            bond_names=self.bond_names,
            schema=self.feature_schema,
        )
        active_native = graph.active_native_indices
        x = graph.x.to(self.device)
        selected_features = features[list(active_native)].to(self.device)
        node_distribution = self._mapped_node_distribution(selected_features)
        node_hidden = None
        for field, embedding in enumerate(self.model.node_encoder.embeddings):
            hard_value = embedding(x[:, field])
            if field == self._atomic_field:
                soft_value = node_distribution @ embedding.weight
                hard_value = _straight_through(hard_value, soft_value)
            node_hidden = hard_value if node_hidden is None else node_hidden + hard_value
        presence_probability = _normalize_distribution(
            selected_features, name="node_presence"
        )[:, 1:].sum(dim=-1)
        node_gate = _straight_through(
            torch.ones_like(presence_probability), presence_probability
        )
        node_hidden = node_hidden * node_gate.unsqueeze(-1)

        sources: list[int] = []
        destinations: list[int] = []
        hard_gates: list[float] = []
        soft_gates: list[Any] = []
        hard_edge_rows: list[tuple[int, ...]] = []
        soft_edge_rows: list[Any] = []
        default_edge = tuple(
            field.encode(
                {
                    "bond_type": "SINGLE",
                    "stereo": "STEREONONE",
                    "is_conjugated": 0,
                    "is_in_ring": 0,
                }[field.name]
            )
            for field in self.feature_schema.edge_fields
        )
        for source_active, source_native in enumerate(active_native):
            for target_active, target_native in enumerate(active_native):
                if source_active == target_active:
                    continue
                pair = tuple(sorted((source_active, target_active)))
                sources.append(source_active)
                destinations.append(target_active)
                hard_gates.append(1.0 if pair in graph.hard_edges else 0.0)
                soft_gates.append(
                    0.5
                    * (
                        adjacency[source_native, target_native]
                        + adjacency[target_native, source_native]
                    )
                )
                hard_edge_rows.append(graph.edge_features.get(pair, default_edge))
                soft_edge_rows.append(
                    edge_attributes[_edge_position(source_native, target_native)]
                )
        if not sources:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_hidden = node_hidden.new_empty((0, int(node_hidden.shape[-1])))
            edge_gate = node_hidden.new_empty((0,))
        else:
            edge_index = torch.tensor(
                [sources, destinations], dtype=torch.long, device=self.device
            )
            hard_edge = torch.tensor(
                hard_edge_rows, dtype=torch.long, device=self.device
            )
            soft_edge = torch.stack(soft_edge_rows).to(self.device)
            edge_distribution, bond_presence = self._mapped_edge_distribution(
                soft_edge
            )
            edge_hidden = None
            for field, embedding in enumerate(self.model.edge_encoder.embeddings):
                hard_value = embedding(hard_edge[:, field])
                if field == self._bond_field:
                    soft_value = edge_distribution @ embedding.weight
                    hard_value = _straight_through(hard_value, soft_value)
                edge_hidden = hard_value if edge_hidden is None else edge_hidden + hard_value
            adjacency_gate = torch.stack(soft_gates).to(self.device)
            edge_gate = _straight_through(
                torch.tensor(hard_gates, dtype=features.dtype, device=self.device),
                adjacency_gate * bond_presence,
            )

        for layer, normalization in zip(
            self.model.layers, self.model.normalizations, strict=True
        ):
            if str(layer.backbone) != "gine":
                raise FrozenGINEBridgeError("Bridge encountered a non-GINE layer")
            source_index, target_index = edge_index[0], edge_index[1]
            messages = torch.relu(node_hidden[source_index] + edge_hidden)
            messages = messages * edge_gate.unsqueeze(-1)
            aggregated = layer._aggregate_sum(
                messages, target_index, int(node_hidden.shape[0])
            )
            updated = layer.update_mlp((1.0 + layer.eps) * node_hidden + aggregated)
            if self.model.config.residual and updated.shape == node_hidden.shape:
                updated = updated + node_hidden
            node_hidden = self.model.dropout(torch.relu(normalization(updated)))
        pooled = (node_hidden * node_gate.unsqueeze(-1)).sum(dim=0, keepdim=True)
        pooled = pooled / node_gate.sum().clamp_min(1.0)
        logits = self.model.classifier(pooled)
        detached_edge_attributes = edge_attributes.detach()
        edge_score_count = int(detached_edge_attributes.numel())
        return logits, {
            "active_node_count": len(active_native),
            "hard_edge_count": len(graph.hard_edges),
            "hard_graph_sanitized": graph.sanitized,
            "hard_graph_failure_reason": graph.failure_reason,
            "edge_score_relaxation": EDGE_SCORE_RELAXATION,
            "edge_score_negative_value_count": int(
                (edge_attributes < 0.0).sum().detach().item()
            ),
            "edge_score_min": (
                float(detached_edge_attributes.min().item())
                if edge_score_count
                else None
            ),
            "edge_score_max": (
                float(detached_edge_attributes.max().item())
                if edge_score_count
                else None
            ),
        }

    def __call__(self, features: Any, adjacency: Any, edge_attributes: Any) -> dict[str, Any]:
        torch = _torch()
        if features.ndim != 3 or adjacency.ndim != 3 or edge_attributes.ndim != 3:
            raise FrozenGINEBridgeError(
                "GlobalGCE bridge expects batched dense node/adjacency/edge tensors"
            )
        if int(features.shape[0]) != int(adjacency.shape[0]) or int(
            features.shape[0]
        ) != int(edge_attributes.shape[0]):
            raise FrozenGINEBridgeError("GlobalGCE bridge batch dimensions differ")
        if tuple(adjacency.shape[1:]) != (
            int(features.shape[1]),
            int(features.shape[1]),
        ):
            raise FrozenGINEBridgeError("GlobalGCE bridge adjacency shape is invalid")
        expected_edges = int(features.shape[1]) * (int(features.shape[1]) - 1) // 2
        if int(edge_attributes.shape[1]) != expected_edges:
            raise FrozenGINEBridgeError("GlobalGCE bridge edge-vector shape is invalid")
        logits: list[Any] = []
        audits: list[dict[str, Any]] = []
        for index in range(int(features.shape[0])):
            value, audit = self._one_graph(
                features[index], adjacency[index], edge_attributes[index]
            )
            logits.append(value)
            audits.append(audit)
        joined = torch.cat(logits, dim=0)
        log_probabilities = torch.log_softmax(joined / self.temperature, dim=-1)
        self.last_audit = {
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "checkpoint_id": self.checkpoint_id,
            "classifier_family": "gine",
            "num_classes": self.num_classes,
            "rf_oracle_used": False,
            "edge_score_contract": "pinned_official_unbounded_affine_class_scores",
            "edge_score_relaxation": EDGE_SCORE_RELAXATION,
            "hard_edge_decode": "argmax",
            "batch_size": len(audits),
            "hard_graph_sanitized_count": sum(
                row["hard_graph_sanitized"] is True for row in audits
            ),
            "graphs": audits,
        }
        return {
            "y_pred": log_probabilities,
            "logits": joined,
            "bridge_audit": dict(self.last_audit),
        }


class GlobalGCETargetClassAdapter:
    """Expose one reviewed multiclass target as official internal class one.

    Pinned GlobalGCE optimizes ``y_cf=1``.  This adapter is only a loss view:
    internal column zero is the frozen source class, internal column one is
    the requested destination, and remaining frozen classes follow in numeric
    order.  The bridge and every final hard-oracle check retain the original
    calibrated class order.
    """

    def __init__(
        self,
        bridge: FrozenGINEDifferentiableBridge,
        *,
        source_label: int,
        target_label: int,
    ) -> None:
        if (
            type(source_label) is not int
            or type(target_label) is not int
            or not 0 <= source_label < bridge.num_classes
            or not 0 <= target_label < bridge.num_classes
            or source_label == target_label
        ):
            raise FrozenGINEBridgeError(
                "GlobalGCE source/target labels must be distinct exact classes"
            )
        self.bridge = bridge
        self.source_label = source_label
        self.target_label = target_label
        self.class_order = (
            source_label,
            target_label,
            *(
                label
                for label in range(bridge.num_classes)
                if label not in {source_label, target_label}
            ),
        )
        self.last_audit: dict[str, Any] = {}

    def train(self, mode: bool = True) -> "GlobalGCETargetClassAdapter":
        self.bridge.train(mode)
        return self

    def eval(self) -> "GlobalGCETargetClassAdapter":
        self.bridge.eval()
        return self

    def parameters(self, recurse: bool = True) -> Any:
        return self.bridge.parameters(recurse=recurse)

    def named_parameters(self, recurse: bool = True) -> Any:
        return self.bridge.named_parameters(recurse=recurse)

    def to(self, device: str | Any) -> "GlobalGCETargetClassAdapter":
        self.bridge.to(device)
        return self

    def __call__(self, features: Any, adjacency: Any, edge_attributes: Any) -> dict[str, Any]:
        result = self.bridge(features, adjacency, edge_attributes)
        self.last_audit = {
            **self.bridge.last_audit,
            "official_internal_source_label": 0,
            "official_internal_target_label": 1,
            "frozen_source_label": self.source_label,
            "frozen_target_label": self.target_label,
            "frozen_class_order_seen_by_official": list(self.class_order),
            "class_order_adapter": "official_0_1_maps_to_frozen_source_target",
        }
        indices = list(self.class_order)
        return {
            **result,
            "y_pred": result["y_pred"][:, indices],
            "logits": result["logits"][:, indices],
            "bridge_audit": dict(self.last_audit),
        }


class GlobalGCEClassZeroTargetAdapter:
    """Map official GlobalGCE target ``1`` to frozen-BACE class ``0``.

    Pinned GlobalGCE hard-codes ``y_cf=1`` in ``run_one_batch``.  BACE's
    scientific source label is one and its untargeted binary destination is
    zero.  This adapter only swaps the two output columns seen by that loss;
    :class:`FrozenGINEDifferentiableBridge` itself and every final hard-oracle
    verification retain the original calibrated class order.
    """

    def __init__(self, bridge: FrozenGINEDifferentiableBridge) -> None:
        self.bridge = bridge
        self.last_audit: dict[str, Any] = {}

    def train(self, mode: bool = True) -> "GlobalGCEClassZeroTargetAdapter":
        self.bridge.train(mode)
        return self

    def eval(self) -> "GlobalGCEClassZeroTargetAdapter":
        self.bridge.eval()
        return self

    def parameters(self, recurse: bool = True) -> Any:
        return self.bridge.parameters(recurse=recurse)

    def named_parameters(self, recurse: bool = True) -> Any:
        return self.bridge.named_parameters(recurse=recurse)

    def to(self, device: str | Any) -> "GlobalGCEClassZeroTargetAdapter":
        self.bridge.to(device)
        return self

    def __call__(self, features: Any, adjacency: Any, edge_attributes: Any) -> dict[str, Any]:
        result = self.bridge(features, adjacency, edge_attributes)
        self.last_audit = {
            **self.bridge.last_audit,
            "official_internal_target_label": 1,
            "frozen_bace_destination_label": 0,
            "class_order_adapter": "official_1_maps_to_frozen_gine_0",
        }
        return {
            **result,
            "y_pred": result["y_pred"][:, [1, 0]],
            "logits": result["logits"][:, [1, 0]],
            "bridge_audit": dict(self.last_audit),
        }


__all__ = [
    "BRIDGE_SCHEMA_VERSION",
    "EDGE_SCORE_RELAXATION",
    "FrozenGINEBridgeError",
    "FrozenGINEDifferentiableBridge",
    "GlobalGCEClassZeroTargetAdapter",
    "GlobalGCETargetClassAdapter",
]
