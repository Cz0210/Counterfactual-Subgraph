"""Deterministic RDKit-to-categorical-graph featurization for molecular GNNs.

The feature vocabulary is deliberately fixed rather than fitted on one split.
That keeps train/validation/calibration/test encoding identical and makes the
schema a portable part of every frozen classifier checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


UNKNOWN_TOKEN = "<UNK>"
NODE_FEATURE_SCHEMA_VERSION = "molecular_node_features_v1"
EDGE_FEATURE_SCHEMA_VERSION = "molecular_edge_features_v1"
MOLECULAR_GRAPH_SCHEMA_VERSION = "molecular_graph_v1"


def _tokens(values: Iterable[Any]) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if UNKNOWN_TOKEN in normalized:
        raise ValueError(f"{UNKNOWN_TOKEN} is reserved for unknown categories.")
    return normalized + (UNKNOWN_TOKEN,)


@dataclass(frozen=True, slots=True)
class CategoricalFeatureField:
    """One named categorical feature and its immutable vocabulary."""

    name: str
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Categorical feature names must be non-empty.")
        if not self.values or self.values[-1] != UNKNOWN_TOKEN:
            raise ValueError(
                f"Feature {self.name!r} must end with the {UNKNOWN_TOKEN!r} token."
            )
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"Feature {self.name!r} contains duplicate vocabulary items.")

    @property
    def cardinality(self) -> int:
        return len(self.values)

    @property
    def unknown_index(self) -> int:
        return len(self.values) - 1

    def encode(self, value: Any) -> int:
        token = str(value)
        try:
            return self.values.index(token)
        except ValueError:
            return self.unknown_index

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "values": list(self.values),
            "cardinality": self.cardinality,
            "unknown_index": self.unknown_index,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CategoricalFeatureField":
        return cls(
            name=str(payload["name"]),
            values=tuple(str(value) for value in payload["values"]),
        )


@dataclass(frozen=True, slots=True)
class MolecularFeatureSchema:
    """Complete node/edge vocabulary required to reconstruct a classifier."""

    node_fields: tuple[CategoricalFeatureField, ...]
    edge_fields: tuple[CategoricalFeatureField, ...]
    node_schema_version: str = NODE_FEATURE_SCHEMA_VERSION
    edge_schema_version: str = EDGE_FEATURE_SCHEMA_VERSION
    graph_schema_version: str = MOLECULAR_GRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        node_names = [field.name for field in self.node_fields]
        edge_names = [field.name for field in self.edge_fields]
        if len(set(node_names)) != len(node_names):
            raise ValueError("Node feature field names must be unique.")
        if len(set(edge_names)) != len(edge_names):
            raise ValueError("Edge feature field names must be unique.")
        if not node_names or not edge_names:
            raise ValueError("Molecular feature schemas require node and edge fields.")

    @property
    def node_cardinalities(self) -> tuple[int, ...]:
        return tuple(field.cardinality for field in self.node_fields)

    @property
    def edge_cardinalities(self) -> tuple[int, ...]:
        return tuple(field.cardinality for field in self.edge_fields)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "graph_schema_version": self.graph_schema_version,
            "node_feature_schema": {
                "version": self.node_schema_version,
                "fields": [field.to_dict() for field in self.node_fields],
            },
            "edge_feature_schema": {
                "version": self.edge_schema_version,
                "fields": [field.to_dict() for field in self.edge_fields],
            },
        }
        payload["schema_sha256"] = _stable_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MolecularFeatureSchema":
        node = payload["node_feature_schema"]
        edge = payload["edge_feature_schema"]
        schema = cls(
            node_fields=tuple(
                CategoricalFeatureField.from_dict(field) for field in node["fields"]
            ),
            edge_fields=tuple(
                CategoricalFeatureField.from_dict(field) for field in edge["fields"]
            ),
            node_schema_version=str(node["version"]),
            edge_schema_version=str(edge["version"]),
            graph_schema_version=str(payload["graph_schema_version"]),
        )
        claimed = payload.get("schema_sha256")
        if claimed is not None:
            canonical = schema.to_dict()["schema_sha256"]
            if str(claimed) != canonical:
                raise ValueError(
                    "Molecular feature schema SHA256 does not match its content."
                )
        return schema


def default_molecular_feature_schema() -> MolecularFeatureSchema:
    """Return the project-wide v1 atom and bond categorical vocabulary."""

    return MolecularFeatureSchema(
        node_fields=(
            CategoricalFeatureField("atomic_num", _tokens(range(0, 119))),
            CategoricalFeatureField("degree", _tokens(range(0, 11))),
            CategoricalFeatureField("formal_charge", _tokens(range(-5, 6))),
            CategoricalFeatureField(
                "chirality",
                _tokens(
                    (
                        "CHI_UNSPECIFIED",
                        "CHI_TETRAHEDRAL_CW",
                        "CHI_TETRAHEDRAL_CCW",
                        "CHI_OTHER",
                    )
                ),
            ),
            CategoricalFeatureField("total_hydrogens", _tokens(range(0, 9))),
            CategoricalFeatureField(
                "hybridization",
                _tokens(("S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "UNSPECIFIED")),
            ),
            CategoricalFeatureField("is_aromatic", _tokens((0, 1))),
            CategoricalFeatureField("is_in_ring", _tokens((0, 1))),
        ),
        edge_fields=(
            CategoricalFeatureField(
                "bond_type", _tokens(("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "OTHER"))
            ),
            CategoricalFeatureField(
                "stereo",
                _tokens(
                    (
                        "STEREONONE",
                        "STEREOANY",
                        "STEREOZ",
                        "STEREOE",
                        "STEREOCIS",
                        "STEREOTRANS",
                    )
                ),
            ),
            CategoricalFeatureField("is_conjugated", _tokens((0, 1))),
            CategoricalFeatureField("is_in_ring", _tokens((0, 1))),
        ),
    )


@dataclass(frozen=True, slots=True)
class MolecularGraphFeatures:
    """Portable graph tensors represented as immutable Python tuples."""

    canonical_smiles: str
    node_features: tuple[tuple[int, ...], ...]
    edge_index: tuple[tuple[int, ...], tuple[int, ...]]
    edge_features: tuple[tuple[int, ...], ...]
    schema_sha256: str
    graph_sha256: str

    def __post_init__(self) -> None:
        if not self.canonical_smiles:
            raise ValueError("A molecular graph requires non-empty canonical SMILES.")
        if not self.node_features:
            raise ValueError("A molecular graph requires at least one atom.")
        if len(self.edge_index) != 2:
            raise ValueError("edge_index must have exactly two rows.")
        if len(self.edge_index[0]) != len(self.edge_index[1]):
            raise ValueError("edge_index source and destination sizes differ.")
        if len(self.edge_features) != len(self.edge_index[0]):
            raise ValueError("Each directed edge must have one feature row.")

    @property
    def num_nodes(self) -> int:
        return len(self.node_features)

    @property
    def num_directed_edges(self) -> int:
        return len(self.edge_features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_smiles": self.canonical_smiles,
            "node_features": [list(row) for row in self.node_features],
            "edge_index": [list(row) for row in self.edge_index],
            "edge_features": [list(row) for row in self.edge_features],
            "num_nodes": self.num_nodes,
            "num_directed_edges": self.num_directed_edges,
            "schema_sha256": self.schema_sha256,
            "graph_sha256": self.graph_sha256,
        }


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_rdkit() -> Any:
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - checked in runtime environments.
        raise RuntimeError(
            "Molecular graph featurization requires RDKit. Activate the project "
            "environment before preparing BACE or TasteMolNet graphs."
        ) from exc
    return Chem


class MolecularGraphFeaturizer:
    """Convert one sanitized SMILES molecule into the frozen categorical schema."""

    def __init__(
        self,
        schema: MolecularFeatureSchema | None = None,
        *,
        require_single_component: bool = True,
    ) -> None:
        self.schema = schema or default_molecular_feature_schema()
        self.require_single_component = bool(require_single_component)
        self._node_by_name = {field.name: field for field in self.schema.node_fields}
        self._edge_by_name = {field.name: field for field in self.schema.edge_fields}

    def featurize(self, smiles: str) -> MolecularGraphFeatures:
        Chem = _require_rdkit()
        raw = str(smiles or "").strip()
        if not raw:
            raise ValueError("Cannot featurize empty SMILES.")
        try:
            molecule = Chem.MolFromSmiles(raw)
            if molecule is not None:
                Chem.SanitizeMol(molecule)
        except Exception as exc:
            raise ValueError(f"SMILES sanitization failed: {raw!r}") from exc
        if molecule is None:
            raise ValueError(f"SMILES parsing failed: {raw!r}")
        fragments = Chem.GetMolFrags(molecule)
        if self.require_single_component and len(fragments) != 1:
            raise ValueError(
                "Molecular GNN input must contain exactly one connected component: "
                f"{raw!r}"
            )
        canonical = str(
            Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        )
        node_features = tuple(self._encode_atom(atom) for atom in molecule.GetAtoms())
        sources: list[int] = []
        targets: list[int] = []
        edge_features: list[tuple[int, ...]] = []
        bonds = sorted(
            molecule.GetBonds(),
            key=lambda bond: (
                min(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())),
                max(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())),
                int(bond.GetIdx()),
            ),
        )
        for bond in bonds:
            begin = int(bond.GetBeginAtomIdx())
            end = int(bond.GetEndAtomIdx())
            encoded = self._encode_bond(bond)
            sources.extend((begin, end))
            targets.extend((end, begin))
            edge_features.extend((encoded, encoded))
        schema_sha256 = str(self.schema.to_dict()["schema_sha256"])
        graph_payload = {
            "canonical_smiles": canonical,
            "node_features": node_features,
            "edge_index": (tuple(sources), tuple(targets)),
            "edge_features": edge_features,
            "schema_sha256": schema_sha256,
        }
        return MolecularGraphFeatures(
            canonical_smiles=canonical,
            node_features=node_features,
            edge_index=(tuple(sources), tuple(targets)),
            edge_features=tuple(edge_features),
            schema_sha256=schema_sha256,
            graph_sha256=_stable_hash(graph_payload),
        )

    def featurize_many(self, smiles_values: Sequence[str]) -> list[MolecularGraphFeatures]:
        return [self.featurize(smiles) for smiles in smiles_values]

    def _encode_atom(self, atom: Any) -> tuple[int, ...]:
        values = {
            "atomic_num": int(atom.GetAtomicNum()),
            "degree": int(atom.GetDegree()),
            "formal_charge": int(atom.GetFormalCharge()),
            "chirality": str(atom.GetChiralTag()),
            "total_hydrogens": int(atom.GetTotalNumHs()),
            "hybridization": str(atom.GetHybridization()),
            "is_aromatic": int(bool(atom.GetIsAromatic())),
            "is_in_ring": int(bool(atom.IsInRing())),
        }
        return tuple(
            self._node_by_name[field.name].encode(values[field.name])
            for field in self.schema.node_fields
        )

    def _encode_bond(self, bond: Any) -> tuple[int, ...]:
        bond_type = str(bond.GetBondType()).upper()
        if bond_type not in {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}:
            bond_type = "OTHER"
        values = {
            "bond_type": bond_type,
            "stereo": str(bond.GetStereo()),
            "is_conjugated": int(bool(bond.GetIsConjugated())),
            "is_in_ring": int(bool(bond.IsInRing())),
        }
        return tuple(
            self._edge_by_name[field.name].encode(values[field.name])
            for field in self.schema.edge_fields
        )


__all__ = [
    "CategoricalFeatureField",
    "EDGE_FEATURE_SCHEMA_VERSION",
    "MOLECULAR_GRAPH_SCHEMA_VERSION",
    "MolecularFeatureSchema",
    "MolecularGraphFeatures",
    "MolecularGraphFeaturizer",
    "NODE_FEATURE_SCHEMA_VERSION",
    "UNKNOWN_TOKEN",
    "default_molecular_feature_schema",
]
