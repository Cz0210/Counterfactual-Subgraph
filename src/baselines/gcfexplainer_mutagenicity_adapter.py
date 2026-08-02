"""Strict Mutagenicity adapter for the official GCFExplainer pipeline.

The official implementation owns the GNN, edit-map, VRRW, NeuroSED distance,
importance, teleportation, reinforcement, and greedy summary algorithms.  This
module owns only project data contracts, chemistry sidecars, deterministic
cohort selection, runtime wiring, and RF-target export.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import os
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence

from rdkit import Chem


EXPECTED_TRAIN_SOURCE_ROWS = 1448
EXPECTED_TRAIN_TARGET_ROWS = 1437
EXPECTED_VAL_SOURCE_ROWS = 260
EXPECTED_VAL_TARGET_ROWS = 95
EXPECTED_MODEL_TRAIN_ROWS = 2885
EXPECTED_MODEL_VAL_ROWS = 355
EXPECTED_GENERATION_SOURCE_ROWS = 1448

SOURCE_LABEL = 1
TARGET_LABEL = 0
SEED = 13

# TUDataset Mutagenicity node-label order.  The original dataset has 14 atom
# labels; official data.py removes labels occurring <= 50 times, leaving these
# ten.  Explicit H nodes are materialized with Chem.AddHs so the strict project
# graphs retain the official ten-channel edit space without inventing a label.
OFFICIAL_MUTAGENICITY_ATOMIC_NUMBERS = (6, 8, 17, 1, 7, 9, 35, 16, 15, 53)
REQUIRED_PROJECT_ATOMIC_NUMBERS = frozenset((6, 7, 8, 9, 15, 16, 17, 35, 53))
SUPPORTED_BOND_TYPES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
ATOM_SIDECAR_SCHEMA_VERSION = "gcfexplainer_mutagenicity_atom_sidecar_v1"
GENERATED_CODEC_VERSION = "gcfexplainer_mutagenicity_fullgraph_codec_v1"

OFFICIAL_AIDS_ROUTE_AUDIT: dict[str, Any] = {
    "wrappers": [
        "scripts/slurm/gcfexplainer/train_aids_gnn.sh",
        "scripts/slurm/gcfexplainer/reproduce_aids_vrrw.sh",
        "scripts/slurm/gcfexplainer/reproduce_aids_summary.sh",
        "scripts/slurm/gcfexplainer/reproduce_aids_all.sh",
    ],
    "official_dataset_argument": "aids",
    "official_input_root": "baselines/gcfexplainer_official/data/aids",
    "node_label_vocabulary": {
        "dimension": 9,
        "source": "data.AIDS.num_classes",
        "chemical_atom_mapping_available": False,
    },
    "checkpoint": "baselines/gcfexplainer_official/data/aids/gnn/model_best.pth",
    "vrrw": {
        "alpha": 0.5,
        "theta": 0.05,
        "max_steps": 50000,
        "seed": 0,
        "teleport": 0.1,
    },
    "summary": {
        "theta": 0.1,
        "official_output": "baselines/gcfexplainer_official/results/aids",
        "project_collector": "scripts/eval/collect_gcf_official_results.py",
    },
    "graph_to_smiles": {
        "existing_project_route": "src/eval/gcf_graph_to_smiles.py",
        "reusable_for_mutagenicity": False,
        "reason": "AIDS-specific feature vocabulary and chemistry mapping",
    },
    "generic_official_components": [
        "gnn.GNN",
        "importance.prepare_and_get",
        "vrrw.counterfactual_summary_with_randomwalk",
        "summary.greedy_counterfactual_summary_from_covering_sets",
        "distance.load_neurosed",
    ],
    "aids_specific_components": [
        "dataset path",
        "nine-channel node vocabulary",
        "GNN checkpoint",
        "VRRW alpha",
        "result path",
        "legacy graph-to-SMILES mapping",
    ],
}


class GCFExplainerMutagenicityError(RuntimeError):
    """Base error for strict project adaptation failures."""


class GCFExplainerMutagenicityCodecError(GCFExplainerMutagenicityError):
    """Raised when source or generated graph chemistry cannot be decoded."""


class GCFExplainerEmptyCandidateSetError(GCFExplainerMutagenicityError):
    """Raised when native-rank filtering cannot produce twenty candidates."""


class RunProfile(str, Enum):
    SMOKE = "smoke"
    FULL = "full"


class TeacherProtocol(Protocol):
    available: bool

    def score_smiles(
        self,
        smiles: str,
        label: int | None = None,
        parent_smiles: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class StrictMolecule:
    molecule_id: str
    smiles: str
    canonical_smiles: str
    label: int
    split: str
    semantic_label: str
    source_row_index: int
    source_path: str


@dataclass(frozen=True, slots=True)
class MutagenicityGraphSchema:
    atom_vocabulary: tuple[int, ...]
    feature_atomic_numbers: tuple[int, ...]
    formal_charge_vocabulary: tuple[int, ...]
    aromaticity_vocabulary: tuple[bool, ...]
    bond_type_vocabulary: tuple[str, ...]
    max_num_nodes: int
    atom_sidecar_schema_version: str = ATOM_SIDECAR_SCHEMA_VERSION
    explicit_h_nodes: bool = True

    @property
    def node_feature_dim(self) -> int:
        return len(self.feature_atomic_numbers)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["atom_vocabulary"] = list(self.atom_vocabulary)
        payload["feature_atomic_numbers"] = list(
            self.feature_atomic_numbers
        )
        payload["formal_charge_vocabulary"] = list(
            self.formal_charge_vocabulary
        )
        payload["aromaticity_vocabulary"] = list(
            self.aromaticity_vocabulary
        )
        payload["bond_type_vocabulary"] = list(
            self.bond_type_vocabulary
        )
        payload["node_feature_dim"] = self.node_feature_dim
        payload["native_node_label_semantics"] = "one_hot_atom_type"
        payload["official_dataset_name"] = "mutagenicity"
        payload["official_label_direction"] = {
            "project_source_label_1": 0,
            "project_target_label_0": 1,
        }
        payload["bond_decoder_native"] = False
        payload["native_adjacency_semantics"] = "binary_connectivity"
        payload["retained_bond_type_source"] = "source_sidecar"
        payload["new_edge_bond_rule"] = "provisional_single"
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MutagenicityGraphSchema":
        return cls(
            atom_vocabulary=tuple(int(v) for v in payload["atom_vocabulary"]),
            feature_atomic_numbers=tuple(
                int(v) for v in payload["feature_atomic_numbers"]
            ),
            formal_charge_vocabulary=tuple(
                int(v) for v in payload["formal_charge_vocabulary"]
            ),
            aromaticity_vocabulary=tuple(
                bool(v) for v in payload["aromaticity_vocabulary"]
            ),
            bond_type_vocabulary=tuple(
                str(v) for v in payload["bond_type_vocabulary"]
            ),
            max_num_nodes=int(payload["max_num_nodes"]),
            atom_sidecar_schema_version=str(
                payload.get(
                    "atom_sidecar_schema_version",
                    ATOM_SIDECAR_SCHEMA_VERSION,
                )
            ),
            explicit_h_nodes=bool(payload.get("explicit_h_nodes", True)),
        )


@dataclass(frozen=True, slots=True)
class GeneratedDecodeResult:
    decode_ok: bool
    canonical_smiles: str
    raw_smiles: str
    failure_reason: str
    projected_new_edge_count: int
    retained_edge_count: int
    removed_source_edge_count: int
    inherited_atom_state_count: int
    reset_atom_state_count: int
    source_parent_id: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cohort_hash(rows: Sequence[StrictMolecule] | Sequence[Mapping[str, Any]]) -> str:
    ids = [
        row.molecule_id if isinstance(row, StrictMolecule) else str(row["molecule_id"])
        for row in rows
    ]
    return stable_json_sha256(ids)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return payload


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False))
            handle.write("\n")
    os.replace(temporary, destination)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object.")
            rows.append(payload)
    return rows


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def reject_forbidden_path(path: str | Path) -> None:
    value = str(Path(path)).lower()
    forbidden = (
        "calibration_source",
        "calibration_target",
        "test_source",
        "test_target",
    )
    if any(token in value for token in forbidden):
        raise ValueError(f"Calibration/test input is forbidden: {path}")


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def _canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() <= 0:
        raise ValueError(f"Invalid or empty SMILES: {smiles!r}")
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def load_strict_molecules(
    path: str | Path,
    *,
    expected_split: str,
    expected_label: int,
    expected_rows: int | None,
) -> list[StrictMolecule]:
    source = Path(path).expanduser().resolve()
    reject_forbidden_path(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = [dict(row) for row in csv.DictReader(handle)]
    if expected_rows is not None and len(raw_rows) != int(expected_rows):
        raise ValueError(
            f"Strict cohort row count mismatch for {source.name}: "
            f"expected={expected_rows}, found={len(raw_rows)}."
        )
    required = {
        "molecule_id",
        "smiles",
        "label",
        "split",
        "teacher_pred",
        "teacher_correct",
    }
    missing = sorted(required - set(raw_rows[0] if raw_rows else {}))
    if missing:
        raise ValueError(f"{source.name} is missing columns: {missing}")
    rows: list[StrictMolecule] = []
    for index, raw in enumerate(raw_rows):
        molecule_id = str(raw.get("molecule_id") or "").strip()
        smiles = str(raw.get("smiles") or "").strip()
        split = str(raw.get("split") or "").strip().lower()
        label = int(float(str(raw.get("label"))))
        teacher_pred = int(float(str(raw.get("teacher_pred"))))
        teacher_correct = _parse_bool(raw.get("teacher_correct"))
        if not molecule_id:
            raise ValueError(f"{source.name}:{index + 2} has empty molecule_id.")
        if split != expected_split:
            raise ValueError(
                f"{source.name}:{index + 2} split={split!r}, expected "
                f"{expected_split!r}."
            )
        if label != expected_label or teacher_pred != expected_label:
            raise ValueError(
                f"{source.name}:{index + 2} violates label/teacher contract."
            )
        if not teacher_correct:
            raise ValueError(
                f"{source.name}:{index + 2} is not teacher-correct."
            )
        rows.append(
            StrictMolecule(
                molecule_id=molecule_id,
                smiles=smiles,
                canonical_smiles=_canonical_smiles(smiles),
                label=label,
                split=split,
                semantic_label=str(raw.get("semantic_label") or ""),
                source_row_index=index,
                source_path=str(source),
            )
        )
    ids = [row.molecule_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{source.name} contains duplicate molecule_id values.")
    return sorted(rows, key=lambda row: row.molecule_id)


def _expanded_mol(row: StrictMolecule) -> Any:
    mol = Chem.MolFromSmiles(row.canonical_smiles)
    if mol is None:
        raise GCFExplainerMutagenicityCodecError(
            f"Cannot parse canonical source molecule {row.molecule_id}."
        )
    Chem.SanitizeMol(mol)
    expanded = Chem.AddHs(mol, addCoords=False)
    Chem.SanitizeMol(expanded)
    return expanded


def _bond_name(bond: Any) -> str:
    bond_type = bond.GetBondType()
    mapping = {
        Chem.BondType.SINGLE: "SINGLE",
        Chem.BondType.DOUBLE: "DOUBLE",
        Chem.BondType.TRIPLE: "TRIPLE",
        Chem.BondType.AROMATIC: "AROMATIC",
    }
    if bond_type not in mapping:
        raise GCFExplainerMutagenicityCodecError(
            f"Unsupported source bond type: {bond_type}."
        )
    return mapping[bond_type]


def derive_schema(
    train_rows: Sequence[StrictMolecule],
    val_rows: Sequence[StrictMolecule],
) -> MutagenicityGraphSchema:
    train_atoms: set[int] = set()
    train_charges: set[int] = set()
    train_aromatic: set[bool] = set()
    train_bonds: set[str] = set()
    for row in train_rows:
        mol = _expanded_mol(row)
        train_atoms.update(int(atom.GetAtomicNum()) for atom in mol.GetAtoms())
        train_charges.update(int(atom.GetFormalCharge()) for atom in mol.GetAtoms())
        train_aromatic.update(bool(atom.GetIsAromatic()) for atom in mol.GetAtoms())
        train_bonds.update(_bond_name(bond) for bond in mol.GetBonds())
    if not REQUIRED_PROJECT_ATOMIC_NUMBERS.issubset(train_atoms):
        missing = sorted(REQUIRED_PROJECT_ATOMIC_NUMBERS - train_atoms)
        raise ValueError(f"Strict train is missing required atom categories: {missing}")
    unsupported = train_atoms - set(OFFICIAL_MUTAGENICITY_ATOMIC_NUMBERS)
    if unsupported:
        raise ValueError(
            "Strict train contains atom labels outside the official active "
            f"Mutagenicity edit space: {sorted(unsupported)}"
        )
    feature_atoms = tuple(
        atomic_num
        for atomic_num in OFFICIAL_MUTAGENICITY_ATOMIC_NUMBERS
        if atomic_num in train_atoms
    )
    if feature_atoms != OFFICIAL_MUTAGENICITY_ATOMIC_NUMBERS:
        raise ValueError(
            "Strict train must materialize the complete official ten-label "
            f"Mutagenicity vocabulary; found={feature_atoms}."
        )
    val_atoms: set[int] = set()
    val_charges: set[int] = set()
    val_aromatic: set[bool] = set()
    val_bonds: set[str] = set()
    for row in val_rows:
        mol = _expanded_mol(row)
        val_atoms.update(int(atom.GetAtomicNum()) for atom in mol.GetAtoms())
        val_charges.update(int(atom.GetFormalCharge()) for atom in mol.GetAtoms())
        val_aromatic.update(bool(atom.GetIsAromatic()) for atom in mol.GetAtoms())
        val_bonds.update(_bond_name(bond) for bond in mol.GetBonds())
    unseen = {
        "atoms": sorted(val_atoms - train_atoms),
        "formal_charges": sorted(val_charges - train_charges),
        "aromaticity": sorted(val_aromatic - train_aromatic),
        "bond_types": sorted(val_bonds - train_bonds),
    }
    if any(unseen.values()):
        raise ValueError(f"Validation contains train-unseen categories: {unseen}")
    max_nodes = max(
        _expanded_mol(row).GetNumAtoms() for row in (*train_rows, *val_rows)
    )
    atom_vocab = tuple(
        value for value in OFFICIAL_MUTAGENICITY_ATOMIC_NUMBERS if value in train_atoms
    )
    return MutagenicityGraphSchema(
        atom_vocabulary=atom_vocab,
        feature_atomic_numbers=feature_atoms,
        formal_charge_vocabulary=tuple(sorted(train_charges)),
        aromaticity_vocabulary=tuple(sorted(train_aromatic)),
        bond_type_vocabulary=tuple(
            value for value in SUPPORTED_BOND_TYPES if value in train_bonds
        ),
        max_num_nodes=int(max_nodes),
    )


def _one_hot(index: int, size: int) -> list[float]:
    values = [0.0] * size
    values[int(index)] = 1.0
    return values


def _source_graph_payload_for_hash(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "x": record["x"],
        "edge_index": record["edge_index"],
        "atoms": record["atom_sidecar"],
        "bonds": record["bond_sidecar"],
        "canonical_smiles": record["canonical_smiles"],
    }


def encode_source_graph(
    row: StrictMolecule,
    schema: MutagenicityGraphSchema,
) -> dict[str, Any]:
    source_mol = Chem.MolFromSmiles(row.canonical_smiles)
    if source_mol is None:
        raise GCFExplainerMutagenicityCodecError(row.molecule_id)
    Chem.SanitizeMol(source_mol)
    original_atom_count = int(source_mol.GetNumAtoms())
    mol = Chem.AddHs(source_mol, addCoords=False)
    Chem.SanitizeMol(mol)
    feature_index = {
        atomic_num: index
        for index, atomic_num in enumerate(schema.feature_atomic_numbers)
    }
    x: list[list[float]] = []
    atoms: list[dict[str, Any]] = []
    for atom in mol.GetAtoms():
        atomic_num = int(atom.GetAtomicNum())
        if atomic_num not in feature_index:
            raise GCFExplainerMutagenicityCodecError(
                f"{row.molecule_id} atom {atomic_num} is outside frozen vocabulary."
            )
        graph_index = int(atom.GetIdx())
        is_original = graph_index < original_atom_count
        attached_original = None
        if not is_original and atom.GetDegree() == 1:
            neighbor = atom.GetNeighbors()[0]
            if int(neighbor.GetIdx()) < original_atom_count:
                attached_original = int(neighbor.GetIdx())
        x.append(_one_hot(feature_index[atomic_num], schema.node_feature_dim))
        atoms.append(
            {
                "graph_node_index": graph_index,
                "original_atom_index": graph_index if is_original else -1,
                "attached_original_atom_index": attached_original,
                "atomic_num": atomic_num,
                "formal_charge": int(atom.GetFormalCharge()),
                "is_aromatic": bool(atom.GetIsAromatic()),
                "num_explicit_hs": int(atom.GetNumExplicitHs()),
                "no_implicit": bool(atom.GetNoImplicit()),
                "chiral_tag": int(atom.GetChiralTag()),
                "isotope": int(atom.GetIsotope()),
            }
        )
    directed_edges: list[tuple[int, int]] = []
    bonds: list[dict[str, Any]] = []
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        directed_edges.extend(((begin, end), (end, begin)))
        bonds.append(
            {
                "begin": min(begin, end),
                "end": max(begin, end),
                "rdkit_begin": begin,
                "rdkit_end": end,
                "source_bond_index": int(bond.GetIdx()),
                "bond_type": _bond_name(bond),
                "is_aromatic": bool(bond.GetIsAromatic()),
                "is_conjugated": bool(bond.GetIsConjugated()),
                "stereo": int(bond.GetStereo()),
                "bond_dir": int(bond.GetBondDir()),
                "stereo_atoms": [int(value) for value in bond.GetStereoAtoms()],
            }
        )
    directed_edges.sort()
    bonds.sort(key=lambda item: (int(item["begin"]), int(item["end"])))
    record: dict[str, Any] = {
        "molecule_id": row.molecule_id,
        "canonical_smiles": row.canonical_smiles,
        "original_smiles": row.smiles,
        "split": row.split,
        "label": int(row.label),
        "gnn_label": 0 if int(row.label) == SOURCE_LABEL else 1,
        "source_label": SOURCE_LABEL,
        "target_label": TARGET_LABEL,
        "semantic_label": row.semantic_label,
        "source_row_index": int(row.source_row_index),
        "source_path": row.source_path,
        "num_nodes": int(mol.GetNumAtoms()),
        "num_heavy_atoms": int(source_mol.GetNumHeavyAtoms()),
        "num_edges": int(mol.GetNumBonds()),
        "x": x,
        "edge_index": [
            [edge[0] for edge in directed_edges],
            [edge[1] for edge in directed_edges],
        ],
        "atom_sidecar": atoms,
        "bond_sidecar": bonds,
        "atom_sidecar_schema_version": schema.atom_sidecar_schema_version,
        "explicit_h_nodes": True,
    }
    record["source_graph_hash"] = stable_json_sha256(
        _source_graph_payload_for_hash(record)
    )
    return record


def _bond_type_from_name(name: str) -> Any:
    mapping = {
        "SINGLE": Chem.BondType.SINGLE,
        "DOUBLE": Chem.BondType.DOUBLE,
        "TRIPLE": Chem.BondType.TRIPLE,
        "AROMATIC": Chem.BondType.AROMATIC,
    }
    if name not in mapping:
        raise GCFExplainerMutagenicityCodecError(
            f"Unsupported bond sidecar value: {name!r}."
        )
    return mapping[name]


def _decode_one_hot(row: Sequence[Any], schema: MutagenicityGraphSchema) -> int:
    if len(row) != schema.node_feature_dim:
        raise GCFExplainerMutagenicityCodecError(
            f"Node feature dimension {len(row)} != {schema.node_feature_dim}."
        )
    values = [float(value) for value in row]
    active = [index for index, value in enumerate(values) if math.isclose(value, 1.0, abs_tol=1e-8)]
    inactive_ok = all(
        math.isclose(value, 0.0, abs_tol=1e-8) or index in active
        for index, value in enumerate(values)
    )
    if len(active) != 1 or not inactive_ok:
        raise GCFExplainerMutagenicityCodecError(
            "generated_atom_feature_out_of_vocabulary"
        )
    return int(schema.feature_atomic_numbers[active[0]])


def reconstruct_source_graph(
    record: Mapping[str, Any],
    schema: MutagenicityGraphSchema,
) -> tuple[Any, dict[str, Any]]:
    x = list(record["x"])
    atoms = list(record["atom_sidecar"])
    bonds = list(record["bond_sidecar"])
    if len(x) != len(atoms) or len(x) != int(record["num_nodes"]):
        raise GCFExplainerMutagenicityCodecError(
            "Source graph node/sidecar count mismatch."
        )
    rw_mol = Chem.RWMol()
    decoded_atomic_numbers: list[int] = []
    for row, sidecar in zip(x, atoms, strict=True):
        atomic_num = _decode_one_hot(row, schema)
        if atomic_num != int(sidecar["atomic_num"]):
            raise GCFExplainerMutagenicityCodecError(
                "Source one-hot atom does not match atom sidecar."
            )
        atom = Chem.Atom(atomic_num)
        atom.SetFormalCharge(int(sidecar["formal_charge"]))
        atom.SetNumExplicitHs(int(sidecar["num_explicit_hs"]))
        atom.SetNoImplicit(bool(sidecar["no_implicit"]))
        atom.SetChiralTag(Chem.ChiralType(int(sidecar["chiral_tag"])))
        atom.SetIsotope(int(sidecar.get("isotope", 0)))
        atom.SetIsAromatic(bool(sidecar["is_aromatic"]))
        rw_mol.AddAtom(atom)
        decoded_atomic_numbers.append(atomic_num)
    reconstructed_bonds: list[tuple[Any, Mapping[str, Any]]] = []
    for sidecar in sorted(
        bonds, key=lambda item: int(item.get("source_bond_index", 0))
    ):
        begin = int(sidecar.get("rdkit_begin", sidecar["begin"]))
        end = int(sidecar.get("rdkit_end", sidecar["end"]))
        bond_type = _bond_type_from_name(str(sidecar["bond_type"]))
        rw_mol.AddBond(begin, end, bond_type)
        bond = rw_mol.GetBondBetweenAtoms(begin, end)
        if bond_type == Chem.BondType.AROMATIC:
            bond.SetIsAromatic(True)
        bond.SetIsConjugated(bool(sidecar.get("is_conjugated", False)))
        bond.SetBondDir(Chem.BondDir(int(sidecar.get("bond_dir", 0))))
        reconstructed_bonds.append((bond, sidecar))
    for bond, sidecar in reconstructed_bonds:
        stereo_atoms = [int(value) for value in sidecar.get("stereo_atoms", [])]
        if len(stereo_atoms) == 2:
            bond.SetStereoAtoms(stereo_atoms[0], stereo_atoms[1])
        bond.SetStereo(Chem.BondStereo(int(sidecar.get("stereo", 0))))
    mol = rw_mol.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol)
    actual_atoms = [
        {
            "atomic_num": int(atom.GetAtomicNum()),
            "formal_charge": int(atom.GetFormalCharge()),
            "is_aromatic": bool(atom.GetIsAromatic()),
            "num_explicit_hs": int(atom.GetNumExplicitHs()),
            "no_implicit": bool(atom.GetNoImplicit()),
            "chiral_tag": int(atom.GetChiralTag()),
        }
        for atom in mol.GetAtoms()
    ]
    expected_atoms = [
        {
            key: value
            for key, value in sidecar.items()
            if key
            in {
                "atomic_num",
                "formal_charge",
                "is_aromatic",
                "num_explicit_hs",
                "no_implicit",
                "chiral_tag",
            }
        }
        for sidecar in atoms
    ]
    actual_bonds = sorted(
        (
            min(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())),
            max(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())),
            _bond_name(bond),
            bool(bond.GetIsConjugated()),
            int(bond.GetStereo()),
            int(bond.GetBondDir()),
            tuple(int(value) for value in bond.GetStereoAtoms()),
        )
        for bond in mol.GetBonds()
    )
    expected_bonds = sorted(
        (
            int(bond["begin"]),
            int(bond["end"]),
            str(bond["bond_type"]),
            bool(bond.get("is_conjugated", False)),
            int(bond.get("stereo", 0)),
            int(bond.get("bond_dir", 0)),
            tuple(int(value) for value in bond.get("stereo_atoms", [])),
        )
        for bond in bonds
    )
    heavy = Chem.RemoveHs(mol, sanitize=True)
    canonical = Chem.MolToSmiles(heavy, canonical=True, isomericSmiles=True)
    checks = {
        "node_count_exact": mol.GetNumAtoms() == int(record["num_nodes"]),
        "atomic_numbers_exact": [item["atomic_num"] for item in actual_atoms]
        == [int(item["atomic_num"]) for item in atoms],
        "formal_charges_exact": [item["formal_charge"] for item in actual_atoms]
        == [int(item["formal_charge"]) for item in atoms],
        "aromaticity_exact": [item["is_aromatic"] for item in actual_atoms]
        == [bool(item["is_aromatic"]) for item in atoms],
        "explicit_hs_exact": [item["num_explicit_hs"] for item in actual_atoms]
        == [int(item["num_explicit_hs"]) for item in atoms],
        "no_implicit_exact": [item["no_implicit"] for item in actual_atoms]
        == [bool(item["no_implicit"]) for item in atoms],
        "chiral_tags_exact": [item["chiral_tag"] for item in actual_atoms]
        == [int(item["chiral_tag"]) for item in atoms],
        "connectivity_exact": actual_bonds == expected_bonds,
        "bond_types_exact": actual_bonds == expected_bonds,
        "bond_stereo_exact": actual_bonds == expected_bonds,
        "canonical_smiles_exact": canonical == str(record["canonical_smiles"]),
        "atom_mapping_unique": [int(item["graph_node_index"]) for item in atoms]
        == list(range(len(atoms))),
        "atom_state_exact": actual_atoms == expected_atoms,
    }
    checks["round_trip_passed"] = all(checks.values())
    if not checks["round_trip_passed"]:
        failed = sorted(key for key, value in checks.items() if not value)
        raise GCFExplainerMutagenicityCodecError(
            f"Source round-trip failed for {record['molecule_id']}: {failed}"
        )
    return mol, checks


def _record_categories(record: Mapping[str, Any]) -> set[str]:
    atoms = list(record["atom_sidecar"])
    bonds = list(record["bond_sidecar"])
    atomic_numbers = {int(atom["atomic_num"]) for atom in atoms}
    categories: set[str] = set()
    for name, number in (("cl", 17), ("br", 35), ("i", 53), ("p", 15)):
        if number in atomic_numbers:
            categories.add(name)
    if any(int(atom["formal_charge"]) != 0 for atom in atoms):
        categories.add("charged")
    if any(
        bool(atom["is_aromatic"]) and int(atom["atomic_num"]) not in {1, 6}
        for atom in atoms
    ):
        categories.add("aromatic_hetero")
    if "[nh" in str(record["original_smiles"]).lower():
        categories.add("n_h")
    bond_names = {str(bond["bond_type"]).upper() for bond in bonds}
    if "DOUBLE" in bond_names:
        categories.add("double")
    if "TRIPLE" in bond_names:
        categories.add("triple")
    if "AROMATIC" in bond_names:
        categories.add("aromatic_bond")
    return categories


PROBE_REQUIRED_CATEGORIES = (
    "cl",
    "br",
    "i",
    "p",
    "charged",
    "aromatic_hetero",
    "n_h",
    "double",
    "triple",
    "aromatic_bond",
    "near_max_nodes",
)


def select_codec_probe_records(
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int = 64,
    require_all: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    if limit <= 0:
        raise ValueError("Codec probe limit must be positive.")
    ordered = sorted((dict(row) for row in records), key=lambda row: str(row["molecule_id"]))
    if len(ordered) < limit:
        raise ValueError(f"Codec probe requires {limit} rows; found {len(ordered)}.")
    by_category: dict[str, list[dict[str, Any]]] = {
        key: [] for key in PROBE_REQUIRED_CATEGORIES
    }
    max_nodes = max(int(row["num_nodes"]) for row in ordered)
    near_threshold = max(1, int(math.floor(max_nodes * 0.9)))
    for row in ordered:
        categories = _record_categories(row)
        if int(row["num_nodes"]) >= near_threshold:
            categories.add("near_max_nodes")
        for category in categories:
            if category in by_category:
                by_category[category].append(row)
    missing = [key for key, values in by_category.items() if not values]
    if missing and require_all:
        raise ValueError(f"Codec probe cannot cover required categories: {missing}")
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    coverage: dict[str, list[str]] = {}
    for category in PROBE_REQUIRED_CATEGORIES:
        candidates = by_category[category]
        if not candidates:
            coverage[category] = []
            continue
        chosen = candidates[0]
        molecule_id = str(chosen["molecule_id"])
        coverage[category] = [molecule_id]
        if molecule_id not in selected_ids:
            selected.append(chosen)
            selected_ids.add(molecule_id)
    for row in ordered:
        if len(selected) >= limit:
            break
        molecule_id = str(row["molecule_id"])
        if molecule_id not in selected_ids:
            selected.append(row)
            selected_ids.add(molecule_id)
    selected = selected[:limit]
    for category, candidates in by_category.items():
        matching = [
            str(row["molecule_id"])
            for row in selected
            if row in candidates
        ]
        coverage[category] = matching
    return selected, coverage


def run_codec_probe(
    records: Sequence[Mapping[str, Any]],
    schema: MutagenicityGraphSchema,
    *,
    output_dir: str | Path,
    limit: int = 64,
    require_all: bool = True,
) -> dict[str, Any]:
    selected, coverage = select_codec_probe_records(
        records, limit=limit, require_all=require_all
    )
    rows: list[dict[str, Any]] = []
    for record in selected:
        base = {
            "molecule_id": record["molecule_id"],
            "canonical_smiles": record["canonical_smiles"],
            "split": record["split"],
            "num_nodes": int(record["num_nodes"]),
            "source_graph_hash": record["source_graph_hash"],
            "categories": sorted(_record_categories(record)),
        }
        try:
            _mol, checks = reconstruct_source_graph(record, schema)
            rows.append({**base, **checks, "failure_reason": ""})
        except Exception as exc:
            rows.append(
                {
                    **base,
                    "round_trip_passed": False,
                    "failure_reason": f"{type(exc).__name__}:{exc}",
                }
            )
    failed = [row for row in rows if row.get("round_trip_passed") is not True]
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "codec_probe_rows.jsonl", rows)
    summary = {
        "probe_rows": len(rows),
        "probe_passed_rows": len(rows) - len(failed),
        "probe_failed_rows": len(failed),
        "probe_passed": not failed,
        "required_category_coverage": coverage,
        "required_categories": list(PROBE_REQUIRED_CATEGORIES),
        "atom_sidecar_schema_version": schema.atom_sidecar_schema_version,
        "node_feature_dim": schema.node_feature_dim,
        "formal_charge_exact": not failed,
        "aromaticity_exact": not failed,
        "explicit_h_exact": not failed,
        "no_implicit_exact": not failed,
        "chirality_exact": not failed,
        "calibration_loaded": False,
        "test_loaded": False,
        "run_complete": not failed,
    }
    write_json(destination / "codec_probe_summary.json", summary)
    if failed:
        raise GCFExplainerMutagenicityCodecError(
            f"Codec probe failed for {len(failed)} source molecules."
        )
    return summary


def _torch_save(payload: Any, path: Path) -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyTorch is required to write GCF graph artifacts.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _torch_load(path: str | Path, *, map_location: str = "cpu") -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyTorch is required to read GCF graph artifacts.") from exc
    try:
        return torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location=map_location)


def prepare_mutagenicity_dataset(
    *,
    train_source_csv: str | Path,
    train_target_csv: str | Path,
    val_source_csv: str | Path,
    val_target_csv: str | Path,
    output_dir: str | Path,
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    expected = {
        "train_source": EXPECTED_TRAIN_SOURCE_ROWS,
        "train_target": EXPECTED_TRAIN_TARGET_ROWS,
        "val_source": EXPECTED_VAL_SOURCE_ROWS,
        "val_target": EXPECTED_VAL_TARGET_ROWS,
    }
    if expected_counts is not None:
        expected.update({key: int(value) for key, value in expected_counts.items()})
    train_source = load_strict_molecules(
        train_source_csv,
        expected_split="train",
        expected_label=1,
        expected_rows=expected["train_source"],
    )
    train_target = load_strict_molecules(
        train_target_csv,
        expected_split="train",
        expected_label=0,
        expected_rows=expected["train_target"],
    )
    val_source = load_strict_molecules(
        val_source_csv,
        expected_split="val",
        expected_label=1,
        expected_rows=expected["val_source"],
    )
    val_target = load_strict_molecules(
        val_target_csv,
        expected_split="val",
        expected_label=0,
        expected_rows=expected["val_target"],
    )
    train_rows = sorted((*train_source, *train_target), key=lambda row: row.molecule_id)
    val_rows = sorted((*val_source, *val_target), key=lambda row: row.molecule_id)
    all_ids = [row.molecule_id for row in (*train_rows, *val_rows)]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Strict train/validation molecule_id overlap detected.")
    train_smiles = {row.canonical_smiles for row in train_rows}
    val_smiles = {row.canonical_smiles for row in val_rows}
    overlap = sorted(train_smiles & val_smiles)
    if overlap:
        raise ValueError(
            f"Strict train/validation canonical SMILES overlap detected: {overlap[:3]}"
        )
    schema = derive_schema(train_rows, val_rows)
    train_graphs = [encode_source_graph(row, schema) for row in train_rows]
    val_graphs = [encode_source_graph(row, schema) for row in val_rows]
    generation_graphs = [encode_source_graph(row, schema) for row in train_source]
    destination = Path(output_dir).expanduser().resolve()
    if (destination / "_PHASE_A_COMPLETE.json").exists():
        raise FileExistsError(f"Completed dataset cannot be overwritten: {destination}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Dataset output is non-empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    _torch_save(train_graphs, destination / "train_graphs.pt")
    _torch_save(val_graphs, destination / "val_graphs.pt")
    _torch_save(
        generation_graphs, destination / "generation_source_graphs.pt"
    )
    manifest_rows = [
        {
            key: graph[key]
            for key in (
                "molecule_id",
                "canonical_smiles",
                "original_smiles",
                "split",
                "label",
                "gnn_label",
                "source_label",
                "target_label",
                "source_row_index",
                "num_nodes",
                "num_edges",
                "source_graph_hash",
                "atom_sidecar_schema_version",
            )
        }
        for graph in (*train_graphs, *val_graphs)
    ]
    write_jsonl(destination / "source_graph_manifest.jsonl", manifest_rows)
    write_json(
        destination / "official_aids_route_audit.json",
        OFFICIAL_AIDS_ROUTE_AUDIT,
    )
    probe_summary = run_codec_probe(
        (*train_graphs, *val_graphs),
        schema,
        output_dir=destination / "codec_probe_64",
        limit=64,
        require_all=True,
    )
    input_paths = {
        "train_source": str(Path(train_source_csv).expanduser().resolve()),
        "train_target": str(Path(train_target_csv).expanduser().resolve()),
        "val_source": str(Path(val_source_csv).expanduser().resolve()),
        "val_target": str(Path(val_target_csv).expanduser().resolve()),
    }
    summary = {
        "dataset": "Mutagenicity",
        "adapter": "official_gcfexplainer_strict_project_data",
        "train_rows": len(train_graphs),
        "train_source_rows": len(train_source),
        "train_target_rows": len(train_target),
        "val_rows": len(val_graphs),
        "val_source_rows": len(val_source),
        "val_target_rows": len(val_target),
        "generation_source_rows": len(generation_graphs),
        "train_label_counts": dict(Counter(int(row["label"]) for row in train_graphs)),
        "val_label_counts": dict(Counter(int(row["label"]) for row in val_graphs)),
        "gnn_label_mapping": {"project_1": 0, "project_0": 1},
        **schema.to_dict(),
        "train_ids_hash": cohort_hash(train_graphs),
        "val_ids_hash": cohort_hash(val_graphs),
        "generation_source_cohort_hash": cohort_hash(generation_graphs),
        "input_paths": input_paths,
        "input_sha256": {
            key: sha256_file(path) for key, path in input_paths.items()
        },
        "probe": probe_summary,
        "official_aids_route_audit": "official_aids_route_audit.json",
        "calibration_loaded": False,
        "test_loaded": False,
        "run_complete": True,
    }
    write_json(destination / "dataset_summary.json", summary)
    artifact_paths = (
        "train_graphs.pt",
        "val_graphs.pt",
        "generation_source_graphs.pt",
        "source_graph_manifest.jsonl",
        "official_aids_route_audit.json",
        "codec_probe_64/codec_probe_rows.jsonl",
        "codec_probe_64/codec_probe_summary.json",
        "dataset_summary.json",
    )
    run_manifest = {
        **summary,
        "artifacts": {
            relative: {
                "size": (destination / relative).stat().st_size,
                "sha256": sha256_file(destination / relative),
            }
            for relative in artifact_paths
        },
    }
    write_json(destination / "run_manifest.json", run_manifest)
    write_json(
        destination / "_PHASE_A_COMPLETE.json",
        {
            "phase_a_complete": True,
            "dataset_summary_sha256": sha256_file(
                destination / "dataset_summary.json"
            ),
            "calibration_loaded": False,
            "test_loaded": False,
        },
    )
    write_json(
        destination / "_RUN_COMPLETE.json",
        {
            "run_complete": True,
            "run_manifest_sha256": sha256_file(destination / "run_manifest.json"),
            "calibration_loaded": False,
            "test_loaded": False,
        },
    )
    return summary


def load_dataset_artifacts(
    dataset_dir: str | Path,
) -> tuple[
    MutagenicityGraphSchema,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    root = Path(dataset_dir).expanduser().resolve()
    summary = read_json(root / "dataset_summary.json")
    if summary.get("calibration_loaded") is not False or summary.get("test_loaded") is not False:
        raise ValueError("Prepared GCF dataset reports calibration/test leakage.")
    schema = MutagenicityGraphSchema.from_dict(summary)
    train = list(_torch_load(root / "train_graphs.pt"))
    val = list(_torch_load(root / "val_graphs.pt"))
    generation = list(_torch_load(root / "generation_source_graphs.pt"))
    expected_counts = {
        "train": int(summary.get("train_rows", -1)),
        "val": int(summary.get("val_rows", -1)),
        "generation": int(summary.get("generation_source_rows", -1)),
    }
    actual_counts = {
        "train": len(train),
        "val": len(val),
        "generation": len(generation),
    }
    if actual_counts != expected_counts:
        raise ValueError(
            f"Prepared graph artifact count mismatch: expected={expected_counts}, "
            f"actual={actual_counts}."
        )
    all_ids = [str(row["molecule_id"]) for row in (*train, *val)]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Prepared train/validation graph IDs overlap.")
    generation_ids = [str(row["molecule_id"]) for row in generation]
    if len(generation_ids) != len(set(generation_ids)):
        raise ValueError("Prepared generation-source graph IDs are duplicated.")
    if any(
        str(row.get("split")) != "train" or int(row.get("label", -1)) != 1
        for row in generation
    ):
        raise ValueError("Generation artifacts are not strict train-source graphs.")
    return schema, train, val, generation, summary


def deterministic_balanced_prefix(
    records: Sequence[Mapping[str, Any]], limit: int,
) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in records), key=lambda row: str(row["molecule_id"]))
    if limit <= 0 or limit >= len(ordered):
        return ordered
    by_label: dict[int, list[dict[str, Any]]] = {}
    for row in ordered:
        by_label.setdefault(int(row["gnn_label"]), []).append(row)
    selected: list[dict[str, Any]] = []
    labels = sorted(by_label)
    cursor = {label: 0 for label in labels}
    while len(selected) < limit:
        progressed = False
        for label in labels:
            index = cursor[label]
            if index < len(by_label[label]) and len(selected) < limit:
                selected.append(by_label[label][index])
                cursor[label] += 1
                progressed = True
        if not progressed:
            break
    return selected


def validate_gnn_profile(
    profile: str | RunProfile,
    *,
    epochs: int,
    train_rows: int,
    val_rows: int,
) -> RunProfile:
    resolved = RunProfile(profile)
    if resolved is RunProfile.SMOKE:
        if int(epochs) != 5:
            raise ValueError("GNN smoke requires epochs=5.")
        if train_rows <= 0 or train_rows > EXPECTED_MODEL_TRAIN_ROWS:
            raise ValueError("GNN smoke train row count is invalid.")
        if val_rows <= 0 or val_rows > EXPECTED_MODEL_VAL_ROWS:
            raise ValueError("GNN smoke validation row count is invalid.")
    else:
        if int(epochs) != 1000:
            raise ValueError("GNN full requires epochs=1000.")
        if train_rows != EXPECTED_MODEL_TRAIN_ROWS:
            raise ValueError("GNN full requires all 2885 train rows.")
        if val_rows != EXPECTED_MODEL_VAL_ROWS:
            raise ValueError("GNN full requires all 355 validation rows.")
    return resolved


def validate_vrrw_profile(
    profile: str | RunProfile,
    *,
    parent_count: int,
    max_steps: int,
    alpha: float,
    seed: int,
) -> RunProfile:
    resolved = RunProfile(profile)
    expected_parents = 64 if resolved is RunProfile.SMOKE else 1448
    expected_steps = {500, 1000} if resolved is RunProfile.SMOKE else {50000}
    if int(parent_count) != expected_parents:
        raise ValueError(
            f"VRRW {resolved.value} requires {expected_parents} parents."
        )
    if int(max_steps) not in expected_steps:
        raise ValueError(
            f"VRRW {resolved.value} max_steps must be one of "
            f"{sorted(expected_steps)}."
        )
    if not math.isclose(float(alpha), 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Mutagenicity VRRW requires alpha=1.0.")
    if int(seed) != SEED:
        raise ValueError("Mutagenicity VRRW requires seed=13.")
    return resolved


def stable_graph_candidate_id(graph: Any) -> str:
    x = _tensor_like_to_list(getattr(graph, "x"))
    edges = _unique_edge_pairs(getattr(graph, "edge_index"))
    payload = {"x": x, "edges": sorted(edges), "num_nodes": len(x)}
    return "GCFM_" + stable_json_sha256(payload)[:20].upper()


def _tensor_like_to_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _unique_edge_pairs(edge_index: Any) -> set[tuple[int, int]]:
    values = _tensor_like_to_list(edge_index)
    if len(values) != 2:
        raise GCFExplainerMutagenicityCodecError(
            "generated_invalid_active_node_mask"
        )
    pairs: set[tuple[int, int]] = set()
    for source, target in zip(values[0], values[1], strict=True):
        a, b = int(source), int(target)
        if a == b:
            continue
        pairs.add((min(a, b), max(a, b)))
    return pairs


def decode_generated_fullgraph(
    graph: Any,
    *,
    source_record: Mapping[str, Any],
    schema: MutagenicityGraphSchema,
) -> GeneratedDecodeResult:
    source_parent_id = str(source_record["molecule_id"])
    try:
        x = _tensor_like_to_list(getattr(graph, "x"))
        num_nodes = int(getattr(graph, "num_nodes", len(x)))
        if num_nodes <= 0 or len(x) != num_nodes:
            raise GCFExplainerMutagenicityCodecError(
                "generated_invalid_active_node_mask"
            )
        decoded_atomic_numbers = [_decode_one_hot(row, schema) for row in x]
        origins_value = getattr(graph, "gcf_node_origin", None)
        if origins_value is None:
            raise GCFExplainerMutagenicityCodecError(
                "generated_missing_source_lineage"
            )
        origins = [int(value) for value in _tensor_like_to_list(origins_value)]
        if len(origins) != num_nodes:
            raise GCFExplainerMutagenicityCodecError(
                "generated_invalid_active_node_mask"
            )
        pairs = _unique_edge_pairs(getattr(graph, "edge_index"))
        if any(a < 0 or b >= num_nodes for a, b in pairs):
            raise GCFExplainerMutagenicityCodecError(
                "generated_invalid_active_node_mask"
            )
        source_atoms = list(source_record["atom_sidecar"])
        source_bonds = {
            (int(item["begin"]), int(item["end"])): dict(item)
            for item in source_record["bond_sidecar"]
        }
        source_neighbors: dict[int, set[int]] = {
            index: set() for index in range(len(source_atoms))
        }
        for a, b in source_bonds:
            source_neighbors[a].add(b)
            source_neighbors[b].add(a)
        generated_neighbors: dict[int, set[int]] = {
            index: set() for index in range(num_nodes)
        }
        for a, b in pairs:
            generated_neighbors[a].add(b)
            generated_neighbors[b].add(a)

        rw_mol = Chem.RWMol()
        inherited_count = 0
        reset_count = 0
        inherit_flags: list[bool] = []
        for index, (atomic_num, origin) in enumerate(
            zip(decoded_atomic_numbers, origins, strict=True)
        ):
            source_atom = source_atoms[origin] if 0 <= origin < len(source_atoms) else None
            mapped_neighbor_origins = {
                origins[neighbor]
                for neighbor in generated_neighbors[index]
                if 0 <= origins[neighbor] < len(source_atoms)
            }
            has_new_neighbor = any(
                origins[neighbor] < 0 for neighbor in generated_neighbors[index]
            )
            all_incident_retained = all(
                origin >= 0
                and origins[neighbor] >= 0
                and (
                    min(origin, origins[neighbor]),
                    max(origin, origins[neighbor]),
                )
                in source_bonds
                for neighbor in generated_neighbors[index]
            )
            identity_unchanged = bool(
                source_atom is not None
                and atomic_num == int(source_atom["atomic_num"])
            )
            unchanged = bool(
                identity_unchanged
                and not has_new_neighbor
                and all_incident_retained
                and mapped_neighbor_origins == source_neighbors[origin]
            )
            atom = Chem.Atom(atomic_num)
            # The official edit space models atom type only.  Formal charge,
            # aromaticity, and isotope are source-anchored only while the atom
            # identity is retained.  Explicit-H/noImplicit/chirality require
            # both identity and the complete incident environment to match.
            if identity_unchanged:
                atom.SetFormalCharge(int(source_atom["formal_charge"]))
                atom.SetIsotope(int(source_atom.get("isotope", 0)))
                atom.SetIsAromatic(bool(source_atom["is_aromatic"]))
            else:
                atom.SetFormalCharge(0)
                atom.SetIsotope(0)
                atom.SetIsAromatic(False)
            if unchanged:
                atom.SetNumExplicitHs(int(source_atom["num_explicit_hs"]))
                atom.SetNoImplicit(bool(source_atom["no_implicit"]))
                atom.SetChiralTag(Chem.ChiralType(int(source_atom["chiral_tag"])))
                inherited_count += 1
            else:
                atom.SetNumExplicitHs(0)
                atom.SetNoImplicit(False)
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                reset_count += 1
            rw_mol.AddAtom(atom)
            inherit_flags.append(unchanged)

        retained_count = 0
        new_count = 0
        generated_stereo: list[tuple[Any, list[int], Mapping[str, Any]]] = []
        origin_to_generated = {
            origin: index
            for index, origin in enumerate(origins)
            if origin >= 0
        }
        def generated_bond_order(pair: tuple[int, int]) -> tuple[int, int, int]:
            origin_a, origin_b = origins[pair[0]], origins[pair[1]]
            source_key = (min(origin_a, origin_b), max(origin_a, origin_b))
            source_sidecar = source_bonds.get(source_key)
            if source_sidecar is not None:
                return (0, int(source_sidecar.get("source_bond_index", 0)), 0)
            return (1, pair[0], pair[1])

        for a, b in sorted(pairs, key=generated_bond_order):
            origin_a, origin_b = origins[a], origins[b]
            source_key = (
                min(origin_a, origin_b),
                max(origin_a, origin_b),
            )
            retained = (
                origin_a >= 0
                and origin_b >= 0
                and source_key in source_bonds
            )
            if retained:
                sidecar = source_bonds[source_key]
                bond_type = _bond_type_from_name(str(sidecar["bond_type"]))
                retained_count += 1
                if bond_type == Chem.BondType.AROMATIC and not (
                    rw_mol.GetAtomWithIdx(a).GetIsAromatic()
                    and rw_mol.GetAtomWithIdx(b).GetIsAromatic()
                ):
                    raise GCFExplainerMutagenicityCodecError(
                        "generated_aromaticity_bond_inconsistent"
                    )
            else:
                bond_type = Chem.BondType.SINGLE
                new_count += 1
            rw_mol.AddBond(a, b, bond_type)
            generated_bond = rw_mol.GetBondBetweenAtoms(a, b)
            if bond_type == Chem.BondType.AROMATIC:
                generated_bond.SetIsAromatic(True)
            if retained:
                generated_bond.SetIsConjugated(
                    bool(sidecar.get("is_conjugated", False))
                )
                stereo_origins = [
                    int(value) for value in sidecar.get("stereo_atoms", [])
                ]
                if (
                    len(stereo_origins) == 2
                    and int(sidecar.get("rdkit_begin", source_key[0]))
                    != source_key[0]
                ):
                    stereo_origins.reverse()
                stereo_can_be_inherited = bool(
                    inherit_flags[a]
                    and inherit_flags[b]
                    and len(stereo_origins) == 2
                    and all(value in origin_to_generated for value in stereo_origins)
                )
                if stereo_can_be_inherited:
                    generated_bond.SetBondDir(
                        Chem.BondDir(int(sidecar.get("bond_dir", 0)))
                    )
                    generated_stereo.append(
                        (generated_bond, stereo_origins, sidecar)
                    )
        for generated_bond, stereo_origins, sidecar in generated_stereo:
            generated_bond.SetStereoAtoms(
                origin_to_generated[stereo_origins[0]],
                origin_to_generated[stereo_origins[1]],
            )
            generated_bond.SetStereo(
                Chem.BondStereo(int(sidecar.get("stereo", 0)))
            )
        mol = rw_mol.GetMol()
        if mol.GetNumAtoms() <= 0:
            raise GCFExplainerMutagenicityCodecError(
                "generated_disconnected_or_empty"
            )
        mol.UpdatePropertyCache(strict=False)
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            message = str(exc).lower()
            if "kekul" in message:
                reason = "generated_kekulization_failed"
            elif "valence" in message:
                reason = "generated_valence_sanitize_failed"
            else:
                reason = "generated_other_sanitize_failed"
            raise GCFExplainerMutagenicityCodecError(reason) from exc
        fragments = Chem.GetMolFrags(mol)
        if len(fragments) != 1:
            raise GCFExplainerMutagenicityCodecError(
                "generated_disconnected_or_empty"
            )
        raw_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        heavy = Chem.RemoveHs(mol, sanitize=True)
        canonical = Chem.MolToSmiles(
            heavy, canonical=True, isomericSmiles=True
        )
        removed = max(0, len(source_bonds) - retained_count)
        return GeneratedDecodeResult(
            decode_ok=True,
            canonical_smiles=canonical,
            raw_smiles=raw_smiles,
            failure_reason="",
            projected_new_edge_count=new_count,
            retained_edge_count=retained_count,
            removed_source_edge_count=removed,
            inherited_atom_state_count=inherited_count,
            reset_atom_state_count=reset_count,
            source_parent_id=source_parent_id,
        )
    except GCFExplainerMutagenicityCodecError as exc:
        return GeneratedDecodeResult(
            decode_ok=False,
            canonical_smiles="",
            raw_smiles="",
            failure_reason=str(exc),
            projected_new_edge_count=0,
            retained_edge_count=0,
            removed_source_edge_count=0,
            inherited_atom_state_count=0,
            reset_atom_state_count=0,
            source_parent_id=source_parent_id,
        )
    except Exception:
        return GeneratedDecodeResult(
            decode_ok=False,
            canonical_smiles="",
            raw_smiles="",
            failure_reason="generated_other_sanitize_failed",
            projected_new_edge_count=0,
            retained_edge_count=0,
            removed_source_edge_count=0,
            inherited_atom_state_count=0,
            reset_atom_state_count=0,
            source_parent_id=source_parent_id,
        )


class GraphRecordDataset:
    """List-like adapter matching the official dataset indexing contract."""

    def __init__(self, graphs: Sequence[Any], num_features: int) -> None:
        self.graphs = list(graphs)
        self.num_features = int(num_features)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return self.graphs[index]
        if isinstance(index, (list, tuple)):
            return [self.graphs[int(value)] for value in index]
        if hasattr(index, "detach"):
            values = index.detach().cpu().tolist()
            if isinstance(values, list):
                return [self.graphs[int(value)] for value in values]
            index = values
        return self.graphs[int(index)]


def record_to_pyg(record: Mapping[str, Any], *, origin_index: int | None = None) -> Any:
    try:
        import torch
        from torch_geometric.data import Data
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "GCFExplainer runtime requires torch and torch_geometric."
        ) from exc
    data = Data(
        x=torch.tensor(record["x"], dtype=torch.float32),
        edge_index=torch.tensor(record["edge_index"], dtype=torch.long),
        y=torch.tensor(int(record["gnn_label"]), dtype=torch.long),
        num_nodes=int(record["num_nodes"]),
    )
    data.gcf_node_origin = torch.arange(int(record["num_nodes"]), dtype=torch.long)
    data.gcf_origin_index = torch.tensor(
        [int(origin_index if origin_index is not None else -1)],
        dtype=torch.long,
    )
    return data


@contextmanager
def official_python_path(official_root: str | Path) -> Iterator[Path]:
    root = Path(official_root).expanduser().resolve()
    required = ("data.py", "gnn.py", "vrrw.py", "summary.py", "importance.py", "distance.py")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Official GCFExplainer files missing: {missing}")
    sys.path.insert(0, str(root))
    try:
        yield root
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass


def import_official_modules(official_root: str | Path) -> dict[str, Any]:
    with official_python_path(official_root):
        modules = {
            name: importlib.import_module(name)
            for name in ("gnn", "vrrw", "summary", "importance", "distance")
        }
    return modules


def graph_lineage_neighbor_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap official edit-map output with lineage metadata only.

    The returned graph tensors are exactly those produced by official
    ``neighbor_graph_access``.  Only project-side origin indices are updated;
    transition probabilities and the edit neighborhood are untouched.
    """

    def wrapped(graph: Any, action: tuple[Any, ...]) -> Any:
        candidate = original(graph, action)
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PyTorch is required for lineage tracking.") from exc
        origins = getattr(graph, "gcf_node_origin", None)
        origin_index = getattr(graph, "gcf_origin_index", None)
        if origins is None or origin_index is None:
            raise GCFExplainerMutagenicityError(
                "Official edit graph is missing project lineage metadata."
            )
        origins = origins.detach().clone()
        operation = str(action[0])
        if operation in {"NA", "INA"}:
            origins = torch.cat([origins, torch.tensor([-1], dtype=origins.dtype)])
        elif operation in {"NR", "INR"}:
            removed = int(action[1])
            origins = torch.cat((origins[:removed], origins[removed + 1 :]))
        candidate.gcf_node_origin = origins
        candidate.gcf_origin_index = origin_index.detach().clone()
        if len(origins) != int(candidate.num_nodes):
            raise GCFExplainerMutagenicityError(
                "Lineage metadata no longer matches official graph node count."
            )
        return candidate

    return wrapped


def checkpoint_is_aids(path: str | Path) -> bool:
    parts = {part.lower() for part in Path(path).expanduser().parts}
    return "aids" in parts or "hiv" in parts


def score_teacher_probabilities(
    teacher: TeacherProtocol, smiles: str
) -> tuple[int, float, float]:
    result0 = teacher.score_smiles(smiles, label=0)
    result1 = teacher.score_smiles(smiles, label=1)
    if not result0.get("teacher_result_ok") or not result1.get("teacher_result_ok"):
        raise GCFExplainerMutagenicityError(
            f"RF teacher could not score candidate: {smiles}"
        )
    pred = int(result0["teacher_label"])
    return pred, float(result0["teacher_prob"]), float(result1["teacher_prob"])


def build_native_rank_candidate_universe(
    native_rows: Sequence[Mapping[str, Any]],
    graphs: Sequence[Any],
    source_records: Sequence[Mapping[str, Any]],
    schema: MutagenicityGraphSchema,
    teacher: TeacherProtocol,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(native_rows) != len(graphs):
        raise ValueError("Native rank rows and graph payload length differ.")
    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_smiles: set[str] = set()
    ordered = sorted(
        zip(native_rows, graphs, strict=True),
        key=lambda item: int(item[0]["native_rank"]),
    )
    for native, graph in ordered:
        origin_value = getattr(graph, "gcf_origin_index", None)
        if origin_value is None:
            skipped.append({**dict(native), "skip_reason": "missing_source_lineage"})
            continue
        origin_list = _tensor_like_to_list(origin_value)
        origin_index = int(origin_list[0] if isinstance(origin_list, list) else origin_list)
        if origin_index < 0 or origin_index >= len(source_records):
            skipped.append({**dict(native), "skip_reason": "invalid_source_lineage"})
            continue
        decoded = decode_generated_fullgraph(
            graph,
            source_record=source_records[origin_index],
            schema=schema,
        )
        base = {
            **dict(native),
            **asdict(decoded),
            "source_parent_index": origin_index,
        }
        if not decoded.decode_ok:
            skipped.append({**base, "skip_reason": decoded.failure_reason})
            continue
        if decoded.canonical_smiles in seen_smiles:
            skipped.append({**base, "skip_reason": "canonical_duplicate"})
            continue
        pred, prob0, prob1 = score_teacher_probabilities(
            teacher, decoded.canonical_smiles
        )
        candidate_id = "GCFMOL_" + hashlib.sha256(
            decoded.canonical_smiles.encode("utf-8")
        ).hexdigest()[:20].upper()
        row = {
            **base,
            "candidate_id": candidate_id,
            "smiles": decoded.canonical_smiles,
            "canonical_smiles": decoded.canonical_smiles,
            "rdkit_valid": True,
            "rf_pred": pred,
            "rf_prob_0": prob0,
            "rf_prob_1": prob1,
            "source_method": "official_gcfexplainer_mutagenicity",
            "selection_method": "native_gcf_summary_rank_filtered_by_validity",
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
            "calibration_loaded": False,
            "test_loaded": False,
        }
        if pred != TARGET_LABEL:
            skipped.append({**row, "skip_reason": "rf_not_target_label_0"})
            continue
        seen_smiles.add(decoded.canonical_smiles)
        eligible.append(row)
    return eligible, skipped


def filter_native_rank_candidates(
    native_rows: Sequence[Mapping[str, Any]],
    graphs: Sequence[Any],
    source_records: Sequence[Mapping[str, Any]],
    schema: MutagenicityGraphSchema,
    teacher: TeacherProtocol,
    *,
    top_k: int = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible, skipped = build_native_rank_candidate_universe(
        native_rows,
        graphs,
        source_records,
        schema,
        teacher,
    )
    return eligible[: int(top_k)], skipped


def write_failure_artifacts(
    output_dir: str | Path,
    *,
    error: BaseException,
    resolved_config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    destination = Path(output_dir).expanduser().resolve()
    if (destination / "_FINALIZED.json").exists() or (
        destination / "_RUN_COMPLETE.json"
    ).exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    if resolved_config is not None and not (destination / "resolved_config.json").exists():
        write_json(destination / "resolved_config.json", resolved_config)
    payload = {
        "run_complete": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "calibration_loaded": False,
        "test_loaded": False,
        **dict(extra or {}),
    }
    if not (destination / "failure_summary.json").exists():
        write_json(destination / "failure_summary.json", payload)
    if not (destination / "run_manifest.json").exists():
        write_json(destination / "run_manifest.json", payload)
    if not (destination / "_RUN_FAILED.json").exists():
        write_json(destination / "_RUN_FAILED.json", payload)


__all__ = [
    "ATOM_SIDECAR_SCHEMA_VERSION",
    "EXPECTED_GENERATION_SOURCE_ROWS",
    "EXPECTED_MODEL_TRAIN_ROWS",
    "EXPECTED_MODEL_VAL_ROWS",
    "GENERATED_CODEC_VERSION",
    "GCFExplainerEmptyCandidateSetError",
    "GCFExplainerMutagenicityCodecError",
    "GCFExplainerMutagenicityError",
    "GraphRecordDataset",
    "MutagenicityGraphSchema",
    "OFFICIAL_AIDS_ROUTE_AUDIT",
    "OFFICIAL_MUTAGENICITY_ATOMIC_NUMBERS",
    "RunProfile",
    "SEED",
    "SOURCE_LABEL",
    "TARGET_LABEL",
    "checkpoint_is_aids",
    "build_native_rank_candidate_universe",
    "cohort_hash",
    "decode_generated_fullgraph",
    "derive_schema",
    "deterministic_balanced_prefix",
    "encode_source_graph",
    "filter_native_rank_candidates",
    "graph_lineage_neighbor_wrapper",
    "import_official_modules",
    "load_dataset_artifacts",
    "load_strict_molecules",
    "prepare_mutagenicity_dataset",
    "read_json",
    "read_jsonl",
    "reconstruct_source_graph",
    "record_to_pyg",
    "reject_forbidden_path",
    "run_codec_probe",
    "score_teacher_probabilities",
    "select_codec_probe_records",
    "sha256_file",
    "stable_graph_candidate_id",
    "stable_json_sha256",
    "validate_gnn_profile",
    "validate_vrrw_profile",
    "write_csv",
    "write_failure_artifacts",
    "write_json",
    "write_jsonl",
]
