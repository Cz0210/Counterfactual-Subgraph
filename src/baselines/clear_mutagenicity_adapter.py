"""Strict Mutagenicity data and chemistry codec for official CLEAR/GraphCFE.

Official CLEAR reconstructs node features and a binary adjacency matrix.  It
does not reconstruct molecular bond classes.  This adapter therefore keeps
source bond types in a sidecar and applies the following deterministic
project-side projection:

* retained source edges inherit the source bond type;
* deleted source edges are omitted;
* newly added edges receive a provisional SINGLE bond;
* RDKit sanitization and connectedness remain hard validity gates.

No atomic-number nearest-neighbour or unknown-to-carbon fallback is used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

try:  # pragma: no cover - availability depends on runtime environment
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None


DATASET_NAME = "mutagenicity"
BOND_DECODER_NATIVE = False
NATIVE_ADJACENCY_SEMANTICS = "binary_connectivity"
RETAINED_BOND_TYPE_SOURCE = "parent_sidecar"
NEW_EDGE_BOND_RULE = "provisional_single"
RDKIT_SANITIZE_REQUIRED = True
ATOM_SIDECAR_SCHEMA_VERSION = "clear_mutagenicity_atom_sidecar_v2"
DEFAULT_EXPECTED_COUNTS = {
    "train_positive": 1448,
    "train_negative": 1437,
    "val_positive": 260,
    "val_negative": 95,
}
BOND_TYPE_ORDER = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")


class ClearMutagenicityError(ValueError):
    """Base error for strict Mutagenicity/CLEAR adaptation."""


class ClearMutagenicityCodecError(ClearMutagenicityError):
    """Raised when graph chemistry cannot be decoded without guessing."""


@dataclass(frozen=True, slots=True)
class ClearMutagenicitySchema:
    atom_vocabulary: tuple[int, ...]
    formal_charge_vocabulary: tuple[int, ...]
    aromaticity_vocabulary: tuple[bool, ...]
    bond_type_vocabulary: tuple[str, ...]
    atom_feature_start: int
    atom_feature_end: int
    charge_feature_start: int
    charge_feature_end: int
    aromatic_feature_index: int
    node_present_feature_index: int
    feature_dim: int
    vocabulary_source: str = "strict_train_only"
    max_num_nodes_source: str = "strict_train_plus_validation"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["atom_vocabulary"] = list(self.atom_vocabulary)
        payload["formal_charge_vocabulary"] = list(
            self.formal_charge_vocabulary
        )
        payload["aromaticity_vocabulary"] = list(
            self.aromaticity_vocabulary
        )
        payload["bond_type_vocabulary"] = list(self.bond_type_vocabulary)
        payload.update(codec_provenance())
        payload["atomic_number_decoding"] = "strict_discrete_one_hot"
        payload["unknown_atom_policy"] = "error"
        payload["nearest_atomic_number_mapping"] = False
        return payload


@dataclass(frozen=True, slots=True)
class PreparedMolecule:
    molecule_id: str
    original_smiles: str
    canonical_smiles: str
    label: int
    split: str
    source_row_index: int
    mol: Any
    atom_categories: tuple[int, ...]
    charge_categories: tuple[int, ...]
    aromatic_categories: tuple[bool, ...]
    bond_categories: tuple[str, ...]
    explicit_h_categories: tuple[int, ...] = ()
    implicit_h_categories: tuple[int, ...] = ()
    no_implicit_categories: tuple[bool, ...] = ()
    chiral_tag_categories: tuple[int, ...] = ()


@dataclass(slots=True)
class CodecResult:
    ok: bool
    canonical_smiles: str | None
    molecule: Any | None
    error_type: str | None
    error: str | None
    num_real_nodes: int
    projected_new_edge_count: int
    retained_edge_count: int
    deleted_edge_count: int
    atom_mapping_unique: bool
    decoded_atomic_numbers: list[int]
    decoded_formal_charges: list[int]
    decoded_aromaticity: list[bool]
    decoded_bond_types: list[str]
    decoded_num_explicit_hs: list[int] = field(default_factory=list)
    decoded_no_implicit: list[bool] = field(default_factory=list)
    decoded_chiral_tags: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("molecule", None)
        return payload


def codec_provenance() -> dict[str, Any]:
    return {
        "bond_decoder_native": BOND_DECODER_NATIVE,
        "native_adjacency_semantics": NATIVE_ADJACENCY_SEMANTICS,
        "retained_bond_type_source": RETAINED_BOND_TYPE_SOURCE,
        "new_edge_bond_rule": NEW_EDGE_BOND_RULE,
        "rdkit_sanitize_required": RDKIT_SANITIZE_REQUIRED,
        "atom_sidecar_schema_version": ATOM_SIDECAR_SCHEMA_VERSION,
        "stores_num_explicit_hs": True,
        "stores_no_implicit": True,
        "stores_chiral_tag": True,
        "explicit_h_decoder_native": False,
    }


def _require_rdkit() -> Any:
    if Chem is None:
        raise RuntimeError("RDKit is required for CLEAR Mutagenicity adaptation.")
    return Chem


def _text(value: Any) -> str:
    return str(value or "").strip()


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _text(value).lower() in {"1", "true", "yes", "y", "on"}


def _forbid_non_train_val_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    # Inspect the input filename, not arbitrary ancestor directories such as a
    # pytest case named ``test_source_*``.
    lowered = resolved.name.lower()
    forbidden = (
        "calibration_source",
        "calibration_target",
        "test_source",
        "test_target",
    )
    if any(token in lowered for token in forbidden):
        raise ClearMutagenicityError(
            f"Calibration/test input is forbidden in CLEAR Phase A: {resolved}"
        )
    return resolved


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    source = _forbid_non_train_val_path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _bond_type_name(bond: Any) -> str:
    name = str(bond.GetBondType()).upper()
    if name not in BOND_TYPE_ORDER:
        raise ClearMutagenicityCodecError(f"Unsupported source bond type: {name}")
    return name


def _canonical_smiles(mol: Any) -> str:
    return str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))


def load_strict_cohort(
    path: str | Path,
    *,
    expected_split: str,
    expected_label: int,
    expected_rows: int | None,
) -> list[PreparedMolecule]:
    chem = _require_rdkit()
    rows = _read_csv(path)
    if expected_rows is not None and len(rows) != int(expected_rows):
        raise ClearMutagenicityError(
            f"{Path(path).name} row count mismatch: "
            f"expected={expected_rows}, found={len(rows)}"
        )
    prepared: list[PreparedMolecule] = []
    seen_ids: set[str] = set()
    for source_row_index, row in enumerate(rows):
        molecule_id = _text(row.get("molecule_id"))
        smiles = _text(row.get("smiles"))
        split = _text(row.get("split"))
        label = _as_int(row.get("label"))
        teacher_pred = _as_int(row.get("teacher_pred"))
        teacher_correct = _as_bool(row.get("teacher_correct"))
        if not molecule_id or molecule_id in seen_ids:
            raise ClearMutagenicityError(
                f"Missing/duplicate molecule_id in {path}: {molecule_id!r}"
            )
        if split != expected_split:
            raise ClearMutagenicityError(
                f"Unexpected split for {molecule_id}: {split!r} != "
                f"{expected_split!r}"
            )
        if label != int(expected_label) or teacher_pred != int(expected_label):
            raise ClearMutagenicityError(
                f"Label/teacher mismatch for {molecule_id}: "
                f"label={label}, teacher_pred={teacher_pred}, "
                f"expected={expected_label}"
            )
        if not teacher_correct:
            raise ClearMutagenicityError(
                f"teacher_correct is false for {molecule_id}"
            )
        mol = chem.MolFromSmiles(smiles)
        if mol is None:
            raise ClearMutagenicityCodecError(
                f"RDKit could not parse {molecule_id}: {smiles!r}"
            )
        try:
            chem.SanitizeMol(mol)
        except Exception as exc:
            raise ClearMutagenicityCodecError(
                f"RDKit sanitization failed for {molecule_id}: {smiles!r}"
            ) from exc
        if mol.GetNumAtoms() <= 0:
            raise ClearMutagenicityCodecError(
                f"Empty molecule in strict cohort: {molecule_id}"
            )
        canonical = _canonical_smiles(mol)
        seen_ids.add(molecule_id)
        prepared.append(
            PreparedMolecule(
                molecule_id=molecule_id,
                original_smiles=smiles,
                canonical_smiles=canonical,
                label=int(label),
                split=split,
                source_row_index=source_row_index,
                mol=mol,
                atom_categories=tuple(
                    int(atom.GetAtomicNum()) for atom in mol.GetAtoms()
                ),
                charge_categories=tuple(
                    int(atom.GetFormalCharge()) for atom in mol.GetAtoms()
                ),
                aromatic_categories=tuple(
                    bool(atom.GetIsAromatic()) for atom in mol.GetAtoms()
                ),
                bond_categories=tuple(
                    _bond_type_name(bond) for bond in mol.GetBonds()
                ),
                explicit_h_categories=tuple(
                    int(atom.GetNumExplicitHs()) for atom in mol.GetAtoms()
                ),
                implicit_h_categories=tuple(
                    int(atom.GetNumImplicitHs()) for atom in mol.GetAtoms()
                ),
                no_implicit_categories=tuple(
                    bool(atom.GetNoImplicit()) for atom in mol.GetAtoms()
                ),
                chiral_tag_categories=tuple(
                    int(atom.GetChiralTag()) for atom in mol.GetAtoms()
                ),
            )
        )
    return sorted(prepared, key=lambda item: item.molecule_id)


def load_phase_a_cohorts(
    *,
    train_positive_csv: str | Path,
    train_negative_csv: str | Path,
    val_positive_csv: str | Path,
    val_negative_csv: str | Path,
    expected_counts: dict[str, int | None] | None = None,
) -> dict[str, list[PreparedMolecule]]:
    counts = {**DEFAULT_EXPECTED_COUNTS, **(expected_counts or {})}
    cohorts = {
        "train_positive": load_strict_cohort(
            train_positive_csv,
            expected_split="train",
            expected_label=1,
            expected_rows=counts.get("train_positive"),
        ),
        "train_negative": load_strict_cohort(
            train_negative_csv,
            expected_split="train",
            expected_label=0,
            expected_rows=counts.get("train_negative"),
        ),
        "val_positive": load_strict_cohort(
            val_positive_csv,
            expected_split="val",
            expected_label=1,
            expected_rows=counts.get("val_positive"),
        ),
        "val_negative": load_strict_cohort(
            val_negative_csv,
            expected_split="val",
            expected_label=0,
            expected_rows=counts.get("val_negative"),
        ),
    }
    all_rows = [item for values in cohorts.values() for item in values]
    all_ids = [item.molecule_id for item in all_rows]
    if len(set(all_ids)) != len(all_ids):
        raise ClearMutagenicityError("molecule_id overlaps across train/validation.")
    train_smiles = {
        item.canonical_smiles
        for key in ("train_positive", "train_negative")
        for item in cohorts[key]
    }
    val_smiles = {
        item.canonical_smiles
        for key in ("val_positive", "val_negative")
        for item in cohorts[key]
    }
    overlap = sorted(train_smiles & val_smiles)
    if overlap:
        raise ClearMutagenicityError(
            f"Canonical SMILES overlap across train/validation: {overlap[:5]}"
        )
    return cohorts


def build_train_schema(
    train_rows: Sequence[PreparedMolecule],
) -> ClearMutagenicitySchema:
    if not train_rows:
        raise ClearMutagenicityError("Cannot build schema from empty train data.")
    atoms = tuple(
        sorted({value for row in train_rows for value in row.atom_categories})
    )
    charges = tuple(
        sorted({value for row in train_rows for value in row.charge_categories})
    )
    aromaticity = tuple(
        sorted({value for row in train_rows for value in row.aromatic_categories})
    )
    bonds_present = {value for row in train_rows for value in row.bond_categories}
    bonds = tuple(value for value in BOND_TYPE_ORDER if value in bonds_present)
    atom_start = 0
    atom_end = atom_start + len(atoms)
    charge_start = atom_end
    charge_end = charge_start + len(charges)
    aromatic_index = charge_end
    present_index = aromatic_index + 1
    return ClearMutagenicitySchema(
        atom_vocabulary=atoms,
        formal_charge_vocabulary=charges,
        aromaticity_vocabulary=aromaticity,
        bond_type_vocabulary=bonds,
        atom_feature_start=atom_start,
        atom_feature_end=atom_end,
        charge_feature_start=charge_start,
        charge_feature_end=charge_end,
        aromatic_feature_index=aromatic_index,
        node_present_feature_index=present_index,
        feature_dim=present_index + 1,
    )


def validate_validation_schema(
    val_rows: Sequence[PreparedMolecule],
    schema: ClearMutagenicitySchema,
) -> None:
    atom_vocab = set(schema.atom_vocabulary)
    charge_vocab = set(schema.formal_charge_vocabulary)
    aromatic_vocab = set(schema.aromaticity_vocabulary)
    bond_vocab = set(schema.bond_type_vocabulary)
    for row in val_rows:
        unseen_atoms = sorted(set(row.atom_categories) - atom_vocab)
        unseen_charges = sorted(set(row.charge_categories) - charge_vocab)
        unseen_aromatic = sorted(set(row.aromatic_categories) - aromatic_vocab)
        unseen_bonds = sorted(set(row.bond_categories) - bond_vocab)
        if unseen_atoms or unseen_charges or unseen_aromatic or unseen_bonds:
            raise ClearMutagenicityCodecError(
                f"Validation molecule {row.molecule_id} contains train-unseen "
                f"categories: atoms={unseen_atoms}, charges={unseen_charges}, "
                f"aromaticity={unseen_aromatic}, bonds={unseen_bonds}"
            )


def encode_atom_feature(
    *,
    atomic_num: int,
    formal_charge: int,
    is_aromatic: bool,
    schema: ClearMutagenicitySchema,
) -> np.ndarray:
    if int(atomic_num) not in schema.atom_vocabulary:
        raise ClearMutagenicityCodecError(
            f"Unknown atom category {atomic_num}; no fallback is permitted."
        )
    if int(formal_charge) not in schema.formal_charge_vocabulary:
        raise ClearMutagenicityCodecError(
            f"Unknown formal-charge category {formal_charge}."
        )
    if bool(is_aromatic) not in schema.aromaticity_vocabulary:
        raise ClearMutagenicityCodecError(
            f"Unknown aromaticity category {is_aromatic}."
        )
    feature = np.zeros(schema.feature_dim, dtype=np.float64)
    feature[
        schema.atom_feature_start
        + schema.atom_vocabulary.index(int(atomic_num))
    ] = 1.0
    feature[
        schema.charge_feature_start
        + schema.formal_charge_vocabulary.index(int(formal_charge))
    ] = 1.0
    feature[schema.aromatic_feature_index] = float(bool(is_aromatic))
    feature[schema.node_present_feature_index] = 1.0
    return feature


def _strict_one_hot_index(values: np.ndarray, *, field: str) -> int:
    if values.ndim != 1 or values.size == 0:
        raise ClearMutagenicityCodecError(f"Malformed {field} feature block.")
    maximum = float(np.max(values))
    winners = np.flatnonzero(np.isclose(values, maximum, rtol=0.0, atol=1e-12))
    if maximum <= 0.0 or len(winners) != 1:
        raise ClearMutagenicityCodecError(
            f"Ambiguous {field} category: values={values.tolist()}"
        )
    return int(winners[0])


def decode_atom_feature(
    feature: Sequence[float],
    *,
    schema: ClearMutagenicitySchema,
    present_threshold: float = 0.5,
) -> dict[str, Any] | None:
    values = np.asarray(feature, dtype=np.float64).reshape(-1)
    if values.size != schema.feature_dim:
        raise ClearMutagenicityCodecError(
            f"Feature dimension mismatch: {values.size} != {schema.feature_dim}"
        )
    if not np.all(np.isfinite(values)):
        raise ClearMutagenicityCodecError("Node feature contains non-finite values.")
    if values[schema.node_present_feature_index] <= float(present_threshold):
        return None
    atom_index = _strict_one_hot_index(
        values[schema.atom_feature_start : schema.atom_feature_end],
        field="atom",
    )
    charge_index = _strict_one_hot_index(
        values[schema.charge_feature_start : schema.charge_feature_end],
        field="formal_charge",
    )
    return {
        "atomic_num": int(schema.atom_vocabulary[atom_index]),
        "formal_charge": int(schema.formal_charge_vocabulary[charge_index]),
        "is_aromatic": bool(values[schema.aromatic_feature_index] > 0.5),
    }


def molecule_to_clear_graph(
    row: PreparedMolecule,
    *,
    schema: ClearMutagenicitySchema,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    num_nodes = int(row.mol.GetNumAtoms())
    adjacency = np.eye(num_nodes, dtype=np.float64)
    features = np.zeros((num_nodes, schema.feature_dim), dtype=np.float64)
    atoms: list[dict[str, Any]] = []
    bonds: list[dict[str, Any]] = []
    for atom in row.mol.GetAtoms():
        index = int(atom.GetIdx())
        features[index] = encode_atom_feature(
            atomic_num=int(atom.GetAtomicNum()),
            formal_charge=int(atom.GetFormalCharge()),
            is_aromatic=bool(atom.GetIsAromatic()),
            schema=schema,
        )
        atoms.append(
            {
                "original_atom_index": index,
                "atomic_num": int(atom.GetAtomicNum()),
                "formal_charge": int(atom.GetFormalCharge()),
                "is_aromatic": bool(atom.GetIsAromatic()),
                "num_explicit_hs": int(atom.GetNumExplicitHs()),
                "num_implicit_hs": int(atom.GetNumImplicitHs()),
                "no_implicit": bool(atom.GetNoImplicit()),
                "chiral_tag": int(atom.GetChiralTag()),
            }
        )
    for bond in row.mol.GetBonds():
        left = int(bond.GetBeginAtomIdx())
        right = int(bond.GetEndAtomIdx())
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
        bonds.append(
            {
                "begin_atom_index": min(left, right),
                "end_atom_index": max(left, right),
                "bond_type": _bond_type_name(bond),
                "is_aromatic": bool(bond.GetIsAromatic()),
            }
        )
    sidecar = {
        "molecule_id": row.molecule_id,
        "canonical_smiles": row.canonical_smiles,
        "original_smiles": row.original_smiles,
        "source_split": row.split,
        "source_row_index": int(row.source_row_index),
        "num_real_nodes": num_nodes,
        "atom_sidecar_schema_version": ATOM_SIDECAR_SCHEMA_VERSION,
        "original_atom_indices": list(range(num_nodes)),
        "atoms": atoms,
        "bonds": sorted(
            bonds,
            key=lambda item: (
                item["begin_atom_index"],
                item["end_atom_index"],
            ),
        ),
    }
    return adjacency, features, sidecar


def _rdkit_bond_type(name: str) -> Any:
    mapping = {
        "SINGLE": Chem.BondType.SINGLE,
        "DOUBLE": Chem.BondType.DOUBLE,
        "TRIPLE": Chem.BondType.TRIPLE,
        "AROMATIC": Chem.BondType.AROMATIC,
    }
    try:
        return mapping[str(name)]
    except KeyError as exc:
        raise ClearMutagenicityCodecError(
            f"Unknown bond category {name!r}."
        ) from exc


def project_binary_graph_to_molecule(
    *,
    features: Sequence[Sequence[float]],
    adjacency: Sequence[Sequence[float]],
    schema: ClearMutagenicitySchema,
    parent_sidecar: dict[str, Any],
    adjacency_threshold: float = 0.5,
    require_connected: bool = True,
    atom_attribute_mode: str = "source",
) -> CodecResult:
    """Project CLEAR binary connectivity into a sanitized RDKit molecule.

    CLEAR predicts atom/charge/aromatic features but does not predict explicit
    hydrogen or chirality state.  Source round trips restore those attributes
    exactly.  Generated nodes may inherit them only when the decoded
    atomic-number/charge/aromatic identity matches the same parent node slot.
    """

    chem = _require_rdkit()
    if atom_attribute_mode not in {"source", "generated"}:
        raise ValueError(
            "atom_attribute_mode must be 'source' or 'generated', found "
            f"{atom_attribute_mode!r}."
        )
    feature_array = np.asarray(features, dtype=np.float64)
    adjacency_array = np.asarray(adjacency, dtype=np.float64)
    if feature_array.ndim != 2 or adjacency_array.ndim != 2:
        raise ClearMutagenicityCodecError("Features/adjacency must be matrices.")
    if adjacency_array.shape[0] != adjacency_array.shape[1]:
        raise ClearMutagenicityCodecError("Adjacency must be square.")
    if feature_array.shape[0] != adjacency_array.shape[0]:
        raise ClearMutagenicityCodecError(
            "Feature and adjacency node dimensions differ."
        )
    decoded: list[tuple[int, dict[str, Any]]] = []
    try:
        for slot, feature in enumerate(feature_array):
            atom = decode_atom_feature(feature, schema=schema)
            if atom is not None:
                decoded.append((slot, atom))
    except ClearMutagenicityCodecError as exc:
        return CodecResult(
            ok=False,
            canonical_smiles=None,
            molecule=None,
            error_type="ambiguous_node_category",
            error=str(exc),
            num_real_nodes=0,
            projected_new_edge_count=0,
            retained_edge_count=0,
            deleted_edge_count=0,
            atom_mapping_unique=False,
            decoded_atomic_numbers=[],
            decoded_formal_charges=[],
            decoded_aromaticity=[],
            decoded_bond_types=[],
        )
    if not decoded:
        return CodecResult(
            ok=False,
            canonical_smiles=None,
            molecule=None,
            error_type="empty_molecule",
            error="No node-present feature survived decoding.",
            num_real_nodes=0,
            projected_new_edge_count=0,
            retained_edge_count=0,
            deleted_edge_count=0,
            atom_mapping_unique=True,
            decoded_atomic_numbers=[],
            decoded_formal_charges=[],
            decoded_aromaticity=[],
            decoded_bond_types=[],
        )
    slot_to_new = {slot: index for index, (slot, _) in enumerate(decoded)}
    atom_mapping_unique = len(slot_to_new) == len(decoded)
    source_atoms = {
        int(item["original_atom_index"]): dict(item)
        for item in parent_sidecar.get("atoms", [])
    }
    source_bonds = {
        (
            int(item["begin_atom_index"]),
            int(item["end_atom_index"]),
        ): str(item["bond_type"])
        for item in parent_sidecar.get("bonds", [])
    }

    decoded_atomic_numbers = [
        int(values["atomic_num"]) for _, values in decoded
    ]
    decoded_formal_charges = [
        int(values["formal_charge"]) for _, values in decoded
    ]
    decoded_aromaticity = [
        bool(values["is_aromatic"]) for _, values in decoded
    ]
    resolved_attributes: list[dict[str, Any]] = []
    required_source_attributes = {
        "atomic_num",
        "formal_charge",
        "is_aromatic",
        "num_explicit_hs",
        "no_implicit",
        "chiral_tag",
    }

    def make_result(
        *,
        ok: bool,
        canonical_smiles: str | None,
        molecule: Any | None,
        error_type: str | None,
        error: str | None,
        projected_new_edge_count: int = 0,
        retained_edge_count: int = 0,
        deleted_edge_count: int = 0,
        decoded_bond_types: list[str] | None = None,
    ) -> CodecResult:
        return CodecResult(
            ok=ok,
            canonical_smiles=canonical_smiles,
            molecule=molecule,
            error_type=error_type,
            error=error,
            num_real_nodes=len(decoded),
            projected_new_edge_count=projected_new_edge_count,
            retained_edge_count=retained_edge_count,
            deleted_edge_count=deleted_edge_count,
            atom_mapping_unique=atom_mapping_unique,
            decoded_atomic_numbers=decoded_atomic_numbers,
            decoded_formal_charges=decoded_formal_charges,
            decoded_aromaticity=decoded_aromaticity,
            decoded_bond_types=list(decoded_bond_types or []),
            decoded_num_explicit_hs=[
                int(item["num_explicit_hs"]) for item in resolved_attributes
            ],
            decoded_no_implicit=[
                bool(item["no_implicit"]) for item in resolved_attributes
            ],
            decoded_chiral_tags=[
                int(item["chiral_tag"]) for item in resolved_attributes
            ],
        )

    for slot, values in decoded:
        source_atom = source_atoms.get(int(slot))
        missing = (
            sorted(required_source_attributes - set(source_atom or {}))
            if source_atom is not None
            else sorted(required_source_attributes)
        )
        if source_atom is None or missing:
            error_type = (
                "source_atom_attribute_mapping_failed"
                if atom_attribute_mode == "source"
                else "ambiguous_generated_atom_hydrogen_state"
            )
            return make_result(
                ok=False,
                canonical_smiles=None,
                molecule=None,
                error_type=error_type,
                error=(
                    f"Parent atom sidecar is incomplete for node slot {slot}: "
                    f"missing={missing}"
                ),
            )
        decoded_identity = (
            int(values["atomic_num"]),
            int(values["formal_charge"]),
            bool(values["is_aromatic"]),
        )
        source_identity = (
            int(source_atom["atomic_num"]),
            int(source_atom["formal_charge"]),
            bool(source_atom["is_aromatic"]),
        )
        if decoded_identity != source_identity:
            error_type = (
                "source_atom_attribute_mapping_failed"
                if atom_attribute_mode == "source"
                else "ambiguous_generated_atom_hydrogen_state"
            )
            return make_result(
                ok=False,
                canonical_smiles=None,
                molecule=None,
                error_type=error_type,
                error=(
                    f"Node slot {slot} identity changed: "
                    f"source={source_identity}, decoded={decoded_identity}; "
                    "explicit-H state cannot be inferred."
                ),
            )
        resolved_attributes.append(source_atom)

    rw_mol = chem.RWMol()
    try:
        for (_slot, values), source_atom in zip(
            decoded, resolved_attributes, strict=True
        ):
            atom = chem.Atom(int(values["atomic_num"]))
            atom.SetFormalCharge(int(values["formal_charge"]))
            atom.SetNumExplicitHs(int(source_atom["num_explicit_hs"]))
            atom.SetNoImplicit(bool(source_atom["no_implicit"]))
            atom.SetChiralTag(chem.ChiralType(int(source_atom["chiral_tag"])))
            atom.SetIsAromatic(bool(values["is_aromatic"]))
            rw_mol.AddAtom(atom)
    except Exception as exc:
        return make_result(
            ok=False,
            canonical_smiles=None,
            molecule=None,
            error_type="atom_attribute_projection_failure",
            error=str(exc),
        )

    retained = 0
    new_edges = 0
    decoded_bonds: list[str] = []
    active_source_edges = 0
    try:
        for left_position, (left_slot, _left_values) in enumerate(decoded):
            for right_position in range(left_position + 1, len(decoded)):
                right_slot = decoded[right_position][0]
                connected = (
                    max(
                        float(adjacency_array[left_slot, right_slot]),
                        float(adjacency_array[right_slot, left_slot]),
                    )
                    > float(adjacency_threshold)
                )
                source_key = (min(left_slot, right_slot), max(left_slot, right_slot))
                if source_key in source_bonds:
                    active_source_edges += 1
                if not connected:
                    continue
                if source_key in source_bonds:
                    bond_name = source_bonds[source_key]
                    retained += 1
                else:
                    bond_name = "SINGLE"
                    new_edges += 1
                rw_mol.AddBond(
                    int(slot_to_new[left_slot]),
                    int(slot_to_new[right_slot]),
                    _rdkit_bond_type(bond_name),
                )
                if bond_name == "AROMATIC":
                    bond = rw_mol.GetBondBetweenAtoms(
                        int(slot_to_new[left_slot]),
                        int(slot_to_new[right_slot]),
                    )
                    bond.SetIsAromatic(True)
                decoded_bonds.append(bond_name)
    except Exception as exc:
        return make_result(
            ok=False,
            canonical_smiles=None,
            molecule=None,
            error_type="bond_projection_failure",
            error=str(exc),
            projected_new_edge_count=new_edges,
            retained_edge_count=retained,
            deleted_edge_count=max(0, len(source_bonds) - active_source_edges),
            decoded_bond_types=decoded_bonds,
        )
    molecule = rw_mol.GetMol()
    try:
        molecule.UpdatePropertyCache(strict=False)
        chem.SanitizeMol(molecule)
    except Exception as exc:
        message = str(exc)
        error_type = (
            "invalid_valence"
            if "valence" in message.lower()
            else "sanitize_failure"
        )
        return make_result(
            ok=False,
            canonical_smiles=None,
            molecule=None,
            error_type=error_type,
            error=message,
            projected_new_edge_count=new_edges,
            retained_edge_count=retained,
            deleted_edge_count=max(0, len(source_bonds) - retained),
            decoded_bond_types=decoded_bonds,
        )
    if require_connected and len(chem.GetMolFrags(molecule)) != 1:
        return make_result(
            ok=False,
            canonical_smiles=None,
            molecule=None,
            error_type="disconnected_output",
            error="Projected graph contains multiple connected components.",
            projected_new_edge_count=new_edges,
            retained_edge_count=retained,
            deleted_edge_count=max(0, len(source_bonds) - retained),
            decoded_bond_types=decoded_bonds,
        )
    return make_result(
        ok=True,
        canonical_smiles=_canonical_smiles(molecule),
        molecule=molecule,
        error_type=None,
        error=None,
        projected_new_edge_count=new_edges,
        retained_edge_count=retained,
        deleted_edge_count=max(0, len(source_bonds) - retained),
        decoded_bond_types=decoded_bonds,
    )


def round_trip_source_graph(
    row: PreparedMolecule,
    *,
    schema: ClearMutagenicitySchema,
) -> tuple[CodecResult, dict[str, Any]]:
    adjacency, features, sidecar = molecule_to_clear_graph(row, schema=schema)
    result = project_binary_graph_to_molecule(
        features=features,
        adjacency=adjacency,
        schema=schema,
        parent_sidecar=sidecar,
        atom_attribute_mode="source",
    )
    expected_atoms = list(row.atom_categories)
    expected_charges = list(row.charge_categories)
    expected_aromatic = list(row.aromatic_categories)
    expected_explicit_hs = [
        int(atom.GetNumExplicitHs()) for atom in row.mol.GetAtoms()
    ]
    expected_no_implicit = [
        bool(atom.GetNoImplicit()) for atom in row.mol.GetAtoms()
    ]
    expected_chiral_tags = [
        int(atom.GetChiralTag()) for atom in row.mol.GetAtoms()
    ]
    expected_connectivity = sorted(
        (
            min(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())),
            max(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())),
        )
        for bond in row.mol.GetBonds()
    )
    expected_bonds = sorted(
        (
            min(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())),
            max(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())),
            _bond_type_name(bond),
        )
        for bond in row.mol.GetBonds()
    )
    actual_atoms = (
        list(result.molecule.GetAtoms())
        if result.ok and result.molecule is not None
        else []
    )
    actual_atomic_numbers = [
        int(atom.GetAtomicNum()) for atom in actual_atoms
    ]
    actual_formal_charges = [
        int(atom.GetFormalCharge()) for atom in actual_atoms
    ]
    actual_aromaticity = [
        bool(atom.GetIsAromatic()) for atom in actual_atoms
    ]
    actual_explicit_hs = [
        int(atom.GetNumExplicitHs()) for atom in actual_atoms
    ]
    actual_no_implicit = [
        bool(atom.GetNoImplicit()) for atom in actual_atoms
    ]
    actual_chiral_tags = [
        int(atom.GetChiralTag()) for atom in actual_atoms
    ]
    decoded_connectivity: list[tuple[int, int]] = []
    decoded_bonds: list[tuple[int, int, str]] = []
    if result.ok and result.molecule is not None:
        decoded_connectivity = sorted(
            (
                min(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())),
                max(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())),
            )
            for bond in result.molecule.GetBonds()
        )
        decoded_bonds = sorted(
            (
                min(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())),
                max(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())),
                _bond_type_name(bond),
            )
            for bond in result.molecule.GetBonds()
        )
    checks = {
        "atomic_numbers_exact": actual_atomic_numbers == expected_atoms,
        "formal_charges_exact": actual_formal_charges == expected_charges,
        "aromaticity_exact": actual_aromaticity == expected_aromatic,
        "explicit_hs_exact": actual_explicit_hs == expected_explicit_hs,
        "no_implicit_exact": actual_no_implicit == expected_no_implicit,
        "chiral_tags_exact": actual_chiral_tags == expected_chiral_tags,
        "bond_types_exact": decoded_bonds == expected_bonds,
        "connectivity_exact": decoded_connectivity == expected_connectivity,
        "canonical_smiles_exact": (
            result.canonical_smiles == row.canonical_smiles
        ),
        "atom_mapping_unique": result.atom_mapping_unique,
        "atom_ordering_exact": list(
            sidecar["original_atom_indices"]
        )
        == list(range(row.mol.GetNumAtoms())),
        "node_count_exact": result.num_real_nodes == row.mol.GetNumAtoms(),
    }
    checks["round_trip_passed"] = bool(result.ok and all(checks.values()))
    checks["mismatch_fields"] = sorted(
        key
        for key, value in checks.items()
        if key.endswith("_exact") and not bool(value)
    )
    return result, checks


def _cohort_hash(rows: Sequence[PreparedMolecule]) -> str:
    payload = [
        {
            "molecule_id": row.molecule_id,
            "canonical_smiles": row.canonical_smiles,
            "label": row.label,
            "split": row.split,
        }
        for row in rows
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_clear_dataset_payload(
    *,
    cohorts: dict[str, list[PreparedMolecule]],
    graph_data_class: Callable[..., Any],
    num_split_repetitions: int = 10,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    train_rows = cohorts["train_positive"] + cohorts["train_negative"]
    val_rows = cohorts["val_positive"] + cohorts["val_negative"]
    schema = build_train_schema(train_rows)
    validate_validation_schema(val_rows, schema)
    all_rows = train_rows + val_rows
    max_num_nodes = max(row.mol.GetNumAtoms() for row in all_rows)
    adjacency_all: list[np.ndarray] = []
    features_all: list[np.ndarray] = []
    u_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    sidecars: list[dict[str, Any]] = []
    for row in all_rows:
        adjacency, features, sidecar = molecule_to_clear_graph(row, schema=schema)
        adjacency_all.append(adjacency)
        features_all.append(features)
        u_all.append(np.asarray([0.0], dtype=np.float64))
        labels_all.append(np.asarray([float(row.label)], dtype=np.float64))
        sidecars.append(sidecar)
    data = graph_data_class(
        adjacency_all,
        features_all,
        u_all,
        labels_all,
        max_num_nodes,
        padded=True,
    )
    data.molecule_id_all = [row.molecule_id for row in all_rows]
    data.canonical_smiles_all = [row.canonical_smiles for row in all_rows]
    data.original_smiles_all = [row.original_smiles for row in all_rows]
    data.original_atom_index_all = [
        sidecar["original_atom_indices"] for sidecar in sidecars
    ]
    data.source_split_all = [row.split for row in all_rows]
    data.source_row_index_all = [row.source_row_index for row in all_rows]
    data.atom_sidecar_all = [sidecar["atoms"] for sidecar in sidecars]
    data.bond_sidecar_all = [sidecar["bonds"] for sidecar in sidecars]
    data.num_real_nodes_all = [sidecar["num_real_nodes"] for sidecar in sidecars]
    data.molecule_sidecar_all = sidecars
    data.feature_schema = schema.to_dict()
    data.atom_sidecar_schema_version = ATOM_SIDECAR_SCHEMA_VERSION
    data.clear_dataset_name = DATASET_NAME
    data.calibration_loaded = False
    data.test_loaded = False

    train_indices = np.arange(len(train_rows), dtype=np.int64)
    val_indices = np.arange(
        len(train_rows), len(train_rows) + len(val_rows), dtype=np.int64
    )
    split_payload = {
        "idx_train_list": [
            train_indices.copy() for _ in range(num_split_repetitions)
        ],
        "idx_val_list": [
            val_indices.copy() for _ in range(num_split_repetitions)
        ],
        # Official code requires a third loader.  It aliases validation and
        # never introduces project calibration/test data.
        "idx_test_list": [
            val_indices.copy() for _ in range(num_split_repetitions)
        ],
        "split_source": "strict_project_train_val",
        "official_test_loader_semantics": "validation_alias_no_external_test",
        "num_splits": int(num_split_repetitions),
        "calibration_loaded": False,
        "test_loaded": False,
    }
    summary = {
        "dataset": DATASET_NAME,
        "train_rows": len(train_rows),
        "train_positive_rows": len(cohorts["train_positive"]),
        "train_negative_rows": len(cohorts["train_negative"]),
        "val_rows": len(val_rows),
        "val_positive_rows": len(cohorts["val_positive"]),
        "val_negative_rows": len(cohorts["val_negative"]),
        "generation_source_parent_rows": len(cohorts["train_positive"]),
        "label_counts_train": {
            "0": len(cohorts["train_negative"]),
            "1": len(cohorts["train_positive"]),
        },
        "label_counts_val": {
            "0": len(cohorts["val_negative"]),
            "1": len(cohorts["val_positive"]),
        },
        "atom_vocabulary": list(schema.atom_vocabulary),
        "formal_charge_vocabulary": list(
            schema.formal_charge_vocabulary
        ),
        "aromaticity_vocabulary": list(schema.aromaticity_vocabulary),
        "bond_type_vocabulary": list(schema.bond_type_vocabulary),
        "feature_schema": schema.to_dict(),
        "feature_dim": schema.feature_dim,
        "max_num_nodes": int(max_num_nodes),
        "max_num_nodes_source": "strict_train_plus_validation",
        "vocabulary_source": "strict_train_only",
        "train_hash": _cohort_hash(train_rows),
        "val_hash": _cohort_hash(val_rows),
        "calibration_loaded": False,
        "test_loaded": False,
        **codec_provenance(),
    }
    return data, split_payload, summary


def write_clear_dataset(
    *,
    data: Any,
    split_payload: dict[str, Any],
    summary: dict[str, Any],
    official_dataset_dir: str | Path,
    summary_path: str | Path,
) -> dict[str, Any]:
    official_dir = Path(official_dataset_dir).expanduser().resolve()
    summary_destination = Path(summary_path).expanduser().resolve()
    official_dir.mkdir(parents=True, exist_ok=True)
    summary_destination.parent.mkdir(parents=True, exist_ok=True)
    full_pickle = official_dir / "mutagenicity_full.pickle"
    split_pickle = official_dir / "mutagenicity_datasplit.pickle"
    with full_pickle.open("wb") as handle:
        pickle.dump({"data": data}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with split_pickle.open("wb") as handle:
        pickle.dump(split_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    payload = {
        **summary,
        "output_full_pickle": str(full_pickle),
        "output_datasplit_pickle": str(split_pickle),
        "summary_path": str(summary_destination),
    }
    summary_destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def required_category_probe_selection(
    rows: Sequence[PreparedMolecule],
    *,
    limit: int,
) -> tuple[list[PreparedMolecule], dict[str, Any]]:
    if limit <= 0:
        raise ClearMutagenicityError("Probe limit must be positive.")
    ordered = sorted(rows, key=lambda row: row.molecule_id)
    predicates: tuple[tuple[str, Callable[[PreparedMolecule], bool]], ...] = (
        ("atom_Br", lambda row: 35 in row.atom_categories),
        ("atom_Cl", lambda row: 17 in row.atom_categories),
        ("atom_P", lambda row: 15 in row.atom_categories),
        ("atom_I", lambda row: 53 in row.atom_categories),
        ("formal_charge_-1", lambda row: -1 in row.charge_categories),
        ("formal_charge_+1", lambda row: 1 in row.charge_categories),
        ("bond_AROMATIC", lambda row: "AROMATIC" in row.bond_categories),
        ("bond_DOUBLE", lambda row: "DOUBLE" in row.bond_categories),
        ("bond_TRIPLE", lambda row: "TRIPLE" in row.bond_categories),
    )
    selected: list[PreparedMolecule] = []
    selected_ids: set[str] = set()
    coverage: dict[str, str | None] = {}
    for name, predicate in predicates:
        match = next((row for row in ordered if predicate(row)), None)
        coverage[name] = match.molecule_id if match is not None else None
        if match is not None and match.molecule_id not in selected_ids:
            selected.append(match)
            selected_ids.add(match.molecule_id)
    max_nodes = max(row.mol.GetNumAtoms() for row in ordered)
    near_max = next(
        row for row in ordered if row.mol.GetNumAtoms() == max_nodes
    )
    coverage["near_max_nodes"] = near_max.molecule_id
    if near_max.molecule_id not in selected_ids:
        selected.append(near_max)
        selected_ids.add(near_max.molecule_id)
    missing = [name for name, molecule_id in coverage.items() if molecule_id is None]
    if missing:
        raise ClearMutagenicityCodecError(
            f"Required codec probe categories are absent from train/val: {missing}"
        )
    if len(selected) > limit:
        raise ClearMutagenicityError(
            f"Probe limit={limit} is too small for required-category rows="
            f"{len(selected)}."
        )
    for row in ordered:
        if len(selected) >= limit:
            break
        if row.molecule_id not in selected_ids:
            selected.append(row)
            selected_ids.add(row.molecule_id)
    return selected, {
        "sampling": "deterministic_required_category_first_then_molecule_id",
        "required_category_representatives": coverage,
        "max_nodes_in_train_val": int(max_nodes),
        "selected_rows": len(selected),
    }


def run_codec_probe(
    *,
    cohorts: dict[str, list[PreparedMolecule]],
    limit: int = 64,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_rows = cohorts["train_positive"] + cohorts["train_negative"]
    val_rows = cohorts["val_positive"] + cohorts["val_negative"]
    schema = build_train_schema(train_rows)
    validate_validation_schema(val_rows, schema)
    selected, selection_audit = required_category_probe_selection(
        train_rows + val_rows,
        limit=int(limit),
    )
    output_rows: list[dict[str, Any]] = []
    for row in selected:
        result, checks = round_trip_source_graph(row, schema=schema)
        contains_aromatic_explicit_h = any(
            bool(atom.GetIsAromatic()) and int(atom.GetNumExplicitHs()) > 0
            for atom in row.mol.GetAtoms()
        )
        output_rows.append(
            {
                "molecule_id": row.molecule_id,
                "source_split": row.split,
                "label": row.label,
                "original_smiles": row.original_smiles,
                "source_canonical_smiles": row.canonical_smiles,
                "num_real_nodes": row.mol.GetNumAtoms(),
                "atomic_numbers": list(row.atom_categories),
                "formal_charges": list(row.charge_categories),
                "aromaticity": list(row.aromatic_categories),
                "num_explicit_hs": [
                    int(atom.GetNumExplicitHs())
                    for atom in row.mol.GetAtoms()
                ],
                "num_implicit_hs": [
                    int(atom.GetNumImplicitHs())
                    for atom in row.mol.GetAtoms()
                ],
                "no_implicit": [
                    bool(atom.GetNoImplicit()) for atom in row.mol.GetAtoms()
                ],
                "chiral_tags": [
                    int(atom.GetChiralTag()) for atom in row.mol.GetAtoms()
                ],
                "contains_aromatic_explicit_h": (
                    contains_aromatic_explicit_h
                ),
                "bond_types": list(row.bond_categories),
                **result.to_dict(),
                **checks,
            }
        )
    failed = [row for row in output_rows if not row["round_trip_passed"]]
    failure_audit = [
        {
            "molecule_id": row["molecule_id"],
            "original_smiles": row["original_smiles"],
            "contains_aromatic_explicit_h": row[
                "contains_aromatic_explicit_h"
            ],
            "failure_stage": (
                row.get("error_type") or "round_trip_comparison"
            ),
            "failure_message": (
                row.get("error")
                or (
                    "Exact source round-trip mismatch: "
                    + ",".join(row.get("mismatch_fields", []))
                )
            ),
        }
        for row in failed
    ]
    summary = {
        "probe_rows": len(output_rows),
        "probe_passed_rows": len(output_rows) - len(failed),
        "probe_failed_rows": len(failed),
        "probe_passed": not failed,
        "failure_examples": failed[:10],
        "failed_count": len(failed),
        "failed_molecule_ids": [
            row["molecule_id"] for row in failure_audit
        ],
        "failed_smiles": [
            row["original_smiles"] for row in failure_audit
        ],
        "failure_audit": failure_audit,
        "atom_vocabulary": list(schema.atom_vocabulary),
        "formal_charge_vocabulary": list(
            schema.formal_charge_vocabulary
        ),
        "aromaticity_vocabulary": list(schema.aromaticity_vocabulary),
        "bond_type_vocabulary": list(schema.bond_type_vocabulary),
        "vocabulary_source": "strict_train_only",
        "max_num_nodes": max(
            row.mol.GetNumAtoms() for row in train_rows + val_rows
        ),
        "max_num_nodes_source": "strict_train_plus_validation",
        "calibration_loaded": False,
        "test_loaded": False,
        **selection_audit,
        **codec_provenance(),
    }
    return output_rows, summary


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


__all__ = [
    "ATOM_SIDECAR_SCHEMA_VERSION",
    "BOND_DECODER_NATIVE",
    "ClearMutagenicityCodecError",
    "ClearMutagenicityError",
    "ClearMutagenicitySchema",
    "CodecResult",
    "DEFAULT_EXPECTED_COUNTS",
    "NATIVE_ADJACENCY_SEMANTICS",
    "NEW_EDGE_BOND_RULE",
    "PreparedMolecule",
    "build_clear_dataset_payload",
    "build_train_schema",
    "codec_provenance",
    "decode_atom_feature",
    "encode_atom_feature",
    "load_phase_a_cohorts",
    "molecule_to_clear_graph",
    "project_binary_graph_to_molecule",
    "required_category_probe_selection",
    "round_trip_source_graph",
    "run_codec_probe",
    "validate_validation_schema",
    "write_clear_dataset",
    "write_jsonl",
]
