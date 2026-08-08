"""BACE data and checkpoint adapter for the official GCFExplainer runtime.

The official implementation continues to own the GNN architecture, edit map,
importance calculation, NeuroSED model, VRRW, and greedy summary.  This module
only maps the frozen project BACE cohorts into that runtime and records the
deterministic NeuroSED input-channel projection needed by BACE's atom alphabet.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rdkit import Chem

from src.baselines.gcfexplainer_mutagenicity_adapter import (
    GCFExplainerMutagenicityCodecError,
    StrictMolecule,
    cohort_hash,
    encode_source_graph,
    load_strict_molecules,
    read_json,
    run_codec_probe,
    sha256_file,
    write_json,
    write_jsonl,
)


DATASET = "BACE"
SOURCE_LABEL = 1
TARGET_LABEL = 0
SEED = 13

EXPECTED_TRAIN_SOURCE_ROWS = 360
EXPECTED_TRAIN_TARGET_ROWS = 509
EXPECTED_VAL_SOURCE_ROWS = 92
EXPECTED_VAL_TARGET_ROWS = 70
EXPECTED_MODEL_TRAIN_ROWS = 869
EXPECTED_MODEL_VAL_ROWS = 162
EXPECTED_GENERATION_SOURCE_ROWS = 360

# Keep the official Mutagenicity channel order for the shared atom types and
# remove only phosphorus, which is absent from the frozen BACE train/val data.
# This preserves channel semantics when projecting the verified ten-channel
# NeuroSED checkpoint to BACE's nine-channel edit space.
MUTAGENICITY_FEATURE_ATOMIC_NUMBERS = (6, 8, 17, 1, 7, 9, 35, 16, 15, 53)
BACE_FEATURE_ATOMIC_NUMBERS = (6, 8, 17, 1, 7, 9, 35, 16, 53)
BACE_FEATURE_KEEP_INDICES = tuple(
    MUTAGENICITY_FEATURE_ATOMIC_NUMBERS.index(value)
    for value in BACE_FEATURE_ATOMIC_NUMBERS
)
SUPPORTED_BOND_TYPES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
ATOM_SIDECAR_SCHEMA_VERSION = "gcfexplainer_bace_atom_sidecar_v1"


@dataclass(frozen=True, slots=True)
class BACEGraphSchema:
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
        for key in (
            "atom_vocabulary",
            "feature_atomic_numbers",
            "formal_charge_vocabulary",
            "aromaticity_vocabulary",
            "bond_type_vocabulary",
        ):
            payload[key] = list(payload[key])
        payload.update(
            {
                "node_feature_dim": self.node_feature_dim,
                "native_node_label_semantics": "one_hot_atom_type",
                "official_dataset_name": "bace",
                "official_label_direction": {
                    "project_source_label_1": 0,
                    "project_target_label_0": 1,
                },
                "native_adjacency_semantics": "binary_connectivity",
                "retained_bond_type_source": "source_sidecar",
                "new_edge_bond_rule": "provisional_single",
            }
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BACEGraphSchema":
        return cls(
            atom_vocabulary=tuple(int(value) for value in payload["atom_vocabulary"]),
            feature_atomic_numbers=tuple(
                int(value) for value in payload["feature_atomic_numbers"]
            ),
            formal_charge_vocabulary=tuple(
                int(value) for value in payload["formal_charge_vocabulary"]
            ),
            aromaticity_vocabulary=tuple(
                bool(value) for value in payload["aromaticity_vocabulary"]
            ),
            bond_type_vocabulary=tuple(
                str(value) for value in payload["bond_type_vocabulary"]
            ),
            max_num_nodes=int(payload["max_num_nodes"]),
            atom_sidecar_schema_version=str(
                payload.get("atom_sidecar_schema_version", ATOM_SIDECAR_SCHEMA_VERSION)
            ),
            explicit_h_nodes=bool(payload.get("explicit_h_nodes", True)),
        )


def _expanded_molecule(row: StrictMolecule) -> Any:
    molecule = Chem.MolFromSmiles(row.canonical_smiles)
    if molecule is None:
        raise GCFExplainerMutagenicityCodecError(
            f"Cannot parse BACE source molecule {row.molecule_id}."
        )
    Chem.SanitizeMol(molecule)
    expanded = Chem.AddHs(molecule, addCoords=False)
    Chem.SanitizeMol(expanded)
    return expanded


def _bond_name(bond: Any) -> str:
    mapping = {
        Chem.BondType.SINGLE: "SINGLE",
        Chem.BondType.DOUBLE: "DOUBLE",
        Chem.BondType.TRIPLE: "TRIPLE",
        Chem.BondType.AROMATIC: "AROMATIC",
    }
    try:
        return mapping[bond.GetBondType()]
    except KeyError as exc:
        raise GCFExplainerMutagenicityCodecError(
            f"Unsupported BACE source bond type: {bond.GetBondType()}."
        ) from exc


def derive_bace_schema(
    train_rows: Sequence[StrictMolecule],
    val_rows: Sequence[StrictMolecule],
) -> BACEGraphSchema:
    def categories(rows: Sequence[StrictMolecule]) -> dict[str, set[Any]]:
        values: dict[str, set[Any]] = {
            "atoms": set(),
            "charges": set(),
            "aromaticity": set(),
            "bonds": set(),
        }
        for row in rows:
            molecule = _expanded_molecule(row)
            values["atoms"].update(
                int(atom.GetAtomicNum()) for atom in molecule.GetAtoms()
            )
            values["charges"].update(
                int(atom.GetFormalCharge()) for atom in molecule.GetAtoms()
            )
            values["aromaticity"].update(
                bool(atom.GetIsAromatic()) for atom in molecule.GetAtoms()
            )
            values["bonds"].update(_bond_name(bond) for bond in molecule.GetBonds())
        return values

    train = categories(train_rows)
    validation = categories(val_rows)
    expected_atoms = set(BACE_FEATURE_ATOMIC_NUMBERS)
    if train["atoms"] != expected_atoms:
        raise ValueError(
            "Frozen BACE train atom vocabulary differs from the preregistered "
            f"nine-channel edit space: actual={sorted(train['atoms'])}, "
            f"expected={sorted(expected_atoms)}."
        )
    unseen = {
        "atoms": sorted(validation["atoms"] - train["atoms"]),
        "formal_charges": sorted(validation["charges"] - train["charges"]),
        "aromaticity": sorted(validation["aromaticity"] - train["aromaticity"]),
        "bond_types": sorted(validation["bonds"] - train["bonds"]),
    }
    if any(unseen.values()):
        raise ValueError(f"BACE validation contains train-unseen categories: {unseen}")
    max_nodes = max(
        _expanded_molecule(row).GetNumAtoms() for row in (*train_rows, *val_rows)
    )
    return BACEGraphSchema(
        atom_vocabulary=BACE_FEATURE_ATOMIC_NUMBERS,
        feature_atomic_numbers=BACE_FEATURE_ATOMIC_NUMBERS,
        formal_charge_vocabulary=tuple(sorted(train["charges"])),
        aromaticity_vocabulary=tuple(sorted(train["aromaticity"])),
        bond_type_vocabulary=tuple(
            value for value in SUPPORTED_BOND_TYPES if value in train["bonds"]
        ),
        max_num_nodes=int(max_nodes),
    )


def prepare_bace_gcf_dataset(
    *,
    train_source_csv: str | Path,
    train_target_csv: str | Path,
    val_source_csv: str | Path,
    val_target_csv: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    cohorts = {
        "train_source": load_strict_molecules(
            train_source_csv,
            expected_split="train",
            expected_label=SOURCE_LABEL,
            expected_rows=EXPECTED_TRAIN_SOURCE_ROWS,
        ),
        "train_target": load_strict_molecules(
            train_target_csv,
            expected_split="train",
            expected_label=TARGET_LABEL,
            expected_rows=EXPECTED_TRAIN_TARGET_ROWS,
        ),
        "val_source": load_strict_molecules(
            val_source_csv,
            expected_split="val",
            expected_label=SOURCE_LABEL,
            expected_rows=EXPECTED_VAL_SOURCE_ROWS,
        ),
        "val_target": load_strict_molecules(
            val_target_csv,
            expected_split="val",
            expected_label=TARGET_LABEL,
            expected_rows=EXPECTED_VAL_TARGET_ROWS,
        ),
    }
    train_rows = sorted(
        (*cohorts["train_source"], *cohorts["train_target"]),
        key=lambda row: row.molecule_id,
    )
    val_rows = sorted(
        (*cohorts["val_source"], *cohorts["val_target"]),
        key=lambda row: row.molecule_id,
    )
    all_ids = [row.molecule_id for row in (*train_rows, *val_rows)]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("BACE strict train/validation molecule_id overlap detected.")
    if {row.canonical_smiles for row in train_rows} & {
        row.canonical_smiles for row in val_rows
    }:
        raise ValueError("BACE strict train/validation canonical SMILES overlap detected.")

    schema = derive_bace_schema(train_rows, val_rows)
    train_graphs = [encode_source_graph(row, schema) for row in train_rows]
    val_graphs = [encode_source_graph(row, schema) for row in val_rows]
    generation_graphs = [
        encode_source_graph(row, schema) for row in cohorts["train_source"]
    ]
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"BACE GCF dataset output is non-empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    try:
        import torch
    except Exception as exc:  # pragma: no cover - HPC dependency
        raise RuntimeError("BACE GCF preparation requires PyTorch.") from exc
    torch.save(train_graphs, destination / "train_graphs.pt")
    torch.save(val_graphs, destination / "val_graphs.pt")
    torch.save(generation_graphs, destination / "generation_source_graphs.pt")
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
    probe = run_codec_probe(
        (*train_graphs, *val_graphs),
        schema,
        output_dir=destination / "codec_probe_64",
        limit=64,
        require_all=True,
    )
    inputs = {
        "train_source": str(Path(train_source_csv).expanduser().resolve()),
        "train_target": str(Path(train_target_csv).expanduser().resolve()),
        "val_source": str(Path(val_source_csv).expanduser().resolve()),
        "val_target": str(Path(val_target_csv).expanduser().resolve()),
    }
    summary = {
        "schema_version": "gcfexplainer_bace_dataset_v1",
        "dataset": DATASET,
        "adapter": "official_gcfexplainer_bace_project_data",
        "train_rows": len(train_graphs),
        "train_source_rows": len(cohorts["train_source"]),
        "train_target_rows": len(cohorts["train_target"]),
        "val_rows": len(val_graphs),
        "val_source_rows": len(cohorts["val_source"]),
        "val_target_rows": len(cohorts["val_target"]),
        "generation_source_rows": len(generation_graphs),
        "train_label_counts": dict(Counter(int(row["label"]) for row in train_graphs)),
        "val_label_counts": dict(Counter(int(row["label"]) for row in val_graphs)),
        "gnn_label_mapping": {"project_1": 0, "project_0": 1},
        **schema.to_dict(),
        "train_ids_hash": cohort_hash(train_graphs),
        "val_ids_hash": cohort_hash(val_graphs),
        "generation_source_cohort_hash": cohort_hash(generation_graphs),
        "input_paths": inputs,
        "input_sha256": {key: sha256_file(path) for key, path in inputs.items()},
        "probe": probe,
        "calibration_loaded": False,
        "test_loaded": False,
        "run_complete": True,
    }
    write_json(destination / "dataset_summary.json", summary)
    artifacts = (
        "train_graphs.pt",
        "val_graphs.pt",
        "generation_source_graphs.pt",
        "source_graph_manifest.jsonl",
        "codec_probe_64/codec_probe_rows.jsonl",
        "codec_probe_64/codec_probe_summary.json",
        "dataset_summary.json",
    )
    write_json(
        destination / "run_manifest.json",
        {
            **summary,
            "artifacts": {
                relative: {
                    "bytes": (destination / relative).stat().st_size,
                    "sha256": sha256_file(destination / relative),
                }
                for relative in artifacts
            },
        },
    )
    write_json(destination / "_RUN_COMPLETE.json", {"run_complete": True})
    return summary


def load_bace_gcf_dataset(
    dataset_dir: str | Path,
) -> tuple[
    BACEGraphSchema,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    root = Path(dataset_dir).expanduser().resolve()
    summary = read_json(root / "dataset_summary.json")
    if summary.get("dataset") != DATASET:
        raise ValueError("Prepared GCF dataset is not BACE.")
    if summary.get("calibration_loaded") is not False or summary.get("test_loaded") is not False:
        raise ValueError("Prepared BACE GCF dataset reports calibration/test leakage.")
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("BACE GCF loading requires PyTorch.") from exc

    def load(path: Path) -> Any:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    schema = BACEGraphSchema.from_dict(summary)
    train = list(load(root / "train_graphs.pt"))
    val = list(load(root / "val_graphs.pt"))
    generation = list(load(root / "generation_source_graphs.pt"))
    expected = (
        int(summary["train_rows"]),
        int(summary["val_rows"]),
        int(summary["generation_source_rows"]),
    )
    actual = (len(train), len(val), len(generation))
    if actual != expected:
        raise ValueError(f"BACE GCF graph artifact count mismatch: {actual} != {expected}")
    ids = [str(row["molecule_id"]) for row in (*train, *val)]
    if len(ids) != len(set(ids)):
        raise ValueError("BACE prepared train/validation graph IDs overlap.")
    generation_ids = [str(row["molecule_id"]) for row in generation]
    if len(generation_ids) != len(set(generation_ids)):
        raise ValueError("BACE generation parent IDs are duplicated.")
    if any(
        str(row.get("split")) != "train" or int(row.get("label", -1)) != SOURCE_LABEL
        for row in generation
    ):
        raise ValueError("BACE generation artifact is not the strict train-source cohort.")
    return schema, train, val, generation, summary


def adapt_bace_neurosed_checkpoint(
    *,
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Project the verified Mutagenicity NeuroSED input layer to BACE channels."""

    source = Path(source_checkpoint).expanduser().resolve()
    target = Path(output_checkpoint).expanduser().resolve()
    manifest_target = Path(manifest_path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"Source NeuroSED checkpoint missing: {source}")
    if target.exists() or manifest_target.exists():
        raise FileExistsError("BACE NeuroSED projection output already exists.")
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("BACE NeuroSED adaptation requires PyTorch.") from exc
    try:
        state = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(source, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("NeuroSED source checkpoint must be a state_dict mapping.")
    key = "embed_model.pre.weight"
    weight = state.get(key)
    if weight is None or tuple(weight.shape)[1:] != (10,):
        raise ValueError("Expected a ten-channel official Mutagenicity NeuroSED input layer.")
    projected = {
        name: value.detach().clone() if hasattr(value, "detach") else value
        for name, value in state.items()
    }
    projected[key] = weight[:, list(BACE_FEATURE_KEEP_INDICES)].detach().clone()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(projected, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    if tuple(projected[key].shape)[1:] != (len(BACE_FEATURE_ATOMIC_NUMBERS),):
        raise AssertionError("Projected BACE NeuroSED input dimension is not nine.")
    manifest = {
        "schema_version": "gcfexplainer_bace_neurosed_projection_v1",
        "dataset": DATASET,
        "adaptation": "deterministic_input_channel_projection",
        "source_dataset": "Mutagenicity",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": sha256_file(source),
        "output_checkpoint": str(target),
        "output_checkpoint_sha256": sha256_file(target),
        "source_feature_atomic_numbers": list(MUTAGENICITY_FEATURE_ATOMIC_NUMBERS),
        "target_feature_atomic_numbers": list(BACE_FEATURE_ATOMIC_NUMBERS),
        "kept_source_indices": list(BACE_FEATURE_KEEP_INDICES),
        "removed_atomic_numbers": [15],
        "architecture_changed": False,
        "non_input_weights_changed": False,
        "training_performed": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    write_json(manifest_target, manifest)
    return manifest


def validate_bace_gnn_profile(
    profile: str,
    *,
    epochs: int,
    train_rows: int,
    val_rows: int,
) -> None:
    if profile == "smoke":
        if int(epochs) != 5 or not (0 < train_rows <= EXPECTED_MODEL_TRAIN_ROWS):
            raise ValueError("BACE GNN smoke requires epochs=5 and a non-empty train prefix.")
        if not 0 < val_rows <= EXPECTED_MODEL_VAL_ROWS:
            raise ValueError("BACE GNN smoke validation prefix is invalid.")
        return
    if profile != "full":
        raise ValueError(f"Unknown BACE GNN profile: {profile}")
    if (int(epochs), int(train_rows), int(val_rows)) != (
        1000,
        EXPECTED_MODEL_TRAIN_ROWS,
        EXPECTED_MODEL_VAL_ROWS,
    ):
        raise ValueError("BACE GNN full requires epochs=1000, train=869, val=162.")


def validate_bace_vrrw_profile(
    profile: str,
    *,
    parent_limit: int,
    m: int,
    alpha: float,
    theta: float,
    seed: int,
) -> None:
    if profile == "smoke":
        expected_parent = 64
        allowed_m = {500, 1000}
    elif profile == "full":
        expected_parent = EXPECTED_GENERATION_SOURCE_ROWS
        allowed_m = {50000}
    else:
        raise ValueError(f"Unknown BACE VRRW profile: {profile}")
    if int(parent_limit) != expected_parent:
        raise ValueError(
            f"BACE {profile} parent_limit={parent_limit}, expected={expected_parent}."
        )
    if int(m) not in allowed_m:
        raise ValueError(f"BACE {profile} M={m}, expected={sorted(allowed_m)}.")
    if not math.isclose(float(alpha), 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("BACE VRRW requires alpha=1.0.")
    if not math.isclose(float(theta), 0.05, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("BACE VRRW requires theta=0.05.")
    if int(seed) != SEED:
        raise ValueError("BACE VRRW requires seed=13.")


__all__ = [
    "ATOM_SIDECAR_SCHEMA_VERSION",
    "BACE_FEATURE_ATOMIC_NUMBERS",
    "BACE_FEATURE_KEEP_INDICES",
    "BACEGraphSchema",
    "DATASET",
    "EXPECTED_GENERATION_SOURCE_ROWS",
    "EXPECTED_MODEL_TRAIN_ROWS",
    "EXPECTED_MODEL_VAL_ROWS",
    "SEED",
    "SOURCE_LABEL",
    "TARGET_LABEL",
    "adapt_bace_neurosed_checkpoint",
    "derive_bace_schema",
    "load_bace_gcf_dataset",
    "prepare_bace_gcf_dataset",
    "validate_bace_gnn_profile",
    "validate_bace_vrrw_profile",
]
