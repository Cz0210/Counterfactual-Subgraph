"""Deterministic BACE CSV adapter for project molecular graph workflows."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SMILES_COL = "smiles"
LABEL_COL = "label"
SPLIT_NAMES = ("train", "val", "calibration", "test")
DEFAULT_SPLIT_RATIOS = (0.70, 0.10, 0.10, 0.10)


@dataclass(frozen=True, slots=True)
class BACEGraph:
    """Portable molecular graph; conversion to PyG is explicit and lossless."""

    atom_features: tuple[tuple[int, int, int, int, int], ...]
    atom_symbols: tuple[str, ...]
    edge_index: tuple[tuple[int, ...], tuple[int, ...]]
    bond_types: tuple[str, ...]

    @property
    def num_nodes(self) -> int:
        return len(self.atom_features)

    @property
    def num_edges(self) -> int:
        return len(self.bond_types) // 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_features": [list(values) for values in self.atom_features],
            "atom_symbols": list(self.atom_symbols),
            "edge_index": [list(values) for values in self.edge_index],
            "bond_types": list(self.bond_types),
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
        }

    def to_pyg(self, *, label: int, molecule_id: str, smiles: str) -> Any:
        """Return a PyG Data object without changing the stored graph ordering."""

        try:
            import torch
            from torch_geometric.data import Data
        except Exception as exc:  # pragma: no cover - exercised on HPC.
            raise RuntimeError(
                "BACE PyG conversion requires torch and torch_geometric."
            ) from exc
        x = torch.tensor(self.atom_features, dtype=torch.long)
        edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        edge_attr = torch.tensor(
            [[_bond_type_index(value)] for value in self.bond_types],
            dtype=torch.long,
        )
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([int(label)], dtype=torch.long),
        )
        graph.molecule_id = str(molecule_id)
        graph.parent_id = str(molecule_id)
        graph.smiles = str(smiles)
        graph.atom_symbols = list(self.atom_symbols)
        graph.bond_types = list(self.bond_types)
        return graph


@dataclass(frozen=True, slots=True)
class BACERecord:
    molecule_id: str
    smiles: str
    label: int
    graph: BACEGraph
    split: str
    original_row_index: int
    scaffold: str
    source_graph_hash: str

    def csv_row(self, graph_index: int) -> dict[str, Any]:
        return {
            "molecule_id": self.molecule_id,
            "parent_id": self.molecule_id,
            SMILES_COL: self.smiles,
            LABEL_COL: int(self.label),
            "split": self.split,
            "original_row_index": int(self.original_row_index),
            "scaffold": self.scaffold,
            "num_atoms": self.graph.num_nodes,
            "num_bonds": self.graph.num_edges,
            "graph_index": int(graph_index),
            "source_graph_hash": self.source_graph_hash,
        }


CSV_FIELDS = (
    "molecule_id",
    "parent_id",
    SMILES_COL,
    LABEL_COL,
    "split",
    "original_row_index",
    "scaffold",
    "num_atoms",
    "num_bonds",
    "graph_index",
    "source_graph_hash",
)


def _require_rdkit() -> tuple[Any, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:  # pragma: no cover - checked on HPC.
        raise RuntimeError("BACE preparation requires RDKit.") from exc
    return Chem, MurckoScaffold


def _bond_type_name(bond: Any) -> str:
    name = str(bond.GetBondType()).upper()
    return name if name in {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"} else "OTHER"


def _bond_type_index(value: str) -> int:
    return {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}.get(
        str(value).upper(), 4
    )


def graph_from_molecule(molecule: Any) -> BACEGraph:
    atom_features = tuple(
        (
            int(atom.GetAtomicNum()),
            int(atom.GetFormalCharge()),
            int(atom.GetIsAromatic()),
            int(atom.GetTotalNumHs()),
            int(atom.GetChiralTag()),
        )
        for atom in molecule.GetAtoms()
    )
    atom_symbols = tuple(str(atom.GetSymbol()) for atom in molecule.GetAtoms())
    sources: list[int] = []
    targets: list[int] = []
    bond_types: list[str] = []
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
        bond_type = _bond_type_name(bond)
        sources.extend((begin, end))
        targets.extend((end, begin))
        bond_types.extend((bond_type, bond_type))
    return BACEGraph(
        atom_features=atom_features,
        atom_symbols=atom_symbols,
        edge_index=(tuple(sources), tuple(targets)),
        bond_types=tuple(bond_types),
    )


def _stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_label(value: Any, *, row_number: int) -> int:
    text = str(value or "").strip()
    if text in {"0", "0.0"}:
        return 0
    if text in {"1", "1.0"}:
        return 1
    raise ValueError(f"BACE label must be binary at CSV row {row_number}: {value!r}")


def _split_for_scaffold(
    scaffold: str,
    *,
    seed: int,
    split_ratios: Sequence[float],
) -> str:
    digest = hashlib.sha256(f"{int(seed)}\0{scaffold}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    cumulative = 0.0
    for name, ratio in zip(SPLIT_NAMES, split_ratios, strict=True):
        cumulative += float(ratio)
        if value < cumulative or name == SPLIT_NAMES[-1]:
            return name
    raise AssertionError("unreachable BACE split assignment")


def _validate_split_ratios(values: Sequence[float]) -> tuple[float, ...]:
    ratios = tuple(float(value) for value in values)
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError(f"Expected {len(SPLIT_NAMES)} split ratios, received {ratios}")
    if any(not math.isfinite(value) or value <= 0.0 for value in ratios):
        raise ValueError("BACE split ratios must be finite and positive.")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"BACE split ratios must sum to one: {ratios}")
    return ratios


def load_bace_records(
    raw_csv: str | Path,
    *,
    raw_smiles_col: str = SMILES_COL,
    raw_label_col: str = LABEL_COL,
    split_seed: int = 13,
    split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
) -> tuple[list[BACERecord], dict[str, Any]]:
    """Normalize one raw BACE CSV while preserving row and graph lineage."""

    Chem, MurckoScaffold = _require_rdkit()
    source = Path(raw_csv).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"BACE raw CSV does not exist: {source}")
    ratios = _validate_split_ratios(split_ratios)
    records: list[BACERecord] = []
    invalid_smiles: list[dict[str, Any]] = []
    duplicates = 0
    canonical_labels: dict[str, int] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted({raw_smiles_col, raw_label_col} - fields)
        if missing:
            raise ValueError(f"BACE raw CSV is missing columns {missing}: {source}")
        raw_rows = 0
        for row_index, row in enumerate(reader):
            raw_rows += 1
            raw_smiles = str(row.get(raw_smiles_col) or "").strip()
            try:
                molecule = Chem.MolFromSmiles(raw_smiles) if raw_smiles else None
                if molecule is not None:
                    Chem.SanitizeMol(molecule)
            except Exception as exc:
                molecule = None
                failure = type(exc).__name__
            else:
                failure = "rdkit_parse_failed"
            if molecule is None:
                invalid_smiles.append(
                    {
                        "original_row_index": row_index,
                        "smiles": raw_smiles,
                        "reason": failure,
                    }
                )
                continue
            label = _parse_label(row.get(raw_label_col), row_number=row_index + 2)
            canonical = str(
                Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
            )
            if canonical in canonical_labels:
                if canonical_labels[canonical] != label:
                    raise ValueError(
                        "BACE duplicate canonical SMILES has conflicting labels: "
                        f"{canonical!r}"
                    )
                duplicates += 1
                continue
            canonical_labels[canonical] = label
            scaffold = str(
                MurckoScaffold.MurckoScaffoldSmiles(
                    mol=molecule, includeChirality=True
                )
                or canonical
            )
            split = _split_for_scaffold(
                scaffold, seed=split_seed, split_ratios=ratios
            )
            graph = graph_from_molecule(molecule)
            molecule_id = "BACE_" + hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest()[:16]
            source_graph_hash = _stable_json_hash(graph.to_dict())
            records.append(
                BACERecord(
                    molecule_id=molecule_id,
                    smiles=canonical,
                    label=label,
                    graph=graph,
                    split=split,
                    original_row_index=row_index,
                    scaffold=scaffold,
                    source_graph_hash=source_graph_hash,
                )
            )
    if not records:
        raise ValueError(f"No valid BACE molecules remained after normalization: {source}")
    if len({record.molecule_id for record in records}) != len(records):
        raise AssertionError("Stable BACE molecule IDs are not unique.")
    audit = {
        "raw_rows": raw_rows,
        "valid_unique_rows": len(records),
        "invalid_smiles": invalid_smiles,
        "duplicate_canonical_smiles_count": duplicates,
        "raw_smiles_col": raw_smiles_col,
        "raw_label_col": raw_label_col,
        "split_seed": int(split_seed),
        "split_ratios": dict(zip(SPLIT_NAMES, ratios, strict=True)),
    }
    return records, audit


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    lines: list[str] = []
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_bace_dataset(
    *,
    raw_csv: str | Path,
    output_dir: str | Path,
    raw_smiles_col: str = SMILES_COL,
    raw_label_col: str = LABEL_COL,
    split_seed: int = 13,
    split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
) -> dict[str, Any]:
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"BACE processed output is non-empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    records, source_audit = load_bace_records(
        raw_csv,
        raw_smiles_col=raw_smiles_col,
        raw_label_col=raw_label_col,
        split_seed=split_seed,
        split_ratios=split_ratios,
    )
    indexed = [(record, index) for index, record in enumerate(records)]
    all_rows = [record.csv_row(index) for record, index in indexed]
    _write_csv(destination / "all.csv", all_rows)
    for split in SPLIT_NAMES:
        rows = [
            record.csv_row(index)
            for record, index in indexed
            if record.split == split
        ]
        if not rows:
            raise ValueError(f"Deterministic BACE split is empty: {split}")
        if {int(row[LABEL_COL]) for row in rows} != {0, 1}:
            raise ValueError(
                f"Deterministic BACE split lacks both labels: split={split}; "
                "change --split-seed before freezing the dataset."
            )
        _write_csv(destination / f"{split}.csv", rows)
    graph_rows = [
        {
            "molecule_id": record.molecule_id,
            "smiles": record.smiles,
            "label": record.label,
            "split": record.split,
            "graph": record.graph.to_dict(),
            "source_graph_hash": record.source_graph_hash,
        }
        for record in records
    ]
    _atomic_write_text(
        destination / "graphs.jsonl",
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n"
            for row in graph_rows
        ),
    )
    split_counts = Counter(record.split for record in records)
    label_counts = Counter(record.label for record in records)
    atom_vocab = sorted({symbol for record in records for symbol in record.graph.atom_symbols})
    bond_vocab = sorted({value for record in records for value in record.graph.bond_types})
    summary = {
        "schema_version": "bace_processed_v1",
        "dataset": "BACE",
        "source_csv": str(Path(raw_csv).expanduser().resolve()),
        "source_csv_sha256": _file_sha256(Path(raw_csv).expanduser().resolve()),
        "smiles_col": SMILES_COL,
        "label_col": LABEL_COL,
        "num_graphs": len(records),
        "label_distribution": {str(key): int(label_counts[key]) for key in (0, 1)},
        "avg_atoms": sum(record.graph.num_nodes for record in records) / len(records),
        "avg_bonds": sum(record.graph.num_edges for record in records) / len(records),
        "invalid_smiles_count": len(source_audit["invalid_smiles"]),
        "duplicate_canonical_smiles_count": source_audit[
            "duplicate_canonical_smiles_count"
        ],
        "split_counts": {name: int(split_counts[name]) for name in SPLIT_NAMES},
        "split_seed": int(split_seed),
        "split_ratios": source_audit["split_ratios"],
        "atom_vocabulary": atom_vocab,
        "bond_type_vocabulary": bond_vocab,
        "molecule_ids_sha256": _stable_json_hash(
            [record.molecule_id for record in records]
        ),
        "dataset_fingerprint": _stable_json_hash(
            [
                {
                    "molecule_id": record.molecule_id,
                    "smiles": record.smiles,
                    "label": record.label,
                    "split": record.split,
                    "source_graph_hash": record.source_graph_hash,
                }
                for record in records
            ]
        ),
        "calibration_used_for_split_assignment_only": True,
        "test_used_for_split_assignment_only": True,
    }
    _atomic_write_text(
        destination / "bace_dataset_summary.json",
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )
    _atomic_write_text(
        destination / "invalid_smiles.json",
        json.dumps(
            source_audit["invalid_smiles"], indent=2, sort_keys=True, ensure_ascii=True
        )
        + "\n",
    )
    inventory = {
        path.name: {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}
        for path in sorted(destination.iterdir())
        if path.is_file()
    }
    manifest = {
        "schema_version": "bace_processed_manifest_v1",
        "dataset": "BACE",
        "dataset_fingerprint": summary["dataset_fingerprint"],
        "files": inventory,
        "source_audit": source_audit,
    }
    _atomic_write_text(
        destination / "run_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )
    return summary


__all__ = [
    "BACEGraph",
    "BACERecord",
    "DEFAULT_SPLIT_RATIOS",
    "LABEL_COL",
    "SMILES_COL",
    "SPLIT_NAMES",
    "graph_from_molecule",
    "load_bace_records",
    "prepare_bace_dataset",
]
