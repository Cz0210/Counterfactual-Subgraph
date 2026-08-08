"""Deterministic BBBP CSV adapter for project molecular graph workflows."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data.molecular_split import (
    DEFAULT_SPLIT_RATIOS,
    SPLIT_NAMES,
    audit_split_overlap,
    file_sha256,
    hashed_group_split,
    resolve_unique_column,
    stable_json_sha256,
    validate_split_ratios,
)
from src.data.scaffold_split import (
    DEFAULT_ACYCLIC_POLICY,
    bemis_murcko_scaffold,
)


SMILES_COL = "smiles"
LABEL_COL = "label"
SMILES_ALIASES = ("smiles", "mol", "canonical_smiles")
LABEL_ALIASES = ("p_np", "label", "target")


@dataclass(frozen=True, slots=True)
class BBBPGraph:
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
                "BBBP PyG conversion requires torch and torch_geometric."
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
class BBBPRecord:
    molecule_id: str
    raw_smiles: str
    canonical_smiles: str
    smiles: str
    label: int
    graph: BBBPGraph
    split: str
    raw_row_index: int
    scaffold_smiles: str
    source_graph_hash: str

    def csv_row(self, graph_index: int) -> dict[str, Any]:
        return {
            "molecule_id": self.molecule_id,
            "parent_id": self.molecule_id,
            "raw_smiles": self.raw_smiles,
            "canonical_smiles": self.canonical_smiles,
            SMILES_COL: self.smiles,
            LABEL_COL: int(self.label),
            "split": self.split,
            "raw_row_index": int(self.raw_row_index),
            "scaffold_smiles": self.scaffold_smiles,
            "num_atoms": self.graph.num_nodes,
            "num_bonds": self.graph.num_edges,
            "graph_index": int(graph_index),
            "source_graph_hash": self.source_graph_hash,
            "source_dataset": "BBBP",
        }


CSV_FIELDS = (
    "molecule_id",
    "parent_id",
    "raw_smiles",
    "canonical_smiles",
    SMILES_COL,
    LABEL_COL,
    "split",
    "raw_row_index",
    "scaffold_smiles",
    "num_atoms",
    "num_bonds",
    "graph_index",
    "source_graph_hash",
    "source_dataset",
)


def _require_rdkit() -> Any:
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - checked on HPC.
        raise RuntimeError("BBBP preparation requires RDKit.") from exc
    return Chem


def _bond_type_name(bond: Any) -> str:
    name = str(bond.GetBondType()).upper()
    return name if name in {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"} else "OTHER"


def _bond_type_index(value: str) -> int:
    return {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}.get(
        str(value).upper(), 4
    )


def graph_from_molecule(molecule: Any) -> BBBPGraph:
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
    return BBBPGraph(
        atom_features=atom_features,
        atom_symbols=atom_symbols,
        edge_index=(tuple(sources), tuple(targets)),
        bond_types=tuple(bond_types),
    )


def _parse_label(value: Any, *, row_number: int) -> int:
    text = str(value or "").strip()
    if text in {"0", "0.0"}:
        return 0
    if text in {"1", "1.0"}:
        return 1
    raise ValueError(f"BBBP label must be binary at CSV row {row_number}: {value!r}")


def _split_for_scaffold(
    scaffold: str,
    *,
    seed: int,
    split_ratios: Sequence[float],
) -> str:
    return hashed_group_split(scaffold, seed=seed, ratios=split_ratios)


def _validate_split_ratios(values: Sequence[float]) -> tuple[float, ...]:
    return validate_split_ratios(values)


def load_bbbp_records(
    raw_csv: str | Path,
    *,
    raw_smiles_col: str | None = None,
    raw_label_col: str | None = None,
    split_seed: int = 13,
    split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    acyclic_policy: str = DEFAULT_ACYCLIC_POLICY,
) -> tuple[list[BBBPRecord], dict[str, Any]]:
    """Normalize one raw BBBP CSV while preserving row and graph lineage."""

    Chem = _require_rdkit()
    source = Path(raw_csv).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"BBBP raw CSV does not exist: {source}")
    ratios = _validate_split_ratios(split_ratios)
    records: list[BBBPRecord] = []
    invalid_smiles: list[dict[str, Any]] = []
    duplicates = 0
    canonical_labels: dict[str, int] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        smiles_column = resolve_unique_column(
            fieldnames,
            explicit=raw_smiles_col,
            aliases=SMILES_ALIASES,
            role="SMILES",
        )
        label_column = resolve_unique_column(
            fieldnames,
            explicit=raw_label_col,
            aliases=LABEL_ALIASES,
            role="label",
        )
        raw_rows = 0
        for row_index, row in enumerate(reader):
            raw_rows += 1
            raw_smiles = str(row.get(smiles_column) or "").strip()
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
                        "raw_row_index": row_index,
                        "raw_smiles": raw_smiles,
                        "reason_code": failure,
                    }
                )
                continue
            label = _parse_label(row.get(label_column), row_number=row_index + 2)
            canonical = str(
                Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
            )
            if canonical in canonical_labels:
                if canonical_labels[canonical] != label:
                    raise ValueError(
                        "BBBP duplicate canonical SMILES has conflicting labels: "
                        f"{canonical!r}"
                    )
                duplicates += 1
                continue
            canonical_labels[canonical] = label
            scaffold, verified_canonical = bemis_murcko_scaffold(
                canonical,
                acyclic_policy=acyclic_policy,
            )
            if verified_canonical != canonical:
                raise AssertionError("BBBP canonicalization changed during scaffold audit.")
            split = _split_for_scaffold(
                scaffold, seed=split_seed, split_ratios=ratios
            )
            graph = graph_from_molecule(molecule)
            molecule_id = "BBBP_" + hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest()[:16]
            source_graph_hash = stable_json_sha256(graph.to_dict())
            records.append(
                BBBPRecord(
                    molecule_id=molecule_id,
                    raw_smiles=raw_smiles,
                    canonical_smiles=canonical,
                    smiles=canonical,
                    label=label,
                    graph=graph,
                    split=split,
                    raw_row_index=row_index,
                    scaffold_smiles=scaffold,
                    source_graph_hash=source_graph_hash,
                )
            )
    if not records:
        raise ValueError(f"No valid BBBP molecules remained after normalization: {source}")
    if len({record.molecule_id for record in records}) != len(records):
        raise AssertionError("Stable BBBP molecule IDs are not unique.")
    audit = {
        "raw_rows": raw_rows,
        "valid_unique_rows": len(records),
        "invalid_smiles": invalid_smiles,
        "duplicate_canonical_smiles_count": duplicates,
        "raw_smiles_col": smiles_column,
        "raw_label_col": label_column,
        "split_seed": int(split_seed),
        "split_ratios": dict(zip(SPLIT_NAMES, ratios, strict=True)),
        "split_strategy": "sha256_scaffold_group_v1",
        "acyclic_policy": acyclic_policy,
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


def prepare_bbbp_dataset(
    *,
    raw_csv: str | Path,
    output_dir: str | Path,
    raw_smiles_col: str | None = None,
    raw_label_col: str | None = None,
    split_seed: int = 13,
    split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    acyclic_policy: str = DEFAULT_ACYCLIC_POLICY,
) -> dict[str, Any]:
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"BBBP processed output is non-empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    records, source_audit = load_bbbp_records(
        raw_csv,
        raw_smiles_col=raw_smiles_col,
        raw_label_col=raw_label_col,
        split_seed=split_seed,
        split_ratios=split_ratios,
        acyclic_policy=acyclic_policy,
    )
    indexed = [(record, index) for index, record in enumerate(records)]
    all_rows = [record.csv_row(index) for record, index in indexed]
    _write_csv(destination / "all.csv", all_rows)
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in SPLIT_NAMES:
        rows = [
            record.csv_row(index)
            for record, index in indexed
            if record.split == split
        ]
        if not rows:
            raise ValueError(f"Deterministic BBBP split is empty: {split}")
        if {int(row[LABEL_COL]) for row in rows} != {0, 1}:
            raise ValueError(
                f"Deterministic BBBP split lacks both labels: split={split}; "
                "change --split-seed before freezing the dataset."
            )
        rows_by_split[split] = rows
        _write_csv(destination / f"{split}.csv", rows)
    leakage_audit = audit_split_overlap(
        rows_by_split,
        require_scaffold_disjoint=True,
    )
    graph_rows = [
        {
            "molecule_id": record.molecule_id,
            "raw_smiles": record.raw_smiles,
            "canonical_smiles": record.canonical_smiles,
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
        "schema_version": "bbbp_processed_v1",
        "dataset": "BBBP",
        "source_csv": str(Path(raw_csv).expanduser().resolve()),
        "source_csv_sha256": file_sha256(Path(raw_csv).expanduser().resolve()),
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
        "molecule_ids_sha256": stable_json_sha256(
            [record.molecule_id for record in records]
        ),
        "canonical_smiles_sha256": stable_json_sha256(
            [record.canonical_smiles for record in records]
        ),
        "scaffold_sha256": stable_json_sha256(
            [record.scaffold_smiles for record in records]
        ),
        "dataset_fingerprint": stable_json_sha256(
            [
                {
                    "molecule_id": record.molecule_id,
                    "canonical_smiles": record.canonical_smiles,
                    "label": record.label,
                    "split": record.split,
                    "source_graph_hash": record.source_graph_hash,
                }
                for record in records
            ]
        ),
        "calibration_used_for_split_assignment_only": True,
        "test_used_for_split_assignment_only": True,
        "source_dataset": "BBBP",
    }
    _atomic_write_text(
        destination / "bbbp_dataset_summary.json",
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )
    _atomic_write_text(
        destination / "invalid_smiles.json",
        json.dumps(
            source_audit["invalid_smiles"], indent=2, sort_keys=True, ensure_ascii=True
        )
        + "\n",
    )
    split_manifest = {
        "schema_version": "bbbp_split_manifest_v1",
        "dataset": "BBBP",
        "protocol": "main_scaffold_split_v1",
        "raw_sha256": summary["source_csv_sha256"],
        "split_seed": int(split_seed),
        "split_strategy": source_audit["split_strategy"],
        "acyclic_policy": source_audit["acyclic_policy"],
        "split_sizes": summary["split_counts"],
        "label_counts": {
            split: dict(Counter(str(row["label"]) for row in rows_by_split[split]))
            for split in SPLIT_NAMES
        },
        "molecule_ids_hash": summary["molecule_ids_sha256"],
        "canonical_smiles_hash": summary["canonical_smiles_sha256"],
        "scaffold_hash": summary["scaffold_sha256"],
        "candidate_source_splits": ["train", "val"],
        "selector_source_splits": ["calibration"],
        "threshold_source": "calibration",
        "test_usage": "final_evaluation_only",
    }
    _atomic_write_text(
        destination / "split_manifest.json",
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write_text(
        destination / "split_leakage_audit.json",
        json.dumps(leakage_audit, indent=2, sort_keys=True) + "\n",
    )
    inventory = {
        path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in sorted(destination.iterdir())
        if path.is_file()
    }
    manifest = {
        "schema_version": "bbbp_processed_manifest_v1",
        "dataset": "BBBP",
        "dataset_fingerprint": summary["dataset_fingerprint"],
        "files": inventory,
        "source_audit": source_audit,
        "split_manifest_sha256": file_sha256(destination / "split_manifest.json"),
        "split_leakage_audit_sha256": file_sha256(
            destination / "split_leakage_audit.json"
        ),
        "git_commit": _git_commit(),
    }
    _atomic_write_text(
        destination / "run_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )
    return summary


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def validate_bbbp_source(
    raw_csv: str | Path,
    *,
    raw_smiles_col: str | None = None,
    raw_label_col: str | None = None,
) -> dict[str, Any]:
    """Validate BBBP input identity without materializing processed outputs."""

    source = Path(raw_csv).expanduser().resolve()
    if not source.is_file():
        return {
            "status": "INPUT_REQUIRED",
            "dataset": "BBBP",
            "raw_csv": str(source),
            "input_exists": False,
            "formal_output_written": False,
        }
    records, audit = load_bbbp_records(
        source,
        raw_smiles_col=raw_smiles_col,
        raw_label_col=raw_label_col,
    )
    return {
        "status": "VALIDATED",
        "dataset": "BBBP",
        "raw_csv": str(source),
        "raw_sha256": file_sha256(source),
        "valid_unique_rows": len(records),
        "invalid_smiles_count": len(audit["invalid_smiles"]),
        "resolved_smiles_col": audit["raw_smiles_col"],
        "resolved_label_col": audit["raw_label_col"],
        "formal_output_written": False,
    }


__all__ = [
    "BBBPGraph",
    "BBBPRecord",
    "DEFAULT_SPLIT_RATIOS",
    "LABEL_COL",
    "SMILES_COL",
    "SPLIT_NAMES",
    "LABEL_ALIASES",
    "SMILES_ALIASES",
    "graph_from_molecule",
    "load_bbbp_records",
    "prepare_bbbp_dataset",
    "validate_bbbp_source",
]
