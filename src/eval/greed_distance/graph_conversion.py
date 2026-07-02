"""SMILES-to-graph conversion for GREED-style HIV distance training."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.utils.io import ensure_directory

try:  # pragma: no cover - runtime dependency
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None


def _parse_label(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def graph_from_smiles(smiles: str, graph_id: str = "", label: int | None = None) -> dict[str, Any]:
    """Convert one molecule SMILES into a lightweight labeled graph record."""

    record: dict[str, Any] = {
        "graph_id": str(graph_id),
        "smiles": str(smiles or "").strip(),
        "label": label,
        "nodes": [],
        "edges": [],
        "num_atoms": 0,
        "num_bonds": 0,
        "parse_ok": False,
        "error": None,
    }
    if Chem is None:
        record["error"] = "rdkit_unavailable"
        return record
    if not record["smiles"]:
        record["error"] = "empty_smiles"
        return record
    try:
        mol = Chem.MolFromSmiles(record["smiles"], sanitize=True)
    except Exception as exc:
        record["error"] = f"rdkit_parse_failed:{exc}"
        return record
    if mol is None:
        record["error"] = "rdkit_parse_failed"
        return record
    try:
        canonical = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        canonical = record["smiles"]
    nodes = [
        {
            "node_id": int(atom.GetIdx()),
            "atomic_num": int(atom.GetAtomicNum()),
            "formal_charge": int(atom.GetFormalCharge()),
            "is_aromatic": bool(atom.GetIsAromatic()),
        }
        for atom in mol.GetAtoms()
    ]
    edges = [
        {
            "source": int(bond.GetBeginAtomIdx()),
            "target": int(bond.GetEndAtomIdx()),
            "bond_type": str(bond.GetBondType()),
            "is_aromatic": bool(bond.GetIsAromatic()),
        }
        for bond in mol.GetBonds()
    ]
    record.update(
        {
            "smiles": canonical,
            "nodes": nodes,
            "edges": edges,
            "num_atoms": int(mol.GetNumAtoms()),
            "num_bonds": int(mol.GetNumBonds()),
            "parse_ok": True,
            "error": None,
        }
    )
    return record


def _resolve_label_col(rows: list[dict[str, Any]], requested: str) -> str:
    if not rows:
        return requested
    fields = set(rows[0])
    if requested in fields:
        return requested
    for fallback in ("label", "HIV_active", "target", "y", "activity"):
        if fallback in fields:
            return fallback
    return requested


def prepare_hiv_graph_dataset(
    *,
    dataset_csv: str | Path,
    output_jsonl: str | Path,
    smiles_col: str = "smiles",
    label_col: str = "label",
    label: int | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Read HIV CSV rows and write GREED graph records as JSONL."""

    source = Path(dataset_csv).expanduser().resolve()
    destination = Path(output_jsonl).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"dataset CSV not found: {source}")
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    actual_label_col = _resolve_label_col(rows, label_col)

    ensure_directory(destination.parent)
    total = 0
    written = 0
    parse_ok = 0
    with destination.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            row_label = _parse_label(row.get(actual_label_col))
            if label is not None and row_label != int(label):
                continue
            smiles = str(row.get(smiles_col) or "").strip()
            graph_id = str(row.get("graph_id") or row.get("id") or row.get("parent_id") or row_index)
            record = graph_from_smiles(smiles, graph_id=graph_id, label=row_label)
            record["source_row_index"] = row_index
            record["source_dataset_csv"] = str(source)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1
            written += 1
            parse_ok += int(bool(record.get("parse_ok")))
            if max_rows is not None and written >= int(max_rows):
                break
    return {
        "dataset_csv": str(source),
        "output_jsonl": str(destination),
        "smiles_col": smiles_col,
        "label_col": actual_label_col,
        "label_filter": label,
        "num_graphs": total,
        "num_parse_ok": parse_ok,
        "parse_ok_rate": (parse_ok / total) if total else 0.0,
    }


def read_graphs_jsonl(path: str | Path, *, parse_ok_only: bool = True) -> list[dict[str, Any]]:
    graphs: list[dict[str, Any]] = []
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if parse_ok_only and not record.get("parse_ok"):
                continue
            graphs.append(dict(record))
    return graphs
