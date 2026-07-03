#!/usr/bin/env python3
"""Prepare PyG graphs from the canonical AIDS/HIV CSV for GCF-HIVCSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "OTHER"]


def _load_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch_geometric.data import Data
        from rdkit import Chem

        return torch, Data, Chem
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("Preparing GCF HIVCSV graphs requires RDKit, torch, and torch_geometric.") from exc


def _parse_label(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def _bond_name(Chem: Any, bond: Any) -> str:
    bond_type = bond.GetBondType()
    if bond_type == Chem.BondType.SINGLE:
        return "SINGLE"
    if bond_type == Chem.BondType.DOUBLE:
        return "DOUBLE"
    if bond_type == Chem.BondType.TRIPLE:
        return "TRIPLE"
    if bond_type == Chem.BondType.AROMATIC:
        return "AROMATIC"
    return "OTHER"


def _one_hot(index: int, size: int) -> list[float]:
    row = [0.0] * size
    row[int(index)] = 1.0
    return row


def _read_rows(path: Path, max_rows: int | None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    return rows[: int(max_rows)] if max_rows is not None else rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--csv-path", default="data/raw/AIDS/HIV.csv")
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--label-col", default="HIV_active")
    parser.add_argument("--out-dir", default="outputs/hpc/gcfexplainer_hiv_csv/dataset")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    torch, Data, Chem = _load_runtime()
    csv_path = Path(args.csv_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[GCF_HIV_CSV_PREPARE_CONFIG]", flush=True)
    print("DATASET_SOURCE=HIV_CSV", flush=True)
    print("GCF_MODE=hiv_csv_adapted", flush=True)
    print("CF_MODE=strict_flip", flush=True)
    print(f"CSV_PATH={csv_path}", flush=True)
    print(f"SMILES_COL={args.smiles_col}", flush=True)
    print(f"LABEL_COL={args.label_col}", flush=True)

    rows = _read_rows(csv_path, args.max_rows)
    parsed: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    atom_symbols: set[str] = set()
    for raw_index, row in enumerate(rows):
        smiles = str(row.get(args.smiles_col) or "").strip()
        label = _parse_label(row.get(args.label_col))
        activity = str(row.get("activity") or "")
        if label is None:
            details.append(
                {
                    "raw_index": raw_index,
                    "smiles": smiles,
                    "activity": activity,
                    "label": "",
                    "conversion_ok": False,
                    "reason": "invalid_label",
                }
            )
            continue
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            details.append(
                {
                    "raw_index": raw_index,
                    "smiles": smiles,
                    "activity": activity,
                    "label": label,
                    "conversion_ok": False,
                    "reason": "rdkit_parse_failed",
                }
            )
            continue
        if mol.GetNumAtoms() <= 0:
            details.append(
                {
                    "raw_index": raw_index,
                    "smiles": smiles,
                    "activity": activity,
                    "label": label,
                    "conversion_ok": False,
                    "reason": "empty_molecule",
                }
            )
            continue
        for atom in mol.GetAtoms():
            atom_symbols.add(atom.GetSymbol())
        parsed.append({"raw_index": raw_index, "row": row, "smiles": smiles, "label": label, "activity": activity, "mol": mol})
        details.append(
            {
                "raw_index": raw_index,
                "smiles": smiles,
                "activity": activity,
                "label": label,
                "conversion_ok": True,
                "reason": "ok",
            }
        )

    atom_vocab = {symbol: idx for idx, symbol in enumerate(sorted(atom_symbols))}
    bond_vocab = {name: idx for idx, name in enumerate(BOND_TYPES)}
    graphs: list[Any] = []
    for item in parsed:
        mol = item["mol"]
        x = [_one_hot(atom_vocab[atom.GetSymbol()], len(atom_vocab)) for atom in mol.GetAtoms()]
        edge_pairs: list[list[int]] = []
        edge_attrs: list[list[float]] = []
        bond_type_names: list[str] = []
        for bond in mol.GetBonds():
            begin = int(bond.GetBeginAtomIdx())
            end = int(bond.GetEndAtomIdx())
            name = _bond_name(Chem, bond)
            attr = _one_hot(bond_vocab[name], len(bond_vocab))
            edge_pairs.append([begin, end])
            edge_pairs.append([end, begin])
            edge_attrs.append(attr)
            edge_attrs.append(attr)
            bond_type_names.append(name)
        if edge_pairs:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, len(bond_vocab)), dtype=torch.float32)
        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(int(item["label"]), dtype=torch.long),
            raw_index=torch.tensor(int(item["raw_index"]), dtype=torch.long),
            num_nodes=int(mol.GetNumAtoms()),
        )
        data.smiles = item["smiles"]
        data.activity = item["activity"]
        data.label = int(item["label"])
        data.atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        data.bond_types = bond_type_names
        graphs.append(data)

    torch.save(graphs, out_dir / "graphs.pt")
    (out_dir / "atom_vocab.json").write_text(json.dumps(atom_vocab, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "bond_vocab.json").write_text(json.dumps(bond_vocab, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (out_dir / "conversion_details.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["raw_index", "smiles", "activity", "label", "conversion_ok", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)
    label_counts = Counter(int(item["label"]) for item in parsed)
    activity_counts = Counter(str(item["activity"]) for item in parsed)
    summary = {
        "DATASET_SOURCE": "HIV_CSV",
        "GCF_MODE": "hiv_csv_adapted",
        "CF_MODE": "strict_flip",
        "csv_path": str(csv_path),
        "smiles_col": args.smiles_col,
        "label_col": args.label_col,
        "num_total": len(rows),
        "num_valid_smiles": len(graphs),
        "valid_rate": (len(graphs) / len(rows)) if rows else 0.0,
        "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "activity_counts": {str(k): int(v) for k, v in sorted(activity_counts.items())},
        "atom_vocab": atom_vocab,
        "bond_vocab": bond_vocab,
        "num_features": len(atom_vocab),
        "num_edge_features": len(bond_vocab),
        "num_classes": max(label_counts.keys()) + 1 if label_counts else 2,
        "graphs_pt": str(out_dir / "graphs.pt"),
        "seed": int(args.seed),
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[GCF_HIV_CSV_PREPARE_DONE]", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

