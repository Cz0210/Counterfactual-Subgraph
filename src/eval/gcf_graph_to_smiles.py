"""Conservative graph-to-SMILES conversion for official GCFExplainer graphs."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


AIDS_NODE_SYMBOLS_6 = {0: "C", 1: "O", 2: "N", 3: "Cl", 4: "F", 5: "S"}


@dataclass
class GCFGraphSmilesResult:
    candidate_id: str
    graph_hash: str
    smiles: str
    convert_ok: bool
    sanitize_ok: bool
    reason: str


def _load_torch() -> Any:
    try:
        import torch

        return torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("GCF graph-to-SMILES conversion requires PyTorch to read selected .pt files.") from exc


def _stable_hash(value: Any) -> str:
    text = repr(value)
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _load_graph_payload(path: str | Path) -> tuple[list[Any], list[dict[str, Any]]]:
    torch = _load_torch()
    payload = torch.load(Path(path).expanduser().resolve(), map_location="cpu")
    if isinstance(payload, dict):
        graphs = list(payload.get("selected_graphs") or payload.get("graphs") or [])
        records = [dict(row) for row in payload.get("selected_records") or [] if isinstance(row, dict)]
        return graphs, records
    if isinstance(payload, list):
        return payload, []
    raise TypeError(f"Unsupported graph payload type: {type(payload).__name__}")


def _tensor_to_list(value: Any) -> list[Any]:
    torch = _load_torch()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return list(value)


def _graph_to_mol(graph: Any) -> tuple[Any | None, str]:
    try:
        from rdkit import Chem
    except Exception:
        return None, "rdkit_unavailable"

    x = getattr(graph, "x", None)
    edge_index = getattr(graph, "edge_index", None)
    edge_attr = getattr(graph, "edge_attr", None)
    if x is None or edge_index is None:
        return None, "missing_atom_or_bond_mapping"

    try:
        x_list = _tensor_to_list(x)
    except Exception:
        return None, "missing_atom_or_bond_mapping"
    if not x_list:
        return None, "empty_graph"
    if not isinstance(x_list[0], list):
        return None, "missing_atom_or_bond_mapping"
    feature_dim = len(x_list[0])
    if feature_dim != 6:
        # Official AIDS processed graphs usually use 9 frequent node labels.
        # Without the original TUDataset label vocabulary, mapping them to
        # atom symbols would be guesswork, so we fail explicitly.
        return None, "missing_atom_or_bond_mapping"

    mol = Chem.RWMol()
    try:
        for row in x_list:
            label = max(range(len(row)), key=lambda idx: float(row[idx]))
            symbol = AIDS_NODE_SYMBOLS_6.get(int(label))
            if symbol is None:
                return None, "missing_atom_or_bond_mapping"
            mol.AddAtom(Chem.Atom(symbol))
    except Exception:
        return None, "missing_atom_or_bond_mapping"

    try:
        edges = _tensor_to_list(edge_index)
        srcs, dsts = edges[0], edges[1]
    except Exception:
        return None, "missing_atom_or_bond_mapping"
    if edge_attr is None:
        return None, "missing_atom_or_bond_mapping"
    try:
        attrs = _tensor_to_list(edge_attr)
    except Exception:
        return None, "missing_atom_or_bond_mapping"

    bond_map = {
        0: Chem.BondType.SINGLE,
        1: Chem.BondType.DOUBLE,
        2: Chem.BondType.TRIPLE,
    }
    added: set[tuple[int, int]] = set()
    for edge_pos, (src, dst) in enumerate(zip(srcs, dsts)):
        i, j = int(src), int(dst)
        if i == j:
            continue
        a, b = sorted((i, j))
        if (a, b) in added:
            continue
        label_value = attrs[edge_pos] if edge_pos < len(attrs) else None
        if isinstance(label_value, list):
            label = max(range(len(label_value)), key=lambda idx: float(label_value[idx]))
        else:
            try:
                label = int(label_value)
            except Exception:
                return None, "missing_atom_or_bond_mapping"
        bond_type = bond_map.get(label)
        if bond_type is None:
            return None, "missing_atom_or_bond_mapping"
        try:
            mol.AddBond(a, b, bond_type)
        except Exception:
            return None, "rdkit_add_bond_failed"
        added.add((a, b))
    return mol.GetMol(), "ok"


def convert_one_graph(graph: Any, *, candidate_id: str, graph_hash: str) -> GCFGraphSmilesResult:
    try:
        from rdkit import Chem
    except Exception:
        return GCFGraphSmilesResult(candidate_id, graph_hash, "", False, False, "rdkit_unavailable")

    mol, reason = _graph_to_mol(graph)
    if mol is None:
        return GCFGraphSmilesResult(candidate_id, graph_hash, "", False, False, reason)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return GCFGraphSmilesResult(candidate_id, graph_hash, "", True, False, "rdkit_sanitize_failed")
    smiles = Chem.MolToSmiles(mol, canonical=True)
    return GCFGraphSmilesResult(candidate_id, graph_hash, smiles, True, True, "ok")


def convert_selected_graphs_to_smiles(
    *,
    selected_graphs_path: str | Path,
    out_csv: str | Path,
    out_report: str | Path,
) -> dict[str, Any]:
    graphs, records = _load_graph_payload(selected_graphs_path)
    rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    for idx, graph in enumerate(graphs):
        record = records[idx] if idx < len(records) else {}
        candidate_id = str(record.get("candidate_id") or f"gcf_official_{idx}")
        graph_hash = str(record.get("graph_hash") or _stable_hash(graph))
        result = convert_one_graph(graph, candidate_id=candidate_id, graph_hash=graph_hash)
        row = asdict(result)
        rows.append(row)
        reason_counts[result.reason] = reason_counts.get(result.reason, 0) + 1

    out_csv_path = Path(out_csv).expanduser().resolve()
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate_id", "graph_hash", "smiles", "convert_ok", "sanitize_ok", "reason"],
        )
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    convert_ok = sum(1 for row in rows if row["convert_ok"])
    sanitize_ok = sum(1 for row in rows if row["sanitize_ok"])
    report = {
        "selected_graphs_path": str(Path(selected_graphs_path).expanduser().resolve()),
        "out_csv": str(out_csv_path),
        "num_candidates": total,
        "convert_ok": convert_ok,
        "sanitize_ok": sanitize_ok,
        "smiles_convert_ok_rate": convert_ok / total if total else 0.0,
        "sanitize_ok_rate": sanitize_ok / total if total else 0.0,
        "reason_counts": reason_counts,
    }
    out_report_path = Path(out_report).expanduser().resolve()
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    out_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report

