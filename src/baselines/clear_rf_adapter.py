"""Utilities for adapting CLEAR full-graph exports to RF/SMILES evaluation.

The CLEAR AIDS graph tensors are dense graph arrays, not guaranteed chemical
graphs. This module therefore performs conservative graph-to-SMILES conversion:
it recovers atom identity only from the CLEAR AIDS preparation feature slot
``features[:, 2] == atomic_num / 100`` and treats adjacency as single-bond
topology. Invalid graphs stay invalid with an explicit reason.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:  # pragma: no cover - runtime dependency
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


FULL_GRAPH_FIELDS = ("original_adj", "cf_adj", "original_x", "cf_x")
ALLOWED_AIDS_ATOMIC_NUMBERS = (6, 7, 8, 9, 16, 17)


@dataclass(frozen=True)
class GraphToSmilesResult:
    ok: bool
    smiles: str | None
    reason: str | None
    num_atoms: int | None
    num_bonds: int | None
    error: str | None = None


def read_jsonl(path: str | Path, *, max_records: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
            if max_records is not None and len(rows) >= int(max_records):
                break
    return rows


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "y"}:
            return True
        if token in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def matrix_shape(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, list) or not value:
        return None
    if not isinstance(value[0], list):
        return None
    return (len(value), len(value[0]))


def row_has_signal(row: Any, *, eps: float = 1e-9) -> bool:
    if not isinstance(row, list):
        return False
    for value in row:
        number = as_float(value)
        if number is not None and abs(number) > eps:
            return True
    return False


def infer_num_nodes(record: dict[str, Any], x_field: str, adj_field: str) -> int | None:
    for field in ("original_num_nodes", "num_nodes", "source_num_nodes"):
        value = as_int(record.get(field))
        if value is not None and value > 0:
            return value
    x = record.get(x_field)
    if isinstance(x, list):
        active = [idx for idx, row in enumerate(x) if row_has_signal(row)]
        if active:
            return max(active) + 1
    adj = record.get(adj_field)
    if isinstance(adj, list):
        active_adj: set[int] = set()
        for i, row in enumerate(adj):
            if not isinstance(row, list):
                continue
            for j, value in enumerate(row):
                if i != j and (as_float(value) or 0.0) > 0.5:
                    active_adj.add(i)
                    active_adj.add(j)
        if active_adj:
            return max(active_adj) + 1
    return None


def canonicalize_smiles(smiles: Any) -> GraphToSmilesResult:
    if Chem is None:
        return GraphToSmilesResult(False, None, "rdkit_unavailable", None, None)
    text = str(smiles or "").strip()
    if not text:
        return GraphToSmilesResult(False, None, "missing_smiles", None, None)
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return GraphToSmilesResult(False, None, "rdkit_parse_failed", None, None)
    try:
        Chem.SanitizeMol(mol)
        canonical = Chem.MolToSmiles(mol, canonical=True)
    except Exception as exc:  # noqa: BLE001
        return GraphToSmilesResult(False, None, "rdkit_sanitize_failed", None, None, str(exc))
    return GraphToSmilesResult(True, canonical, None, int(mol.GetNumAtoms()), int(mol.GetNumBonds()))


def atomic_num_from_feature(row: Any, *, tolerance: float = 0.75) -> tuple[int | None, str | None]:
    if not isinstance(row, list) or len(row) < 3:
        return None, "missing_atom_feature_slot"
    raw = as_float(row[2])
    if raw is None:
        return None, "non_numeric_atom_feature"
    scaled = raw * 100.0
    nearest = min(ALLOWED_AIDS_ATOMIC_NUMBERS, key=lambda item: abs(float(item) - scaled))
    if abs(float(nearest) - scaled) <= float(tolerance):
        return int(nearest), None
    rounded = int(round(scaled))
    if rounded in ALLOWED_AIDS_ATOMIC_NUMBERS:
        return int(rounded), None
    return None, f"unknown_atom_feature:{scaled:.4g}"


def graph_arrays_to_smiles(
    *,
    x: Any,
    adj: Any,
    num_nodes: int | None,
    atom_tolerance: float = 0.75,
    adjacency_threshold: float = 0.5,
) -> GraphToSmilesResult:
    if Chem is None:
        return GraphToSmilesResult(False, None, "rdkit_unavailable", None, None)
    if not isinstance(x, list) or not isinstance(adj, list):
        return GraphToSmilesResult(False, None, "missing_graph_arrays", None, None)
    shape = matrix_shape(adj)
    if shape is None:
        return GraphToSmilesResult(False, None, "malformed_adjacency", None, None)
    n = int(num_nodes or 0)
    n = min(n, len(x), shape[0], shape[1])
    if n <= 0:
        return GraphToSmilesResult(False, None, "empty_graph", None, None)

    rw_mol = Chem.RWMol()
    for idx in range(n):
        atomic_num, reason = atomic_num_from_feature(x[idx], tolerance=atom_tolerance)
        if atomic_num is None:
            return GraphToSmilesResult(False, None, reason or "unknown_atom_label", None, None)
        try:
            rw_mol.AddAtom(Chem.Atom(int(atomic_num)))
        except Exception as exc:  # noqa: BLE001
            return GraphToSmilesResult(False, None, "rdkit_add_atom_failed", None, None, str(exc))

    num_bonds = 0
    for i in range(n):
        row = adj[i]
        if not isinstance(row, list):
            return GraphToSmilesResult(False, None, "malformed_adjacency_row", None, None)
        for j in range(i + 1, n):
            value = as_float(row[j])
            if value is None or value <= float(adjacency_threshold):
                continue
            try:
                rw_mol.AddBond(int(i), int(j), Chem.BondType.SINGLE)
                num_bonds += 1
            except Exception as exc:  # noqa: BLE001
                return GraphToSmilesResult(False, None, "rdkit_add_bond_failed", None, None, str(exc))

    try:
        mol = rw_mol.GetMol()
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol, canonical=True)
    except Exception as exc:  # noqa: BLE001
        return GraphToSmilesResult(False, None, "rdkit_sanitize_failed", n, num_bonds, str(exc))
    if not smiles:
        return GraphToSmilesResult(False, None, "empty_smiles", n, num_bonds)
    return GraphToSmilesResult(True, smiles, None, n, num_bonds)


def convert_clear_record_graphs(record: dict[str, Any]) -> dict[str, GraphToSmilesResult]:
    num_nodes = infer_num_nodes(record, "original_x", "original_adj")
    return {
        "original": canonicalize_smiles(record.get("original_smiles"))
        if record.get("original_smiles")
        else graph_arrays_to_smiles(
            x=record.get("original_x"),
            adj=record.get("original_adj"),
            num_nodes=num_nodes,
        ),
        "cf": canonicalize_smiles(record.get("cf_smiles"))
        if record.get("cf_smiles")
        else graph_arrays_to_smiles(
            x=record.get("cf_x"),
            adj=record.get("cf_adj"),
            num_nodes=num_nodes,
        ),
    }


def record_has_full_graph_fields(record: dict[str, Any]) -> bool:
    return all(field in record and record.get(field) not in (None, "") for field in FULL_GRAPH_FIELDS)

