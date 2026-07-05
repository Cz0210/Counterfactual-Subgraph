"""Utilities for adapting CLEAR full-graph exports to RF/SMILES evaluation.

The CLEAR AIDS graph tensors are dense graph arrays, not guaranteed chemical
graphs. This module therefore performs conservative graph-to-SMILES conversion.
For AIDS data prepared by ``prepare_clear_aids_dataset.py``, the atom identity
lives in the original graph descriptor slot ``features[:, 2] == atomic_num /
100``. CLEAR's decoded ``cf_x`` is a continuous reconstruction tensor, so it is
audited but not used as an atom vocabulary by default. The counterfactual graph
uses original node identities plus thresholded/symmetrized ``cf_adj`` topology.
Invalid graphs stay invalid with an explicit reason.
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
CLEAR_AIDS_ATOMIC_NUM_TO_SYMBOL = {6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl"}
CLEAR_AIDS_FEATURE_SCHEMA = (
    "descriptor_v1:[confounder,label_proxy,atomic_num/100,degree/6,formal_charge/3,"
    "is_aromatic,is_in_ring,mass/200,is_C,is_hetero,valence/8]"
)


@dataclass(frozen=True)
class GraphToSmilesResult:
    ok: bool
    smiles: str | None
    reason: str | None
    num_atoms: int | None
    num_bonds: int | None
    error: str | None = None
    node_mask_source: str | None = None
    num_nodes_used: int | None = None


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


def infer_num_nodes(record: dict[str, Any], x_field: str = "original_x", adj_field: str = "original_adj") -> tuple[int | None, str]:
    for field in ("original_num_nodes", "num_nodes", "source_num_nodes"):
        value = as_int(record.get(field))
        if value is not None and value > 0:
            return value, field
    x = record.get(x_field)
    cf_x = record.get("cf_x")
    if isinstance(x, list):
        active = [idx for idx, row in enumerate(x) if row_has_signal(row)]
        if active:
            return max(active) + 1, "nonzero_original_x_rows"
        if isinstance(cf_x, list):
            cf_active = [idx for idx, row in enumerate(cf_x) if row_has_signal(row)]
            if cf_active:
                return max(cf_active) + 1, "nonzero_cf_x_rows"
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
            return max(active_adj) + 1, f"{adj_field}_nonzero_topology"
    return None, "unresolved"


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


def argmax_index(row: Any) -> int | None:
    if not isinstance(row, list) or not row:
        return None
    best_index: int | None = None
    best_value: float | None = None
    for index, value in enumerate(row):
        number = as_float(value)
        if number is None:
            continue
        if best_value is None or number > best_value:
            best_index = index
            best_value = number
    return best_index


def is_onehot_like_row(row: Any, *, eps: float = 1e-3) -> bool:
    if not isinstance(row, list) or not row_has_signal(row):
        return False
    values = [as_float(value) for value in row]
    if any(value is None for value in values):
        return False
    clean = [float(value) for value in values if value is not None]
    near_one = sum(1 for value in clean if abs(value - 1.0) <= eps)
    near_zero = sum(1 for value in clean if abs(value) <= eps)
    return near_one == 1 and near_zero >= len(clean) - 1


def is_continuous_like_row(row: Any, *, eps: float = 1e-3) -> bool:
    if not isinstance(row, list) or not row_has_signal(row):
        return False
    values = [as_float(value) for value in row]
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return False
    non_binary = [value for value in clean if abs(value) > eps and abs(value - 1.0) > eps]
    return len(non_binary) > 0


def atomic_num_from_original_feature(row: Any, *, tolerance: float = 0.75) -> tuple[int | None, str | None]:
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
    argmax = argmax_index(row)
    return None, f"unknown_atom_type_idx:{argmax if argmax is not None else 'none'}"


def sym_adj_value(adj: Any, i: int, j: int) -> float | None:
    try:
        left = as_float(adj[i][j])
        right = as_float(adj[j][i])
    except Exception:
        return None
    if left is None and right is None:
        return None
    if left is None:
        return right
    if right is None:
        return left
    return float((left + right) / 2.0)


def adjacency_stats(adj: Any, *, num_nodes: int | None = None) -> dict[str, float | None]:
    if not isinstance(adj, list):
        return {"min": None, "max": None, "mean": None}
    n = int(num_nodes or len(adj))
    values: list[float] = []
    for i in range(min(n, len(adj))):
        row = adj[i]
        if not isinstance(row, list):
            continue
        for j in range(min(n, len(row))):
            if i == j:
                continue
            value = as_float(row[j])
            if value is not None:
                values.append(float(value))
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def feature_matrix_stats(x: Any, *, num_nodes: int | None = None) -> dict[str, Any]:
    rows = x if isinstance(x, list) else []
    n = min(int(num_nodes or len(rows)), len(rows))
    active_rows = [rows[index] for index in range(n) if row_has_signal(rows[index])]
    argmax_counts: dict[str, int] = {}
    for row in active_rows:
        index = argmax_index(row)
        key = str(index) if index is not None else "none"
        argmax_counts[key] = argmax_counts.get(key, 0) + 1
    onehot_count = sum(1 for row in active_rows if is_onehot_like_row(row))
    continuous_count = sum(1 for row in active_rows if is_continuous_like_row(row))
    return {
        "shape": [len(rows), len(rows[0]) if rows and isinstance(rows[0], list) else None],
        "active_rows": len(active_rows),
        "onehot_like_count": onehot_count,
        "onehot_like_rate": (onehot_count / len(active_rows)) if active_rows else 0.0,
        "continuous_count": continuous_count,
        "continuous_rate": (continuous_count / len(active_rows)) if active_rows else 0.0,
        "argmax_distribution": argmax_counts,
    }


def graph_arrays_to_smiles(
    *,
    x: Any,
    adj: Any,
    num_nodes: int | None,
    atom_tolerance: float = 0.75,
    adjacency_threshold: float = 0.5,
    node_mask_source: str | None = None,
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
        atomic_num, reason = atomic_num_from_original_feature(x[idx], tolerance=atom_tolerance)
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
            value = sym_adj_value(adj, i, j)
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
        return GraphToSmilesResult(False, None, "rdkit_sanitize_failed", n, num_bonds, str(exc), node_mask_source, n)
    if not smiles:
        return GraphToSmilesResult(False, None, "empty_smiles", n, num_bonds, None, node_mask_source, n)
    return GraphToSmilesResult(True, smiles, None, n, num_bonds, None, node_mask_source, n)


def convert_clear_record_graphs(record: dict[str, Any], *, adjacency_threshold: float = 0.5) -> dict[str, GraphToSmilesResult]:
    num_nodes, node_mask_source = infer_num_nodes(record, "original_x", "original_adj")
    return {
        "original": canonicalize_smiles(record.get("original_smiles"))
        if record.get("original_smiles")
        else graph_arrays_to_smiles(
            x=record.get("original_x"),
            adj=record.get("original_adj"),
            num_nodes=num_nodes,
            adjacency_threshold=adjacency_threshold,
            node_mask_source=node_mask_source,
        ),
        "cf": canonicalize_smiles(record.get("cf_smiles"))
        if record.get("cf_smiles")
        else graph_arrays_to_smiles(
            x=record.get("original_x"),
            adj=record.get("cf_adj"),
            num_nodes=num_nodes,
            adjacency_threshold=adjacency_threshold,
            node_mask_source=node_mask_source,
        ),
    }


def record_has_full_graph_fields(record: dict[str, Any]) -> bool:
    return all(field in record and record.get(field) not in (None, "") for field in FULL_GRAPH_FIELDS)


def analyze_clear_record_schema(record: dict[str, Any], *, adjacency_threshold: float = 0.5) -> dict[str, Any]:
    num_nodes, node_mask_source = infer_num_nodes(record, "original_x", "original_adj")
    original_x = feature_matrix_stats(record.get("original_x"), num_nodes=num_nodes)
    cf_x = feature_matrix_stats(record.get("cf_x"), num_nodes=num_nodes)
    cf_adj = adjacency_stats(record.get("cf_adj"), num_nodes=num_nodes)
    return {
        "feature_schema": CLEAR_AIDS_FEATURE_SCHEMA,
        "node_mask_source": node_mask_source,
        "num_nodes_used": num_nodes,
        "original_x": original_x,
        "cf_x": cf_x,
        "cf_adj_min": cf_adj["min"],
        "cf_adj_max": cf_adj["max"],
        "cf_adj_mean": cf_adj["mean"],
        "cf_adj_threshold": float(adjacency_threshold),
    }
