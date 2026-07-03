"""Adapters for GlobalGCE official outputs.

This module is intentionally independent from ``baselines/globalgce_official``.
It reads copied/exported artifacts and converts them into project-owned records
for unified baseline evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import pickle
import statistics
from pathlib import Path
from typing import Any, Iterable

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - depends on runtime environment.
    Chem = None

try:
    import networkx as nx
except ImportError:  # pragma: no cover - optional.
    nx = None


AIDS_NODE_LABEL_TO_ATOM = {
    0: "C",
    1: "O",
    2: "N",
    3: "Cl",
    4: "F",
    5: "S",
}
AIDS_EDGE_LABEL_TO_BOND_ORDER = {
    0: "single",
    1: "double",
    2: "triple",
}
AIDS_CLASS_LABEL_MAP = {
    0: "a",
    1: "i",
}


@dataclass
class ConversionResult:
    """Result of converting a GlobalGCE graph record to an RDKit molecule."""

    ok: bool
    mol: Any | None = None
    smiles: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    num_nodes: int = 0
    num_edges: int = 0
    invalid_reason: str | None = None

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "smiles": self.smiles,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "invalid_reason": self.invalid_reason,
        }


def map_aids_node_label_to_atom_symbol(label: int) -> str | None:
    """Map an AIDS raw node label to an atom symbol."""

    return AIDS_NODE_LABEL_TO_ATOM.get(int(label))


def map_aids_edge_label_to_bond_order(label: int) -> str | None:
    """Map an AIDS raw edge label to a bond-order name."""

    return AIDS_EDGE_LABEL_TO_BOND_ORDER.get(int(label))


def label_alignment_audit() -> dict[str, Any]:
    """Return the known GlobalGCE/AIDS label-alignment caveat."""

    return {
        "dataset": "AIDS",
        "node_label_map": AIDS_NODE_LABEL_TO_ATOM,
        "edge_label_map": AIDS_EDGE_LABEL_TO_BOND_ORDER,
        "class_label_map": AIDS_CLASS_LABEL_MAP,
        "globalgce_internal_node_shift": (
            "Official preprocessing stores molecule atom labels as raw_label + 1 "
            "and reserves 0 for padding."
        ),
        "globalgce_preprocess_label_flip": (
            "Official GlobalGCE preprocessing flips AIDS graph labels with y = 1 - y."
        ),
        "label_alignment_warning": (
            "Do not assume GlobalGCE internal label=1 is identical to the project "
            "label=1 without checking the preprocessing and split alignment."
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _load_pickle_or_torch(path: Path) -> Any:
    try:
        import torch  # type: ignore

        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
        except Exception:
            pass
    except Exception:
        pass

    with path.open("rb") as handle:
        return pickle.load(handle)


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(item) for item in shape]
    except Exception:
        return None


def _to_list(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_to_list(item) for item in value]
    if isinstance(value, list):
        return [_to_list(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_list(item) for key, item in value.items()}
    return value


def _len0(value: Any) -> int:
    if value is None:
        return 0
    shape = _shape(value)
    if shape:
        return int(shape[0])
    try:
        return len(value)
    except Exception:
        return 0


def _item_at(value: Any, index: int) -> Any:
    if value is None:
        return None
    try:
        return value[index]
    except Exception:
        return None


def _argmax(row: Any) -> int:
    values = _to_list(row)
    if isinstance(values, list):
        if not values:
            return 0
        if all(not isinstance(item, list) for item in values):
            max_index = 0
            max_value = float("-inf")
            for index, item in enumerate(values):
                try:
                    number = float(item)
                except Exception:
                    number = float("-inf")
                if number > max_value:
                    max_value = number
                    max_index = index
            return int(max_index)
    try:
        return int(float(values))
    except Exception:
        return 0


def _feature_to_internal_labels(feature: Any) -> list[int]:
    values = _to_list(feature)
    if not isinstance(values, list):
        return []
    labels: list[int] = []
    for row in values:
        if isinstance(row, list):
            labels.append(_argmax(row))
        else:
            try:
                labels.append(int(float(row)))
            except Exception:
                labels.append(0)
    return labels


def _threshold_adjacency(adjacency: Any) -> list[list[int]]:
    values = _to_list(adjacency)
    if not isinstance(values, list):
        return []
    matrix: list[list[int]] = []
    for row in values:
        if not isinstance(row, list):
            continue
        matrix.append([1 if _safe_float(item) > 0.5 else 0 for item in row])
    return matrix


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return 0.0
    return number if math.isfinite(number) else 0.0


def _edge_label_matrix(edge_attr: Any, n_nodes: int) -> list[list[int | None]]:
    matrix: list[list[int | None]] = [[None for _ in range(n_nodes)] for _ in range(n_nodes)]
    values = _to_list(edge_attr)
    if values is None:
        return matrix
    flat = values if isinstance(values, list) else []
    cursor = 0
    for row in range(1, n_nodes):
        for col in range(row):
            if cursor < len(flat):
                label = _argmax(flat[cursor])
                matrix[row][col] = label
                matrix[col][row] = label
            cursor += 1
    return matrix


def _active_nodes(adjacency: list[list[int]], internal_labels: list[int]) -> list[int]:
    nodes: list[int] = []
    for index, label in enumerate(internal_labels):
        degree = sum(adjacency[index]) if index < len(adjacency) else 0
        if int(label) > 0 or degree > 0:
            nodes.append(index)
    return nodes


def _bond_type_from_internal_edge_label(label: int | None) -> Any:
    if Chem is None:
        return None
    raw_label = int(label) - 1 if label is not None and int(label) > 0 else 0
    order = map_aids_edge_label_to_bond_order(raw_label)
    if order == "double":
        return Chem.BondType.DOUBLE
    if order == "triple":
        return Chem.BondType.TRIPLE
    return Chem.BondType.SINGLE


def _matrix_from_record(value: Any) -> list[list[Any]]:
    values = _to_list(value)
    if not isinstance(values, list):
        return []
    matrix: list[list[Any]] = []
    for row in values:
        if isinstance(row, list):
            matrix.append(row)
    return matrix


def _label_list_from_record(value: Any) -> list[Any]:
    values = _to_list(value)
    return values if isinstance(values, list) else []


def _is_present_symbol(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return text not in {"", "None", "none", "PAD", "pad", "0"}


def _node_symbol_from_record(record: dict[str, Any], node_index: int) -> tuple[str | None, str | None]:
    node_symbols = _label_list_from_record(record.get("node_symbols"))
    if node_index < len(node_symbols) and _is_present_symbol(node_symbols[node_index]):
        return str(node_symbols[node_index]), None

    raw_labels = _label_list_from_record(record.get("node_labels_aids_raw"))
    if node_index < len(raw_labels) and raw_labels[node_index] is not None:
        try:
            symbol = map_aids_node_label_to_atom_symbol(int(raw_labels[node_index]))
        except Exception:
            symbol = None
        if symbol is not None:
            return symbol, None
        return None, f"unknown_node_label:{raw_labels[node_index]}"

    internal_labels = _label_list_from_record(record.get("node_labels_internal") or record.get("node_labels"))
    if node_index < len(internal_labels) and internal_labels[node_index] is not None:
        try:
            internal = int(float(internal_labels[node_index]))
        except Exception:
            return None, f"unknown_node_label:{internal_labels[node_index]}"
        if internal <= 0:
            return None, "unknown_node_label:padding"
        raw_label = internal - 1
        symbol = map_aids_node_label_to_atom_symbol(raw_label)
        if symbol is not None:
            return symbol, None
        return None, f"unknown_node_label:{raw_label}"

    return None, "unknown_node_label:missing"


def _active_nodes_from_any_labels(record: dict[str, Any], adjacency: list[list[Any]]) -> list[int]:
    n_nodes = len(adjacency)
    node_symbols = _label_list_from_record(record.get("node_symbols"))
    raw_labels = _label_list_from_record(record.get("node_labels_aids_raw"))
    internal_labels = _label_list_from_record(record.get("node_labels_internal") or record.get("node_labels"))
    active: list[int] = []
    for index in range(n_nodes):
        degree = 0
        if index < len(adjacency):
            for value in adjacency[index]:
                if _safe_float(value) > 0.0:
                    degree += 1
        has_symbol = index < len(node_symbols) and _is_present_symbol(node_symbols[index])
        has_raw_label = index < len(raw_labels) and raw_labels[index] is not None
        has_internal_label = False
        if index < len(internal_labels):
            try:
                has_internal_label = int(float(internal_labels[index])) > 0
            except Exception:
                has_internal_label = False
        if has_symbol or has_raw_label or has_internal_label or degree > 0:
            active.append(index)
    return active


def _edge_label_value(record: dict[str, Any], source: int, target: int) -> tuple[int | None, str | None]:
    matrix = _matrix_from_record(record.get("edge_labels_internal_matrix") or record.get("edge_labels"))
    if not matrix:
        return None, None
    value = None
    if source < len(matrix) and isinstance(matrix[source], list) and target < len(matrix[source]):
        value = matrix[source][target]
    elif target < len(matrix) and isinstance(matrix[target], list) and source < len(matrix[target]):
        value = matrix[target][source]
    if value is None:
        return None, None
    try:
        label = int(float(value))
    except Exception:
        return None, f"unknown_edge_label:{value}"
    if label in {0, 1, 2}:
        return label, None
    if label in {3}:  # tolerate 1/2/3 encodings by mapping 3 to triple.
        return label - 1, None
    return None, f"unknown_edge_label:{label}"


def _bond_type_from_raw_edge_label(raw_label: int | None) -> Any:
    if Chem is None:
        return None
    if raw_label is None:
        raw_label = 0
    order = map_aids_edge_label_to_bond_order(int(raw_label))
    if order == "single":
        return Chem.BondType.SINGLE
    if order == "double":
        return Chem.BondType.DOUBLE
    if order == "triple":
        return Chem.BondType.TRIPLE
    return None


def globalgce_graph_record_to_mol(record: dict[str, Any]) -> ConversionResult:
    """Convert an exported GlobalGCE graph record to RDKit Mol and SMILES.

    The official AIDS preprocessing may store node labels as ``raw_label + 1``
    with 0 reserved for padding. This converter therefore prefers explicit
    ``node_symbols``, then raw AIDS labels, then shifted internal labels.
    """

    if Chem is None:
        return ConversionResult(
            ok=False,
            error_type="rdkit_unavailable",
            error_message="RDKit is not importable",
            invalid_reason="malformed_record",
        )

    adjacency = _matrix_from_record(record.get("adjacency") or record.get("adj") or record.get("cf_adj"))
    if not adjacency:
        return ConversionResult(
            ok=False,
            error_type="malformed_record",
            error_message="Missing adjacency matrix",
            invalid_reason="malformed_record",
        )

    active = _active_nodes_from_any_labels(record, adjacency)
    if not active:
        return ConversionResult(
            ok=False,
            error_type="empty_graph",
            error_message="No active atoms in graph record",
            num_nodes=0,
            num_edges=0,
            invalid_reason="empty_graph",
        )

    rw_mol = Chem.RWMol()
    node_to_mol: dict[int, int] = {}
    for node in active:
        symbol, error = _node_symbol_from_record(record, node)
        if error is not None or symbol is None:
            return ConversionResult(
                ok=False,
                error_type="unknown_node_label",
                error_message=error or f"Unknown label for node {node}",
                num_nodes=len(active),
                num_edges=0,
                invalid_reason="unknown_node_label",
            )
        try:
            node_to_mol[node] = rw_mol.AddAtom(Chem.Atom(symbol))
        except Exception as exc:
            return ConversionResult(
                ok=False,
                error_type="unknown_node_label",
                error_message=str(exc),
                num_nodes=len(active),
                num_edges=0,
                invalid_reason="unknown_node_label",
            )

    num_edges = 0
    for source_index, source in enumerate(active):
        for target in active[source_index + 1 :]:
            if source >= len(adjacency) or target >= len(adjacency[source]):
                continue
            if _safe_float(adjacency[source][target]) <= 0.0:
                continue
            raw_edge_label, edge_error = _edge_label_value(record, source, target)
            if edge_error is not None:
                return ConversionResult(
                    ok=False,
                    error_type="unknown_edge_label",
                    error_message=edge_error,
                    num_nodes=len(active),
                    num_edges=num_edges,
                    invalid_reason="unknown_edge_label",
                )
            bond_type = _bond_type_from_raw_edge_label(raw_edge_label)
            if bond_type is None:
                return ConversionResult(
                    ok=False,
                    error_type="unknown_edge_label",
                    error_message=f"Unsupported raw edge label: {raw_edge_label}",
                    num_nodes=len(active),
                    num_edges=num_edges,
                    invalid_reason="unknown_edge_label",
                )
            try:
                rw_mol.AddBond(node_to_mol[source], node_to_mol[target], bond_type)
                num_edges += 1
            except Exception as exc:
                return ConversionResult(
                    ok=False,
                    error_type="rdkit_add_bond_error",
                    error_message=str(exc),
                    num_nodes=len(active),
                    num_edges=num_edges,
                    invalid_reason="rdkit_add_bond_error",
                )

    mol = rw_mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        message = str(exc)
        invalid_reason = "valence_error" if "valence" in message.lower() else "sanitize_error"
        return ConversionResult(
            ok=False,
            mol=mol,
            error_type=invalid_reason,
            error_message=message,
            num_nodes=len(active),
            num_edges=num_edges,
            invalid_reason=invalid_reason,
        )

    smiles = Chem.MolToSmiles(mol, canonical=True)
    if not smiles:
        return ConversionResult(
            ok=False,
            mol=mol,
            error_type="empty_graph",
            error_message="RDKit returned empty SMILES",
            num_nodes=len(active),
            num_edges=num_edges,
            invalid_reason="empty_graph",
        )
    return ConversionResult(
        ok=True,
        mol=mol,
        smiles=smiles,
        num_nodes=len(active),
        num_edges=num_edges,
    )


def graph_record_to_smiles(record: dict[str, Any]) -> tuple[str | None, str | None]:
    """Convert an exported GlobalGCE graph record to a molecule SMILES if possible."""

    conversion = globalgce_graph_record_to_mol(record)
    if conversion.ok:
        return conversion.smiles, None
    return None, conversion.invalid_reason or conversion.error_type or conversion.error_message


def globalgce_cf_to_graph_record(cf: Any) -> dict[str, Any]:
    """Convert a raw or exported GlobalGCE CF object to a serializable graph record."""

    if isinstance(cf, dict) and "adjacency" in cf:
        record = dict(cf)
        if "smiles" not in record or not record.get("smiles"):
            smiles, error = graph_record_to_smiles(record)
            record["smiles"] = smiles
            record["smiles_error"] = error
        return record

    if not isinstance(cf, dict):
        raise TypeError(f"Expected CF dict-like object, got {type(cf).__name__}")

    feature = cf.get("cf_feat", cf.get("feature", cf.get("feat")))
    adjacency = cf.get("cf_adj", cf.get("adjacency", cf.get("adj")))
    edge_attr = cf.get("cf_edge", cf.get("edge_attr", cf.get("edge")))
    graph_idx = cf.get("graph_idx")
    cf_index = cf.get("cf_index")

    adjacency_matrix = _threshold_adjacency(adjacency)
    internal_labels = _feature_to_internal_labels(feature)
    edge_matrix = _edge_label_matrix(edge_attr, len(adjacency_matrix)) if edge_attr is not None else []
    raw_labels = [int(label) - 1 if int(label) > 0 else None for label in internal_labels]

    record = {
        "cf_index": cf_index,
        "graph_idx": int(graph_idx) if graph_idx is not None else None,
        "adjacency": adjacency_matrix,
        "node_labels_internal": internal_labels,
        "node_labels_aids_raw": raw_labels,
        "node_symbols": [
            map_aids_node_label_to_atom_symbol(label) if label is not None else None
            for label in raw_labels
        ],
        "edge_labels_internal_matrix": edge_matrix,
        "source_path": cf.get("source_path"),
    }
    smiles, error = graph_record_to_smiles(record)
    record["smiles"] = smiles
    record["smiles_error"] = error
    return record


def _raw_cfs_payload_to_records(payload: Any, source_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(payload, (list, tuple)) and len(payload) >= 4:
        feats, adjs, edges, graph_idxs = payload[:4]
        count = min(_len0(feats), _len0(adjs), _len0(graph_idxs))
        for index in range(count):
            row = {
                "cf_index": index,
                "graph_idx": _to_list(_item_at(graph_idxs, index)),
                "cf_feat": _item_at(feats, index),
                "cf_adj": _item_at(adjs, index),
                "cf_edge": _item_at(edges, index) if edges is not None else None,
                "source_path": str(source_path),
            }
            records.append(globalgce_cf_to_graph_record(row))
    return records


def load_globalgce_cfs(path: str | Path) -> list[dict[str, Any]]:
    """Load exported JSONL or raw GlobalGCE CF pickle/pt records."""

    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [globalgce_cf_to_graph_record(row) for row in _read_jsonl(path)]
    payload = _load_pickle_or_torch(path)
    return _raw_cfs_payload_to_records(payload, path)


def _make_rule_record(
    *,
    rule_id: int,
    source_path: Path,
    lhs_feat: Any,
    lhs_adj: Any,
    lhs_edge: Any,
    rhs_feat: Any,
    rhs_adj: Any,
    rhs_edge: Any,
) -> dict[str, Any]:
    lhs = globalgce_cf_to_graph_record(
        {
            "cf_index": rule_id,
            "cf_feat": lhs_feat,
            "cf_adj": lhs_adj,
            "cf_edge": lhs_edge,
            "source_path": str(source_path),
        }
    )
    rhs = globalgce_cf_to_graph_record(
        {
            "cf_index": rule_id,
            "cf_feat": rhs_feat,
            "cf_adj": rhs_adj,
            "cf_edge": rhs_edge,
            "source_path": str(source_path),
        }
    )
    return {
        "rule_id": rule_id,
        "source_path": str(source_path),
        "lhs": lhs,
        "rhs": rhs,
        "lhs_num_nodes": len(_active_nodes(lhs.get("adjacency") or [], lhs.get("node_labels_internal") or [])),
        "rhs_num_nodes": len(_active_nodes(rhs.get("adjacency") or [], rhs.get("node_labels_internal") or [])),
    }


def _rules_payload_to_records(payload: Any, source_path: Path) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    lhs_feat = payload.get("feat")
    lhs_adj = payload.get("adj")
    lhs_edge = payload.get("edge_attr")
    rhs_feat = payload.get("features_reconst")
    rhs_adj = payload.get("adj_reconst")
    rhs_edge = payload.get("edge_attrs_reconst")
    count = min(_len0(lhs_feat), _len0(lhs_adj), _len0(rhs_feat), _len0(rhs_adj))
    records: list[dict[str, Any]] = []
    for index in range(count):
        records.append(
            _make_rule_record(
                rule_id=index,
                source_path=source_path,
                lhs_feat=_item_at(lhs_feat, index),
                lhs_adj=_item_at(lhs_adj, index),
                lhs_edge=_item_at(lhs_edge, index) if lhs_edge is not None else None,
                rhs_feat=_item_at(rhs_feat, index),
                rhs_adj=_item_at(rhs_adj, index),
                rhs_edge=_item_at(rhs_edge, index) if rhs_edge is not None else None,
            )
        )
    return records


def load_globalgce_rules(path: str | Path) -> list[dict[str, Any]]:
    """Load exported JSONL rules or raw GlobalGCE rule pickle/pt files."""

    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    payload = _load_pickle_or_torch(path)
    return _rules_payload_to_records(payload, path)


def globalgce_rule_to_action(rule: dict[str, Any]) -> dict[str, Any]:
    """Convert a GlobalGCE rule record to a project action descriptor."""

    return {
        "method": "GlobalGCE",
        "action_type": "graph_rule_replacement",
        "rule_id": rule.get("rule_id"),
        "lhs": rule.get("lhs"),
        "rhs": rule.get("rhs"),
        "rule_action_supported": False,
        "unsupported_reason": (
            "Safe LHS->RHS graph replacement is not implemented in the project "
            "adapter yet; report SuppCov/StructRed/CovRed and use native-cf "
            "CCRCov for first-stage unified evaluation."
        ),
    }


def _graph_signature(record: dict[str, Any]) -> set[str]:
    adjacency = record.get("adjacency") or []
    labels = record.get("node_labels_aids_raw") or record.get("node_labels_internal") or []
    signature: set[str] = set()
    for index, label in enumerate(labels):
        if label is not None:
            signature.add(f"n:{label}")
    for row_index, row in enumerate(adjacency):
        if not isinstance(row, list):
            continue
        for col_index, value in enumerate(row[:row_index]):
            if int(value) > 0:
                label_a = labels[row_index] if row_index < len(labels) else None
                label_b = labels[col_index] if col_index < len(labels) else None
                if label_a is None or label_b is None:
                    continue
                signature.add(f"e:{min(label_a, label_b)}-{max(label_a, label_b)}")
    return signature


def _jaccard(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def compute_globalgce_structural_redundancy(
    rules: list[dict[str, Any]],
    metric: str = "tanimoto_or_graph",
) -> float | None:
    """Compute a simple graph-signature redundancy proxy for GlobalGCE rules."""

    if len(rules) < 2:
        return 0.0 if rules else None
    signatures = [
        _graph_signature(rule.get("rhs") or rule.get("lhs") or rule)
        for rule in rules
    ]
    scores: list[float] = []
    for left_index in range(len(signatures)):
        for right_index in range(left_index + 1, len(signatures)):
            scores.append(_jaccard(signatures[left_index], signatures[right_index]))
    return float(statistics.mean(scores)) if scores else None


def compute_globalgce_coverage_redundancy(rule_cover_sets: dict[Any, Iterable[Any]]) -> float | None:
    """Compute average pairwise Jaccard redundancy over rule coverage sets."""

    keys = list(rule_cover_sets.keys())
    if len(keys) < 2:
        return 0.0 if keys else None
    sets = {key: set(rule_cover_sets[key]) for key in keys}
    scores: list[float] = []
    for left_index in range(len(keys)):
        for right_index in range(left_index + 1, len(keys)):
            scores.append(_jaccard(sets[keys[left_index]], sets[keys[right_index]]))
    return float(statistics.mean(scores)) if scores else None


def load_globalgce_processed_graphs(dataset_dir: str | Path) -> list[dict[str, Any]]:
    """Load official GlobalGCE processed graph tensors from a copied run tree."""

    dataset_dir = Path(dataset_dir)
    required = ["feat.pkl", "adj.pkl", "label.pkl"]
    missing = [name for name in required if not (dataset_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing GlobalGCE processed files under {dataset_dir}: {missing}")

    with (dataset_dir / "feat.pkl").open("rb") as handle:
        feats = pickle.load(handle)
    with (dataset_dir / "adj.pkl").open("rb") as handle:
        adjs = pickle.load(handle)
    with (dataset_dir / "label.pkl").open("rb") as handle:
        labels = pickle.load(handle)
    edges = None
    if (dataset_dir / "edge.pkl").exists():
        with (dataset_dir / "edge.pkl").open("rb") as handle:
            edges = pickle.load(handle)

    count = min(_len0(feats), _len0(adjs), _len0(labels))
    records: list[dict[str, Any]] = []
    for index in range(count):
        row = globalgce_cf_to_graph_record(
            {
                "cf_index": index,
                "graph_idx": index,
                "cf_feat": _item_at(feats, index),
                "cf_adj": _item_at(adjs, index),
                "cf_edge": _item_at(edges, index) if edges is not None else None,
                "source_path": str(dataset_dir),
            }
        )
        label_value = _to_list(_item_at(labels, index))
        try:
            row["label"] = int(label_value)
        except Exception:
            row["label"] = None
        records.append(row)
    return records


def graph_record_to_networkx(record: dict[str, Any]) -> Any | None:
    """Convert a graph record to a NetworkX graph with node/edge labels."""

    if nx is None:
        return None
    adjacency = record.get("adjacency") or []
    labels = record.get("node_labels_aids_raw") or record.get("node_labels_internal") or []
    active = _active_nodes(adjacency, record.get("node_labels_internal") or labels)
    graph = nx.Graph()
    for node in active:
        graph.add_node(node, label=labels[node] if node < len(labels) else None)
    for left_index, left in enumerate(active):
        for right in active[left_index + 1 :]:
            if left < len(adjacency) and right < len(adjacency[left]) and int(adjacency[left][right]) > 0:
                graph.add_edge(left, right, label=1)
    return graph


__all__ = [
    "AIDS_CLASS_LABEL_MAP",
    "AIDS_EDGE_LABEL_TO_BOND_ORDER",
    "AIDS_NODE_LABEL_TO_ATOM",
    "ConversionResult",
    "compute_globalgce_coverage_redundancy",
    "compute_globalgce_structural_redundancy",
    "globalgce_cf_to_graph_record",
    "globalgce_graph_record_to_mol",
    "globalgce_rule_to_action",
    "graph_record_to_networkx",
    "graph_record_to_smiles",
    "label_alignment_audit",
    "load_globalgce_cfs",
    "load_globalgce_processed_graphs",
    "load_globalgce_rules",
    "map_aids_edge_label_to_bond_order",
    "map_aids_node_label_to_atom_symbol",
    "write_jsonl",
]
