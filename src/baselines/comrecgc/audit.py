"""Data identity and final artifact gates for COMRECGC."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    ADAPTATION_MODE,
    CF_MODE,
    DISTANCE_LINE,
    METHOD,
    ContractError,
    stable_json_sha256,
)


def tensor_graph_fingerprint(graphs: Sequence[Any]) -> str:
    rows: list[dict[str, Any]] = []
    for index, graph in enumerate(graphs):
        edges = graph.edge_index.detach().cpu().tolist()
        features = graph.x.detach().cpu().tolist()
        label = getattr(graph, "y", -1)
        if hasattr(label, "item"):
            label = label.item()
        rows.append(
            {
                "index": index,
                "num_nodes": int(graph.num_nodes),
                "x": features,
                "edge_index": edges,
                "label": int(label),
            }
        )
    return stable_json_sha256(rows)


def official_dataset_audit(dataset: str, graphs: Sequence[Any]) -> dict[str, Any]:
    labels = Counter()
    node_feature_dim = 0
    edge_feature_dim = 0
    atom_indices: set[int] = set()
    bond_indices: set[int] = set()
    for graph in graphs:
        label = graph.y.item() if hasattr(graph.y, "item") else graph.y
        labels[int(label)] += 1
        node_feature_dim = int(graph.x.shape[1])
        atom_indices.update(int(value) for value in graph.x.argmax(dim=1).tolist())
        edge_attr = getattr(graph, "edge_attr", None)
        if edge_attr is not None:
            if getattr(edge_attr, "ndim", 0) == 2:
                edge_feature_dim = int(edge_attr.shape[1])
                bond_indices.update(int(value) for value in edge_attr.argmax(dim=1).tolist())
            else:
                edge_feature_dim = 1
                bond_indices.update(int(value) for value in edge_attr.tolist())
    return {
        "dataset": f"TU/{dataset}",
        "source": f"torch_geometric.datasets.TUDataset(name={dataset!r}) via pinned upstream data.py",
        "num_graphs": len(graphs),
        "num_label0": labels[0],
        "num_label1": labels[1],
        "node_feature_dim": node_feature_dim,
        "edge_feature_dim": edge_feature_dim,
        "atom_types": sorted(atom_indices),
        "bond_types": sorted(bond_indices),
        "label_semantics": "official COMRECGC internal source=0,target=1",
        "graph_id_source": "TUDataset processed order",
        "smiles_available": False,
        "dataset_fingerprint": tensor_graph_fingerprint(graphs),
        "eligible_for_project_figures": False,
    }


def validate_monotonic(values: Sequence[float], *, field: str) -> None:
    resolved = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0 for value in resolved):
        raise ContractError(f"{field} contains negative or non-finite values.")
    if any(right + 1e-12 < left for left, right in zip(resolved, resolved[1:])):
        raise ContractError(f"{field} is not monotonically non-decreasing.")


def validate_final_manifest(payload: Mapping[str, Any]) -> None:
    expected = {
        "method": METHOD,
        "cf_mode": CF_MODE,
        "distance_line": DISTANCE_LINE,
        "adaptation_mode": ADAPTATION_MODE,
        "candidate_set_preselected": True,
        "selection_performed_in_eval": False,
    }
    mismatches = {
        field: {"actual": payload.get(field), "expected": value}
        for field, value in expected.items()
        if payload.get(field) != value
    }
    if mismatches:
        raise ContractError(f"Final COMRECGC semantic gate failed: {mismatches}")
    for field in ("calibration_loaded", "test_used_for_selection", "threshold_fitted_on_test"):
        if payload.get(field) is True:
            raise ContractError(f"Final COMRECGC leakage gate failed: {field}=true")
