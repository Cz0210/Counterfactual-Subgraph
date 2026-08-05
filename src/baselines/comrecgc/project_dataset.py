"""Frozen project datasets exposed through the graph interface COMRECGC needs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import ContractError, ordered_ids_sha256, sha256_file, stable_json_sha256


@dataclass(frozen=True)
class ProjectDatasetBundle:
    dataset: str
    graphs: list[Any]
    parent_ids: list[str]
    source_label: int
    target_label: int
    node_feature_dim: int
    atom_vocabulary: tuple[str | int, ...]
    dataset_source: str
    source_files: tuple[str, ...]
    dataset_fingerprint: str
    generation_source_parent_rows: int
    calibration_loaded: bool = False
    test_loaded: bool = False

    def audit(self) -> dict[str, Any]:
        labels = Counter(int(_scalar(getattr(graph, "y", -1))) for graph in self.graphs)
        edge_dim = 0
        if self.graphs:
            edge_attr = getattr(self.graphs[0], "edge_attr", None)
            if edge_attr is not None and getattr(edge_attr, "ndim", 0) == 2:
                edge_dim = int(edge_attr.shape[1])
        return {
            "dataset": self.dataset,
            "dataset_source": self.dataset_source,
            "num_graphs": len(self.graphs),
            "num_label0": int(labels.get(0, 0)),
            "num_label1": int(labels.get(1, 0)),
            "node_feature_dim": self.node_feature_dim,
            "edge_feature_dim": edge_dim,
            "atom_types": list(self.atom_vocabulary),
            "label_semantics": {
                "project_source_label": self.source_label,
                "project_target_label": self.target_label,
                "comrecgc_source_internal_label": 0,
                "comrecgc_target_internal_label": 1,
            },
            "graph_id_source": "frozen_project_artifact",
            "smiles_available": all(bool(getattr(graph, "smiles", "")) for graph in self.graphs),
            "dataset_fingerprint": self.dataset_fingerprint,
            "generation_source_parent_rows": self.generation_source_parent_rows,
            "generation_parent_ids_sha256": ordered_ids_sha256(self.parent_ids),
            "source_files": list(self.source_files),
            "calibration_loaded": self.calibration_loaded,
            "test_loaded": self.test_loaded,
        }


class GraphListDataset:
    """Small indexable dataset compatible with upstream list-style access."""

    def __init__(self, graphs: Sequence[Any], num_features: int) -> None:
        self.graphs = list(graphs)
        self.num_features = int(num_features)
        self.num_classes = 2

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return self.graphs[index]
        if isinstance(index, (list, tuple)):
            return [self.graphs[int(value)] for value in index]
        if hasattr(index, "detach"):
            index = index.detach().cpu()
        if hasattr(index, "ndim") and int(index.ndim) > 0:
            return [self.graphs[int(value)] for value in index.tolist()]
        if hasattr(index, "item"):
            index = index.item()
        return self.graphs[int(index)]


def project_label_to_internal(project_label: int, *, source_label: int = 1, target_label: int = 0) -> int:
    """Map project direction to COMRECGC's source=0,target=1 convention."""

    value = int(project_label)
    if value == int(source_label):
        return 0
    if value == int(target_label):
        return 1
    raise ContractError(
        f"Project label {value} is outside the frozen source/target direction "
        f"{source_label}->{target_label}."
    )


def _scalar(value: Any) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"Expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _torch_stack() -> tuple[Any, Any]:
    try:
        import torch
        from torch_geometric.data import Data
    except Exception as exc:  # pragma: no cover - HPC runtime dependency
        raise RuntimeError(
            "COMRECGC project dataset loading requires torch and torch_geometric."
        ) from exc
    return torch, Data


def _attach_lineage(
    graph: Any,
    *,
    parent_id: str,
    source_index: int,
    smiles: str,
    project_label: int,
) -> Any:
    torch, _Data = _torch_stack()
    cloned = graph.clone()
    cloned.comrecgc_parent_id = str(parent_id)
    cloned.comrecgc_source_index = int(source_index)
    cloned.comrecgc_source_smiles = str(smiles)
    cloned.comrecgc_project_label = int(project_label)
    cloned.comrecgc_node_origin = torch.arange(int(cloned.num_nodes), dtype=torch.long)
    return cloned


def load_aids_generation_bundle(
    *,
    dataset_dir: str | Path,
    source_csv: str | Path,
    parent_limit: int | None = None,
) -> ProjectDatasetBundle:
    """Load project HIV graphs corresponding exactly to the frozen source CSV.

    The CSV order is authoritative. Exact SMILES matching is used first, with a
    canonical-SMILES fallback that must remain unique.
    """

    from src.baselines.gcf_hiv_csv_dataset import HIVCSVGraphDataset

    root = Path(dataset_dir).expanduser().resolve()
    source = Path(source_csv).expanduser().resolve()
    summary = _load_json(root / "dataset_summary.json")
    if str(summary.get("DATASET_SOURCE")) != "HIV_CSV":
        raise ContractError("AIDS adapter requires the project HIV.csv graph artifact.")
    rows = _read_csv(source)
    if parent_limit is not None:
        rows = rows[: int(parent_limit)]
    if not rows or any(int(row.get("label", -1)) != 1 for row in rows):
        raise ContractError("AIDS generation CSV must contain only source label 1 rows.")
    dataset = HIVCSVGraphDataset(root)
    exact: dict[str, list[tuple[int, Any]]] = {}
    for index, graph in enumerate(dataset.graphs):
        exact.setdefault(str(getattr(graph, "smiles", "")), []).append((index, graph))

    canonical_cache: dict[str, list[tuple[int, Any]]] | None = None
    used_graph_indices: set[int] = set()
    selected: list[Any] = []
    parent_ids: list[str] = []
    for row_index, row in enumerate(rows):
        smiles = str(row.get("smiles") or "").strip()
        matches = [item for item in exact.get(smiles, []) if item[0] not in used_graph_indices]
        selected_match: tuple[int, Any] | None = None
        if matches:
            selected_match = matches[0]
        if selected_match is None:
            if canonical_cache is None:
                try:
                    from rdkit import Chem
                except ImportError as exc:  # pragma: no cover
                    raise RuntimeError("RDKit is required for canonical AIDS matching.") from exc
                canonical_cache = {}
                for graph_index, graph in enumerate(dataset.graphs):
                    molecule = Chem.MolFromSmiles(str(getattr(graph, "smiles", "")))
                    if molecule is not None:
                        canonical = Chem.MolToSmiles(molecule, canonical=True)
                        canonical_cache.setdefault(canonical, []).append((graph_index, graph))
            molecule = Chem.MolFromSmiles(smiles)
            canonical = Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else ""
            matches = [
                item
                for item in canonical_cache.get(canonical, [])
                if item[0] not in used_graph_indices
            ]
            if matches:
                selected_match = matches[0]
        if selected_match is None:
            raise ContractError(
                f"AIDS source row {row_index} does not map uniquely to graphs.pt: "
                f"matches={len(matches)}, smiles={smiles!r}"
            )
        graph_index, graph = selected_match
        used_graph_indices.add(graph_index)
        parent_id = f"AIDS_HIV_{row_index:06d}"
        selected.append(
            _attach_lineage(
                graph,
                parent_id=parent_id,
                source_index=graph_index,
                smiles=smiles,
                project_label=1,
            )
        )
        parent_ids.append(parent_id)
    atom_vocab_raw = summary.get("atom_vocab") or {}
    atom_vocab = tuple(
        symbol for symbol, _index in sorted(atom_vocab_raw.items(), key=lambda item: int(item[1]))
    )
    fingerprint_payload = {
        "graphs_sha256": sha256_file(root / "graphs.pt"),
        "summary_sha256": sha256_file(root / "dataset_summary.json"),
        "source_csv_sha256": sha256_file(source),
        "parent_ids": parent_ids,
        "smiles": [str(row["smiles"]) for row in rows],
    }
    return ProjectDatasetBundle(
        dataset="AIDS/HIV",
        graphs=selected,
        parent_ids=parent_ids,
        source_label=1,
        target_label=0,
        node_feature_dim=int(summary["num_features"]),
        atom_vocabulary=atom_vocab,
        dataset_source="data/raw/AIDS/HIV.csv via frozen graphs.pt and source CSV",
        source_files=(str(root / "graphs.pt"), str(root / "dataset_summary.json"), str(source)),
        dataset_fingerprint=stable_json_sha256(fingerprint_payload),
        generation_source_parent_rows=len(selected),
    )


def load_mutagenicity_generation_bundle(
    *, dataset_dir: str | Path,
    parent_limit: int | None = None,
) -> ProjectDatasetBundle:
    from src.baselines.gcfexplainer_mutagenicity_adapter import (
        load_dataset_artifacts,
        record_to_pyg,
    )

    root = Path(dataset_dir).expanduser().resolve()
    schema, _train, _val, generation, summary = load_dataset_artifacts(root)
    selected_records = sorted(generation, key=lambda row: str(row["molecule_id"]))
    if parent_limit is not None:
        selected_records = selected_records[: int(parent_limit)]
    graphs = [
        record_to_pyg(record, origin_index=index)
        for index, record in enumerate(selected_records)
    ]
    parent_ids = [str(record["molecule_id"]) for record in selected_records]
    for index, (graph, record) in enumerate(zip(graphs, selected_records, strict=True)):
        graph.comrecgc_parent_id = parent_ids[index]
        graph.comrecgc_source_index = index
        graph.comrecgc_source_smiles = str(record["canonical_smiles"])
        graph.comrecgc_project_label = 1
        graph.comrecgc_source_record = record
        if not hasattr(graph, "comrecgc_node_origin"):
            graph.comrecgc_node_origin = getattr(graph, "gcf_node_origin")
    fingerprint_payload = {
        "dataset_summary_sha256": sha256_file(root / "dataset_summary.json"),
        "generation_sha256": sha256_file(root / "generation_source_graphs.pt"),
        "parent_ids": parent_ids,
        "source_graph_hashes": [str(record["source_graph_hash"]) for record in selected_records],
    }
    return ProjectDatasetBundle(
        dataset="Mutagenicity",
        graphs=graphs,
        parent_ids=parent_ids,
        source_label=1,
        target_label=0,
        node_feature_dim=schema.node_feature_dim,
        atom_vocabulary=tuple(schema.atom_vocabulary),
        dataset_source="mutagenicity_v1_teacher_consistent strict train-source",
        source_files=(
            str(root / "dataset_summary.json"),
            str(root / "generation_source_graphs.pt"),
        ),
        dataset_fingerprint=stable_json_sha256(fingerprint_payload),
        generation_source_parent_rows=len(generation),
    )


def verify_evaluation_parent_ids(
    path: str | Path,
    *,
    expected_count: int,
    id_field: str,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    rows = _read_csv(source)
    values = [str(row.get(id_field) or "").strip() for row in rows]
    if len(values) != expected_count or any(not value for value in values):
        raise ContractError(
            f"Evaluation parent contract mismatch for {source}: "
            f"count={len(values)}, expected={expected_count}."
        )
    if len(set(values)) != len(values):
        raise ContractError(f"Evaluation parent IDs are not unique: {source}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "count": len(values),
        "id_field": id_field,
        "ordered_ids_sha256": ordered_ids_sha256(values),
    }


def graph_ids(graphs: Iterable[Any]) -> list[str]:
    return [str(getattr(graph, "comrecgc_parent_id")) for graph in graphs]
