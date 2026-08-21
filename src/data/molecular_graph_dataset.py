"""Dataset and batching contracts for task-specific molecular classifiers."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from src.data.molecular_graph_featurizer import (
    MolecularFeatureSchema,
    MolecularGraphFeatures,
    MolecularGraphFeaturizer,
    default_molecular_feature_schema,
)


SMILES_COLUMN_CANDIDATES = (
    "model_smiles",
    "canonical_smiles",
    "smiles",
    "PROCESSED_SMILES",
)
ID_COLUMN_CANDIDATES = (
    "molecule_id",
    "compound_id",
    "COMPOUND_ID",
    "id",
)
LABEL_COLUMN_CANDIDATES = ("label", "target", "TARGET")
SPLIT_ALIASES = {"validation": "val", "valid": "val", "val": "val"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_split(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    return SPLIT_ALIASES.get(normalized, normalized)


def _resolve_column(
    fieldnames: Sequence[str],
    explicit: str | None,
    candidates: Sequence[str],
    *,
    role: str,
    required: bool = True,
) -> str | None:
    available = set(fieldnames)
    if explicit:
        if explicit not in available:
            raise ValueError(f"Configured {role} column {explicit!r} is absent.")
        return explicit
    matches = [candidate for candidate in candidates if candidate in available]
    if not matches:
        if required:
            raise ValueError(
                f"Could not detect {role} column. Available columns: {fieldnames}"
            )
        return None
    return matches[0]


def _stratified_limit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    label_column: str,
    num_classes: int,
    limit: int,
) -> list[Mapping[str, Any]]:
    """Select a deterministic class-aware smoke subset.

    Molecular split exports may be grouped by label.  Taking a raw prefix can
    therefore turn an otherwise valid binary or multiclass split into a
    single-class smoke dataset.  Round-robin selection retains source order
    within each class while guaranteeing coverage of every registered class.
    """

    if limit < num_classes:
        raise ValueError(
            "A stratified dataset limit must be at least the number of classes: "
            f"limit={limit}, num_classes={num_classes}"
        )
    buckets: dict[int, list[Mapping[str, Any]]] = {
        label: [] for label in range(num_classes)
    }
    for row in rows:
        try:
            label = int(str(row.get(label_column)).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid molecular label while building stratified subset: "
                f"{row.get(label_column)!r}"
            ) from exc
        if label not in buckets:
            raise ValueError(
                f"Molecular label {label} falls outside [0, {num_classes - 1}]."
            )
        buckets[label].append(row)
    missing = [label for label, values in buckets.items() if not values]
    if missing:
        raise ValueError(
            "A stratified smoke subset requires every class in the source split: "
            f"missing={missing}"
        )

    selected: list[Mapping[str, Any]] = []
    offsets = {label: 0 for label in buckets}
    while len(selected) < min(limit, len(rows)):
        progressed = False
        for label in range(num_classes):
            offset = offsets[label]
            if offset >= len(buckets[label]):
                continue
            selected.append(buckets[label][offset])
            offsets[label] = offset + 1
            progressed = True
            if len(selected) >= min(limit, len(rows)):
                break
        if not progressed:
            break
    return selected


@dataclass(frozen=True, slots=True)
class MolecularGraphRecord:
    """One validated classification row plus its portable graph."""

    molecule_id: str
    smiles: str
    label: int
    graph: MolecularGraphFeatures
    split: str | None = None
    source_row_index: int | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class MolecularGraphData:
    """One graph using immutable integer tuples, independent of PyG."""

    x: tuple[tuple[int, ...], ...]
    edge_index: tuple[tuple[int, ...], tuple[int, ...]]
    edge_attr: tuple[tuple[int, ...], ...]
    y: int
    molecule_id: str
    smiles: str
    split: str | None
    graph_sha256: str

    @property
    def num_nodes(self) -> int:
        return len(self.x)


@dataclass(frozen=True, slots=True)
class MolecularGraphBatch:
    """Minimal tensor batch accepted by :class:`MolecularGNN` and the oracle."""

    x: Any
    edge_index: Any
    edge_attr: Any
    batch: Any
    y: Any
    molecule_ids: tuple[str, ...]
    smiles: tuple[str, ...]
    splits: tuple[str | None, ...]
    graph_sha256s: tuple[str, ...]

    @property
    def num_graphs(self) -> int:
        return len(self.molecule_ids)

    def to(self, device: str | Any) -> "MolecularGraphBatch":
        return replace(
            self,
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            batch=self.batch.to(device),
            y=self.y.to(device),
        )


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - checked in runtime environment.
        raise RuntimeError("Molecular GNN batching requires PyTorch.") from exc
    return torch


def _tensor_graph_parts(graph: Any) -> tuple[Any, Any, Any, int, str, str, Any, str]:
    """Read either a portable graph or a PyG-compatible graph object."""

    torch = _require_torch()
    if isinstance(graph, MolecularGraphData):
        x = torch.tensor(graph.x, dtype=torch.long)
        edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
        edge_attr = torch.tensor(graph.edge_attr, dtype=torch.long)
        if edge_index.numel() == 0:
            edge_index = edge_index.reshape(2, 0)
        if edge_attr.numel() == 0:
            # The portable row shape is unavailable after tensorizing an empty tuple;
            # callers repair it from a non-empty peer or the explicit dataset schema.
            edge_attr = edge_attr.reshape(0, 0)
        return (
            x,
            edge_index,
            edge_attr,
            int(graph.y),
            graph.molecule_id,
            graph.smiles,
            graph.split,
            graph.graph_sha256,
        )

    required = ("x", "edge_index", "edge_attr")
    missing = [name for name in required if not hasattr(graph, name)]
    if missing:
        raise TypeError(f"Graph object is missing tensor attributes: {missing}")
    x = torch.as_tensor(graph.x, dtype=torch.long)
    edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long).reshape(2, -1)
    edge_attr = torch.as_tensor(graph.edge_attr, dtype=torch.long)
    if edge_attr.ndim == 1:
        edge_attr = edge_attr.reshape(-1, 1)
    raw_y = getattr(graph, "y", -1)
    if hasattr(raw_y, "reshape"):
        values = raw_y.reshape(-1).tolist()
        label = int(values[0]) if values else -1
    else:
        label = int(raw_y)
    molecule_id = str(
        getattr(graph, "molecule_id", getattr(graph, "parent_id", "")) or ""
    )
    smiles = str(getattr(graph, "smiles", "") or "")
    split = getattr(graph, "split", None)
    graph_sha256 = str(getattr(graph, "graph_sha256", "") or "")
    return x, edge_index, edge_attr, label, molecule_id, smiles, split, graph_sha256


def collate_molecular_graphs(
    graphs: Sequence[Any],
    *,
    edge_feature_dim: int | None = None,
) -> MolecularGraphBatch:
    """Batch portable or PyG-compatible graphs without requiring PyG itself."""

    torch = _require_torch()
    if not graphs:
        raise ValueError("Cannot collate an empty molecular graph batch.")
    parts = [_tensor_graph_parts(graph) for graph in graphs]
    inferred_edge_dims = {
        int(edge_attr.shape[1])
        for _x, _edge_index, edge_attr, *_rest in parts
        if edge_attr.ndim == 2 and int(edge_attr.shape[1]) > 0
    }
    if edge_feature_dim is None:
        if len(inferred_edge_dims) > 1:
            raise ValueError(f"Graph batch mixes edge feature dimensions: {inferred_edge_dims}")
        edge_feature_dim = next(iter(inferred_edge_dims), 0)
    if int(edge_feature_dim) <= 0:
        raise ValueError(
            "edge_feature_dim is required when every graph in a batch has zero bonds."
        )

    xs: list[Any] = []
    edges: list[Any] = []
    attrs: list[Any] = []
    assignments: list[Any] = []
    labels: list[int] = []
    molecule_ids: list[str] = []
    smiles_values: list[str] = []
    splits: list[str | None] = []
    graph_hashes: list[str] = []
    offset = 0
    for graph_index, part in enumerate(parts):
        x, edge_index, edge_attr, label, molecule_id, smiles, split, graph_hash = part
        if x.ndim != 2 or int(x.shape[0]) <= 0:
            raise ValueError("Every graph must have a non-empty rank-2 node tensor.")
        if edge_attr.numel() == 0:
            edge_attr = torch.empty((0, int(edge_feature_dim)), dtype=torch.long)
        if edge_attr.ndim != 2 or int(edge_attr.shape[1]) != int(edge_feature_dim):
            raise ValueError(
                "Edge feature dimension mismatch while batching molecular graphs."
            )
        if int(edge_index.shape[1]) != int(edge_attr.shape[0]):
            raise ValueError("edge_index and edge_attr row counts differ.")
        xs.append(x)
        edges.append(edge_index + int(offset))
        attrs.append(edge_attr)
        assignments.append(
            torch.full((int(x.shape[0]),), graph_index, dtype=torch.long)
        )
        labels.append(label)
        molecule_ids.append(molecule_id or f"graph_{graph_index}")
        smiles_values.append(smiles)
        splits.append(None if split is None else str(split))
        graph_hashes.append(graph_hash)
        offset += int(x.shape[0])

    return MolecularGraphBatch(
        x=torch.cat(xs, dim=0),
        edge_index=torch.cat(edges, dim=1),
        edge_attr=torch.cat(attrs, dim=0),
        batch=torch.cat(assignments, dim=0),
        y=torch.tensor(labels, dtype=torch.long),
        molecule_ids=tuple(molecule_ids),
        smiles=tuple(smiles_values),
        splits=tuple(splits),
        graph_sha256s=tuple(graph_hashes),
    )


class MolecularGraphDataset:
    """In-memory, schema-bound molecular graph classification dataset."""

    def __init__(
        self,
        records: Sequence[MolecularGraphRecord],
        *,
        feature_schema: MolecularFeatureSchema,
        num_classes: int,
        source_path: str | Path | None = None,
        source_sha256: str | None = None,
    ) -> None:
        if int(num_classes) < 2:
            raise ValueError("Molecular classification requires num_classes >= 2.")
        if not records:
            raise ValueError("MolecularGraphDataset cannot be empty.")
        ids = [record.molecule_id for record in records]
        smiles = [record.graph.canonical_smiles for record in records]
        if len(set(ids)) != len(ids):
            raise ValueError("MolecularGraphDataset contains duplicate molecule_id values.")
        if len(set(smiles)) != len(smiles):
            raise ValueError("MolecularGraphDataset contains duplicate canonical SMILES.")
        invalid_labels = sorted(
            {int(record.label) for record in records if not 0 <= int(record.label) < num_classes}
        )
        if invalid_labels:
            raise ValueError(
                f"Dataset labels fall outside [0, {num_classes}): {invalid_labels}"
            )
        schema_hash = str(feature_schema.to_dict()["schema_sha256"])
        mismatches = [
            record.molecule_id
            for record in records
            if record.graph.schema_sha256 != schema_hash
        ]
        if mismatches:
            raise ValueError(
                "Graph/schema fingerprint mismatch for records: " + ", ".join(mismatches[:5])
            )
        self.records = tuple(records)
        self.feature_schema = feature_schema
        self.num_classes = int(num_classes)
        self.source_path = None if source_path is None else Path(source_path).resolve()
        self.source_sha256 = source_sha256
        self.dataset_fingerprint = _stable_hash(
            [
                {
                    "molecule_id": record.molecule_id,
                    "canonical_smiles": record.graph.canonical_smiles,
                    "label": int(record.label),
                    "split": record.split,
                    "graph_sha256": record.graph.graph_sha256,
                }
                for record in self.records
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> MolecularGraphData:
        record = self.records[int(index)]
        return MolecularGraphData(
            x=record.graph.node_features,
            edge_index=record.graph.edge_index,
            edge_attr=record.graph.edge_features,
            y=int(record.label),
            molecule_id=record.molecule_id,
            smiles=record.graph.canonical_smiles,
            split=record.split,
            graph_sha256=record.graph.graph_sha256,
        )

    def __iter__(self) -> Iterator[MolecularGraphData]:
        for index in range(len(self)):
            yield self[index]

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(int(record.label) for record in self.records)

    def subset(self, indices: Sequence[int]) -> "MolecularGraphDataset":
        return MolecularGraphDataset(
            [self.records[int(index)] for index in indices],
            feature_schema=self.feature_schema,
            num_classes=self.num_classes,
            source_path=self.source_path,
            source_sha256=self.source_sha256,
        )

    def collate(self, rows: Sequence[Any]) -> MolecularGraphBatch:
        return collate_molecular_graphs(
            rows, edge_feature_dim=len(self.feature_schema.edge_fields)
        )

    def manifest(self) -> dict[str, Any]:
        split_counts: dict[str, int] = {}
        label_counts = {str(label): 0 for label in range(self.num_classes)}
        for record in self.records:
            split = record.split or "unspecified"
            split_counts[split] = split_counts.get(split, 0) + 1
            label_counts[str(int(record.label))] += 1
        return {
            "schema_version": "molecular_graph_dataset_v1",
            "num_records": len(self),
            "num_classes": self.num_classes,
            "label_counts": label_counts,
            "split_counts": split_counts,
            "source_path": None if self.source_path is None else str(self.source_path),
            "source_sha256": self.source_sha256,
            "dataset_fingerprint": self.dataset_fingerprint,
            "feature_schema_sha256": self.feature_schema.to_dict()["schema_sha256"],
        }

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        num_classes: int,
        featurizer: MolecularGraphFeaturizer | None = None,
        smiles_column: str | None = None,
        label_column: str | None = None,
        id_column: str | None = None,
        split_column: str | None = "split",
        expected_split: str | None = None,
        limit: int | None = None,
        stratified_limit: bool = False,
    ) -> "MolecularGraphDataset":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Molecular dataset CSV does not exist: {source}")
        graph_featurizer = featurizer or MolecularGraphFeaturizer()
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = tuple(reader.fieldnames or ())
            smiles_key = _resolve_column(
                fieldnames, smiles_column, SMILES_COLUMN_CANDIDATES, role="SMILES"
            )
            label_key = _resolve_column(
                fieldnames, label_column, LABEL_COLUMN_CANDIDATES, role="label"
            )
            id_key = _resolve_column(
                fieldnames,
                id_column,
                ID_COLUMN_CANDIDATES,
                role="molecule id",
                required=False,
            )
            if split_column and split_column not in fieldnames:
                split_column = None
            rows = list(reader)
        if limit is not None:
            if int(limit) <= 0:
                raise ValueError("Dataset limit must be positive when supplied.")
            rows = (
                list(
                    _stratified_limit_rows(
                        rows,
                        label_column=str(label_key),
                        num_classes=int(num_classes),
                        limit=int(limit),
                    )
                )
                if stratified_limit
                else rows[: int(limit)]
            )
        wanted_split = _normalize_split(expected_split)
        records: list[MolecularGraphRecord] = []
        for row_index, row in enumerate(rows):
            raw_smiles = str(row.get(str(smiles_key)) or "").strip()
            try:
                label = int(str(row.get(str(label_key))).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid molecular label at {source}:{row_index + 2}: "
                    f"{row.get(str(label_key))!r}"
                ) from exc
            row_split = _normalize_split(
                None if split_column is None else row.get(split_column)
            )
            if wanted_split is not None and row_split not in (None, wanted_split):
                raise ValueError(
                    f"Unexpected split at {source}:{row_index + 2}: "
                    f"expected={wanted_split!r}, observed={row_split!r}"
                )
            graph = graph_featurizer.featurize(raw_smiles)
            molecule_id = (
                str(row.get(str(id_key)) or "").strip() if id_key else ""
            )
            if not molecule_id:
                molecule_id = "MOL_" + hashlib.sha256(
                    graph.canonical_smiles.encode("utf-8")
                ).hexdigest()[:16]
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {smiles_key, label_key, id_key}
            }
            records.append(
                MolecularGraphRecord(
                    molecule_id=molecule_id,
                    smiles=raw_smiles,
                    label=label,
                    graph=graph,
                    split=wanted_split or row_split,
                    source_row_index=row_index,
                    metadata=metadata,
                )
            )
        return cls(
            records,
            feature_schema=graph_featurizer.schema,
            num_classes=num_classes,
            source_path=source,
            source_sha256=_sha256_file(source),
        )


def build_molecular_data_loader(
    dataset: MolecularGraphDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    sampler: Any = None,
    num_workers: int = 0,
) -> Any:
    """Construct a torch DataLoader while enforcing sampler/shuffle exclusivity."""

    torch = _require_torch()
    if sampler is not None and shuffle:
        raise ValueError("A weighted sampler and shuffle cannot be enabled together.")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=int(num_workers),
        collate_fn=dataset.collate,
    )


__all__ = [
    "ID_COLUMN_CANDIDATES",
    "LABEL_COLUMN_CANDIDATES",
    "MolecularGraphBatch",
    "MolecularGraphData",
    "MolecularGraphDataset",
    "MolecularGraphRecord",
    "SMILES_COLUMN_CANDIDATES",
    "build_molecular_data_loader",
    "collate_molecular_graphs",
    "default_molecular_feature_schema",
]
