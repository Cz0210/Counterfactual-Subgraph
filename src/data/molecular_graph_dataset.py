"""Dataset and batching contracts for task-specific molecular classifiers."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence

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
MOLECULAR_GRAPH_CACHE_SCHEMA_VERSION = "molecular_graph_tensor_cache_v1"


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


def _graph_cache_payload(
    dataset: MolecularGraphDataset,
    *,
    split_name: str,
) -> dict[str, Any]:
    """Flatten one split into a ``weights_only=True`` compatible payload.

    The payload intentionally contains only tensors and Python primitives.  In
    particular, it never serializes ``MolecularGraphData`` or another custom
    class, so consumers do not need to weaken PyTorch's safe-loading policy.
    Edge indices stay local to each graph and are sliced using ``edge_ptr``.
    """

    torch = _require_torch()
    normalized_split = _normalize_split(split_name)
    if not normalized_split:
        raise ValueError("A molecular graph cache requires a non-empty split name.")
    if dataset.source_path is None or not dataset.source_sha256:
        raise ValueError("A molecular graph cache requires source CSV provenance.")

    graphs = list(dataset)
    node_feature_dim = len(dataset.feature_schema.node_fields)
    edge_feature_dim = len(dataset.feature_schema.edge_fields)
    node_rows: list[tuple[int, ...]] = []
    edge_sources: list[int] = []
    edge_targets: list[int] = []
    edge_rows: list[tuple[int, ...]] = []
    node_ptr = [0]
    edge_ptr = [0]
    molecule_ids: list[str] = []
    smiles_values: list[str] = []
    record_splits: list[str] = []
    graph_hashes: list[str] = []
    labels: list[int] = []

    for graph in graphs:
        record_split = _normalize_split(graph.split)
        if record_split not in (None, normalized_split):
            raise ValueError(
                "Molecular graph cache split mismatch: "
                f"expected={normalized_split!r}, observed={record_split!r}, "
                f"molecule_id={graph.molecule_id!r}."
            )
        if any(len(row) != node_feature_dim for row in graph.x):
            raise ValueError(f"Node feature width mismatch for {graph.molecule_id!r}.")
        if any(len(row) != edge_feature_dim for row in graph.edge_attr):
            raise ValueError(f"Edge feature width mismatch for {graph.molecule_id!r}.")
        node_rows.extend(graph.x)
        edge_sources.extend(graph.edge_index[0])
        edge_targets.extend(graph.edge_index[1])
        edge_rows.extend(graph.edge_attr)
        node_ptr.append(len(node_rows))
        edge_ptr.append(len(edge_rows))
        molecule_ids.append(graph.molecule_id)
        smiles_values.append(graph.smiles)
        record_splits.append(record_split or normalized_split)
        graph_hashes.append(graph.graph_sha256)
        labels.append(int(graph.y))

    x = torch.tensor(node_rows, dtype=torch.long)
    edge_index = torch.tensor(
        (edge_sources, edge_targets), dtype=torch.long
    ).reshape(2, -1)
    edge_attr = torch.tensor(edge_rows, dtype=torch.long)
    if edge_attr.numel() == 0:
        edge_attr = edge_attr.reshape(0, edge_feature_dim)
    return {
        "cache_schema_version": MOLECULAR_GRAPH_CACHE_SCHEMA_VERSION,
        "split": normalized_split,
        "graph_count": len(graphs),
        "num_classes": dataset.num_classes,
        "feature_schema": dataset.feature_schema.to_dict(),
        "source_csv": str(dataset.source_path),
        "source_csv_sha256": dataset.source_sha256,
        "dataset_fingerprint": dataset.dataset_fingerprint,
        "x": x,
        "node_ptr": torch.tensor(node_ptr, dtype=torch.long),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_ptr": torch.tensor(edge_ptr, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "molecule_ids": molecule_ids,
        "smiles": smiles_values,
        "record_splits": record_splits,
        "graph_sha256s": graph_hashes,
    }


def save_molecular_graph_cache(
    dataset: MolecularGraphDataset,
    path: str | Path,
    *,
    split_name: str,
) -> dict[str, Any]:
    """Persist one split to a fresh, atomic, safe-loadable ``.pt`` file."""

    torch = _require_torch()
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Molecular graph cache must be fresh: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _graph_cache_payload(dataset, split_name=split_name)
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "cache_schema_version": MOLECULAR_GRAPH_CACHE_SCHEMA_VERSION,
        "path": str(target),
        "sha256": _sha256_file(target),
        "split": str(payload["split"]),
        "graph_count": int(payload["graph_count"]),
        "num_classes": int(payload["num_classes"]),
        "source_csv": str(payload["source_csv"]),
        "source_csv_sha256": str(payload["source_csv_sha256"]),
        "dataset_fingerprint": str(payload["dataset_fingerprint"]),
        "feature_schema_sha256": str(
            payload["feature_schema"]["schema_sha256"]
        ),
    }


def _checked_pointer_values(
    tensor: Any,
    *,
    name: str,
    graph_count: int,
    terminal: int,
) -> list[int]:
    torch = _require_torch()
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1:
        raise ValueError(f"Cached {name} must be a rank-1 tensor.")
    values = [int(value) for value in tensor.tolist()]
    if len(values) != graph_count + 1:
        raise ValueError(f"Cached {name} length does not match graph_count.")
    if not values or values[0] != 0 or values[-1] != terminal:
        raise ValueError(f"Cached {name} has invalid boundary values.")
    if any(right < left for left, right in zip(values, values[1:])):
        raise ValueError(f"Cached {name} is not monotonic.")
    return values


def load_molecular_graph_cache(
    path: str | Path | BinaryIO,
    *,
    expected_num_classes: int | None = None,
    expected_source_sha256: str | None = None,
    expected_feature_schema: MolecularFeatureSchema | None = None,
) -> MolecularGraphDataset:
    """Load and validate a plain-tensor graph cache without custom unpickling.

    ``path`` may be either an ordinary path or an already-open seekable binary
    stream.  The latter is required by long-lived authority holders: callers
    can deserialize the inode they opened and authenticated without reopening
    a replaceable pathname between verification and ``torch.load``.
    """

    torch = _require_torch()
    if isinstance(path, (str, Path)):
        source: Any = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Molecular graph cache does not exist: {source}")
    else:
        source = path
        if not all(
            callable(getattr(source, name, None))
            for name in ("read", "seek", "tell")
        ):
            raise TypeError(
                "Molecular graph cache input must be a path or seekable binary stream."
            )
        try:
            source.seek(0)
        except (OSError, ValueError) as exc:
            raise ValueError(
                "Molecular graph cache binary stream is not seekable."
            ) from exc
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Molecular graph cache payload must be a dictionary.")
    if payload.get("cache_schema_version") != MOLECULAR_GRAPH_CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported molecular graph cache schema version.")

    required = {
        "split",
        "graph_count",
        "num_classes",
        "feature_schema",
        "source_csv",
        "source_csv_sha256",
        "dataset_fingerprint",
        "x",
        "node_ptr",
        "edge_index",
        "edge_attr",
        "edge_ptr",
        "labels",
        "molecule_ids",
        "smiles",
        "record_splits",
        "graph_sha256s",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Molecular graph cache is missing fields: {missing}")

    graph_count = int(payload["graph_count"])
    num_classes = int(payload["num_classes"])
    if graph_count <= 0 or num_classes < 2:
        raise ValueError("Molecular graph cache has invalid graph/class counts.")
    if expected_num_classes is not None and num_classes != int(expected_num_classes):
        raise ValueError(
            f"Molecular graph cache num_classes mismatch: {num_classes} != "
            f"{int(expected_num_classes)}."
        )
    source_sha256 = str(payload["source_csv_sha256"])
    if expected_source_sha256 is not None and source_sha256 != str(
        expected_source_sha256
    ):
        raise ValueError("Molecular graph cache source CSV SHA256 mismatch.")

    feature_schema = MolecularFeatureSchema.from_dict(payload["feature_schema"])
    if expected_feature_schema is not None and feature_schema.to_dict() != (
        expected_feature_schema.to_dict()
    ):
        raise ValueError("Molecular graph cache feature schema mismatch.")
    schema_sha256 = str(feature_schema.to_dict()["schema_sha256"])

    x = payload["x"]
    edge_index = payload["edge_index"]
    edge_attr = payload["edge_attr"]
    labels = payload["labels"]
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError("Cached x must be a rank-2 tensor.")
    if int(x.shape[1]) != len(feature_schema.node_fields):
        raise ValueError("Cached x width does not match the feature schema.")
    if not isinstance(edge_index, torch.Tensor) or tuple(edge_index.shape[:1]) != (2,):
        raise ValueError("Cached edge_index must have shape [2, num_edges].")
    if edge_index.ndim != 2:
        raise ValueError("Cached edge_index must be rank 2.")
    if not isinstance(edge_attr, torch.Tensor) or edge_attr.ndim != 2:
        raise ValueError("Cached edge_attr must be a rank-2 tensor.")
    if int(edge_attr.shape[1]) != len(feature_schema.edge_fields):
        raise ValueError("Cached edge_attr width does not match the feature schema.")
    if int(edge_index.shape[1]) != int(edge_attr.shape[0]):
        raise ValueError("Cached edge_index/edge_attr counts differ.")
    if not isinstance(labels, torch.Tensor) or tuple(labels.shape) != (graph_count,):
        raise ValueError("Cached labels shape does not match graph_count.")

    node_ptr = _checked_pointer_values(
        payload["node_ptr"],
        name="node_ptr",
        graph_count=graph_count,
        terminal=int(x.shape[0]),
    )
    edge_ptr = _checked_pointer_values(
        payload["edge_ptr"],
        name="edge_ptr",
        graph_count=graph_count,
        terminal=int(edge_attr.shape[0]),
    )
    sequence_fields = (
        "molecule_ids",
        "smiles",
        "record_splits",
        "graph_sha256s",
    )
    if any(
        not isinstance(payload[name], list) or len(payload[name]) != graph_count
        for name in sequence_fields
    ):
        raise ValueError("Cached graph metadata lengths do not match graph_count.")

    records: list[MolecularGraphRecord] = []
    for index in range(graph_count):
        node_start, node_end = node_ptr[index : index + 2]
        edge_start, edge_end = edge_ptr[index : index + 2]
        local_edge_index = edge_index[:, edge_start:edge_end]
        if local_edge_index.numel() and (
            int(local_edge_index.min()) < 0
            or int(local_edge_index.max()) >= node_end - node_start
        ):
            raise ValueError(f"Cached graph {index} has out-of-range edge indices.")
        node_features = tuple(
            tuple(int(value) for value in row)
            for row in x[node_start:node_end].tolist()
        )
        edge_indices = tuple(
            tuple(int(value) for value in row)
            for row in local_edge_index.tolist()
        )
        edge_features = tuple(
            tuple(int(value) for value in row)
            for row in edge_attr[edge_start:edge_end].tolist()
        )
        canonical_smiles = str(payload["smiles"][index])
        graph_payload = {
            "canonical_smiles": canonical_smiles,
            "node_features": node_features,
            "edge_index": edge_indices,
            "edge_features": edge_features,
            "schema_sha256": schema_sha256,
        }
        graph_sha256 = _stable_hash(graph_payload)
        if graph_sha256 != str(payload["graph_sha256s"][index]):
            raise ValueError(f"Cached graph fingerprint mismatch at index {index}.")
        label = int(labels[index])
        graph = MolecularGraphFeatures(
            canonical_smiles=canonical_smiles,
            node_features=node_features,
            edge_index=edge_indices,  # type: ignore[arg-type]
            edge_features=edge_features,
            schema_sha256=schema_sha256,
            graph_sha256=graph_sha256,
        )
        records.append(
            MolecularGraphRecord(
                molecule_id=str(payload["molecule_ids"][index]),
                smiles=canonical_smiles,
                label=label,
                graph=graph,
                split=str(payload["record_splits"][index]),
                source_row_index=index,
            )
        )

    dataset = MolecularGraphDataset(
        records,
        feature_schema=feature_schema,
        num_classes=num_classes,
        source_path=str(payload["source_csv"]),
        source_sha256=source_sha256,
    )
    if dataset.dataset_fingerprint != str(payload["dataset_fingerprint"]):
        raise ValueError("Molecular graph cache dataset fingerprint mismatch.")
    return dataset


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
    "MOLECULAR_GRAPH_CACHE_SCHEMA_VERSION",
    "MolecularGraphBatch",
    "MolecularGraphData",
    "MolecularGraphDataset",
    "MolecularGraphRecord",
    "SMILES_COLUMN_CANDIDATES",
    "build_molecular_data_loader",
    "collate_molecular_graphs",
    "default_molecular_feature_schema",
    "load_molecular_graph_cache",
    "save_molecular_graph_cache",
]
