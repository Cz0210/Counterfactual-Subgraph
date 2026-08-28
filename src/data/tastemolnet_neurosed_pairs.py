"""Leakage-safe TasteMolNet graph-pair construction for NeuroSED.

Only the caller-supplied ``train.csv`` and ``validation.csv`` payloads are
opened.  Each nested graph pair contains a connected induced BFS subgraph and
its own parent.  The ordered SED input is parent then subgraph: under GREED's
zero insertion/unit deletion costs, omitted nodes and edges are therefore the
exact directional edit count and both interval bounds (``lb == ub``).

No labels are consulted: the auxiliary distance model is source-label
independent and is never a classifier.
"""

from __future__ import annotations

from collections import deque
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import stat
from typing import Any, Mapping, Sequence


ALLOWED_PAYLOAD_SPLITS = frozenset({"train", "validation"})
FORBIDDEN_PAYLOAD_SPLITS = frozenset({"calibration", "test"})


class TasteNeuroSEDPairError(RuntimeError):
    """Raised when split or graph-pair provenance is not closed."""


@dataclass(frozen=True, slots=True)
class TasteSplitRow:
    molecule_id: str
    smiles: str
    split: str


@dataclass(frozen=True, slots=True)
class TastePair:
    pair_id: str
    split: str
    query: Any
    parent: Any
    lb: float
    ub: float
    parent_id_sha256: str
    selected_nodes_sha256: str
    removed_nodes: int
    removed_edges: int


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _open_regular_no_symlink(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    if not path.is_absolute():
        raise TasteNeuroSEDPairError(f"{label} must be absolute")
    unresolved = Path(os.path.abspath(path))
    if unresolved != path:
        raise TasteNeuroSEDPairError(f"{label} contains an alias")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise TasteNeuroSEDPairError(f"{label} contains a symlink")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    info = os.fstat(descriptor)
    named = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_dev != named.st_dev
        or info.st_ino != named.st_ino
    ):
        os.close(descriptor)
        raise TasteNeuroSEDPairError(f"{label} physical identity changed")
    return descriptor, info


def read_taste_split_rows(
    path: str | Path,
    *,
    expected_split: str,
) -> tuple[list[TasteSplitRow], dict[str, Any]]:
    """Read exactly one admitted split through a held, no-follow descriptor."""

    if expected_split not in ALLOWED_PAYLOAD_SPLITS:
        raise TasteNeuroSEDPairError(
            "NeuroSED may open only train and validation payloads"
        )
    source = Path(path)
    if source.name != f"{expected_split}.csv":
        raise TasteNeuroSEDPairError(
            f"{expected_split} payload must be named {expected_split}.csv"
        )
    descriptor, before = _open_regular_no_symlink(
        source, label=f"Taste {expected_split} CSV"
    )
    try:
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        source_bytes = b"".join(chunks)
        after = os.fstat(descriptor)
        named_after = os.stat(source, follow_symlinks=False)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or after.st_dev != named_after.st_dev
            or after.st_ino != named_after.st_ino
        ):
            raise TasteNeuroSEDPairError(
                f"Taste {expected_split} CSV changed while read"
            )
    finally:
        os.close(descriptor)

    rows, evidence = parse_taste_split_rows_bytes(
        source_bytes, expected_split=expected_split
    )
    return rows, {
        **evidence,
        "source_csv_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }


def parse_taste_split_rows_bytes(
    source_bytes: bytes,
    *,
    expected_split: str,
) -> tuple[list[TasteSplitRow], dict[str, Any]]:
    """Parse descriptor-held split bytes without reopening a named path."""

    if expected_split not in ALLOWED_PAYLOAD_SPLITS:
        raise TasteNeuroSEDPairError(
            "NeuroSED may parse only train and validation payloads"
        )
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TasteNeuroSEDPairError(
            f"Taste {expected_split} CSV is not UTF-8"
        ) from exc
    reader = csv.DictReader(io.StringIO(source_text, newline=""), strict=True)
    rows_raw = [dict(row) for row in reader]
    if not rows_raw:
        raise TasteNeuroSEDPairError(f"Taste {expected_split} CSV is empty")
    rows: list[TasteSplitRow] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows_raw):
        observed_split = str(raw.get("split") or "").strip()
        if observed_split != expected_split:
            raise TasteNeuroSEDPairError(
                f"Taste {expected_split} CSV contains split={observed_split!r}"
            )
        molecule_id = str(raw.get("molecule_id") or "").strip()
        smiles = str(raw.get("model_smiles") or raw.get("canonical_smiles") or "").strip()
        if not molecule_id or not smiles:
            raise TasteNeuroSEDPairError(
                f"Taste {expected_split} row {index} lacks molecule identity or SMILES"
            )
        if molecule_id in seen:
            raise TasteNeuroSEDPairError(
                f"Taste {expected_split} molecule IDs are not unique"
            )
        seen.add(molecule_id)
        rows.append(TasteSplitRow(molecule_id=molecule_id, smiles=smiles, split=expected_split))
    ids_hash = _canonical_hash(sorted(seen))
    return rows, {
        "split": expected_split,
        "row_count": len(rows),
        "graph_ids_hash": ids_hash,
        "all_rows_declared_expected_split": True,
        "labels_opened_but_not_consumed": True,
    }


def read_preparation_split_manifest(path: str | Path) -> dict[str, Any]:
    """Read only the non-payload preparation manifest and validate its task."""

    source = Path(path)
    if source.name != "split_manifest.json":
        raise TasteNeuroSEDPairError("split manifest filename changed")
    descriptor, before = _open_regular_no_symlink(source, label="Taste split manifest")
    try:
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 64 * 1024)
            if not block:
                break
            chunks.append(block)
            if sum(map(len, chunks)) > 1024 * 1024:
                raise TasteNeuroSEDPairError("Taste split manifest is too large")
        manifest_bytes = b"".join(chunks)
        after = os.fstat(descriptor)
        named_after = os.stat(source, follow_symlinks=False)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or after.st_dev != named_after.st_dev
            or after.st_ino != named_after.st_ino
        ):
            raise TasteNeuroSEDPairError("Taste split manifest changed while read")
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteNeuroSEDPairError("Taste split manifest is not JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("dataset") != "tastemolnet"
        or payload.get("num_classes") != 3
        or payload.get("label_map")
        != {"0": "Bitter", "1": "Sweet", "2": "Tasteless"}
        or payload.get("source_label") != 1
        or payload.get("scaffold_overlap_gate_passed") is not True
        or payload.get("same_canonical_smiles_cross_split_forbidden") is not True
    ):
        raise TasteNeuroSEDPairError("Taste split manifest contract changed")
    return {
        "path": str(source),
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "schema_version": payload.get("schema_version"),
        "dataset": "tastemolnet",
        "seed": payload.get("seed"),
        "scaffold_overlap_gate_passed": True,
        "same_canonical_smiles_cross_split_forbidden": True,
    }


def _require_chemistry() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch_geometric.data import Data
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - AutoDL dependency gate.
        raise TasteNeuroSEDPairError(
            "Taste pair generation requires torch, torch_geometric, and RDKit"
        ) from exc
    return torch, Data, Chem


def _expanded_molecule(smiles: str) -> Any:
    _torch, _Data, Chem = _require_chemistry()
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise TasteNeuroSEDPairError("Taste split contains an unparsable SMILES")
    Chem.SanitizeMol(molecule)
    expanded = Chem.AddHs(molecule, addCoords=False)
    Chem.SanitizeMol(expanded)
    if len(Chem.GetMolFrags(expanded)) != 1:
        raise TasteNeuroSEDPairError("Taste NeuroSED parent graph is disconnected")
    return expanded


def derive_feature_schema(
    train_rows: Sequence[TasteSplitRow],
    validation_rows: Sequence[TasteSplitRow],
) -> dict[str, Any]:
    """Derive atom channels from train and reject validation-unseen atoms."""

    train_atoms: set[int] = set()
    validation_atoms: set[int] = set()
    max_nodes = 0
    for row in train_rows:
        molecule = _expanded_molecule(row.smiles)
        train_atoms.update(int(atom.GetAtomicNum()) for atom in molecule.GetAtoms())
        max_nodes = max(max_nodes, int(molecule.GetNumAtoms()))
    for row in validation_rows:
        molecule = _expanded_molecule(row.smiles)
        validation_atoms.update(int(atom.GetAtomicNum()) for atom in molecule.GetAtoms())
        max_nodes = max(max_nodes, int(molecule.GetNumAtoms()))
    unseen = sorted(validation_atoms - train_atoms)
    if unseen:
        raise TasteNeuroSEDPairError(
            f"Taste validation contains train-unseen atom channels: {unseen}"
        )
    vocabulary = sorted(train_atoms)
    if not vocabulary:
        raise TasteNeuroSEDPairError("Taste train atom vocabulary is empty")
    return {
        "schema_version": "tastemolnet_gcf_neurosed_feature_schema_v1",
        "dataset": "tastemolnet",
        "node_feature_semantics": "one_hot_atomic_number",
        "feature_atomic_numbers": vocabulary,
        "input_dim": len(vocabulary),
        "explicit_h_nodes": True,
        "native_adjacency_semantics": "binary_connectivity_directed_both_ways",
        "edge_features_used": False,
        "validation_unseen_atomic_numbers": [],
        "train_derived_only": True,
        "maximum_train_or_validation_nodes": int(max_nodes),
    }


def rows_to_graphs(
    rows: Sequence[TasteSplitRow],
    feature_schema: Mapping[str, Any],
) -> list[Any]:
    torch, Data, _Chem = _require_chemistry()
    vocabulary = [int(value) for value in feature_schema["feature_atomic_numbers"]]
    lookup = {atomic_number: index for index, atomic_number in enumerate(vocabulary)}
    graphs: list[Any] = []
    for row in rows:
        molecule = _expanded_molecule(row.smiles)
        x = torch.zeros((molecule.GetNumAtoms(), len(vocabulary)), dtype=torch.float32)
        for atom in molecule.GetAtoms():
            atomic_number = int(atom.GetAtomicNum())
            if atomic_number not in lookup:
                raise TasteNeuroSEDPairError(
                    f"split={row.split} contains an unregistered atom channel"
                )
            x[int(atom.GetIdx()), lookup[atomic_number]] = 1.0
        edges: list[tuple[int, int]] = []
        for bond in molecule.GetBonds():
            left = int(bond.GetBeginAtomIdx())
            right = int(bond.GetEndAtomIdx())
            edges.extend(((left, right), (right, left)))
        edge_index = (
            torch.tensor(edges, dtype=torch.long).T.contiguous()
            if edges
            else torch.empty((2, 0), dtype=torch.long)
        )
        graph = Data(x=x, edge_index=edge_index, num_nodes=int(x.shape[0]))
        graph.neurosed_parent_id_sha256 = hashlib.sha256(
            row.molecule_id.encode("utf-8")
        ).hexdigest()
        graph.neurosed_split = row.split
        graphs.append(graph)
    return graphs


def _adjacency(graph: Any) -> list[list[int]]:
    adjacency: list[set[int]] = [set() for _ in range(int(graph.num_nodes))]
    for left, right in graph.edge_index.detach().cpu().T.tolist():
        adjacency[int(left)].add(int(right))
    return [sorted(values) for values in adjacency]


def _bfs_nodes(graph: Any, *, count: int, rng: random.Random) -> list[int]:
    total = int(graph.num_nodes)
    if not 2 <= count < total:
        raise TasteNeuroSEDPairError("BFS subgraph size must be in [2,parent-1]")
    adjacency = _adjacency(graph)
    start = rng.randrange(total)
    queue: deque[int] = deque([start])
    seen = {start}
    selected: list[int] = []
    while queue and len(selected) < count:
        node = queue.popleft()
        selected.append(node)
        neighbours = list(adjacency[node])
        rng.shuffle(neighbours)
        for neighbour in neighbours:
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    if len(selected) != count:
        raise TasteNeuroSEDPairError("Taste parent is not connected for BFS sampling")
    return sorted(selected)


def _induced_subgraph(graph: Any, selected: Sequence[int]) -> Any:
    torch, Data, _Chem = _require_chemistry()
    selected_tensor = torch.tensor(list(selected), dtype=torch.long)
    reverse = torch.full((int(graph.num_nodes),), -1, dtype=torch.long)
    reverse[selected_tensor] = torch.arange(len(selected), dtype=torch.long)
    mask = (reverse[graph.edge_index[0]] >= 0) & (reverse[graph.edge_index[1]] >= 0)
    edge_index = reverse[graph.edge_index[:, mask]]
    result = Data(
        x=graph.x[selected_tensor].clone(),
        edge_index=edge_index.clone(),
        num_nodes=len(selected),
    )
    return result


def _is_connected(graph: Any) -> bool:
    if int(graph.num_nodes) <= 0:
        return False
    adjacency = _adjacency(graph)
    seen = {0}
    queue: deque[int] = deque([0])
    while queue:
        node = queue.popleft()
        for neighbour in adjacency[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return len(seen) == int(graph.num_nodes)


def build_connected_bfs_pairs(
    graphs: Sequence[Any],
    *,
    split: str,
    num_pairs: int,
    seed: int,
) -> list[TastePair]:
    """Build nested graphs whose ordered SED direction is parent-to-subgraph."""

    if split not in ALLOWED_PAYLOAD_SPLITS:
        raise TasteNeuroSEDPairError("pair split must be train or validation")
    eligible = [graph for graph in graphs if int(graph.num_nodes) >= 3]
    if not eligible:
        raise TasteNeuroSEDPairError(f"Taste {split} has no graph with >=3 nodes")
    if int(num_pairs) <= 0:
        raise TasteNeuroSEDPairError("NeuroSED pair count must be positive")
    rng = random.Random(int(seed))
    pairs: list[TastePair] = []
    for index in range(int(num_pairs)):
        parent = eligible[index % len(eligible)]
        keep_count = rng.randint(2, int(parent.num_nodes) - 1)
        selected = _bfs_nodes(parent, count=keep_count, rng=rng)
        query = _induced_subgraph(parent, selected)
        if not _is_connected(query):
            raise TasteNeuroSEDPairError("BFS pair query is disconnected")
        removed_nodes = int(parent.num_nodes) - int(query.num_nodes)
        if int(parent.num_edges) % 2 or int(query.num_edges) % 2:
            raise TasteNeuroSEDPairError("molecular adjacency is not symmetric")
        removed_edges = int(parent.num_edges - query.num_edges) // 2
        edit_count = float(removed_nodes + removed_edges)
        if not math.isfinite(edit_count) or edit_count <= 0:
            raise TasteNeuroSEDPairError("nested edit label is not positive finite")
        parent_hash = str(parent.neurosed_parent_id_sha256)
        selected_hash = _canonical_hash(list(selected))
        pair_id = hashlib.sha256(
            f"{split}\0{index}\0{parent_hash}\0{selected_hash}".encode("utf-8")
        ).hexdigest()
        pairs.append(
            TastePair(
                pair_id=pair_id,
                split=split,
                query=query,
                parent=parent,
                lb=edit_count,
                ub=edit_count,
                parent_id_sha256=parent_hash,
                selected_nodes_sha256=selected_hash,
                removed_nodes=removed_nodes,
                removed_edges=removed_edges,
            )
        )
    return pairs


def pair_manifest(pairs: Sequence[TastePair], *, split: str) -> dict[str, Any]:
    if not pairs or any(pair.split != split or pair.lb != pair.ub for pair in pairs):
        raise TasteNeuroSEDPairError("pair manifest received invalid interval pairs")
    labels = [float(pair.lb) for pair in pairs]
    return {
        "schema_version": "tastemolnet_gcf_neurosed_pair_manifest_v1",
        "split": split,
        "pair_count": len(pairs),
        "pair_builder": "taste_connected_induced_bfs_nested_pair_v1",
        "pair_direction": "parent_to_connected_induced_bfs_subgraph",
        "upstream_greed_pair_sampling_unchanged": False,
        "upstream_greed_pair_sampling": (
            "independent_query_subgraph_to_random_target_with_pyged_bounds"
        ),
        "gcf_runtime_direction": "generated_query_to_original_parent_target",
        "training_direction_matches_gcf_runtime": False,
        "full_official_pair_semantics_claimed": False,
        "distance_label": "known_directional_deletion_node_plus_edge_edit_count",
        "edit_cost_contract": {
            "node_insertion": 0,
            "node_deletion": 1,
            "edge_insertion": 0,
            "edge_deletion": 1,
            "node_relabel": 1,
            "edge_relabel": 0,
        },
        "reverse_subgraph_to_parent_cost_is_zero": True,
        "interval_bounds_exact": True,
        "all_lb_equal_ub": True,
        "minimum_edit_count": min(labels),
        "maximum_edit_count": max(labels),
        "mean_edit_count": sum(labels) / len(labels),
        "pair_ids_hash": _canonical_hash([pair.pair_id for pair in pairs]),
        "parent_graph_ids_hash": _canonical_hash(
            sorted({pair.parent_id_sha256 for pair in pairs})
        ),
        "connected_queries": True,
        "cross_parent_pairs": False,
        "labels_used": False,
    }


class TastePairDataset:
    """Return the explicit Taste adaptation order: parent, then subgraph."""

    def __init__(self, pairs: Sequence[TastePair]) -> None:
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Any, Any, Any, Any]:
        torch, _Data, _Chem = _require_chemistry()
        pair = self.pairs[index]
        return (
            pair.parent,
            pair.query,
            torch.tensor(pair.lb, dtype=torch.float32),
            torch.tensor(pair.ub, dtype=torch.float32),
        )


def split_boundary_manifest(
    *,
    train_evidence: Mapping[str, Any],
    validation_evidence: Mapping[str, Any],
    preparation_manifest: Mapping[str, Any],
    train_validation_intersection_empty: bool,
) -> dict[str, Any]:
    """Record proof from split-local payloads without opening excluded data."""

    return {
        "schema_version": "tastemolnet_gcf_neurosed_split_manifest_v1",
        "dataset": "tastemolnet",
        "neurosed_train_graph_ids_hash": train_evidence["graph_ids_hash"],
        "neurosed_validation_graph_ids_hash": validation_evidence["graph_ids_hash"],
        "train_graph_count": train_evidence["row_count"],
        "validation_graph_count": validation_evidence["row_count"],
        "train_source_csv_sha256": train_evidence["source_csv_sha256"],
        "validation_source_csv_sha256": validation_evidence["source_csv_sha256"],
        "preparation_split_manifest_sha256": preparation_manifest["sha256"],
        "train_neurosed_ids_subset_of_train": True,
        "validation_neurosed_ids_subset_of_validation": True,
        "train_validation_id_intersection_empty": bool(
            train_validation_intersection_empty
        ),
        "calibration_intersection_empty": True,
        "test_intersection_empty": True,
        "partition_proof": (
            "each opened CSV row declared its expected split; the authenticated "
            "preparation manifest forbids canonical cross-split overlap"
        ),
        "opened_payload_splits": ["train", "validation"],
        "calibration_loaded": False,
        "test_loaded": False,
        "calibration_graph_ids_observed": False,
        "test_graph_ids_observed": False,
        "calibration_graph_hashes_observed": False,
        "test_graph_hashes_observed": False,
        "calibration_smiles_observed": False,
        "test_smiles_observed": False,
        "test_labels_observed": False,
        "test_embeddings_observed": False,
    }


__all__ = [
    "ALLOWED_PAYLOAD_SPLITS",
    "FORBIDDEN_PAYLOAD_SPLITS",
    "TasteNeuroSEDPairError",
    "TastePair",
    "TastePairDataset",
    "TasteSplitRow",
    "build_connected_bfs_pairs",
    "derive_feature_schema",
    "pair_manifest",
    "parse_taste_split_rows_bytes",
    "read_preparation_split_manifest",
    "read_taste_split_rows",
    "rows_to_graphs",
    "sha256_file",
    "split_boundary_manifest",
]
