"""Exact, partitioned execution of the pinned official GlobalGCE gSpan tree.

The implementation in this module is intentionally narrow.  It loads only the
two pinned official gSpan source files, preserves their insertion-order DFS
walk, and divides *report/candidate events* into disjoint contiguous units:

``ROOT_SUBTREE``
    One complete ordinary top-level gSpan root.
``PREFIX_HEADER``
    Exactly one candidate at a proper prefix of a split root.  Accepted
    headers contribute their normal official pattern, while rejected headers
    preserve the corresponding support/minimality event.
``PREFIX_SUBTREE``
    One canonical frontier prefix and every official descendant below it.

The prefix headers followed by the frontier subtrees are an exact cut of the
original DFS preorder.  Navigation to a prefix repeats only ancestor checks;
it never emits those ancestors twice and it does not prune a scientific
candidate.  Outputs are written by one process per shard and become visible
only after a partition-boundary atomic rename.  This is a CPU mining surface;
it has no matrix/publisher integration.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import types
import uuid
from typing import Any, Iterable, Iterator, Mapping, Sequence, TextIO


OFFICIAL_GLOBALGCE_COMMIT = "157e65c2850bc787f229a1ee8c60564906b933f2"
OFFICIAL_GSPAN_SHA256 = {
    "models/gSpan/gSpan.py": (
        "65f2dcd2e5d19e066992f60124cae35b873f4df8b06ad4a866da3d21171ab726"
    ),
    "models/gSpan/graph.py": (
        "b8697f50b3650c3db8385570e2d51addffbfd3103d91608aedd5f10dad26d232"
    ),
}
PARTITION_MANIFEST_SCHEMA = "globalgce_hpc_exact_partition_manifest_v1"
UNIT_RESULT_SCHEMA = "globalgce_hpc_exact_partition_result_v1"
SHARD_RESULT_SCHEMA = "globalgce_hpc_exact_shard_result_v1"
MERGE_RESULT_SCHEMA = "globalgce_hpc_exact_merge_v1"
PARITY_RESULT_SCHEMA = "globalgce_hpc_exact_parity_v1"
RESULT_BUNDLE_SCHEMA = "globalgce_hpc_exact_result_bundle_v1"
UNIT_TYPES = frozenset({"ROOT_SUBTREE", "PREFIX_HEADER", "PREFIX_SUBTREE"})
EVENT_STATUSES = frozenset(
    {"ACCEPTED", "REJECTED_MIN_SUPPORT", "REJECTED_NON_MINIMAL"}
)
Scalar = int | str


class GlobalGCEHPCExactError(RuntimeError):
    """An exactness, provenance, partition, or publication gate failed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_commit(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _canonical_jsonl_sha256(value: Mapping[str, Any]) -> str:
    """Hash the canonical newline-terminated JSON used by the input builder."""

    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: str | Path) -> Any:
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {value}")
        ),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    atomic_write_bytes(path, _canonical_bytes(dict(payload)) + b"\n")


def _self_hashed(payload: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = canonical_sha256(result)
    return result


def validate_hpc_cli_contract(
    config: str | Path, overrides: Sequence[str] | None
) -> Path:
    """Enforce the deliberately tiny HPC command-line configuration surface."""

    value = Path(config).expanduser()
    if value.as_posix() != "configs/hpc.yaml" and not value.as_posix().endswith(
        "/configs/hpc.yaml"
    ):
        raise GlobalGCEHPCExactError("--config must be configs/hpc.yaml")
    resolved = value.resolve(strict=True)
    normalized = list(overrides or ())
    if normalized not in ([], ["inference.fallback_to_heuristic=false"]):
        raise GlobalGCEHPCExactError(
            "only --set inference.fallback_to_heuristic=false is accepted"
        )
    return resolved


def _normalize_scalar(value: Any, *, field: str) -> Scalar:
    if type(value) in (int, str):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if type(converted) in (int, str):
            return converted
    raise GlobalGCEHPCExactError(f"{field} must be an exact integer or string")


@dataclass(frozen=True)
class TypedDFSEdge:
    """Portable, typed representation of the official ``DFSedge`` value."""

    frm: int
    to: int
    vevlb: tuple[Scalar, Scalar, Scalar]

    def to_json(self) -> dict[str, Any]:
        return {
            "frm": self.frm,
            "to": self.to,
            "vevlb": list(self.vevlb),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "TypedDFSEdge":
        if type(value) is not dict or set(value) != {"frm", "to", "vevlb"}:
            raise GlobalGCEHPCExactError("typed DFS edge schema mismatch")
        labels = value["vevlb"]
        if (
            type(value["frm"]) is not int
            or type(value["to"]) is not int
            or type(labels) is not list
            or len(labels) != 3
        ):
            raise GlobalGCEHPCExactError("typed DFS edge value is malformed")
        return cls(
            value["frm"],
            value["to"],
            tuple(
                _normalize_scalar(label, field="DFS label") for label in labels
            ),
        )


def typed_dfs_code(value: Iterable[Any]) -> tuple[TypedDFSEdge, ...]:
    result: list[TypedDFSEdge] = []
    for edge in value:
        labels = tuple(
            _normalize_scalar(label, field="DFS label") for label in edge.vevlb
        )
        if len(labels) != 3 or type(edge.frm) is not int or type(edge.to) is not int:
            raise GlobalGCEHPCExactError("official DFS edge has an invalid shape")
        result.append(TypedDFSEdge(edge.frm, edge.to, labels))
    if not result:
        raise GlobalGCEHPCExactError("DFS code may not be empty")
    return tuple(result)


def dfs_code_to_json(value: Sequence[TypedDFSEdge]) -> list[dict[str, Any]]:
    return [edge.to_json() for edge in value]


def dfs_code_from_json(value: Any) -> tuple[TypedDFSEdge, ...]:
    if type(value) is not list or not value:
        raise GlobalGCEHPCExactError("typed DFS code must be a non-empty list")
    return tuple(TypedDFSEdge.from_json(edge) for edge in value)


def dfs_code_sha256(value: Sequence[TypedDFSEdge]) -> str:
    return canonical_sha256(dfs_code_to_json(value))


def _is_prefix(
    prefix: Sequence[TypedDFSEdge], value: Sequence[TypedDFSEdge]
) -> bool:
    return len(prefix) <= len(value) and tuple(prefix) == tuple(value[: len(prefix)])


def load_graph_jsonl(path: str | Path) -> tuple[list[Any], dict[str, Any]]:
    """Load graph bytes without changing row, node, or edge insertion order."""

    source = Path(path).expanduser().resolve(strict=True)
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - dependency preflight
        raise GlobalGCEHPCExactError("networkx is required") from exc
    graphs: list[Any] = []
    graph_ids: list[int | str] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, text in enumerate(stream, start=1):
            try:
                row = json.loads(
                    text,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON constant {value}")
                    ),
                )
            except (ValueError, json.JSONDecodeError) as exc:
                raise GlobalGCEHPCExactError(
                    f"invalid graph JSON at line {line_number}"
                ) from exc
            if type(row) is not dict or set(row) != {"graph_id", "nodes", "edges"}:
                raise GlobalGCEHPCExactError("graph JSONL row schema mismatch")
            graph_id = _normalize_scalar(row["graph_id"], field="graph_id")
            if type(graph_id) is not int or graph_id != line_number - 1:
                raise GlobalGCEHPCExactError(
                    "graph_id must equal its zero-based JSONL row index"
                )
            if graph_id in graph_ids:
                raise GlobalGCEHPCExactError("duplicate graph_id")
            graph_ids.append(graph_id)
            if type(row["nodes"]) is not list or not row["nodes"]:
                raise GlobalGCEHPCExactError("graph must contain nodes")
            if type(row["edges"]) is not list:
                raise GlobalGCEHPCExactError("graph edges must be a list")
            graph = nx.Graph()
            node_ids: list[int] = []
            for node in row["nodes"]:
                if type(node) is not dict or set(node) != {"id", "label"}:
                    raise GlobalGCEHPCExactError("node schema mismatch")
                node_id = node["id"]
                if type(node_id) is not int or node_id < 0:
                    raise GlobalGCEHPCExactError("node id is invalid")
                if type(node["label"]) is not int:
                    raise GlobalGCEHPCExactError(
                        "production graph node label must be an integer"
                    )
                node_ids.append(node_id)
                graph.add_node(
                    node_id,
                    label=_normalize_scalar(node["label"], field="node label"),
                )
            if node_ids != list(range(len(node_ids))):
                raise GlobalGCEHPCExactError(
                    "nodes must preserve consecutive native insertion order"
                )
            for edge in row["edges"]:
                if type(edge) is not dict or set(edge) != {
                    "source",
                    "target",
                    "label",
                }:
                    raise GlobalGCEHPCExactError("edge schema mismatch")
                left, right = edge["source"], edge["target"]
                if (
                    type(left) is not int
                    or type(right) is not int
                    or left not in graph
                    or right not in graph
                    or left == right
                    or graph.has_edge(left, right)
                ):
                    raise GlobalGCEHPCExactError("edge endpoints are invalid")
                if type(edge["label"]) is not int:
                    raise GlobalGCEHPCExactError(
                        "production graph edge label must be an integer"
                    )
                graph.add_edge(
                    left,
                    right,
                    label=_normalize_scalar(edge["label"], field="edge label"),
                )
            if not nx.is_connected(graph):
                raise GlobalGCEHPCExactError("production graph must be connected")
            graphs.append(graph)
    if not graphs:
        raise GlobalGCEHPCExactError("graph JSONL is empty")
    return graphs, {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "graph_count": len(graphs),
        "ordered_graph_ids_sha256": canonical_sha256(graph_ids),
    }


def _official_source_identity(official_src: str | Path) -> dict[str, Any]:
    root = Path(official_src).expanduser().resolve(strict=True)
    observed: dict[str, dict[str, Any]] = {}
    for relative, expected_sha in OFFICIAL_GSPAN_SHA256.items():
        source = root / relative
        if not source.is_file() or source.is_symlink():
            raise GlobalGCEHPCExactError(
                f"pinned official gSpan source is missing: {relative}"
            )
        digest = sha256_file(source)
        if digest != expected_sha:
            raise GlobalGCEHPCExactError(
                f"pinned official gSpan source hash changed: {relative}"
            )
        observed[relative] = {"bytes": source.stat().st_size, "sha256": digest}
    identity = {
        "commit": OFFICIAL_GLOBALGCE_COMMIT,
        "source_root": str(root),
        "files": observed,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


@contextmanager
def _narrow_official_gspan_import(
    official_src: str | Path,
) -> Iterator[tuple[Any, dict[str, Any]]]:
    """Import only ``gSpan.py`` and ``graph.py`` under a disposable package."""

    identity = _official_source_identity(official_src)
    source_root = Path(identity["source_root"])
    token = f"_globalgce_hpc_exact_{os.getpid()}_{uuid.uuid4().hex}"
    models_name = f"{token}.models"
    gspan_package_name = f"{models_name}.gSpan"
    created: list[str] = []
    try:
        for name, location in (
            (token, source_root),
            (models_name, source_root / "models"),
            (gspan_package_name, source_root / "models" / "gSpan"),
        ):
            package = types.ModuleType(name)
            package.__path__ = [str(location)]  # type: ignore[attr-defined]
            package.__package__ = name
            sys.modules[name] = package
            created.append(name)
        graph_name = f"{gspan_package_name}.graph"
        graph_spec = importlib.util.spec_from_file_location(
            graph_name, source_root / "models/gSpan/graph.py"
        )
        if graph_spec is None or graph_spec.loader is None:
            raise GlobalGCEHPCExactError("cannot construct official graph import")
        graph_module = importlib.util.module_from_spec(graph_spec)
        sys.modules[graph_name] = graph_module
        created.append(graph_name)
        graph_spec.loader.exec_module(graph_module)

        gspan_name = f"{gspan_package_name}.gSpan"
        gspan_spec = importlib.util.spec_from_file_location(
            gspan_name, source_root / "models/gSpan/gSpan.py"
        )
        if gspan_spec is None or gspan_spec.loader is None:
            raise GlobalGCEHPCExactError("cannot construct official gSpan import")
        gspan_module = importlib.util.module_from_spec(gspan_spec)
        sys.modules[gspan_name] = gspan_module
        created.append(gspan_name)
        gspan_spec.loader.exec_module(gspan_module)
        if Path(gspan_module.__file__).resolve() != (
            source_root / "models/gSpan/gSpan.py"
        ).resolve():
            raise GlobalGCEHPCExactError("official gSpan module origin changed")
        yield gspan_module, identity
    finally:
        for name in reversed(created):
            sys.modules.pop(name, None)


def _miner_and_roots(
    module: Any,
    graphs: Sequence[Any],
    *,
    min_support: int,
    min_vertices: int,
    max_vertices: int,
) -> tuple[Any, list[tuple[Any, Any]]]:
    miner = module.gSpan(
        list(graphs),
        min_support,
        min_vertices,
        max_vertices,
        len(graphs),
        where=False,
    )
    miner._read_graphs()
    miner._generate_1edge_frequent_subgraphs()
    roots: dict[Any, Any] = {}
    for graph_id, graph in miner.graphs.items():
        for vertex_id, vertex in graph.vertices.items():
            for edge in miner._get_forward_root_edges(graph, vertex_id):
                key = (vertex.vlb, edge.elb, graph.vertices[edge.to].vlb)
                if key not in roots:
                    roots[key] = module.Projected()
                roots[key].append(module.PDFS(graph_id, edge, None))
    return miner, list(roots.items())


def _root_code(module: Any, labels: Any) -> TypedDFSEdge:
    edge = module.DFSedge(0, 1, labels)
    return typed_dfs_code((edge,))[0]


def _candidate_input_sha256(projected: Sequence[Any]) -> str:
    projections: list[dict[str, Any]] = []
    for projection in projected:
        chain: list[dict[str, Any]] = []
        current = projection
        while current is not None:
            edge = current.edge
            chain.append(
                {
                    "eid": int(edge.eid),
                    "frm": int(edge.frm),
                    "to": int(edge.to),
                    "elb": _normalize_scalar(edge.elb, field="projected edge label"),
                }
            )
            current = current.prev
        projections.append({"gid": int(projection.gid), "edges": list(reversed(chain))})
    return canonical_sha256(projections)


def _candidate_observation(miner: Any, projected: Any) -> dict[str, Any]:
    support = int(miner._get_support(projected))
    if support < int(miner._min_support):
        status = "REJECTED_MIN_SUPPORT"
    elif not bool(miner._is_min()):
        status = "REJECTED_NON_MINIMAL"
    else:
        status = "ACCEPTED"
    return {
        "support": support,
        "status": status,
        "candidate_input_sha256": _candidate_input_sha256(projected),
    }


def _unit_id(root_index: int, unit_type: str, code: Sequence[TypedDFSEdge]) -> str:
    return f"r{root_index:04d}-{unit_type.lower()}-{dfs_code_sha256(code)[:20]}"


def _unit_payload(
    *,
    root_index: int,
    unit_type: str,
    code: Sequence[TypedDFSEdge],
    observation: Mapping[str, Any] | None,
    support_hint: int,
) -> dict[str, Any]:
    payload = {
        "partition_id": _unit_id(root_index, unit_type, code),
        "partition_type": unit_type,
        "root_index": root_index,
        "dfs_code": dfs_code_to_json(code),
        "dfs_code_sha256": dfs_code_sha256(code),
        "support_hint": support_hint,
        "expected_candidate": dict(observation) if observation is not None else None,
    }
    return payload


def _enumerate_split_root_units(
    *,
    miner: Any,
    module: Any,
    root_index: int,
    root_labels: Any,
    projected: Any,
    split_depth: int,
) -> list[dict[str, Any]]:
    if split_depth < 2:
        raise GlobalGCEHPCExactError("split depth must be at least two")
    gspan_class = module.gSpan
    original_subgraph_mining = gspan_class._subgraph_mining
    original_report = gspan_class._report
    units: list[dict[str, Any]] = []

    def no_report(_self: Any, _projected: Any) -> None:
        return None

    def enumerate_node(self: Any, current_projected: Any) -> Any:
        code = typed_dfs_code(self._DFScode)
        observation = _candidate_observation(self, current_projected)
        accepted = observation["status"] == "ACCEPTED"
        if accepted and len(code) >= split_depth:
            units.append(
                _unit_payload(
                    root_index=root_index,
                    unit_type="PREFIX_SUBTREE",
                    code=code,
                    observation=observation,
                    support_hint=int(observation["support"]),
                )
            )
            return self
        units.append(
            _unit_payload(
                root_index=root_index,
                unit_type="PREFIX_HEADER",
                code=code,
                observation=observation,
                support_hint=max(int(observation["support"]), 1),
            )
        )
        if not accepted:
            return self
        return original_subgraph_mining(self, current_projected)

    gspan_class._subgraph_mining = enumerate_node
    gspan_class._report = no_report
    try:
        miner._DFScode.append(module.DFSedge(0, 1, root_labels))
        miner._subgraph_mining(projected)
        miner._DFScode.pop()
    finally:
        gspan_class._subgraph_mining = original_subgraph_mining
        gspan_class._report = original_report
    if not units:
        raise GlobalGCEHPCExactError("split root produced no partition units")
    return units


def _assign_shards(
    units: Sequence[Mapping[str, Any]], shard_count: int
) -> list[dict[str, Any]]:
    if type(shard_count) is not int or shard_count < 1 or shard_count > len(units):
        raise GlobalGCEHPCExactError("shard_count must be in [1, partition_count]")
    loads = [0 for _ in range(shard_count)]
    assignment: dict[str, int] = {}
    weighted = sorted(
        units,
        key=lambda row: (-int(row["support_hint"]), int(row["global_partition_order"])),
    )
    for unit in weighted:
        selected = min(range(shard_count), key=lambda index: (loads[index], index))
        assignment[str(unit["partition_id"])] = selected
        loads[selected] += max(int(unit["support_hint"]), 1)
    return [{**dict(unit), "shard_index": assignment[str(unit["partition_id"])]} for unit in units]


def _validate_partition_geometry(units: Sequence[Mapping[str, Any]]) -> None:
    ids: set[str] = set()
    orders: list[int] = []
    frontier: dict[int, list[tuple[TypedDFSEdge, ...]]] = {}
    for unit in units:
        unit_id = unit.get("partition_id")
        unit_type = unit.get("partition_type")
        if type(unit_id) is not str or not unit_id or unit_id in ids:
            raise GlobalGCEHPCExactError("partition IDs are missing or duplicated")
        ids.add(unit_id)
        if unit_type not in UNIT_TYPES:
            raise GlobalGCEHPCExactError("unknown partition type")
        code = dfs_code_from_json(unit.get("dfs_code"))
        if unit.get("dfs_code_sha256") != dfs_code_sha256(code):
            raise GlobalGCEHPCExactError("partition DFS code hash mismatch")
        if type(unit.get("root_index")) is not int or unit["root_index"] < 0:
            raise GlobalGCEHPCExactError("partition root index is invalid")
        if type(unit.get("global_partition_order")) is not int:
            raise GlobalGCEHPCExactError("partition order is invalid")
        orders.append(unit["global_partition_order"])
        if unit_type == "PREFIX_SUBTREE":
            frontier.setdefault(unit["root_index"], []).append(code)
    if sorted(orders) != list(range(len(units))):
        raise GlobalGCEHPCExactError("partition order is not complete and consecutive")
    for codes in frontier.values():
        for index, left in enumerate(codes):
            for right in codes[index + 1 :]:
                if _is_prefix(left, right) or _is_prefix(right, left):
                    raise GlobalGCEHPCExactError("prefix subtrees overlap")


def _validate_input_bundle_manifest(
    path: str | Path,
    *,
    graph_identity: Mapping[str, Any],
    expected_commit: str,
    min_support: int,
    min_vertices: int,
    max_vertices: int,
    top_k: int,
) -> dict[str, Any]:
    """Validate and freeze the narrow train-only T8 transfer provenance."""

    source = Path(path).expanduser().resolve(strict=True)
    payload = _load_json(source)
    if type(payload) is not dict:
        raise GlobalGCEHPCExactError("input bundle manifest is not an object")
    claimed = payload.get("manifest_sha256")
    without_hash = dict(payload)
    without_hash.pop("manifest_sha256", None)
    if not _is_sha256(claimed) or _canonical_jsonl_sha256(without_hash) != claimed:
        raise GlobalGCEHPCExactError("input bundle manifest self-hash mismatch")
    normalized_commit = expected_commit.lower()
    if not _is_commit(normalized_commit):
        raise GlobalGCEHPCExactError("expected execution commit must be exact 40-hex")
    mining = payload.get("mining_config")
    transaction = payload.get("transaction_binding")
    transfer = payload.get("transfer_policy")
    if (
        payload.get("state") != "PASS"
        or payload.get("dataset") != "tastemolnet"
        or payload.get("method") != "globalgce"
        or payload.get("stage") != "EXACT_GSPAN_CPU_INPUT"
        or payload.get("route_kind")
        != "T8_T13_GRADE_GLOBALGCE_EXACT_CPU_OFFLOAD"
        or payload.get("split_scope") != "train_only"
        or payload.get("calibration_payload_included") is not False
        or payload.get("test_payload_included") is not False
        or payload.get("matrix_publication_allowed_from_hpc") is not False
        or payload.get("official_globalgce_commit") != OFFICIAL_GLOBALGCE_COMMIT
        or not _is_commit(payload.get("source_commit"))
        or type(mining) is not dict
        or type(transaction) is not dict
        or type(transfer) is not dict
    ):
        raise GlobalGCEHPCExactError("input bundle scientific provenance is invalid")
    expected_mining = {
        "min_support": min_support,
        "min_vertices": min_vertices,
        "max_vertices": max_vertices,
        "top_k": top_k,
    }
    if any(mining.get(key) != value for key, value in expected_mining.items()):
        raise GlobalGCEHPCExactError("CLI mining configuration differs from input bundle")
    if (
        mining.get("source_label") != 1
        or mining.get("seed") != 7
        or mining.get("epochs") != 100
        or mining.get("exact") is not True
        or mining.get("approximate_pruning") is not False
        or transaction.get("shared_transaction_database") is not True
        or transaction.get("target_labels") != [0, 2]
        or transaction.get("target_semantics_do_not_modify_transaction_database")
        is not True
        or transaction.get("graph_jsonl_sha256") != graph_identity["sha256"]
        or transaction.get("graph_count") != graph_identity["graph_count"]
        or transfer.get("source_data_is_train_only_derived") is not True
        or transfer.get("hpc_may_modify_autodl_matrix") is not False
    ):
        raise GlobalGCEHPCExactError("input bundle T8/T13-grade contract mismatch")
    declared_config_sha = payload.get("mining_config_sha256")
    if not _is_sha256(declared_config_sha):
        raise GlobalGCEHPCExactError("input bundle mining config hash is missing")
    # The input builder uses canonical JSON followed by one newline for this hash.
    if _canonical_jsonl_sha256(mining) != declared_config_sha:
        raise GlobalGCEHPCExactError("input bundle mining config self-hash mismatch")
    files = payload.get("files")
    graph_entries = (
        [row for row in files if type(row) is dict and row.get("role") == "graph_jsonl"]
        if type(files) is list
        else []
    )
    if (
        len(graph_entries) != 1
        or graph_entries[0].get("sha256") != graph_identity["sha256"]
    ):
        raise GlobalGCEHPCExactError("input manifest graph payload binding mismatch")
    root_count = mining.get("root_count")
    if type(root_count) is not int or root_count < 1:
        raise GlobalGCEHPCExactError("input bundle root_count is invalid")
    provenance = {
        "input_manifest": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "file_sha256": sha256_file(source),
            "manifest_sha256": claimed,
        },
        "execution_commit": normalized_commit,
        "source_commit": payload["source_commit"].lower(),
        "route_kind": payload["route_kind"],
        "dataset": "tastemolnet",
        "method": "globalgce",
        "split_scope": "train_only",
        "source_label": 1,
        "target_branches": [0, 2],
        "seed": 7,
        "epochs": 100,
        "mining_config_sha256": declared_config_sha,
        "hpc_runtime_config": payload.get("hpc_runtime_config"),
        "calibration_loaded": False,
        "test_loaded": False,
        "matrix_write_enabled": False,
        "expected_root_count": root_count,
    }
    provenance["provenance_sha256"] = canonical_sha256(provenance)
    return provenance


def build_partition_manifest(
    *,
    graph_jsonl: str | Path,
    input_manifest: str | Path,
    expected_commit: str,
    official_src: str | Path,
    output: str | Path,
    shard_count: int,
    min_support: int,
    min_vertices: int,
    max_vertices: int,
    top_k: int,
    split_root_indices: Sequence[int],
    split_depth: int,
    canary_root_indices: Sequence[int],
    included_root_indices: Sequence[int] | None = None,
    included_unit_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Enumerate a shallow exact cut and publish its disjoint/complete proof."""

    if (
        type(min_support) is not int
        or min_support <= 0
        or type(min_vertices) is not int
        or min_vertices <= 0
        or type(max_vertices) is not int
        or max_vertices < min_vertices
        or type(top_k) is not int
        or top_k <= 0
    ):
        raise GlobalGCEHPCExactError("gSpan bounds are invalid")
    destination = Path(output).expanduser().absolute()
    if destination.exists():
        raise GlobalGCEHPCExactError("partition manifest output must be fresh")
    graphs, graph_identity = load_graph_jsonl(graph_jsonl)
    provenance = _validate_input_bundle_manifest(
        input_manifest,
        graph_identity=graph_identity,
        expected_commit=expected_commit,
        min_support=min_support,
        min_vertices=min_vertices,
        max_vertices=max_vertices,
        top_k=top_k,
    )
    with _narrow_official_gspan_import(official_src) as (module, official_identity):
        miner, roots = _miner_and_roots(
            module,
            graphs,
            min_support=min_support,
            min_vertices=min_vertices,
            max_vertices=max_vertices,
        )
        if not roots:
            raise GlobalGCEHPCExactError("official gSpan root universe is empty")
        split_roots = tuple(sorted(set(split_root_indices)))
        canary_roots = tuple(sorted(set(canary_root_indices)))
        whole_roots = tuple(
            range(len(roots))
            if included_root_indices is None
            else sorted(set(included_root_indices))
        )
        selected_unit_ids = tuple(dict.fromkeys(included_unit_ids or ()))
        selected_partition_scope = bool(selected_unit_ids)
        if (
            not split_roots
            or any(type(value) is not int or value < 0 for value in split_roots)
            or split_roots[-1] >= len(roots)
            or not canary_roots
            or any(type(value) is not int or value < 0 for value in canary_roots)
            or canary_roots[-1] >= len(roots)
            or not whole_roots
            or any(type(value) is not int or value < 0 for value in whole_roots)
            or whole_roots[-1] >= len(roots)
            or (
                not selected_partition_scope
                and not set(split_roots).issubset(whole_roots)
            )
            or (
                selected_partition_scope
                and bool(set(split_roots).intersection(whole_roots))
            )
            or any(type(value) is not str or not value for value in selected_unit_ids)
        ):
            raise GlobalGCEHPCExactError(
                "included/split/canary root selection is invalid"
            )
        units: list[dict[str, Any]] = []
        selected_units_found: set[str] = set()
        root_descriptors: list[dict[str, Any]] = []
        for root_index, (labels, projected) in enumerate(roots):
            root_code = (_root_code(module, labels),)
            root_descriptors.append(
                {
                    "root_index": root_index,
                    "dfs_code": dfs_code_to_json(root_code),
                    "dfs_code_sha256": dfs_code_sha256(root_code),
                    "projected_support": int(miner._get_support(projected)),
                }
            )
            if root_index in split_roots:
                split_miner, split_universe = _miner_and_roots(
                    module,
                    graphs,
                    min_support=min_support,
                    min_vertices=min_vertices,
                    max_vertices=max_vertices,
                )
                split_labels, split_projected = split_universe[root_index]
                split_units = _enumerate_split_root_units(
                        miner=split_miner,
                        module=module,
                        root_index=root_index,
                        root_labels=split_labels,
                        projected=split_projected,
                        split_depth=split_depth,
                    )
                if selected_partition_scope:
                    for unit in split_units:
                        if unit["partition_id"] in selected_unit_ids:
                            if unit["partition_type"] != "PREFIX_SUBTREE":
                                raise GlobalGCEHPCExactError(
                                    "selected canary unit must be a PREFIX_SUBTREE"
                                )
                            units.append(unit)
                            selected_units_found.add(unit["partition_id"])
                elif root_index in whole_roots:
                    units.extend(split_units)
            elif root_index in whole_roots:
                units.append(
                    _unit_payload(
                        root_index=root_index,
                        unit_type="ROOT_SUBTREE",
                        code=root_code,
                        observation=None,
                        support_hint=max(int(miner._get_support(projected)), 1),
                    )
                )
        if selected_partition_scope and selected_units_found != set(selected_unit_ids):
            missing = sorted(set(selected_unit_ids).difference(selected_units_found))
            raise GlobalGCEHPCExactError(
                f"selected canary prefix unit IDs were not found: {missing}"
            )
        covered_roots = tuple(sorted({int(unit["root_index"]) for unit in units}))
        if selected_partition_scope and canary_roots != covered_roots:
            raise GlobalGCEHPCExactError(
                "canary roots must exactly describe the selected search space"
            )
        if not selected_partition_scope and not set(canary_roots).issubset(
            covered_roots
        ):
            raise GlobalGCEHPCExactError(
                "canary roots are outside the selected search space"
            )
        if int(provenance["expected_root_count"]) != len(root_descriptors):
            raise GlobalGCEHPCExactError(
                "official root universe count differs from input bundle"
            )
    ordered = [
        {**unit, "global_partition_order": index}
        for index, unit in enumerate(units)
    ]
    assigned = _assign_shards(ordered, shard_count)
    _validate_partition_geometry(assigned)
    type_counts = {
        name: sum(unit["partition_type"] == name for unit in assigned)
        for name in sorted(UNIT_TYPES)
    }
    if (
        not selected_partition_scope
        and (
            type_counts["PREFIX_HEADER"] < 1
            or type_counts["PREFIX_SUBTREE"] < 1
        )
    ):
        raise GlobalGCEHPCExactError(
            "selected split roots did not yield both header and subtree partitions"
        )
    if selected_partition_scope and type_counts["PREFIX_SUBTREE"] != len(
        selected_unit_ids
    ):
        raise GlobalGCEHPCExactError("selected prefix canary did not close exactly")
    root_universe_sha = canonical_sha256(root_descriptors)
    scientific_input = {
        "graph_input": {
            key: graph_identity[key]
            for key in (
                "bytes",
                "sha256",
                "graph_count",
                "ordered_graph_ids_sha256",
            )
        },
        "official_gspan": {
            "commit": official_identity["commit"],
            "files": official_identity["files"],
            "identity_sha256": official_identity["identity_sha256"],
        },
        "provenance": provenance,
        "configuration": {
            "min_support": min_support,
            "min_vertices": min_vertices,
            "max_vertices": max_vertices,
            "is_undirected": True,
            "exact_top_k_pruning": False,
            "approximation_used": False,
            "top_k": top_k,
        },
        "root_universe_sha256": root_universe_sha,
    }
    completeness_payload = {
        "root_count": len(root_descriptors),
        "included_root_count": len(covered_roots),
        "whole_root_indices": list(whole_roots),
        "selected_partition_ids": list(selected_unit_ids),
        "root_universe_sha256": root_universe_sha,
        "partition_count": len(assigned),
        "ordered_partition_ids": [unit["partition_id"] for unit in assigned],
        "split_root_indices": list(split_roots),
        "included_root_indices": list(covered_roots),
        "split_depth": split_depth,
        "partition_type_counts": type_counts,
        "disjoint": True,
        "complete": True,
        "scientific_search_pruned": False,
    }
    payload = {
        "schema_version": PARTITION_MANIFEST_SCHEMA,
        "status": "PASS",
        "scope": (
            "FULL_ROOT_UNIVERSE"
            if not selected_partition_scope
            and whole_roots == tuple(range(len(roots)))
            else (
                "SELECTED_PARTITION_CANARY"
                if selected_partition_scope
                else "SELECTED_ROOTS_CANARY"
            )
        ),
        "created_at": _utc_now(),
        "graph_input": graph_identity,
        "official_gspan": official_identity,
        "provenance": provenance,
        "configuration": {
            "min_support": min_support,
            "min_vertices": min_vertices,
            "max_vertices": max_vertices,
            "is_undirected": True,
            "exact_top_k_pruning": False,
            "approximation_used": False,
            "top_k": top_k,
        },
        "scientific_input_sha256": canonical_sha256(scientific_input),
        "root_universe": root_descriptors,
        "root_universe_sha256": root_universe_sha,
        "split_root_indices": list(split_roots),
        "included_root_indices": list(covered_roots),
        "whole_root_indices": list(whole_roots),
        "selected_partition_ids": list(selected_unit_ids),
        "split_depth": split_depth,
        "canary_root_indices": list(canary_roots),
        "shard_count": shard_count,
        "partitions": assigned,
        "completeness_proof": {
            **completeness_payload,
            "proof_sha256": canonical_sha256(completeness_payload),
        },
        "matrix_write_enabled": False,
    }
    payload = _self_hashed(payload, field="manifest_sha256")
    atomic_write_json(destination, payload)
    return payload


def validate_partition_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    payload = _load_json(source)
    if (
        type(payload) is not dict
        or payload.get("schema_version") != PARTITION_MANIFEST_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("matrix_write_enabled") is not False
        or not _is_sha256(payload.get("scientific_input_sha256"))
        or not _is_sha256(payload.get("manifest_sha256"))
    ):
        raise GlobalGCEHPCExactError("partition manifest header is invalid")
    claimed = payload.pop("manifest_sha256")
    observed = canonical_sha256(payload)
    payload["manifest_sha256"] = claimed
    if claimed != observed:
        raise GlobalGCEHPCExactError("partition manifest self-hash mismatch")
    partitions = payload.get("partitions")
    if type(partitions) is not list or not partitions:
        raise GlobalGCEHPCExactError("partition manifest has no partitions")
    _validate_partition_geometry(partitions)
    if type(payload.get("shard_count")) is not int or payload["shard_count"] < 1:
        raise GlobalGCEHPCExactError("partition shard count is invalid")
    if any(
        type(unit.get("shard_index")) is not int
        or unit["shard_index"] < 0
        or unit["shard_index"] >= payload["shard_count"]
        for unit in partitions
    ):
        raise GlobalGCEHPCExactError("partition shard assignment is invalid")
    included = payload.get("included_root_indices")
    whole = payload.get("whole_root_indices")
    selected_partition_ids = payload.get("selected_partition_ids")
    split = payload.get("split_root_indices")
    canary = payload.get("canary_root_indices")
    scope = payload.get("scope")
    if (
        scope
        not in {
            "FULL_ROOT_UNIVERSE",
            "SELECTED_ROOTS_CANARY",
            "SELECTED_PARTITION_CANARY",
        }
        or type(included) is not list
        or not included
        or included != sorted(set(included))
        or type(whole) is not list
        or not whole
        or whole != sorted(set(whole))
        or type(selected_partition_ids) is not list
        or type(split) is not list
        or not split
        or split != sorted(set(split))
        or type(canary) is not list
        or not canary
        or canary != sorted(set(canary))
    ):
        raise GlobalGCEHPCExactError("manifest root scopes are invalid")
    units_by_root: dict[int, list[Mapping[str, Any]]] = {}
    for unit in partitions:
        units_by_root.setdefault(int(unit["root_index"]), []).append(unit)
    if set(units_by_root) != set(included):
        raise GlobalGCEHPCExactError("partition roots do not cover the included scope")
    split_depth = payload.get("split_depth")
    if type(split_depth) is not int or split_depth < 2:
        raise GlobalGCEHPCExactError("manifest split depth is invalid")
    if scope == "SELECTED_PARTITION_CANARY":
        if (
            not selected_partition_ids
            or len(selected_partition_ids) != len(set(selected_partition_ids))
            or set(whole).intersection(split)
            or set(included) != set(whole).union(split)
            or canary != included
        ):
            raise GlobalGCEHPCExactError("selected-prefix canary scope is invalid")
    elif (
        selected_partition_ids
        or not set(split).issubset(whole)
        or included != whole
        or not set(canary).issubset(included)
    ):
        raise GlobalGCEHPCExactError("whole-root partition scope is invalid")
    for root_index in included:
        root_units = units_by_root[root_index]
        types_for_root = {row["partition_type"] for row in root_units}
        if root_index in whole and root_index not in split:
            if len(root_units) != 1 or types_for_root != {"ROOT_SUBTREE"}:
                raise GlobalGCEHPCExactError("ordinary root is not one exact subtree")
            continue
        if scope == "SELECTED_PARTITION_CANARY":
            if (
                types_for_root != {"PREFIX_SUBTREE"}
                or any(len(row["dfs_code"]) != split_depth for row in root_units)
            ):
                raise GlobalGCEHPCExactError(
                    "selected-prefix canary contains a non-frontier unit"
                )
            continue
        if "ROOT_SUBTREE" in types_for_root:
            raise GlobalGCEHPCExactError("split root also owns a full root subtree")
        root_headers = [
            row
            for row in root_units
            if row["partition_type"] == "PREFIX_HEADER"
            and len(row["dfs_code"]) == 1
        ]
        frontier = [
            row for row in root_units if row["partition_type"] == "PREFIX_SUBTREE"
        ]
        if len(root_headers) != 1 or not frontier or any(
            len(row["dfs_code"]) != split_depth for row in frontier
        ):
            raise GlobalGCEHPCExactError("split-root DFS cut is incomplete")
    if selected_partition_ids != [
        row["partition_id"]
        for row in partitions
        if row["partition_type"] == "PREFIX_SUBTREE"
    ] and scope == "SELECTED_PARTITION_CANARY":
        raise GlobalGCEHPCExactError("selected prefix IDs do not match partition units")
    graph = payload.get("graph_input")
    if type(graph) is not dict or not _is_sha256(graph.get("sha256")):
        raise GlobalGCEHPCExactError("partition graph identity is invalid")
    graph_path = Path(graph.get("path", "")).resolve(strict=True)
    if sha256_file(graph_path) != graph["sha256"]:
        raise GlobalGCEHPCExactError("partition graph input bytes changed")
    observed_official = _official_source_identity(
        payload.get("official_gspan", {}).get("source_root", "")
    )
    if observed_official != payload.get("official_gspan"):
        raise GlobalGCEHPCExactError("partition official source identity changed")
    provenance = payload.get("provenance")
    if type(provenance) is not dict or not _is_sha256(
        provenance.get("provenance_sha256")
    ):
        raise GlobalGCEHPCExactError("partition provenance is missing")
    provenance_without_hash = dict(provenance)
    provenance_sha = provenance_without_hash.pop("provenance_sha256")
    if canonical_sha256(provenance_without_hash) != provenance_sha:
        raise GlobalGCEHPCExactError("partition provenance self-hash mismatch")
    input_identity = provenance.get("input_manifest")
    if type(input_identity) is not dict:
        raise GlobalGCEHPCExactError("input manifest identity is missing")
    input_path = Path(str(input_identity.get("path") or "")).resolve(strict=True)
    if (
        sha256_file(input_path) != input_identity.get("file_sha256")
        or input_path.stat().st_size != input_identity.get("bytes")
        or provenance.get("route_kind")
        != "T8_T13_GRADE_GLOBALGCE_EXACT_CPU_OFFLOAD"
        or provenance.get("split_scope") != "train_only"
        or provenance.get("source_label") != 1
        or provenance.get("target_branches") != [0, 2]
        or provenance.get("calibration_loaded") is not False
        or provenance.get("test_loaded") is not False
        or provenance.get("matrix_write_enabled") is not False
        or not _is_commit(provenance.get("execution_commit"))
    ):
        raise GlobalGCEHPCExactError("partition provenance binding changed")
    scientific_input = {
        "graph_input": {
            key: graph[key]
            for key in (
                "bytes",
                "sha256",
                "graph_count",
                "ordered_graph_ids_sha256",
            )
        },
        "official_gspan": {
            "commit": observed_official["commit"],
            "files": observed_official["files"],
            "identity_sha256": observed_official["identity_sha256"],
        },
        "provenance": provenance,
        "configuration": payload["configuration"],
        "root_universe_sha256": payload["root_universe_sha256"],
    }
    if canonical_sha256(scientific_input) != payload["scientific_input_sha256"]:
        raise GlobalGCEHPCExactError("scientific input identity mismatch")
    proof = payload.get("completeness_proof")
    if type(proof) is not dict or not _is_sha256(proof.get("proof_sha256")):
        raise GlobalGCEHPCExactError("partition completeness proof is missing")
    proof_copy = dict(proof)
    proof_sha = proof_copy.pop("proof_sha256")
    if canonical_sha256(proof_copy) != proof_sha:
        raise GlobalGCEHPCExactError("partition completeness proof hash mismatch")
    if proof.get("disjoint") is not True or proof.get("complete") is not True:
        raise GlobalGCEHPCExactError("partition is not disjoint and complete")
    if (
        proof.get("root_count") != len(payload.get("root_universe", ()))
        or proof.get("included_root_count") != len(included)
        or proof.get("whole_root_indices") != whole
        or proof.get("selected_partition_ids") != selected_partition_ids
        or proof.get("root_universe_sha256") != payload["root_universe_sha256"]
        or proof.get("partition_count") != len(partitions)
        or proof.get("ordered_partition_ids")
        != [unit["partition_id"] for unit in partitions]
        or proof.get("split_root_indices") != split
        or proof.get("included_root_indices") != included
        or proof.get("split_depth") != split_depth
    ):
        raise GlobalGCEHPCExactError("partition completeness claims do not close")
    return payload


class _RecordWriter:
    def __init__(
        self,
        root: Path,
        *,
        flush_every: int,
        target_branches: Sequence[int],
    ) -> None:
        if flush_every < 1:
            raise GlobalGCEHPCExactError("flush_every must be positive")
        self.root = root
        self.flush_every = flush_every
        self.target_branches = list(target_branches)
        self.events: TextIO = (root / "events.jsonl").open("x", encoding="utf-8")
        self.patterns: TextIO = (root / "patterns.jsonl").open("x", encoding="utf-8")
        self.event_count = 0
        self.pattern_count = 0
        self.rejection_count = 0

    def _write(self, stream: TextIO, payload: Mapping[str, Any]) -> None:
        stream.write(_canonical_bytes(dict(payload)).decode("ascii") + "\n")

    def event(
        self,
        *,
        unit: Mapping[str, Any],
        code: Sequence[TypedDFSEdge],
        observation: Mapping[str, Any],
    ) -> None:
        self._write(
            self.events,
            {
                "local_preorder": self.event_count,
                "partition_id": unit["partition_id"],
                "root_index": unit["root_index"],
                "dfs_code": dfs_code_to_json(code),
                "dfs_code_sha256": dfs_code_sha256(code),
                "target_branches": self.target_branches,
                **dict(observation),
            },
        )
        self.event_count += 1
        if observation["status"] != "ACCEPTED":
            self.rejection_count += 1
        self._periodic_flush()

    def pattern(
        self,
        *,
        unit: Mapping[str, Any],
        code: Sequence[TypedDFSEdge],
        support: int,
    ) -> None:
        self._write(
            self.patterns,
            {
                "local_preorder": self.pattern_count,
                "partition_id": unit["partition_id"],
                "root_index": unit["root_index"],
                "dfs_code": dfs_code_to_json(code),
                "dfs_code_sha256": dfs_code_sha256(code),
                "pattern_sha256": canonical_sha256(
                    {"dfs_code": dfs_code_to_json(code), "undirected": True}
                ),
                "support": support,
                "target_branches": self.target_branches,
            },
        )
        self.pattern_count += 1
        self._periodic_flush()

    def _periodic_flush(self) -> None:
        if (self.event_count + self.pattern_count) % self.flush_every == 0:
            for stream in (self.events, self.patterns):
                stream.flush()
                os.fsync(stream.fileno())

    def close(self) -> None:
        for stream in (self.events, self.patterns):
            if stream.closed:
                continue
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()


def _verify_live_root_universe(
    manifest: Mapping[str, Any], module: Any, roots: Sequence[tuple[Any, Any]]
) -> None:
    observed = []
    for index, (labels, projected) in enumerate(roots):
        code = (_root_code(module, labels),)
        observed.append(
            {
                "root_index": index,
                "dfs_code": dfs_code_to_json(code),
                "dfs_code_sha256": dfs_code_sha256(code),
                "projected_support": len({int(row.gid) for row in projected}),
            }
        )
    if (
        canonical_sha256(observed) != manifest.get("root_universe_sha256")
        or observed != manifest.get("root_universe")
    ):
        raise GlobalGCEHPCExactError("live official root universe changed")


def _publish_staged_unit(
    source: Path,
    destination: Path,
    *,
    manifest: Mapping[str, Any],
    unit: Mapping[str, Any],
) -> None:
    """Copy one sealed scratch unit to persistent storage, then atomically expose it."""

    persistent_temporary = destination.parent / (
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.copying"
    )
    persistent_temporary.mkdir(mode=0o700)
    try:
        for name in ("events.jsonl", "patterns.jsonl", "partition_manifest.json"):
            source_file = source / name
            target_file = persistent_temporary / name
            shutil.copyfile(source_file, target_file)
            descriptor = os.open(target_file, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        _fsync_directory(persistent_temporary)
        validate_unit_result(
            persistent_temporary, manifest=manifest, expected_unit=unit
        )
        os.rename(persistent_temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        # Keep the uniquely named incomplete copy as evidence.  It is never a
        # resume boundary and can never be mistaken for the sealed unit path.
        raise


def _execute_partition_unit(
    manifest: Mapping[str, Any],
    unit: Mapping[str, Any],
    destination: Path,
    *,
    flush_every: int,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    if destination.is_dir():
        return validate_unit_result(destination, manifest=manifest, expected_unit=unit)
    if destination.exists():
        raise GlobalGCEHPCExactError("partition destination is not a directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = destination.parent if scratch_root is None else scratch_root
    staging_parent.mkdir(parents=True, exist_ok=True)
    temporary = staging_parent / (
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.incomplete"
    )
    temporary.mkdir(mode=0o700)
    writer = _RecordWriter(
        temporary,
        flush_every=flush_every,
        target_branches=manifest["provenance"]["target_branches"],
    )
    try:
        graphs, graph_identity = load_graph_jsonl(manifest["graph_input"]["path"])
        if graph_identity != manifest["graph_input"]:
            raise GlobalGCEHPCExactError("partition graph identity changed")
        with _narrow_official_gspan_import(
            manifest["official_gspan"]["source_root"]
        ) as (module, official_identity):
            if official_identity != manifest["official_gspan"]:
                raise GlobalGCEHPCExactError("partition official source changed")
            config = manifest["configuration"]
            miner, roots = _miner_and_roots(
                module,
                graphs,
                min_support=int(config["min_support"]),
                min_vertices=int(config["min_vertices"]),
                max_vertices=int(config["max_vertices"]),
            )
            _verify_live_root_universe(manifest, module, roots)
            root_index = int(unit["root_index"])
            root_labels, root_projected = roots[root_index]
            target = dfs_code_from_json(unit["dfs_code"])
            unit_type = str(unit["partition_type"])
            gspan_class = module.gSpan
            original_subgraph_mining = gspan_class._subgraph_mining
            original_report = gspan_class._report
            full_subtree_depth: int | None = None

            def capture_report(self: Any, _projected: Any) -> None:
                # An exact prefix unit replays accepted ancestors only to
                # reconstruct its projected database.  Those ancestor reports
                # belong exclusively to their PREFIX_HEADER units.
                if full_subtree_depth is None:
                    return
                if self._DFScode.get_num_vertices() < self._min_num_vertices:
                    return
                code = typed_dfs_code(self._DFScode)
                writer.pattern(
                    unit=unit,
                    code=code,
                    support=int(self._support),
                )

            def visit(self: Any, projected: Any) -> Any:
                nonlocal full_subtree_depth
                code = typed_dfs_code(self._DFScode)
                if full_subtree_depth is not None:
                    observation = _candidate_observation(self, projected)
                    writer.event(unit=unit, code=code, observation=observation)
                    if observation["status"] != "ACCEPTED":
                        return self
                    return original_subgraph_mining(self, projected)

                if unit_type == "ROOT_SUBTREE":
                    if len(code) != 1 or code != target:
                        raise GlobalGCEHPCExactError("root partition navigation drift")
                    full_subtree_depth = 1
                    try:
                        return visit(self, projected)
                    finally:
                        full_subtree_depth = None

                if code == target:
                    observation = _candidate_observation(self, projected)
                    expected = unit.get("expected_candidate")
                    if expected is not None and observation != expected:
                        raise GlobalGCEHPCExactError(
                            "partition prefix candidate observation changed"
                        )
                    writer.event(unit=unit, code=code, observation=observation)
                    if unit_type == "PREFIX_HEADER":
                        if (
                            observation["status"] == "ACCEPTED"
                            and self._DFScode.get_num_vertices()
                            >= self._min_num_vertices
                        ):
                            writer.pattern(
                                unit=unit,
                                code=code,
                                support=int(observation["support"]),
                            )
                        return self
                    if observation["status"] != "ACCEPTED":
                        raise GlobalGCEHPCExactError(
                            "prefix subtree stopped being an accepted candidate"
                        )
                    full_subtree_depth = len(code)
                    try:
                        # The official method repeats the same deterministic
                        # support/minimality check, reports this prefix, and
                        # visits every descendant in its original order.
                        return original_subgraph_mining(self, projected)
                    finally:
                        full_subtree_depth = None

                if _is_prefix(code, target):
                    # Navigate through an official accepted ancestor without
                    # emitting it in this unit.  Its own PREFIX_HEADER unit is
                    # the sole owner of that candidate/report event.
                    observation = _candidate_observation(self, projected)
                    if observation["status"] != "ACCEPTED":
                        raise GlobalGCEHPCExactError(
                            "partition target is below a rejected ancestor"
                        )
                    return original_subgraph_mining(self, projected)
                return self

            gspan_class._subgraph_mining = visit
            gspan_class._report = capture_report
            try:
                miner._DFScode.append(module.DFSedge(0, 1, root_labels))
                miner._subgraph_mining(root_projected)
                miner._DFScode.pop()
            finally:
                gspan_class._subgraph_mining = original_subgraph_mining
                gspan_class._report = original_report
        writer.close()
        writer = None  # type: ignore[assignment]
        events_path = temporary / "events.jsonl"
        patterns_path = temporary / "patterns.jsonl"
        if writer is not None:  # pragma: no cover - defensive type narrowing
            raise AssertionError
        result = {
            "schema_version": UNIT_RESULT_SCHEMA,
            "status": "PASS",
            "completed_at": _utc_now(),
            "manifest_sha256": manifest["manifest_sha256"],
            "partition": dict(unit),
            "event_count": _count_jsonl(events_path),
            "pattern_count": _count_jsonl(patterns_path),
            "rejection_count": _count_rejections(events_path),
            "events_sha256": sha256_file(events_path),
            "patterns_sha256": sha256_file(patterns_path),
            "scientific_search_pruned": False,
            "approximation_used": False,
            "single_writer": True,
            "matrix_write_enabled": False,
            "scientific_input_sha256": manifest["scientific_input_sha256"],
            "provenance_sha256": manifest["provenance"]["provenance_sha256"],
            "target_branches": manifest["provenance"]["target_branches"],
            "scratch_staging_used": scratch_root is not None,
        }
        result = _self_hashed(result, field="result_sha256")
        atomic_write_json(temporary / "partition_manifest.json", result)
        _fsync_directory(temporary)
        validate_unit_result(temporary, manifest=manifest, expected_unit=unit)
        if scratch_root is None:
            os.rename(temporary, destination)
            _fsync_directory(destination.parent)
        else:
            _publish_staged_unit(
                temporary, destination, manifest=manifest, unit=unit
            )
            shutil.rmtree(temporary)
        return result
    except BaseException:
        try:
            if writer is not None:
                writer.close()
        finally:
            failure = temporary / "FAILED.json"
            if temporary.is_dir() and not failure.exists():
                atomic_write_json(
                    failure,
                    {
                        "schema_version": "globalgce_hpc_exact_partition_failure_v1",
                        "status": "FAILED",
                        "failed_at": _utc_now(),
                        "partition_id": unit.get("partition_id"),
                    },
                )
        raise


def _count_jsonl(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for _line in stream)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(
                    line,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON constant {token}")
                    ),
                )
            except (ValueError, json.JSONDecodeError) as exc:
                raise GlobalGCEHPCExactError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from exc
            if type(value) is not dict:
                raise GlobalGCEHPCExactError("JSONL record is not an object")
            yield value


def _count_rejections(path: Path) -> int:
    return sum(row.get("status") != "ACCEPTED" for row in _iter_jsonl(path))


def validate_unit_result(
    root: str | Path,
    *,
    manifest: Mapping[str, Any],
    expected_unit: Mapping[str, Any],
) -> dict[str, Any]:
    directory = Path(root).resolve(strict=True)
    result = _load_json(directory / "partition_manifest.json")
    if (
        type(result) is not dict
        or result.get("schema_version") != UNIT_RESULT_SCHEMA
        or result.get("status") != "PASS"
        or result.get("manifest_sha256") != manifest.get("manifest_sha256")
        or result.get("partition") != dict(expected_unit)
        or result.get("matrix_write_enabled") is not False
        or result.get("scientific_input_sha256")
        != manifest.get("scientific_input_sha256")
        or result.get("provenance_sha256")
        != manifest.get("provenance", {}).get("provenance_sha256")
        or result.get("target_branches")
        != manifest.get("provenance", {}).get("target_branches")
        or not _is_sha256(result.get("result_sha256"))
    ):
        raise GlobalGCEHPCExactError("partition result manifest is invalid")
    claimed = result.pop("result_sha256")
    observed = canonical_sha256(result)
    result["result_sha256"] = claimed
    if claimed != observed:
        raise GlobalGCEHPCExactError("partition result self-hash mismatch")
    events = directory / "events.jsonl"
    patterns = directory / "patterns.jsonl"
    if (
        sha256_file(events) != result["events_sha256"]
        or sha256_file(patterns) != result["patterns_sha256"]
        or _count_jsonl(events) != result["event_count"]
        or _count_jsonl(patterns) != result["pattern_count"]
        or _count_rejections(events) != result["rejection_count"]
    ):
        raise GlobalGCEHPCExactError("partition result payload changed")
    return result


def _write_or_validate_run_spec(path: Path, payload: Mapping[str, Any]) -> None:
    expected = _self_hashed(payload, field="run_spec_sha256")
    if path.exists():
        if _load_json(path) != expected:
            raise GlobalGCEHPCExactError("existing run spec differs from requested run")
        return
    atomic_write_json(path, expected)


def run_mining_shard(
    *,
    partition_manifest: str | Path,
    shard_index: int,
    output_root: str | Path,
    flush_every: int = 256,
    scratch_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run or boundary-resume one manifest shard with one physical writer."""

    manifest = validate_partition_manifest(partition_manifest)
    if type(shard_index) is not int or not 0 <= shard_index < manifest["shard_count"]:
        raise GlobalGCEHPCExactError("shard index is outside the manifest")
    root = Path(output_root).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    scratch = (
        None
        if scratch_root is None
        else Path(scratch_root).expanduser().absolute() / f"shard-{shard_index:03d}"
    )
    if scratch is not None:
        if scratch == root or root in scratch.parents or scratch in root.parents:
            raise GlobalGCEHPCExactError("scratch root must be separate from persistent output")
        scratch.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".single-writer.lock"
    lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GlobalGCEHPCExactError("shard already has an active writer") from exc
        if (root / "shard_manifest.json").is_file():
            return _validate_shard(root, manifest, shard_index)
        run_spec = {
            "schema_version": "globalgce_hpc_exact_shard_run_spec_v1",
            "manifest_sha256": manifest["manifest_sha256"],
            "shard_index": shard_index,
            "flush_every": flush_every,
            "scratch_policy": (
                "NODE_LOCAL_ACTIVE_PARTITION_THEN_ATOMIC_PERSISTENT_SEAL"
                if scratch is not None
                else "PERSISTENT_ACTIVE_PARTITION"
            ),
            "matrix_write_enabled": False,
        }
        _write_or_validate_run_spec(root / "run_spec.json", run_spec)
        units = [
            unit for unit in manifest["partitions"] if unit["shard_index"] == shard_index
        ]
        units.sort(key=lambda row: row["global_partition_order"])
        if not units:
            raise GlobalGCEHPCExactError("shard owns no partitions")
        completed: list[dict[str, Any]] = []
        partitions_root = root / "partitions"
        partitions_root.mkdir(exist_ok=True)
        for unit in units:
            atomic_write_json(
                root / "checkpoint.json",
                {
                    "schema_version": "globalgce_hpc_exact_shard_checkpoint_v1",
                    "state": "RUNNING",
                    "manifest_sha256": manifest["manifest_sha256"],
                    "scientific_input_sha256": manifest["scientific_input_sha256"],
                    "provenance_sha256": manifest["provenance"]["provenance_sha256"],
                    "graph_input_sha256": manifest["graph_input"]["sha256"],
                    "official_gspan_identity_sha256": manifest["official_gspan"][
                        "identity_sha256"
                    ],
                    "configuration_sha256": canonical_sha256(
                        manifest["configuration"]
                    ),
                    "shard_index": shard_index,
                    "completed_partition_ids": [
                        row["partition"]["partition_id"] for row in completed
                    ],
                    "current_partition_id": unit["partition_id"],
                    "current_partition_dfs_code_sha256": unit["dfs_code_sha256"],
                    "resume_boundary": "COMPLETED_PERSISTENT_PARTITION_ONLY",
                    "scratch_disposable": scratch is not None,
                    "written_at": _utc_now(),
                },
            )
            result = _execute_partition_unit(
                manifest,
                unit,
                partitions_root / unit["partition_id"],
                flush_every=flush_every,
                scratch_root=scratch,
            )
            completed.append(result)
            atomic_write_json(
                root / "checkpoint.json",
                {
                    "schema_version": "globalgce_hpc_exact_shard_checkpoint_v1",
                    "state": "RUNNING",
                    "manifest_sha256": manifest["manifest_sha256"],
                    "scientific_input_sha256": manifest["scientific_input_sha256"],
                    "provenance_sha256": manifest["provenance"]["provenance_sha256"],
                    "graph_input_sha256": manifest["graph_input"]["sha256"],
                    "official_gspan_identity_sha256": manifest["official_gspan"][
                        "identity_sha256"
                    ],
                    "configuration_sha256": canonical_sha256(
                        manifest["configuration"]
                    ),
                    "shard_index": shard_index,
                    "completed_partition_ids": [
                        row["partition"]["partition_id"] for row in completed
                    ],
                    "current_partition_id": None,
                    "resume_boundary": "COMPLETED_PERSISTENT_PARTITION_ONLY",
                    "scratch_disposable": scratch is not None,
                    "written_at": _utc_now(),
                },
            )
        payload = {
            "schema_version": SHARD_RESULT_SCHEMA,
            "status": "PASS",
            "completed_at": _utc_now(),
            "manifest_sha256": manifest["manifest_sha256"],
            "scientific_input_sha256": manifest["scientific_input_sha256"],
            "provenance_sha256": manifest["provenance"]["provenance_sha256"],
            "target_branches": manifest["provenance"]["target_branches"],
            "shard_index": shard_index,
            "partition_ids": [unit["partition_id"] for unit in units],
            "partition_result_sha256s": [row["result_sha256"] for row in completed],
            "event_count": sum(int(row["event_count"]) for row in completed),
            "pattern_count": sum(int(row["pattern_count"]) for row in completed),
            "rejection_count": sum(int(row["rejection_count"]) for row in completed),
            "resume_boundary": "COMPLETED_PERSISTENT_PARTITION_ONLY",
            "scratch_staging_used": scratch is not None,
            "scientific_search_pruned": False,
            "approximation_used": False,
            "matrix_write_enabled": False,
        }
        payload = _self_hashed(payload, field="result_sha256")
        atomic_write_json(root / "shard_manifest.json", payload)
        atomic_write_json(
            root / "checkpoint.json",
            {
                "schema_version": "globalgce_hpc_exact_shard_checkpoint_v1",
                "state": "COMPLETE",
                "manifest_sha256": manifest["manifest_sha256"],
                "scientific_input_sha256": manifest["scientific_input_sha256"],
                "provenance_sha256": manifest["provenance"]["provenance_sha256"],
                "graph_input_sha256": manifest["graph_input"]["sha256"],
                "official_gspan_identity_sha256": manifest["official_gspan"][
                    "identity_sha256"
                ],
                "configuration_sha256": canonical_sha256(
                    manifest["configuration"]
                ),
                "shard_index": shard_index,
                "completed_partition_ids": payload["partition_ids"],
                "current_partition_id": None,
                "resume_boundary": "COMPLETED_PERSISTENT_PARTITION_ONLY",
                "scratch_disposable": scratch is not None,
                "written_at": _utc_now(),
            },
        )
        return payload
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def _validate_shard(
    root: Path, manifest: Mapping[str, Any], shard_index: int
) -> dict[str, Any]:
    payload = _load_json(root / "shard_manifest.json")
    if (
        type(payload) is not dict
        or payload.get("schema_version") != SHARD_RESULT_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("manifest_sha256") != manifest["manifest_sha256"]
        or payload.get("scientific_input_sha256")
        != manifest["scientific_input_sha256"]
        or payload.get("provenance_sha256")
        != manifest["provenance"]["provenance_sha256"]
        or payload.get("target_branches")
        != manifest["provenance"]["target_branches"]
        or payload.get("shard_index") != shard_index
        or payload.get("matrix_write_enabled") is not False
        or not _is_sha256(payload.get("result_sha256"))
    ):
        raise GlobalGCEHPCExactError("shard manifest is invalid")
    claimed = payload.pop("result_sha256")
    observed = canonical_sha256(payload)
    payload["result_sha256"] = claimed
    if claimed != observed:
        raise GlobalGCEHPCExactError("shard result self-hash mismatch")
    expected = [
        unit
        for unit in manifest["partitions"]
        if unit["shard_index"] == shard_index
    ]
    expected.sort(key=lambda row: row["global_partition_order"])
    if payload.get("partition_ids") != [row["partition_id"] for row in expected]:
        raise GlobalGCEHPCExactError("shard partition coverage mismatch")
    results = [
        validate_unit_result(
            root / "partitions" / unit["partition_id"],
            manifest=manifest,
            expected_unit=unit,
        )
        for unit in expected
    ]
    if payload.get("partition_result_sha256s") != [
        row["result_sha256"] for row in results
    ]:
        raise GlobalGCEHPCExactError("shard partition hashes changed")
    return payload


def _merge_partition_results(
    *,
    manifest: Mapping[str, Any],
    unit_roots: Mapping[str, Path],
    output_root: Path,
    scope: str,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise GlobalGCEHPCExactError("merge output root must be fresh")
    staging_parent = output_root.parent if scratch_root is None else scratch_root
    staging_parent.mkdir(parents=True, exist_ok=True)
    temporary = staging_parent / (
        f".{output_root.name}.{os.getpid()}.{uuid.uuid4().hex}.incomplete"
    )
    temporary.mkdir(parents=True, mode=0o700)
    event_output = (temporary / "events.jsonl").open("x", encoding="utf-8")
    pattern_output = (temporary / "patterns.jsonl").open("x", encoding="utf-8")
    rejection_output = (temporary / "rejection_events.jsonl").open(
        "x", encoding="utf-8"
    )
    seen_database = sqlite3.connect(temporary / "seen_patterns.sqlite3")
    seen_database.execute("CREATE TABLE seen_patterns(pattern_sha256 TEXT PRIMARY KEY)")
    seen_database.execute("CREATE TABLE seen_events(dfs_code_sha256 TEXT PRIMARY KEY)")
    event_count = pattern_count = rejection_count = 0
    stable_top_k: list[dict[str, Any]] = []
    top_k = int(manifest["configuration"]["top_k"])
    ordered_units = [
        unit
        for unit in manifest["partitions"]
        if unit["partition_id"] in unit_roots
    ]
    ordered_units.sort(key=lambda row: row["global_partition_order"])
    try:
        for unit in ordered_units:
            root = unit_roots[unit["partition_id"]]
            validate_unit_result(root, manifest=manifest, expected_unit=unit)
            for row in _iter_jsonl(root / "events.jsonl"):
                try:
                    seen_database.execute(
                        "INSERT INTO seen_events(dfs_code_sha256) VALUES(?)",
                        (row["dfs_code_sha256"],),
                    )
                except sqlite3.IntegrityError as exc:
                    raise GlobalGCEHPCExactError(
                        "duplicate DFS candidate event across exact partitions"
                    ) from exc
                merged = {
                    **row,
                    "global_preorder": event_count,
                    "global_partition_order": unit["global_partition_order"],
                }
                event_output.write(_canonical_bytes(merged).decode("ascii") + "\n")
                event_count += 1
                if row["status"] != "ACCEPTED":
                    rejection_output.write(
                        _canonical_bytes(merged).decode("ascii") + "\n"
                    )
                    rejection_count += 1
            for row in _iter_jsonl(root / "patterns.jsonl"):
                try:
                    seen_database.execute(
                        "INSERT INTO seen_patterns(pattern_sha256) VALUES(?)",
                        (row["pattern_sha256"],),
                    )
                except sqlite3.IntegrityError as exc:
                    raise GlobalGCEHPCExactError(
                        "duplicate canonical pattern across exact partitions"
                    ) from exc
                merged = {
                    **row,
                    "global_preorder": pattern_count,
                    "global_partition_order": unit["global_partition_order"],
                }
                pattern_output.write(_canonical_bytes(merged).decode("ascii") + "\n")
                pattern_count += 1
                stable_top_k.append(_normalized_pattern(merged) | {
                    "global_preorder": merged["global_preorder"]
                })
                stable_top_k.sort(
                    key=lambda candidate: (
                        -int(candidate["support"]),
                        int(candidate["global_preorder"]),
                    )
                )
                if len(stable_top_k) > top_k:
                    stable_top_k.pop()
        seen_database.commit()
        for stream in (event_output, pattern_output, rejection_output):
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        seen_database.close()
        (temporary / "seen_patterns.sqlite3").unlink()
        stable_top_k_payload = {
            "schema_version": "globalgce_hpc_exact_stable_top_k_v1",
            "top_k": top_k,
            "selected_count": len(stable_top_k),
            "ordering": "SUPPORT_DESC_OFFICIAL_PREORDER_ASC",
            "selected": stable_top_k,
            "selected_sha256": canonical_sha256(stable_top_k),
        }
        atomic_write_json(temporary / "stable_top_k.json", stable_top_k_payload)
        payload = {
            "schema_version": MERGE_RESULT_SCHEMA,
            "status": "PASS",
            "scope": scope,
            "completed_at": _utc_now(),
            "manifest_sha256": manifest["manifest_sha256"],
            "scientific_input_sha256": manifest["scientific_input_sha256"],
            "provenance_sha256": manifest["provenance"]["provenance_sha256"],
            "target_branches": manifest["provenance"]["target_branches"],
            "ordered_partition_ids": [row["partition_id"] for row in ordered_units],
            "event_count": event_count,
            "pattern_count": pattern_count,
            "rejection_count": rejection_count,
            "events_sha256": sha256_file(temporary / "events.jsonl"),
            "patterns_sha256": sha256_file(temporary / "patterns.jsonl"),
            "rejection_events_sha256": sha256_file(
                temporary / "rejection_events.jsonl"
            ),
            "stable_top_k_sha256": sha256_file(temporary / "stable_top_k.json"),
            "stable_top_k_selected_sha256": stable_top_k_payload["selected_sha256"],
            "stable_top_k_selected_count": len(stable_top_k),
            "global_order": "OFFICIAL_ROOT_AND_DFS_PREORDER",
            "partition_disjoint": True,
            "partition_complete": True,
            "full_root_universe_complete": scope == "FULL_MANIFEST",
            "duplicate_pattern_count": 0,
            "duplicate_event_count": 0,
            "scientific_search_pruned": False,
            "approximation_used": False,
            "matrix_write_enabled": False,
            "scratch_staging_used": scratch_root is not None,
        }
        payload = _self_hashed(payload, field="result_sha256")
        atomic_write_json(temporary / "merge_manifest.json", payload)
        _fsync_directory(temporary)
        if scratch_root is None:
            os.rename(temporary, output_root)
            _fsync_directory(output_root.parent)
        else:
            persistent_temporary = output_root.parent / (
                f".{output_root.name}.{os.getpid()}.{uuid.uuid4().hex}.copying"
            )
            persistent_temporary.mkdir(parents=True, mode=0o700)
            for name in (
                "events.jsonl",
                "patterns.jsonl",
                "rejection_events.jsonl",
                "stable_top_k.json",
                "merge_manifest.json",
            ):
                target = persistent_temporary / name
                shutil.copyfile(temporary / name, target)
                descriptor = os.open(target, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            _fsync_directory(persistent_temporary)
            validate_merge_result(
                persistent_temporary, manifest=manifest, allowed_scopes=(scope,)
            )
            os.rename(persistent_temporary, output_root)
            _fsync_directory(output_root.parent)
            shutil.rmtree(temporary)
        return payload
    except BaseException:
        for stream in (event_output, pattern_output, rejection_output):
            if not stream.closed:
                stream.close()
        try:
            seen_database.close()
        except Exception:
            pass
        raise


def merge_exact_shards(
    *,
    partition_manifest: str | Path,
    shards_root: str | Path,
    output_root: str | Path,
    scratch_root: str | Path | None = None,
) -> dict[str, Any]:
    manifest = validate_partition_manifest(partition_manifest)
    root = Path(shards_root).expanduser().resolve(strict=True)
    for shard_index in range(manifest["shard_count"]):
        _validate_shard(root / f"shard-{shard_index:03d}", manifest, shard_index)
    unit_roots = {
        unit["partition_id"]: root
        / f"shard-{unit['shard_index']:03d}"
        / "partitions"
        / unit["partition_id"]
        for unit in manifest["partitions"]
    }
    if set(unit_roots) != {unit["partition_id"] for unit in manifest["partitions"]}:
        raise GlobalGCEHPCExactError("full merge partition coverage is incomplete")
    return _merge_partition_results(
        manifest=manifest,
        unit_roots=unit_roots,
        output_root=Path(output_root).expanduser().absolute(),
        scope=(
            "FULL_MANIFEST"
            if manifest.get("scope") == "FULL_ROOT_UNIVERSE"
            else (
                "SELECTED_PARTITION_CANARY"
                if manifest.get("scope") == "SELECTED_PARTITION_CANARY"
                else "SELECTED_ROOTS_CANARY"
            )
        ),
        scratch_root=(
            None
            if scratch_root is None
            else Path(scratch_root).expanduser().absolute() / "merge"
        ),
    )


def validate_merge_result(
    root: str | Path,
    *,
    manifest: Mapping[str, Any],
    allowed_scopes: Sequence[str],
) -> dict[str, Any]:
    directory = Path(root).expanduser().resolve(strict=True)
    payload = _load_json(directory / "merge_manifest.json")
    if (
        type(payload) is not dict
        or payload.get("schema_version") != MERGE_RESULT_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("scope") not in set(allowed_scopes)
        or payload.get("manifest_sha256") != manifest["manifest_sha256"]
        or payload.get("scientific_input_sha256")
        != manifest["scientific_input_sha256"]
        or payload.get("provenance_sha256")
        != manifest["provenance"]["provenance_sha256"]
        or payload.get("target_branches")
        != manifest["provenance"]["target_branches"]
        or payload.get("matrix_write_enabled") is not False
        or payload.get("scientific_search_pruned") is not False
        or payload.get("approximation_used") is not False
        or payload.get("duplicate_pattern_count") != 0
        or payload.get("duplicate_event_count") != 0
        or payload.get("partition_disjoint") is not True
        or payload.get("partition_complete") is not True
        or not _is_sha256(payload.get("result_sha256"))
    ):
        raise GlobalGCEHPCExactError("merge manifest is invalid")
    claimed = payload.pop("result_sha256")
    observed = canonical_sha256(payload)
    payload["result_sha256"] = claimed
    if claimed != observed:
        raise GlobalGCEHPCExactError("merge manifest self-hash mismatch")
    expected_partition_ids = [
        row["partition_id"]
        for row in sorted(
            manifest["partitions"], key=lambda row: row["global_partition_order"]
        )
    ]
    if (
        payload.get("scope") != "REFERENCE_ROOTS"
        and payload.get("ordered_partition_ids") != expected_partition_ids
    ):
        raise GlobalGCEHPCExactError("merged partition coverage/order is incomplete")
    events_path = directory / "events.jsonl"
    patterns_path = directory / "patterns.jsonl"
    rejections_path = directory / "rejection_events.jsonl"
    top_k_path = directory / "stable_top_k.json"
    if (
        sha256_file(events_path) != payload["events_sha256"]
        or sha256_file(patterns_path) != payload["patterns_sha256"]
        or sha256_file(rejections_path) != payload["rejection_events_sha256"]
        or sha256_file(top_k_path) != payload["stable_top_k_sha256"]
        or _count_jsonl(events_path) != payload["event_count"]
        or _count_jsonl(patterns_path) != payload["pattern_count"]
        or _count_jsonl(rejections_path) != payload["rejection_count"]
    ):
        raise GlobalGCEHPCExactError("merge payload hash/count mismatch")
    for expected, row in enumerate(_iter_jsonl(events_path)):
        if row.get("global_preorder") != expected:
            raise GlobalGCEHPCExactError("merged event preorder is not consecutive")
        code = dfs_code_from_json(row.get("dfs_code"))
        if row.get("dfs_code_sha256") != dfs_code_sha256(code):
            raise GlobalGCEHPCExactError("merged event DFS identity changed")
        if row.get("status") not in EVENT_STATUSES:
            raise GlobalGCEHPCExactError("merged event status is invalid")
        if row.get("target_branches") != manifest["provenance"]["target_branches"]:
            raise GlobalGCEHPCExactError("merged event target branches changed")
    for expected, row in enumerate(_iter_jsonl(patterns_path)):
        if row.get("global_preorder") != expected:
            raise GlobalGCEHPCExactError("merged pattern preorder is not consecutive")
        code = dfs_code_from_json(row.get("dfs_code"))
        if (
            row.get("dfs_code_sha256") != dfs_code_sha256(code)
            or row.get("pattern_sha256")
            != canonical_sha256(
                {"dfs_code": dfs_code_to_json(code), "undirected": True}
            )
            or row.get("target_branches")
            != manifest["provenance"]["target_branches"]
        ):
            raise GlobalGCEHPCExactError("merged pattern identity changed")
    if any(row.get("status") == "ACCEPTED" for row in _iter_jsonl(rejections_path)):
        raise GlobalGCEHPCExactError("rejection stream contains an accepted event")
    stable_top_k = _load_json(top_k_path)
    if (
        type(stable_top_k) is not dict
        or stable_top_k.get("schema_version")
        != "globalgce_hpc_exact_stable_top_k_v1"
        or stable_top_k.get("top_k") != manifest["configuration"]["top_k"]
        or stable_top_k.get("selected_count")
        != payload["stable_top_k_selected_count"]
        or stable_top_k.get("selected_sha256")
        != payload["stable_top_k_selected_sha256"]
        or canonical_sha256(stable_top_k.get("selected"))
        != stable_top_k.get("selected_sha256")
    ):
        raise GlobalGCEHPCExactError("stable support top-k identity changed")
    return payload


class _IndependentReferenceWriter:
    """Reference-only writer, intentionally separate from the shard writer."""

    def __init__(
        self, root: Path, *, flush_every: int, target_branches: Sequence[int]
    ) -> None:
        self.flush_every = flush_every
        self.target_branches = list(target_branches)
        self.events = (root / "events.jsonl").open("x", encoding="utf-8")
        self.patterns = (root / "patterns.jsonl").open("x", encoding="utf-8")
        self.event_count = 0
        self.pattern_count = 0
        self.rejection_count = 0

    def event(
        self,
        *,
        unit: Mapping[str, Any],
        code: Sequence[TypedDFSEdge],
        projected: Sequence[Any],
        support: int,
        status: str,
    ) -> None:
        row = {
            "local_preorder": self.event_count,
            "partition_id": unit["partition_id"],
            "root_index": unit["root_index"],
            "dfs_code": dfs_code_to_json(code),
            "dfs_code_sha256": dfs_code_sha256(code),
            "target_branches": self.target_branches,
            "support": support,
            "status": status,
            "candidate_input_sha256": _candidate_input_sha256(projected),
        }
        self.events.write(_canonical_bytes(row).decode("ascii") + "\n")
        self.event_count += 1
        if status != "ACCEPTED":
            self.rejection_count += 1
        self._flush_if_needed()

    def pattern(
        self,
        *,
        unit: Mapping[str, Any],
        code: Sequence[TypedDFSEdge],
        support: int,
    ) -> None:
        row = {
            "local_preorder": self.pattern_count,
            "partition_id": unit["partition_id"],
            "root_index": unit["root_index"],
            "dfs_code": dfs_code_to_json(code),
            "dfs_code_sha256": dfs_code_sha256(code),
            "pattern_sha256": canonical_sha256(
                {"dfs_code": dfs_code_to_json(code), "undirected": True}
            ),
            "support": support,
            "target_branches": self.target_branches,
        }
        self.patterns.write(_canonical_bytes(row).decode("ascii") + "\n")
        self.pattern_count += 1
        self._flush_if_needed()

    def _flush_if_needed(self) -> None:
        if (self.event_count + self.pattern_count) % self.flush_every == 0:
            for stream in (self.events, self.patterns):
                stream.flush()
                os.fsync(stream.fileno())

    def close(self) -> None:
        for stream in (self.events, self.patterns):
            if not stream.closed:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()


def _independent_reference_observation(
    miner: Any, projected: Sequence[Any]
) -> tuple[int, str]:
    support = int(miner._get_support(projected))
    if support < int(miner._min_support):
        return support, "REJECTED_MIN_SUPPORT"
    if not bool(miner._is_min()):
        return support, "REJECTED_NON_MINIMAL"
    return support, "ACCEPTED"


def _run_independent_reference_unit(
    manifest: Mapping[str, Any],
    unit: Mapping[str, Any],
    destination: Path,
    *,
    flush_every: int,
    scratch_root: Path | None,
) -> dict[str, Any]:
    """Run one serial reference unit without the production partition executor."""

    if destination.is_dir():
        return validate_unit_result(destination, manifest=manifest, expected_unit=unit)
    if destination.exists():
        raise GlobalGCEHPCExactError("reference unit destination is invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = destination.parent if scratch_root is None else scratch_root
    staging_parent.mkdir(parents=True, exist_ok=True)
    temporary = staging_parent / (
        f".reference-{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.incomplete"
    )
    temporary.mkdir(mode=0o700)
    writer = _IndependentReferenceWriter(
        temporary,
        flush_every=flush_every,
        target_branches=manifest["provenance"]["target_branches"],
    )
    found_target = False
    try:
        graphs, graph_identity = load_graph_jsonl(manifest["graph_input"]["path"])
        if graph_identity != manifest["graph_input"]:
            raise GlobalGCEHPCExactError("reference graph input identity changed")
        with _narrow_official_gspan_import(
            manifest["official_gspan"]["source_root"]
        ) as (module, official_identity):
            if official_identity != manifest["official_gspan"]:
                raise GlobalGCEHPCExactError("reference official source changed")
            config = manifest["configuration"]
            miner, roots = _miner_and_roots(
                module,
                graphs,
                min_support=int(config["min_support"]),
                min_vertices=int(config["min_vertices"]),
                max_vertices=int(config["max_vertices"]),
            )
            _verify_live_root_universe(manifest, module, roots)
            root_labels, root_projected = roots[int(unit["root_index"])]
            target = dfs_code_from_json(unit["dfs_code"])
            unit_type = unit["partition_type"]
            if unit_type not in {"ROOT_SUBTREE", "PREFIX_SUBTREE"}:
                raise GlobalGCEHPCExactError(
                    "independent reference accepts only complete declared subtrees"
                )
            gspan_class = module.gSpan
            original_mining = gspan_class._subgraph_mining
            original_report = gspan_class._report
            recording = False

            def reference_report(self: Any, _projected: Any) -> None:
                if not recording:
                    return
                if self._DFScode.get_num_vertices() < self._min_num_vertices:
                    return
                writer.pattern(
                    unit=unit,
                    code=typed_dfs_code(self._DFScode),
                    support=int(self._support),
                )

            def reference_visit(self: Any, projected: Sequence[Any]) -> Any:
                nonlocal recording, found_target
                code = typed_dfs_code(self._DFScode)
                if recording:
                    support, status = _independent_reference_observation(self, projected)
                    writer.event(
                        unit=unit,
                        code=code,
                        projected=projected,
                        support=support,
                        status=status,
                    )
                    if status != "ACCEPTED":
                        return self
                    return original_mining(self, projected)

                is_target = code == target
                if unit_type == "ROOT_SUBTREE" and len(code) == 1 and is_target:
                    found_target = True
                elif unit_type == "PREFIX_SUBTREE" and is_target:
                    found_target = True
                elif unit_type == "PREFIX_SUBTREE" and _is_prefix(code, target):
                    support, status = _independent_reference_observation(self, projected)
                    if status != "ACCEPTED":
                        raise GlobalGCEHPCExactError(
                            "reference prefix lies below a rejected official ancestor"
                        )
                    return original_mining(self, projected)
                else:
                    return self

                support, status = _independent_reference_observation(self, projected)
                expected = unit.get("expected_candidate")
                observed = {
                    "support": support,
                    "status": status,
                    "candidate_input_sha256": _candidate_input_sha256(projected),
                }
                if expected is not None and observed != expected:
                    raise GlobalGCEHPCExactError(
                        "independent reference prefix observation differs from manifest"
                    )
                writer.event(
                    unit=unit,
                    code=code,
                    projected=projected,
                    support=support,
                    status=status,
                )
                if status != "ACCEPTED":
                    if unit_type == "PREFIX_SUBTREE":
                        raise GlobalGCEHPCExactError(
                            "selected reference prefix is no longer accepted"
                        )
                    return self
                recording = True
                try:
                    return original_mining(self, projected)
                finally:
                    recording = False

            gspan_class._subgraph_mining = reference_visit
            gspan_class._report = reference_report
            try:
                miner._DFScode.append(module.DFSedge(0, 1, root_labels))
                miner._subgraph_mining(root_projected)
                miner._DFScode.pop()
            finally:
                gspan_class._subgraph_mining = original_mining
                gspan_class._report = original_report
        if not found_target:
            raise GlobalGCEHPCExactError("independent reference target was not reached")
        writer.close()
        writer = None  # type: ignore[assignment]
        result = {
            "schema_version": UNIT_RESULT_SCHEMA,
            "status": "PASS",
            "completed_at": _utc_now(),
            "manifest_sha256": manifest["manifest_sha256"],
            "partition": dict(unit),
            "event_count": _count_jsonl(temporary / "events.jsonl"),
            "pattern_count": _count_jsonl(temporary / "patterns.jsonl"),
            "rejection_count": _count_rejections(temporary / "events.jsonl"),
            "events_sha256": sha256_file(temporary / "events.jsonl"),
            "patterns_sha256": sha256_file(temporary / "patterns.jsonl"),
            "scientific_search_pruned": False,
            "approximation_used": False,
            "single_writer": True,
            "matrix_write_enabled": False,
            "scientific_input_sha256": manifest["scientific_input_sha256"],
            "provenance_sha256": manifest["provenance"]["provenance_sha256"],
            "target_branches": manifest["provenance"]["target_branches"],
            "scratch_staging_used": scratch_root is not None,
            "execution_engine": "INDEPENDENT_SERIAL_OFFICIAL_TRAVERSAL",
        }
        result = _self_hashed(result, field="result_sha256")
        atomic_write_json(temporary / "partition_manifest.json", result)
        _fsync_directory(temporary)
        validate_unit_result(temporary, manifest=manifest, expected_unit=unit)
        if scratch_root is None:
            os.rename(temporary, destination)
            _fsync_directory(destination.parent)
        else:
            _publish_staged_unit(
                temporary, destination, manifest=manifest, unit=unit
            )
            shutil.rmtree(temporary)
        return result
    except BaseException:
        if writer is not None:
            writer.close()
        raise


def _merge_independent_reference_units(
    *,
    manifest: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
    partitions_root: Path,
    output_root: Path,
    scratch_root: Path | None,
) -> dict[str, Any]:
    """Independently concatenate serial reference units in declared preorder."""

    if output_root.is_dir():
        return validate_merge_result(
            output_root, manifest=manifest, allowed_scopes=("REFERENCE_ROOTS",)
        )
    staging_parent = output_root.parent if scratch_root is None else scratch_root
    staging_parent.mkdir(parents=True, exist_ok=True)
    temporary = staging_parent / (
        f".reference-merge.{os.getpid()}.{uuid.uuid4().hex}.incomplete"
    )
    temporary.mkdir(mode=0o700)
    event_output = (temporary / "events.jsonl").open("x", encoding="utf-8")
    pattern_output = (temporary / "patterns.jsonl").open("x", encoding="utf-8")
    rejection_output = (temporary / "rejection_events.jsonl").open(
        "x", encoding="utf-8"
    )
    seen_events: set[str] = set()
    seen_patterns: set[str] = set()
    event_count = pattern_count = rejection_count = 0
    top_k = int(manifest["configuration"]["top_k"])
    stable_top_k: list[dict[str, Any]] = []
    try:
        ordered = sorted(units, key=lambda row: row["global_partition_order"])
        for unit in ordered:
            root = partitions_root / unit["partition_id"]
            validate_unit_result(root, manifest=manifest, expected_unit=unit)
            for row in _iter_jsonl(root / "events.jsonl"):
                identity = row["dfs_code_sha256"]
                if identity in seen_events:
                    raise GlobalGCEHPCExactError(
                        "duplicate DFS event in independent reference"
                    )
                seen_events.add(identity)
                merged = {
                    **row,
                    "global_preorder": event_count,
                    "global_partition_order": unit["global_partition_order"],
                }
                event_output.write(_canonical_bytes(merged).decode("ascii") + "\n")
                event_count += 1
                if row["status"] != "ACCEPTED":
                    rejection_output.write(
                        _canonical_bytes(merged).decode("ascii") + "\n"
                    )
                    rejection_count += 1
            for row in _iter_jsonl(root / "patterns.jsonl"):
                identity = row["pattern_sha256"]
                if identity in seen_patterns:
                    raise GlobalGCEHPCExactError(
                        "duplicate pattern in independent reference"
                    )
                seen_patterns.add(identity)
                merged = {
                    **row,
                    "global_preorder": pattern_count,
                    "global_partition_order": unit["global_partition_order"],
                }
                pattern_output.write(_canonical_bytes(merged).decode("ascii") + "\n")
                stable_top_k.append(
                    _normalized_pattern(merged)
                    | {"global_preorder": merged["global_preorder"]}
                )
                stable_top_k.sort(
                    key=lambda candidate: (
                        -int(candidate["support"]),
                        int(candidate["global_preorder"]),
                    )
                )
                if len(stable_top_k) > top_k:
                    stable_top_k.pop()
                pattern_count += 1
        for stream in (event_output, pattern_output, rejection_output):
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        top_k_payload = {
            "schema_version": "globalgce_hpc_exact_stable_top_k_v1",
            "top_k": top_k,
            "selected_count": len(stable_top_k),
            "ordering": "SUPPORT_DESC_OFFICIAL_PREORDER_ASC",
            "selected": stable_top_k,
            "selected_sha256": canonical_sha256(stable_top_k),
        }
        atomic_write_json(temporary / "stable_top_k.json", top_k_payload)
        payload = {
            "schema_version": MERGE_RESULT_SCHEMA,
            "status": "PASS",
            "scope": "REFERENCE_ROOTS",
            "completed_at": _utc_now(),
            "manifest_sha256": manifest["manifest_sha256"],
            "scientific_input_sha256": manifest["scientific_input_sha256"],
            "provenance_sha256": manifest["provenance"]["provenance_sha256"],
            "target_branches": manifest["provenance"]["target_branches"],
            "ordered_partition_ids": [unit["partition_id"] for unit in ordered],
            "event_count": event_count,
            "pattern_count": pattern_count,
            "rejection_count": rejection_count,
            "events_sha256": sha256_file(temporary / "events.jsonl"),
            "patterns_sha256": sha256_file(temporary / "patterns.jsonl"),
            "rejection_events_sha256": sha256_file(
                temporary / "rejection_events.jsonl"
            ),
            "stable_top_k_sha256": sha256_file(temporary / "stable_top_k.json"),
            "stable_top_k_selected_sha256": top_k_payload["selected_sha256"],
            "stable_top_k_selected_count": len(stable_top_k),
            "global_order": "OFFICIAL_ROOT_AND_DFS_PREORDER",
            "partition_disjoint": True,
            "partition_complete": True,
            "full_root_universe_complete": False,
            "duplicate_pattern_count": 0,
            "duplicate_event_count": 0,
            "scientific_search_pruned": False,
            "approximation_used": False,
            "matrix_write_enabled": False,
            "scratch_staging_used": scratch_root is not None,
            "execution_engine": "INDEPENDENT_SERIAL_REFERENCE_MERGE",
        }
        payload = _self_hashed(payload, field="result_sha256")
        atomic_write_json(temporary / "merge_manifest.json", payload)
        _fsync_directory(temporary)
        validate_merge_result(
            temporary, manifest=manifest, allowed_scopes=("REFERENCE_ROOTS",)
        )
        if scratch_root is None:
            os.rename(temporary, output_root)
            _fsync_directory(output_root.parent)
        else:
            persistent_temporary = output_root.parent / (
                f".{output_root.name}.{os.getpid()}.{uuid.uuid4().hex}.copying"
            )
            persistent_temporary.mkdir(parents=True, mode=0o700)
            for name in (
                "events.jsonl",
                "patterns.jsonl",
                "rejection_events.jsonl",
                "stable_top_k.json",
                "merge_manifest.json",
            ):
                target = persistent_temporary / name
                shutil.copyfile(temporary / name, target)
                descriptor = os.open(target, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            _fsync_directory(persistent_temporary)
            validate_merge_result(
                persistent_temporary,
                manifest=manifest,
                allowed_scopes=("REFERENCE_ROOTS",),
            )
            os.rename(persistent_temporary, output_root)
            _fsync_directory(output_root.parent)
            shutil.rmtree(temporary)
        return payload
    except BaseException:
        for stream in (event_output, pattern_output, rejection_output):
            if not stream.closed:
                stream.close()
        raise


def run_exact_reference(
    *,
    partition_manifest: str | Path,
    output_root: str | Path,
    root_indices: Sequence[int] | None = None,
    flush_every: int = 256,
    scratch_root: str | Path | None = None,
) -> dict[str, Any]:
    manifest = validate_partition_manifest(partition_manifest)
    selected = tuple(
        sorted(
            set(
                manifest["canary_root_indices"]
                if root_indices is None
                else root_indices
            )
        )
    )
    if (
        not selected
        or any(type(index) is not int or index < 0 for index in selected)
        or selected[-1] >= len(manifest["root_universe"])
    ):
        raise GlobalGCEHPCExactError("reference root selection is invalid")
    output = Path(output_root).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=True)
    scratch = (
        None
        if scratch_root is None
        else Path(scratch_root).expanduser().absolute() / "reference"
    )
    if scratch is not None:
        if scratch == output or output in scratch.parents or scratch in output.parents:
            raise GlobalGCEHPCExactError("reference scratch must be separate from output")
        scratch.mkdir(parents=True, exist_ok=True)
    reference_units: list[dict[str, Any]] = []
    if manifest["scope"] == "SELECTED_PARTITION_CANARY":
        if root_indices is not None and tuple(selected) != tuple(
            manifest["canary_root_indices"]
        ):
            raise GlobalGCEHPCExactError(
                "selected-prefix reference cannot widen or narrow manifest scope"
            )
        reference_units = [
            {**dict(unit), "shard_index": 0} for unit in manifest["partitions"]
        ]
    else:
        for order, root_index in enumerate(selected):
            descriptor = manifest["root_universe"][root_index]
            code = dfs_code_from_json(descriptor["dfs_code"])
            unit = {
                **_unit_payload(
                    root_index=root_index,
                    unit_type="ROOT_SUBTREE",
                    code=code,
                    observation=None,
                    support_hint=max(int(descriptor["projected_support"]), 1),
                ),
                "global_partition_order": order,
                "shard_index": 0,
            }
            # Validation binds unit equality, so expose a temporary manifest view
            # containing the exact reference root units.
            reference_units.append(unit)
    reference_manifest = {
        **manifest,
        "partitions": reference_units,
        "root_universe_sha256": manifest["root_universe_sha256"],
    }
    lock_descriptor = os.open(
        output / ".single-writer.lock", os.O_RDWR | os.O_CREAT, 0o600
    )
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GlobalGCEHPCExactError(
                "independent reference already has an active writer"
            ) from exc
        terminal = output / "reference_manifest.json"
        if terminal.is_file():
            payload = _load_json(terminal)
            if (
                type(payload) is not dict
                or payload.get("schema_version")
                != "globalgce_hpc_exact_reference_v1"
                or payload.get("status") != "PASS"
                or payload.get("manifest_sha256") != manifest["manifest_sha256"]
                or payload.get("root_indices") != list(selected)
                or payload.get("reference_unit_ids")
                != [unit["partition_id"] for unit in reference_units]
                or not _is_sha256(payload.get("result_sha256"))
            ):
                raise GlobalGCEHPCExactError("existing reference terminal is invalid")
            claimed = payload.pop("result_sha256")
            observed = canonical_sha256(payload)
            payload["result_sha256"] = claimed
            if claimed != observed:
                raise GlobalGCEHPCExactError("reference terminal self-hash mismatch")
            merged = validate_merge_result(
                output / "merged",
                manifest=reference_manifest,
                allowed_scopes=("REFERENCE_ROOTS",),
            )
            if payload.get("merge_result_sha256") != merged["result_sha256"]:
                raise GlobalGCEHPCExactError("reference merge binding changed")
            return payload

        run_spec = {
            "schema_version": "globalgce_hpc_exact_reference_run_spec_v1",
            "manifest_sha256": manifest["manifest_sha256"],
            "scientific_input_sha256": manifest["scientific_input_sha256"],
            "provenance_sha256": manifest["provenance"]["provenance_sha256"],
            "root_indices": list(selected),
            "reference_units": reference_units,
            "flush_every": flush_every,
            "scratch_policy": (
                "NODE_LOCAL_ACTIVE_UNIT_THEN_ATOMIC_PERSISTENT_SEAL"
                if scratch is not None
                else "PERSISTENT_ACTIVE_UNIT"
            ),
            "execution_engine": "INDEPENDENT_SERIAL_OFFICIAL_TRAVERSAL",
            "matrix_write_enabled": False,
        }
        _write_or_validate_run_spec(output / "run_spec.json", run_spec)
        partitions_root = output / "partitions"
        partitions_root.mkdir(exist_ok=True)
        completed: list[str] = []
        for unit in reference_units:
            atomic_write_json(
                output / "checkpoint.json",
                {
                    "schema_version": "globalgce_hpc_exact_reference_checkpoint_v1",
                    "state": "RUNNING",
                    "manifest_sha256": manifest["manifest_sha256"],
                    "scientific_input_sha256": manifest["scientific_input_sha256"],
                    "provenance_sha256": manifest["provenance"]["provenance_sha256"],
                    "completed_unit_ids": completed,
                    "current_unit_id": unit["partition_id"],
                    "resume_boundary": "COMPLETED_PERSISTENT_REFERENCE_UNIT_ONLY",
                    "written_at": _utc_now(),
                },
            )
            _run_independent_reference_unit(
                reference_manifest,
                unit,
                partitions_root / unit["partition_id"],
                flush_every=flush_every,
                scratch_root=scratch,
            )
            completed.append(unit["partition_id"])
        merged = _merge_independent_reference_units(
            manifest=reference_manifest,
            units=reference_units,
            partitions_root=partitions_root,
            output_root=output / "merged",
            scratch_root=(None if scratch is None else scratch / "merge"),
        )
        payload = {
            "schema_version": "globalgce_hpc_exact_reference_v1",
            "status": "PASS",
            "manifest_sha256": manifest["manifest_sha256"],
            "scientific_input_sha256": manifest["scientific_input_sha256"],
            "provenance_sha256": manifest["provenance"]["provenance_sha256"],
            "target_branches": manifest["provenance"]["target_branches"],
            "root_indices": list(selected),
            "selected_partition_ids": manifest["selected_partition_ids"],
            "reference_unit_ids": [unit["partition_id"] for unit in reference_units],
            "scratch_staging_used": scratch is not None,
            "merge_result_sha256": merged["result_sha256"],
            "execution_engine": "INDEPENDENT_SERIAL_OFFICIAL_TRAVERSAL",
            "scientific_search_pruned": False,
            "approximation_used": False,
            "matrix_write_enabled": False,
        }
        payload = _self_hashed(payload, field="result_sha256")
        atomic_write_json(terminal, payload)
        atomic_write_json(
            output / "checkpoint.json",
            {
                "schema_version": "globalgce_hpc_exact_reference_checkpoint_v1",
                "state": "COMPLETE",
                "manifest_sha256": manifest["manifest_sha256"],
                "scientific_input_sha256": manifest["scientific_input_sha256"],
                "provenance_sha256": manifest["provenance"]["provenance_sha256"],
                "completed_unit_ids": completed,
                "current_unit_id": None,
                "resume_boundary": "COMPLETED_PERSISTENT_REFERENCE_UNIT_ONLY",
                "written_at": _utc_now(),
            },
        )
        return payload
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def _normalized_event(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "root_index",
            "dfs_code",
            "dfs_code_sha256",
            "support",
            "status",
            "candidate_input_sha256",
            "target_branches",
        )
    }


def _normalized_pattern(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "root_index",
            "dfs_code",
            "dfs_code_sha256",
            "pattern_sha256",
            "support",
            "target_branches",
        )
    }


def verify_exact_parity(
    *,
    partition_manifest: str | Path,
    reference_root: str | Path,
    merged_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Compare all scientific event surfaces on the selected canary roots."""

    manifest = validate_partition_manifest(partition_manifest)
    destination = Path(output).expanduser().absolute()
    if destination.exists():
        raise GlobalGCEHPCExactError("parity receipt output must be fresh")
    reference = Path(reference_root).expanduser().resolve(strict=True) / "merged"
    optimized = Path(merged_root).expanduser().resolve(strict=True)
    reference_manifest = validate_merge_result(
        reference,
        manifest=manifest,
        allowed_scopes=("REFERENCE_ROOTS",),
    )
    optimized_manifest = validate_merge_result(
        optimized,
        manifest=manifest,
        allowed_scopes=(
            "SELECTED_ROOTS_CANARY",
            "SELECTED_PARTITION_CANARY",
            "FULL_MANIFEST",
        ),
    )
    canary_roots = set(manifest["canary_root_indices"])
    reference_events = [
        _normalized_event(row) for row in _iter_jsonl(reference / "events.jsonl")
    ]
    optimized_events = [
        _normalized_event(row)
        for row in _iter_jsonl(optimized / "events.jsonl")
        if row["root_index"] in canary_roots
    ]
    reference_patterns = [
        _normalized_pattern(row)
        for row in _iter_jsonl(reference / "patterns.jsonl")
    ]
    optimized_patterns = [
        _normalized_pattern(row)
        for row in _iter_jsonl(optimized / "patterns.jsonl")
        if row["root_index"] in canary_roots
    ]
    first_event_divergence = next(
        (
            index
            for index, (left, right) in enumerate(
                zip(reference_events, optimized_events, strict=False)
            )
            if left != right
        ),
        None,
    )
    if first_event_divergence is None and len(reference_events) != len(optimized_events):
        first_event_divergence = min(len(reference_events), len(optimized_events))
    first_pattern_divergence = next(
        (
            index
            for index, (left, right) in enumerate(
                zip(reference_patterns, optimized_patterns, strict=False)
            )
            if left != right
        ),
        None,
    )
    if first_pattern_divergence is None and len(reference_patterns) != len(
        optimized_patterns
    ):
        first_pattern_divergence = min(
            len(reference_patterns), len(optimized_patterns)
        )
    event_equal = reference_events == optimized_events
    pattern_equal = reference_patterns == optimized_patterns
    support_equal = [row["support"] for row in reference_patterns] == [
        row["support"] for row in optimized_patterns
    ]
    preorder_equal = [row["dfs_code_sha256"] for row in reference_patterns] == [
        row["dfs_code_sha256"] for row in optimized_patterns
    ]
    candidate_input_equal = [
        row["candidate_input_sha256"] for row in reference_events
    ] == [row["candidate_input_sha256"] for row in optimized_events]
    rejection_equal = [
        row for row in reference_events if row["status"] != "ACCEPTED"
    ] == [row for row in optimized_events if row["status"] != "ACCEPTED"]
    passed = all(
        (
            event_equal,
            pattern_equal,
            support_equal,
            preorder_equal,
            candidate_input_equal,
            rejection_equal,
        )
    )
    payload = {
        "schema_version": PARITY_RESULT_SCHEMA,
        "status": "PASS" if passed else "FAILED",
        "verified_at": _utc_now(),
        "manifest_sha256": manifest["manifest_sha256"],
        "scientific_input_sha256": manifest["scientific_input_sha256"],
        "provenance_sha256": manifest["provenance"]["provenance_sha256"],
        "target_branches": manifest["provenance"]["target_branches"],
        "search_space_scope": manifest["scope"],
        "selected_partition_ids": manifest["selected_partition_ids"],
        "canary_partition_manifest": {
            "path": str(Path(partition_manifest).expanduser().resolve(strict=True)),
            "file_sha256": sha256_file(partition_manifest),
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "reference_merge": {
            "result_sha256": reference_manifest["result_sha256"],
            "events_sha256": reference_manifest["events_sha256"],
            "patterns_sha256": reference_manifest["patterns_sha256"],
            "rejection_events_sha256": reference_manifest[
                "rejection_events_sha256"
            ],
        },
        "optimized_merge": {
            "result_sha256": optimized_manifest["result_sha256"],
            "events_sha256": optimized_manifest["events_sha256"],
            "patterns_sha256": optimized_manifest["patterns_sha256"],
            "rejection_events_sha256": optimized_manifest[
                "rejection_events_sha256"
            ],
        },
        "canary_root_indices": sorted(canary_roots),
        "patterns_equal": pattern_equal,
        "supports_equal": support_equal,
        "stable_preorder_equal": preorder_equal,
        "candidate_inputs_equal": candidate_input_equal,
        "rejection_events_equal": rejection_equal,
        "all_events_equal": event_equal,
        "reference_event_count": len(reference_events),
        "optimized_event_count": len(optimized_events),
        "reference_pattern_count": len(reference_patterns),
        "optimized_pattern_count": len(optimized_patterns),
        "first_event_divergence": first_event_divergence,
        "first_pattern_divergence": first_pattern_divergence,
        "scientific_search_pruned": False,
        "approximation_used": False,
        "matrix_write_enabled": False,
    }
    payload = _self_hashed(payload, field="result_sha256")
    atomic_write_json(destination, payload)
    return payload


def build_result_bundle(
    *,
    partition_manifest: str | Path,
    merge_root: str | Path,
    parity_receipt: str | Path,
    output_tar: str | Path,
    output_manifest: str | Path,
    environment_manifest: str | Path | None = None,
    slurm_inventory: str | Path | None = None,
    resource_metrics: str | Path | None = None,
) -> dict[str, Any]:
    """Create a deterministic, matrix-inert tar archive of verified results."""

    import tarfile

    manifest_path = Path(partition_manifest).expanduser().resolve(strict=True)
    manifest = validate_partition_manifest(manifest_path)
    merge = Path(merge_root).expanduser().resolve(strict=True)
    merge_manifest = validate_merge_result(
        merge,
        manifest=manifest,
        allowed_scopes=("FULL_MANIFEST",),
    )
    parity_path = Path(parity_receipt).expanduser().resolve(strict=True)
    parity = _load_json(parity_path)
    if type(parity) is not dict or not _is_sha256(parity.get("result_sha256")):
        raise GlobalGCEHPCExactError("parity receipt is malformed")
    parity_copy = dict(parity)
    parity_sha = parity_copy.pop("result_sha256")
    if canonical_sha256(parity_copy) != parity_sha:
        raise GlobalGCEHPCExactError("parity receipt self-hash mismatch")
    canary_manifest_identity = parity.get("canary_partition_manifest")
    if (
        type(canary_manifest_identity) is not dict
        or not _is_sha256(canary_manifest_identity.get("file_sha256"))
        or not _is_sha256(canary_manifest_identity.get("manifest_sha256"))
    ):
        raise GlobalGCEHPCExactError("parity receipt lacks its canary manifest")
    canary_manifest_path = Path(
        str(canary_manifest_identity.get("path") or "")
    ).resolve(strict=True)
    if sha256_file(canary_manifest_path) != canary_manifest_identity["file_sha256"]:
        raise GlobalGCEHPCExactError("canary partition manifest bytes changed")
    canary_manifest = validate_partition_manifest(canary_manifest_path)
    if (
        canary_manifest["manifest_sha256"]
        != canary_manifest_identity["manifest_sha256"]
        or canary_manifest.get("scope")
        not in {"SELECTED_ROOTS_CANARY", "SELECTED_PARTITION_CANARY"}
    ):
        raise GlobalGCEHPCExactError("canary partition manifest identity mismatch")
    if (
        merge_manifest.get("schema_version") != MERGE_RESULT_SCHEMA
        or merge_manifest.get("status") != "PASS"
        or merge_manifest.get("scope") != "FULL_MANIFEST"
        or merge_manifest.get("manifest_sha256") != manifest["manifest_sha256"]
        or parity.get("schema_version") != PARITY_RESULT_SCHEMA
        or parity.get("status") != "PASS"
        or parity.get("scientific_input_sha256")
        != manifest["scientific_input_sha256"]
        or parity.get("provenance_sha256")
        != manifest["provenance"]["provenance_sha256"]
        or parity.get("target_branches")
        != manifest["provenance"]["target_branches"]
    ):
        raise GlobalGCEHPCExactError("result bundle requires bound full merge and parity PASS")
    evidence_arguments = {
        "environment_manifest": environment_manifest,
        "slurm_inventory": slurm_inventory,
        "resource_metrics": resource_metrics,
    }
    if any(value is None for value in evidence_arguments.values()):
        raise GlobalGCEHPCExactError(
            "production result bundle requires environment, Slurm inventory, and resource metrics"
        )
    evidence: list[tuple[Path, str]] = []
    evidence_identities: dict[str, dict[str, Any]] = {}
    for role, value in evidence_arguments.items():
        source = Path(value).expanduser().resolve(strict=True)  # type: ignore[arg-type]
        if source.is_symlink() or not source.is_file():
            raise GlobalGCEHPCExactError(f"{role} must be a regular immutable JSON file")
        content = _load_json(source)
        if type(content) is not dict or not content:
            raise GlobalGCEHPCExactError(f"{role} JSON is empty or malformed")
        archive_name = f"evidence/{role}.json"
        evidence.append((source, archive_name))
        evidence_identities[role] = {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "content_sha256": canonical_sha256(content),
        }
    archive = Path(output_tar).expanduser().absolute()
    receipt = Path(output_manifest).expanduser().absolute()
    if archive.exists() or receipt.exists():
        raise GlobalGCEHPCExactError("result bundle outputs must be fresh")
    files = [
        (manifest_path, "partition_manifest.json"),
        (canary_manifest_path, "canary_partition_manifest.json"),
        (merge / "merge_manifest.json", "merge/merge_manifest.json"),
        (merge / "events.jsonl", "merge/events.jsonl"),
        (merge / "patterns.jsonl", "merge/patterns.jsonl"),
        (merge / "rejection_events.jsonl", "merge/rejection_events.jsonl"),
        (merge / "stable_top_k.json", "merge/stable_top_k.json"),
        (parity_path, "parity_receipt.json"),
    ] + evidence
    inventory = [
        {"name": name, "bytes": source.stat().st_size, "sha256": sha256_file(source)}
        for source, name in files
    ]
    bundle_inner = _self_hashed(
        {
            "schema_version": RESULT_BUNDLE_SCHEMA,
            "status": "PASS",
            "manifest_sha256": manifest["manifest_sha256"],
            "merge_result_sha256": merge_manifest["result_sha256"],
            "parity_result_sha256": parity["result_sha256"],
            "parity_manifest_sha256": parity["manifest_sha256"],
            "scientific_input_sha256": manifest["scientific_input_sha256"],
            "provenance_sha256": manifest["provenance"]["provenance_sha256"],
            "target_branches": manifest["provenance"]["target_branches"],
            "external_evidence": evidence_identities,
            "files": inventory,
            "matrix_write_enabled": False,
        },
        field="bundle_content_sha256",
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{archive.name}.", suffix=".tmp", dir=archive.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as tar:
            for source, name in files:
                info = tar.gettarinfo(str(source), arcname=name)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                info.mode = 0o600
                with source.open("rb") as stream:
                    tar.addfile(info, stream)
            inner_bytes = _canonical_bytes(bundle_inner) + b"\n"
            info = tarfile.TarInfo("RESULT_MANIFEST.json")
            info.size = len(inner_bytes)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o600
            import io

            tar.addfile(info, io.BytesIO(inner_bytes))
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, archive)
        _fsync_directory(archive.parent)
    finally:
        temporary.unlink(missing_ok=True)
    outer = _self_hashed(
        {
            **bundle_inner,
            "archive": str(archive),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": sha256_file(archive),
            "created_at": _utc_now(),
        },
        field="receipt_sha256",
    )
    atomic_write_json(receipt, outer)
    return outer


__all__ = [
    "GlobalGCEHPCExactError",
    "TypedDFSEdge",
    "build_partition_manifest",
    "build_result_bundle",
    "dfs_code_from_json",
    "dfs_code_sha256",
    "dfs_code_to_json",
    "merge_exact_shards",
    "run_exact_reference",
    "run_mining_shard",
    "validate_hpc_cli_contract",
    "validate_merge_result",
    "validate_partition_manifest",
    "verify_exact_parity",
]
