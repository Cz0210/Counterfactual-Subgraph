"""Exact, isolated root-sharding canary for pinned GlobalGCE gSpan.

This module is deliberately not wired into a production runner.  It compares
an exhaustive serial run over a caller-selected root subset with the union of
independent exact-top-k root shards.  A PASS proves only that selected canary
input; it never authorizes replacement of an active T8/T13 process.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from collections import defaultdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sqlite3
import time
from typing import Any, Mapping, Sequence

from src.baselines.globalgce_resumable import (
    _graph_input_fingerprint,
    resumable_gspan_root_chunks,
    validate_exact_top_k_audit,
)


CANARY_SCHEMA = "globalgce_exact_root_sharding_canary_v1"


class RootShardingCanaryError(RuntimeError):
    """The isolated root-sharding canary could not close exactly."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(_canonical_bytes(dict(payload)) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def normalize_root_indices(values: Sequence[int]) -> tuple[int, ...]:
    roots = tuple(values)
    if (
        not roots
        or any(type(value) is not int or value < 0 for value in roots)
        or tuple(sorted(set(roots))) != roots
    ):
        raise RootShardingCanaryError(
            "root indices must be non-empty, unique, non-negative, and increasing"
        )
    return roots


def plan_disjoint_root_shards(
    root_indices: Sequence[int], *, shard_count: int
) -> tuple[tuple[int, ...], ...]:
    roots = normalize_root_indices(root_indices)
    if type(shard_count) is not int or shard_count < 1 or shard_count > len(roots):
        raise RootShardingCanaryError("shard_count must be in [1, root_count]")
    shards = tuple(tuple(roots[offset::shard_count]) for offset in range(shard_count))
    flattened = [value for shard in shards for value in shard]
    if sorted(flattened) != list(roots) or len(flattened) != len(set(flattened)):
        raise AssertionError("root shard plan is not an exact partition")
    return shards


def _json_scalar(value: Any, *, field: str) -> int | str:
    if type(value) not in (int, str):
        raise RootShardingCanaryError(f"{field} must be an integer or string")
    return value


def load_graph_jsonl(path: str | Path) -> tuple[list[Any], str]:
    """Load a frozen real gSpan graph list without changing insertion order."""

    source = Path(path).expanduser().resolve(strict=True)
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - remote dependency gate
        raise RootShardingCanaryError("networkx is required for the T8 canary") from exc
    graphs: list[Any] = []
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
                raise RootShardingCanaryError(
                    f"invalid graph JSON at line {line_number}"
                ) from exc
            if type(row) is not dict or set(row) != {"graph_id", "nodes", "edges"}:
                raise RootShardingCanaryError(
                    "each graph row must contain exactly graph_id, nodes, and edges"
                )
            if type(row["nodes"]) is not list or not row["nodes"]:
                raise RootShardingCanaryError("each graph must have at least one node")
            if type(row["edges"]) is not list:
                raise RootShardingCanaryError("graph edges must be a list")
            graph = nx.Graph()
            node_ids: list[int] = []
            for node in row["nodes"]:
                if type(node) is not dict or set(node) != {"id", "label"}:
                    raise RootShardingCanaryError("node schema mismatch")
                node_id = node["id"]
                if type(node_id) is not int or node_id < 0:
                    raise RootShardingCanaryError("node IDs must be non-negative integers")
                node_ids.append(node_id)
                graph.add_node(
                    node_id, label=_json_scalar(node["label"], field="node label")
                )
            if node_ids != list(range(len(node_ids))):
                raise RootShardingCanaryError(
                    "node IDs must be consecutive and in native insertion order"
                )
            for edge in row["edges"]:
                if type(edge) is not dict or set(edge) != {"source", "target", "label"}:
                    raise RootShardingCanaryError("edge schema mismatch")
                left, right = edge["source"], edge["target"]
                if type(left) is not int or type(right) is not int:
                    raise RootShardingCanaryError("edge endpoints must be integers")
                if left not in graph or right not in graph or left == right:
                    raise RootShardingCanaryError("edge endpoint or self-loop is invalid")
                if graph.has_edge(left, right):
                    raise RootShardingCanaryError("parallel/duplicate edges are unsupported")
                graph.add_edge(
                    left,
                    right,
                    label=_json_scalar(edge["label"], field="edge label"),
                )
            graphs.append(graph)
    if not graphs:
        raise RootShardingCanaryError("graph JSONL is empty")
    return graphs, _sha256_file(source)


def _pattern_sha256(graph: Any) -> str:
    nodes = sorted(
        ((repr(node), repr(attributes.get("label"))) for node, attributes in graph.nodes(data=True))
    )
    edges = sorted(
        (
            min(repr(left), repr(right)),
            max(repr(left), repr(right)),
            repr(attributes.get("label")),
        )
        for left, right, attributes in graph.edges(data=True)
    )
    return hashlib.sha256(_canonical_bytes({"nodes": nodes, "edges": edges})).hexdigest()


def _find_single_support_root(checkpoint_root: Path) -> Path:
    candidates = sorted(checkpoint_root.glob("support_*"))
    if len(candidates) != 1 or not candidates[0].is_dir():
        raise RootShardingCanaryError("expected exactly one gSpan support directory")
    return candidates[0]


def _read_ranked_rows(checkpoint_root: Path) -> list[dict[str, Any]]:
    support_root = _find_single_support_root(checkpoint_root)
    database = support_root / "frequent_patterns.sqlite3"
    if not database.is_file():
        raise RootShardingCanaryError("gSpan canary SQLite output is missing")
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=120) as connection:
        rows = connection.execute(
            "SELECT support, root_index, local_index, payload FROM patterns "
            "ORDER BY support DESC, root_index ASC, local_index ASC"
        ).fetchall()
    return [
        {
            "support": int(support),
            "root_index": int(root_index),
            "local_index": int(local_index),
            "pattern_sha256": _pattern_sha256(pickle.loads(payload)),
        }
        for support, root_index, local_index, payload in rows
    ]


def merge_ranked_shard_rows(
    shard_rows: Sequence[Sequence[Mapping[str, Any]]], *, top_k: int
) -> list[dict[str, Any]]:
    if type(top_k) is not int or top_k <= 0:
        raise RootShardingCanaryError("top_k must be positive")
    flattened: list[dict[str, Any]] = []
    seen_positions: set[tuple[int, int]] = set()
    for rows in shard_rows:
        for raw in rows:
            if set(raw) != {"support", "root_index", "local_index", "pattern_sha256"}:
                raise RootShardingCanaryError("shard row schema mismatch")
            row = dict(raw)
            if (
                type(row["support"]) is not int
                or type(row["root_index"]) is not int
                or type(row["local_index"]) is not int
                or type(row["pattern_sha256"]) is not str
                or len(row["pattern_sha256"]) != 64
            ):
                raise RootShardingCanaryError("shard row type mismatch")
            position = (row["root_index"], row["local_index"])
            if position in seen_positions:
                raise RootShardingCanaryError("duplicate root/local position across shards")
            seen_positions.add(position)
            flattened.append(row)
    flattened.sort(key=lambda row: (-row["support"], row["root_index"], row["local_index"]))
    return flattened[:top_k]


@dataclass(frozen=True)
class RootPartitionJob:
    graph_jsonl: str
    official_src: str
    checkpoint_root: str
    scratch_root: str | None
    selected_root_indices: tuple[int, ...]
    min_support: int
    min_vertices: int
    max_vertices: int
    top_k: int
    exact_top_k_pruning: bool


def _import_official(official_src: str) -> Any:
    # Import in a disposable worker: the pinned adapter validates and closes
    # the official module namespace rather than trusting ambient sys.path.
    from src.baselines.globalgce_mutagenicity_adapter import _import_official_modules

    return _import_official_modules(Path(official_src).resolve(strict=True))["gspan_module"]


def inspect_root_universe(job: RootPartitionJob) -> tuple[str, ...]:
    graphs, _input_sha = load_graph_jsonl(job.graph_jsonl)
    module = _import_official(job.official_src)
    miner = module.gSpan(
        graphs,
        job.min_support,
        job.min_vertices,
        job.max_vertices,
        len(graphs),
        where=False,
    )
    miner._read_graphs()
    miner._generate_1edge_frequent_subgraphs()
    globals_ = miner.run.__globals__
    roots: dict[Any, Any] = defaultdict(globals_["Projected"])
    for graph_id, graph in miner.graphs.items():
        for vertex_id, vertex in graph.vertices.items():
            for edge in miner._get_forward_root_edges(graph, vertex_id):
                roots[(vertex.vlb, edge.elb, graph.vertices[edge.to].vlb)].append(
                    globals_["PDFS"](graph_id, edge, None)
                )
    return tuple(repr(value) for value in roots)


def inspect_production_input_identity(job: RootPartitionJob) -> dict[str, Any]:
    """Rebuild the unsharded production fingerprint before a canary starts.

    The selected-root canary necessarily has a different checkpoint
    fingerprint because its execution scope is smaller.  This preflight uses
    the exact settings of the protected exhaustive route and therefore binds
    the exported graph list to that route without reading or modifying its
    active SQLite database.
    """

    graphs, graph_jsonl_sha256 = load_graph_jsonl(job.graph_jsonl)
    module = _import_official(job.official_src)
    miner = module.gSpan(
        graphs,
        job.min_support,
        job.min_vertices,
        job.max_vertices,
        len(graphs),
        where=False,
    )
    miner._read_graphs()
    miner._generate_1edge_frequent_subgraphs()
    globals_ = miner.run.__globals__
    roots: dict[Any, Any] = defaultdict(globals_["Projected"])
    for graph_id, graph in miner.graphs.items():
        for vertex_id, vertex in graph.vertices.items():
            for edge in miner._get_forward_root_edges(graph, vertex_id):
                roots[(vertex.vlb, edge.elb, graph.vertices[edge.to].vlb)].append(
                    globals_["PDFS"](graph_id, edge, None)
                )
    settings = {
        "min_support": miner._min_support,
        "min_vertices": miner._min_num_vertices,
        "max_vertices": miner._max_num_vertices,
        "is_undirected": miner._is_undirected,
        "root_order": [repr(value) for value in roots],
        "top_k": int(job.top_k),
        "spill_schema": "sqlite_stable_support_topk_v2",
    }
    return {
        "graph_jsonl_sha256": graph_jsonl_sha256,
        "graph_count": len(graphs),
        "root_count": len(roots),
        "root_order": settings["root_order"],
        "production_input_fingerprint": _graph_input_fingerprint(graphs, settings),
    }


def run_root_partition(job: RootPartitionJob) -> dict[str, Any]:
    roots = normalize_root_indices(job.selected_root_indices)
    checkpoint_root = Path(job.checkpoint_root).expanduser().absolute()
    if checkpoint_root.exists():
        raise RootShardingCanaryError("partition checkpoint root must be fresh")
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    scratch_root: Path | None = None
    if job.scratch_root is not None:
        scratch_root = Path(job.scratch_root).expanduser().absolute()
        if scratch_root.exists():
            raise RootShardingCanaryError("partition scratch root must be fresh")
        scratch_root.mkdir(parents=True, exist_ok=False)
    graphs, graph_jsonl_sha256 = load_graph_jsonl(job.graph_jsonl)
    module = _import_official(job.official_src)
    miner = module.gSpan(
        graphs,
        job.min_support,
        job.min_vertices,
        job.max_vertices,
        len(graphs),
        where=str(checkpoint_root / "official-report-unused.txt"),
    )
    started = time.monotonic()
    with resumable_gspan_root_chunks(
        module,
        checkpoint_root=checkpoint_root,
        scratch_root=scratch_root,
        top_k=job.top_k,
        exact_top_k_pruning=job.exact_top_k_pruning,
        selected_root_indices=roots,
    ):
        miner.run()
    elapsed = time.monotonic() - started
    support_root = _find_single_support_root(checkpoint_root)
    if job.exact_top_k_pruning:
        validate_exact_top_k_audit(support_root / "exact_top_k_audit.json")
    rows = _read_ranked_rows(checkpoint_root)[: job.top_k]
    payload = {
        "schema_version": "globalgce_root_partition_canary_v1",
        "scope": "SELECTED_ROOTS_ONLY",
        "status": "PASS",
        "graph_jsonl_sha256": graph_jsonl_sha256,
        "selected_root_indices": list(roots),
        "exact_top_k_pruning": job.exact_top_k_pruning,
        "top_k": job.top_k,
        "elapsed_seconds": elapsed,
        "rows": rows,
        "rows_sha256": hashlib.sha256(_canonical_bytes(rows)).hexdigest(),
        "scientific_parity_claimed": False,
        "replacement_authorized": False,
    }
    _atomic_json(checkpoint_root / "partition_manifest.json", payload)
    return payload


def run_exact_root_sharding_canary(
    *,
    graph_jsonl: str | Path,
    official_src: str | Path,
    output_root: str | Path,
    root_indices: Sequence[int],
    shard_count: int,
    min_support: int,
    min_vertices: int,
    max_vertices: int,
    top_k: int,
    scratch_root: str | Path | None = None,
    expected_production_input_fingerprint: str | None = None,
) -> dict[str, Any]:
    roots = normalize_root_indices(root_indices)
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
        raise RootShardingCanaryError("gSpan canary bounds are invalid")
    output = Path(output_root).expanduser().absolute()
    if output.exists():
        raise RootShardingCanaryError("canary output root must be fresh")
    output.mkdir(parents=True, exist_ok=False)
    scratch = Path(scratch_root).expanduser().absolute() if scratch_root else None
    if scratch is not None:
        if scratch.exists():
            raise RootShardingCanaryError("canary scratch root must be fresh")
        scratch.mkdir(parents=True, exist_ok=False)
    graph_path = Path(graph_jsonl).expanduser().resolve(strict=True)
    official_path = Path(official_src).expanduser().resolve(strict=True)
    base = dict(
        graph_jsonl=str(graph_path),
        official_src=str(official_path),
        min_support=int(min_support),
        min_vertices=int(min_vertices),
        max_vertices=int(max_vertices),
        top_k=int(top_k),
    )
    probe = RootPartitionJob(
        **base,
        checkpoint_root=str(output / "probe-unused"),
        scratch_root=None,
        selected_root_indices=roots,
        exact_top_k_pruning=False,
    )
    production_input = inspect_production_input_identity(probe)
    root_universe = tuple(production_input["root_order"])
    if expected_production_input_fingerprint is not None:
        expected = str(expected_production_input_fingerprint).strip().lower()
        if (
            len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or production_input["production_input_fingerprint"] != expected
        ):
            raise RootShardingCanaryError(
                "exported graph input differs from the protected production route"
            )
    if roots[-1] >= len(root_universe):
        raise RootShardingCanaryError("selected root is outside the real root universe")
    reference_job = RootPartitionJob(
        **base,
        checkpoint_root=str(output / "serial_reference"),
        scratch_root=None,
        selected_root_indices=roots,
        exact_top_k_pruning=False,
    )
    reference = run_root_partition(reference_job)
    shards = plan_disjoint_root_shards(roots, shard_count=shard_count)
    jobs = [
        RootPartitionJob(
            **base,
            checkpoint_root=str(output / "shards" / f"shard-{index:03d}"),
            scratch_root=(
                str(scratch / f"shard-{index:03d}") if scratch is not None else None
            ),
            selected_root_indices=shard,
            exact_top_k_pruning=True,
        )
        for index, shard in enumerate(shards)
    ]
    parallel_started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=len(jobs), mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        shard_results = list(executor.map(run_root_partition, jobs))
    parallel_seconds = time.monotonic() - parallel_started
    merged = merge_ranked_shard_rows(
        [result["rows"] for result in shard_results], top_k=top_k
    )
    reference_semantics = [
        {"support": row["support"], "pattern_sha256": row["pattern_sha256"]}
        for row in reference["rows"]
    ]
    merged_semantics = [
        {"support": row["support"], "pattern_sha256": row["pattern_sha256"]}
        for row in merged
    ]
    equivalent = reference_semantics == merged_semantics
    report = {
        "schema_version": CANARY_SCHEMA,
        "scope": "SELECTED_REAL_ROOTS_EXACTNESS_CANARY",
        "status": "PASS" if equivalent else "FAILED",
        "graph_jsonl": str(graph_path),
        "graph_jsonl_sha256": _sha256_file(graph_path),
        "official_src": str(official_path),
        "root_universe_count": len(root_universe),
        "root_universe_sha256": hashlib.sha256(
            _canonical_bytes(list(root_universe))
        ).hexdigest(),
        "selected_root_indices": list(roots),
        "production_input_identity": production_input,
        "protected_production_input_match": (
            expected_production_input_fingerprint is not None
        ),
        "shards": [list(shard) for shard in shards],
        "reference_rows": reference_semantics,
        "merged_rows": merged_semantics,
        "canonical_patterns_equal": equivalent,
        "supports_equal": [row["support"] for row in reference_semantics]
        == [row["support"] for row in merged_semantics],
        "stable_order_equal": equivalent,
        "serial_seconds": reference["elapsed_seconds"],
        "parallel_seconds": parallel_seconds,
        "observed_canary_speedup": (
            reference["elapsed_seconds"] / parallel_seconds if parallel_seconds else None
        ),
        "scientific_parity_claimed": False,
        "full_root_universe_parity_claimed": False,
        "score_parity_claimed": False,
        "rejection_count_parity_claimed": False,
        "replacement_authorized": False,
    }
    _atomic_json(output / "canary_report.json", report)
    if not equivalent:
        raise RootShardingCanaryError("root-shard union diverged from serial reference")
    return report


__all__ = [
    "CANARY_SCHEMA",
    "RootPartitionJob",
    "RootShardingCanaryError",
    "inspect_root_universe",
    "inspect_production_input_identity",
    "load_graph_jsonl",
    "merge_ranked_shard_rows",
    "normalize_root_indices",
    "plan_disjoint_root_shards",
    "run_exact_root_sharding_canary",
    "run_root_partition",
]
