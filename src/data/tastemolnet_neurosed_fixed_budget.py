"""Deterministic, fixed-budget GREED pair sampling for TasteMolNet.

The upstream GREED ``make_inner_dataset`` contract has two independent draws:
one graph supplies a sampled query and a second graph supplies the target.  The
project extension implemented here caps the number of such draws; it does not
replace them with parent/own-subgraph pairs and it never manufactures GED/SED
labels.  Labels must be added later by the pinned ``pyged/GEDLIB`` route.

This module intentionally has no RDKit, PyTorch, or pyged import.  Production
entrypoints construct :class:`FixedBudgetGraph` objects from held Taste split
bytes, while the sampling contract remains independently unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any, Iterable, Mapping, Sequence


PAIR_SAMPLING_SEED = 7
BENCHMARK_BUDGETS = (100, 500, 1000)
ALLOWED_TRAIN_PAIR_BUDGETS = (5000, 10000, 20000)
ALLOWED_SPLITS = frozenset({"train", "validation"})
FORBIDDEN_LABEL_FIELDS = frozenset(
    {
        "ged",
        "sed",
        "label",
        "lower_bound",
        "upper_bound",
        "lb",
        "ub",
    }
)


class FixedBudgetPairError(RuntimeError):
    """The fixed-budget pair contract is invalid or cannot be satisfied."""


def _stable_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FixedBudgetGraph:
    """One immutable graph from exactly one admitted Taste split."""

    graph_id: str
    split: str
    node_labels: tuple[int, ...]
    directed_edges: tuple[tuple[int, int], ...]
    scaffold: str
    class_label: int | None = None

    def __post_init__(self) -> None:
        if not self.graph_id:
            raise FixedBudgetPairError("graph_id must be non-empty")
        if self.split not in ALLOWED_SPLITS:
            raise FixedBudgetPairError("fixed-budget graphs must be train or validation")
        if len(self.node_labels) < 2:
            raise FixedBudgetPairError("fixed-budget graphs need at least two nodes")
        if not self.scaffold:
            raise FixedBudgetPairError("graph scaffold diagnostic must be non-empty")
        if self.class_label is not None and self.class_label not in (0, 1, 2):
            raise FixedBudgetPairError("Taste diagnostic class must be 0, 1, or 2")
        total = len(self.node_labels)
        edges = set(self.directed_edges)
        if len(edges) != len(self.directed_edges):
            raise FixedBudgetPairError("directed graph edges must be unique")
        for source, target in edges:
            if not 0 <= source < total or not 0 <= target < total or source == target:
                raise FixedBudgetPairError("directed graph edge endpoint is invalid")
            if (target, source) not in edges:
                raise FixedBudgetPairError("molecular graph edges must be symmetric")
        if not edges:
            raise FixedBudgetPairError("fixed-budget graphs need at least one edge")
        if not _is_connected(total, edges):
            raise FixedBudgetPairError("fixed-budget graph must be connected")

    @property
    def num_nodes(self) -> int:
        return len(self.node_labels)

    @property
    def num_undirected_edges(self) -> int:
        return len(self.directed_edges) // 2

    @property
    def graph_sha256(self) -> str:
        return _stable_sha256(
            {
                "graph_id": self.graph_id,
                "split": self.split,
                "node_labels": self.node_labels,
                "directed_edges": self.directed_edges,
            }
        )

    @property
    def canonical_graph_sha256(self) -> str:
        """Hash graph content only, for directional GED cache identities."""

        return _stable_sha256(
            {
                "node_labels": self.node_labels,
                "directed_edges": self.directed_edges,
            }
        )

    def pyged_data(self) -> tuple[list[int], list[tuple[int, int]]]:
        """Return the exact tuple shape consumed by upstream ``pyged.sed``."""

        return list(self.node_labels), list(self.directed_edges)


@dataclass(frozen=True, slots=True)
class FixedBudgetQuery:
    """One deterministic official-style query sampled from a source graph."""

    source_graph_id: str
    split: str
    node_labels: tuple[int, ...]
    directed_edges: tuple[tuple[int, int], ...]
    selected_source_nodes: tuple[int, ...]
    sampling_seed: int

    @property
    def num_nodes(self) -> int:
        return len(self.node_labels)

    @property
    def num_undirected_edges(self) -> int:
        return len(self.directed_edges) // 2

    @property
    def graph_sha256(self) -> str:
        return _stable_sha256(
            {
                "source_graph_id": self.source_graph_id,
                "split": self.split,
                "node_labels": self.node_labels,
                "directed_edges": self.directed_edges,
                "selected_source_nodes": self.selected_source_nodes,
                "sampling_seed": self.sampling_seed,
            }
        )

    @property
    def canonical_graph_sha256(self) -> str:
        """Hash sampled graph content without its source/replay metadata."""

        return _stable_sha256(
            {
                "node_labels": self.node_labels,
                "directed_edges": self.directed_edges,
            }
        )

    def pyged_data(self) -> tuple[list[int], list[tuple[int, int]]]:
        return list(self.node_labels), list(self.directed_edges)


@dataclass(frozen=True, slots=True)
class FixedBudgetPair:
    """An independent query-source/target pair with no GED label."""

    pair_id: str
    query: FixedBudgetQuery
    target: FixedBudgetGraph
    query_scaffold: str
    target_scaffold: str
    sampling_seed: int
    sampling_stratum: str

    def __post_init__(self) -> None:
        if self.query.source_graph_id == self.target.graph_id:
            raise FixedBudgetPairError("query_graph_id must differ from target_graph_id")
        if self.query.split != self.target.split:
            raise FixedBudgetPairError("query and target must come from the same split")
        expected = _pair_id(
            split=self.query.split,
            query_graph_id=self.query.source_graph_id,
            target_graph_id=self.target.graph_id,
            query_instance_sha256=self.query.graph_sha256,
            sampling_seed=self.sampling_seed,
            sampling_stratum=self.sampling_stratum,
        )
        if self.pair_id != expected:
            raise FixedBudgetPairError("pair_id does not bind the pair payload")

    def metadata(self) -> dict[str, Any]:
        """Return pair metadata only; deliberately omit GED/SED labels."""

        row = {
            "pair_id": self.pair_id,
            "query_graph_id": self.query.source_graph_id,
            "target_graph_id": self.target.graph_id,
            "query_split": self.query.split,
            "target_split": self.target.split,
            "query_num_nodes": self.query.num_nodes,
            "target_num_nodes": self.target.num_nodes,
            "query_num_edges": self.query.num_undirected_edges,
            "target_num_edges": self.target.num_undirected_edges,
            "query_scaffold": self.query_scaffold,
            "target_scaffold": self.target_scaffold,
            "sampling_seed": self.sampling_seed,
            "query_sampling_seed": self.query.sampling_seed,
            "sampling_stratum": self.sampling_stratum,
            "query_instance_sha256": self.query.graph_sha256,
            "target_graph_sha256": self.target.graph_sha256,
            "query_canonical_graph_sha256": self.query.canonical_graph_sha256,
            "target_canonical_graph_sha256": self.target.canonical_graph_sha256,
            "ged_direction": "query_to_target",
            "ged_label_present": False,
        }
        if FORBIDDEN_LABEL_FIELDS.intersection(row):  # pragma: no cover - invariant.
            raise FixedBudgetPairError("sampler metadata contains a GED label field")
        return row


def _is_connected(total: int, edges: Iterable[tuple[int, int]]) -> bool:
    adjacency = [set() for _ in range(total)]
    for source, target in edges:
        adjacency[source].add(target)
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbour in adjacency[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return len(seen) == total


def _seed_for(*parts: Any) -> int:
    digest = hashlib.sha256(
        "\0".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def sample_official_style_query(
    source: FixedBudgetGraph,
    *,
    n_hops: int,
    traversal_probability: float,
    node_limit: int | None,
    sampling_seed: int,
    max_attempts: int = 256,
) -> FixedBudgetQuery:
    """Reproduce the topology semantics of GREED ``make_queries`` deterministically.

    Upstream starts its frontier at node zero, samples traversed directed edges
    for a fixed number of hops, symmetrizes the retained edge set, and accepts
    queries with at least one undirected edge and at least five nodes.  Python's
    local RNG is used instead of ambient PyTorch global state so a persisted
    per-query seed reconstructs exactly the same query.
    """

    if int(n_hops) <= 0:
        raise FixedBudgetPairError("n_hops must be positive")
    if not 0.0 < float(traversal_probability) <= 1.0:
        raise FixedBudgetPairError("traversal_probability must be in (0,1]")
    if node_limit is not None and int(node_limit) < 5:
        raise FixedBudgetPairError("node_limit cannot be smaller than five")
    ordered_edges = tuple(sorted(source.directed_edges))
    for attempt in range(int(max_attempts)):
        rng = random.Random(_seed_for(sampling_seed, "query", attempt))
        front = {0}
        seen_nodes = {0}
        seen_edges: set[tuple[int, int]] = set()
        for _ in range(int(n_hops)):
            traversed = {
                edge
                for edge in ordered_edges
                if edge[0] in front and rng.random() <= traversal_probability
            }
            seen_edges.update(traversed)
            next_front = {target for _source, target in traversed} - seen_nodes
            seen_nodes.update(next_front)
            front = next_front
        symmetrized = seen_edges | {(target, source) for source, target in seen_edges}
        selected = tuple(sorted(seen_nodes))
        if (
            len(selected) < 5
            or not symmetrized
            or (node_limit is not None and len(selected) > int(node_limit))
        ):
            continue
        reverse = {
            source_index: result_index
            for result_index, source_index in enumerate(selected)
        }
        retained = tuple(
            sorted(
                (reverse[source_index], reverse[target_index])
                for source_index, target_index in symmetrized
                if source_index in reverse and target_index in reverse
            )
        )
        if not retained or not _is_connected(len(selected), retained):
            continue
        return FixedBudgetQuery(
            source_graph_id=source.graph_id,
            split=source.split,
            node_labels=tuple(source.node_labels[index] for index in selected),
            directed_edges=retained,
            selected_source_nodes=selected,
            sampling_seed=int(sampling_seed),
        )
    raise FixedBudgetPairError(
        f"could not sample an official-style query from {source.graph_id!r}"
    )


def _size_bins(graphs: Sequence[FixedBudgetGraph]) -> dict[str, str]:
    ordered = sorted(graphs, key=lambda graph: (graph.num_nodes, graph.graph_id))
    result: dict[str, str] = {}
    names = ("small", "medium", "large")
    for index, graph in enumerate(ordered):
        bucket = min(2, (3 * index) // len(ordered))
        result[graph.graph_id] = names[bucket]
    return result


def _strata(
    graphs: Sequence[FixedBudgetGraph],
    bins: Mapping[str, str],
) -> list[tuple[str, str, str]]:
    by_bin: dict[str, list[FixedBudgetGraph]] = {
        name: [graph for graph in graphs if bins[graph.graph_id] == name]
        for name in ("small", "medium", "large")
    }
    all_labeled = all(graph.class_label is not None for graph in graphs)
    result: list[tuple[str, str, str]] = []
    for query_bin in ("small", "medium", "large"):
        for target_bin in ("small", "medium", "large"):
            query_graphs = by_bin[query_bin]
            target_graphs = by_bin[target_bin]
            if not query_graphs or not target_graphs:
                continue
            if not all_labeled:
                if any(q.graph_id != t.graph_id for q in query_graphs for t in target_graphs):
                    result.append((query_bin, target_bin, "unknown"))
                continue
            if any(
                q.graph_id != t.graph_id and q.class_label == t.class_label
                for q in query_graphs
                for t in target_graphs
            ):
                result.append((query_bin, target_bin, "same"))
            if any(
                q.graph_id != t.graph_id and q.class_label != t.class_label
                for q in query_graphs
                for t in target_graphs
            ):
                result.append((query_bin, target_bin, "cross"))
    if not result:
        raise FixedBudgetPairError("no independent query-target stratum is viable")
    return result


def _draw_independent_sources(
    *,
    graphs: Sequence[FixedBudgetGraph],
    bins: Mapping[str, str],
    stratum: tuple[str, str, str],
    rng: random.Random,
    max_attempts: int = 512,
) -> tuple[FixedBudgetGraph, FixedBudgetGraph]:
    query_bin, target_bin, relation = stratum
    query_pool = [graph for graph in graphs if bins[graph.graph_id] == query_bin]
    target_pool = [graph for graph in graphs if bins[graph.graph_id] == target_bin]
    for _ in range(max_attempts):
        query_source = query_pool[rng.randrange(len(query_pool))]
        target = target_pool[rng.randrange(len(target_pool))]
        if query_source.graph_id == target.graph_id:
            continue
        if relation == "same" and query_source.class_label != target.class_label:
            continue
        if relation == "cross" and query_source.class_label == target.class_label:
            continue
        return query_source, target
    raise FixedBudgetPairError(f"could not draw independent sources for {stratum}")


def _pair_id(
    *,
    split: str,
    query_graph_id: str,
    target_graph_id: str,
    query_instance_sha256: str,
    sampling_seed: int,
    sampling_stratum: str,
) -> str:
    return _stable_sha256(
        {
            "split": split,
            "query_graph_id": query_graph_id,
            "target_graph_id": target_graph_id,
            "query_instance_sha256": query_instance_sha256,
            "sampling_seed": int(sampling_seed),
            "sampling_stratum": sampling_stratum,
        }
    )


def sample_fixed_budget_pairs(
    graphs: Sequence[FixedBudgetGraph],
    *,
    split: str,
    pair_count: int,
    seed: int = PAIR_SAMPLING_SEED,
    n_hops_query: int,
    traversal_probability_query: float,
    node_limit_query: int | None = None,
    max_pair_attempts: int | None = None,
) -> list[FixedBudgetPair]:
    """Sample a deterministic budget without constructing a Cartesian product."""

    if split not in ALLOWED_SPLITS:
        raise FixedBudgetPairError("pair split must be train or validation")
    if int(pair_count) <= 0:
        raise FixedBudgetPairError("pair_count must be positive")
    if not graphs or any(graph.split != split for graph in graphs):
        raise FixedBudgetPairError("all graphs must belong to the requested split")
    if len({graph.graph_id for graph in graphs}) != len(graphs):
        raise FixedBudgetPairError("graph IDs must be unique within the split")
    bins = _size_bins(graphs)
    strata = _strata(graphs, bins)
    rng = random.Random(int(seed))
    pairs: list[FixedBudgetPair] = []
    seen_pair_ids: set[str] = set()
    maximum = int(max_pair_attempts or max(10_000, pair_count * 64))
    attempt = 0
    while len(pairs) < int(pair_count) and attempt < maximum:
        stratum = strata[len(pairs) % len(strata)]
        query_source, target = _draw_independent_sources(
            graphs=graphs,
            bins=bins,
            stratum=stratum,
            rng=rng,
        )
        query_seed = _seed_for(seed, split, len(pairs), attempt, query_source.graph_id)
        attempt += 1
        try:
            query = sample_official_style_query(
                query_source,
                n_hops=n_hops_query,
                traversal_probability=traversal_probability_query,
                node_limit=node_limit_query,
                sampling_seed=query_seed,
            )
        except FixedBudgetPairError:
            continue
        stratum_name = (
            f"query_size={stratum[0]}|target_size={stratum[1]}|class={stratum[2]}"
        )
        pair_seed = _seed_for(seed, split, len(pairs), attempt, target.graph_id)
        identifier = _pair_id(
            split=split,
            query_graph_id=query_source.graph_id,
            target_graph_id=target.graph_id,
            query_instance_sha256=query.graph_sha256,
            sampling_seed=pair_seed,
            sampling_stratum=stratum_name,
        )
        if identifier in seen_pair_ids:
            continue
        seen_pair_ids.add(identifier)
        pairs.append(
            FixedBudgetPair(
                pair_id=identifier,
                query=query,
                target=target,
                query_scaffold=query_source.scaffold,
                target_scaffold=target.scaffold,
                sampling_seed=pair_seed,
                sampling_stratum=stratum_name,
            )
        )
    if len(pairs) != int(pair_count):
        raise FixedBudgetPairError(
            f"sampled {len(pairs)} of {pair_count} requested independent pairs"
        )
    return pairs


def partition_disjoint_benchmarks(
    pairs: Sequence[FixedBudgetPair],
) -> dict[int, list[FixedBudgetPair]]:
    """Return disjoint 100-, 500-, and 1000-pair benchmark cohorts."""

    required = sum(BENCHMARK_BUDGETS)
    if len(pairs) < required:
        raise FixedBudgetPairError(
            f"disjoint GED benchmarks require at least {required} pairs"
        )
    result: dict[int, list[FixedBudgetPair]] = {}
    offset = 0
    seen: set[str] = set()
    for budget in BENCHMARK_BUDGETS:
        cohort = list(pairs[offset : offset + budget])
        identifiers = {pair.pair_id for pair in cohort}
        if len(identifiers) != budget or identifiers.intersection(seen):
            raise FixedBudgetPairError("GED benchmark cohorts are not disjoint")
        result[budget] = cohort
        seen.update(identifiers)
        offset += budget
    return result


def fixed_budget_pair_manifest(
    pairs: Sequence[FixedBudgetPair],
    *,
    split: str,
    seed: int,
    n_hops_query: int,
    traversal_probability_query: float,
    node_limit_query: int | None,
) -> dict[str, Any]:
    """Build a fail-closed manifest for unlabeled independent pairs."""

    if not pairs or any(pair.query.split != split for pair in pairs):
        raise FixedBudgetPairError("pair manifest split mismatch")
    rows = [pair.metadata() for pair in pairs]
    query_ids = {row["query_graph_id"] for row in rows}
    target_ids = {row["target_graph_id"] for row in rows}
    query_ids_hash = _stable_sha256(sorted(query_ids))
    target_ids_hash = _stable_sha256(sorted(target_ids))
    relations = {
        row["sampling_stratum"].rsplit("=", 1)[-1]
        for row in rows
    }
    return {
        "schema_version": "tastemolnet_neurosed_fixed_budget_pairs_v1",
        "dataset": "tastemolnet",
        "split": split,
        "pair_count": len(rows),
        "pair_sampling_seed": int(seed),
        "pair_builder": "deterministic_official_style_independent_query_target_v1",
        "official_pair_builder_signature": (
            "neuro.datasets.make_inner_dataset(graphs,n_pairs,n_hops_query,"
            "trav_prob_query,node_lim_query=None,n_hops_target=None,targets=None)"
        ),
        "independent_query_target_pairs": True,
        "query_graph_id_differs_from_target_graph_id": all(
            row["query_graph_id"] != row["target_graph_id"] for row in rows
        ),
        "parent_own_subgraph_shortcut": False,
        "cartesian_product_materialized": False,
        "ged_labels_present": False,
        "class_label_used_as_supervision": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "all_query_ids_subset_of_declared_split": True,
        "all_target_ids_subset_of_declared_split": True,
        "class_label_sampling_diagnostic_relations": sorted(relations),
        "query_generation": {
            "semantics": "upstream_random_bfs_sample_topology_deterministic_rng_extension",
            "n_hops_query": int(n_hops_query),
            "traversal_probability_query": float(traversal_probability_query),
            "node_limit_query": node_limit_query,
            "minimum_query_nodes": 5,
            "minimum_query_undirected_edges": 1,
        },
        "pair_ids_sha256": _stable_sha256([row["pair_id"] for row in rows]),
        "query_graph_ids_sha256": query_ids_hash,
        "target_graph_ids_sha256": target_ids_hash,
        f"{split}_query_ids_hash": query_ids_hash,
        f"{split}_target_ids_hash": target_ids_hash,
        "metadata_rows_sha256": _stable_sha256(rows),
    }


def reserve_pair_count(budget: int) -> int:
    """Return the exact deterministic 10 percent reserve requirement."""

    if int(budget) <= 0:
        raise FixedBudgetPairError("budget must be positive")
    return int(math.ceil(int(budget) * 1.10))


__all__ = [
    "ALLOWED_SPLITS",
    "ALLOWED_TRAIN_PAIR_BUDGETS",
    "BENCHMARK_BUDGETS",
    "FixedBudgetGraph",
    "FixedBudgetPair",
    "FixedBudgetPairError",
    "FixedBudgetQuery",
    "PAIR_SAMPLING_SEED",
    "fixed_budget_pair_manifest",
    "partition_disjoint_benchmarks",
    "reserve_pair_count",
    "sample_fixed_budget_pairs",
    "sample_official_style_query",
]
