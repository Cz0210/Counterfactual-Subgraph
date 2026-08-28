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
PAIR_SAMPLER_MANIFEST_SCHEMA = "tastemolnet_neurosed_fixed_budget_pairs_v1"
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
    """An independent query-source/target pair with no GED label.

    ``sampling_stratum`` is a backward-compatible metadata name for a
    diagnostic computed *after* the ordered source/target draws are complete.
    It must never influence which graphs are drawn, accepted, or ordered.
    """

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


def _diagnostic_size_bins(
    graphs: Sequence[FixedBudgetGraph],
) -> dict[str, str]:
    """Return post-sampling size diagnostics; never use these bins to draw."""

    ordered = sorted(graphs, key=lambda graph: (graph.num_nodes, graph.graph_id))
    result: dict[str, str] = {}
    names = ("small", "medium", "large")
    for index, graph in enumerate(ordered):
        bucket = min(2, (3 * index) // len(ordered))
        result[graph.graph_id] = names[bucket]
    return result


def _diagnostic_stratum(
    query_source: FixedBudgetGraph,
    target: FixedBudgetGraph,
    bins: Mapping[str, str],
) -> str:
    """Describe an already-sampled pair without affecting its identity."""

    if query_source.class_label is None or target.class_label is None:
        class_relation = "unknown"
    elif query_source.class_label == target.class_label:
        class_relation = "same"
    else:
        class_relation = "cross"
    return (
        f"query_size={bins[query_source.graph_id]}|"
        f"target_size={bins[target.graph_id]}|class={class_relation}"
    )


@dataclass(frozen=True, slots=True)
class _FixedBudgetPairDraft:
    """A sampled scientific pair before any class/size diagnostics are read."""

    query_source: FixedBudgetGraph
    query: FixedBudgetQuery
    target: FixedBudgetGraph
    sampling_seed: int


def _pair_id(
    *,
    split: str,
    query_graph_id: str,
    target_graph_id: str,
    query_instance_sha256: str,
    sampling_seed: int,
) -> str:
    return _stable_sha256(
        {
            "split": split,
            "query_graph_id": query_graph_id,
            "target_graph_id": target_graph_id,
            "query_instance_sha256": query_instance_sha256,
            "sampling_seed": int(sampling_seed),
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
    """Sample official-style independent draws under a deterministic budget.

    Query sources and targets are drawn with replacement from the complete
    same-split input sequence using separate deterministic RNG streams.  The
    only pair-level rejection is the required distinct-graph-ID constraint
    (plus failure to construct a valid official-style query).  Class labels,
    graph sizes, and scaffolds are not read until all ordered draws have been
    accepted, when they are attached as diagnostics only.
    """

    if split not in ALLOWED_SPLITS:
        raise FixedBudgetPairError("pair split must be train or validation")
    if type(seed) is not int or seed != PAIR_SAMPLING_SEED:
        raise FixedBudgetPairError("pair sampling seed must remain fixed at 7")
    if int(pair_count) <= 0:
        raise FixedBudgetPairError("pair_count must be positive")
    if not graphs or any(graph.split != split for graph in graphs):
        raise FixedBudgetPairError("all graphs must belong to the requested split")
    if len({graph.graph_id for graph in graphs}) != len(graphs):
        raise FixedBudgetPairError("graph IDs must be unique within the split")
    if len(graphs) < 2:
        raise FixedBudgetPairError(
            "independent query-target sampling requires at least two graphs"
        )

    graph_pool = tuple(graphs)
    query_source_rng = random.Random(
        _seed_for(seed, split, "official_query_source_draws")
    )
    target_rng = random.Random(_seed_for(seed, split, "official_target_draws"))
    drafts: list[_FixedBudgetPairDraft] = []
    maximum = int(max_pair_attempts or max(10_000, pair_count * 64))
    attempt = 0
    while len(drafts) < int(pair_count) and attempt < maximum:
        query_source = graph_pool[query_source_rng.randrange(len(graph_pool))]
        target = graph_pool[target_rng.randrange(len(graph_pool))]
        attempt += 1
        if query_source.graph_id == target.graph_id:
            continue
        query_seed = _seed_for(
            seed,
            split,
            "query_instance",
            len(drafts),
            attempt,
            query_source.graph_id,
        )
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
        pair_seed = _seed_for(
            seed,
            split,
            "pair_instance",
            len(drafts),
            attempt,
            target.graph_id,
        )
        drafts.append(
            _FixedBudgetPairDraft(
                query_source=query_source,
                query=query,
                target=target,
                sampling_seed=pair_seed,
            )
        )
    if len(drafts) != int(pair_count):
        raise FixedBudgetPairError(
            f"sampled {len(drafts)} of {pair_count} requested independent pairs"
        )

    # This is deliberately a second phase: diagnostics cannot feed back into
    # source/target draws, query retries, acceptance, or ordered pair identity.
    bins = _diagnostic_size_bins(graph_pool)
    pairs: list[FixedBudgetPair] = []
    seen_pair_ids: set[str] = set()
    for draft in drafts:
        stratum_name = _diagnostic_stratum(
            draft.query_source,
            draft.target,
            bins,
        )
        identifier = _pair_id(
            split=split,
            query_graph_id=draft.query_source.graph_id,
            target_graph_id=draft.target.graph_id,
            query_instance_sha256=draft.query.graph_sha256,
            sampling_seed=draft.sampling_seed,
        )
        if identifier in seen_pair_ids:
            raise FixedBudgetPairError("sampled pair instance identity collision")
        seen_pair_ids.add(identifier)
        pairs.append(
            FixedBudgetPair(
                pair_id=identifier,
                query=draft.query,
                target=draft.target,
                query_scaffold=draft.query_source.scaffold,
                target_scaffold=draft.target.scaffold,
                sampling_seed=draft.sampling_seed,
                sampling_stratum=stratum_name,
            )
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
    if type(seed) is not int or seed != PAIR_SAMPLING_SEED:
        raise FixedBudgetPairError("pair manifest seed must remain fixed at 7")
    rows = [pair.metadata() for pair in pairs]
    query_ids = {row["query_graph_id"] for row in rows}
    target_ids = {row["target_graph_id"] for row in rows}
    query_ids_hash = _stable_sha256(sorted(query_ids))
    target_ids_hash = _stable_sha256(sorted(target_ids))
    relations = {
        row["sampling_stratum"].rsplit("=", 1)[-1]
        for row in rows
    }
    payload = {
        "schema_version": PAIR_SAMPLER_MANIFEST_SCHEMA,
        "dataset": "tastemolnet",
        "split": split,
        "pair_count": len(rows),
        "pair_sampling_seed": int(seed),
        "pair_builder": (
            "deterministic_official_style_independent_unstratified_query_target_v2"
        ),
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
        "source_target_draws_with_replacement": True,
        "source_target_rng_streams_independent": True,
        "distinct_graph_ids_enforced_by_rejection": True,
        "size_or_class_used_to_select_filter_or_order_pairs": False,
        "size_and_class_diagnostics_computed_after_sampling": True,
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
    return bind_fixed_budget_pair_manifest(payload)


def bind_fixed_budget_pair_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the canonical self-hashed sampler manifest after all additions."""

    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = _stable_sha256(payload)
    return payload


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
    "PAIR_SAMPLER_MANIFEST_SCHEMA",
    "bind_fixed_budget_pair_manifest",
    "fixed_budget_pair_manifest",
    "partition_disjoint_benchmarks",
    "reserve_pair_count",
    "sample_fixed_budget_pairs",
    "sample_official_style_query",
]
