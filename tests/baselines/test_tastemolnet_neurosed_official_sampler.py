from __future__ import annotations

from dataclasses import replace

import pytest

import src.data.tastemolnet_neurosed_fixed_budget as sampler


def _chain(graph_id: str, nodes: int, *, class_label: int | None) -> sampler.FixedBudgetGraph:
    edges: list[tuple[int, int]] = []
    for node in range(nodes - 1):
        edges.extend(((node, node + 1), (node + 1, node)))
    return sampler.FixedBudgetGraph(
        graph_id=graph_id,
        split="train",
        node_labels=tuple(node % 4 for node in range(nodes)),
        directed_edges=tuple(edges),
        scaffold=f"scaffold-{graph_id}",
        class_label=class_label,
    )


def _graphs() -> list[sampler.FixedBudgetGraph]:
    return [
        _chain(f"graph-{index:02d}", 8 + index, class_label=index % 3)
        for index in range(9)
    ]


def _sample(graphs: list[sampler.FixedBudgetGraph]) -> list[sampler.FixedBudgetPair]:
    return sampler.sample_fixed_budget_pairs(
        graphs,
        split="train",
        pair_count=80,
        seed=7,
        n_hops_query=64,
        traversal_probability_query=1.0,
    )


def _scientific_sampling_identity(
    pairs: list[sampler.FixedBudgetPair],
) -> list[tuple[object, ...]]:
    return [
        (
            pair.pair_id,
            pair.query.source_graph_id,
            pair.target.graph_id,
            pair.query.selected_source_nodes,
            pair.query.sampling_seed,
            pair.sampling_seed,
        )
        for pair in pairs
    ]


def test_class_labels_are_post_sampling_diagnostics_only() -> None:
    labeled = _graphs()
    relabeled = [replace(graph, class_label=0) for graph in labeled]

    first = _sample(labeled)
    second = _sample(relabeled)

    assert _scientific_sampling_identity(first) == _scientific_sampling_identity(second)
    assert [pair.sampling_stratum for pair in first] != [
        pair.sampling_stratum for pair in second
    ]


def test_size_bins_are_computed_after_sampling_and_cannot_change_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graphs = _graphs()
    baseline = _sample(graphs)

    monkeypatch.setattr(
        sampler,
        "_diagnostic_size_bins",
        lambda rows: {graph.graph_id: "diagnostic-only" for graph in rows},
    )
    changed_diagnostics = _sample(graphs)

    assert _scientific_sampling_identity(baseline) == _scientific_sampling_identity(
        changed_diagnostics
    )
    assert {pair.sampling_stratum for pair in changed_diagnostics} == {
        "query_size=diagnostic-only|target_size=diagnostic-only|class=same",
        "query_size=diagnostic-only|target_size=diagnostic-only|class=cross",
    }


def test_manifest_declares_unstratified_independent_draw_contract() -> None:
    pairs = _sample(_graphs())
    manifest = sampler.fixed_budget_pair_manifest(
        pairs,
        split="train",
        seed=7,
        n_hops_query=64,
        traversal_probability_query=1.0,
        node_limit_query=None,
    )

    assert manifest["pair_builder"].endswith("unstratified_query_target_v2")
    assert manifest["source_target_draws_with_replacement"] is True
    assert manifest["source_target_rng_streams_independent"] is True
    assert manifest["distinct_graph_ids_enforced_by_rejection"] is True
    assert manifest["size_or_class_used_to_select_filter_or_order_pairs"] is False
    assert manifest["size_and_class_diagnostics_computed_after_sampling"] is True
    assert all(pair.query.source_graph_id != pair.target.graph_id for pair in pairs)


def test_independent_draw_requires_at_least_two_graphs() -> None:
    with pytest.raises(
        sampler.FixedBudgetPairError,
        match="requires at least two graphs",
    ):
        sampler.sample_fixed_budget_pairs(
            [_chain("only", 8, class_label=1)],
            split="train",
            pair_count=1,
            seed=7,
            n_hops_query=64,
            traversal_probability_query=1.0,
        )


def test_pair_sampling_seed_is_not_operator_selectable() -> None:
    with pytest.raises(sampler.FixedBudgetPairError, match="fixed at 7"):
        sampler.sample_fixed_budget_pairs(
            _graphs(),
            split="train",
            pair_count=8,
            seed=17,
            n_hops_query=64,
            traversal_probability_query=1.0,
        )
    pairs = _sample(_graphs())
    with pytest.raises(sampler.FixedBudgetPairError, match="fixed at 7"):
        sampler.fixed_budget_pair_manifest(
            pairs,
            split="train",
            seed=17,
            n_hops_query=64,
            traversal_probability_query=1.0,
            node_limit_query=None,
        )
