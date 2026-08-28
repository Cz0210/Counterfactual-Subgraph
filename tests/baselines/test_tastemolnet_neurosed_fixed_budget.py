from __future__ import annotations

import hashlib

import pytest

from src.data.tastemolnet_neurosed_fixed_budget import (
    FixedBudgetGraph,
    FixedBudgetPairError,
    fixed_budget_pair_manifest,
    partition_disjoint_benchmarks,
    reserve_pair_count,
    sample_fixed_budget_pairs,
    sample_official_style_query,
)
from src.eval.tastemolnet_neurosed_fixed_budget import (
    NeuroSEDFixedBudgetError,
    combine_disjoint_benchmark_reports,
    directional_ged_cache_key,
    select_fixed_pair_budget,
    select_gedlib_worker_count,
    select_successful_reserve_pairs,
    summarize_real_gedlib_observations,
    validation_pair_budget,
)
from src.utils.tastemolnet_neurosed_gedlib_build import blocked_build_manifest


SHA = "a" * 64
COMMIT = "b" * 40


def _chain(graph_id: str, nodes: int, *, split: str = "train", label: int = 0) -> FixedBudgetGraph:
    edges: list[tuple[int, int]] = []
    for node in range(nodes - 1):
        edges.extend(((node, node + 1), (node + 1, node)))
    return FixedBudgetGraph(
        graph_id=graph_id,
        split=split,
        node_labels=tuple((node + label) % 4 for node in range(nodes)),
        directed_edges=tuple(edges),
        scaffold=f"scaffold-{graph_id}",
        class_label=label,
    )


def _graphs(split: str = "train") -> list[FixedBudgetGraph]:
    return [
        _chain(f"g-{index:02d}", 7 + index, split=split, label=index % 3)
        for index in range(12)
    ]


def _pairs(count: int = 1600):
    return sample_fixed_budget_pairs(
        _graphs(),
        split="train",
        pair_count=count,
        seed=7,
        n_hops_query=64,
        traversal_probability_query=1.0,
    )


def _observations(prefix: str, count: int, *, latency: float = 1.0, timeouts: int = 0):
    rows = []
    for index in range(count):
        timeout = index < timeouts
        rows.append(
            {
                "pair_id": f"{prefix}-{index}",
                "status": "TIMEOUT" if timeout else "SUCCESS",
                "latency_seconds": latency,
                "lower_bound": None if timeout else 1.0,
                "upper_bound": None if timeout else 2.0,
                "exact_bound": None if timeout else False,
                "query_num_nodes": 5 + index % 3,
                "target_num_nodes": 8 + index % 5,
                "query_num_edges": 4 + index % 3,
                "target_num_edges": 7 + index % 5,
            }
        )
    return rows


def _report(
    prefix: str,
    count: int,
    *,
    latency: float = 1.0,
    timeouts: int = 0,
    workers: int = 1,
    resource_gates: bool = False,
):
    return summarize_real_gedlib_observations(
        _observations(prefix, count, latency=latency, timeouts=timeouts),
        benchmark_budget=count,
        worker_count=workers,
        wall_seconds=max(1.0, count * latency / workers),
        pyged_module_sha256=SHA,
        gedlib_commit=COMMIT,
        gedlib_config_sha256=SHA,
        feature_schema_sha256=SHA,
        resource_metrics=(
            {
                "host_load_gate_pass": True,
                "iowait_gate_pass": True,
                "bace_legacy_throughput_drop_le_10pct": True,
                "aids_exact_throughput_drop_le_10pct": True,
            }
            if resource_gates
            else None
        ),
    )


def test_query_sampler_is_reconstructable_and_uses_upstream_node_zero_frontier() -> None:
    graph = _chain("source", 8)
    first = sample_official_style_query(
        graph,
        n_hops=5,
        traversal_probability=1.0,
        node_limit=None,
        sampling_seed=19,
    )
    second = sample_official_style_query(
        graph,
        n_hops=5,
        traversal_probability=1.0,
        node_limit=None,
        sampling_seed=19,
    )
    assert first == second
    assert first.selected_source_nodes == (0, 1, 2, 3, 4, 5)
    assert first.num_nodes == 6
    assert first.num_undirected_edges == 5


def test_fixed_sampler_is_deterministic_independent_and_never_labels_pairs() -> None:
    first = _pairs(80)
    second = _pairs(80)
    assert first == second
    assert all(pair.query.source_graph_id != pair.target.graph_id for pair in first)
    assert len({pair.pair_id for pair in first}) == len(first)
    rows = [pair.metadata() for pair in first]
    assert all(row["ged_label_present"] is False for row in rows)
    assert all("lower_bound" not in row and "upper_bound" not in row for row in rows)
    relations = {row["sampling_stratum"].rsplit("=", 1)[-1] for row in rows}
    assert relations == {"same", "cross"}
    manifest = fixed_budget_pair_manifest(
        first,
        split="train",
        seed=7,
        n_hops_query=64,
        traversal_probability_query=1.0,
        node_limit_query=None,
    )
    assert manifest["independent_query_target_pairs"] is True
    assert manifest["parent_own_subgraph_shortcut"] is False
    assert manifest["cartesian_product_materialized"] is False
    assert manifest["ged_labels_present"] is False


def test_sampler_rejects_split_drift() -> None:
    with pytest.raises(FixedBudgetPairError, match="requested split"):
        sample_fixed_budget_pairs(
            _graphs("validation"),
            split="train",
            pair_count=1,
            n_hops_query=64,
            traversal_probability_query=1.0,
        )


def test_benchmark_cohorts_are_disjoint_and_have_exact_sizes() -> None:
    cohorts = partition_disjoint_benchmarks(_pairs())
    assert {budget: len(rows) for budget, rows in cohorts.items()} == {
        100: 100,
        500: 500,
        1000: 1000,
    }
    sets = [{pair.pair_id for pair in cohorts[budget]} for budget in (100, 500, 1000)]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])


def test_reserve_and_validation_budgets_follow_fixed_contract() -> None:
    assert reserve_pair_count(5000) == 5500
    assert validation_pair_budget(5000) == 1000
    assert validation_pair_budget(10000) == 2000
    assert validation_pair_budget(20000) == 4000
    with pytest.raises(NeuroSEDFixedBudgetError):
        validation_pair_budget(7500)


def test_real_benchmark_summary_rejects_labels_on_timeouts() -> None:
    observations = _observations("x", 100, timeouts=1)
    observations[0]["lower_bound"] = 0.0
    with pytest.raises(NeuroSEDFixedBudgetError, match="must not retain"):
        summarize_real_gedlib_observations(
            observations,
            benchmark_budget=100,
            worker_count=1,
            wall_seconds=100.0,
            pyged_module_sha256=SHA,
            gedlib_commit=COMMIT,
            gedlib_config_sha256=SHA,
            feature_schema_sha256=SHA,
        )


def test_benchmark_summary_proves_pair_level_disjointness() -> None:
    reports = {
        100: _report("a", 100),
        500: _report("b", 500),
        1000: _report("c", 1000),
    }
    summary = combine_disjoint_benchmark_reports(reports)
    assert summary["benchmark_pair_cohorts_disjoint"] is True
    reports[500]["pair_ids"][0] = reports[100]["pair_ids"][0]
    reports[500]["pair_ids_sha256"] = hashlib.sha256(
        json_bytes(reports[500]["pair_ids"])
    ).hexdigest()
    with pytest.raises(NeuroSEDFixedBudgetError, match="reused"):
        combine_disjoint_benchmark_reports(reports)


def json_bytes(value) -> bytes:
    import json

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def test_budget_planner_selects_largest_affordable_approved_budget() -> None:
    report = _report("p", 1000, latency=4.0, workers=1)
    plan = select_fixed_pair_budget(
        report,
        selected_workers=1,
        disk_reservation_pass=True,
        cpu_contention_gate_pass=True,
    )
    assert plan["status"] == "PASS"
    assert plan["selected_neurosed_train_pair_budget"] == 10000
    assert plan["selected_neurosed_validation_pair_budget"] == 2000
    assert plan["shortcut_fallback_used"] is False
    assert plan["approximate_label_fallback_used"] is False


def test_budget_planner_blocks_instead_of_falling_back_to_shortcut() -> None:
    report = _report("p", 1000, latency=20.0, workers=1)
    plan = select_fixed_pair_budget(
        report,
        selected_workers=1,
        disk_reservation_pass=True,
        cpu_contention_gate_pass=True,
    )
    assert plan["status"] == "BLOCKED_GEDLIB_THROUGHPUT"
    assert plan["selected_neurosed_train_pair_budget"] is None
    assert plan["selected_neurosed_validation_pair_budget"] is None
    assert set(plan["projections"]) == {"5000", "10000", "20000"}


def test_timeout_or_reservation_gate_blocks_budget_selection() -> None:
    report = _report("p", 1000, latency=1.0, timeouts=51, workers=1)
    plan = select_fixed_pair_budget(
        report,
        selected_workers=1,
        disk_reservation_pass=True,
        cpu_contention_gate_pass=True,
    )
    assert plan["status"] == "BLOCKED_GEDLIB_THROUGHPUT"
    healthy = _report("q", 1000, latency=1.0, workers=1)
    no_disk = select_fixed_pair_budget(
        healthy,
        selected_workers=1,
        disk_reservation_pass=False,
        cpu_contention_gate_pass=True,
    )
    assert no_disk["status"] == "BLOCKED_GEDLIB_THROUGHPUT"


def test_directional_cache_never_shares_reverse_sed_pairs() -> None:
    forward = directional_ged_cache_key(
        query_canonical_graph_sha256="1" * 64,
        target_canonical_graph_sha256="2" * 64,
        gedlib_config_sha256="3" * 64,
        feature_schema_sha256="4" * 64,
    )
    reverse = directional_ged_cache_key(
        query_canonical_graph_sha256="2" * 64,
        target_canonical_graph_sha256="1" * 64,
        gedlib_config_sha256="3" * 64,
        feature_schema_sha256="4" * 64,
    )
    assert forward != reverse
    with pytest.raises(NeuroSEDFixedBudgetError, match="direction"):
        directional_ged_cache_key(
            query_canonical_graph_sha256="1" * 64,
            target_canonical_graph_sha256="2" * 64,
            gedlib_config_sha256="3" * 64,
            feature_schema_sha256="4" * 64,
            direction="target_to_query",
        )


def test_timeout_reserve_uses_sampler_order_and_never_invents_labels() -> None:
    rows = []
    for index in range(11):
        timeout = index in {0, 3}
        rows.append(
            {
                "pair_id": f"reserve-{index}",
                "status": "TIMEOUT" if timeout else "SUCCESS",
                "lower_bound": None if timeout else 1.0,
                "upper_bound": None if timeout else 2.0,
            }
        )
    selection = select_successful_reserve_pairs(rows, requested_pair_count=10)
    assert selection["status"] == "BLOCKED_GEDLIB_LABEL_YIELD"
    assert selection["successful_pair_count"] == 9
    rows[3].update(status="SUCCESS", lower_bound=1.0, upper_bound=2.0)
    selection = select_successful_reserve_pairs(rows, requested_pair_count=10)
    assert selection["status"] == "PASS"
    assert selection["selected_pair_ids"] == [
        f"reserve-{index}" for index in range(1, 11)
    ]
    assert selection["reserve_used"] == 1


def test_worker_selector_uses_fresh_pairs_and_contention_gates() -> None:
    reports = {
        1: _report("w1", 100, workers=1, resource_gates=True),
        2: _report("w2", 100, workers=2, resource_gates=True),
    }
    assert select_gedlib_worker_count(reports, physical_core_count=4) == 2
    reports[2]["pair_ids"][0] = reports[1]["pair_ids"][0]
    reports[2]["pair_ids_sha256"] = hashlib.sha256(
        json_bytes(reports[2]["pair_ids"])
    ).hexdigest()
    with pytest.raises(NeuroSEDFixedBudgetError, match="reused"):
        select_gedlib_worker_count(reports, physical_core_count=4)


def test_blocked_build_can_never_emit_a_pass_marker_or_fallback() -> None:
    payload = blocked_build_manifest(
        error=RuntimeError("GEDLIB absent"),
        greed_root="/pinned/greed",
        greed_expts_root="/pinned/greed-expts",
        gedlib_root="/missing/gedlib",
        output_root="/fresh/build",
    )
    assert payload["status"] == "BLOCKED_GEDLIB_BUILD"
    assert payload["marker"] is None
    assert payload["real_pyged_smoke_passed"] is False
    assert payload["approximate_or_neural_fallback_used"] is False
    assert payload["network_install_performed"] is False
