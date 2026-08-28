from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

import pytest

import src.eval.tastemolnet_neurosed_fixed_budget as fixed_budget_eval
from src.data.tastemolnet_neurosed_fixed_budget import (
    FixedBudgetGraph,
    FixedBudgetPairError,
    PAIR_SAMPLER_MANIFEST_SCHEMA,
    fixed_budget_pair_manifest,
    partition_disjoint_benchmarks,
    reserve_pair_count,
    sample_fixed_budget_pairs,
    sample_official_style_query,
)
from src.eval.tastemolnet_neurosed_fixed_budget import (
    NeuroSEDFixedBudgetError,
    OFFICIAL_SED_EDIT_COSTS,
    PAIR_LABELS_MANIFEST_SCHEMA,
    build_official_pair_labels_manifest,
    combine_disjoint_benchmark_reports,
    directional_ged_cache_key,
    ged_cache_key,
    ged_cache_symmetry_policy,
    official_ged_interval_label,
    select_fixed_pair_budget,
    select_gedlib_worker_count,
    select_successful_reserve_pairs,
    summarize_real_gedlib_observations,
    validation_pair_budget,
)
from src.eval.tastemolnet_neurosed_official_fixed_budget import (
    GENERATED_QUERY_ROLE,
    OFFICIAL_GCF_COMMIT,
    OFFICIAL_GCF_REPOSITORY,
    OFFICIAL_FIXED_MODEL_CARD_SCHEMA,
    ORIGINAL_TARGET_ROLE,
    VENDORED_GCF_RETAINED_INVENTORY_SHA256,
    VENDORED_GCF_SOURCE_SHA256,
    GeneratedQueryOriginalTargetBinding,
    OfficialFixedBudgetGateError,
    validate_official_fixed_budget_model_card,
    verify_official_fixed_budget_readiness,
    verify_vendored_gcf_retained_inventory,
)
from src.eval.tastemolnet_neurosed_gate import STRICT_OFFICIAL_PROVENANCE
from src.train.tastemolnet_neurosed_official_selector import (
    OfficialBatchInterleavedSelector,
    OfficialSelectorError,
)
from src.utils.tastemolnet_neurosed_gedlib_build import blocked_build_manifest


SHA = "a" * 64
COMMIT = "b" * 40
GCF_ROOT = Path(__file__).resolve().parents[2] / "baselines/gcfexplainer_official"


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
                "worker_resource_evidence": _pass_resource_evidence()
            }
            if resource_gates
            else None
        ),
    )


def _pass_resource_evidence() -> dict:
    payload = {
        "schema_version": fixed_budget_eval.GEDLIB_WORKER_RESOURCE_EVIDENCE_SCHEMA,
        "status": "PASS",
        "marker": None,
        "machine_generated": True,
        "producer_implemented": True,
        "producer_source_sha256": "9" * 64,
        "benchmark_process_identity": {"pid": 123, "pid_start_ticks": 456},
        "pre_sample": {"unix_time_ns": 1, "iowait_ticks": 10},
        "during_samples": [{"unix_time_ns": 2, "iowait_ticks": 11}],
        "post_sample": {"unix_time_ns": 3, "iowait_ticks": 12},
        "protected_processes": {
            "bace_legacy": {"pid": 10, "pid_start_ticks": 20},
            "aids_exact": {"pid": 30, "pid_start_ticks": 40},
        },
        "missing_required_evidence": [],
        "host_load_gate_pass": True,
        "iowait_gate_pass": True,
        "bace_legacy_throughput_drop_percent": 0.0,
        "aids_exact_throughput_drop_percent": 0.0,
    }
    payload["evidence_sha256"] = _stable(payload)
    return payload


def _future_worker_manifest(
    monkeypatch: pytest.MonkeyPatch,
    *,
    physical_core_count: int = 1,
) -> dict:
    monkeypatch.setattr(
        fixed_budget_eval,
        "REVIEWED_WORKER_RESOURCE_EVIDENCE_PRODUCER_SHA256",
        "9" * 64,
    )
    monkeypatch.setattr(
        fixed_budget_eval, "WORKER_TRIAL_COHORT_BUILDER_IMPLEMENTED", True
    )
    monkeypatch.setattr(
        fixed_budget_eval,
        "REVIEWED_WORKER_TRIAL_COHORT_BUILDER_SHA256",
        "8" * 64,
    )
    reports = {
        workers: _report(
            f"future-worker-{workers}",
            100,
            workers=workers,
            resource_gates=True,
        )
        for workers in (1, 2, 4, 8)
        if workers <= physical_core_count
    }
    return fixed_budget_eval.build_gedlib_worker_selection_manifest(
        reports, physical_core_count=physical_core_count
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


def test_budget_planner_selects_largest_affordable_approved_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report("p", 1000, latency=4.0, workers=1)
    plan = select_fixed_pair_budget(
        report,
        worker_selection_manifest=_future_worker_manifest(monkeypatch),
        disk_reservation_pass=True,
    )
    assert plan["status"] == "PASS"
    assert plan["selected_neurosed_train_pair_budget"] == 10000
    assert plan["selected_neurosed_validation_pair_budget"] == 2000
    assert plan["shortcut_fallback_used"] is False
    assert plan["approximate_label_fallback_used"] is False


def test_budget_planner_blocks_instead_of_falling_back_to_shortcut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report("p", 1000, latency=20.0, workers=1)
    plan = select_fixed_pair_budget(
        report,
        worker_selection_manifest=_future_worker_manifest(monkeypatch),
        disk_reservation_pass=True,
    )
    assert plan["status"] == "BLOCKED_GEDLIB_THROUGHPUT"
    assert plan["selected_neurosed_train_pair_budget"] is None
    assert plan["selected_neurosed_validation_pair_budget"] is None
    assert set(plan["projections"]) == {"5000", "10000", "20000"}


def test_timeout_or_reservation_gate_blocks_budget_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_selection = _future_worker_manifest(monkeypatch)
    report = _report("p", 1000, latency=1.0, timeouts=51, workers=1)
    plan = select_fixed_pair_budget(
        report,
        worker_selection_manifest=worker_selection,
        disk_reservation_pass=True,
    )
    assert plan["status"] == "BLOCKED_GEDLIB_THROUGHPUT"
    healthy = _report("q", 1000, latency=1.0, workers=1)
    no_disk = select_fixed_pair_budget(
        healthy,
        worker_selection_manifest=worker_selection,
        disk_reservation_pass=False,
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


def test_worker_selector_uses_fresh_pairs_and_contention_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fixed_budget_eval,
        "REVIEWED_WORKER_RESOURCE_EVIDENCE_PRODUCER_SHA256",
        "9" * 64,
    )
    monkeypatch.setattr(
        fixed_budget_eval, "WORKER_TRIAL_COHORT_BUILDER_IMPLEMENTED", True
    )
    monkeypatch.setattr(
        fixed_budget_eval,
        "REVIEWED_WORKER_TRIAL_COHORT_BUILDER_SHA256",
        "8" * 64,
    )
    reports = {
        1: _report("w1", 100, workers=1, resource_gates=True),
        2: _report("w2", 100, workers=2, resource_gates=True),
    }
    assert select_gedlib_worker_count(reports, physical_core_count=2) == 2
    reports[2]["pair_ids"][0] = reports[1]["pair_ids"][0]
    reports[2]["pair_ids_sha256"] = hashlib.sha256(
        json_bytes(reports[2]["pair_ids"])
    ).hexdigest()
    with pytest.raises(NeuroSEDFixedBudgetError, match="reused"):
        select_gedlib_worker_count(reports, physical_core_count=2)


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


def test_cache_symmetry_is_derived_before_reverse_sharing() -> None:
    asymmetric = ged_cache_symmetry_policy(OFFICIAL_SED_EDIT_COSTS)
    assert asymmetric["symmetric"] is False
    assert asymmetric["share_reverse_cache"] is False
    symmetric_costs = {
        **OFFICIAL_SED_EDIT_COSTS,
        "node_insertion": 1,
        "edge_insertion": 1,
    }
    symmetric = ged_cache_symmetry_policy(symmetric_costs)
    assert symmetric["symmetric"] is True
    forward = ged_cache_key(
        query_canonical_graph_sha256="1" * 64,
        target_canonical_graph_sha256="2" * 64,
        gedlib_config_sha256="3" * 64,
        feature_schema_sha256="4" * 64,
        edit_cost_contract=symmetric_costs,
    )
    reverse = ged_cache_key(
        query_canonical_graph_sha256="2" * 64,
        target_canonical_graph_sha256="1" * 64,
        gedlib_config_sha256="3" * 64,
        feature_schema_sha256="4" * 64,
        edit_cost_contract=symmetric_costs,
    )
    assert forward == reverse


def _reserve_observations(requested: int) -> list[dict]:
    rows = []
    for index in range(requested + 1):
        timeout = index == 0
        rows.append(
            {
                "pair_id": hashlib.sha256(f"pair-{index}".encode()).hexdigest(),
                "query_graph_id": f"query-{index}",
                "target_graph_id": f"target-{index}",
                "query_split": "train",
                "target_split": "train",
                "status": "TIMEOUT" if timeout else "SUCCESS",
                "lower_bound": None if timeout else 1.25,
                "upper_bound": None if timeout else 2.75,
                "exact_bound": None if timeout else False,
                "query_canonical_graph_sha256": f"{index + 1:064x}",
                "target_canonical_graph_sha256": f"{index + 101:064x}",
            }
        )
    return rows


def test_official_label_contract_keeps_both_bounds_and_reserve_order() -> None:
    observations = _reserve_observations(10)
    reserve = select_successful_reserve_pairs(observations, requested_pair_count=10)
    assert reserve["selected_pair_ids"] == [row["pair_id"] for row in observations[1:]]
    labels = [
        official_ged_interval_label(
            row,
            gedlib_commit=COMMIT,
            pyged_module_sha256=SHA,
            gedlib_config_sha256=SHA,
            feature_schema_sha256=SHA,
            pair_sampler_manifest_sha256=SHA,
            gedlib_build_manifest_sha256=SHA,
            ged_method_args="--threads 1 --time-limit 1",
        )
        for row in observations[1:]
    ]
    assert all(label["bound_average_used"] is False for label in labels)
    assert all(label["lower_bound"] == 1.25 for label in labels)
    assert all(label["upper_bound"] == 2.75 for label in labels)
    assert all(label["reverse_cache_shared"] is False for label in labels)
    manifest = build_official_pair_labels_manifest(
        labels,
        split="train",
        requested_pair_count=10,
        reserve_selection=reserve,
        compact_storage_format="numpy_npz",
        compact_labels_sha256="c" * 64,
    )
    assert manifest["schema_version"] == PAIR_LABELS_MANIFEST_SCHEMA
    assert manifest["successful_pair_count"] == 10
    assert manifest["timeout_count"] == 1
    assert manifest["reserve_used"] == 1
    assert manifest["interval_bound_pair_count"] == 10
    with pytest.raises(NeuroSEDFixedBudgetError, match="only SUCCESS"):
        official_ged_interval_label(
            observations[0],
            gedlib_commit=COMMIT,
            pyged_module_sha256=SHA,
            gedlib_config_sha256=SHA,
            feature_schema_sha256=SHA,
            pair_sampler_manifest_sha256=SHA,
            gedlib_build_manifest_sha256=SHA,
            ged_method_args="--threads 1 --time-limit 1",
        )


def _stopped_selector_trace() -> dict:
    selector = OfficialBatchInterleavedSelector(
        cycle_patience=1,
        step_size_up=1,
        step_size_down=1,
    )
    metrics = (5.0, 4.0, 4.0, 4.0, 4.0)
    for index, metric in enumerate(metrics):
        decision = selector.observe_validation(metric, training_batch_index=index)
        if decision.checkpoint_candidate:
            selector.bind_checkpoint_candidate(
                validation_event_index=decision.validation_event_index,
                checkpoint_sha256=("a" if index == 0 else "b") * 64,
            )
        if decision.paired_training_batch_allowed:
            selector.record_training_update(
                training_batch_index=index,
                optimizer_step_completed=True,
                cyclic_lr_step_completed=True,
                gradient_clip_norm=0.1,
            )
    return selector.trace_manifest()


def test_official_selector_is_validation_before_batch_and_strict_on_ties() -> None:
    trace = _stopped_selector_trace()
    assert trace["validation_event_count"] == 5
    assert trace["completed_training_batch_count"] == 4
    assert trace["selected_validation_event_index"] == 1
    assert trace["selected_checkpoint_sha256"] == "b" * 64
    assert trace["trace"][2]["strict_improvement"] is False
    assert trace["trace"][-1]["stop_before_training_batch"] is True
    assert trace["trace"][-1]["training_update_completed"] is False
    selector = OfficialBatchInterleavedSelector(
        cycle_patience=1, step_size_up=1, step_size_down=1
    )
    with pytest.raises(OfficialSelectorError, match="preceding validation"):
        selector.record_training_update(
            training_batch_index=0,
            optimizer_step_completed=True,
            cyclic_lr_step_completed=True,
            gradient_clip_norm=0.1,
        )
    selector.observe_validation(1.0, training_batch_index=0)
    with pytest.raises(OfficialSelectorError, match="pre-update bytes"):
        selector.record_training_update(
            training_batch_index=0,
            optimizer_step_completed=True,
            cyclic_lr_step_completed=True,
            gradient_clip_norm=0.1,
        )


class _FakeDistanceModel:
    def __init__(self) -> None:
        self.targets = []

    def embed_targets(self, targets) -> None:
        self.targets = list(targets)

    def predict_outer_with_queries(self, queries, batch_size=None):
        del batch_size
        return [[0.0 for _target in self.targets] for _query in queries]


def test_distance_binding_exposes_generated_query_to_original_target_only() -> None:
    binding = GeneratedQueryOriginalTargetBinding.create(
        _FakeDistanceModel(),
        original_targets=["original-a", "original-b"],
        original_target_hashes=["1" * 64, "2" * 64],
    )
    distances = binding.predict_generated_queries(
        ["generated"], generated_query_hashes=["3" * 64]
    )
    assert distances == [[0.0, 0.0]]
    trace = binding.direction_manifest()
    assert trace["query_role"] == GENERATED_QUERY_ROLE
    assert trace["target_role"] == ORIGINAL_TARGET_ROLE
    assert trace["reverse_direction_used"] is False
    assert trace["records"][0]["query_graph_hash"] == "3" * 64
    assert trace["records"][0]["target_graph_hash"] == "1" * 64


def _stable(value) -> str:
    import json

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _sampler_manifest(split: str, selected_budget: int) -> dict:
    payload = {
        "schema_version": PAIR_SAMPLER_MANIFEST_SCHEMA,
        "dataset": "tastemolnet",
        "split": split,
        "pair_count": reserve_pair_count(selected_budget),
        "pair_sampling_seed": 7,
        "pair_builder": (
            "deterministic_official_style_independent_unstratified_query_target_v2"
        ),
        "independent_query_target_pairs": True,
        "query_graph_id_differs_from_target_graph_id": True,
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
        "source_csv_sha256": "e" * 64,
        "feature_schema_sha256": SHA,
        "graph_inventory_sha256": "f" * 64,
        "pair_ids_sha256": "1" * 64,
        "query_graph_ids_sha256": "2" * 64,
        "target_graph_ids_sha256": "3" * 64,
        "metadata_rows_sha256": "4" * 64,
    }
    payload["manifest_sha256"] = _stable(payload)
    return payload


def _label_manifest(
    split: str, count: int, *, pair_sampler_manifest_sha256: str
) -> dict:
    payload = {
        "schema_version": PAIR_LABELS_MANIFEST_SCHEMA,
        "status": "READY_FOR_INDEPENDENT_VERIFICATION",
        "split": split,
        "requested_pair_count": count,
        "successful_pair_count": count,
        "exact_bound_pair_count": 0,
        "interval_bound_pair_count": count,
        "real_pyged_gedlib_labels": True,
        "timeout_or_error_rows_used_as_labels": False,
        "selected_in_sampler_order": True,
        "ged_value_based_selection_used": False,
        "finite_labels": True,
        "all_lower_bounds_le_upper_bounds": True,
        "cache_symmetric": False,
        "reverse_cache_shared": False,
        "query_target_order_in_cache_key": True,
        "large_per_pair_json_debug_dump_used": False,
        "compact_storage_format": "numpy_npz",
        "compact_labels_sha256": "d" * 64,
        "gedlib_commit": COMMIT,
        "pyged_module_sha256": SHA,
        "gedlib_config_sha256": SHA,
        "feature_schema_sha256": SHA,
        "pair_sampler_manifest_sha256": pair_sampler_manifest_sha256,
        "gedlib_build_manifest_sha256": SHA,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    payload["manifest_sha256"] = _stable(payload)
    return payload


def _strict_fixed_model_card(train_labels, val_labels, selector, direction) -> dict:
    selected_checkpoint = selector["selected_checkpoint_sha256"]
    return {
        "schema_version": OFFICIAL_FIXED_MODEL_CARD_SCHEMA,
        "dataset": "tastemolnet",
        "role": "GCF_AUXILIARY_DISTANCE_MODEL",
        "classifier": False,
        "source_label_independent": True,
        "train_only_fit": True,
        "validation_only_selection": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "pair_budget_strategy": "fixed_budget_resource_control",
        "fixed_pair_budget_is_project_extension": True,
        "official_pair_semantics": True,
        "fixed_budget_extension_documented": True,
        "upstream_greed_independent_pair_role_semantics_unchanged": True,
        "upstream_greed_sampler_byte_for_byte_unchanged": False,
        "exhaustive_pairs": False,
        "cartesian_product_materialized": False,
        "independent_query_target_pairs": True,
        "query_graph_id_differs_from_target_graph_id": True,
        "parent_own_subgraph_shortcut": False,
        "class_label_used_as_supervision": False,
        "real_pyged_gedlib_labels": True,
        "ged_method": "f2",
        "ged_method_switched_from_official": False,
        "approximate_or_neural_labels_used": False,
        "timeout_or_error_rows_used_as_labels": False,
        "label_representation": "ordered_query_target_lower_upper_interval",
        "pyged_return_dtype": "float64",
        "label_dtype": "float32",
        "label_transform": "official_torch_float32_storage_cast_only",
        "bound_average_used": False,
        "single_bound_substitution_used": False,
        "training_loop_authority": "neuro.train.train_full_batch_interleaved_validation",
        "upstream_greed_batch_interleaved_selection_loop_unchanged": True,
        "strict_official_batch_interleaved_selector_implemented": True,
        "gcf_runtime_direction": "generated_query_to_original_target",
        "training_direction_matches_gcf_runtime": True,
        "checkpoint_reload_passed": True,
        "batch_single_inference_passed": True,
        "finite_labels": True,
        "all_lower_bounds_le_upper_bounds": True,
        "official_selection_trace_authenticated": True,
        "gcf_runner_load_passed": True,
        "feature_schema_compatible": True,
        "pair_sampling_seed": 7,
        "deterministic_reserve_fraction": 0.10,
        "disk_reservation_pass": True,
        "cpu_contention_gate_pass": True,
        "worker_wrote_pass": False,
        "scientific_release_eligible": True,
        "full_official_neurosed_semantics_claimed": True,
        "train_pair_budget": 5000,
        "validation_pair_budget": 1000,
        "successful_train_pair_count": 5000,
        "successful_validation_pair_count": 1000,
        "edit_cost_contract": dict(OFFICIAL_SED_EDIT_COSTS),
        "strict_official_provenance": dict(STRICT_OFFICIAL_PROVENANCE),
        "vendored_gcf_source_sha256": dict(VENDORED_GCF_SOURCE_SHA256),
        "vendored_gcf_retained_inventory_sha256": (
            VENDORED_GCF_RETAINED_INVENTORY_SHA256
        ),
        "official_gcf_repository": OFFICIAL_GCF_REPOSITORY,
        "official_gcf_commit": OFFICIAL_GCF_COMMIT,
        "official_greed_commit": STRICT_OFFICIAL_PROVENANCE["greed_commit"],
        "gedlib_commit": COMMIT,
        "pyged_module_sha256": SHA,
        "gedlib_build_manifest_sha256": SHA,
        "gedlib_config_sha256": SHA,
        "feature_schema_sha256": SHA,
        "gedlib_benchmark_summary_sha256": SHA,
        "pair_budget_plan_sha256": SHA,
        "train_pair_labels_manifest_sha256": train_labels["manifest_sha256"],
        "validation_pair_labels_manifest_sha256": val_labels["manifest_sha256"],
        "train_pair_sampler_manifest_sha256": train_labels[
            "pair_sampler_manifest_sha256"
        ],
        "validation_pair_sampler_manifest_sha256": val_labels[
            "pair_sampler_manifest_sha256"
        ],
        "selector_trace_sha256": selector["trace_sha256"],
        "distance_direction_trace_sha256": direction["trace_sha256"],
        "selected_checkpoint_sha256": selected_checkpoint,
    }


def test_model_card_and_health_gate_are_ready_not_self_signed_pass() -> None:
    train_sampler = _sampler_manifest("train", 5000)
    validation_sampler = _sampler_manifest("validation", 1000)
    train_labels = _label_manifest(
        "train",
        5000,
        pair_sampler_manifest_sha256=train_sampler["manifest_sha256"],
    )
    validation_labels = _label_manifest(
        "validation",
        1000,
        pair_sampler_manifest_sha256=validation_sampler["manifest_sha256"],
    )
    selector = _stopped_selector_trace()
    binding = GeneratedQueryOriginalTargetBinding.create(
        _FakeDistanceModel(),
        original_targets=["original"],
        original_target_hashes=["1" * 64],
    )
    binding.predict_generated_queries(
        ["generated"], generated_query_hashes=["2" * 64]
    )
    direction = binding.direction_manifest()
    card = _strict_fixed_model_card(train_labels, validation_labels, selector, direction)
    validate_official_fixed_budget_model_card(card, vendored_gcf_root=GCF_ROOT)
    readiness = verify_official_fixed_budget_readiness(
        model_card=card,
        train_pair_sampler_manifest=train_sampler,
        validation_pair_sampler_manifest=validation_sampler,
        train_pair_labels_manifest=train_labels,
        validation_pair_labels_manifest=validation_labels,
        selector_trace=selector,
        distance_direction_trace=direction,
        vendored_gcf_root=GCF_ROOT,
    )
    assert readiness["status"] == "READY_FOR_MANAGED_INDEPENDENT_VERIFICATION"
    assert readiness["marker"] is None
    assert readiness["scientific_pass_claimed"] is False
    wrong_seed = dict(train_sampler)
    wrong_seed["pair_sampling_seed"] = 17
    wrong_seed["manifest_sha256"] = _stable(
        {key: value for key, value in wrong_seed.items() if key != "manifest_sha256"}
    )
    with pytest.raises(OfficialFixedBudgetGateError, match="sampler contract changed"):
        verify_official_fixed_budget_readiness(
            model_card=card,
            train_pair_sampler_manifest=wrong_seed,
            validation_pair_sampler_manifest=validation_sampler,
            train_pair_labels_manifest=train_labels,
            validation_pair_labels_manifest=validation_labels,
            selector_trace=selector,
            distance_direction_trace=direction,
            vendored_gcf_root=GCF_ROOT,
        )
    blocked = dict(card)
    blocked["official_gcf_commit"] = "UNAVAILABLE_FROM_VENDORED_SNAPSHOT"
    with pytest.raises(OfficialFixedBudgetGateError, match="GCF commit"):
        validate_official_fixed_budget_model_card(
            blocked, vendored_gcf_root=GCF_ROOT
        )
    wrong_commit = dict(card)
    wrong_commit["official_gcf_commit"] = "c" * 40
    with pytest.raises(OfficialFixedBudgetGateError, match="GCF commit changed"):
        validate_official_fixed_budget_model_card(
            wrong_commit, vendored_gcf_root=GCF_ROOT
        )
    wrong_repository = dict(card)
    wrong_repository["official_gcf_repository"] = "https://example.invalid/fork"
    with pytest.raises(OfficialFixedBudgetGateError, match="GCF repository changed"):
        validate_official_fixed_budget_model_card(
            wrong_repository, vendored_gcf_root=GCF_ROOT
        )
    shortcut = dict(card)
    shortcut["parent_own_subgraph_shortcut"] = True
    with pytest.raises(OfficialFixedBudgetGateError, match="contract changed"):
        validate_official_fixed_budget_model_card(
            shortcut, vendored_gcf_root=GCF_ROOT
        )


def test_vendored_gcf_inventory_is_reopened_and_tamper_fails(tmp_path: Path) -> None:
    verified = verify_vendored_gcf_retained_inventory(GCF_ROOT)
    assert verified["file_count"] == 17
    assert verified["inventory_sha256"] == VENDORED_GCF_RETAINED_INVENTORY_SHA256

    tampered = tmp_path / "gcf"
    shutil.copytree(GCF_ROOT, tampered)
    (tampered / "distance.py").write_bytes(
        (tampered / "distance.py").read_bytes() + b"\n# tampered\n"
    )
    with pytest.raises(OfficialFixedBudgetGateError, match="file hash changed"):
        verify_vendored_gcf_retained_inventory(tampered)

    shutil.copytree(GCF_ROOT, tmp_path / "gcf-extra")
    (tmp_path / "gcf-extra" / "unexpected.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(OfficialFixedBudgetGateError, match="file inventory changed"):
        verify_vendored_gcf_retained_inventory(tmp_path / "gcf-extra")

    real_parent = tmp_path / "real-parent"
    shutil.copytree(GCF_ROOT, real_parent / "gcf")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(OfficialFixedBudgetGateError, match="symlink ancestor"):
        verify_vendored_gcf_retained_inventory(linked_parent / "gcf")
