from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import src.baselines.comrecgc.graph_trace as graph_trace_module
from src.baselines.comrecgc.graph_trace import (
    ActionTraceRecorder,
    TRACE_IMPORTANCE_ABS_TOLERANCE,
    apply_action_to_normalized_payload,
    assert_trace_parity,
    infer_official_single_edit,
    iter_candidate_lineage_from_selected_trace,
    iter_selected_trace,
    load_selected_trace,
    normalized_untyped_graph_payload,
    recover_candidate_lineage_from_selected_trace,
    stable_graph_sha256,
    stable_untyped_graph_sha256,
)


@dataclass
class Graph:
    x: list[list[float]]
    edge_index: list[list[int]]
    num_nodes: int
    comrecgc_parent_id: str = "parent-1"
    comrecgc_trace_node_ids: tuple[str, ...] = ("source:0", "source:1")


def graph(*, swapped_edges: bool = False, atom: int = 0) -> Graph:
    edges = [[0, 1], [1, 0]]
    if swapped_edges:
        edges = [[1, 0], [0, 1]]
    return Graph(
        x=[[1.0 - atom, float(atom)], [0.0, 1.0]],
        edge_index=edges,
        num_nodes=2,
    )


def payload(first: Graph, second: Graph) -> dict:
    return {
        "graph_map": {"a": [first], "b": [second]},
        "counterfactual_candidates": [
            {"graph_hash": "a", "frequency": 4, "importance_parts": [0.7, 1.0]},
            {"graph_hash": "b", "frequency": 2, "importance_parts": [0.8, 1.0]},
        ],
    }


def test_stable_graph_sha256_normalizes_edge_order() -> None:
    assert stable_graph_sha256(graph()) == stable_graph_sha256(graph(swapped_edges=True))
    assert stable_graph_sha256(graph()) != stable_graph_sha256(graph(atom=1))


def test_untyped_identity_explicitly_ignores_stale_official_edge_sidecar() -> None:
    first = SimpleNamespace(
        x=[[1.0, 0.0], [0.0, 1.0]],
        edge_index=[[0, 1], [1, 0]],
        edge_attr=[1],
        num_nodes=2,
    )
    reordered = SimpleNamespace(
        x=[[1.0, 0.0], [0.0, 1.0]],
        edge_index=[[1, 0], [0, 1]],
        edge_attr=[1],
        num_nodes=2,
    )
    with pytest.raises(ValueError, match="edge_attr is not aligned"):
        stable_graph_sha256(first)
    assert stable_untyped_graph_sha256(first) == stable_untyped_graph_sha256(reordered)


def test_action_trace_uses_untyped_identity_for_stale_official_edge_sidecar() -> None:
    source = SimpleNamespace(
        x=[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        edge_index=[[0, 1], [1, 0]],
        edge_attr=[[1.0], [1.0]],
        num_nodes=3,
        comrecgc_parent_id="parent-1",
        comrecgc_trace_node_ids=("source:0", "source:1", "source:2"),
    )
    target = SimpleNamespace(
        x=[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        edge_index=[[0, 1, 1, 2], [1, 0, 2, 1]],
        edge_attr=[[1.0], [1.0]],
        num_nodes=3,
        comrecgc_parent_id="parent-1",
        comrecgc_trace_node_ids=("source:0", "source:1", "source:2"),
    )
    recorder = ActionTraceRecorder()
    recorder.record_enumerated(
        source_graph=source,
        target_graph=target,
        action=("EA", 1, 2),
    )
    assert len(recorder.enumerated) == 1
    ((source_sha, target_sha),) = recorder.enumerated.keys()
    assert source_sha == stable_untyped_graph_sha256(source)
    assert target_sha == stable_untyped_graph_sha256(target)

    module = SimpleNamespace(graph_map={"source": [source], "target": [target]})
    wrapped = recorder.wrap_move(
        lambda *_args, **_kwargs: (["target"], False, None, None, None),
        module,
    )
    wrapped(
        graphs_hash=["source"],
        start_graphs_hash=["source"],
        importance_args={},
        teleport_probability=0.1,
    )
    lineage = recorder.candidate_lineage(
        {
            "graph_map": module.graph_map,
            "counterfactual_candidates": [{"graph_hash": "target"}],
        }
    )
    assert lineage[0]["action_lineage_resolved"] is True
    assert lineage[0]["actions"][0]["action"] == ["EA", 1, 2]


def test_full_compact_enumeration_defers_graph_hashing_until_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = graph()
    target = graph(atom=1)
    recorder = ActionTraceRecorder(compact_enumeration=True)

    def forbidden_hash(_graph: object) -> str:
        raise AssertionError("enumerated neighbors must not be hashed in compact mode")

    monkeypatch.setattr(
        graph_trace_module,
        "stable_untyped_graph_sha256",
        forbidden_hash,
    )
    target_fields = dict(vars(target))
    recorder.record_enumerated(
        source_graph=source,
        target_graph=target,
        action=("NLC", 0, 1),
    )

    assert recorder.enumerated == {}
    assert recorder.enumerated_transition_count == 1
    assert dict(vars(target)) == target_fields


def test_full_compact_trace_resolves_selected_action_from_upstream_transition(
    tmp_path,
) -> None:
    source = graph()
    target = graph(atom=1)
    recorder = ActionTraceRecorder(
        output_dir=tmp_path,
        chunk_size=1,
        compact_enumeration=True,
    )
    recorder.record_enumerated(
        source_graph=source,
        target_graph=target,
        action=("NLC", 0, 1),
    )
    module = SimpleNamespace(
        graph_map={"source": [source], "target": [target]},
        transitions={},
    )

    def first_move(*_args: object, **_kwargs: object) -> tuple:
        module.transitions["source"] = (
            ["target"],
            [target],
            [[0.7, 1.0]],
            [[0.0]],
        )
        return (["target"], False, None, None, None)

    recorder.wrap_move(first_move, module)(
        graphs_hash=["source"],
        start_graphs_hash=["source"],
        importance_args={},
        teleport_probability=0.1,
    )
    payload_value = {
        "graph_map": module.graph_map,
        "counterfactual_candidates": [{"graph_hash": "target"}],
    }
    summary = recorder.write(
        tmp_path,
        payload_value,
        source_graphs_by_parent_id={"parent-1": source},
        compact_candidate_lineage=True,
    )
    events = load_selected_trace(summary["selected_trace_path"])

    assert events[0]["action_resolution"] == "exact"
    assert events[0]["action"] == ["NLC", 0, 1]
    assert summary["enumeration_trace_mode"] == "weak_target_object_action_index_v1"
    assert summary["transition_cache_hit_count"] == 0
    assert summary["transition_cache_miss_count"] == 1
    assert summary["candidate_payload_mutated"] is False


def test_full_compact_trace_audits_upstream_transition_cache_hit(tmp_path) -> None:
    source = graph()
    target = graph(atom=1)
    recorder = ActionTraceRecorder(
        output_dir=tmp_path,
        chunk_size=1,
        compact_enumeration=True,
    )
    recorder.record_enumerated(
        source_graph=source,
        target_graph=target,
        action=("NLC", 0, 1),
    )
    module = SimpleNamespace(
        graph_map={"source": [source], "target": [target]},
        transitions={
            "source": (["target"], [target], [[0.7, 1.0]], [[0.0]])
        },
    )
    recorder.wrap_move(
        lambda *_args, **_kwargs: (["target"], False, None, None, None),
        module,
    )(
        graphs_hash=["source"],
        start_graphs_hash=["source"],
        importance_args={},
        teleport_probability=0.1,
    )

    assert recorder.transition_cache_hit_count == 1
    assert recorder.transition_cache_miss_count == 0


def test_full_compact_trace_reads_actions_from_memory_bounded_cache() -> None:
    source = graph()
    target = graph(atom=1)
    recorder = ActionTraceRecorder(compact_enumeration=True)

    class CompactTransitions(dict):
        def action_records(self, source_hash: str, target_hash: str) -> list[dict]:
            assert source_hash == "source"
            assert target_hash == "target"
            return [{"action": ["NLC", 0, 1]}]

    module = SimpleNamespace(
        graph_map={"source": [source], "target": [target]},
        transitions=CompactTransitions(source=object()),
    )
    recorder.wrap_move(
        lambda *_args, **_kwargs: (["target"], False, None, None, None),
        module,
    )(
        graphs_hash=["source"],
        start_graphs_hash=["source"],
        importance_args={},
        teleport_probability=0.1,
    )

    assert recorder.predecessor_by_official_hash["target"]["action"] == [
        "NLC",
        0,
        1,
    ]


def test_trace_does_not_change_candidates() -> None:
    reference = payload(graph(), graph(atom=1))
    traced = payload(graph(swapped_edges=True), graph(atom=1))
    result = assert_trace_parity(reference, traced)
    assert result["trace_parity_passed"] is True
    assert result["candidate_count"] == 2


def test_trace_parity_rejects_frequency_or_order_change() -> None:
    reference = payload(graph(), graph(atom=1))
    traced = payload(graph(), graph(atom=1))
    traced["counterfactual_candidates"][0]["frequency"] = 5
    with pytest.raises(ValueError, match="Trace-on/off"):
        assert_trace_parity(reference, traced)


def _record_one_transition(recorder: ActionTraceRecorder) -> tuple[Graph, Graph, SimpleNamespace]:
    source = graph()
    target = graph(atom=1)
    recorder.record_enumerated(
        source_graph=source,
        target_graph=target,
        action=("NLC", 0, 1),
    )
    module = SimpleNamespace(graph_map={"source": [source], "target": [target]})

    return source, target, module


def test_selected_action_lineage_is_exact_and_ordered() -> None:
    recorder = ActionTraceRecorder()
    source, target, module = _record_one_transition(recorder)

    def move(*_args: object, **_kwargs: object) -> tuple:
        return (["target"], False, None, None, None)

    wrapped = recorder.wrap_move(move, module)
    wrapped(
        graphs_hash=["source"],
        start_graphs_hash=["source"],
        importance_args={},
        teleport_probability=0.1,
    )
    candidate_payload = {
        "graph_map": {"source": [source], "target": [target]},
        "counterfactual_candidates": [{"graph_hash": "target"}],
    }
    lineage = recorder.candidate_lineage(candidate_payload)
    assert lineage[0]["action_lineage_resolved"] is True
    assert lineage[0]["actions"][0]["action"] == ["NLC", 0, 1]
    assert lineage[0]["actions"][0]["action_replay_exact"] is True
    assert lineage[0]["actions"][0]["source_node_ids"] == ["source:0", "source:1"]
    assert lineage[0]["actions"][0]["target_node_ids"] == ["source:0", "source:1"]


def test_trace_parity_accepts_numpy_importance_without_truth_coercion() -> None:
    reference = payload(graph(), graph(atom=1))
    traced = payload(graph(), graph(atom=1))
    for value in (reference, traced):
        value["counterfactual_candidates"][0]["importance_parts"] = np.asarray(
            [0.7, 1.0], dtype=np.float64
        )
    assert assert_trace_parity(reference, traced)["trace_parity_passed"] is True


def test_trace_parity_accepts_audited_cuda_float32_importance_drift() -> None:
    reference = payload(graph(), graph(atom=1))
    traced = payload(graph(), graph(atom=1))
    traced["counterfactual_candidates"][0]["importance_parts"][0] += 7.75e-7

    result = assert_trace_parity(reference, traced)

    assert result["trace_parity_passed"] is True
    assert result["importance_exact_match"] is False
    assert result["importance_exact_mismatch_count"] == 1
    assert result["importance_max_abs_difference"] <= TRACE_IMPORTANCE_ABS_TOLERANCE
    assert result["importance_threshold_mask_exact"] is True
    assert result["model_cf_id_set_exact"] is True
    assert result["model_cf_order_exact"] is True
    assert result["dbscan_input_id_set_exact"] is True
    assert result["dbscan_input_order_exact"] is True
    assert result["num_diff_gt_1e_7"] == 1
    assert result["num_diff_gt_1e_6"] == 0
    assert (
        result["importance_comparison_policy"]
        == "float32_cuda_replay_abs_tolerance_v1"
    )


def test_trace_parity_rejects_importance_beyond_audited_tolerance() -> None:
    reference = payload(graph(), graph(atom=1))
    traced = payload(graph(), graph(atom=1))
    traced["counterfactual_candidates"][0]["importance_parts"][0] += 1.1e-6

    with pytest.raises(ValueError, match="exceeds the audited CUDA float32"):
        assert_trace_parity(reference, traced)


def test_trace_parity_rejects_candidate_order_change() -> None:
    reference = payload(graph(), graph(atom=1))
    traced = payload(graph(), graph(atom=1))
    traced["counterfactual_candidates"].reverse()

    with pytest.raises(ValueError, match="topology, features, frequency, or order"):
        assert_trace_parity(reference, traced)


def test_trace_parity_rejects_importance_threshold_mask_change() -> None:
    reference = payload(graph(), graph(atom=1))
    traced = payload(graph(), graph(atom=1))
    reference["counterfactual_candidates"][0]["importance_parts"][0] = 0.4999996
    traced["counterfactual_candidates"][0]["importance_parts"][0] = 0.5000004

    with pytest.raises(ValueError, match="importance_threshold_mask_exact"):
        assert_trace_parity(reference, traced)


def test_trace_parity_rejects_model_cf_set_or_order_change() -> None:
    reference = payload(graph(), graph(atom=1))
    traced = payload(graph(), graph(atom=1))
    with pytest.raises(ValueError, match="model_cf_id_set_exact"):
        assert_trace_parity(
            reference,
            traced,
            reference_model_cf_ids=["a", "b"],
            traced_model_cf_ids=["a", "c"],
        )
    with pytest.raises(ValueError, match="model_cf_order_exact"):
        assert_trace_parity(
            reference,
            traced,
            reference_model_cf_ids=["a", "b"],
            traced_model_cf_ids=["b", "a"],
        )


def test_trace_parity_rejects_dbscan_input_set_or_order_change() -> None:
    reference = payload(graph(), graph(atom=1))
    traced = payload(graph(), graph(atom=1))
    with pytest.raises(ValueError, match="dbscan_input_id_set_exact"):
        assert_trace_parity(
            reference,
            traced,
            reference_dbscan_input_ids=["p0:c0", "p1:c0"],
            traced_dbscan_input_ids=["p0:c0", "p2:c0"],
        )
    with pytest.raises(ValueError, match="dbscan_input_order_exact"):
        assert_trace_parity(
            reference,
            traced,
            reference_dbscan_input_ids=["p0:c0", "p1:c0"],
            traced_dbscan_input_ids=["p1:c0", "p0:c0"],
        )


def test_infer_official_single_edit_recovers_node_label_change() -> None:
    assert infer_official_single_edit(graph(), graph(atom=1)) == ["NLC", 0, 1]


def test_recover_lineage_infers_action_missing_from_cached_transition() -> None:
    source = graph()
    target = graph(atom=1)
    candidate_payload = {
        "graph_map": {"source": [source], "target": [target]},
        "counterfactual_candidates": [
            {"graph_hash": "target", "frequency": 2, "importance_parts": [0.8, 1.0]}
        ],
    }
    selected_events = [
        {
            "move_index": 3,
            "head_index": 1,
            "event": "selected_transition",
            "source_official_hash": "source",
            "target_official_hash": "target",
            "source_graph_sha256": stable_graph_sha256(source),
            "target_graph_sha256": stable_graph_sha256(target),
            "action_resolution": "missing",
            "action": None,
            "parent_id": "parent-1",
        }
    ]

    lineage = recover_candidate_lineage_from_selected_trace(
        candidate_payload, selected_events
    )

    assert lineage[0]["action_lineage_resolved"] is True
    assert lineage[0]["actions"][0]["action"] == ["NLC", 0, 1]
    assert (
        lineage[0]["actions"][0]["action_recovery"]
        == "inferred_exact_graph_delta_v1"
    )
    assert lineage[0]["actions"][0]["source_node_ids"] == [
        "source:0",
        "source:1",
    ]
    assert lineage[0]["actions"][0]["target_node_ids"] == [
        "source:0",
        "source:1",
    ]


def test_recorded_action_bypasses_ambiguous_graph_delta_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Graph(
        x=[[1.0, 0.0], [1.0, 0.0]],
        edge_index=[[], []],
        num_nodes=2,
    )
    target = Graph(
        x=[[1.0, 0.0]],
        edge_index=[[], []],
        num_nodes=1,
        comrecgc_trace_node_ids=("source:1",),
    )
    candidate_payload = {
        "graph_map": {"source": [source], "target": [target]},
        "counterfactual_candidates": [{"graph_hash": "target"}],
    }
    selected_events = [
        {
            "move_index": 7,
            "head_index": 2,
            "event": "selected_transition",
            "source_official_hash": "source",
            "target_official_hash": "target",
            "source_graph_sha256": stable_graph_sha256(source),
            "target_graph_sha256": stable_graph_sha256(target),
            "action_resolution": "exact",
            "action": ["INR", 0, 0],
            "parent_id": "parent-1",
        }
    ]

    def fail_if_inferred(*_args: object, **_kwargs: object) -> list[object]:
        pytest.fail("recorded lineage must not invoke graph-delta inference")

    monkeypatch.setattr(
        graph_trace_module, "infer_official_single_edit", fail_if_inferred
    )
    audit: dict[str, object] = {}
    lineage = recover_candidate_lineage_from_selected_trace(
        candidate_payload,
        selected_events,
        source_graphs_by_parent_id={"parent-1": source},
        recovery_audit=audit,
    )

    assert lineage[0]["action_lineage_resolved"] is True
    assert lineage[0]["actions"][0]["action"] == ["INR", 0, 0]
    assert lineage[0]["actions"][0]["lineage_source"] == (
        "recorded_action_replay_v1"
    )
    assert audit["recorded_action_selected_count"] == 1
    assert audit["recorded_action_present_count"] == 1
    assert audit["recorded_action_replay_ok_count"] == 1
    assert audit["recorded_action_replay_mismatch_count"] == 0
    assert audit["recorded_action_replay_verified_count"] == 1
    assert audit["legacy_inference_called_count"] == 0
    assert audit["legacy_inference_invocation_count"] == 0


def test_recorded_action_exact_replay_is_verified() -> None:
    source = graph()
    target = graph(atom=1)
    candidate_payload = {
        "graph_map": {"source": [source], "target": [target]},
        "counterfactual_candidates": [{"graph_hash": "target"}],
    }
    selected_events = [
        {
            "move_index": 8,
            "head_index": 0,
            "event": "selected_transition",
            "source_official_hash": "source",
            "target_official_hash": "target",
            "source_graph_sha256": stable_graph_sha256(source),
            "target_graph_sha256": stable_graph_sha256(target),
            "action_resolution": "exact",
            "action": ["NLC", 0, 1],
            "parent_id": "parent-1",
        }
    ]
    audit: dict[str, object] = {}

    lineage = recover_candidate_lineage_from_selected_trace(
        candidate_payload,
        selected_events,
        source_graphs_by_parent_id={"parent-1": source},
        recovery_audit=audit,
    )

    action = lineage[0]["actions"][0]
    assert action["action_recovery"] == "recorded_exact"
    assert action["action_replay_exact"] is True
    assert audit["recorded_action_replay_verified_count"] == 1
    assert audit["recorded_action_replay_failed_count"] == 0


def test_recorded_action_replay_mismatch_fails_closed_without_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = graph()
    target = graph(atom=1)
    candidate_payload = {
        "graph_map": {"source": [source], "target": [target]},
        "counterfactual_candidates": [{"graph_hash": "target"}],
    }
    selected_events = [
        {
            "move_index": 9,
            "head_index": 0,
            "event": "selected_transition",
            "source_official_hash": "source",
            "target_official_hash": "target",
            "source_graph_sha256": stable_graph_sha256(source),
            "target_graph_sha256": stable_graph_sha256(target),
            "action_resolution": "exact",
            "action": ["NOTHING", 0, 0],
            "parent_id": "parent-1",
        }
    ]

    def fail_if_inferred(*_args: object, **_kwargs: object) -> list[object]:
        pytest.fail("mismatched recorded action must fail closed without inference")

    monkeypatch.setattr(
        graph_trace_module, "infer_official_single_edit", fail_if_inferred
    )
    audit: dict[str, object] = {}
    with pytest.raises(
        ValueError, match="Recorded selected action does not replay"
    ) as error:
        recover_candidate_lineage_from_selected_trace(
            candidate_payload,
            selected_events,
            recovery_audit=audit,
        )

    assert '"official_graph_diff_diagnostic"' in str(error.value)
    assert '"candidates": [["NLC", 0, 1]]' in str(error.value)
    assert audit["recorded_action_selected_count"] == 1
    assert audit["recorded_action_replay_failed_count"] == 1
    assert audit["recorded_action_replay_mismatch_count"] == 1
    assert audit["legacy_inference_invocation_count"] == 0


def test_missing_recorded_action_invokes_legacy_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = graph()
    target = graph(atom=1)
    candidate_payload = {
        "graph_map": {"source": [source], "target": [target]},
        "counterfactual_candidates": [{"graph_hash": "target"}],
    }
    selected_events = [
        {
            "move_index": 10,
            "head_index": 0,
            "event": "selected_transition",
            "source_official_hash": "source",
            "target_official_hash": "target",
            "source_graph_sha256": stable_graph_sha256(source),
            "target_graph_sha256": stable_graph_sha256(target),
            "action_resolution": "missing",
            "action": None,
            "parent_id": "parent-1",
        }
    ]
    calls = 0

    def tracked_inference(source_graph: object, target_graph: object) -> list[object]:
        nonlocal calls
        calls += 1
        return infer_official_single_edit(source_graph, target_graph)

    monkeypatch.setattr(
        graph_trace_module, "infer_official_single_edit", tracked_inference
    )
    audit: dict[str, object] = {}
    lineage = recover_candidate_lineage_from_selected_trace(
        candidate_payload,
        selected_events,
        recovery_audit=audit,
    )

    assert calls == 1
    assert lineage[0]["actions"][0]["lineage_source"] == (
        "legacy_graph_diff_inference_v1"
    )
    assert audit["missing_action_fallback_count"] == 1
    assert audit["legacy_missing_action_count"] == 1
    assert audit["legacy_inference_called_count"] == 1
    assert audit["legacy_inference_invocation_count"] == 1
    assert audit["legacy_inference_success_count"] == 1


def test_missing_recorded_action_ambiguous_legacy_inference_fails_closed() -> None:
    source = Graph(
        x=[[1.0, 0.0], [1.0, 0.0]],
        edge_index=[[], []],
        num_nodes=2,
    )
    target = Graph(
        x=[[1.0, 0.0]],
        edge_index=[[], []],
        num_nodes=1,
        comrecgc_trace_node_ids=("source:1",),
    )
    candidate_payload = {
        "graph_map": {"source": [source], "target": [target]},
        "counterfactual_candidates": [{"graph_hash": "target"}],
    }
    selected_events = [
        {
            "move_index": 11,
            "head_index": 0,
            "event": "selected_transition",
            "source_official_hash": "source",
            "target_official_hash": "target",
            "source_graph_sha256": stable_graph_sha256(source),
            "target_graph_sha256": stable_graph_sha256(target),
            "action_resolution": "missing",
            "action": None,
            "parent_id": "parent-1",
        }
    ]
    audit: dict[str, object] = {}

    with pytest.raises(ValueError, match="not one unique pinned-upstream"):
        recover_candidate_lineage_from_selected_trace(
            candidate_payload,
            selected_events,
            recovery_audit=audit,
        )

    assert audit["missing_action_fallback_count"] == 1
    assert audit["legacy_inference_invocation_count"] == 1
    assert audit["legacy_inference_failure_count"] == 1
    assert audit["legacy_inference_ambiguous_count"] == 1


def test_recorded_action_json_round_trip_replays_exact_target(tmp_path) -> None:
    source = graph()
    target = graph(atom=1)
    action_path = tmp_path / "selected_action.json"
    action_path.write_text(
        json.dumps({"action": ["NLC", 0, 1]}, sort_keys=True),
        encoding="utf-8",
    )

    loaded = json.loads(action_path.read_text(encoding="utf-8"))["action"]

    assert apply_action_to_normalized_payload(source, loaded) == (
        normalized_untyped_graph_payload(target)
    )


def test_recover_lineage_accepts_exact_frozen_zero_action_source_root() -> None:
    source = graph()
    candidate_payload = {
        "graph_map": {"source": [source]},
        "counterfactual_candidates": [{"graph_hash": "source"}],
    }

    lineage = recover_candidate_lineage_from_selected_trace(
        candidate_payload,
        [],
        source_graphs_by_parent_id={"parent-1": source},
    )

    assert lineage[0]["action_lineage_resolved"] is True
    assert lineage[0]["zero_action_source_root"] is True
    assert lineage[0]["lineage_root_status"] == "frozen_source_graph_exact_zero_action"
    assert lineage[0]["actions"] == []


def test_recover_lineage_rejects_unverified_zero_action_candidate() -> None:
    candidate = graph()
    different_source = graph(atom=1)
    candidate_payload = {
        "graph_map": {"candidate": [candidate]},
        "counterfactual_candidates": [{"graph_hash": "candidate"}],
    }

    lineage = recover_candidate_lineage_from_selected_trace(
        candidate_payload,
        [],
        source_graphs_by_parent_id={"parent-1": different_source},
    )

    assert lineage[0]["action_lineage_resolved"] is False
    assert lineage[0]["zero_action_source_root"] is False
    assert lineage[0]["lineage_root_status"] == "unresolved"


def test_recover_lineage_deduplicates_identical_target_events() -> None:
    source = graph()
    target = graph(atom=1)
    candidate_payload = {
        "graph_map": {"source": [source], "target": [target]},
        "counterfactual_candidates": [{"graph_hash": "target"}],
    }
    base_event = {
        "move_index": 3,
        "head_index": 1,
        "event": "selected_transition",
        "source_official_hash": "source",
        "target_official_hash": "target",
        "source_graph_sha256": stable_graph_sha256(source),
        "target_graph_sha256": stable_graph_sha256(target),
        "parent_id": "parent-1",
    }
    selected_events = [
        {**base_event, "action_resolution": "exact", "action": ["NLC", 0, 1]},
        {**base_event, "action_resolution": "missing", "action": None},
    ]

    audit: dict[str, object] = {}
    lineage = recover_candidate_lineage_from_selected_trace(
        candidate_payload,
        selected_events,
        source_graphs_by_parent_id={"parent-1": source},
        recovery_audit=audit,
    )

    assert lineage[0]["action_lineage_resolved"] is True
    assert len(lineage[0]["actions"]) == 1
    assert lineage[0]["actions"][0]["action_recovery"] == "recorded_exact"
    assert audit["predecessor_target_count"] == 1
    assert audit["predecessor_duplicate_event_count"] == 1
    assert audit["predecessor_duplicate_exact_transition_count"] == 1
    assert audit["predecessor_unverified_conflict_count"] == 0


def test_recover_lineage_keeps_first_recorded_event_for_source_hash_alias() -> None:
    source = graph()
    source_alias = graph()
    target = graph(atom=1)
    candidate_payload = {
        "graph_map": {
            "source-first": [source],
            "source-alias": [source_alias],
            "target": [target],
        },
        "counterfactual_candidates": [{"graph_hash": "target"}],
    }
    selected_events = [
        {
            "move_index": 273,
            "head_index": 0,
            "event": "selected_transition",
            "source_official_hash": "source-first",
            "target_official_hash": "target",
            "source_graph_sha256": stable_untyped_graph_sha256(source),
            "target_graph_sha256": stable_untyped_graph_sha256(target),
            "action_resolution": "exact",
            "action": ["NLC", 0, 1],
            "parent_id": "parent-1",
        },
        {
            "move_index": 529,
            "head_index": 3,
            "event": "selected_transition",
            "source_official_hash": "source-alias",
            "target_official_hash": "target",
            "source_graph_sha256": stable_untyped_graph_sha256(source_alias),
            "target_graph_sha256": stable_untyped_graph_sha256(target),
            "action_resolution": "exact",
            "action": ["NLC", 0, 1],
            "parent_id": "parent-1",
        },
    ]
    audit: dict[str, object] = {}

    lineage = recover_candidate_lineage_from_selected_trace(
        candidate_payload,
        selected_events,
        source_graphs_by_parent_id={"parent-1": source},
        recovery_audit=audit,
    )

    action = lineage[0]["actions"][0]
    assert lineage[0]["action_lineage_resolved"] is True
    assert action["source_official_hash"] == "source-first"
    assert action["move_index"] == 273
    assert action["selected_trace_row_index"] == 0
    assert action["selected_transition_index"] == 0
    assert audit["predecessor_target_count"] == 1
    assert audit["predecessor_duplicate_event_count"] == 1
    assert audit["predecessor_duplicate_exact_transition_count"] == 0
    assert audit["predecessor_duplicate_content_equivalent_count"] == 1
    assert audit["predecessor_source_official_alias_count"] == 1
    assert audit["predecessor_conflicting_exact_event_count"] == 0
    assert audit["predecessor_cross_parent_convergence_count"] == 0
    assert audit["predecessor_unverified_conflict_count"] == 0
    assert audit["predecessor_index_sha256"]


def test_recover_lineage_mirrors_global_first_recorded_cross_parent_target() -> None:
    source_first = graph()
    source_later = Graph(
        x=[[0.0, 0.0], [0.0, 1.0]],
        edge_index=[[0, 1], [1, 0]],
        num_nodes=2,
        comrecgc_parent_id="parent-2",
        comrecgc_trace_node_ids=("parent-2:0", "parent-2:1"),
    )
    target = graph(atom=1)
    candidate_payload = {
        "graph_map": {
            "source-first": [source_first],
            "source-later": [source_later],
            "shared-target": [target],
        },
        "counterfactual_candidates": [{"graph_hash": "shared-target"}],
    }
    selected_events = [
        {
            "move_index": 11,
            "head_index": 0,
            "event": "selected_transition",
            "source_official_hash": "source-first",
            "target_official_hash": "shared-target",
            "source_graph_sha256": stable_untyped_graph_sha256(source_first),
            "target_graph_sha256": stable_untyped_graph_sha256(target),
            "action_resolution": "exact",
            "action": ["NLC", 0, 1],
            "parent_id": "parent-1",
        },
        {
            "move_index": 29,
            "head_index": 2,
            "event": "selected_transition",
            "source_official_hash": "source-later",
            "target_official_hash": "shared-target",
            "source_graph_sha256": stable_untyped_graph_sha256(source_later),
            "target_graph_sha256": stable_untyped_graph_sha256(target),
            "action_resolution": "exact",
            "action": ["NLC", 0, 1],
            "parent_id": "parent-2",
        },
    ]
    audit: dict[str, object] = {}

    lineage = recover_candidate_lineage_from_selected_trace(
        candidate_payload,
        selected_events,
        source_graphs_by_parent_id={"parent-1": source_first},
        recovery_audit=audit,
    )

    action = lineage[0]["actions"][0]
    assert lineage[0]["action_lineage_resolved"] is True
    assert action["parent_id"] == "parent-1"
    assert action["source_official_hash"] == "source-first"
    assert action["move_index"] == 11
    assert action["selected_trace_row_index"] == 0
    assert action["selected_transition_index"] == 0
    assert audit["recorded_action_replay_verified_count"] == 2
    assert audit["predecessor_target_count"] == 1
    assert audit["predecessor_duplicate_event_count"] == 1
    assert audit["predecessor_conflicting_exact_event_count"] == 1
    assert audit["predecessor_cross_parent_convergence_count"] == 1
    assert audit["predecessor_unverified_conflict_count"] == 0
    assert audit["selected_event_source_parent_mismatch_count"] == 0
    assert audit["selected_event_target_parent_mismatch_count"] == 1


def test_recover_lineage_global_lookup_rejects_official_hash_collision() -> None:
    source = graph()
    target = graph(atom=1)
    candidate_payload = {
        "graph_map": {"source": [source], "target": [target]},
        "frozen_graph_closure": {"target": source},
        "counterfactual_candidates": [{"graph_hash": "target"}],
    }

    with pytest.raises(
        RuntimeError, match="official_hash_collision_between_payload_sections"
    ):
        recover_candidate_lineage_from_selected_trace(
            candidate_payload,
            [],
            source_graphs_by_parent_id={"parent-1": source},
        )


def test_recover_lineage_rejects_conflicting_predecessors_for_one_target() -> None:
    source_a = graph()
    source_b = Graph(
        x=[[0.0, 1.0], [1.0, 0.0]],
        edge_index=[[0, 1], [1, 0]],
        num_nodes=2,
    )
    target = graph(atom=1)
    candidate_payload = {
        "graph_map": {
            "source-a": [source_a],
            "source-b": [source_b],
            "target": [target],
        },
        "counterfactual_candidates": [{"graph_hash": "target"}],
    }
    selected_events = []
    for source_hash, source in (("source-a", source_a), ("source-b", source_b)):
        selected_events.append(
            {
                "move_index": 3,
                "head_index": 1,
                "event": "selected_transition",
                "source_official_hash": source_hash,
                "target_official_hash": "target",
                "source_graph_sha256": stable_graph_sha256(source),
                "target_graph_sha256": stable_graph_sha256(target),
                "action_resolution": "missing",
                "action": None,
                "parent_id": "parent-1",
            }
        )

    audit: dict[str, object] = {}
    with pytest.raises(ValueError, match="Ambiguous legacy COMRECGC predecessor"):
        recover_candidate_lineage_from_selected_trace(
            candidate_payload,
            selected_events,
            source_graphs_by_parent_id={"parent-1": source_a},
            recovery_audit=audit,
        )

    assert audit["predecessor_conflicting_exact_event_count"] == 1
    assert audit["predecessor_unverified_conflict_count"] == 1
    assert audit["predecessor_unresolved_legacy_conflict_count"] == 1

    resolved_audit: dict[str, object] = {}
    lineage = recover_candidate_lineage_from_selected_trace(
        candidate_payload,
        [
            *selected_events,
            {
                **selected_events[0],
                "move_index": 4,
                "action_resolution": "exact",
                "action": ["NLC", 0, 1],
            },
        ],
        source_graphs_by_parent_id={"parent-1": source_a},
        recovery_audit=resolved_audit,
    )

    assert lineage[0]["action_lineage_resolved"] is True
    assert lineage[0]["actions"][0]["selected_transition_index"] == 2
    assert lineage[0]["actions"][0]["action_recovery"] == "recorded_exact"
    assert resolved_audit["predecessor_recorded_upgrade_count"] == 1
    assert resolved_audit["predecessor_unverified_conflict_count"] == 1
    assert resolved_audit["predecessor_unresolved_legacy_conflict_count"] == 0


def test_selected_trace_streams_to_reloadable_bounded_chunks(tmp_path) -> None:
    recorder = ActionTraceRecorder(output_dir=tmp_path, chunk_size=1)
    source, target, module = _record_one_transition(recorder)

    def move(*_args: object, **_kwargs: object) -> tuple:
        return (["target"], False, None, None, None)

    recorder.wrap_move(move, module)(
        graphs_hash=["source"],
        start_graphs_hash=["source"],
        importance_args={},
        teleport_probability=0.1,
    )
    payload_value = {
        "graph_map": {"source": [source], "target": [target]},
        "counterfactual_candidates": [{"graph_hash": "target"}],
    }
    summary = recorder.write(tmp_path, payload_value)
    rows = load_selected_trace(summary["selected_trace_path"])
    assert len(rows) == 1
    assert rows[0]["action"] == ["NLC", 0, 1]
    assert summary["selected_trace_chunk_count"] == 1
    assert recorder._pending_events == []
    assert not hasattr(recorder, "selected")


def test_streamed_trace_write_resolves_frozen_zero_action_root(tmp_path) -> None:
    source = graph()
    recorder = ActionTraceRecorder(output_dir=tmp_path, chunk_size=1)
    payload_value = {
        "graph_map": {"source": [source]},
        "counterfactual_candidates": [{"graph_hash": "source"}],
    }

    summary = recorder.write(
        tmp_path,
        payload_value,
        source_graphs_by_parent_id={"parent-1": source},
    )
    lineage = __import__("json").loads(
        (tmp_path / "candidate_action_lineage.json").read_text(encoding="utf-8")
    )

    assert summary["candidate_lineage_resolved_count"] == 1
    assert summary["lineage_recovery_policy"] == (
        "pinned_upstream_official_hash_source_root_v2"
    )
    assert lineage[0]["zero_action_source_root"] is True


def test_trace_resume_reuses_identical_completed_chunks_without_duplicates(tmp_path) -> None:
    payload_value = None
    for _run in range(2):
        recorder = ActionTraceRecorder(output_dir=tmp_path, chunk_size=1)
        source, target, module = _record_one_transition(recorder)

        def move(*_args: object, **_kwargs: object) -> tuple:
            return (["target"], False, None, None, None)

        recorder.wrap_move(move, module)(
            graphs_hash=["source"],
            start_graphs_hash=["source"],
            importance_args={},
            teleport_probability=0.1,
        )
        payload_value = {
            "graph_map": {"source": [source], "target": [target]},
            "counterfactual_candidates": [{"graph_hash": "target"}],
        }
        summary = recorder.write(tmp_path, payload_value)

    chunks = list((tmp_path / "selected_action_trace_chunks").glob("part-*.jsonl"))
    assert len(chunks) == 1
    assert len(load_selected_trace(summary["selected_trace_path"])) == 1
    manifest = __import__("json").loads(
        (tmp_path / "selected_action_trace_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["chunks"][0]["materialization"] == "adopt_existing_identical"


def test_trace_checkpoint_round_trip_preserves_completed_and_pending_rows(tmp_path) -> None:
    recorder = ActionTraceRecorder(output_dir=tmp_path, chunk_size=2)
    recorder._stream_event({"event": "teleport", "move_index": 0})
    recorder._stream_event({"event": "teleport", "move_index": 1})
    recorder._stream_event({"event": "teleport", "move_index": 2})
    recorder.move_index = 3
    state = recorder.export_checkpoint_state()
    assert len(state["chunks"]) == 1
    assert len(state["pending_events"]) == 1

    restored = ActionTraceRecorder(output_dir=tmp_path, chunk_size=2)
    restored.restore_checkpoint_state(state)
    assert restored.export_checkpoint_state() == state
    restored._stream_event({"event": "teleport", "move_index": 3})
    assert len(restored.export_checkpoint_state()["chunks"]) == 2


def test_full_trace_writes_compact_reloadable_lineage_without_inline_actions(tmp_path) -> None:
    recorder = ActionTraceRecorder(output_dir=tmp_path, chunk_size=1)
    source, target, module = _record_one_transition(recorder)

    def move(*_args: object, **_kwargs: object) -> tuple:
        return (["target"], False, None, None, None)

    recorder.wrap_move(move, module)(
        graphs_hash=["source"],
        start_graphs_hash=["source"],
        importance_args={},
        teleport_probability=0.1,
    )
    payload_value = {
        "graph_map": {"source": [source], "target": [target]},
        "counterfactual_candidates": [{"graph_hash": "target"}],
    }
    summary = recorder.write(
        tmp_path,
        payload_value,
        source_graphs_by_parent_id={"parent-1": source},
        compact_candidate_lineage=True,
    )
    contract = __import__("json").loads(
        (tmp_path / "candidate_action_lineage.json").read_text(encoding="utf-8")
    )

    assert contract["format"] == "selected_trace_predecessor_index"
    assert contract["candidate_actions_inlined"] is False
    assert summary["candidate_lineage_format"] == "selected_trace_predecessor_index"
    assert summary["max_materialized_candidate_lineages"] == 1
    assert summary["lineage_recovery_audit"][
        "recorded_action_replay_verified_count"
    ] == 1
    assert summary["lineage_recovery_audit"][
        "legacy_inference_invocation_count"
    ] == 0
    index_row = __import__("json").loads(
        (tmp_path / contract["candidate_index_path"])
        .read_text(encoding="utf-8")
        .strip()
    )
    assert index_row["actions"] == []
    recovered = list(
        iter_candidate_lineage_from_selected_trace(
            payload_value,
            iter_selected_trace(tmp_path / contract["selected_trace_manifest_path"]),
            source_graphs_by_parent_id={"parent-1": source},
        )
    )
    assert recovered[0]["actions"][0]["action"] == ["NLC", 0, 1]


def test_compact_trace_resume_reuses_index_without_duplicate_rows(tmp_path) -> None:
    for _run in range(2):
        recorder = ActionTraceRecorder(output_dir=tmp_path, chunk_size=1)
        source, target, module = _record_one_transition(recorder)

        def move(*_args: object, **_kwargs: object) -> tuple:
            return (["target"], False, None, None, None)

        recorder.wrap_move(move, module)(
            graphs_hash=["source"],
            start_graphs_hash=["source"],
            importance_args={},
            teleport_probability=0.1,
        )
        recorder.write(
            tmp_path,
            {
                "graph_map": {"source": [source], "target": [target]},
                "counterfactual_candidates": [{"graph_hash": "target"}],
            },
            source_graphs_by_parent_id={"parent-1": source},
            compact_candidate_lineage=True,
        )

    rows = (
        tmp_path / "candidate_action_lineage_index.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
