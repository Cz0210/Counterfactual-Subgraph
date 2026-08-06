from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from src.baselines.comrecgc.graph_trace import (
    ActionTraceRecorder,
    TRACE_IMPORTANCE_ABS_TOLERANCE,
    assert_trace_parity,
    infer_official_single_edit,
    load_selected_trace,
    recover_candidate_lineage_from_selected_trace,
    stable_graph_sha256,
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
