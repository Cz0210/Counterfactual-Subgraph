from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.baselines.comrecgc.graph_trace import (
    apply_action_to_normalized_payload,
    enumerate_official_single_edits,
    recover_candidate_lineage_from_selected_trace,
    stable_json_sha256,
    stable_untyped_graph_sha256,
)


FIXTURE = Path(__file__).parent / "fixtures/comrecgc_lineage/aids_move_37600.json"


def _graph(document: dict, *, parent_id: str) -> SimpleNamespace:
    width = int(document["feature_width"])
    rows = []
    for active in document["node_active_indices"]:
        row = [0.0] * width
        row[int(active)] = 1.0
        rows.append(row)
    sources: list[int] = []
    targets: list[int] = []
    for first, second in document["undirected_edges"]:
        sources.extend((int(first), int(second)))
        targets.extend((int(second), int(first)))
    return SimpleNamespace(
        x=rows,
        edge_index=[sources, targets],
        num_nodes=len(rows),
        comrecgc_parent_id=parent_id,
    )


def _with_nlc(graph: SimpleNamespace, node: int, label: int) -> SimpleNamespace:
    result = deepcopy(graph)
    result.x[node] = [0.0] * len(result.x[node])
    result.x[node][label] = 1.0
    return result


def test_aids_historical_failure_payload() -> None:
    """Replay the immutable AIDS move 37600 row, not a toy transition."""

    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    event = document["event"]
    source = _graph(document, parent_id="global-representative-parent")
    target = _with_nlc(source, *document["official_single_edit"][1:])
    target.comrecgc_parent_id = "another-representative-parent"

    assert stable_untyped_graph_sha256(source) == event["source_graph_sha256"]
    assert stable_untyped_graph_sha256(target) == event["target_graph_sha256"]
    assert enumerate_official_single_edits(source, target) == [
        document["official_single_edit"]
    ]
    assert stable_json_sha256(
        apply_action_to_normalized_payload(source, event["action"])
    ) == document["recorded_replay_sha256"]

    frozen_source = deepcopy(source)
    frozen_source.comrecgc_parent_id = event["parent_id"]
    audit: dict[str, object] = {}
    lineage = recover_candidate_lineage_from_selected_trace(
        {
            "dataset": "aids",
            "graph_map": {
                event["source_official_hash"]: [source],
                event["target_official_hash"]: [target],
            },
            "counterfactual_candidates": [
                {"graph_hash": event["target_official_hash"]}
            ],
        },
        [event],
        source_graphs_by_parent_id={event["parent_id"]: frozen_source},
        recovery_audit=audit,
    )

    action = lineage[0]["actions"][0]
    assert lineage[0]["action_lineage_resolved"] is True
    assert lineage[0]["parent_id"] == event["parent_id"]
    assert action["recorded_action"] == ["NLC", 3, 33]
    assert action["action"] == ["NLC", 2, 33]
    assert action["recorded_action_index_remap"] == {
        "operation": "NLC",
        "recorded_node_index": 3,
        "resolved_node_index": 2,
        "label": 33,
    }
    assert action["action_recovery"] == "recorded_exact_global_index_remap_v1"
    assert audit["recorded_action_index_remap_count"] == 1
    assert audit["recorded_action_replay_ok_count"] == 1
    assert audit["recorded_action_replay_mismatch_count"] == 0


def test_aids_recorded_index_remap_still_fails_closed_on_label_mismatch() -> None:
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    event = {**document["event"], "action": ["NLC", 3, 32]}
    source = _graph(document, parent_id=event["parent_id"])
    target = _with_nlc(source, *document["official_single_edit"][1:])

    with pytest.raises(ValueError, match="does not replay to the exact target"):
        recover_candidate_lineage_from_selected_trace(
            {
                "graph_map": {
                    event["source_official_hash"]: [source],
                    event["target_official_hash"]: [target],
                },
                "counterfactual_candidates": [
                    {"graph_hash": event["target_official_hash"]}
                ],
            },
            [event],
        )
