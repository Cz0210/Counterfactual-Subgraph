from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.baselines.comrecgc.frozen_payload import (
    ComRecGCFrozenPayloadClosureError,
    materialize_frozen_payload_closure,
    payload_graphs_by_official_hash,
)
from src.baselines.comrecgc.graph_trace import stable_untyped_graph_sha256
from src.baselines.comrecgc.live_graph_state import AuthoritativeGraphStore


def graph(values: list[int], edges: list[tuple[int, int]]) -> SimpleNamespace:
    return SimpleNamespace(
        x=np.asarray([[value] for value in values], dtype=np.int64),
        edge_index=np.asarray(edges, dtype=np.int64).T
        if edges
        else np.empty((2, 0), dtype=np.int64),
        num_nodes=len(values),
        comrecgc_parent_id="parent",
    )


def entry(value: SimpleNamespace) -> list[object]:
    return [value, np.asarray([0.5, 1.0]), np.asarray([1.0, 2.0])]


def event(source: SimpleNamespace, target: SimpleNamespace) -> dict[str, object]:
    return {
        "event": "selected_transition",
        "parent_id": "parent",
        "source_official_hash": "source",
        "target_official_hash": "target",
        "source_graph_sha256": stable_untyped_graph_sha256(source),
        "target_graph_sha256": stable_untyped_graph_sha256(target),
    }


def test_selected_trace_graph_is_rehydrated_into_frozen_payload(tmp_path) -> None:
    source = graph([1, 2], [(0, 1), (1, 0)])
    target = graph([2], [])
    store_path = tmp_path / "graphs.sqlite3"
    store = AuthoritativeGraphStore(store_path)
    store.put("source", entry(source))
    store.close()
    candidates = [{"graph_hash": "target", "frequency": 1}]
    payload = {
        "graph_map": {"target": entry(target)},
        "counterfactual_candidates": candidates,
        "traversed_hashes": ["source", "target"],
    }

    frozen, audit = materialize_frozen_payload_closure(
        payload, [event(source, target)], backing_store_path=store_path
    )

    assert audit["closure_complete"] is True
    assert audit["resolved_from_backing_store"] == 1
    assert frozen["counterfactual_candidates"] is candidates
    assert list(frozen["graph_map"]) == ["target", "source"]
    assert set(payload_graphs_by_official_hash(frozen)) == {"source", "target"}


def test_frozen_payload_fails_closed_when_required_graph_is_absent(tmp_path) -> None:
    source = graph([1, 2], [(0, 1), (1, 0)])
    target = graph([2], [])
    store_path = tmp_path / "graphs.sqlite3"
    AuthoritativeGraphStore(store_path).close()
    payload = {
        "graph_map": {"target": entry(target)},
        "counterfactual_candidates": [{"graph_hash": "target"}],
    }

    with pytest.raises(ComRecGCFrozenPayloadClosureError, match="unresolved_hash_count"):
        materialize_frozen_payload_closure(
            payload, [event(source, target)], backing_store_path=store_path
        )


def test_transition_destination_is_retained_in_graph_only_closure(tmp_path) -> None:
    source = graph([1, 2], [(0, 1), (1, 0)])
    target = graph([2], [])
    payload = {
        "graph_map": {"source": entry(source)},
        "counterfactual_candidates": [{"graph_hash": "source"}],
        "transitions": {
            "source": (["target"], [target], [[0.5, 1.0]], [[1.0, 2.0]])
        },
    }

    frozen, audit = materialize_frozen_payload_closure(
        payload, (), backing_store_path=None
    )

    assert audit["transition_hash_count"] == 2
    assert audit["resolved_from_inline_transition"] == 1
    assert stable_untyped_graph_sha256(frozen["frozen_graph_closure"]["target"]) == (
        stable_untyped_graph_sha256(target)
    )
