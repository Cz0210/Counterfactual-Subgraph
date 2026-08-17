from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.baselines.comrecgc.frozen_payload import (
    build_frozen_payload_closure,
    payload_graphs_by_official_hash,
)
from src.baselines.comrecgc.graph_trace import stable_untyped_graph_sha256


def _graph(values: list[int], parent_id: str = "parent") -> SimpleNamespace:
    return SimpleNamespace(
        x=np.asarray([[value] for value in values], dtype=np.int64),
        edge_index=(
            np.asarray([[0, 1], [1, 0]], dtype=np.int64)
            if len(values) > 1
            else np.empty((2, 0), dtype=np.int64)
        ),
        num_nodes=len(values),
        comrecgc_parent_id=parent_id,
    )


def _entry(graph: SimpleNamespace) -> list[object]:
    return [graph, np.asarray([0.5]), np.asarray([1.0])]


def _event(source: SimpleNamespace, target: SimpleNamespace) -> dict[str, object]:
    return {
        "event": "selected_transition",
        "parent_id": "parent",
        "source_official_hash": "evicted-source-alias",
        "target_official_hash": "evicted-target-alias",
        "source_graph_sha256": stable_untyped_graph_sha256(source),
        "target_graph_sha256": stable_untyped_graph_sha256(target),
    }


def test_alias_chain_materialization() -> None:
    source = _graph([1, 2])
    payload = {
        "graph_map": {"canonical-source": _entry(source)},
        "counterfactual_candidates": [{"graph_hash": "canonical-source"}],
        "alias_to_canonical": {
            "old-source": "middle-source",
            "middle-source": "canonical-source",
        },
    }

    frozen, audit = build_frozen_payload_closure(
        payload, (), backing_store_path=None
    )
    resolved = payload_graphs_by_official_hash(frozen)

    assert audit["closure_complete"] is True
    assert frozen["alias_to_canonical"]["old-source"] == "canonical-source"
    assert stable_untyped_graph_sha256(resolved["old-source"]) == (
        stable_untyped_graph_sha256(source)
    )


def test_selected_trace_payload_roundtrip_recovers_changed_official_hashes() -> None:
    source = _graph([1, 2])
    target = _graph([2])
    event = _event(source, target)
    payload = {
        "graph_map": {
            "canonical-source": _entry(source),
            "canonical-target": _entry(target),
        },
        "counterfactual_candidates": [{"graph_hash": "canonical-target"}],
    }

    frozen, first = build_frozen_payload_closure(
        payload, [event], backing_store_path=None
    )
    reloaded, second = build_frozen_payload_closure(
        frozen, [event], backing_store_path=None
    )
    resolved = payload_graphs_by_official_hash(reloaded)

    assert first["resolved_by_fingerprint_alias"] == 2
    assert second["unresolved_hash_count"] == 0
    assert second["closure_complete"] is True
    assert set(reloaded["original_trace_hashes"]) == {
        "evicted-source-alias",
        "evicted-target-alias",
    }
    assert stable_untyped_graph_sha256(resolved["evicted-target-alias"]) == (
        stable_untyped_graph_sha256(target)
    )


def test_freeze_validator_recover_parity_uses_same_pure_builder() -> None:
    source = _graph([1, 2])
    target = _graph([2])
    payload = {
        "graph_map": {"source": _entry(source), "target": _entry(target)},
        "counterfactual_candidates": [{"graph_hash": "target"}],
    }
    event = {
        **_event(source, target),
        "source_official_hash": "source",
        "target_official_hash": "target",
    }

    validated, validation = build_frozen_payload_closure(
        payload, [event], backing_store_path=None
    )
    recovered, recovery = build_frozen_payload_closure(
        validated, [event], backing_store_path=None
    )

    assert validation["requirements_sha256"] == recovery["requirements_sha256"]
    assert validation["required_hash_count"] == recovery["required_hash_count"]
    assert payload_graphs_by_official_hash(validated).keys() == (
        payload_graphs_by_official_hash(recovered).keys()
    )
