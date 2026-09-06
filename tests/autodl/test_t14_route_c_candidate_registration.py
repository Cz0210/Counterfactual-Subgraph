"""Candidate-first official ordering, using tiny CPU tensor graphs only.

T14_OFFICIAL_SOURCE may point to the immutable 122f9341 comrecgc.py (or its
complete prefix through move_to_next_graph). The optional integration case
executes the unmodified official function AST, with fixed tiny transition
inputs, no model, dataset, GPU, or scientific generation job.
"""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.baselines import tastemolnet_t14_route_c_fresh as route_c


def _graph(atom: int) -> SimpleNamespace:
    # The store's real graph serializer accepts this x/edge_index protocol.
    # No graph-fingerprint function is mocked and PyG need not be installed.
    return SimpleNamespace(
        x=torch.tensor([[atom, 0], [atom, 1]], dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        num_nodes=2,
    )


def _entry(atom: int) -> list:
    return [_graph(atom), np.array([float(atom), 0.0]), np.array([2])]


def _candidate(key: str) -> dict:
    return {
        "frequency": 2,
        "graph_hash": key,
        "importance_parts": (0.75, 0.0),
        "input_graphs_covering_list": None,
    }


def _updater(root: Path, *, resume: bool = False):
    return route_c.RouteCStateUpdater(
        root, candidate_capacity=8, record_capacity=32, lru_capacity=2, resume=resume
    )


def test_candidate_key_reservation_does_not_materialize_or_advance_rng(tmp_path):
    updater = _updater(tmp_path / "state")
    graph_map = route_c.RouteCGraphMap(
        updater.graph_store, {}, next_sequence=updater.next_sequence
    )
    before = random.getstate()
    updater.candidates.append(_candidate("nonlead"))
    assert random.getstate() == before
    assert updater._sequence == 1  # Candidate append only; key reservation has none.
    assert "nonlead" not in graph_map
    assert not graph_map.contains_resolvable("nonlead")
    assert updater.graph_store.graph_id("nonlead") is None
    assert updater.graph_store.data_path.stat().st_size == 0
    reserved = int(updater.candidates.metadata[0]["graph_id"])
    graph_map["nonlead"] = _entry(6)
    assert updater.graph_store.graph_id("nonlead") == reserved
    assert dict(updater.candidates[0]) == _candidate("nonlead")
    updater.close()


def test_unmaterialized_candidate_can_be_replaced_without_fabricated_graph(tmp_path):
    updater = _updater(tmp_path / "state")
    updater.candidates.append(_candidate("evicted-before-registration"))
    updater.candidates[0] = _candidate("replacement")
    assert updater.graph_store.count() == 0
    assert updater.graph_store.checkpoint_state()["reserved_key_count"] == 2
    assert [dict(value) for value in updater.candidates] == [_candidate("replacement")]
    updater.close()


def test_pending_numeric_id_collision_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(route_c, "_stable_numeric_graph_id", lambda _key: 71)
    updater = _updater(tmp_path / "state")
    updater.candidates.append(_candidate("one"))
    with pytest.raises(route_c.T14RouteCFreshError, match="collision"):
        updater.candidates.append(_candidate("other"))
    with pytest.raises(route_c.T14RouteCFreshError, match="collision"):
        updater.graph_store.put("other", _entry(6), sequence_id=2)
    assert len(updater.candidates) == 1
    assert updater.graph_store.count() == 0
    updater.close()


def test_pending_key_checkpoint_reload_preserves_id_and_materialization(tmp_path):
    root = tmp_path / "state"
    updater = _updater(root)
    updater.candidates.append(_candidate("pending"))
    reserved = int(updater.candidates.metadata[0]["graph_id"])
    state = updater.checkpoint_state()
    assert state["graph_store"]["schema_version"].endswith("_v2")
    assert state["graph_store"]["reserved_key_count"] == 1
    updater.close()
    restored = _updater(root, resume=True)
    restored.restore_checkpoint_state(state)
    assert restored.graph_store.graph_id("pending") is None
    assert restored.graph_store.reserve_key("pending") == reserved
    restored.graph_store.put("pending", _entry(7), sequence_id=restored.next_sequence())
    assert restored.graph_store.graph_id("pending") == reserved
    assert dict(restored.candidates[0]) == _candidate("pending")
    restored.close()


def test_key_only_suffix_is_rejected_by_checkpoint_boundary(tmp_path):
    updater = _updater(tmp_path / "state")
    state = updater.checkpoint_state()
    updater.candidates.append(_candidate("suffix"))
    with pytest.raises(route_c.T14RouteCFreshError, match="key boundary"):
        updater.graph_store.validate_checkpoint_state(state["graph_store"])
    updater.close()


OFFICIAL_FUNCTIONS = {
    "is_counterfactual_array_full", "get_minimum_frequency",
    "is_graph_counterfactual", "reorder_counterfactual_candidates",
    "update_input_graphs_covered", "check_reinforcement_condition",
    "populate_counterfactual_candidates", "move_from_known_graph",
    "move_to_next_graph",
}


def _official_module(source: Path, updater=None):
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    definitions = [node for node in parsed.body
                   if isinstance(node, ast.FunctionDef) and node.name in OFFICIAL_FUNCTIONS]
    assert {node.name for node in definitions} == OFFICIAL_FUNCTIONS
    selected = ast.Module(body=definitions, type_ignores=[])
    assert hashlib.sha256(ast.dump(selected, include_attributes=False).encode()).hexdigest() == (
        "59d9523a2e1d74658475b4fbac830f0d19eb56ed2c6bb57c394de08162c51e0e"
    ), "Official 122f9341 functions changed"
    code = compile(selected, str(source), "exec")
    module = SimpleNamespace(
        random=random.Random(7), np=np,
        util=SimpleNamespace(graph_element_counts=lambda graphs: np.array(
            [graph.num_nodes for graph in graphs]
        )),
        MAX_COUNTERFACTUAL_SIZE=8, graph_index_map={"s0": 0, "s1": 1},
        counterfactual_candidates=[_candidate("s0"), _candidate("s1")],
        input_graphs_covered=np.array([], dtype=np.int64), covering_graphs=set(),
        graph_map={"s0": _entry(1), "s1": _entry(2)},
        is_sample=True, sample_size=2,
    )
    targets = [_entry(6), _entry(7)]
    module.transitions = {
        key: (["t0", "t1"], [entry[0] for entry in targets],
              [(0.75, 0.0), (0.6, 0.0)], [entry[1] for entry in targets])
        for key in ("s0", "s1", "t0", "t1")
    }
    if updater is not None:
        module.graph_map = route_c.RouteCGraphMap(
            updater.graph_store, module.graph_map,
            next_sequence=updater.next_sequence, module=module,
        )
        for row in module.counterfactual_candidates:
            updater.candidates.append(row)
        module.counterfactual_candidates = updater.candidates
    exec(code, module.__dict__)
    return module


def _scientific_projection(module):
    from src.baselines.comrecgc.live_graph_state import _entry_graph_sha256

    return {
        "candidate_records": [dict(row) for row in module.counterfactual_candidates],
        "index_order": list(module.graph_index_map.items()),
        "active_graphs": [(key, _entry_graph_sha256(module.graph_map[key]))
                          for key in module.graph_map],
        "covering_graphs": module.covering_graphs.copy(),
        "rng": module.random.getstate(),
    }


def test_actual_official_two_head_move_candidate_first_parity(tmp_path):
    source_value = os.environ.get("T14_OFFICIAL_SOURCE")
    if not source_value:
        pytest.skip("Supply pinned official source for tiny no-model integration")
    source = Path(source_value)
    assert source.is_file()
    updater = _updater(tmp_path / "state")
    plain = _official_module(source)
    lowmemory = _official_module(source, updater)
    observed_candidate_first = []
    original_insert = updater.candidates.insert

    def observe_insert(index, value):
        observed_candidate_first.append(value["graph_hash"] not in lowmemory.graph_map)
        return original_insert(index, value)

    updater.candidates.insert = observe_insert
    heads = ["s0", "s1"]
    for _ in range(4):
        left = plain.move_to_next_graph(heads, ["s0", "s1"], None, 0.0)
        right = lowmemory.move_to_next_graph(heads, ["s0", "s1"], None, 0.0)
        assert left[0] == right[0]
        np.testing.assert_array_equal(left[2], right[2])
        assert left[3:] == right[3:]
        assert _scientific_projection(plain) == _scientific_projection(lowmemory)
        heads = left[0]
    assert any(observed_candidate_first), "Must exercise official non-lead append-before-put"
    assert updater.graph_store.count() == len(lowmemory.graph_map)
    updater.close()
