from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.baselines.comrecgc.graph_trace import stable_untyped_graph_sha256
from src.baselines.comrecgc.live_graph_state import (
    ComRecGCGraphHashCollisionError,
    ComRecGCLiveGraphResolutionError,
    LiveGraphState,
)


def graph(label: int) -> SimpleNamespace:
    return SimpleNamespace(
        x=np.asarray([[label], [label + 1]], dtype=np.int64),
        edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        num_nodes=2,
    )


def module() -> SimpleNamespace:
    return SimpleNamespace(
        graph_map={}, graph_index_map={}, counterfactual_candidates=[], transitions={}
    )


def test_evicted_live_graph_is_resolved_without_reactivating_membership(tmp_path) -> None:
    owner = module()
    state = LiveGraphState(owner, {}, store_path=tmp_path / "graphs.sqlite3", seed=13)
    owner.graph_map = state.graph_map
    owner.comrecgc_live_graph_state = state
    try:
        value = [graph(1), np.asarray([1.0]), np.asarray([2.0])]
        owner.graph_map[-7] = value
        expected = stable_untyped_graph_sha256(value[0])
        del owner.graph_map[-7]

        assert -7 not in owner.graph_map
        assert state.contains(-7)
        assert stable_untyped_graph_sha256(state.resolve_graph(-7)) == expected
        assert -7 not in owner.graph_map
        assert state.graph_map.rehydrations == 1
    finally:
        state.close()


def test_unresolved_lookup_fails_closed_with_diagnostics(tmp_path) -> None:
    owner = module()
    state = LiveGraphState(owner, {}, store_path=tmp_path / "graphs.sqlite3", seed=0)
    owner.graph_map = state.graph_map
    try:
        state.graph_map.begin_move([123], current_step=46690)
        with pytest.raises(ComRecGCLiveGraphResolutionError) as captured:
            state.resolve_graph(123)
        assert captured.value.diagnostics["current_step"] == 46690
        assert captured.value.diagnostics["graph_hash"] == "123"
        assert state.graph_map.unresolved_lookups == 1
    finally:
        state.close()


def test_same_official_hash_with_different_graph_fails_collision_gate(tmp_path) -> None:
    owner = module()
    state = LiveGraphState(owner, {}, store_path=tmp_path / "graphs.sqlite3", seed=0)
    owner.graph_map = state.graph_map
    try:
        owner.graph_map[9] = [graph(1), np.asarray([1.0]), np.asarray([2.0])]
        del owner.graph_map[9]
        with pytest.raises(ComRecGCGraphHashCollisionError):
            owner.graph_map[9] = [graph(8), np.asarray([1.0]), np.asarray([2.0])]
    finally:
        state.close()


def test_live_graph_checkpoint_restores_boundary_counters(tmp_path) -> None:
    owner = module()
    database = tmp_path / "graphs.sqlite3"
    state = LiveGraphState(owner, {}, store_path=database, seed=4)
    owner.graph_map = state.graph_map
    try:
        owner.graph_map[3] = [graph(1), np.asarray([1.0]), np.asarray([2.0])]
        state.move_count = 7
        state.graph_map.current_step = 7
        del owner.graph_map[3]
        exported = state.export_checkpoint_state()
    finally:
        state.close()

    restored_owner = module()
    restored = LiveGraphState(
        restored_owner, {}, store_path=database, seed=4
    )
    restored_owner.graph_map = restored.graph_map
    try:
        restored.restore_checkpoint_state(exported)
        assert restored.move_count == 7
        assert restored.graph_map.eviction_committed == 1
        assert restored.contains(3)
    finally:
        restored.close()


def test_live_graph_checkpoint_rejects_active_pin(tmp_path) -> None:
    owner = module()
    state = LiveGraphState(owner, {}, store_path=tmp_path / "graphs.sqlite3", seed=0)
    owner.graph_map = state.graph_map
    try:
        state.graph_map.pin_counts[9] = 1
        with pytest.raises(RuntimeError, match="inside a move"):
            state.export_checkpoint_state()
    finally:
        state.close()
