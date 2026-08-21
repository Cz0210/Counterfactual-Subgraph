from __future__ import annotations

import random
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.baselines.comrecgc.generation_checkpoint import (
    list_generation_checkpoints,
    load_generation_checkpoint,
    restore_rng_state,
    save_generation_checkpoint,
    scientific_command_sha256,
)
from src.baselines.comrecgc.generation_loop import (
    GenerationLoopState,
    restore_official_state,
    run_generation_loop,
    snapshot_official_state,
)


class _FakeOfficial(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(
            graph_map={},
            graph_index_map={},
            counterfactual_candidates=[],
            input_graphs_covered=[0, 0, 0, 0],
            covering_graphs=set(),
            transitions={},
            start={},
            is_sample=True,
            starting_step=1,
            traversed_hashes=[],
            sample_size=17,
            MAX_COUNTERFACTUAL_SIZE=200,
            restart_count=0,
            move_count=0,
        )

    def restart_randomwalk(self, input_graphs, heads, importance_args):
        del input_graphs, importance_args
        self.restart_count += 1
        indices = random.choices(range(4), k=heads)
        base = random.randrange(1, 1_000_000_000)
        hashes = [base + head for head in range(heads)]
        for value in hashes:
            self.graph_map[value] = [f"graph-{value}"]
            self.graph_index_map[value] = len(self.graph_index_map)
        return hashes, indices

    def move_to_next_graph(
        self,
        *,
        graphs_hash,
        start_graphs_hash,
        importance_args,
        teleport_probability,
    ):
        del start_graphs_hash, importance_args
        self.move_count += 1
        if random.uniform(0.0, 1.0) < teleport_probability:
            return None, True, None, None, None
        target = [value + 1_000_000 for value in graphs_hash]
        self.counterfactual_candidates.append(
            {"graph_hash": graphs_hash[0], "frequency": random.randint(1, 9)}
        )
        return target, False, [0], [[1.0]], [0.0]


def _seed() -> None:
    random.seed(913)
    np.random.seed(914)
    torch.manual_seed(915)


def _run(module: _FakeOfficial, *, maximum: int, initial=None):
    return run_generation_loop(
        module,
        input_graphs=list(range(4)),
        importance_args={},
        teleport_probability=0.1,
        max_steps=maximum,
        heads=3,
        initial_state=initial,
    )


def _observable(module: _FakeOfficial):
    return {
        "graph_map": dict(module.graph_map),
        "graph_index_map": dict(module.graph_index_map),
        "counterfactual_candidates": list(module.counterfactual_candidates),
        "input_graphs_covered": list(module.input_graphs_covered),
        "covering_graphs": set(module.covering_graphs),
        "start": dict(module.start),
        "traversed_hashes": list(module.traversed_hashes),
    }


def test_1000_steps_equal_500_plus_durable_resume(tmp_path) -> None:
    _seed()
    uninterrupted = _FakeOfficial()
    uninterrupted_state = _run(uninterrupted, maximum=1_000)
    uninterrupted_observable = _observable(uninterrupted)
    uninterrupted_next = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )

    _seed()
    interrupted = _FakeOfficial()
    first_state = _run(interrupted, maximum=500)
    database = sqlite3.connect(tmp_path / "live.sqlite3")
    database.execute("CREATE TABLE state (step INTEGER NOT NULL)")
    database.execute("INSERT INTO state VALUES (500)")
    database.commit()
    checkpoint_root = tmp_path / "checkpoints"
    scientific_argv = ("run_generation.py", "--steps=1000")
    provenance = {
        "test": "deterministic-1000-v1",
        "scientific_command_sha256": scientific_command_sha256(scientific_argv),
        "total_steps": "1000",
    }
    save_generation_checkpoint(
        checkpoint_root,
        completed_step=500,
        step_complete=True,
        algorithm_state={
            "loop_state": first_state.to_checkpoint_state(),
            "official_state": snapshot_official_state(interrupted),
        },
        trace_state={"enabled": False},
        sqlite_source=database,
        provenance_fingerprints=provenance,
        scientific_argv=scientific_argv,
        command_sha256=scientific_command_sha256(scientific_argv),
        total_steps=1_000,
    )
    database.close()

    # Crash-left staging directories are not published checkpoints.
    (checkpoint_root / ".step-000000000777-killed.tmp").mkdir()
    (checkpoint_root / ".step-000000000777-killed.tmp" / "partial").write_bytes(b"x")
    assert [path.name for path in list_generation_checkpoints(checkpoint_root)] == [
        "step-000000000500"
    ]

    loaded = load_generation_checkpoint(
        checkpoint_root,
        expected_provenance=provenance,
    )
    resumed = _FakeOfficial()
    restore_official_state(resumed, loaded.algorithm_state["official_state"])
    restore_rng_state(loaded.rng_state)
    resumed_state = _run(
        resumed,
        maximum=1_000,
        initial=GenerationLoopState.from_checkpoint_state(
            loaded.algorithm_state["loop_state"]
        ),
    )
    resumed_next = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )

    assert resumed_state == uninterrupted_state
    assert _observable(resumed) == uninterrupted_observable
    assert resumed_next == uninterrupted_next


def test_nonteleport_next_hash_is_intentionally_not_adopted() -> None:
    module = _FakeOfficial()
    _seed()
    state = run_generation_loop(
        module,
        input_graphs=list(range(4)),
        importance_args={},
        teleport_probability=0.0,
        max_steps=2,
        heads=2,
    )
    assert state.current_graph_hashes == state.start_graph_hashes
    assert module.traversed_hashes == [
        list(state.start_graph_hashes),
        list(state.start_graph_hashes),
    ]


def test_callback_is_after_graph_map_assertions() -> None:
    module = _FakeOfficial()
    observed = []

    def broken_move(**kwargs):
        module.graph_index_map.pop(next(iter(module.graph_index_map)))
        return None, False, None, None, None

    module.move_to_next_graph = broken_move
    with pytest.raises(AssertionError):
        run_generation_loop(
            module,
            input_graphs=list(range(4)),
            importance_args={},
            teleport_probability=0.0,
            max_steps=1,
            heads=2,
            on_step_complete=observed.append,
        )
    assert observed == []
