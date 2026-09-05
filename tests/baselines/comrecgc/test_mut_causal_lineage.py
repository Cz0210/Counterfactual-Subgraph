from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import random
import shutil
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from src.baselines.comrecgc.graph_trace import (
    ActionTraceRecorder, load_selected_trace, normalized_untyped_graph_payload,
    apply_action_to_normalized_payload,
)
from src.baselines.comrecgc.mut_causal_lineage import (
    MutCausalLineageError, MutCausalLineageRecorder, validate_causal_scope,
    validate_causal_recovery_audit,
)
from src.baselines.comrecgc.transition_cache import CompactMoveScopedTransitionMap


@dataclass
class Graph:
    x: list[list[float]]
    edge_index: list[list[int]]
    num_nodes: int = 2
    comrecgc_parent_id: str = "train-parent"
    comrecgc_trace_node_ids: tuple[str, ...] = ("source:0", "source:1")


def graph(label: int, width: int = 1040) -> Graph:
    row = [0.0] * width
    row[label] = 1.0
    return Graph([row, [1.0] + [0.0] * (width - 1)], [[0, 1], [1, 0]])


class FixtureWalk:
    """Tiny indexed move fixture; no model/data load and not production parity."""

    def __init__(self, root=None, *, debug=False):
        self.module = SimpleNamespace(
            graph_map={"g0": [graph(0), np.array([0.0]), 2]},
            graph_index_map={}, counterfactual_candidates=[],
        )
        self.head = "g0"
        self.step = 0
        self.choices = []
        self.cache = CompactMoveScopedTransitionMap(
            self.module, {}, seed=7, expanded_capacity=2,
            rebuild_target=lambda src, action: graph(int(action[2])),
        )
        self.module.transitions = self.cache
        self.recorder = (ActionTraceRecorder(root, chunk_size=64, compact_enumeration=True)
                         if debug else MutCausalLineageRecorder(root, chunk_size=64)) if root else None

    def original_move(self, *, graphs_hash, **_):
        self.step += 1
        if random.uniform(0, 1) < 0.04:
            self.choices.append((self.step, "teleport"))
            return None, True, None, None, None
        source = graphs_hash[0]
        source_graph = self.module.graph_map[source][0]
        if source not in self.cache:
            targets, hashes, actions = [], [], []
            for label in (self.step * 2, self.step * 2 + 1):
                target, action = graph(label), ("NLC", 0, label)
                self.cache.record_enumerated(target, action)
                if self.recorder:
                    self.recorder.record_enumerated(source_graph=source_graph, target_graph=target, action=action)
                targets.append(target); hashes.append(f"g{label}"); actions.append(action)
            self.cache[source] = (hashes, targets, [[0.7, 1.0], [0.8, 1.0]], [[1.0], [2.0]])
        # Same two lead + neighbor draws as the fixture reference without any
        # interception/replacement of random calls by the observer.
        lead = random.choices(range(len(graphs_hash)))[0]
        hashes, targets, importance, embeddings = self.cache[source]
        index = random.choices(range(len(hashes)), weights=[0.7, 0.8])[0]
        target = hashes[index]
        self.choices.append((self.step, lead, source, index, target))
        self.module.graph_map.setdefault(target, [targets[index], embeddings[index], 2])
        registry = self.module.graph_index_map
        if target not in registry:
            registry[target] = len(self.module.counterfactual_candidates)
            self.module.counterfactual_candidates.append({"graph_hash": target, "frequency": 1})
        else:
            self.module.counterfactual_candidates[registry[target]]["frequency"] += 1
        return [target], False, None, importance[index], None

    def advance(self, count):
        move = self.original_move
        if self.recorder:
            move = self.recorder.wrap_move(move, self.module)
        move = self.cache.wrap_move(move)
        for _ in range(count):
            result = move(graphs_hash=[self.head], start_graphs_hash=["g0"],
                          importance_args={}, teleport_probability=0.04)
            self.head = "g0" if result[1] else result[0][0]

    def scientific(self):
        return deepcopy({"head": self.head, "step": self.step, "choices": self.choices,
                         "rng": random.getstate(),
                         "candidates": self.module.counterfactual_candidates,
                         "registry": self.module.graph_index_map,
                         "graphs": {k: normalized_untyped_graph_payload(v[0])
                                    for k, v in self.module.graph_map.items()},
                         "transition_order": list(self.cache)})


def test_fixture_500_steps_rng_action_candidate_frequency_parity(tmp_path):
    values = []
    for mode in ("off", "causal", "debug"):
        random.seed(7)
        walk = FixtureWalk(None if mode == "off" else tmp_path / mode, debug=mode == "debug")
        walk.advance(500)
        values.append(walk.scientific())
        if mode == "causal":
            assert walk.recorder.enumerated == {}
            assert walk.recorder._enumerated_by_target_object == {}
            assert walk.recorder.move_index == 500
    assert values[0] == values[1] == values[2]


def test_checkpoint_500_reload_501_510_matches_uninterrupted(tmp_path):
    from src.baselines.comrecgc.generation_checkpoint import (
        save_generation_checkpoint, load_generation_checkpoint, scientific_command_sha256,
    )
    random.seed(7)
    first = FixtureWalk(tmp_path / "first")
    first.advance(500)
    checkpoint = deepcopy({"module": {k:getattr(first.module,k) for k in
                           ("graph_map", "graph_index_map", "counterfactual_candidates")}, "head": first.head,
                           "step": first.step, "choices": first.choices,
                           "rng": random.getstate(),
                           "transition": first.cache.export_checkpoint_state(),
                           "causal": first.recorder.export_checkpoint_state()})
    # Use the real generation checkpoint serializer + sealed SQLite snapshot,
    # rather than proving resume only with a Python dict copy. This is still a
    # tiny fixture, not the live Mut 1,448-parent production canary.
    argv = ("fixture", "--dataset=mutagenicity", "--causal-lineage=true")
    command_sha = scientific_command_sha256(argv)
    provenance = {"config_sha256": "c" * 64, "dataset_fingerprint": "d" * 64,
                  "external_commit": "e" * 40, "project_commit": "f" * 40,
                  "scientific_command_sha256": command_sha, "total_steps": "50000"}
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE graph_state(key TEXT PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO graph_state VALUES ('head', ?)", (first.head,))
        connection.commit()
        save_generation_checkpoint(
            tmp_path / "checkpoints", completed_step=500, step_complete=True,
            algorithm_state={key: value for key, value in checkpoint.items() if key != "causal"},
            trace_state=checkpoint["causal"], sqlite_source=connection,
            provenance_fingerprints=provenance, scientific_argv=argv,
            command_sha256=command_sha, total_steps=50000,
        )
    loaded = load_generation_checkpoint(
        tmp_path / "checkpoints", expected_provenance=provenance,
        expected_completed_step=500, expected_total_steps=50000,
    )
    checkpoint = {**loaded.algorithm_state, "causal": loaded.trace_state}
    assert loaded.validation.manifest["next_step"] == 501
    assert loaded.rng_state["python"] == checkpoint["rng"]
    # Only closed chunks from the committed checkpoint are copied, not live files.
    resumed = FixtureWalk(tmp_path / "resumed")
    for row in checkpoint["causal"]["causal_state"]["chunks"]:
        shutil.copyfile(tmp_path / "first" / row["path"], tmp_path / "resumed" / row["path"])
    first.advance(10)
    expected = first.scientific()
    for name in ("graph_map", "graph_index_map", "counterfactual_candidates"):
        setattr(resumed.module, name, checkpoint["module"][name])
    resumed.head = checkpoint["head"]; resumed.step = checkpoint["step"]
    resumed.choices = checkpoint["choices"]
    resumed.cache.restore_checkpoint_state(checkpoint["transition"])
    resumed.recorder.restore_checkpoint_state(checkpoint["causal"])
    random.setstate(checkpoint["rng"])
    resumed.advance(10)
    assert resumed.scientific() == expected
    assert resumed.recorder.export_checkpoint_state() == first.recorder.export_checkpoint_state()


def test_event_and_checkpoint_are_deep_snapshots(tmp_path):
    recorder = MutCausalLineageRecorder(tmp_path)
    event = {"event": "selected_transition", "action_resolution": "exact",
             "action": ["NLC", 0, 1], "source_official_hash": "a", "target_official_hash": "b"}
    recorder._stream_event(event)
    event["action"][2] = 99
    state = recorder.export_checkpoint_state()
    state["causal_state"]["pending_events"][0]["action"][2] = 88
    assert recorder._pending_events[0]["action"] == ["NLC", 0, 1]


def test_five_selected_heads_are_observed_after_population_update(tmp_path, monkeypatch):
    walk = FixtureWalk(tmp_path)
    sources, targets = [], []
    for index in range(5):
        source, target = f"s{index}", f"t{index}"
        sources.append(source); targets.append(target)
        walk.module.graph_map[source] = [graph(index)]
        target_graph = graph(10 + index)
        walk.cache.record_enumerated(target_graph, ("NLC", 0, 10 + index))
        walk.cache[source] = ([target], [target_graph], [[0.8, 1.0]], [[float(index)]])

    # The selected head/follower outcome is fixed by the original move. Any
    # additional draw from the observer would fail immediately.
    def forbidden_rng(*args, **kwargs):
        raise AssertionError("observer requested a random draw")
    for name in ("random", "uniform", "choice", "choices", "sample", "randint"):
        monkeypatch.setattr(random, name, forbidden_rng)

    def original(**_):
        for index, (source, target) in enumerate(zip(sources, targets, strict=True)):
            hashes, graphs, _, _ = walk.cache[source]
            walk.module.graph_map[target] = [graphs[0]]
            walk.module.graph_index_map[target] = index
            walk.module.counterfactual_candidates.append({"graph_hash": target, "frequency": 3})
        return targets, False, None, None, None

    observed = []
    stream = walk.recorder._stream_event
    def record(event):
        assert len(walk.module.counterfactual_candidates) == 5
        observed.append(deepcopy(event))
        stream(event)
    monkeypatch.setattr(walk.recorder, "_stream_event", record)
    wrapped = walk.cache.wrap_move(walk.recorder.wrap_move(original, walk.module))
    assert wrapped(graphs_hash=sources)[0] == targets
    assert [row["head_index"] for row in observed] == list(range(5))
    assert [row["action"] for row in observed] == [["NLC", 0, 10 + i] for i in range(5)]
    assert [row["frequency"] for row in walk.module.counterfactual_candidates] == [3] * 5
    assert list(walk.module.graph_index_map) == targets


@pytest.mark.parametrize("reason", ["missing", "ambiguous", "bad_replay"])
def test_actual_action_must_be_unique_and_replay_exact(tmp_path, monkeypatch, reason):
    walk = FixtureWalk(tmp_path)
    walk.module.graph_map["target"] = [graph(1)]
    rows = [] if reason == "missing" else [{"action": ["NLC", 0, 2 if reason == "bad_replay" else 1]}]
    if reason == "ambiguous": rows = rows * 2
    monkeypatch.setattr(walk.cache, "action_records", lambda *_: rows)
    with pytest.raises(MutCausalLineageError):
        walk.recorder._compact_transition_records(walk.module, source_hash="g0", target_hash="target")


def test_short_selected_universe_replays_and_serializes_compactly(tmp_path):
    random.seed(7)
    walk = FixtureWalk(tmp_path)
    walk.advance(4)
    result = {"graph_map": walk.module.graph_map,
              "counterfactual_candidates": walk.module.counterfactual_candidates}
    before = deepcopy(result)
    summary = walk.recorder.write(tmp_path, result,
        source_graphs_by_parent_id={"train-parent": graph(0)}, compact_candidate_lineage=True)
    assert json.dumps(result, default=lambda x: vars(x) if isinstance(x, Graph) else x.tolist(), sort_keys=True) == json.dumps(before, default=lambda x: vars(x) if isinstance(x, Graph) else x.tolist(), sort_keys=True)
    assert summary["candidate_count"] == summary["candidate_lineage_resolved_count"]
    manifest = json.loads((tmp_path / "causal_lineage_manifest.json").read_text())
    assert manifest["debug_trace_enabled"] is False
    assert manifest["generation_production_parity_claimed"] is False
    for key in ("recorded_action_index_remap_count", "legacy_inference_called_count",
                "semantic_transition_lineage_replacement_count"):
        changed = deepcopy(summary)
        changed["lineage_recovery_audit"][key] = 1
        with pytest.raises(MutCausalLineageError, match="infer or replace"):
            validate_causal_recovery_audit(changed)
    for event in load_selected_trace(summary["selected_trace_path"]):
        if event["event"] == "selected_transition":
            a = result["graph_map"][event["source_official_hash"]][0]
            b = normalized_untyped_graph_payload(result["graph_map"][event["target_official_hash"]][0])
            assert apply_action_to_normalized_payload(a, event["action"]) == b


def test_scope_does_not_enable_other_datasets_or_debug(tmp_path):
    assert validate_causal_scope(dataset="bace", mode="full", output_root=tmp_path,
                                debug_trace_root=None, causal_root=None) is None
    for changes in ({"dataset": "bace"}, {"mode": "smoke"}, {"debug_trace_root": tmp_path / "trace"},
                    {"causal_root": tmp_path / "other"}):
        args = dict(dataset="mutagenicity", mode="full", output_root=tmp_path,
                    debug_trace_root=None, causal_root=tmp_path / "causal_lineage")
        args.update(changes)
        with pytest.raises(MutCausalLineageError): validate_causal_scope(**args)


def test_inactive_cli_keeps_existing_scientific_argv():
    from scripts.baselines.comrecgc.run_generation import canonical_scientific_argv
    base = SimpleNamespace(route="project", dataset="mutagenicity", resume=False)
    old = canonical_scientific_argv(base)
    base.mut_causal_lineage_output_dir = None
    assert canonical_scientific_argv(base) == old
    base.mut_causal_lineage_output_dir = "/fresh/causal_lineage"
    assert canonical_scientific_argv(base) != old


def test_cli_does_not_silently_ignore_causal_option_on_native_route(monkeypatch, tmp_path):
    from scripts.baselines.comrecgc import run_generation
    monkeypatch.setattr("sys.argv", ["run_generation.py", "--config", "configs/hpc.yaml",
        "--route", "native", "--dataset", "mutagenicity", "--mode", "full",
        "--upstream-root", "/unused", "--dataset-dir", "/unused",
        "--output-dir", str(tmp_path / "unused"),
        "--mut-causal-lineage-output-dir", str(tmp_path / "unused/causal_lineage")])
    with pytest.raises(ValueError, match="project full route"):
        run_generation.main()
    assert not list(tmp_path.iterdir())
