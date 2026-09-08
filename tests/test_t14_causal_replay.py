import hashlib
from pathlib import Path

import pytest

from src.baselines.t14_causal_replay import copy_prefix, validate_campaign


def test_committed_prefix_clone_does_not_modify_source(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"sealed-prefix" + b"uncommitted-suffix")
    before = source.stat()
    destination = tmp_path / "fresh" / "copy.bin"
    result = copy_prefix(source, destination, len(b"sealed-prefix"))
    assert destination.read_bytes() == b"sealed-prefix"
    assert source.read_bytes() == b"sealed-prefixuncommitted-suffix"
    assert source.stat().st_mtime_ns == before.st_mtime_ns
    assert result["content_sha256"] == hashlib.sha256(b"sealed-prefix").hexdigest()


def test_prefix_clone_rejects_symlink_and_short_source(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"abc")
    alias = tmp_path / "alias"
    alias.symlink_to(source)
    with pytest.raises(ValueError):
        copy_prefix(alias, tmp_path / "copy", 3)
    with pytest.raises(ValueError):
        copy_prefix(source, tmp_path / "copy", 4)


def test_replay_preserves_original_sources_and_has_no_formal_stage():
    source = (Path(__file__).parents[1] / "src/baselines/t14_causal_replay.py").read_text()
    assert "max_steps=335" in source
    assert "if started >= 85" in source
    assert "_restore_checkpoint_state" in source
    assert "DIAGNOSTIC_ONLY_NOT_PROMOTABLE" in source
    assert "promote_generation_checkpoint" not in source
    assert "recover_route_c_external_state(" not in source
    assert "random.seed" not in source
    assert "_seed_all" not in source


def test_campaign_rejects_extra_arm_or_other_root(tmp_path):
    import json
    path = tmp_path / "campaign.json"
    value = {"total_new_transition_cap": 170, "arms": {name: {"start_step": 250, "end_step": 335, "new_transition_cap": 85, "source_spec": str(tmp_path / (name+"-spec.json")), "output_root": str(tmp_path/name)} for name in ("reference", "lowmemory")}}
    path.write_text(json.dumps(value))
    assert len(validate_campaign(path, source_spec_path=tmp_path/"reference-spec.json", output_root=tmp_path/"reference")) == 2
    with pytest.raises(ValueError):
        validate_campaign(path, source_spec_path=tmp_path/"reference-spec.json", output_root=tmp_path/"retry3")
    value["total_new_transition_cap"] = 234
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        validate_campaign(path, source_spec_path=tmp_path/"reference-spec.json", output_root=tmp_path/"reference")


def test_real_route_store_clone_restores250_without_source_rollback(tmp_path):
    import sqlite3
    import torch
    from types import SimpleNamespace
    from src.baselines import t14_causal_diagnostic as diagnostic
    from src.baselines.t14_causal_replay import clone_route_state
    from src.baselines.tastemolnet_t14_route_c_fresh import RouteCStateUpdater
    source = tmp_path / "source"
    def entry(atom):
        graph = SimpleNamespace(x=torch.tensor([[atom, 0], [atom, 1]], dtype=torch.float32), edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), num_nodes=2)
        return [graph, [float(atom), 0.0], [2]]
    updater = RouteCStateUpdater(source / "route_c_state", candidate_capacity=4, record_capacity=12, lru_capacity=2)
    updater.graph_store.put("g0", entry(6), sequence_id=updater.next_sequence())
    row = {"frequency": 7, "graph_hash": "g0", "importance_parts": (0.75, 0.0), "input_graphs_covering_list": None}
    updater.candidates.append(row)
    state = updater.checkpoint_state()
    snapshot = source / "sealed250.sqlite3"
    connection = sqlite3.connect(snapshot)
    updater.graph_store.checkpoint_connection.backup(connection)
    connection.close()
    updater.graph_store.put("g1", entry(7), sequence_id=updater.next_sequence())
    updater.candidates.append(dict(row, graph_hash="g1"))
    updater.close()
    original = (source / "route_c_state/graph_store/graphs.bin").read_bytes()
    loaded = SimpleNamespace(validation=SimpleNamespace(checkpoint_digest="d"*64), sqlite_snapshot_path=snapshot, algorithm_state={"route_c_state": state, "live_graph_state": {"store": state["graph_store"]}})
    destination = tmp_path / "fresh"
    clone_route_state(source, destination, loaded, diagnostic)
    assert (source / "route_c_state/graph_store/graphs.bin").read_bytes() == original
    restored = RouteCStateUpdater(destination / "route_c_state", candidate_capacity=4, record_capacity=12, lru_capacity=2, resume=True)
    restored.restore_checkpoint_state(state)
    assert restored.graph_store.count() == 1
    assert [dict(value) for value in restored.candidates] == [row]
    restored.close()
