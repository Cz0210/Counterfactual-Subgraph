from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import uuid

import numpy as np
import pytest

from src.baselines.tastemolnet_gcf_production_state import (
    T12CompactHistoryJournal, T12ProductionBounds, TasteT12ProductionStateError,
)
from src.utils.main_ready_task_specs import stable_sha256
from src.utils.t12_future_history_cache import load_cache, stage_closed_snapshot


def fixture(tmp_path):
    root = tmp_path / "history"
    journal = T12CompactHistoryJournal(root=root, index_root=tmp_path/"index",
        bounds=T12ProductionBounds.pinned(parent_count=2), contract_sha256="a"*64,
        attempt_id=str(uuid.uuid4()), generation_token="b"*64)
    journal.bind_first_seen_embedding_authority(model_sha256="c"*64, feature_schema_sha256="d"*64)
    values = {}
    for i in (1, 2, 1):
        graph = str(i)*64
        if graph not in values:
            data = np.array([1.25*i, -0.0, 3.5], dtype=np.float32)
            first = journal.append_first_embedding(graph_identity_sha256=graph,
                dtype=data.dtype.str, shape=data.shape, raw_bytes=data.tobytes())
            values[graph] = first
        journal.append_observation(graph_identity_sha256=graph, probabilities=(0.1,0.2,0.7),
            prediction=2, candidate=True, valid_fullgraph=True, coverage_vector=(1,0),
            embedding_sha256=values[graph].embedding_sha256, failure_reason="",
            lineage_sha256=str(i)*64, neurosed_query_sha256="e"*64)
    snapshot = journal.checkpoint_state()
    journal.close()
    proc = tmp_path / "proc"
    proc.mkdir()
    return snapshot, values, proc


def stage(tmp_path, snapshot, proc, **kwargs):
    return stage_closed_snapshot(snapshot, expected_snapshot_sha256=stable_sha256(snapshot),
        cache_root=tmp_path/"cache", producer_pid=987654, producer_start_ticks=123,
        proc_root=proc, min_free_bytes=kwargs.pop("min_free_bytes", 0), min_free_inodes=0, **kwargs)


def reopen(tmp_path, snapshot, cache=None, **kwargs):
    return T12CompactHistoryJournal(root=snapshot["history_root"],
        index_root=tmp_path/str(uuid.uuid4()), bounds=T12ProductionBounds.from_dict(snapshot["bounds"]),
        contract_sha256=snapshot["contract_sha256"], attempt_id=snapshot["attempt_id"],
        generation_token=snapshot["generation_token"], resume_snapshot=snapshot,
        open_writer=kwargs.get("open_writer", False), history_read_cache=cache)


def test_original_codec_records_state_and_first_seen_bytes_exact(tmp_path):
    snapshot, values, proc = fixture(tmp_path)
    original = copy.deepcopy(snapshot)
    before = {p: p.read_bytes() for p in Path(snapshot["history_root"]).rglob("*.bin")}
    result = stage(tmp_path, snapshot, proc)
    cache = load_cache(tmp_path/"cache", expected_manifest_sha256=result["cache_manifest_sha256"])
    uncached, cached = reopen(tmp_path, snapshot), reopen(tmp_path, snapshot, cache)
    try:
        assert uncached.checkpoint_state() == cached.checkpoint_state() == original
        assert uncached.observation_count == cached.observation_count == 3
        for graph, expected in values.items():
            assert cached.lookup_first_embedding(graph) == uncached.lookup_first_embedding(graph) == expected
            assert cached.lookup_first(graph) == uncached.lookup_first(graph)
    finally:
        uncached.close(); cached.close()
    assert snapshot == original
    assert all(p.read_bytes() == data for p, data in before.items())
    assert all(row["source_full_reads"] == row["local_verification_full_reads"] == 1
               for row in result["manifest"]["files"])


def test_active_producer_is_rejected_before_copy(tmp_path):
    snapshot, _, proc = fixture(tmp_path)
    alive = proc/"987654"; alive.mkdir()
    (alive/"stat").write_text("987654 (python) S " + "0 "*18 + "123 " + "0 "*5)
    with pytest.raises(Exception, match="still present"):
        stage(tmp_path, snapshot, proc)
    assert not (tmp_path/"cache").exists()


def test_writable_source_fd_rejected(tmp_path):
    snapshot, _, proc = fixture(tmp_path)
    live = proc/"77"; (live/"fd").mkdir(parents=True); (live/"fdinfo").mkdir()
    source = next(Path(snapshot["history_root"]).glob("history-*.bin"))
    (live/"fd/3").symlink_to(source)
    (live/"fdinfo/3").write_text("flags:\t0100001\n")
    with pytest.raises(Exception, match="writable fd"):
        stage(tmp_path, snapshot, proc)
    assert not (tmp_path/"cache").exists()


def test_uncommitted_tail_is_not_promoted_to_sealed_cache(tmp_path):
    snapshot, _, proc = fixture(tmp_path)
    source = next(Path(snapshot["history_root"]).glob("history-*.bin"))
    with source.open("ab") as stream: stream.write(b"tail")
    with pytest.raises(ValueError, match="NOT_FULLY_SEALED"):
        stage(tmp_path, snapshot, proc)


def test_local_capacity_guard_not_lowered(tmp_path):
    snapshot, _, proc = fixture(tmp_path)
    with pytest.raises(ValueError, match="RESOURCE_ADMISSION"):
        stage(tmp_path, snapshot, proc, min_free_bytes=10**20)
    assert not (tmp_path/"cache").exists()


def test_source_tamper_cannot_be_accepted(tmp_path):
    snapshot, _, proc = fixture(tmp_path)
    source = next(Path(snapshot["history_root"]).glob("history-*.bin"))
    data = bytearray(source.read_bytes()); data[-1] ^= 1; source.write_bytes(data)
    with pytest.raises(ValueError, match="SOURCE_COPY_CHANGED"):
        stage(tmp_path, snapshot, proc)
    assert not (tmp_path/"cache/cache_manifest.json").exists()


def test_cache_mutation_rejected_without_rehashing_large_sources(tmp_path):
    snapshot, _, proc = fixture(tmp_path)
    result = stage(tmp_path, snapshot, proc)
    cache = load_cache(tmp_path/"cache", expected_manifest_sha256=result["cache_manifest_sha256"])
    entry = result["manifest"]["files"][0]
    target = Path(entry["cache_path"]); target.chmod(0o644)
    with pytest.raises(ValueError, match="IDENTITY_CHANGED"):
        cache.open(Path(entry["source"]))


def test_future_cache_cannot_open_writer_or_change_snapshot(tmp_path):
    snapshot, _, proc = fixture(tmp_path)
    result = stage(tmp_path, snapshot, proc)
    cache = load_cache(tmp_path/"cache", expected_manifest_sha256=result["cache_manifest_sha256"])
    with pytest.raises(TasteT12ProductionStateError, match="read-only"):
        reopen(tmp_path, snapshot, cache, open_writer=True)
    changed = copy.deepcopy(snapshot); changed["observation_count"] += 1
    with pytest.raises(ValueError, match="SNAPSHOT_CHANGED"):
        reopen(tmp_path, changed, cache)


def test_fresh_root_never_overwrites_existing_cache(tmp_path):
    snapshot, _, proc = fixture(tmp_path)
    stage(tmp_path, snapshot, proc)
    with pytest.raises(ValueError, match="FRESH_ROOT"):
        stage(tmp_path, snapshot, proc)


def test_hot_codec_and_embedding_lookup_never_reopen_source_payload(tmp_path, monkeypatch):
    snapshot, values, proc = fixture(tmp_path)
    result = stage(tmp_path, snapshot, proc)
    cache = load_cache(tmp_path/"cache", expected_manifest_sha256=result["cache_manifest_sha256"])
    sources = {Path(row["source"]) for row in result["manifest"]["files"]}
    old_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path in sources:
            raise AssertionError("hot cache must not reopen original payload")
        return old_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    for _ in range(2):
        journal = reopen(tmp_path, snapshot, cache)
        try:
            assert journal.checkpoint_state() == snapshot
            for graph, value in values.items():
                assert journal.lookup_first_embedding(graph) == value
        finally:
            journal.close()
