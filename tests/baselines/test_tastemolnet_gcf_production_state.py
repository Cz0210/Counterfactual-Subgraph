from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid

import pytest

from src.baselines import tastemolnet_gcf_full_resume as resume
from src.baselines import tastemolnet_gcf_production_state as production


def _journal(
    tmp_path: Path,
    *,
    attempt_id: str,
    snapshot=None,
    index_name: str,
    open_writer: bool = True,
):
    return production.T12CompactHistoryJournal(
        root=(tmp_path / "history").resolve(),
        index_root=(tmp_path / index_name).resolve(),
        bounds=production.T12ProductionBounds.pinned(parent_count=2),
        contract_sha256="a" * 64,
        attempt_id=attempt_id,
        generation_token="b" * 64,
        resume_snapshot=snapshot,
        open_writer=open_writer,
    )


def _append(journal, *, graph: str, lineage: str, embedding: str = "d" * 64):
    return journal.append_observation(
        graph_identity_sha256=graph,
        probabilities=(0.1, 0.2, 0.7),
        prediction=2,
        candidate=True,
        valid_fullgraph=True,
        coverage_vector=(1, 0),
        embedding_sha256=embedding,
        failure_reason="",
        lineage_sha256=lineage,
        neurosed_query_sha256="e" * 64,
    )


def test_pinned_bounds_bind_official_parameters_and_finite_20k_caps():
    bounds = production.T12ProductionBounds.pinned(parent_count=3_778)
    proof = bounds.proof()
    assert bounds.total_steps == 20_000
    assert bounds.sample_size == 10_000
    assert bounds.candidate_capacity == 100_000
    assert bounds.checkpoint_cursors == (10_000, 20_000)
    assert bounds.max_scored_observations == 200_000_001
    assert bounds.max_full_live_records == 20_001
    assert bounds.max_transient_full_records == 30_001
    assert proof["bound_pass"] is True
    assert proof["history_payload_retained"] is False
    assert proof["history_neurosed_query_sha256_retained"] is True
    assert proof["history_record_bytes"] == 304
    transition = resume.production_transition_bound_report(bounds=bounds)
    assert transition["official_in_memory_transition_dict_allowed"] is False
    assert transition["production_launch_ready"] is False
    assert transition["minimum_bitpacked_coverage_bytes"] > 85 * 1024**3


def test_compact_history_reopens_committed_prefix_and_chains_second_segment(
    tmp_path: Path,
):
    attempt = str(uuid.uuid4())
    first = _journal(tmp_path, attempt_id=attempt, index_name="index-a")
    row = _append(first, graph="1" * 64, lineage="2" * 64)
    assert row.candidate is True
    _append(first, graph="3" * 64, lineage="4" * 64)
    snapshot_10k = first.checkpoint_state()
    assert snapshot_10k["observation_count"] == 2
    assert len(snapshot_10k["segments"]) == 1
    first.close()

    second = _journal(
        tmp_path,
        attempt_id=attempt,
        snapshot=snapshot_10k,
        index_name="index-b",
    )
    assert second.checkpoint_state() == snapshot_10k
    assert second.lookup_first("1" * 64) == row
    _append(second, graph="5" * 64, lineage="6" * 64)
    snapshot_20k = second.checkpoint_state()
    assert snapshot_20k["observation_count"] == 3
    assert len(snapshot_20k["segments"]) == 2
    second.close()

    verifier = _journal(
        tmp_path,
        attempt_id=attempt,
        snapshot=snapshot_20k,
        index_name="index-c",
        open_writer=False,
    )
    assert verifier.checkpoint_state() == snapshot_20k
    assert verifier.first_seen_graph_count == 3
    verifier.close()


def test_compact_history_rejects_committed_prefix_tamper(tmp_path: Path):
    attempt = str(uuid.uuid4())
    writer = _journal(tmp_path, attempt_id=attempt, index_name="index-a")
    _append(writer, graph="1" * 64, lineage="2" * 64)
    snapshot = writer.checkpoint_state()
    writer.close()
    segment = tmp_path / "history" / snapshot["segments"][0]["segment_file"]
    data = bytearray(segment.read_bytes())
    data[-1] ^= 1
    segment.write_bytes(data)
    with pytest.raises(
        production.TasteT12ProductionStateError, match="hash chain"
    ):
        _journal(
            tmp_path,
            attempt_id=attempt,
            snapshot=snapshot,
            index_name="index-b",
            open_writer=False,
        )


def test_compact_history_waits_for_transient_read_lock(tmp_path: Path):
    attempt = str(uuid.uuid4())
    journal = _journal(tmp_path, attempt_id=attempt, index_name="index-a")
    _append(journal, graph="1" * 64, lineage="2" * 64)

    reader_ready = threading.Event()

    def hold_read_lock() -> None:
        connection = sqlite3.connect(journal._index_path)
        try:
            connection.execute("BEGIN")
            connection.execute("SELECT COUNT(*) FROM first_observation").fetchone()
            reader_ready.set()
            time.sleep(0.2)
            connection.commit()
        finally:
            connection.close()

    reader = threading.Thread(target=hold_read_lock)
    reader.start()
    assert reader_ready.wait(timeout=2)
    started = time.monotonic()
    journal._commit_index()
    elapsed = time.monotonic() - started
    reader.join(timeout=2)

    assert not reader.is_alive()
    assert elapsed >= 0.1
    assert journal._connection.execute("PRAGMA busy_timeout").fetchone() == (
        production.HISTORY_INDEX_BUSY_TIMEOUT_MILLISECONDS,
    )
    journal.close()


def test_compact_history_failure_close_rolls_back_disposable_index(tmp_path: Path):
    attempt = str(uuid.uuid4())
    journal = _journal(tmp_path, attempt_id=attempt, index_name="index-a")
    _append(journal, graph="1" * 64, lineage="2" * 64)
    index_path = journal._index_path

    journal.close(commit_index=False)

    with sqlite3.connect(index_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM first_observation"
        ).fetchone() == (0,)


def test_10k_20k_plan_is_the_only_production_orchestration():
    assert resume.production_segment_bounds(0) == (1, 10_000)
    assert resume.production_segment_bounds(10_000) == (10_001, 20_000)
    with pytest.raises(resume.TasteGCFFullResumeError, match="already"):
        resume.production_segment_bounds(20_000)
    with pytest.raises(resume.TasteGCFFullResumeError, match="0/10k/20k"):
        resume.production_segment_bounds(2_500)


def test_bound_document_is_json_serializable():
    bounds = production.T12ProductionBounds.pinned(parent_count=3_778)
    assert production.T12ProductionBounds.from_dict(
        json.loads(json.dumps(bounds.to_dict()))
    ) == bounds
