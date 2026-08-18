from __future__ import annotations

import json

from src.baselines.comrecgc.continuation import decide_resume_or_finalize


def test_timed_out_v6_progress_without_rng_manifest_fails_closed(tmp_path) -> None:
    (tmp_path / "progress.json").write_text(
        json.dumps({"current_step": 21_000, "run_complete": False}),
        encoding="utf-8",
    )
    (tmp_path / "graph_state").mkdir()
    (tmp_path / "graph_state" / "authoritative_graph_store.sqlite3").write_bytes(
        b"state without RNG/transition checkpoint"
    )

    result = decide_resume_or_finalize(tmp_path)

    assert result["status"] == "FAIL_CLOSED"
    assert result["reason"] == "no_atomic_rng_transition_trace_closure_checkpoint"
    assert result["fresh_start_allowed"] is False
