from __future__ import annotations

import json

from src.baselines.comrecgc.continuation import decide_resume_or_finalize


def _write(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_complete_generation_becomes_noop(tmp_path) -> None:
    _write(tmp_path / "_RUN_COMPLETE.json", {"run_complete": True})
    _write(
        tmp_path / "progress.json",
        {"current_step": 50_000, "run_complete": True},
    )
    _write(tmp_path / "run_manifest.json", {"run_complete": True})
    (tmp_path / "counterfactuals.pt").write_bytes(b"payload")

    decision = decide_resume_or_finalize(tmp_path)

    assert decision["status"] == "ALREADY_COMPLETE"
    assert decision["fresh_start_allowed"] is False


def test_nonempty_directory_is_not_a_checkpoint(tmp_path) -> None:
    _write(tmp_path / "progress.json", {"current_step": 12_345})
    (tmp_path / "some_state.sqlite3").write_bytes(b"not-a-checkpoint")

    decision = decide_resume_or_finalize(tmp_path)

    assert decision["status"] == "FAIL_CLOSED"
    assert decision["resume_safe"] is False
    assert decision["fresh_start_allowed"] is False
