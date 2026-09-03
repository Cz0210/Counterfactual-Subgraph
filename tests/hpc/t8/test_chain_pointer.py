from __future__ import annotations

import fcntl
from pathlib import Path

import pytest

from scripts.hpc.t8 import run_stress_followup as followup
from src.utils.hpc_t8_chain_pointer import (
    T8ChainPointerError,
    chain_lock,
    load_current_pointer,
    write_current_pointer,
)


def _refinement_payload(
    *, canary: str = "101", followup_job: str = "102", depth: int = 5
) -> dict[str, object]:
    return {
        "schema_version": "t8_hpc_current_chain_v1",
        "state": "REFINEMENT_CANARY_SUBMITTED",
        "active_stage": "REFINEMENT_CANARY",
        "updated_at": "2026-09-04T00:00:00+00:00",
        "output_root": "/runtime/continuation/control",
        "continuation_root": "/runtime/continuation/artifacts",
        "decision_root": "/runtime/continuation/control/upstream-100",
        "upstream_job_id": "100",
        "controller_commit": "a" * 40,
        "science_commit": "b" * 40,
        "refinement_depth": depth,
        "canary_job_id": canary,
        "followup_job_id": followup_job,
        "followup_dependency": f"afterany:{canary}",
        "canary_root": "/runtime/continuation/artifacts/canary",
        "matrix_write_enabled": False,
        "gpu_requested": False,
    }


def test_current_pointer_is_atomic_self_hashed_and_monotonic(tmp_path: Path) -> None:
    path = tmp_path / "control" / "current.json"
    first = write_current_pointer(path, _refinement_payload())

    assert load_current_pointer(path) == first
    assert path.with_suffix(".json.sha256").is_file()
    assert not list(path.parent.glob("*.tmp"))

    second = write_current_pointer(
        path, _refinement_payload(canary="201", followup_job="202", depth=6)
    )
    assert second["previous_current_sha256"] == first["current_sha256"]
    assert load_current_pointer(path) == second

    stale = write_current_pointer(path, _refinement_payload(depth=5))
    assert stale == second
    assert load_current_pointer(path) == second


def test_one_canary_cannot_be_bound_to_two_followups(tmp_path: Path) -> None:
    path = tmp_path / "current.json"
    write_current_pointer(path, _refinement_payload())
    with pytest.raises(T8ChainPointerError, match="two follow-up"):
        write_current_pointer(path, _refinement_payload(followup_job="999"))


def test_chain_lock_serializes_the_stable_pointer(tmp_path: Path) -> None:
    pointer = tmp_path / "chain" / "current.json"
    with chain_lock(pointer):
        with (pointer.parent / ".chain.lock").open("a+b") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_submitted_refinement_publishes_dynamic_pointer(tmp_path: Path) -> None:
    pointer = tmp_path / "runtime" / "control" / "current.json"
    args = type(
        "Args",
        (),
        {
            "submit": True,
            "current_pointer": pointer,
            "output_root": tmp_path / "decision",
            "continuation_root": tmp_path / "artifacts",
            "expected_controller_commit": "a" * 40,
            "expected_science_commit": "b" * 40,
        },
    )()
    followup._publish_submitted_chain_pointer(
        args,
        {"job_id": "100"},
        tmp_path / "decision" / "upstream-100",
        {
            "state": "REFINEMENT_CANARY_SUBMITTED",
            "refinement_level": 3,
            "refinement_canary_job_id": "201",
            "afterany_followup_job_id": "202",
            "fresh_canary_root": str(tmp_path / "artifacts" / "depth-6"),
        },
    )
    current = load_current_pointer(pointer)
    assert current["refinement_depth"] == 6
    assert current["canary_job_id"] == "201"
    assert current["followup_job_id"] == "202"
    assert current["followup_dependency"] == "afterany:201"


def test_refinement_limit_is_exact_depth_eight() -> None:
    assert followup.INITIAL_STRESS_DEPTH == 3
    assert followup.MAX_REFINEMENT_DEPTH == 8
    assert followup.MAX_REFINEMENT_LEVELS == 5
