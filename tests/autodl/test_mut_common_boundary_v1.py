from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import sqlite3
from types import SimpleNamespace

import pytest

from scripts.autodl import run_mut_trace_mode_equivalence as runner
from src.utils import autodl_mut_common_boundary_v1 as boundary
from src.baselines.comrecgc.generation_checkpoint import (
    load_generation_checkpoint, restore_rng_state, save_generation_checkpoint,
    scientific_command_sha256,
)


class GuardStop(RuntimeError):
    pass


def _actor():
    return SimpleNamespace(candidates=[], frequencies={}, step=0)


def _move(actor):
    actor.step += 1
    head, target = random.randrange(5), random.randrange(17)
    actor.frequencies[target] = actor.frequencies.get(target, 0) + 1
    if target not in actor.candidates:
        actor.candidates.append(target)
    return {"selected_head": head, "selected_action": ["NLC", head, target],
            "candidate_order": list(actor.candidates),
            "frequency": dict(actor.frequencies)}


def _state(actor):
    return SimpleNamespace(completed_step=actor.step, next_step=actor.step + 1,
                           start_graph_hashes=("s",), current_graph_hashes=("t",),
                           restart_indices=(0,))


def _scientific(actor):
    return {"candidates": list(actor.candidates), "frequencies": dict(actor.frequencies)}


def _save(root, actor):
    argv = ("mut-tiny-common-boundary", "M_MAX=50000")
    command = scientific_command_sha256(argv)
    database = sqlite3.connect(":memory:")
    database.execute("CREATE TABLE graphs (key TEXT PRIMARY KEY)")
    try:
        return save_generation_checkpoint(
            root, completed_step=actor.step, step_complete=True,
            algorithm_state={"actor": copy.deepcopy(vars(actor))},
            trace_state={"enabled": False}, sqlite_source=database,
            provenance_fingerprints={"project_commit": "a" * 40,
                                     "scientific_command_sha256": command,
                                     "total_steps": "50000"},
            scientific_argv=argv, command_sha256=command, total_steps=50000,
        )
    finally:
        database.close()


@pytest.fixture
def tiny_observer(monkeypatch):
    monkeypatch.setattr(runner, "_candidate_state", _scientific)
    monkeypatch.setattr(runner, "_rng_state", lambda: {"python": random.getstate()})
    monkeypatch.setattr(runner, "_COMMON_BOUNDARY_MODULE", boundary)


def test_guard_stop_retains_real_250_event_and_joint_checkpoint(tmp_path, tiny_observer):
    path = tmp_path / "common_step_state.jsonl"
    checkpoints = tmp_path / "generation_checkpoints"
    observer = runner._CommonStepObserver(path, phase="continuous", trace_mode="on",
                                           checkpoint_root=checkpoints)
    random.seed(7)
    actor = _actor()
    for step in range(1, 251):
        observer.last_move = _move(actor)
        before_rng = random.getstate()

        def callback(_state):
            # This read proves the actual fsync'd observer event precedes the
            # checkpoint function, including on the storage failure boundary.
            assert json.loads(path.read_text().splitlines()[-1])["step"] == step
            if step == 250:
                _save(checkpoints, actor)
                raise GuardStop("free_inodes_below_limit")

        if step == 250:
            with pytest.raises(GuardStop):
                observer.completed_boundary(_state(actor), actor, callback)
        else:
            observer.completed_boundary(_state(actor), actor, callback)
        assert random.getstate() == before_rng
    joint = runner._resume_common_boundary(tmp_path, path, phase="continuous", trace_mode="on")
    assert joint["checkpoint"]["completed_step"] == 250
    assert joint["observer"]["row_count"] == 250
    assert joint["observer"]["next_step"] == 251
    assert joint["checkpoint"]["payload_reload_verified"] is False
    assert joint["resource_failure_triggers_route_b"] is False


@pytest.mark.parametrize("trace_mode", ["on", "off"])
def test_segmented_500_and_real_checkpoint_reload_501_510_match(tmp_path, tiny_observer, trace_mode):
    random.seed(11)
    actor = _actor()
    baseline = []
    for _ in range(510):
        baseline.append((_move(actor), random.getstate()))

    root = tmp_path / "fresh"
    path = root / "common_step_state.jsonl"
    checkpoints = root / "generation_checkpoints"
    random.seed(11)
    actor = _actor()
    observer = runner._CommonStepObserver(path, phase="continuous", trace_mode=trace_mode,
                                           checkpoint_root=checkpoints)
    for step in range(1, 511):
        observer.last_move = _move(actor)
        assert (observer.last_move, random.getstate()) == baseline[step - 1]
        observer.completed_boundary(
            _state(actor), actor,
            (lambda _s: _save(checkpoints, actor)) if step in (250, 500) else None,
        )
        if step == 250:
            loaded = load_generation_checkpoint(checkpoints, expected_completed_step=250)
            actor = SimpleNamespace(**loaded.algorithm_state["actor"])
            joint = runner._resume_common_boundary(root, path, phase="continuous", trace_mode=trace_mode)
            observer = runner._CommonStepObserver(path, phase="continuous", trace_mode=trace_mode,
                                                   checkpoint_root=checkpoints, resume_boundary=joint)
            restore_rng_state(loaded.rng_state)
    joint500 = runner._resume_common_boundary(root, path, phase="reload", trace_mode=trace_mode)
    assert joint500["observer"]["row_count"] == 500
    loaded = load_generation_checkpoint(checkpoints, expected_completed_step=500)
    actor = SimpleNamespace(**loaded.algorithm_state["actor"])
    observer = runner._CommonStepObserver(path, phase="reload", trace_mode=trace_mode,
                                           checkpoint_root=checkpoints)
    restore_rng_state(loaded.rng_state)
    for step in range(501, 511):
        observer.last_move = _move(actor)
        assert (observer.last_move, random.getstate()) == baseline[step - 1]
        observer.completed_boundary(_state(actor), actor, None)
    rows = runner._read_jsonl(path)
    continuous = {r["step"]: r for r in rows if r["phase"] == "continuous"}
    reloaded = {r["step"]: r for r in rows if r["phase"] == "reload"}
    assert list(continuous) == list(range(1, 511))
    assert list(reloaded) == list(range(501, 511))
    for step in reloaded:
        assert runner._row_science(continuous[step]) == runner._row_science(reloaded[step])


def test_missing_250_cannot_be_inferred_from_checkpoint_or_debug_move_index(tmp_path):
    plan = boundary.recovery_plan(checkpoint_steps=[250], valid_observer_through=249,
                                  joint_committed_steps=[])
    assert plan["common_completed_step"] == 0
    assert plan["replay_range_to_last_algorithm_checkpoint"] == [1, 250]
    assert plan["missing_observer_steps"] == [250, 250]
    assert not plan["route_b_eligible"] and not plan["science_launch_allowed"]
    assert plan["minimum_free_inodes"] == 100000


def test_algorithm_failure_does_not_publish_joint_boundary(tmp_path, tiny_observer):
    actor = _actor()
    observer = runner._CommonStepObserver(tmp_path / "common_step_state.jsonl",
                                           phase="continuous", trace_mode="off",
                                           checkpoint_root=tmp_path / "generation_checkpoints")
    observer.last_move = _move(actor)
    def fail(_state):
        raise OSError("checkpoint write failed")
    with pytest.raises(OSError):
        observer.completed_boundary(_state(actor), actor, fail)
    assert not (tmp_path / "common_boundaries").exists()
    assert json.loads(observer.path.read_text())["step"] == 1


@pytest.mark.parametrize("bad_step", [0, 2])
def test_no_zero_based_or_skipped_common_step(tmp_path, tiny_observer, bad_step):
    actor = _actor()
    observer = runner._CommonStepObserver(tmp_path / "ledger", phase="continuous", trace_mode="off")
    observer.last_move = _move(actor)
    actor.step = bad_step
    with pytest.raises(ValueError, match="skipped/duplicate"):
        observer.completed_boundary(_state(actor), actor, None)
    assert not observer.path.exists()


def test_start_guard_is_not_lowered(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.os, "statvfs", lambda _: SimpleNamespace(
        f_favail=99999, f_bavail=10**9, f_frsize=4096, f_blocks=2*10**9))
    with pytest.raises(RuntimeError, match="MUT_STORAGE_ADMISSION_BLOCKED"):
        runner._require_start_storage(tmp_path / "fresh")


def test_readonly_plan_parser_and_no_science_import():
    args = runner.build_parser().parse_args(["plan-recovery", "--arm-root", "/fresh/trace_on", "--trace-mode", "on"])
    assert args.action == "plan-recovery"


@pytest.mark.parametrize("corruption", ["missing_row", "history", "digest", "mode", "partial_row", "extra_row"])
def test_observer_common_prefix_rejects_missing_or_tampered_events(tmp_path, tiny_observer, corruption):
    actor = _actor()
    path = tmp_path / "common_step_state.jsonl"
    observer = runner._CommonStepObserver(path, phase="continuous", trace_mode="on")
    for _ in range(3):
        observer.last_move = _move(actor)
        observer.completed_boundary(_state(actor), actor, None)
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    if corruption == "missing_row":
        rows.pop(1)
    elif corruption == "history":
        rows[1]["history_digest"] = "0" * 64
    elif corruption == "digest":
        rows[1]["move"]["selected_action"] = ["FORGED"]
    elif corruption == "mode":
        rows[1]["trace_mode"] = "off"
    elif corruption == "extra_row":
        rows.append(rows[-1])
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if corruption == "partial_row":
        path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError):
        boundary.observer_prefix(path, phase="continuous", trace_mode="on", stop_step=3,
                                 science_digest=runner._CommonStepObserver.row_digest)


def test_checkpoint_marker_not_payload_is_reopened(tmp_path, tiny_observer, monkeypatch):
    actor = _actor()
    _move(actor)
    validation = _save(tmp_path / "checkpoints", actor)
    original_read = Path.read_bytes
    def guarded_read(path):
        assert path.suffix not in {".pt", ".sqlite3", ".db"}
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    result = boundary.checkpoint_binding(validation.checkpoint_dir, step=1)
    assert result["completed_step"] == 1
    assert not result["payload_reload_verified"]
    marker = validation.checkpoint_dir / "_CHECKPOINT_COMPLETE.json"
    value = json.loads(marker.read_text())
    value["manifest_sha256"] = "0" * 64
    marker.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="completion marker"):
        boundary.checkpoint_binding(validation.checkpoint_dir, step=1)


def test_start_admission_pass_preserves_all_three_old_limits(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.os, "statvfs", lambda _: SimpleNamespace(
        f_favail=100001, f_bavail=10**9, f_frsize=4096, f_blocks=2*10**9))
    runner._require_start_storage(tmp_path / "fresh")
