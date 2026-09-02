from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.utils import autodl_mut_throttled_continuation_v1 as policy


def _manifest() -> dict[str, Any]:
    return {
        "tasks": [
            {
                "task_id": "taste_t14",
                "pid": 14,
                "start_ticks": 140,
                "progress_path": "/progress.json",
                "counter_field": "step",
                "terminal_value": 20_000,
            }
        ]
    }


def _baseline(rate: float | None = 1.0) -> dict[str, Any]:
    return {
        "status": "PASS",
        "tasks": {
            "taste_t14": {
                "state": "ACTIVE",
                "units_per_second": rate,
            }
        },
    }


def _row(
    second: float,
    counter: float,
    *,
    phase: str = "GENERATION",
    checkpoint: bool = False,
    cpu_ticks: int = 0,
    output_bytes: int = 0,
    memory_pressure: bool = False,
    io_pressure: bool = False,
) -> dict[str, Any]:
    return {
        "task_id": "taste_t14",
        "pid": 14,
        "alive": True,
        "completed": False,
        "counter": counter,
        "sampled_at_unix": second,
        "phase": phase,
        "checkpoint_or_flush_active": checkpoint,
        "cpu_ticks": cpu_ticks,
        "output_bytes": output_bytes,
        "memory_pressure": memory_pressure,
        "io_pressure": io_pressure,
    }


def _reader(rows: list[dict[str, Any]]):
    iterator = iter(rows)
    return lambda _task: next(iterator)


def test_mut_robust_policy_is_frozen() -> None:
    receipt = policy.MutThrottlePolicy().validate().as_receipt()
    assert receipt["workers"] == 2
    assert receipt["nice"] == 10
    assert receipt["ionice_priority"] == 7
    assert receipt["prefetch"] == 1
    assert receipt["baseline_seconds"] == 1_800
    assert receipt["evaluation_seconds"] == 1_200
    assert receipt["maximum_slowdown"] == 0.15
    with pytest.raises(policy.MutContinuationPolicyError):
        policy.MutThrottlePolicy(workers=3).validate()


def test_mut_cpu_selection_avoids_smt_siblings() -> None:
    before = {0: (0, 100), 1: (0, 100), 2: (0, 100), 3: (0, 100)}
    after = {0: (1, 200), 1: (2, 200), 2: (3, 200), 3: (4, 200)}
    siblings = {0: (0, 1), 1: (0, 1), 2: (2, 3), 3: (2, 3)}
    # CPU1 is the second least busy, but shares CPU0's physical core.
    assert policy.select_two_least_busy_cpus(
        before,
        after,
        candidates=(0, 1, 2, 3),
        sibling_groups=siblings,
    ) == (0, 2)


def _fake_process(proc: Path, pid: int, parent: int, argv: list[str]) -> None:
    root = proc / str(pid)
    root.mkdir(parents=True)
    (root / "cmdline").write_bytes(b"\0".join(item.encode() for item in argv) + b"\0")
    (root / "stat").write_text(
        f"{pid} (python) S {parent} 0 0 0 0 0 0 0 0 0 0 0\n",
        encoding="utf-8",
    )


def test_mut_single_continuation_owner_ignores_own_children(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    _fake_process(proc, 100, 1, ["python", "run_mut_trace_on_adoption_worker.py", "run"])
    _fake_process(
        proc,
        101,
        100,
        ["python", "run_mut_trace_mode_equivalence.py", "run-pair"],
    )
    receipt = policy.assert_single_continuation_owner(proc, current_pid=100)
    assert receipt["status"] == "PASS"

    _fake_process(
        proc,
        200,
        1,
        ["python", "run_mut_checkpoint_instrumentation_equivalence.py", "run-pair"],
    )
    with pytest.raises(policy.MutContinuationPolicyError, match="200"):
        policy.assert_single_continuation_owner(proc, current_pid=100)


def test_mut_attached_controller_is_not_misclassified_as_second_writer(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    _fake_process(proc, 50, 1, ["python", "run_mut_fast_accurate_v2.py", "run"])
    _fake_process(proc, 100, 1, ["python", "run_mut_trace_on_adoption_worker.py", "run"])
    receipt = policy.assert_single_continuation_owner(
        proc, current_pid=100, attached_controller_pid=50
    )
    assert receipt["attached_controller_pid"] == 50


def test_mut_checkpoint_window_is_never_used_for_slowdown() -> None:
    gate = policy.RobustProtectedThroughputGate(
        _manifest(),
        _baseline(),
        sample_reader=_reader(
            [
                _row(0, 0, phase="CHECKPOINT_WRITE", checkpoint=True),
                _row(
                    1_200,
                    10,
                    phase="CHECKPOINT_WRITE",
                    checkpoint=True,
                    io_pressure=True,
                ),
            ]
        ),
    )
    assert gate.sample()["status"] == "PASS"
    assert gate.sample()["status"] == "PASS"
    receipt = gate.receipt()
    assert receipt["checked_windows"] == []
    assert receipt["excluded_windows"][0]["reason"] == "NON_COMPARABLE_PHASE"


def test_mut_active_without_step_is_materialization_not_slowdown() -> None:
    gate = policy.RobustProtectedThroughputGate(
        _manifest(),
        _baseline(),
        sample_reader=_reader(
            [
                _row(0, 100, cpu_ticks=1, output_bytes=10),
                _row(
                    1_200,
                    100,
                    cpu_ticks=100,
                    output_bytes=1_000,
                    io_pressure=True,
                ),
            ]
        ),
    )
    gate.sample()
    assert gate.sample()["status"] == "PASS"
    assert gate.receipt()["excluded_windows"][0]["reason"] == (
        "ACTIVE_CHECKPOINT_OR_MATERIALIZATION"
    )


def test_mut_slowdown_without_resource_contention_does_not_pause() -> None:
    gate = policy.RobustProtectedThroughputGate(
        _manifest(),
        _baseline(),
        sample_reader=_reader([_row(0, 0), _row(1_200, 900)]),
    )
    gate.sample()
    assert gate.sample()["status"] == "PASS"
    checked = gate.receipt()["checked_windows"][0]
    assert checked["slowdown_fraction"] == pytest.approx(0.25)
    assert checked["memory_or_io_contention"] is False
    assert checked["actionable_pause"] is False


def test_mut_sustained_slowdown_with_contention_pauses() -> None:
    gate = policy.RobustProtectedThroughputGate(
        _manifest(),
        _baseline(),
        sample_reader=_reader(
            [_row(0, 0), _row(1_200, 900, io_pressure=True)]
        ),
    )
    gate.sample()
    sample = gate.sample()
    assert sample["status"] == "FAIL"
    assert sample["failures"] == [
        "protected_slowdown_gt_15_percent_with_contention:taste_t14"
    ]
    assert gate.receipt()["status"] == "FAIL"


def test_mut_no_parallel_trace_arms_and_throttled_launcher() -> None:
    project = Path(__file__).resolve().parents[2]
    worker = (project / "scripts/autodl/run_mut_trace_on_adoption_worker.py").read_text(
        encoding="utf-8"
    )
    launcher = (
        project / "scripts/autodl/launch_mut_throttled_continuation_v1.sh"
    ).read_text(encoding="utf-8")
    assert "arms_sequential" in worker
    assert 'nice -n 10' in launcher
    assert 'ionice -c 2 -n 7' in launcher
    assert 'taskset -c "$MUT_CPUSET"' in launcher
    assert "--throttle-profile robust-v2" in launcher
    assert "MUT_PROTECTED_BASELINE_MAX_WAIT_SECONDS=1800" in launcher
    assert "MUT_CPU_WORKERS=2" in launcher
    assert "MUT_PREFETCH=1" in launcher
    assert "RUN_GNN_ABLATION=0" in launcher
    assert "fresh_50k_launched=false" in launcher

