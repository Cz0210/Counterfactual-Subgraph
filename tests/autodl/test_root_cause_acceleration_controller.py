from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.utils.autodl_progress_health import (
    PROCESS_EXITED,
    ProgressPolicy,
    RUNNING_PROGRESSING,
    RUNNING_SLOW,
    RUNNING_STALLED,
    RUNNING_UNVIABLE,
    update_progress_health,
)


def _time(minutes: int) -> str:
    return (datetime(2026, 8, 24, tzinfo=timezone.utc) + timedelta(minutes=minutes)).isoformat()


def test_progress_health_distinguishes_slow_unviable_and_stalled() -> None:
    policy = ProgressPolicy(stalled_after_seconds=1200, slow_eta_hours=24, unviable_eta_hours=168)
    first = update_progress_health(None, completed=10, total=100, pid_alive=True, observed_at=_time(0), policy=policy)
    assert first["health_state"] == RUNNING_SLOW
    progressing = update_progress_health(first, completed=50, total=100, pid_alive=True, observed_at=_time(10), policy=policy)
    assert progressing["health_state"] == RUNNING_PROGRESSING
    unviable = update_progress_health(first, completed=11, total=1_000_000, pid_alive=True, observed_at=_time(10), policy=policy)
    assert unviable["health_state"] == RUNNING_UNVIABLE
    stalled = update_progress_health(first, completed=10, total=100, pid_alive=True, observed_at=_time(21), policy=policy)
    assert stalled["health_state"] == RUNNING_STALLED
    assert stalled["automatic_signal_allowed"] is False


def test_dead_pid_is_observed_but_never_called_pass() -> None:
    result = update_progress_health(None, completed=100, total=100, pid_alive=False, observed_at=_time(0), policy=ProgressPolicy())
    assert result["health_state"] == PROCESS_EXITED
    assert result["automatic_signal_allowed"] is False


def test_progress_regression_fails_closed() -> None:
    first = update_progress_health(None, completed=10, total=100, pid_alive=True, observed_at=_time(0), policy=ProgressPolicy())
    with pytest.raises(ValueError, match="regressed"):
        update_progress_health(first, completed=9, total=100, pid_alive=True, observed_at=_time(1), policy=ProgressPolicy())
