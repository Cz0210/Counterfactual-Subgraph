"""Progress/ETA health states for non-owning AutoDL continuation monitors.

These helpers never signal a scientific process.  They turn an observed
counter and a PID identity into an auditable health classification so that a
live but computationally unviable route is not confused with useful progress.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Mapping


RUNNING_PROGRESSING = "RUNNING_PROGRESSING"
RUNNING_SLOW = "RUNNING_SLOW"
RUNNING_UNVIABLE = "RUNNING_UNVIABLE"
RUNNING_STALLED = "RUNNING_STALLED"
SUPERSEDED = "SUPERSEDED"
PROCESS_EXITED = "PROCESS_EXITED"
OBSERVATION_FAILED = "OBSERVATION_FAILED"
HEALTH_STATES = frozenset(
    {
        RUNNING_PROGRESSING,
        RUNNING_SLOW,
        RUNNING_UNVIABLE,
        RUNNING_STALLED,
        SUPERSEDED,
        PROCESS_EXITED,
        OBSERVATION_FAILED,
    }
)

ROUTE_VIABLE = "VIABLE"
ROUTE_SLOW = "SLOW"
ROUTE_UNVIABLE = "UNVIABLE"
ROUTE_STALLED = "STALLED"
ROUTE_SUPERSEDED = "SUPERSEDED"
ROUTE_UNKNOWN = "UNKNOWN"
SUPERSESSION_RECEIPT_SCHEMA = "autodl_route_supersession_receipt_v1"
SUPERSESSION_RECEIPT_STATE = "GRACEFUL_CHECKPOINT_AND_STOP_COMPLETE"


def route_viability_for_progress_state(progress_state: str) -> str:
    """Map an observed worker state to a non-terminal route assessment.

    A controller is not allowed to infer scientific PASS from a completed
    counter or a live PID.  In particular, a missing worker is *unknown*, not
    successful, unless an independent finalizer says otherwise.
    """

    values = {
        RUNNING_PROGRESSING: ROUTE_VIABLE,
        RUNNING_SLOW: ROUTE_SLOW,
        RUNNING_UNVIABLE: ROUTE_UNVIABLE,
        RUNNING_STALLED: ROUTE_STALLED,
        SUPERSEDED: ROUTE_SUPERSEDED,
        PROCESS_EXITED: ROUTE_UNKNOWN,
        OBSERVATION_FAILED: ROUTE_UNKNOWN,
    }
    try:
        return values[progress_state]
    except KeyError as exc:
        raise ValueError(f"Unknown scientific progress state: {progress_state!r}") from exc


def _timestamp(value: str | None = None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Progress timestamps must include a timezone.")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class ProgressPolicy:
    stalled_after_seconds: float = 30 * 60
    slow_eta_hours: float = 24.0
    unviable_eta_hours: float = 7 * 24.0

    def validate(self) -> None:
        values = (
            self.stalled_after_seconds,
            self.slow_eta_hours,
            self.unviable_eta_hours,
        )
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Progress-health limits must be finite and positive.")
        if self.unviable_eta_hours <= self.slow_eta_hours:
            raise ValueError("Unviable ETA must be greater than slow ETA.")


def update_progress_health(
    previous: Mapping[str, Any] | None,
    *,
    completed: int,
    total: int,
    pid_alive: bool,
    observed_at: str,
    policy: ProgressPolicy,
) -> dict[str, Any]:
    """Update one task without ever treating liveness as scientific PASS."""

    policy.validate()
    if isinstance(completed, bool) or isinstance(total, bool):
        raise ValueError("Progress counters cannot be booleans.")
    completed = int(completed)
    total = int(total)
    if total <= 0 or completed < 0 or completed > total:
        raise ValueError("Progress counters are outside [0, total].")
    now = _timestamp(observed_at)
    prior = dict(previous or {})
    prior_completed = int(prior.get("completed", completed))
    prior_observed = _timestamp(prior.get("observed_at", observed_at))
    prior_progress = _timestamp(prior.get("last_progress_at", observed_at))
    elapsed = max(0.0, (now - prior_observed).total_seconds())
    delta = completed - prior_completed
    if delta < 0:
        raise ValueError("Observed progress regressed.")
    if delta > 0:
        last_progress = now
        instant_rate = delta / elapsed if elapsed > 0 else None
    else:
        last_progress = prior_progress
        instant_rate = None
    previous_rate = prior.get("rolling_throughput_per_second")
    if instant_rate is None:
        rolling_rate = (
            float(previous_rate)
            if previous_rate is not None and float(previous_rate) >= 0
            else 0.0
        )
    elif previous_rate is None:
        rolling_rate = float(instant_rate)
    else:
        # Stable enough for an ETA while still responding to a phase change.
        rolling_rate = 0.5 * float(previous_rate) + 0.5 * float(instant_rate)
    stalled_seconds = max(0.0, (now - last_progress).total_seconds())
    eta_hours = (
        (total - completed) / rolling_rate / 3600.0
        if rolling_rate > 0 and completed < total
        else (0.0 if completed == total else None)
    )
    if not pid_alive:
        state = PROCESS_EXITED
    elif completed < total and stalled_seconds >= policy.stalled_after_seconds:
        state = RUNNING_STALLED
    elif eta_hours is not None and eta_hours > policy.unviable_eta_hours:
        state = RUNNING_UNVIABLE
    elif eta_hours is None or eta_hours > policy.slow_eta_hours:
        state = RUNNING_SLOW
    else:
        state = RUNNING_PROGRESSING
    return {
        "health_state": state,
        # ``health_state`` is retained for existing consumers.  The explicit
        # name prevents a live monitor from being mistaken for a live worker.
        "scientific_progress_state": state,
        "route_viability": route_viability_for_progress_state(state),
        "completed": completed,
        "total": total,
        "fraction": completed / total,
        "observed_at": now.isoformat(timespec="seconds"),
        "last_progress_at": last_progress.isoformat(timespec="seconds"),
        "seconds_since_progress": stalled_seconds,
        "rolling_throughput_per_second": rolling_rate,
        "rolling_throughput_per_hour": rolling_rate * 3600.0,
        "eta_hours": eta_hours,
        "pid_alive": bool(pid_alive),
        "scientific_worker_alive": bool(pid_alive),
        "automatic_signal_allowed": False,
    }


def mark_superseded(
    previous: Mapping[str, Any] | None,
    *,
    total: int,
    scientific_worker_alive: bool,
    observed_at: str,
    supersession: Mapping[str, Any],
) -> dict[str, Any]:
    """Record an operator-approved route replacement without signalling it.

    This is deliberately not an action primitive.  The caller must provide an
    immutable receipt for a completed graceful handover/stop.  A superseded
    worker can be already gone (the common case) or still drain briefly; both
    are observable and neither is treated as PASS.
    """

    if isinstance(total, bool) or int(total) <= 0:
        raise ValueError("A superseded task requires a positive total.")
    if supersession.get("schema_version") != SUPERSESSION_RECEIPT_SCHEMA:
        raise ValueError("Supersession evidence has the wrong schema.")
    if supersession.get("state") != SUPERSESSION_RECEIPT_STATE:
        raise ValueError("Supersession evidence has the wrong terminal state.")
    for field in (
        "graceful_checkpoint_completed",
        "graceful_stop_completed",
        "old_worker_exited",
    ):
        if supersession.get(field) is not True:
            raise ValueError(f"Supersession evidence does not prove {field}.")
    if supersession.get("sigkill_used") is not False:
        raise ValueError("Supersession evidence does not prove sigkill_used=false.")
    receipt_sha = str(supersession.get("receipt_sha256") or "")
    if len(receipt_sha) != 64 or any(value not in "0123456789abcdef" for value in receipt_sha):
        raise ValueError("Supersession evidence lacks a valid receipt SHA256.")
    replacement = supersession.get("replacement")
    if not isinstance(replacement, Mapping) or replacement.get("task_gate_state") != "PASS":
        raise ValueError("Supersession evidence does not bind a replacement PASS gate.")
    if scientific_worker_alive:
        raise ValueError("A live scientific worker cannot be marked SUPERSEDED.")
    now = _timestamp(observed_at)
    prior = dict(previous or {})
    completed = int(prior.get("completed", 0))
    if completed < 0 or completed > int(total):
        raise ValueError("Prior progress for a superseded task is outside [0, total].")
    last_progress = _timestamp(prior.get("last_progress_at", observed_at))
    rolling_rate = prior.get("rolling_throughput_per_second")
    if rolling_rate is not None:
        rolling_rate = float(rolling_rate)
        if not math.isfinite(rolling_rate) or rolling_rate < 0:
            rolling_rate = None
    return {
        "health_state": SUPERSEDED,
        "scientific_progress_state": SUPERSEDED,
        "route_viability": ROUTE_SUPERSEDED,
        "completed": completed,
        "total": int(total),
        "fraction": completed / int(total),
        "observed_at": now.isoformat(timespec="seconds"),
        "last_progress_at": last_progress.isoformat(timespec="seconds"),
        "seconds_since_progress": max(0.0, (now - last_progress).total_seconds()),
        "rolling_throughput_per_second": rolling_rate,
        "rolling_throughput_per_hour": (
            None if rolling_rate is None else rolling_rate * 3600.0
        ),
        "eta_hours": None,
        # Kept as aliases for monitor-v1 consumers.
        "pid_alive": bool(scientific_worker_alive),
        "scientific_worker_alive": bool(scientific_worker_alive),
        "automatic_signal_allowed": False,
        "supersession": dict(supersession),
    }


__all__ = [
    "HEALTH_STATES",
    "OBSERVATION_FAILED",
    "PROCESS_EXITED",
    "ProgressPolicy",
    "RUNNING_PROGRESSING",
    "RUNNING_SLOW",
    "RUNNING_STALLED",
    "RUNNING_UNVIABLE",
    "SUPERSEDED",
    "SUPERSESSION_RECEIPT_SCHEMA",
    "SUPERSESSION_RECEIPT_STATE",
    "ROUTE_SLOW",
    "ROUTE_STALLED",
    "ROUTE_SUPERSEDED",
    "ROUTE_UNKNOWN",
    "ROUTE_UNVIABLE",
    "ROUTE_VIABLE",
    "mark_superseded",
    "route_viability_for_progress_state",
    "update_progress_health",
]
