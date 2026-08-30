"""Persistent read-only observer for the authorized BACE ComRecGC cap.

The observer publishes an exact handover recommendation from convergence
receipts.  It deliberately cannot send signals or launch post-processing; the
separate handover remains fail-closed until an execution path with explicit
signal authority consumes the request.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

from src.eval.bace_comrecgc_resource_cap import (
    BaceComRecGCResourceCapError,
    decide_bace_comrecgc_resource_cap,
)
from src.utils.autodl_runtime import atomic_write_json, sha256_file


STATE_SCHEMA = "bace_comrecgc_resource_cap_observer_v1"
REQUEST_SCHEMA = "bace_comrecgc_resource_cap_handover_request_v1"


class ResourceCapObserverError(RuntimeError):
    """The observer cannot safely interpret its immutable inputs."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _absolute(value: str | Path, *, label: str, existing: bool = False) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ResourceCapObserverError(f"{label} must be absolute")
    try:
        return path.resolve(strict=existing)
    except FileNotFoundError as exc:
        raise ResourceCapObserverError(f"{label} does not exist: {path}") from exc


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResourceCapObserverError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResourceCapObserverError(f"expected one JSON object: {path}")
    return dict(value)


def _process_generation_matches(pid: int, start_ticks: int) -> bool:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        closing = raw.rfind(")")
        fields = raw[closing + 2 :].split()
        return int(fields[19]) == start_ticks and fields[0] != "Z"
    except (IndexError, OSError, ValueError):
        return False


class ResourceCapObserver:
    """Materialize cap decisions without owning a science process."""

    def __init__(
        self,
        *,
        convergence_hook_root: str | Path,
        state_root: str | Path,
        science_pid: int,
        science_start_ticks: int,
        poll_seconds: int = 60,
    ) -> None:
        if science_pid <= 0 or science_start_ticks <= 0:
            raise ResourceCapObserverError("science PID/start ticks must be positive")
        if poll_seconds <= 0:
            raise ResourceCapObserverError("poll_seconds must be positive")
        self.hook_root = _absolute(
            convergence_hook_root, label="convergence_hook_root"
        )
        self.state_root = _absolute(state_root, label="state_root")
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.science_pid = int(science_pid)
        self.science_start_ticks = int(science_start_ticks)
        self.poll_seconds = int(poll_seconds)
        self.sequence = 0

    def _audit_rows(self) -> list[tuple[Path, dict[str, Any]]]:
        if not self.hook_root.is_dir():
            return []
        rows: list[tuple[Path, dict[str, Any]]] = []
        for path in sorted(self.hook_root.glob("step-*.json")):
            wrapper = _read_object(path)
            result = wrapper.get("result", wrapper)
            if isinstance(result, Mapping):
                rows.append((path, dict(result)))
        return rows

    def tick(self) -> dict[str, Any]:
        self.sequence += 1
        decisions: list[dict[str, Any]] = []
        for path, audit in self._audit_rows():
            try:
                decision = decide_bace_comrecgc_resource_cap(audit)
            except BaceComRecGCResourceCapError as exc:
                decision = {
                    "status": "FAIL_CLOSED",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "signals_sent": False,
                    "postprocess_started": False,
                }
            receipt = {
                "schema_version": STATE_SCHEMA,
                "audit_path": str(path.resolve(strict=True)),
                "audit_sha256": sha256_file(path),
                "observed_at": utc_now(),
                **decision,
            }
            atomic_write_json(
                self.state_root / f"step-{audit.get('evaluation_step', 'unknown')}.json",
                receipt,
            )
            decisions.append(receipt)
        eligible = [
            row
            for row in decisions
            if row.get("status")
            in {
                "HANDOVER_ELIGIBLE",
                "SCI_FAILED_ELIGIBLE_FOR_EXACT_GRACEFUL_STOP",
            }
        ]
        selected = (
            min(eligible, key=lambda row: int(row["m_effective"]))
            if eligible
            else (decisions[-1] if decisions else None)
        )
        science_alive = _process_generation_matches(
            self.science_pid, self.science_start_ticks
        )
        state = {
            "schema_version": STATE_SCHEMA,
            "pid": os.getpid(),
            "sequence": self.sequence,
            "heartbeat_at": utc_now(),
            "science_pid": self.science_pid,
            "science_start_ticks": self.science_start_ticks,
            "science_generation_alive": science_alive,
            "latest_decision": selected,
            "signals_sent": False,
            "postprocess_started": False,
            "execution_authority": "READ_ONLY_OBSERVER",
            "state": (
                str(selected["status"])
                if selected is not None
                else "WAITING_COMMITTED_AUDIT"
            ),
        }
        atomic_write_json(self.state_root / "heartbeat.json", state)
        atomic_write_json(self.state_root / "state.json", state)
        if selected is not None and selected.get("status") in {
            "HANDOVER_ELIGIBLE",
            "SCI_FAILED_ELIGIBLE_FOR_EXACT_GRACEFUL_STOP",
        }:
            request = {
                "schema_version": REQUEST_SCHEMA,
                "status": selected["status"],
                "m_effective": selected["m_effective"],
                "reason": selected["reason"],
                "valid_unique_count": selected["valid_unique_count"],
                "lineage_error_count": selected["lineage_error_count"],
                "checkpoint_digest": selected["checkpoint_digest"],
                "science_pid": self.science_pid,
                "science_start_ticks": self.science_start_ticks,
                "science_generation_alive": science_alive,
                "requested_signal": "SIGTERM_EXACT_PID_ONLY",
                "signal_sent": False,
                "postprocess_started": False,
                "manual_or_separately_authorized_executor_required": True,
                "created_at": utc_now(),
            }
            request_path = self.state_root / "handover_request.json"
            if request_path.is_file():
                prior = _read_object(request_path)
                stable_keys = (
                    "status",
                    "m_effective",
                    "reason",
                    "valid_unique_count",
                    "lineage_error_count",
                    "checkpoint_digest",
                    "science_pid",
                    "science_start_ticks",
                )
                if any(prior.get(key) != request.get(key) for key in stable_keys):
                    raise ResourceCapObserverError(
                        "existing handover request differs from earliest eligible decision"
                    )
            else:
                atomic_write_json(request_path, request)
        return state

    def run(self, *, once: bool = False) -> int:
        while True:
            self.tick()
            if once:
                return 0
            time.sleep(self.poll_seconds)


__all__ = ["ResourceCapObserver", "ResourceCapObserverError"]
