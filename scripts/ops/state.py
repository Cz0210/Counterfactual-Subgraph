"""Persistent experiment run state and append-only events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import socket
import tempfile
from typing import Any
from uuid import uuid4


class RunStatus(str, Enum):
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    DRY_RUN_COMPLETED = "DRY_RUN_COMPLETED"
    ADOPT_EXISTING_DRY_RUN = "ADOPT_EXISTING_DRY_RUN"
    LOCAL_PREFLIGHT = "LOCAL_PREFLIGHT"
    LOCAL_GATE_RUNNING = "LOCAL_GATE_RUNNING"
    LOCAL_GATE_PASSED = "LOCAL_GATE_PASSED"
    LOCAL_GATE_FAILED = "LOCAL_GATE_FAILED"
    COMMITTING = "COMMITTING"
    PUSHED = "PUSHED"
    REMOTE_SYNCING = "REMOTE_SYNCING"
    REMOTE_PREFLIGHT = "REMOTE_PREFLIGHT"
    REMOTE_PREFLIGHT_RUNNING = "REMOTE_PREFLIGHT_RUNNING"
    REMOTE_PREFLIGHT_PASSED = "REMOTE_PREFLIGHT_PASSED"
    REMOTE_PREFLIGHT_PASSED_WITH_WARNINGS = (
        "REMOTE_PREFLIGHT_PASSED_WITH_WARNINGS"
    )
    REMOTE_PREFLIGHT_BLOCKED = "REMOTE_PREFLIGHT_BLOCKED"
    NEEDS_DEPLOY = "NEEDS_DEPLOY"
    NEEDS_PROXY_SETUP = "NEEDS_PROXY_SETUP"
    ADOPT_EXISTING_VERIFYING = "ADOPT_EXISTING_VERIFYING"
    ADOPTED_EXISTING = "ADOPTED_EXISTING"
    STOPPED_BEFORE_APPROVAL = "STOPPED_BEFORE_APPROVAL"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    AUDITING = "AUDITING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


TERMINAL_STATUSES = {
    RunStatus.DRY_RUN_COMPLETED.value,
    RunStatus.ADOPT_EXISTING_DRY_RUN.value,
    RunStatus.REMOTE_PREFLIGHT_PASSED.value,
    RunStatus.REMOTE_PREFLIGHT_PASSED_WITH_WARNINGS.value,
    RunStatus.REMOTE_PREFLIGHT_BLOCKED.value,
    RunStatus.NEEDS_DEPLOY.value,
    RunStatus.NEEDS_PROXY_SETUP.value,
    RunStatus.STOPPED_BEFORE_APPROVAL.value,
    RunStatus.BLOCKED.value,
    RunStatus.FAILED.value,
    RunStatus.COMPLETED.value,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid4().hex[:8]}"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_jsonl_fsync(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(slots=True)
class RunStore:
    run_dir: Path

    @classmethod
    def create(
        cls,
        reports_root: Path,
        task_id: str,
        run_id: str | None = None,
        spec_path: str | None = None,
    ) -> "RunStore":
        resolved = reports_root.expanduser().resolve()
        selected_run_id = run_id or make_run_id()
        return cls.create_at(
            resolved / task_id / selected_run_id,
            task_id=task_id,
            run_id=selected_run_id,
            spec_path=spec_path,
        )

    @classmethod
    def create_at(
        cls,
        run_dir: str | Path,
        *,
        task_id: str,
        run_id: str | None = None,
        spec_path: str | None = None,
    ) -> "RunStore":
        resolved_run_dir = Path(run_dir).expanduser().resolve()
        selected_run_id = run_id or resolved_run_dir.name or make_run_id()
        if resolved_run_dir.exists():
            if not resolved_run_dir.is_dir():
                raise FileExistsError(
                    f"Run path is not a directory: {resolved_run_dir}"
                )
            if any(resolved_run_dir.iterdir()):
                raise FileExistsError(
                    f"Run directory is not empty: {resolved_run_dir}"
                )
        else:
            resolved_run_dir.mkdir(parents=True, exist_ok=False)
        store = cls(run_dir=resolved_run_dir)
        state = {
            "schema_version": 1,
            "task_id": task_id,
            "run_id": selected_run_id,
            "status": RunStatus.CREATED.value,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "spec_path": spec_path or "",
            "local_commit": None,
            "remote_commit": None,
            "slurm_jobs": [],
            "adopted_stages": [],
            "next_stage": None,
            "remote_write_performed": False,
            "stages": {},
            "approvals": {},
            "stop_reason": None,
        }
        atomic_write_json(store.state_path, state)
        with store.commands_path.open("a", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        store.append_event("run_created", status=RunStatus.CREATED.value)
        return store

    @classmethod
    def open(cls, run_dir: str | Path) -> "RunStore":
        resolved = Path(run_dir).expanduser().resolve()
        store = cls(run_dir=resolved)
        if not store.state_path.is_file() or not store.events_path.is_file():
            raise FileNotFoundError(
                f"Run state is incomplete under {resolved}"
            )
        store.validate_event_history()
        return store

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def events_path(self) -> Path:
        return self.run_dir / "events.jsonl"

    @property
    def commands_path(self) -> Path:
        return self.run_dir / "commands.jsonl"

    def load(self) -> dict[str, Any]:
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid state object: {self.state_path}")
        # State schema v1 predates Slurm job persistence. Keep old run
        # checkpoints resumable while presenting one stable list contract.
        payload.setdefault("slurm_jobs", [])
        payload.setdefault("adopted_stages", [])
        payload.setdefault("next_stage", None)
        payload.setdefault("remote_write_performed", False)
        return payload

    def save(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        atomic_write_json(self.state_path, state)

    def append_event(
        self, event_type: str, **details: Any
    ) -> dict[str, Any]:
        state = self.load()
        event = {
            "schema_version": 1,
            "sequence": self.event_count() + 1,
            "timestamp": utc_now(),
            "task_id": state["task_id"],
            "run_id": state["run_id"],
            "event_type": event_type,
            **details,
        }
        append_jsonl_fsync(self.events_path, event)
        return event

    def event_count(self) -> int:
        if not self.events_path.exists():
            return 0
        with self.events_path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def transition(
        self, status: RunStatus | str, *, reason: str | None = None
    ) -> None:
        value = status.value if isinstance(status, RunStatus) else str(status)
        if value not in {item.value for item in RunStatus}:
            raise ValueError(f"Unknown run status: {value}")
        state = self.load()
        previous = state["status"]
        state["status"] = value
        self.save(state)
        self.append_event(
            "state_transition",
            previous_status=previous,
            status=value,
            reason=reason,
        )

    def record_stage(self, stage_id: str, record: dict[str, Any]) -> None:
        state = self.load()
        state["stages"][stage_id] = dict(record)
        self.save(state)
        self.append_event(
            "stage_recorded",
            stage_id=stage_id,
            stage_status=record.get("status"),
            attempt=record.get("attempt"),
        )

    def stage_succeeded(self, stage_id: str) -> bool:
        stage = self.load().get("stages", {}).get(stage_id, {})
        return stage.get("status") in {
            "PASSED",
            "ADOPTED_EXISTING",
            "APPROVED",
        }

    def approve(
        self,
        stage_id: str,
        reason: str,
        username: str,
        *,
        git_commit: str | None = None,
    ) -> None:
        if not reason.strip():
            raise ValueError("Approval reason must not be empty.")
        state = self.load()
        approval = {
            "timestamp": utc_now(),
            "username": username,
            "hostname": socket.gethostname(),
            "stage_id": stage_id,
            "reason": reason.strip(),
            "git_commit": git_commit,
        }
        state["approvals"][stage_id] = approval
        self.save(state)
        self.append_event("stage_approved", **approval)

    def is_approved(self, stage_id: str) -> bool:
        return stage_id in self.load().get("approvals", {})

    def record_slurm_job(self, record: dict[str, Any]) -> None:
        job_id = str(record.get("job_id") or "")
        if not job_id.isdigit():
            raise ValueError(f"Slurm job_id must be numeric: {job_id!r}")
        state = self.load()
        jobs = list(state.get("slurm_jobs") or [])
        matching = [
            index
            for index, item in enumerate(jobs)
            if str(item.get("job_id")) == job_id
        ]
        if len(matching) > 1:
            raise ValueError(f"Duplicate persisted Slurm job ID: {job_id}")
        if matching:
            jobs[matching[0]] = dict(record)
        else:
            jobs.append(dict(record))
        state["slurm_jobs"] = jobs
        self.save(state)
        self.append_event(
            "slurm_job_recorded",
            stage_id=record.get("stage_id"),
            job_id=job_id,
            slurm_state=record.get("slurm_state"),
            exit_code=record.get("exit_code"),
        )

    def append_command(self, record: dict[str, Any]) -> None:
        append_jsonl_fsync(self.commands_path, record)

    def validate_event_history(self) -> None:
        expected = 1
        with self.events_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("sequence") != expected:
                    raise ValueError(
                        "Event sequence corruption at "
                        f"{self.events_path}:{line_number}; "
                        f"expected={expected}, found={payload.get('sequence')}"
                    )
                expected += 1
