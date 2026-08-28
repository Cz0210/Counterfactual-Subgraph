from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import Any
import uuid

import pytest

from src.utils.managed_execution_v2 import (
    ManagedExecutionV2Error,
    create_checkpoint_directory,
    create_managed_attempt,
    create_worker_staging,
    write_worker_metadata,
)
from src.utils.process_identity_v2 import (
    ProcessSnapshotV2,
    audit_process_lineage,
    register_process_lineage,
    stable_json_sha256,
)


def _snapshot(
    *,
    pid: int,
    ppid: int,
    start: int,
    command: tuple[str, ...] = ("/usr/bin/python3", "worker.py"),
) -> ProcessSnapshotV2:
    return ProcessSnapshotV2(
        pid=pid,
        ppid=ppid,
        pid_start_ticks=start,
        boot_id="boot-generation-1",
        executable_realpath=command[0],
        command=command,
        command_hash=stable_json_sha256(list(command)),
        cwd_realpath="/fixed/project",
        cgroup_path="0::/managed/test",
    )


def _create_attempt(tmp_path: Path, *, attempt_id: str | None = None):
    return create_managed_attempt(
        stage_root=tmp_path,
        controller_id="controller-v2",
        task_id="task-v2",
        git_commit="1" * 40,
        config_hash="2" * 64,
        input_hashes={"input": "3" * 64},
        attempt_id=attempt_id,
        boot_id="boot-generation-1",
    )


def test_managed_attempt_uuid_never_reused(tmp_path: Path) -> None:
    attempt_id = str(uuid.uuid4())
    first = _create_attempt(tmp_path, attempt_id=attempt_id)
    first_path = first.attempt_path
    first.close()
    assert first_path.is_dir()
    with pytest.raises(ManagedExecutionV2Error, match="never be reused"):
        _create_attempt(tmp_path, attempt_id=attempt_id)


def test_managed_checkpoint_uuid_never_reused(tmp_path: Path) -> None:
    attempt = _create_attempt(tmp_path)
    checkpoint_id = str(uuid.uuid4())
    try:
        checkpoint = create_checkpoint_directory(
            attempt, checkpoint_id=checkpoint_id
        )
        checkpoint.close()
        with pytest.raises(ManagedExecutionV2Error, match="never be reused"):
            create_checkpoint_directory(attempt, checkpoint_id=checkpoint_id)
    finally:
        attempt.close()


def test_managed_launcher_exec_worker() -> None:
    attempt_id = str(uuid.uuid4())
    launcher = _snapshot(pid=101, ppid=1, start=900)
    worker_command = ("/usr/bin/python3", "scientific.py")
    worker = replace(
        launcher,
        executable_realpath=worker_command[0],
        command=worker_command,
        command_hash=stable_json_sha256(list(worker_command)),
    )
    lineage = register_process_lineage(
        controller_id="controller-v2",
        attempt_id=attempt_id,
        launcher=launcher,
        worker=worker,
        registered_at="2026-08-28T00:00:00+00:00",
    )
    assert lineage.relationship == "LAUNCHER_EXEC_WORKER"
    assert lineage.to_dict()["launcher_pid_start_ticks"] == 900
    assert lineage.to_dict()["worker_pid_start_ticks"] == 900


def test_managed_legitimate_reparenting() -> None:
    attempt_id = str(uuid.uuid4())
    launcher = _snapshot(pid=101, ppid=1, start=900)
    worker = _snapshot(pid=202, ppid=101, start=901)
    lineage = register_process_lineage(
        controller_id="controller-v2",
        attempt_id=attempt_id,
        launcher=launcher,
        worker=worker,
        registered_at="2026-08-28T00:00:00+00:00",
    )
    observed = replace(worker, ppid=1)
    audit = audit_process_lineage(
        lineage,
        observed_worker=observed,
        launcher_alive=False,
        last_heartbeat="2026-08-28T00:00:01+00:00",
        output_root="/output",
        observed_at="2026-08-28T00:00:02+00:00",
    )
    assert audit["state"] == "RUNNING_LEGITIMATELY_REPARENTED"
    assert audit["science_adopted"] is False


def test_managed_identity_drift_quarantines() -> None:
    attempt_id = str(uuid.uuid4())
    launcher = _snapshot(pid=101, ppid=1, start=900)
    worker = _snapshot(pid=202, ppid=101, start=901)
    lineage = register_process_lineage(
        controller_id="controller-v2",
        attempt_id=attempt_id,
        launcher=launcher,
        worker=worker,
        registered_at="2026-08-28T00:00:00+00:00",
    )
    hostile_command = ("/bin/sh", "hostile")
    observed = replace(
        worker,
        executable_realpath="/bin/sh",
        command=hostile_command,
        command_hash=stable_json_sha256(list(hostile_command)),
    )
    audit = audit_process_lineage(
        lineage,
        observed_worker=observed,
        launcher_alive=True,
        last_heartbeat="2026-08-28T00:00:01+00:00",
        output_root="/output",
        observed_at="2026-08-28T00:00:02+00:00",
    )
    assert audit["state"] == "QUARANTINED"
    assert audit["quarantine_reason"] == "WORKER_PROCESS_IDENTITY_DRIFT"
    assert audit["science_adopted"] is False
    assert audit["downstream_released"] is False
    assert audit["manual_review_required"] is True


@pytest.mark.parametrize("signal_name", ["SIGTERM", "SIGKILL"])
def test_managed_no_signal_termination(
    monkeypatch: pytest.MonkeyPatch, signal_name: str
) -> None:
    calls: list[tuple[Any, ...]] = []

    def forbidden_signal(*args: Any, **kwargs: Any) -> None:
        calls.append((*args, kwargs))
        raise AssertionError(f"unexpected {signal_name}")

    monkeypatch.setattr(os, "kill", forbidden_signal)
    launcher = _snapshot(pid=101, ppid=1, start=900)
    worker = _snapshot(pid=202, ppid=101, start=901)
    lineage = register_process_lineage(
        controller_id="controller-v2",
        attempt_id=str(uuid.uuid4()),
        launcher=launcher,
        worker=worker,
        registered_at="2026-08-28T00:00:00+00:00",
    )
    audit_process_lineage(
        lineage,
        observed_worker=replace(worker, pid_start_ticks=999),
        launcher_alive=True,
        last_heartbeat=None,
        output_root="/output",
        observed_at="2026-08-28T00:00:02+00:00",
    )
    assert calls == []


def test_managed_no_auto_sigterm(monkeypatch: pytest.MonkeyPatch) -> None:
    test_managed_no_signal_termination(monkeypatch, "SIGTERM")


def test_managed_no_auto_sigkill(monkeypatch: pytest.MonkeyPatch) -> None:
    test_managed_no_signal_termination(monkeypatch, "SIGKILL")


def test_worker_cannot_write_verifier_outputs(tmp_path: Path) -> None:
    attempt = _create_attempt(tmp_path)
    staging = create_worker_staging(attempt)
    try:
        for name in (
            "PASS",
            "FAILED",
            "gate.json",
            "verification.json",
            "adoption_receipt.json",
        ):
            with pytest.raises(ManagedExecutionV2Error, match="worker may write"):
                write_worker_metadata(staging, name=name, payload={"status": "PASS"})
    finally:
        staging.close()
        attempt.close()
