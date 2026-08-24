from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.autodl import run_root_cause_acceleration_controller as controller
from scripts.autodl.status_root_cause_acceleration import render_table
from src.utils.autodl_progress_health import (
    OBSERVATION_FAILED,
    PROCESS_EXITED,
    ProgressPolicy,
    ROUTE_SUPERSEDED,
    RUNNING_PROGRESSING,
    RUNNING_SLOW,
    RUNNING_STALLED,
    RUNNING_UNVIABLE,
    SUPERSEDED,
    mark_superseded,
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


def test_progress_health_exports_scientific_worker_and_route_fields() -> None:
    result = update_progress_health(
        None,
        completed=10,
        total=100,
        pid_alive=True,
        observed_at=_time(0),
        policy=ProgressPolicy(),
    )
    assert result["scientific_progress_state"] == result["health_state"]
    assert result["scientific_worker_alive"] is True
    assert result["route_viability"] == "SLOW"
    # The legacy field remains available for monitor-v1 consumers.
    assert result["pid_alive"] is True


def test_superseded_route_requires_external_receipt_and_is_never_pass() -> None:
    result = mark_superseded(
        {"completed": 12, "last_progress_at": _time(1)},
        total=100,
        scientific_worker_alive=False,
        observed_at=_time(2),
        supersession={
            "schema_version": controller.SUPERSESSION_RECEIPT_SCHEMA,
            "state": controller.SUPERSESSION_RECEIPT_STATE,
            "reason": "replacement passed",
            "receipt_sha256": "a" * 64,
            "graceful_checkpoint_completed": True,
            "graceful_stop_completed": True,
            "old_worker_exited": True,
            "sigkill_used": False,
            "replacement": {"task_gate_state": "PASS"},
        },
    )
    assert result["health_state"] == SUPERSEDED
    assert result["scientific_progress_state"] == SUPERSEDED
    assert result["route_viability"] == ROUTE_SUPERSEDED
    assert result["scientific_worker_alive"] is False
    assert result["automatic_signal_allowed"] is False


def _identity(
    pid: int, *, command: str = "legacy-science", state: str = "S"
) -> dict[str, object]:
    return {
        "pid": pid,
        "start_ticks": 42,
        "command": command,
        "state": state,
        "ppid": 1,
        "utime_ticks": 0,
        "stime_ticks": 0,
        "rss_bytes": 0,
        "read_bytes": 0,
        "write_bytes": 0,
    }


def _task_spec(
    tmp_path: Path, *, supersession: dict[str, object] | None = None
) -> dict[str, object]:
    old_root = tmp_path / "old-output"
    old_root.mkdir(exist_ok=True)
    task: dict[str, object] = {
        "task_id": "legacy",
        "ownership": "external_read_only",
        "pid": 12345,
        "start_ticks": 42,
        "command_contains": "legacy-science",
        "output_root": str(old_root),
        "total": 100,
        "progress": {"kind": "json", "path": str(tmp_path / "unused.json"), "pointer": ["done"]},
    }
    if supersession is not None:
        task["supersession"] = supersession
    return {"controller_id": "root-cause-test", "tasks": [task]}


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _supersession_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], Path, dict[str, object]]:
    replacement_root = tmp_path / "replacement-output"
    replacement_root.mkdir()
    controller_manifest = tmp_path / "replacement-controller-manifest.json"
    task_gate = tmp_path / "replacement-task-gate.json"
    final_manifest = replacement_root / "final-manifest.json"
    controller_sha = _write_json(
        controller_manifest, {"controller_id": "replacement-controller"}
    )
    gate_sha = _write_json(
        task_gate, {"task_id": "replacement-task", "status": "PASS"}
    )
    final_sha = _write_json(final_manifest, {"status": "PASS"})
    replacement_receipt = {
        "controller_id": "replacement-controller",
        "task_id": "replacement-task",
        "output_root": str(replacement_root),
        "controller_manifest_path": str(controller_manifest),
        "controller_manifest_sha256": controller_sha,
        "task_gate_path": str(task_gate),
        "task_gate_sha256": gate_sha,
        "task_gate_status_field": "status",
        "task_gate_state": "PASS",
        "final_manifest_path": str(final_manifest),
        "final_manifest_sha256": final_sha,
    }
    receipt_payload: dict[str, object] = {
        "schema_version": controller.SUPERSESSION_RECEIPT_SCHEMA,
        "state": controller.SUPERSESSION_RECEIPT_STATE,
        "reason": "exact paper-faithful route passed",
        "graceful_checkpoint_completed": True,
        "graceful_stop_completed": True,
        "old_worker_exited": True,
        "sigkill_used": False,
        "old_task": {
            "task_id": "legacy",
            "pid": 12345,
            "start_ticks": 42,
            "output_root": str(tmp_path / "old-output"),
        },
        "replacement": replacement_receipt,
    }
    receipt = tmp_path / "graceful-handover.json"
    receipt_sha = _write_json(receipt, receipt_payload)
    supersession: dict[str, object] = {
        "reason": "exact paper-faithful route passed",
        "receipt_path": str(receipt),
        "expected_receipt_sha256": receipt_sha,
        "replacement": {
            "controller_id": replacement_receipt["controller_id"],
            "task_id": replacement_receipt["task_id"],
            "output_root": replacement_receipt["output_root"],
            "controller_manifest_path": replacement_receipt[
                "controller_manifest_path"
            ],
            "expected_controller_manifest_sha256": controller_sha,
            "task_gate_path": replacement_receipt["task_gate_path"],
            "expected_task_gate_sha256": gate_sha,
            "task_gate_status_field": "status",
            "final_manifest_path": replacement_receipt["final_manifest_path"],
            "expected_final_manifest_sha256": final_sha,
        },
    }
    return supersession, receipt, receipt_payload


def _observe_supersession(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    supersession: dict[str, object],
    *,
    task_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    monkeypatch.setattr(controller, "_gpu_snapshot", lambda: {})
    monkeypatch.setattr(
        controller,
        "_proc_identity",
        lambda pid: task_identity if pid == 12345 else _identity(pid, command="monitor"),
    )
    monkeypatch.setattr(
        controller,
        "_progress",
        lambda _task: (_ for _ in ()).throw(AssertionError("unexpected progress probe")),
    )
    monkeypatch.setattr(controller, "utc_now", lambda: _time(2))
    payload = controller.run_once(
        spec=_task_spec(tmp_path, supersession=supersession),
        root=tmp_path / "control",
        spec_sha256="e" * 64,
    )
    return payload["tasks"]["legacy"]


def test_run_once_separates_controller_from_scientific_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(controller, "_gpu_snapshot", lambda: {})
    monkeypatch.setattr(
        controller,
        "_proc_identity",
        lambda pid: _identity(pid, command="legacy-science" if pid == 12345 else "monitor"),
    )
    monkeypatch.setattr(controller, "_progress", lambda task: 10)
    monkeypatch.setattr(controller, "utc_now", lambda: _time(0))

    payload = controller.run_once(
        spec=_task_spec(tmp_path), root=tmp_path / "control", spec_sha256="b" * 64
    )
    task = payload["tasks"]["legacy"]
    assert payload["controller_pid"] == os.getpid()
    assert payload["controller_process_alive"] is True
    assert task["controller_process_alive"] is True
    assert task["scientific_worker_alive"] is True
    assert task["scientific_progress_state"] == RUNNING_SLOW
    assert task["route_viability"] == "SLOW"
    assert task["automatic_signal_allowed"] is False


def test_monitor_v1_spec_without_supersession_remains_valid(tmp_path: Path) -> None:
    spec = _task_spec(tmp_path)
    spec["schema_version"] = controller.SCHEMA
    path = tmp_path / "monitor-v1.json"
    _write_json(path, spec)
    validated = controller._validate_spec(path)
    assert validated["tasks"][0].get("supersession") is None


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("pid", 12345.0),
        ("start_ticks", 42.0),
        ("pid", True),
        ("start_ticks", False),
    ],
)
def test_monitor_spec_pid_identity_rejects_float_and_bool_as_int(
    tmp_path: Path, field: str, invalid_value: object
) -> None:
    spec = _task_spec(tmp_path)
    spec["schema_version"] = controller.SCHEMA
    tasks = spec["tasks"]
    assert isinstance(tasks, list)
    task = tasks[0]
    assert isinstance(task, dict)
    task[field] = invalid_value
    path = tmp_path / "invalid-monitor-v1.json"
    _write_json(path, spec)
    with pytest.raises(ValueError, match="positive JSON integer"):
        controller._validate_spec(path)


def test_run_once_supersedes_only_with_a_receipt_and_skips_removed_progress_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    supersession, receipt, _ = _supersession_fixture(tmp_path)
    monkeypatch.setattr(controller, "_gpu_snapshot", lambda: {})
    monkeypatch.setattr(controller, "_proc_identity", lambda _pid: None)
    monkeypatch.setattr(
        controller,
        "_progress",
        lambda _task: (_ for _ in ()).throw(AssertionError("must not probe superseded root")),
    )
    monkeypatch.setattr(
        controller.os,
        "kill",
        lambda *_args: (_ for _ in ()).throw(AssertionError("monitor must not signal")),
    )
    monkeypatch.setattr(controller, "utc_now", lambda: _time(2))

    payload = controller.run_once(
        spec=_task_spec(tmp_path, supersession=supersession),
        root=tmp_path / "control",
        spec_sha256="c" * 64,
    )
    task = payload["tasks"]["legacy"]
    assert task["health_state"] == SUPERSEDED
    assert task["route_viability"] == ROUTE_SUPERSEDED
    assert task["scientific_worker_alive"] is False
    assert task["supersession"]["receipt_path"] == str(receipt)
    assert task["supersession"]["graceful_checkpoint_completed"] is True
    assert task["supersession"]["graceful_stop_completed"] is True
    assert task["supersession"]["old_worker_exited"] is True
    assert task["supersession"]["sigkill_used"] is False
    assert task["supersession"]["replacement"]["task_gate_state"] == "PASS"
    assert task["automatic_signal_allowed"] is False


def test_missing_supersession_receipt_fails_closed_without_claiming_superseded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    supersession, _, _ = _supersession_fixture(tmp_path)
    supersession["receipt_path"] = str(tmp_path / "missing-receipt.json")
    monkeypatch.setattr(controller, "_gpu_snapshot", lambda: {})
    monkeypatch.setattr(controller, "_proc_identity", lambda _pid: None)
    monkeypatch.setattr(controller, "utc_now", lambda: _time(2))

    payload = controller.run_once(
        spec=_task_spec(tmp_path, supersession=supersession),
        root=tmp_path / "control",
        spec_sha256="d" * 64,
    )
    task = payload["tasks"]["legacy"]
    assert task["health_state"] == OBSERVATION_FAILED
    assert task["scientific_progress_state"] == OBSERVATION_FAILED
    assert task["route_viability"] == "UNKNOWN"
    assert task["automatic_signal_allowed"] is False


def test_supersession_rejects_arbitrary_txt_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    supersession, receipt, _ = _supersession_fixture(tmp_path)
    text_receipt = tmp_path / "arbitrary.txt"
    text_receipt.write_bytes(receipt.read_bytes())
    supersession["receipt_path"] = str(text_receipt)
    task = _observe_supersession(monkeypatch, tmp_path, supersession)
    assert task["health_state"] == OBSERVATION_FAILED
    assert ".json suffix" in str(task["observation_error"])


def test_supersession_rejects_symlink_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    supersession, receipt, _ = _supersession_fixture(tmp_path)
    linked = tmp_path / "linked-receipt.json"
    linked.symlink_to(receipt)
    supersession["receipt_path"] = str(linked)
    task = _observe_supersession(monkeypatch, tmp_path, supersession)
    assert task["health_state"] == OBSERVATION_FAILED
    assert "symlinks" in str(task["observation_error"])


def test_supersession_rejects_wrong_expected_receipt_sha256(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    supersession, _, _ = _supersession_fixture(tmp_path)
    supersession["expected_receipt_sha256"] = "0" * 64
    task = _observe_supersession(monkeypatch, tmp_path, supersession)
    assert task["health_state"] == OBSERVATION_FAILED
    assert "SHA256 mismatch" in str(task["observation_error"])


def test_supersession_rejects_missing_required_receipt_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    supersession, receipt, receipt_payload = _supersession_fixture(tmp_path)
    del receipt_payload["graceful_stop_completed"]
    supersession["expected_receipt_sha256"] = _write_json(receipt, receipt_payload)
    task = _observe_supersession(monkeypatch, tmp_path, supersession)
    assert task["health_state"] == OBSERVATION_FAILED
    assert "missing graceful_stop_completed" in str(task["observation_error"])


@pytest.mark.parametrize(
    ("field", "json_number"),
    [
        ("graceful_checkpoint_completed", 1),
        ("graceful_stop_completed", 1),
        ("old_worker_exited", 1),
        ("sigkill_used", 0),
    ],
)
def test_supersession_receipt_booleans_reject_json_one_and_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    json_number: int,
) -> None:
    supersession, receipt, receipt_payload = _supersession_fixture(tmp_path)
    receipt_payload[field] = json_number
    supersession["expected_receipt_sha256"] = _write_json(receipt, receipt_payload)
    task = _observe_supersession(monkeypatch, tmp_path, supersession)
    assert task["health_state"] == OBSERVATION_FAILED
    assert f"{field} does not match" in str(task["observation_error"])


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("pid", 12345.0),
        ("start_ticks", 42.0),
        ("pid", True),
        ("start_ticks", True),
    ],
)
def test_supersession_receipt_pid_identity_rejects_float_and_bool_as_int(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    supersession, receipt, receipt_payload = _supersession_fixture(tmp_path)
    old_task = receipt_payload["old_task"]
    assert isinstance(old_task, dict)
    old_task[field] = invalid_value
    supersession["expected_receipt_sha256"] = _write_json(receipt, receipt_payload)
    task = _observe_supersession(monkeypatch, tmp_path, supersession)
    assert task["health_state"] == OBSERVATION_FAILED
    assert f"old_task {field} does not match" in str(task["observation_error"])


def test_supersession_rejects_receipt_tampering_after_spec_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    supersession, receipt, receipt_payload = _supersession_fixture(tmp_path)
    receipt_payload["old_worker_exited"] = False
    _write_json(receipt, receipt_payload)
    task = _observe_supersession(monkeypatch, tmp_path, supersession)
    assert task["health_state"] == OBSERVATION_FAILED
    assert "SHA256 mismatch" in str(task["observation_error"])


def test_supersession_rejects_old_pid_rebinding_even_with_recomputed_receipt_sha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    supersession, receipt, receipt_payload = _supersession_fixture(tmp_path)
    old_task = receipt_payload["old_task"]
    assert isinstance(old_task, dict)
    old_task["pid"] = 99999
    supersession["expected_receipt_sha256"] = _write_json(receipt, receipt_payload)
    task = _observe_supersession(monkeypatch, tmp_path, supersession)
    assert task["health_state"] == OBSERVATION_FAILED
    assert "old_task pid does not match" in str(task["observation_error"])


def test_supersession_rejects_sigkill_even_with_recomputed_receipt_sha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    supersession, receipt, receipt_payload = _supersession_fixture(tmp_path)
    receipt_payload["sigkill_used"] = True
    supersession["expected_receipt_sha256"] = _write_json(receipt, receipt_payload)
    task = _observe_supersession(monkeypatch, tmp_path, supersession)
    assert task["health_state"] == OBSERVATION_FAILED
    assert "sigkill_used does not match" in str(task["observation_error"])


def test_supersession_rejects_replacement_manifest_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    supersession, _, _ = _supersession_fixture(tmp_path)
    replacement = supersession["replacement"]
    assert isinstance(replacement, dict)
    controller_manifest = Path(str(replacement["controller_manifest_path"]))
    _write_json(
        controller_manifest,
        {"controller_id": "replacement-controller", "tampered": True},
    )
    task = _observe_supersession(monkeypatch, tmp_path, supersession)
    assert task["health_state"] == OBSERVATION_FAILED
    assert "controller manifest SHA256 mismatch" in str(task["observation_error"])


def test_supersession_rejects_live_old_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    supersession, _, _ = _supersession_fixture(tmp_path)
    task = _observe_supersession(
        monkeypatch,
        tmp_path,
        supersession,
        task_identity=_identity(12345),
    )
    assert task["health_state"] == OBSERVATION_FAILED
    assert task["scientific_worker_alive"] is True
    assert "live scientific worker" in str(task["observation_error"])


@pytest.mark.parametrize("proc_state", ["Z", "X", "x"])
def test_terminal_proc_states_are_not_live_and_allow_valid_supersession(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, proc_state: str
) -> None:
    supersession, _, _ = _supersession_fixture(tmp_path)
    task = _observe_supersession(
        monkeypatch,
        tmp_path,
        supersession,
        task_identity=_identity(12345, state=proc_state),
    )
    assert task["health_state"] == SUPERSEDED
    assert task["scientific_worker_alive"] is False


def test_supersession_rejects_replacement_gate_that_is_not_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    supersession, receipt, receipt_payload = _supersession_fixture(tmp_path)
    replacement_spec = supersession["replacement"]
    assert isinstance(replacement_spec, dict)
    task_gate = Path(str(replacement_spec["task_gate_path"]))
    failed_gate_sha = _write_json(
        task_gate, {"task_id": "replacement-task", "status": "FAILED"}
    )
    replacement_spec["expected_task_gate_sha256"] = failed_gate_sha
    receipt_replacement = receipt_payload["replacement"]
    assert isinstance(receipt_replacement, dict)
    receipt_replacement["task_gate_sha256"] = failed_gate_sha
    supersession["expected_receipt_sha256"] = _write_json(receipt, receipt_payload)
    task = _observe_supersession(monkeypatch, tmp_path, supersession)
    assert task["health_state"] == OBSERVATION_FAILED
    assert "task gate is not PASS" in str(task["observation_error"])


def test_status_table_exposes_all_four_distinct_health_dimensions() -> None:
    rendered = render_table(
        {
            "controller_id": "monitor",
            "controller_pid": 10,
            "controller_process_alive": True,
            "updated_at": _time(0),
            "tasks": {
                "old-brute": {
                    "controller_process_alive": True,
                    "scientific_worker_alive": True,
                    "scientific_progress_state": RUNNING_UNVIABLE,
                    "route_viability": "UNVIABLE",
                    "completed": 27,
                    "total": 100,
                    "eta_hours": 9000,
                }
            },
        }
    )
    assert "controller_process_alive" in rendered
    assert "scientific_worker_alive" in rendered
    assert "scientific_progress_state" in rendered
    assert "route_viability" in rendered
    assert "RUNNING_UNVIABLE" in rendered


def test_status_table_falls_back_for_monitor_v1_state() -> None:
    rendered = render_table(
        {
            "controller_id": "old-monitor",
            "controller_pid": 11,
            "updated_at": _time(0),
            "tasks": {
                "old-brute": {
                    "pid_alive": True,
                    "health_state": RUNNING_UNVIABLE,
                    "completed": 27,
                    "total": 100,
                    "eta_hours": 9000,
                }
            },
        }
    )
    assert "controller_process_alive=UNKNOWN" in rendered
    assert "old-brute\tUNKNOWN\tTrue\tRUNNING_UNVIABLE\tUNVIABLE" in rendered
