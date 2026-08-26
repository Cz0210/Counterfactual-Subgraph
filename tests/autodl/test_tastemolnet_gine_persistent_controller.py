from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

import pytest

from src.utils import autodl_tastemolnet_gine_controller_v1 as controller_module
from src.utils.autodl_exec_startup_barrier import arm_exec_startup_barrier
from src.utils.autodl_tastemolnet_gine_controller_v1 import (
    PASS_NAME,
    STATE_NAME,
    TasteGINEControllerSpec,
    TasteGINEPersistentController,
    inspect_tastemolnet_gine_controller,
    stable_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _spec(tmp_path: Path, worker: Path) -> TasteGINEControllerSpec:
    output = tmp_path / "science-output"
    state = tmp_path / "training-state"
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "PYTHONPATH": str(PROJECT_ROOT),
        "TASTEMOLNET_GNN_FULL_OUTPUT": str(output),
        "TASTEMOLNET_GNN_TRAINING_STATE_ROOT": str(state),
        "TASTEMOLNET_GINE_CONTROLLER_CID": "tastemolnet_gine_v1_20260825T000000Z_deadbeef",
        "TASTEMOLNET_GINE_CONTROLLER_ROOT": str(tmp_path / "controller"),
    }
    return TasteGINEControllerSpec(
        cid="tastemolnet_gine_v1_20260825T000000Z_deadbeef",
        controller_root=tmp_path / "controller",
        project_root=PROJECT_ROOT,
        output_dir=output,
        training_state_root=state,
        worker_argv=(sys.executable, str(worker)),
        source_identity={},
        environment_authority=environment,
        config_files=(),
        poll_seconds=0.01,
        terminal_stability_seconds=0.0,
        resource_wait_deadline_seconds=1,
    )


@pytest.fixture(autouse=True)
def _skip_production_source_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        TasteGINEPersistentController,
        "_validate_spec_sources",
        lambda self: None,
    )


def _write_worker(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _synthetic_process_snapshot(
    *, pid: int, start: int, ppid: int, argv: list[str]
) -> dict[str, object]:
    executable = str(Path(sys.executable).resolve(strict=True))
    return {
        "pid": pid,
        "linux_start_ticks": start,
        "ppid": ppid,
        "argv": list(argv),
        "argv_sha256": stable_sha256(list(argv)),
        "cmdline_sha256": "a" * 64,
        "cwd": str(PROJECT_ROOT.resolve(strict=True)),
        "exe": executable,
        "exe_identity": controller_module._file_identity(os.stat(executable)),
    }


def _write_synthetic_trainer_authority(
    *,
    spec: TasteGINEControllerSpec,
    run_id: str,
    parent_pid: int,
    parent_start: int,
    child_pid: int,
    child_start: int,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    control_root = Path(spec.environment_authority["AUTODL_CONTROL_ROOT"])
    run_root = control_root / "experiment_registry/run_state" / run_id
    run_root.mkdir(parents=True)
    authority_path = run_root / controller_module.TRAINER_CHILD_AUTHORITY_NAME
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train_molecular_gnn.py"),
    ]
    barrier = arm_exec_startup_barrier(
        lock_path=run_root / "trainer-startup.lock",
        record_path=run_root / "trainer-startup.json",
        target_argv=command,
        python_executable=sys.executable,
    )
    barrier_record = barrier.record.to_dict()
    barrier.abort()
    parent = _synthetic_process_snapshot(
        pid=parent_pid,
        start=parent_start,
        ppid=1,
        argv=[sys.executable, str(PROJECT_ROOT / "scripts/autodl/exp_run.py")],
    )
    child = _synthetic_process_snapshot(
        pid=child_pid,
        start=child_start,
        ppid=parent_pid,
        argv=list(barrier.record.launcher_argv),
    )
    payload = {
        "schema_version": controller_module.TRAINER_CHILD_AUTHORITY_SCHEMA,
        "status": "RELEASE_AUTHORIZED",
        "run_id": run_id,
        "dataset": "tastemolnet",
        "stage": "TASTEMOLNET_GINE_FULL_RESEARCH_V1",
        "controller_cid": spec.cid,
        "controller_root": str(spec.controller_root),
        "project_root": str(PROJECT_ROOT.resolve(strict=True)),
        "authority_path": str(authority_path),
        "parent_exp_run": parent,
        "child_registered": child,
        "trainer_command": command,
        "trainer_command_sha256": stable_sha256(command),
        "barrier_record": barrier_record,
    }
    authority_path.write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    authority_path.chmod(0o600)
    return authority_path, parent, child


def _spec_with_control_root(
    tmp_path: Path, worker: Path
) -> TasteGINEControllerSpec:
    base = _spec(tmp_path, worker)
    return TasteGINEControllerSpec(
        **{
            **{
                name: getattr(base, name)
                for name in base.__dataclass_fields__
            },
            "environment_authority": {
                **base.environment_authority,
                "AUTODL_CONTROL_ROOT": str(tmp_path / "autodl-control"),
            },
        }
    )


def _install_linux_proc_fixture(
    monkeypatch: pytest.MonkeyPatch,
    observations: dict[int, tuple[int, str] | None],
    calls: list[int],
) -> None:
    real_is_file = Path.is_file
    real_read_text = Path.read_text

    def fixture_is_file(path: Path) -> bool:
        if path == Path("/proc/self/stat"):
            return True
        return real_is_file(path)

    def fixture_read_text(path: Path, *args: object, **kwargs: object) -> str:
        prefix = "/proc/"
        suffix = "/stat"
        raw_path = str(path)
        if raw_path.startswith(prefix) and raw_path.endswith(suffix):
            pid = int(raw_path[len(prefix) : -len(suffix)])
            calls.append(pid)
            observation = observations[pid]
            if observation is None:
                raise FileNotFoundError(raw_path)
            start, state = observation
            fields = [state, "1", *(["0"] * 17), str(start)]
            return f"{pid} (synthetic trainer) {' '.join(fields)}\n"
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(controller_module.sys, "platform", "linux")
    monkeypatch.setattr(Path, "is_file", fixture_is_file)
    monkeypatch.setattr(Path, "read_text", fixture_read_text)


def test_fresh_controller_recovers_empty_root_creation_window(tmp_path: Path) -> None:
    worker = _write_worker(tmp_path / "worker.py", "raise SystemExit(2)\n")
    spec = _spec(tmp_path, worker)
    spec.controller_root.mkdir(mode=0o700)
    with TasteGINEPersistentController(spec, resume=True) as controller:
        assert (controller.root / "controller_root_claim.json").is_file()
        assert (controller.root / "controller_spec.json").is_file()


def test_controller_rejects_symlinked_parent_before_root_creation(
    tmp_path: Path,
) -> None:
    worker = _write_worker(tmp_path / "worker.py", "raise SystemExit(2)\n")
    physical = tmp_path / "physical"
    physical.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(physical, target_is_directory=True)
    spec = _spec(alias, worker)
    with pytest.raises(
        controller_module.TasteGINEControllerError, match="symlink components"
    ):
        TasteGINEPersistentController(spec, resume=False).open()


def test_child_environment_uses_only_frozen_spec_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _write_worker(tmp_path / "worker.py", "raise SystemExit(2)\n")
    spec = _spec(tmp_path, worker)
    frozen_path = spec.environment_authority["PATH"]
    monkeypatch.setenv("PATH", "/drifted/path")
    monkeypatch.setenv("UNREVIEWED_SCIENCE_FLAG", "drifted")
    controller = TasteGINEPersistentController(spec, resume=False)
    child = controller._child_environment()
    assert child["PATH"] == frozen_path
    assert "UNREVIEWED_SCIENCE_FLAG" not in child


def test_controller_detects_same_byte_root_and_named_lock_replacement(
    tmp_path: Path,
) -> None:
    worker = _write_worker(tmp_path / "worker.py", "raise SystemExit(2)\n")
    spec = _spec(tmp_path, worker)
    controller = TasteGINEPersistentController(spec, resume=False)
    controller.open()
    displaced = tmp_path / "displaced-controller"
    controller.root.rename(displaced)
    shutil.copytree(displaced, controller.root, copy_function=shutil.copy2)
    with pytest.raises(controller_module.TasteGINEControllerError, match="root identity"):
        controller.verify_authority()
    controller.close()

    second_spec = _spec(tmp_path / "second", worker)
    second_spec.controller_root.parent.mkdir()
    second = TasteGINEPersistentController(second_spec, resume=False)
    second.open()
    lock = second.root / ".controller.lock"
    lock.unlink()
    lock.write_bytes(b"replacement")
    with pytest.raises(controller_module.TasteGINEControllerError, match="lock changed"):
        second.verify_authority()
    second.close()


def test_typed_status_rejects_joint_spec_and_state_rewrite(tmp_path: Path) -> None:
    worker = _write_worker(tmp_path / "worker.py", "raise SystemExit(2)\n")
    spec = _spec(tmp_path, worker)
    controller = TasteGINEPersistentController(spec, resume=False)
    controller.open()
    controller._write_state(
        "WAITING_RESOURCES",
        attempt=0,
        launch_index=1,
        retries_used=0,
    )
    controller.close()
    spec_path = spec.controller_root / "controller_spec.json"
    state_path = spec.controller_root / STATE_NAME
    spec_payload = json.loads(spec_path.read_text(encoding="utf-8"))
    spec_payload["output_dir"] = str(tmp_path / "forged-output")
    spec_path.write_text(json.dumps(spec_payload), encoding="utf-8")
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    state_payload["spec_sha256"] = stable_sha256(spec_payload)
    state_path.write_text(json.dumps(state_payload), encoding="utf-8")
    with pytest.raises(controller_module.TasteGINEControllerError, match="status closure"):
        inspect_tastemolnet_gine_controller(spec.controller_root)


def test_arm_record_before_pre_release_state_is_reconciled_and_rearmed(
    tmp_path: Path,
) -> None:
    worker = _write_worker(tmp_path / "worker.py", "raise SystemExit(2)\n")
    spec = _spec(tmp_path, worker)
    first = TasteGINEPersistentController(spec, resume=False)
    first.open()
    lock, record = first._barrier_paths(0)
    first._write_state(
        "ARMING",
        attempt=0,
        launch_index=0,
        barrier_lock=str(lock),
        barrier_record_path=str(record),
        retries_used=0,
    )
    armed = arm_exec_startup_barrier(
        lock_path=lock,
        record_path=record,
        target_argv=spec.worker_argv,
        python_executable=sys.executable,
    )
    armed.abort()
    first.close()

    with TasteGINEPersistentController(spec, resume=True) as resumed:
        assert resumed.run() == 2
        state = json.loads((resumed.root / STATE_NAME).read_text(encoding="utf-8"))
        assert state["phase"] == "FAILED"
        assert state["reason"] == "SCIENTIFIC_OR_NORMAL_PROCESS_FAILURE"


def test_gpu_wait_exit_is_persistently_supervised_to_one_global_deadline(
    tmp_path: Path,
) -> None:
    worker = _write_worker(tmp_path / "wait.py", "raise SystemExit(75)\n")
    base = _spec(tmp_path, worker)
    spec = controller_module.TasteGINEControllerSpec(
        **{
            **{
                name: getattr(base, name)
                for name in base.__dataclass_fields__
            },
            "resource_wait_deadline_seconds": 2,
            "poll_seconds": 0.05,
        }
    )
    with TasteGINEPersistentController(spec, resume=False) as controller:
        started = time.monotonic()
        assert controller.run() == 75
        assert time.monotonic() - started >= 0.8
        state = json.loads((controller.root / STATE_NAME).read_text(encoding="utf-8"))
        assert state["phase"] == "FAILED"
        assert state["reason"] == "RESOURCE_WAIT_GLOBAL_DEADLINE_EXCEEDED"
        assert state["attempt"] == 0
        assert state["retries_used"] == 0
        assert 1 < len(list(controller.root.glob("startup-launch-*.json"))) < 64


def test_release_authorized_crash_rearms_same_attempt_without_science_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = tmp_path / "science-started"
    worker = _write_worker(
        tmp_path / "worker.py",
        f"from pathlib import Path\nPath({str(counter)!r}).write_text('once')\nraise SystemExit(2)\n",
    )
    spec = _spec(tmp_path, worker)
    from src.utils.autodl_exec_startup_barrier import ArmedExecStartupBarrier

    real_release = ArmedExecStartupBarrier.release

    def crash_before_release(self: ArmedExecStartupBarrier) -> None:
        self.abort()
        raise RuntimeError("controller crash before release token")

    monkeypatch.setattr(ArmedExecStartupBarrier, "release", crash_before_release)
    first = TasteGINEPersistentController(spec, resume=False)
    first.open()
    with pytest.raises(RuntimeError, match="before release token"):
        first.run()
    first.close()
    assert not counter.exists()
    state = json.loads((spec.controller_root / STATE_NAME).read_text(encoding="utf-8"))
    assert state["phase"] == "RELEASE_AUTHORIZED"
    assert state["attempt"] == 0
    try:
        os.waitpid(int(state["worker_generation"]["pid"]), 0)
    except ChildProcessError:
        pass

    monkeypatch.setattr(ArmedExecStartupBarrier, "release", real_release)
    with TasteGINEPersistentController(spec, resume=True) as resumed:
        assert resumed.run() == 2
        recovered = json.loads((resumed.root / STATE_NAME).read_text(encoding="utf-8"))
    assert counter.read_text(encoding="utf-8") == "once"
    assert recovered["attempt"] == 0
    assert recovered["retries_used"] == 0


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_real_sigkill_in_release_authorized_rearms_without_science_retry(
    tmp_path: Path,
) -> None:
    counter = tmp_path / "science-started"
    worker = _write_worker(
        tmp_path / "worker.py",
        f"from pathlib import Path\nPath({str(counter)!r}).write_text('once')\nraise SystemExit(2)\n",
    )
    spec = _spec(tmp_path, worker)
    child = os.fork()
    if child == 0:  # pragma: no cover - the child intentionally dies by SIGKILL.
        from src.utils.autodl_exec_startup_barrier import ArmedExecStartupBarrier

        def hard_crash(self: ArmedExecStartupBarrier) -> None:
            os.kill(os.getpid(), signal.SIGKILL)

        ArmedExecStartupBarrier.release = hard_crash
        controller = TasteGINEPersistentController(spec, resume=False)
        controller.open()
        controller.run()
        os._exit(99)
    _, status = os.waitpid(child, 0)
    assert os.WIFSIGNALED(status)
    assert os.WTERMSIG(status) == signal.SIGKILL
    assert not counter.exists()
    state = json.loads((spec.controller_root / STATE_NAME).read_text(encoding="utf-8"))
    assert state["phase"] == "RELEASE_AUTHORIZED"
    with TasteGINEPersistentController(spec, resume=True) as resumed:
        assert resumed.run() == 2
        recovered = json.loads((resumed.root / STATE_NAME).read_text(encoding="utf-8"))
    assert counter.read_text(encoding="utf-8") == "once"
    assert recovered["attempt"] == 0
    assert recovered["retries_used"] == 0


def test_event_cap_never_blocks_terminal_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _write_worker(tmp_path / "worker.py", "raise SystemExit(2)\n")
    spec = _spec(tmp_path, worker)
    with TasteGINEPersistentController(spec, resume=False) as controller:
        monkeypatch.setattr(controller_module, "MAX_EVENTS_BYTES", 512)
        monkeypatch.setattr(controller_module, "TERMINAL_EVENT_RESERVE_BYTES", 256)
        events = controller.root / controller_module.EVENTS_NAME
        events.write_bytes(b"x" * 500)
        events.chmod(0o600)
        state = controller._write_state(
            "FAILED", attempt=0, launch_index=0, retries_used=0, reason="fixture"
        )
        assert state["phase"] == "FAILED"
        assert json.loads((controller.root / STATE_NAME).read_text())["phase"] == "FAILED"
        assert events.stat().st_size == 500


def test_live_worker_log_cap_retains_supervision_until_worker_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finished = tmp_path / "finished"
    worker = _write_worker(
        tmp_path / "worker.py",
        "import pathlib, time\n"
        "print('diagnostic-over-cap', flush=True)\n"
        "time.sleep(0.08)\n"
        f"pathlib.Path({str(finished)!r}).write_text('done')\n"
        "raise SystemExit(2)\n",
    )
    spec = _spec(tmp_path, worker)
    monkeypatch.setattr(controller_module, "MAX_WORKER_LOG_BYTES", 1)
    with TasteGINEPersistentController(spec, resume=False) as controller:
        assert controller.run() == 2
        state = json.loads((controller.root / STATE_NAME).read_text(encoding="utf-8"))
    assert finished.read_text(encoding="utf-8") == "done"
    assert state["reason"] == "SCIENTIFIC_OR_NORMAL_PROCESS_FAILURE"
    assert state["worker_log_cap_observed"] is True


def test_process_loss_retries_same_training_state_root_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = tmp_path / "counter.json"
    worker = _write_worker(
        tmp_path / "loss_then_ok.py",
        "import json, os, signal\n"
        f"path = {str(counter)!r}\n"
        "rows = json.loads(open(path).read()) if os.path.exists(path) else []\n"
        "rows.append(os.environ['TASTEMOLNET_GNN_TRAINING_STATE_ROOT'])\n"
        "open(path, 'w').write(json.dumps(rows))\n"
        "if len(rows) == 1: os.kill(os.getpid(), signal.SIGKILL)\n",
    )
    spec = _spec(tmp_path, worker)

    def terminal(self):
        if counter.exists() and len(json.loads(counter.read_text(encoding="utf-8"))) == 2:
            return {"fixture": "terminal"}
        return None

    def publish(self, evidence, *, attempt, launch_index):
        self._write_state(
            "PASS", attempt=attempt, launch_index=launch_index, retries_used=attempt
        )

    monkeypatch.setattr(TasteGINEPersistentController, "_terminal_evidence", terminal)
    monkeypatch.setattr(TasteGINEPersistentController, "_publish_terminal", publish)
    with TasteGINEPersistentController(spec, resume=False) as controller:
        assert controller.run() == 0
        state = json.loads((controller.root / STATE_NAME).read_text(encoding="utf-8"))
        assert state["attempt"] == 1
        roots = json.loads(counter.read_text(encoding="utf-8"))
        assert roots == [str(spec.training_state_root), str(spec.training_state_root)]


def test_terminal_state_without_pass_marker_fails_shared_strict_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _write_worker(tmp_path / "unused.py", "raise SystemExit(0)\n")
    spec = _spec(tmp_path, worker)
    controller = TasteGINEPersistentController(spec, resume=False)
    controller.open()
    evidence = {"fixture": "stable"}
    hold = {"fixture": "held"}
    monkeypatch.setattr(controller, "_acquire_terminal_hold", lambda: hold)
    monkeypatch.setattr(controller, "_scan_terminal_hold", lambda value: evidence)
    monkeypatch.setattr(controller, "_release_terminal_hold", lambda value: None)
    real_write = controller_module._write_text_new

    def crash_before_pass(path: Path, text: str) -> None:
        if path.name == PASS_NAME:
            raise RuntimeError("crash before PASS")
        real_write(path, text)

    monkeypatch.setattr(controller_module, "_write_text_new", crash_before_pass)
    with pytest.raises(RuntimeError, match="before PASS"):
        controller._publish_terminal(evidence, attempt=0, launch_index=0)
    state = json.loads((controller.root / STATE_NAME).read_text(encoding="utf-8"))
    assert state["phase"] == "PASS"
    assert not (controller.root / PASS_NAME).exists()
    monkeypatch.setattr(controller_module, "_write_text_new", real_write)
    with pytest.raises(
        controller_module.TasteGINEControllerError,
        match="No such file|PASS",
    ):
        controller.run()
    assert not (controller.root / PASS_NAME).exists()
    controller.close()


def test_pass_postscan_failure_revokes_only_marker_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _write_worker(tmp_path / "unused.py", "raise SystemExit(0)\n")
    spec = _spec(tmp_path, worker)
    controller = TasteGINEPersistentController(spec, resume=False)
    controller.open()
    evidence = {"fixture": "stable"}
    hold = {"fixture": "held"}
    monkeypatch.setattr(controller, "_acquire_terminal_hold", lambda: hold)
    scans = 0

    def scan(value):
        nonlocal scans
        assert value is hold
        scans += 1
        return evidence if scans < 4 else None

    monkeypatch.setattr(controller, "_scan_terminal_hold", scan)
    monkeypatch.setattr(controller, "_release_terminal_hold", lambda value: None)
    with pytest.raises(controller_module.TasteGINEControllerError, match="after PASS"):
        controller._publish_terminal(evidence, attempt=0, launch_index=0)
    assert not (controller.root / PASS_NAME).exists()
    controller.close()


def test_pass_postscan_never_unlinks_substituted_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _write_worker(tmp_path / "unused.py", "raise SystemExit(0)\n")
    spec = _spec(tmp_path, worker)
    controller = TasteGINEPersistentController(spec, resume=False)
    controller.open()
    evidence = {"fixture": "stable"}
    hold = {"fixture": "held"}
    monkeypatch.setattr(controller, "_acquire_terminal_hold", lambda: hold)
    scans = 0

    def scan(value):
        nonlocal scans
        assert value is hold
        scans += 1
        if scans == 4:
            marker = controller.root / PASS_NAME
            marker.rename(controller.root / "displaced-pass-marker")
            marker.write_text("replacement\n", encoding="utf-8")
            marker.chmod(0o600)
            return None
        return evidence

    monkeypatch.setattr(controller, "_scan_terminal_hold", scan)
    monkeypatch.setattr(controller, "_release_terminal_hold", lambda value: None)
    with pytest.raises(controller_module.TasteGINEControllerError, match="after PASS"):
        controller._publish_terminal(evidence, attempt=0, launch_index=0)
    assert (controller.root / PASS_NAME).read_text(encoding="utf-8") == "replacement\n"
    controller.close()


def test_status_pass_reopen_is_strictly_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _write_worker(tmp_path / "unused.py", "raise SystemExit(0)\n")
    spec = _spec(tmp_path, worker)
    evidence = {"fixture": "stable"}
    hold = {"fixture": "held"}
    monkeypatch.setattr(
        TasteGINEPersistentController, "_acquire_terminal_hold", lambda self: hold
    )
    monkeypatch.setattr(
        TasteGINEPersistentController,
        "_scan_terminal_hold",
        lambda self, value: evidence,
    )
    monkeypatch.setattr(
        TasteGINEPersistentController,
        "_release_terminal_hold",
        lambda self, value: None,
    )
    monkeypatch.setattr(
        TasteGINEPersistentController,
        "_terminal_evidence",
        lambda self: evidence,
    )
    with TasteGINEPersistentController(spec, resume=False) as controller:
        controller._publish_terminal(evidence, attempt=0, launch_index=0)

    def snapshot() -> dict[str, tuple[int, int, int, str]]:
        rows = {}
        for path in sorted(spec.controller_root.iterdir()):
            if path.is_file():
                info = path.stat()
                rows[path.name] = (
                    info.st_ino,
                    info.st_size,
                    info.st_mtime_ns,
                    controller_module.sha256_file(path),
                )
        return rows

    before = snapshot()
    status = inspect_tastemolnet_gine_controller(spec.controller_root)
    after_status = snapshot()
    assert status["pass"] is True
    assert after_status == before
    with TasteGINEPersistentController(spec, resume=True) as reopened:
        assert reopened.run() == 0
    assert snapshot() == before
    injected = spec.controller_root / "second_terminal_claim.json"
    injected.write_text("{}\n", encoding="utf-8")
    injected.chmod(0o600)
    injected_before = snapshot()
    with pytest.raises(
        controller_module.TasteGINEControllerError,
        match="extra terminal-named",
    ):
        inspect_tastemolnet_gine_controller(spec.controller_root)
    assert snapshot() == injected_before


def test_resume_detects_partial_terminal_name_before_any_writer_reconciliation(
    tmp_path: Path,
) -> None:
    worker = _write_worker(tmp_path / "unused.py", "raise SystemExit(0)\n")
    spec = _spec(tmp_path, worker)
    with TasteGINEPersistentController(spec, resume=False):
        pass
    partial = spec.controller_root / ".controller_terminal.json.crash.tmp"
    partial.write_text("partial\n", encoding="utf-8")

    def snapshot() -> dict[str, tuple[int, int, int, str]]:
        rows = {}
        for path in sorted(spec.controller_root.iterdir()):
            if path.is_file():
                info = path.stat()
                rows[path.name] = (
                    info.st_ino,
                    info.st_size,
                    info.st_mtime_ns,
                    controller_module.sha256_file(path),
                )
        return rows

    before = snapshot()
    with pytest.raises(
        controller_module.TasteGINEControllerError,
        match="missing/extra terminal-named",
    ):
        TasteGINEPersistentController(spec, resume=True).open()
    assert snapshot() == before


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires /proc")
def test_generation_binding_allows_only_reviewed_launcher_to_target_exec(
    tmp_path: Path,
) -> None:
    ready = tmp_path / "worker-ready"
    stop = tmp_path / "worker-stop"
    worker = _write_worker(
        tmp_path / "sleep.py",
        "from pathlib import Path\n"
        "import time\n"
        f"ready = Path({str(ready)!r})\n"
        f"stop = Path({str(stop)!r})\n"
        "ready.write_text('ready\\n', encoding='utf-8')\n"
        "deadline = time.monotonic() + 5\n"
        "while not stop.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.005)\n"
        "raise SystemExit(0 if stop.exists() else 3)\n",
    )
    spec = _spec(tmp_path, worker)
    lock = tmp_path / "barrier.lock"
    record_path = tmp_path / "barrier.json"
    barrier = arm_exec_startup_barrier(
        lock_path=lock,
        record_path=record_path,
        target_argv=spec.worker_argv,
        python_executable=sys.executable,
    )
    process = barrier.launch(
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    generation = controller_module._process_generation(
        process.pid,
        spec=spec,
        barrier_record=barrier.record.to_dict(),
    )
    assert generation["last_observed_phase"] == "startup_launcher"
    barrier.release()
    observed = generation
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        candidate = controller_module._observe_generation(
            observed,
            spec=spec,
            barrier_record=barrier.record.to_dict(),
        )
        if candidate is not None:
            observed = candidate
            if observed["last_observed_phase"] == "worker_target":
                break
        time.sleep(0.005)
    assert observed["last_observed_phase"] == "worker_target"
    assert observed["last_observed"]["cwd"] == str(PROJECT_ROOT)
    assert observed["last_observed"]["argv"] == list(spec.worker_argv)
    assert observed["last_observed"]["exe"] == str(Path(sys.executable).resolve())
    ready_deadline = time.monotonic() + 1
    while not ready.is_file() and time.monotonic() < ready_deadline:
        time.sleep(0.005)
    assert ready.is_file()
    stop.write_text("stop\n", encoding="utf-8")
    exit_deadline = time.monotonic() + 2
    exited = None
    while time.monotonic() < exit_deadline:
        exited = controller_module._read_linux_process_stat(process.pid)
        if (
            exited is None
            or exited.state in controller_module.LINUX_EXITED_PROCESS_STATES
        ):
            break
        time.sleep(0.005)
    assert exited is None or (
        exited.state in controller_module.LINUX_EXITED_PROCESS_STATES
    )
    assert (
        controller_module._observe_generation(
            observed,
            spec=spec,
            barrier_record=barrier.record.to_dict(),
        )
        is None
    )
    assert process.wait(timeout=2) == 0


def test_linux_exit_race_checks_proc_state_before_empty_argv_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = 81001
    start = 701
    live = controller_module._LinuxProcessStatObservation(
        pid=pid,
        state="S",
        ppid=1,
        start_ticks=start,
    )
    zombie = controller_module._LinuxProcessStatObservation(
        pid=pid,
        state="Z",
        ppid=1,
        start_ticks=start,
    )

    worker = _write_worker(tmp_path / "unused.py", "raise SystemExit(0)\n")
    spec = _spec(tmp_path, worker)
    registered = {"pid": pid, "linux_start_ticks": start}
    generation = {
        "pid": pid,
        "linux_start_ticks": start,
        "registered": registered,
        "registered_phase": "startup_launcher",
        "last_observed_phase": "startup_launcher",
        "phase_bindings": {"startup_launcher": registered},
        "ancestry": {},
    }
    monkeypatch.setattr(controller_module.sys, "platform", "linux")
    observations = iter((live, live, zombie))
    monkeypatch.setattr(
        controller_module,
        "_read_linux_process_stat",
        lambda observed_pid: next(observations),
    )
    monkeypatch.setattr(
        controller_module,
        "_process_snapshot",
        lambda observed_pid: {
            "pid": observed_pid,
            "linux_start_ticks": start,
            "argv": [],
        },
    )
    monkeypatch.setattr(
        controller_module,
        "_classify_process_phase",
        lambda *args, **kwargs: pytest.fail(
            "argv phase classification ran after exact-generation exit"
        ),
    )
    assert (
        controller_module._observe_generation(
            generation,
            spec=spec,
            barrier_record={},
        )
        is None
    )

    reused = controller_module._LinuxProcessStatObservation(
        pid=pid,
        state="S",
        ppid=1,
        start_ticks=start + 1,
    )
    observations = iter((live, reused))
    assert (
        controller_module._snapshot_live_linux_generation(
            pid, start, label="worker"
        )
        is None
    )

    for malformed_argv in ([], [""]):
        observations = iter((live, live))
        monkeypatch.setattr(
            controller_module,
            "_process_snapshot",
            lambda observed_pid, argv=malformed_argv: {
                "pid": observed_pid,
                "linux_start_ticks": start,
                "argv": argv,
            },
        )
        with pytest.raises(
            controller_module.TasteGINEControllerError,
            match="live worker process argv is empty or malformed",
        ):
            controller_module._snapshot_live_linux_generation(
                pid, start, label="worker"
            )


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires /proc")
def test_real_trainer_child_authority_survives_exp_run_parent_sigkill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A durably registered trainer remains owned after its exp_run parent dies."""

    worker = _write_worker(tmp_path / "outer.py", "raise SystemExit(0)\n")
    base = _spec(tmp_path, worker)
    control_root = tmp_path / "autodl-control"
    environment = {
        **base.environment_authority,
        "AUTODL_CONTROL_ROOT": str(control_root),
    }
    spec = controller_module.TasteGINEControllerSpec(
        **{
            **{
                name: getattr(base, name)
                for name in base.__dataclass_fields__
            },
            "environment_authority": environment,
        }
    )
    run_id = "real-parent-loss"
    run_root = control_root / "experiment_registry/run_state" / run_id
    run_root.mkdir(parents=True)
    authority_path = run_root / controller_module.TRAINER_CHILD_AUTHORITY_NAME
    trainer = _write_worker(
        tmp_path / "trainer.py",
        "import time\ntime.sleep(30)\n",
    )
    parent_program = _write_worker(
        tmp_path / "exp_run_parent.py",
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "from src.utils.autodl_exec_startup_barrier import arm_exec_startup_barrier\n"
        "from src.utils.autodl_runtime import atomic_write_json\n"
        "from src.utils.autodl_tastemolnet_gine_controller_v1 import _process_snapshot, stable_sha256\n"
        f"authority_path = Path({str(authority_path)!r})\n"
        f"target = [sys.executable, {str(trainer)!r}]\n"
        "barrier = arm_exec_startup_barrier(\n"
        "    lock_path=authority_path.parent / 'trainer-startup.lock',\n"
        "    record_path=authority_path.parent / 'trainer-startup.json',\n"
        "    target_argv=target,\n"
        "    python_executable=sys.executable,\n"
        ")\n"
        "child = barrier.launch(\n"
        f"    cwd=Path({str(PROJECT_ROOT)!r}),\n"
        "    env=os.environ.copy(),\n"
        "    stdin=subprocess.DEVNULL,\n"
        "    stdout=subprocess.DEVNULL,\n"
        "    stderr=subprocess.DEVNULL,\n"
        "    start_new_session=True,\n"
        ")\n"
        "payload = {\n"
        f"    'schema_version': {controller_module.TRAINER_CHILD_AUTHORITY_SCHEMA!r},\n"
        "    'status': 'RELEASE_AUTHORIZED',\n"
        f"    'run_id': {run_id!r},\n"
        "    'dataset': 'tastemolnet',\n"
        "    'stage': 'TASTEMOLNET_GINE_FULL_RESEARCH_V1',\n"
        f"    'controller_cid': {spec.cid!r},\n"
        f"    'controller_root': {str(spec.controller_root)!r},\n"
        f"    'project_root': {str(PROJECT_ROOT)!r},\n"
        f"    'authority_path': {str(authority_path)!r},\n"
        "    'parent_exp_run': _process_snapshot(os.getpid()),\n"
        "    'child_registered': _process_snapshot(child.pid),\n"
        "    'trainer_command': target,\n"
        "    'trainer_command_sha256': stable_sha256(target),\n"
        "    'barrier_record': barrier.record.to_dict(),\n"
        "}\n"
        "atomic_write_json(authority_path, payload)\n"
        "barrier.release()\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(30)\n",
    )
    process = subprocess.Popen(
        [sys.executable, str(parent_program)],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    child_pid: int | None = None
    child_start: int | None = None
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not authority_path.is_file():
            if process.poll() is not None:
                stderr = "" if process.stderr is None else process.stderr.read()
                raise AssertionError(
                    f"trainer parent failed before registration: {stderr}"
                )
            time.sleep(0.01)
        assert authority_path.is_file()
        authority = json.loads(authority_path.read_text(encoding="utf-8"))
        child_pid = int(authority["child_registered"]["pid"])
        parent_snapshot = authority["parent_exp_run"]
        monkeypatch.setattr(
            controller_module,
            "_classify_process_phase",
            lambda snapshot, *, spec, barrier_record: "exp_run_target",
        )
        generation = controller_module._trainer_generation_from_authority(
            authority_path,
            spec=spec,
            worker_generation=parent_snapshot,
            worker_barrier_record={},
        )
        child_start = int(generation["linux_start_ticks"])
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            observed = controller_module._observe_trainer_generation(generation)
            assert observed is not None
            generation = observed
            if generation["last_observed_phase"] == "trainer_target":
                break
            time.sleep(0.01)
        assert generation["last_observed_phase"] == "trainer_target"

        process.send_signal(signal.SIGKILL)
        assert process.wait(timeout=5) < 0
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            observed = controller_module._observe_trainer_generation(generation)
            assert observed is not None
            generation = observed
            if generation["ancestry"]["orphan_adopted"] is True:
                break
            time.sleep(0.01)
        assert generation["ancestry"]["orphan_adopted"] is True
        assert generation["ancestry"]["registered_ppid"] == parent_snapshot["pid"]
        assert generation["last_observed"]["pid"] == child_pid
        assert controller_module._linux_start_ticks(child_pid) == child_start
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if (
            child_pid is not None
            and child_start is not None
            and controller_module._linux_start_ticks(child_pid) == child_start
        ):
            os.kill(child_pid, signal.SIGTERM)


@pytest.mark.parametrize(
    "stale_observation",
    [None, (999_999, "S"), (102, "Z")],
    ids=("absent", "pid-reused", "zombie"),
)
def test_dead_stale_trainer_authority_does_not_hide_current_live_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stale_observation: tuple[int, str] | None,
) -> None:
    worker = _write_worker(tmp_path / "outer.py", "raise SystemExit(0)\n")
    spec = _spec_with_control_root(tmp_path, worker)
    stale_path, _, stale_child = _write_synthetic_trainer_authority(
        spec=spec,
        run_id="00-dead-stale",
        parent_pid=51001,
        parent_start=101,
        child_pid=51002,
        child_start=102,
    )
    current_path, current_parent, current_child = (
        _write_synthetic_trainer_authority(
            spec=spec,
            run_id="01-current-live",
            parent_pid=52001,
            parent_start=201,
            child_pid=52002,
            child_start=202,
        )
    )
    classified: list[int] = []
    _install_linux_proc_fixture(
        monkeypatch,
        {
            int(stale_child["pid"]): stale_observation,
            int(current_child["pid"]): (
                int(current_child["linux_start_ticks"]),
                "S",
            ),
        },
        classified,
    )
    monkeypatch.setattr(
        controller_module,
        "_classify_process_phase",
        lambda snapshot, *, spec, barrier_record: "exp_run_target",
    )
    monkeypatch.setattr(
        controller_module,
        "_observe_trainer_generation",
        lambda generation: dict(generation),
    )
    controller = TasteGINEPersistentController(spec, resume=False)
    observed = controller._discover_trainer_generation(
        {
            "worker_generation": current_parent,
            "barrier_record": {},
        }
    )

    assert observed is not None
    assert observed["authority_path"] == str(current_path)
    assert observed["pid"] == current_child["pid"]
    assert classified == [stale_child["pid"], current_child["pid"]]
    assert stale_path.is_file()


def test_live_stale_trainer_authority_blocks_concurrent_current_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _write_worker(tmp_path / "outer.py", "raise SystemExit(0)\n")
    spec = _spec_with_control_root(tmp_path, worker)
    _write_synthetic_trainer_authority(
        spec=spec,
        run_id="00-live-stale",
        parent_pid=61001,
        parent_start=301,
        child_pid=61002,
        child_start=302,
    )
    _, current_parent, current_child = _write_synthetic_trainer_authority(
        spec=spec,
        run_id="01-current-live",
        parent_pid=62001,
        parent_start=401,
        child_pid=62002,
        child_start=402,
    )
    proc_calls: list[int] = []
    _install_linux_proc_fixture(
        monkeypatch,
        {
            61002: (302, "S"),
            int(current_child["pid"]): (
                int(current_child["linux_start_ticks"]),
                "S",
            ),
        },
        proc_calls,
    )
    monkeypatch.setattr(
        controller_module,
        "_classify_process_phase",
        lambda snapshot, *, spec, barrier_record: "exp_run_target",
    )
    controller = TasteGINEPersistentController(spec, resume=False)
    with pytest.raises(
        controller_module.TasteGINEControllerError,
        match="PID/start/cwd/cmd/exe/ancestry binding changed",
    ):
        controller._discover_trainer_generation(
            {
                "worker_generation": current_parent,
                "barrier_record": {},
            }
        )
    assert proc_calls == [61002]


def test_malformed_dead_stale_authority_fails_before_liveness_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _write_worker(tmp_path / "outer.py", "raise SystemExit(0)\n")
    spec = _spec_with_control_root(tmp_path, worker)
    stale_path, _, _ = _write_synthetic_trainer_authority(
        spec=spec,
        run_id="00-malformed-dead-stale",
        parent_pid=71001,
        parent_start=501,
        child_pid=71002,
        child_start=502,
    )
    raw = json.loads(stale_path.read_text(encoding="utf-8"))
    raw["unreviewed_field"] = "must-not-be-ignored"
    stale_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    stale_path.chmod(0o600)
    calls: list[int] = []
    monkeypatch.setattr(
        controller_module,
        "_declared_trainer_child_is_live",
        lambda child: calls.append(int(child["pid"])) or False,
    )
    controller = TasteGINEPersistentController(spec, resume=False)
    with pytest.raises(
        controller_module.TasteGINEControllerError,
        match="authority fields differ",
    ):
        controller._discover_trainer_generation(
            {
                "worker_generation": {"pid": 72001},
                "barrier_record": {},
            }
        )
    assert calls == []


def test_monitor_never_retries_while_registered_trainer_child_is_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _write_worker(tmp_path / "outer.py", "raise SystemExit(0)\n")
    spec = _spec(tmp_path, worker)
    launches: list[tuple[object, ...]] = []

    class StopAfterAdoption(BaseException):
        pass

    with TasteGINEPersistentController(spec, resume=False) as controller:
        worker_log = controller.root / "worker-attempt-0.log"
        worker_log.write_bytes(b"")
        worker_log.chmod(0o600)
        state = {
            "attempt": 0,
            "launch_index": 0,
            "barrier_record": {},
            "worker_generation": {"pid": 41001, "linux_start_ticks": 11},
            "worker_log": str(worker_log),
            "worker_log_binding": controller_module._file_binding(
                os.lstat(worker_log)
            ),
        }
        trainer_generation = {
            "pid": 41002,
            "linux_start_ticks": 12,
            "authority_path": str(tmp_path / "trainer_child_authority.json"),
        }
        monkeypatch.setattr(controller, "_terminal_evidence", lambda: None)
        monkeypatch.setattr(
            controller,
            "_discover_trainer_generation",
            lambda value: dict(trainer_generation),
        )
        monkeypatch.setattr(
            controller_module,
            "_generation_alive",
            lambda generation: generation.get("pid") == 41002,
        )
        monkeypatch.setattr(
            controller,
            "_launch",
            lambda *args, **kwargs: launches.append((args, kwargs)),
        )
        monkeypatch.setattr(
            controller_module.time,
            "sleep",
            lambda seconds: (_ for _ in ()).throw(StopAfterAdoption()),
        )
        with pytest.raises(StopAfterAdoption):
            controller._monitor(state, process=None)
        persisted = json.loads(
            (controller.root / STATE_NAME).read_text(encoding="utf-8")
        )
        assert persisted["phase"] == "RUNNING_TRAINER_ADOPTED"
        assert persisted["trainer_generation"] == trainer_generation
        assert launches == []


def test_exp_run_exec_phase_requires_the_exact_reviewed_command(tmp_path: Path) -> None:
    worker = _write_worker(tmp_path / "worker.py", "raise SystemExit(0)\n")
    base = _spec(tmp_path, worker)
    environment = {
        **base.environment_authority,
        "AUTODL_PYTHON": sys.executable,
        "AUTODL_DATA_ROOT": str(tmp_path / "data"),
        "PRIMARY_GNN_BACKBONE": "gine",
        "PRIMARY_SEED": "7",
        "TASTEMOLNET_SPLIT_ROOT": str(tmp_path / "splits"),
        "TASTEMOLNET_GRAPH_CACHE_ROOT": str(tmp_path / "cache"),
        "TASTEMOLNET_POLICY_FILE": str(tmp_path / "policy.yaml"),
        "TASTEMOLNET_POLICY_SHA256": "a" * 64,
        "TASTEMOLNET_POLICY_RECEIPT": str(tmp_path / "receipt.json"),
        "TASTEMOLNET_PREPARED_ROOT": str(tmp_path / "prepared"),
    }
    spec = controller_module.TasteGINEControllerSpec(
        **{
            **{
                name: getattr(base, name)
                for name in base.__dataclass_fields__
            },
            "environment_authority": environment,
        }
    )
    argv = controller_module._expected_exp_run_argv(
        spec,
        gpu_uuid="GPU-1234abcd",
        input_manifest=None,
        resume_training=False,
    )
    snapshot = {
        "argv": list(argv),
        "cwd": str(PROJECT_ROOT),
        "exe": str(Path(sys.executable).resolve()),
    }
    assert controller_module._classify_process_phase(
        snapshot,
        spec=spec,
        barrier_record={"launcher_argv": []},
    ) == "exp_run_target"
    snapshot["argv"] = [*snapshot["argv"], "--unreviewed-extra"]
    with pytest.raises(
        controller_module.TasteGINEControllerError, match="not an allowed exec phase"
    ):
        controller_module._classify_process_phase(
            snapshot,
            spec=spec,
            barrier_record={"launcher_argv": []},
        )
