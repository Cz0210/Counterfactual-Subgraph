from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

import pytest

from src.utils.autodl_exec_startup_barrier import (
    RECORD_KIND,
    RECORD_SCHEMA,
    MAX_RECORD_BYTES,
    StartupBarrierBusy,
    StartupBarrierValidationError,
    arm_exec_startup_barrier,
    reconcile_interrupted_startup_barrier_publication,
    validate_reopenable_unreleased_barrier,
    validate_startup_barrier_record,
    wait_for_unreleased_barrier_lock_quiescent,
)


@pytest.mark.parametrize("window", ("temp_only", "final_and_temp"))
def test_public_prearm_publication_reconciliation_closes_hard_crash_windows(
    tmp_path: Path, window: str
) -> None:
    lock_path = tmp_path / "startup.lock"
    record_path = tmp_path / "startup.json"
    target = _write_target(tmp_path / "must-not-run.txt")
    barrier = arm_exec_startup_barrier(
        lock_path=lock_path,
        record_path=record_path,
        target_argv=target,
    )
    barrier.abort()
    temporary = tmp_path / "startup.json.tmp.injected-crash"
    if window == "temp_only":
        record_path.rename(temporary)
    else:
        os.link(record_path, temporary, follow_symlinks=False)
        with pytest.raises(StartupBarrierValidationError, match="link count"):
            validate_startup_barrier_record(record_path)

    record_remains = reconcile_interrupted_startup_barrier_publication(
        lock_path=lock_path,
        record_path=record_path,
        timeout_seconds=1.0,
    )
    assert record_remains is (window == "final_and_temp")
    assert not temporary.exists()
    if record_remains:
        assert validate_startup_barrier_record(record_path).target_argv == tuple(target)
        assert record_path.stat().st_nlink == 1
    else:
        assert not record_path.exists()


def _write_target(path: Path, value: str = "ran") -> list[str]:
    return [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            f"Path({str(path)!r}).write_text({value!r}, encoding='utf-8')"
        ),
    ]


def _wait(process: subprocess.Popen[str]) -> tuple[int, str]:
    return process.wait(timeout=10), process.stderr.read() if process.stderr else ""


def test_record_writer_enforces_the_frozen_64kib_budget(tmp_path: Path) -> None:
    record_path = tmp_path / "startup.json"
    with pytest.raises(StartupBarrierValidationError, match="byte limit"):
        arm_exec_startup_barrier(
            lock_path=tmp_path / "startup.lock",
            record_path=record_path,
            target_argv=[sys.executable, "-c", "x" * MAX_RECORD_BYTES],
        )
    assert not record_path.exists()
    assert not list(tmp_path.glob("startup.json.tmp.*"))


def test_real_wrapper_blocks_until_exact_parent_release(tmp_path: Path) -> None:
    output = tmp_path / "target-ran.txt"
    lock_path = tmp_path / "startup.lock"
    record_path = tmp_path / "startup.json"
    target = _write_target(output)
    barrier = arm_exec_startup_barrier(
        lock_path=lock_path,
        record_path=record_path,
        target_argv=target,
    )

    raw = json.loads(record_path.read_text(encoding="utf-8"))
    assert raw["schema"] == RECORD_SCHEMA
    assert raw["kind"] == RECORD_KIND
    assert raw["lock_path"] == str(lock_path)
    assert raw["target_argv_sha256"]
    assert raw["launcher_argv_sha256"]
    assert raw["release_read_fd"] in barrier.pass_fds
    assert raw["lock_fd"] in barrier.pass_fds
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(record_path.stat().st_mode) == 0o600

    process = barrier.launch(
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.25)
    assert process.poll() is None
    assert not output.exists()
    with pytest.raises(StartupBarrierBusy):
        wait_for_unreleased_barrier_lock_quiescent(
            record_path,
            timeout_seconds=0.05,
            expected_target_argv=target,
        )

    barrier.release()
    returncode, stderr = _wait(process)
    assert returncode == 0, stderr
    assert output.read_text(encoding="utf-8") == "ran"
    reopened = validate_reopenable_unreleased_barrier(
        record_path,
        expected_target_argv=target,
    )
    assert reopened.target_argv == tuple(target)


def test_parent_abort_sends_eof_never_executes_and_same_paths_can_rearm(
    tmp_path: Path,
) -> None:
    output = tmp_path / "must-not-run.txt"
    lock_path = tmp_path / "startup.lock"
    record_path = tmp_path / "startup.json"
    target = _write_target(output)
    first = arm_exec_startup_barrier(
        lock_path=lock_path,
        record_path=record_path,
        target_argv=target,
    )
    process = first.launch(stderr=subprocess.PIPE, text=True)
    time.sleep(0.1)
    first.abort()
    returncode, _ = _wait(process)
    assert returncode != 0
    assert not output.exists()

    previous = validate_reopenable_unreleased_barrier(
        record_path,
        expected_target_argv=target,
        timeout_seconds=2,
    )
    assert previous.target_argv == tuple(target)
    second = arm_exec_startup_barrier(
        lock_path=lock_path,
        record_path=record_path,
        target_argv=target,
        record_policy="resume",
        rearm_timeout_seconds=2,
        expected_unreleased_record=previous,
    )
    second_process = second.launch(stderr=subprocess.PIPE, text=True)
    second.release()
    returncode, stderr = _wait(second_process)
    assert returncode == 0, stderr
    assert output.read_text(encoding="utf-8") == "ran"


def test_real_parent_hard_exit_before_release_never_executes_and_can_resume(
    tmp_path: Path,
) -> None:
    output = tmp_path / "target-after-resume.txt"
    ready = tmp_path / "coordinator-spawned.txt"
    lock_path = tmp_path / "startup.lock"
    record_path = tmp_path / "startup.json"
    target = _write_target(output)
    coordinator_code = (
        "import os, subprocess; from pathlib import Path; "
        "from src.utils.autodl_exec_startup_barrier import arm_exec_startup_barrier; "
        f"b=arm_exec_startup_barrier(lock_path={str(lock_path)!r}, "
        f"record_path={str(record_path)!r}, target_argv={target!r}); "
        "p=b.launch(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, "
        "start_new_session=True); "
        f"Path({str(ready)!r}).write_text(str(p.pid), encoding='utf-8'); "
        "os._exit(91)"
    )
    coordinator = subprocess.Popen(
        [sys.executable, "-c", coordinator_code],
        cwd=Path(__file__).resolve().parents[2],
    )
    assert coordinator.wait(timeout=10) == 91
    assert ready.is_file()
    old = validate_reopenable_unreleased_barrier(
        record_path,
        expected_target_argv=target,
        timeout_seconds=5,
    )
    assert not output.exists()

    resumed = arm_exec_startup_barrier(
        lock_path=lock_path,
        record_path=record_path,
        target_argv=target,
        record_policy="resume",
        expected_unreleased_record=old,
        rearm_timeout_seconds=5,
    )
    worker = resumed.launch(stderr=subprocess.PIPE, text=True)
    resumed.release()
    returncode, stderr = _wait(worker)
    assert returncode == 0, stderr
    assert output.read_text(encoding="utf-8") == "ran"


@pytest.mark.parametrize("replacement", ["symlink", "new_inode"])
def test_live_barrier_rejects_symlink_or_inode_replacement(
    tmp_path: Path, replacement: str
) -> None:
    output = tmp_path / "must-not-run.txt"
    lock_path = tmp_path / "startup.lock"
    record_path = tmp_path / "startup.json"
    target = _write_target(output)
    barrier = arm_exec_startup_barrier(
        lock_path=lock_path,
        record_path=record_path,
        target_argv=target,
    )
    process = barrier.launch(stderr=subprocess.PIPE, text=True)
    time.sleep(0.1)

    original_inode = barrier.record.lock_inode
    lock_path.unlink()
    if replacement == "symlink":
        victim = tmp_path / "victim.lock"
        victim.touch(mode=0o600)
        lock_path.symlink_to(victim)
    else:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.close(descriptor)
        assert lock_path.stat().st_ino != original_inode

    with pytest.raises(StartupBarrierValidationError):
        barrier.release()
    returncode, _ = _wait(process)
    assert returncode != 0
    assert not output.exists()
    with pytest.raises(StartupBarrierValidationError):
        validate_startup_barrier_record(record_path)


def test_runtime_target_argv_drift_is_rejected_before_exec(tmp_path: Path) -> None:
    expected_output = tmp_path / "expected.txt"
    drift_output = tmp_path / "drift.txt"
    lock_path = tmp_path / "startup.lock"
    record_path = tmp_path / "startup.json"
    expected_target = _write_target(expected_output, "expected")
    barrier = arm_exec_startup_barrier(
        lock_path=lock_path,
        record_path=record_path,
        target_argv=expected_target,
    )

    drifted_launcher = list(barrier.launcher_argv)
    drifted_launcher[-1] = _write_target(drift_output, "drift")[-1]
    process = subprocess.Popen(
        drifted_launcher,
        pass_fds=barrier.pass_fds,
        close_fds=True,
        stderr=subprocess.PIPE,
        text=True,
    )
    barrier.mark_spawned()
    # The child may reject the launcher before or while the parent writes.  A
    # BrokenPipe-style release error is acceptable and remains fail-closed.
    try:
        barrier.release()
    except OSError:
        pass
    returncode, _ = _wait(process)
    assert returncode != 0
    assert not expected_output.exists()
    assert not drift_output.exists()

    with pytest.raises(StartupBarrierValidationError, match="target argv"):
        validate_reopenable_unreleased_barrier(
            record_path,
            expected_target_argv=_write_target(drift_output, "drift"),
            timeout_seconds=2,
        )


def test_durable_record_target_digest_tamper_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "must-not-run.txt"
    record_path = tmp_path / "startup.json"
    barrier = arm_exec_startup_barrier(
        lock_path=tmp_path / "startup.lock",
        record_path=record_path,
        target_argv=_write_target(output),
    )
    barrier.abort()
    raw = json.loads(record_path.read_text(encoding="utf-8"))
    raw["target_argv"][-1] += "; raise SystemExit(0)"
    record_path.write_text(json.dumps(raw), encoding="utf-8")
    os.chmod(record_path, 0o600)

    with pytest.raises(StartupBarrierValidationError, match="target argv digest"):
        validate_startup_barrier_record(record_path)
    assert not output.exists()


def test_arm_never_silently_overwrites_existing_record_or_target_drift(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "startup.lock"
    record_path = tmp_path / "startup.json"
    original_target = _write_target(tmp_path / "original.txt", "original")
    drifted_target = _write_target(tmp_path / "drift.txt", "drift")
    first = arm_exec_startup_barrier(
        lock_path=lock_path,
        record_path=record_path,
        target_argv=original_target,
    )
    first.abort()
    original_bytes = record_path.read_bytes()

    with pytest.raises(StartupBarrierValidationError, match="fresh barrier"):
        arm_exec_startup_barrier(
            lock_path=lock_path,
            record_path=record_path,
            target_argv=original_target,
        )
    assert record_path.read_bytes() == original_bytes

    with pytest.raises(StartupBarrierValidationError, match="target argv"):
        arm_exec_startup_barrier(
            lock_path=lock_path,
            record_path=record_path,
            target_argv=drifted_target,
            record_policy="resume",
            expected_unreleased_record=first.record,
        )
    assert record_path.read_bytes() == original_bytes

    record_path.write_text("{not-json", encoding="utf-8")
    os.chmod(record_path, 0o600)
    corrupted_bytes = record_path.read_bytes()
    with pytest.raises(StartupBarrierValidationError, match="valid JSON"):
        arm_exec_startup_barrier(
            lock_path=lock_path,
            record_path=record_path,
            target_argv=original_target,
            record_policy="resume",
            expected_unreleased_record=first.record,
        )
    assert record_path.read_bytes() == corrupted_bytes


def test_fresh_arm_reconciles_sigkill_before_no_replace_publication(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "startup.lock"
    record_path = tmp_path / "startup.json"
    target = _write_target(tmp_path / "target.txt")
    abandoned = arm_exec_startup_barrier(
        lock_path=lock_path,
        record_path=record_path,
        target_argv=target,
    )
    abandoned.abort()
    pre_link_temp = Path(f"{record_path}.tmp.999.deadbeef")
    record_path.replace(pre_link_temp)
    assert not record_path.exists()

    recovered = arm_exec_startup_barrier(
        lock_path=lock_path,
        record_path=record_path,
        target_argv=target,
    )
    assert record_path.exists()
    assert not pre_link_temp.exists()
    recovered.abort()


def test_resume_reconciles_sigkill_after_link_before_temp_unlink(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "startup.lock"
    record_path = tmp_path / "startup.json"
    target = _write_target(tmp_path / "target.txt")
    abandoned = arm_exec_startup_barrier(
        lock_path=lock_path,
        record_path=record_path,
        target_argv=target,
    )
    abandoned.abort()
    linked_temp = Path(f"{record_path}.tmp.999.deadbeef")
    os.link(record_path, linked_temp)
    assert record_path.stat().st_nlink == 2

    recovered = arm_exec_startup_barrier(
        lock_path=lock_path,
        record_path=record_path,
        target_argv=target,
        record_policy="resume",
        rearm_timeout_seconds=2,
        expected_unreleased_record=abandoned.record,
    )
    assert not linked_temp.exists()
    assert record_path.stat().st_nlink == 1
    recovered.abort()


@pytest.mark.parametrize(
    "dangerous_kwargs",
    [
        {"shell": True},
        {"executable": sys.executable},
        {"preexec_fn": lambda: None},
    ],
)
def test_launch_rejects_subprocess_options_that_can_bypass_wrapper(
    tmp_path: Path, dangerous_kwargs: dict[str, object]
) -> None:
    barrier = arm_exec_startup_barrier(
        lock_path=tmp_path / "startup.lock",
        record_path=tmp_path / "startup.json",
        target_argv=_write_target(tmp_path / "must-not-run.txt"),
    )
    with pytest.raises(TypeError, match="launch owns"):
        barrier.launch(**dangerous_kwargs)
    barrier.abort()


def test_resume_requires_explicit_typed_prior_unreleased_authority(
    tmp_path: Path,
) -> None:
    target = _write_target(tmp_path / "must-not-run.txt")
    first = arm_exec_startup_barrier(
        lock_path=tmp_path / "startup.lock",
        record_path=tmp_path / "startup.json",
        target_argv=target,
    )
    first.abort()
    with pytest.raises(StartupBarrierValidationError, match="typed"):
        arm_exec_startup_barrier(
            lock_path=tmp_path / "startup.lock",
            record_path=tmp_path / "startup.json",
            target_argv=target,
            record_policy="resume",
        )
