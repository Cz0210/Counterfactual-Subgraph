from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from scripts.autodl import build_main_ready_task_specs as builder
from scripts.autodl import dispatch_bound_task_once as dispatcher
from scripts.autodl import hot_bind_main_ready_task_specs as hot_bind
from scripts.autodl import run_mut_clean_trace_equivalence_v1 as mut_owner
from scripts.autodl import run_t12_reference_500_v1 as t12_owner
from scripts.autodl import run_t14_checkpoint12500_audit_owner as t14_owner
from src.utils.main_ready_task_specs import (
    MainReadyTaskSpecError,
    TASK_SPEC_PATH_TOKEN,
    atomic_json,
    command_from_spec,
    conflicting_output_writers,
    load_pointer,
    materialize_task_spec_path,
    process_identity,
    probe_owner,
    seal_pointer,
    seal_spec,
    stable_sha256,
    validate_manifest,
    validate_spec,
)


SHA = "a" * 64
COMMIT = "b" * 40
ATTEMPT = "123e4567-e89b-42d3-a456-426614174000"


def _raw_spec(tmp_path: Path, *, task_id: str = "t12-reference-500") -> dict:
    root = tmp_path.resolve()
    return {
        "schema_version": "ignored-until-sealed",
        "task_id": task_id,
        "task_kind": "T12_REFERENCE_ACCELERATED_PARITY_AND_FULL",
        "attempt_uuid": ATTEMPT,
        "repo_root": str(root / "repo"),
        "execution_commit": COMMIT,
        "python": str(root / "python"),
        "entrypoint": str(root / "repo" / "scripts" / "owner.py"),
        "config_path": str(root / "repo" / "configs" / "hpc.yaml"),
        "config_sha256": SHA,
        "manifest_path": str(root / "science-manifest.json"),
        "manifest_sha256": SHA,
        "input_roots": {"checkpoint": str(root / "checkpoint")},
        "input_hashes": {"checkpoint": SHA},
        "output_root": str(root / "output"),
        "gpu_request": {"index": 3, "uuid": "GPU-fixture"},
        "cpu_request": {"workers": 1},
        "memory_request": {"minimum_headroom_bytes": 1},
        "required_environment": {"RUN_GNN_ABLATION": "0"},
        "matrix_authority_root": str(root / "matrix-authority"),
        "expected_owner_command_sha256": "0" * 64,
        "expected_heartbeat_path": str(root / "control" / "heartbeat.json"),
        "expected_pid_file": str(root / "control" / "owner.pid.json"),
        "resume_policy": "fresh_reference_then_bound_resume",
        "single_writer_policy": "fail_if_live_owner_or_output_writer",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": "0" * 64,
        "arguments": ["owner", "--task-spec", TASK_SPEC_PATH_TOKEN],
        "owner_timeout_seconds": 60,
    }


def _write_fake_process(
    proc_root: Path,
    *,
    pid: int,
    start_ticks: int,
    argv: list[str],
    cwd: Path,
    comm: str = "owner worker with spaces",
) -> None:
    process = proc_root / str(pid)
    process.mkdir(parents=True)
    process.joinpath("stat").write_text(
        f"{pid} ({comm}) S " + " ".join(["0"] * 18 + [str(start_ticks)]) + "\n",
        encoding="utf-8",
    )
    process.joinpath("cmdline").write_bytes(b"\0".join(item.encode() for item in argv) + b"\0")
    process.joinpath("cwd").symlink_to(cwd)
    process.joinpath("fd").mkdir()


def _sealed(tmp_path: Path) -> dict:
    path = (tmp_path / "bundle" / "t12-reference-500.json").resolve()
    return seal_spec(materialize_task_spec_path(_raw_spec(tmp_path), path))


def test_task_spec_self_hash_and_safe_task_id_fail_closed(tmp_path: Path) -> None:
    spec = _sealed(tmp_path)
    validate_spec(spec, check_files=False)
    changed = dict(spec)
    changed["cpu_request"] = {"workers": 99}
    with pytest.raises(MainReadyTaskSpecError, match="self hash"):
        validate_spec(changed, check_files=False)
    unsafe = _raw_spec(tmp_path, task_id="../escape")
    unsafe = materialize_task_spec_path(unsafe, (tmp_path / "escape.json").resolve())
    with pytest.raises(MainReadyTaskSpecError, match="safe path component"):
        seal_spec(unsafe)


def test_descriptor_token_binds_final_spec_path_before_command_hash(tmp_path: Path) -> None:
    final = (tmp_path / "published" / "t12-reference-500.json").resolve()
    spec = seal_spec(materialize_task_spec_path(_raw_spec(tmp_path), final))
    assert TASK_SPEC_PATH_TOKEN not in spec["arguments"]
    assert spec["arguments"][-1] == str(final)
    assert spec["expected_owner_command_sha256"] == stable_sha256(command_from_spec(spec))


def test_builder_publishes_complete_root_and_materializes_final_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _raw_spec(tmp_path)
    config = Path(raw["config_path"])
    manifest = Path(raw["manifest_path"])
    config.parent.mkdir(parents=True)
    config.write_text("fixture: true\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    descriptor = (tmp_path / "descriptor.json").resolve()
    descriptor.write_text(json.dumps(raw), encoding="utf-8")
    output = (tmp_path / "published").resolve()

    observed: dict[str, object] = {}

    def fake_manifest(paths: list[Path], *, published_paths: list[Path]) -> dict:
        staged = json.loads(paths[0].read_text(encoding="utf-8"))
        observed["argument"] = staged["arguments"][-1]
        observed["published"] = str(published_paths[0])
        return {
            "schema_version": "fixture_manifest",
            "task_specs": [str(published_paths[0])],
        }

    monkeypatch.setattr(builder, "manifest_for_specs", fake_manifest)
    assert builder.main(["--descriptor", str(descriptor), "--output-root", str(output)]) == 0
    assert observed["argument"] == str(output / "t12-reference-500.json")
    assert observed["published"] == observed["argument"]
    assert output.is_dir()
    assert not list(tmp_path.glob(".published.*.tmp"))


def test_builder_removes_staging_and_never_publishes_partial_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _raw_spec(tmp_path)
    config = Path(raw["config_path"])
    manifest = Path(raw["manifest_path"])
    config.parent.mkdir(parents=True)
    config.write_text("fixture: true\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    descriptor = (tmp_path / "descriptor.json").resolve()
    descriptor.write_text(json.dumps(raw), encoding="utf-8")
    output = (tmp_path / "published").resolve()

    def fail_manifest(*_args: object, **_kwargs: object) -> dict:
        raise MainReadyTaskSpecError("fixture failure")

    monkeypatch.setattr(builder, "manifest_for_specs", fail_manifest)
    with pytest.raises(MainReadyTaskSpecError, match="fixture failure"):
        builder.main(["--descriptor", str(descriptor), "--output-root", str(output)])
    assert not output.exists()
    assert not list(tmp_path.glob(".published.*.tmp"))


def test_process_identity_parses_comm_with_spaces(tmp_path: Path) -> None:
    proc = (tmp_path / "proc").resolve()
    cwd = (tmp_path / "repo").resolve()
    cwd.mkdir()
    argv = ["/python", "-I", "owner.py"]
    _write_fake_process(proc, pid=701, start_ticks=9_876_543, argv=argv, cwd=cwd)
    identity = process_identity(701, proc_root=proc)
    assert identity is not None
    assert identity["start_ticks"] == 9_876_543
    assert identity["command_sha256"] == stable_sha256(argv)


def test_owner_probe_requires_fresh_heartbeat_ticks_cwd_and_command(tmp_path: Path) -> None:
    spec = _sealed(tmp_path)
    proc = (tmp_path / "proc").resolve()
    repo = Path(spec["repo_root"])
    repo.mkdir(parents=True)
    argv = command_from_spec(spec)
    _write_fake_process(proc, pid=702, start_ticks=1_234, argv=argv, cwd=repo)
    pid_file = Path(spec["expected_pid_file"])
    heartbeat = Path(spec["expected_heartbeat_path"])
    atomic_json(pid_file, {"owner_pid": 702})
    now = datetime.now(timezone.utc)
    atomic_json(
        heartbeat,
        {
            "task_id": spec["task_id"],
            "owner_pid": 702,
            "owner_start_ticks": 1_234,
            "output_root": spec["output_root"],
            "written_at": now.isoformat(),
        },
    )
    assert probe_owner(spec, proc_root=proc, now_epoch=now.timestamp())["state"] == "OWNER_CONFIRMED"

    stale = now - timedelta(seconds=61)
    atomic_json(
        heartbeat,
        {
            "task_id": spec["task_id"],
            "owner_pid": 702,
            "owner_start_ticks": 1_234,
            "output_root": spec["output_root"],
            "written_at": stale.isoformat(),
        },
    )
    evidence = probe_owner(spec, proc_root=proc, now_epoch=now.timestamp())
    assert evidence["state"] == "INVALID"
    assert "stale" in evidence["reason"]


def test_same_root_writer_is_detected_for_equals_argument_and_open_fd(tmp_path: Path) -> None:
    proc = (tmp_path / "proc").resolve()
    cwd = (tmp_path / "elsewhere").resolve()
    cwd.mkdir()
    output = (tmp_path / "science-output").resolve()
    output.mkdir()
    _write_fake_process(
        proc,
        pid=703,
        start_ticks=2_345,
        argv=["python", "worker.py", f"--output-root={output}"],
        cwd=cwd,
    )
    conflicts = conflicting_output_writers(output, proc_root=proc)
    assert [row["pid"] for row in conflicts] == [703]
    assert conflicts[0]["argv_match"] is True

    proc2 = (tmp_path / "proc2").resolve()
    _write_fake_process(
        proc2,
        pid=704,
        start_ticks=3_456,
        argv=["python", "other.py"],
        cwd=cwd,
    )
    proc2.joinpath("704", "fd", "8").symlink_to(output / "active.jsonl")
    conflicts = conflicting_output_writers(output, proc_root=proc2)
    assert conflicts[0]["fd_match"] is True


def test_unconfirmed_live_launcher_is_never_retryable(tmp_path: Path) -> None:
    proc = (tmp_path / "proc").resolve()
    cwd = (tmp_path / "repo").resolve()
    cwd.mkdir()
    _write_fake_process(proc, pid=705, start_ticks=4_567, argv=["python", "owner.py"], cwd=cwd)
    child = SimpleNamespace(pid=705, poll=lambda: None)
    evidence = dispatcher._launcher_state(child, proc_root=proc)
    assert evidence["state"] == "BLOCKED_UNCONFIRMED_LIVE_LAUNCHER"
    assert evidence["cleanup"]["retry_allowed"] is False
    assert evidence["cleanup"]["signal_sent"] is False


def test_exited_launcher_is_reaped_before_retry(tmp_path: Path) -> None:
    calls: list[str] = []
    child = SimpleNamespace(pid=706, poll=lambda: 17, wait=lambda: calls.append("wait") or 17)
    evidence = dispatcher._launcher_state(child, proc_root=(tmp_path / "empty-proc").resolve())
    assert calls == ["wait"]
    assert evidence["cleanup"]["launcher_reaped"] is True
    assert evidence["cleanup"]["retry_allowed"] is True


def test_pointer_self_hash_and_manifest_binding_are_fail_closed(tmp_path: Path) -> None:
    manifest_path = (tmp_path / "task_specs_manifest.json").resolve()
    manifest = {
        "schema_version": "main_ready_task_specs_manifest_v1",
        "matrix_authority_root": str((tmp_path / "matrix").resolve()),
        "task_specs": [
            {
                "task_id": "t12",
                "task_kind": "T12",
                "path": str((tmp_path / "t12.json").resolve()),
                "file_sha256": "1" * 64,
                "spec_sha256": "2" * 64,
            }
        ],
    }
    manifest["manifest_sha256"] = stable_sha256(manifest)
    atomic_json(manifest_path, manifest)
    validate_manifest(manifest)
    pointer = seal_pointer(
        {
            "manifest_path": str(manifest_path),
            "manifest_file_sha256": __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest(),
            "manifest_sha256": manifest["manifest_sha256"],
            "sidecar_control_root": str((tmp_path / "control").resolve()),
            "sidecar_pid": 707,
            "sidecar_start_ticks": 5_678,
            "sidecar_command_sha256": "3" * 64,
            "published_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    pointer_path = (tmp_path / "pointer.json").resolve()
    atomic_json(pointer_path, pointer)
    assert load_pointer(pointer_path)["pointer_sha256"] == pointer["pointer_sha256"]
    changed = json.loads(pointer_path.read_text(encoding="utf-8"))
    changed["sidecar_pid"] = 999
    pointer_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(MainReadyTaskSpecError, match="self hash"):
        load_pointer(pointer_path)


def test_hot_bind_sidecar_probe_checks_pid_generation_command_and_freshness(tmp_path: Path) -> None:
    root = (tmp_path / "sidecar").resolve()
    root.mkdir()
    proc = (tmp_path / "proc").resolve()
    argv = ["python", "run-main-sidecar.py"]
    _write_fake_process(proc, pid=708, start_ticks=6_789, argv=argv, cwd=tmp_path.resolve())
    atomic_json(
        root / "heartbeat.json",
        {
            "controller_pid": 708,
            "written_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    heartbeat, identity = hot_bind._validate_live_sidecar(
        root=root,
        expected_pid=708,
        expected_start_ticks=6_789,
        expected_command_sha256=stable_sha256(argv),
        heartbeat_max_age_seconds=120,
        proc_root=proc,
    )
    assert heartbeat["controller_pid"] == identity["pid"]
    with pytest.raises(RuntimeError, match="PID generation"):
        hot_bind._validate_live_sidecar(
            root=root,
            expected_pid=708,
            expected_start_ticks=6_790,
            expected_command_sha256=stable_sha256(argv),
            heartbeat_max_age_seconds=120,
            proc_root=proc,
        )
    atomic_json(root / "heartbeat.json", {"controller_pid": 708, "written_at_unix": 1})
    with pytest.raises(RuntimeError, match="stale"):
        hot_bind._validate_live_sidecar(
            root=root,
            expected_pid=708,
            expected_start_ticks=6_789,
            expected_command_sha256=stable_sha256(argv),
            heartbeat_max_age_seconds=120,
            proc_root=proc,
        )


def test_conda_style_python_symlink_is_allowed_but_entrypoint_symlink_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _sealed(tmp_path)
    repo = Path(spec["repo_root"])
    entrypoint = Path(spec["entrypoint"])
    config = Path(spec["config_path"])
    manifest = Path(spec["manifest_path"])
    checkpoint = Path(spec["input_roots"]["checkpoint"])
    entrypoint.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir()
    entrypoint.write_text("pass\n", encoding="utf-8")
    config.write_text("fixture: true\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    target_python = (tmp_path / "python3.10").resolve()
    target_python.write_text("#!/bin/sh\n", encoding="utf-8")
    target_python.chmod(0o755)
    Path(spec["python"]).symlink_to(target_python)
    raw = dict(spec)
    raw["config_sha256"] = __import__("hashlib").sha256(config.read_bytes()).hexdigest()
    raw["manifest_sha256"] = __import__("hashlib").sha256(manifest.read_bytes()).hexdigest()
    spec = seal_spec(raw)
    monkeypatch.setattr(
        "src.utils.main_ready_task_specs.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=COMMIT + "\n"),
    )
    validate_spec(spec, check_files=True)
    replacement = (tmp_path / "owner-real.py").resolve()
    replacement.write_text("pass\n", encoding="utf-8")
    entrypoint.unlink()
    entrypoint.symlink_to(replacement)
    with pytest.raises(MainReadyTaskSpecError, match="entrypoint input"):
        validate_spec(spec, check_files=True)


def test_attempt_must_be_uuid4(tmp_path: Path) -> None:
    raw = _raw_spec(tmp_path)
    raw["attempt_uuid"] = str(UUID(int=0))
    raw = materialize_task_spec_path(raw, (tmp_path / "spec.json").resolve())
    with pytest.raises(MainReadyTaskSpecError, match="UUIDv4"):
        seal_spec(raw)


@pytest.mark.parametrize("owner", [mut_owner, t12_owner, t14_owner])
def test_owner_wrappers_use_shared_robust_process_identity(
    owner: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        owner,
        "process_identity",
        lambda _pid: {"pid": 800, "alive": True, "start_ticks": 98_765},
    )
    monkeypatch.setattr(owner.os, "getpid", lambda: 800)
    assert owner._owner_identity() == (800, 98_765)


def test_mut_owner_takes_bound_nonblocking_exclusive_gpu_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(mut_owner.fcntl, "flock", lambda _fd, flags: calls.append(flags))
    lease_path = (tmp_path / "main-ready-dispatch-leases" / "gpu0.lock").resolve()
    spec = {
        "gpu_request": {
            "index": 0,
            "lease_path": str(lease_path),
            "lease_scope": "MAIN_READY_DISPATCH_OWNER",
        }
    }
    lease = mut_owner._acquire_gpu_lease(spec)
    try:
        assert calls == [mut_owner.fcntl.LOCK_EX | mut_owner.fcntl.LOCK_NB]
    finally:
        lease.close()


def test_mut_heartbeat_truthfully_reports_dispatch_lease(tmp_path: Path) -> None:
    spec = _sealed(tmp_path)
    payload = mut_owner._heartbeat_payload(
        spec,
        owner_pid=801,
        owner_ticks=87_654,
        output=Path(spec["output_root"]),
        state={"child_pid": 802, "phase": "RUNNING"},
        sequence=3,
    )
    assert payload["dispatch_lease_held"] is True
    assert payload["worker_gpu_lock_managed_internally"] is True
    assert "gpu_lock_held" not in payload
    assert payload["gpu_request"] == spec["gpu_request"]
    assert payload["owner_start_ticks"] == 87_654
    assert payload["science_gpu_selected_by_reviewed_worker"] is True


def test_mut_wrapper_rejects_worker_global_gpu_lock_namespace(tmp_path: Path) -> None:
    spec = {
        "gpu_request": {
            "index": 0,
            "lease_path": str((tmp_path / "runtime" / "locks" / "gpu0.lock").resolve()),
            "lease_scope": "MAIN_READY_DISPATCH_OWNER",
        }
    }
    with pytest.raises(ValueError, match="main-ready-dispatch-leases"):
        mut_owner._acquire_gpu_lease(spec)


def test_t12_terminal_binds_output_root_and_pid_generation(tmp_path: Path) -> None:
    spec = _sealed(tmp_path)
    terminal = t12_owner._owner_terminal(
        spec=spec,
        state={"phase": "REFERENCE_500_AND_RELOAD_510_PASS", "step": 510},
        owner_pid=803,
        owner_ticks=76_543,
    )
    assert terminal["task_id"] == spec["task_id"]
    assert terminal["output_root"] == spec["output_root"]
    assert terminal["owner_start_ticks"] == 76_543
    assert terminal["gpu_lock_held"] is False
