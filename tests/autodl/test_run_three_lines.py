from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from scripts.autodl import run_three_lines as orchestrator
from scripts.autodl.run_three_lines import (
    LoadedSpec,
    OrchestratorError,
    RUN_ID,
    resume,
    start,
    status,
    stop,
)


requires_linux_procfs = pytest.mark.skipif(
    not Path("/proc/self/stat").is_file(),
    reason="real worker identity integration requires Linux procfs",
)


def _git_repo(path: Path) -> tuple[str, str]:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "AutoDL Test"],
        check=True,
    )
    (path / "README").write_text("test\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "README"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "init"], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "-C", str(path), "branch", "--show-current"], text=True
    ).strip()
    return commit, branch


SUCCESS_CODE = """
import json, os, pathlib, time
path = pathlib.Path(os.environ['THREE_LINES_REQUIRED_SUCCESS_SENTINEL'])
path.parent.mkdir(parents=True, exist_ok=True)
time.sleep(float(os.environ.get('TEST_DELAY', '0')))
payload = {
    'status': 'PASS',
    'completed_step': 50000,
    'canonical_method_count': 4,
    'disallow_generation': os.environ.get('DISALLOW_GENERATION'),
    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
}
path.write_text(json.dumps(payload) + '\\n', encoding='utf-8')
"""


FAIL_CODE = "import sys; sys.exit(17)"


def _stage(
    stage_id: str,
    output_root: Path,
    *,
    dependencies: list[str] | None = None,
    command: list[str] | None = None,
    resume_command: list[str] | None = None,
    environment: dict[str, str] | None = None,
) -> dict:
    sentinel = output_root / f"{stage_id}.json"
    fields: dict[str, object] = {"status": "PASS"}
    if stage_id == "bace_generation":
        fields["completed_step"] = 50000
    if stage_id == "bace_common4":
        fields["canonical_method_count"] = 4
    return {
        "id": stage_id,
        "dependencies": dependencies or [],
        "command": command or [sys.executable, "-c", SUCCESS_CODE],
        "resume_command": resume_command or [sys.executable, "-c", SUCCESS_CODE],
        "environment": environment or {},
        "required_success_sentinel": str(sentinel),
        "required_success_fields": fields,
        "output_manifest": str(sentinel),
    }


def _write_spec(
    tmp_path: Path,
    *,
    first_commands: dict[str, list[str]] | None = None,
    delay: str = "0",
) -> Path:
    project = tmp_path / "project"
    external = tmp_path / "external"
    _commit, branch = _git_repo(project)
    external_commit, _external_branch = _git_repo(external)
    persistent = tmp_path / "persistent"
    fast = tmp_path / "fast"
    first_commands = first_commands or {}

    lanes = []
    topology = [
        ("mut_recovery", 0, "Mutagenicity", "preserved_freeze_only"),
        ("aids_recovery", 1, "AIDS", "preserved_freeze_only"),
        ("bace_comrecgc", 2, "BACE", "fresh_step0_checkpointed"),
        ("bace_globalgce_common4", 3, "BACE", "reuse_frozen_inputs_only"),
    ]
    for lane_id, gpu_id, dataset, generation_policy in topology:
        input_root = persistent / "inputs" / lane_id
        input_root.mkdir(parents=True)
        input_payload = input_root / "fixture.txt"
        input_payload.write_text(f"{lane_id}\n", encoding="utf-8")
        manifest = input_root / "MANIFEST.sha256"
        manifest.write_text(
            f"{hashlib.sha256(input_payload.read_bytes()).hexdigest()}  fixture.txt\n",
            encoding="utf-8",
        )
        output_root = persistent / "outputs" / lane_id
        base_environment = {"TEST_DELAY": delay}
        if dataset in {"Mutagenicity", "AIDS"}:
            base_environment["DISALLOW_GENERATION"] = "1"
        if lane_id == "bace_comrecgc":
            stages = [
                _stage(
                    "bace_generation",
                    output_root,
                    command=first_commands.get(lane_id),
                    environment=base_environment,
                ),
                _stage(
                    "bace_comrecgc_final",
                    output_root,
                    dependencies=["bace_comrecgc:bace_generation"],
                    environment=base_environment,
                ),
            ]
        elif lane_id == "bace_globalgce_common4":
            stages = [
                _stage(
                    "bace_globalgce_wnode",
                    output_root,
                    command=first_commands.get(lane_id),
                    environment=base_environment,
                ),
                _stage(
                    "bace_common4",
                    output_root,
                    dependencies=[
                        "bace_comrecgc:bace_comrecgc_final",
                        "bace_globalgce_common4:bace_globalgce_wnode",
                    ],
                    environment=base_environment,
                ),
            ]
        else:
            stages = [
                _stage(
                    f"{lane_id}_stage",
                    output_root,
                    command=first_commands.get(lane_id),
                    environment=base_environment,
                )
            ]
        lanes.append(
            {
                "id": lane_id,
                "gpu_id": gpu_id,
                "dataset": dataset,
                "method": (
                    "ComRecGC"
                    if lane_id != "bace_globalgce_common4"
                    else "GlobalGCE+common4"
                ),
                "generation_policy": generation_policy,
                "input_root": str(input_root),
                "input_manifest": str(manifest),
                "output_root": str(output_root),
                "cache_root": str(fast / "cache" / lane_id),
                "active_root": str(fast / "active" / lane_id),
                "stages": stages,
            }
        )
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "roots": {
            "project": str(project),
            "external_comrecgc": str(external),
            "persistent_run": str(persistent),
            "fast_run": str(fast),
        },
        "provenance": {
            "branch": branch,
            "code_commit": "HEAD",
            "external_comrecgc_commit": external_commit,
        },
        "runtime": {
            "require_nvidia_smi": False,
            "require_gpu_count": 4,
            "heartbeat_seconds": 0.05,
            "dependency_poll_seconds": 0.05,
            "stop_grace_seconds": 1,
        },
        "policy": {
            "strict_three_line_topology": True,
            "require_input_read_only": False,
            "require_clean_code": True,
            "require_clean_external": True,
        },
        "lanes": lanes,
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return spec_path


def _write_bound_run_state(spec: LoadedSpec, *, status_value: str = "BLOCKED") -> None:
    spec.state_root.mkdir(parents=True, exist_ok=True)
    orchestrator.atomic_write_json(
        spec.run_state_path,
        {
            **orchestrator._spec_state_binding(spec),
            "status": status_value,
            "created_at": orchestrator.utc_now(),
            "updated_at": orchestrator.utc_now(),
            "commands": orchestrator._canonical_commands(spec),
            "lanes": {},
            "slurm_jobs": [],
            "autodl_pid_is_slurm_job_id": False,
        },
    )


def _wait_for_status(spec: LoadedSpec, expected: set[str], timeout: float = 12) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        last = status(spec)
        if last["status"] in expected:
            return last
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {expected}; last={last}")


def _wait_for_lane_status(
    spec: LoadedSpec, lane_id: str, expected: set[str], timeout: float = 12
) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        last = status(spec)
        if last["lanes"][lane_id]["status"] in expected:
            return last
        time.sleep(0.05)
    raise AssertionError(
        f"Timed out waiting for {lane_id}={expected}; last={last}"
    )


@requires_linux_procfs
def test_lanes_start_incrementally_and_common4_waits_for_cross_lane_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "_pid_matches_worker",
        lambda pid, _spec, _lane: orchestrator._pid_alive(pid),
    )
    spec = LoadedSpec.load(_write_spec(tmp_path))

    first = start(spec, ["mut_recovery"])
    assert set(first["launched_pids"]) == {"mut_recovery"}
    partial = _wait_for_lane_status(spec, "mut_recovery", {"SUCCEEDED"})
    assert partial["status"] == "PARTIALLY_COMPLETED"
    assert partial["not_started_lanes"] == [
        "aids_recovery",
        "bace_comrecgc",
        "bace_globalgce_common4",
    ]

    aids = resume(spec, ["aids_recovery"])
    assert set(aids["resumed_pids"]) == {"aids_recovery"}
    assert _wait_for_lane_status(spec, "aids_recovery", {"SUCCEEDED"})[
        "lanes"
    ]["aids_recovery"]["retry_count"] == 0

    globalgce = resume(spec, ["bace_globalgce_common4"])
    assert set(globalgce["resumed_pids"]) == {"bace_globalgce_common4"}
    waiting = _wait_for_lane_status(
        spec, "bace_globalgce_common4", {"WAITING_DEPENDENCY"}
    )
    assert waiting["lanes"]["bace_comrecgc"]["status"] == "NOT_STARTED"

    bace = resume(spec, ["bace_comrecgc"])
    assert set(bace["resumed_pids"]) == {"bace_comrecgc"}
    final = _wait_for_status(spec, {"LANES_COMPLETED"})
    assert final["not_started_lanes"] == []
    assert all(value["retry_count"] == 0 for value in final["lanes"].values())


@requires_linux_procfs
def test_repeated_lane_selection_is_idempotent_and_does_not_start_other_lanes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "_pid_matches_worker",
        lambda pid, _spec, _lane: orchestrator._pid_alive(pid),
    )
    spec = LoadedSpec.load(_write_spec(tmp_path, delay="2"))
    first = start(spec, ["mut_recovery", "mut_recovery"])
    first_pid = first["launched_pids"]["mut_recovery"]
    assert len(first["launched_pids"]) == 1

    repeated = resume(spec, ["mut_recovery", "mut_recovery"])
    assert repeated["resumed_pids"] == {}
    assert repeated["skipped_lanes"] == {"mut_recovery": "ALREADY_ACTIVE"}
    current = status(spec)
    assert current["lanes"]["mut_recovery"]["worker_pid"] in {None, first_pid}
    assert current["lanes"]["mut_recovery"]["retry_count"] == 0
    assert set(current["not_started_lanes"]) == set(spec.lane_by_id) - {
        "mut_recovery"
    }
    assert all(
        not (spec.lane_state_root(lane_id) / "worker_pid.json").exists()
        for lane_id in current["not_started_lanes"]
    )
    stop(spec)


@requires_linux_procfs
def test_selected_lane_preflight_ignores_unready_unselected_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "_pid_matches_worker",
        lambda pid, _spec, _lane: orchestrator._pid_alive(pid),
    )
    spec = LoadedSpec.load(_write_spec(tmp_path, delay="2"))
    aids_manifest = Path(spec.lane_by_id["aids_recovery"]["input_manifest"])
    aids_manifest.unlink()

    started = start(spec, ["mut_recovery"])
    assert set(started["launched_pids"]) == {"mut_recovery"}
    with pytest.raises(OrchestratorError, match="Input manifest missing"):
        resume(spec, ["aids_recovery"])
    current = status(spec)
    assert current["lanes"]["aids_recovery"]["status"] == "NOT_STARTED"
    assert current["lanes"]["mut_recovery"]["status"] in {
        "STARTING",
        "RUNNING",
        "SUCCEEDED",
    }
    stop(spec)


def test_lane_cli_is_repeatable() -> None:
    parser = orchestrator.build_parser()
    parsed = parser.parse_args(
        ["start", "--lane", "mut_recovery", "--lane", "aids_recovery"]
    )
    assert parsed.lane == ["mut_recovery", "aids_recovery"]


@requires_linux_procfs
def test_four_lanes_complete_with_common4_dependency_and_autodl_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "_pid_matches_worker",
        lambda pid, _spec, _lane: orchestrator._pid_alive(pid),
    )
    spec = LoadedSpec.load(_write_spec(tmp_path))
    launched = start(spec)
    assert sorted(launched["launched_pids"]) == sorted(spec.lane_by_id)
    final = _wait_for_status(spec, {"LANES_COMPLETED"})
    assert final["autodl_pid_is_slurm_job_id"] is False
    assert final["slurm_jobs"] == []

    for lane_id, lane in spec.lane_by_id.items():
        lane_status = final["lanes"][lane_id]
        assert lane_status["status"] == "SUCCEEDED"
        lane_success = spec.lane_state_root(lane_id) / "LANE_SUCCESS.json"
        assert json.loads(lane_success.read_text(encoding="utf-8"))["status"] == "PASS"
        scientific = json.loads(
            Path(lane["stages"][0]["required_success_sentinel"]).read_text(
                encoding="utf-8"
            )
        )
        assert scientific["cuda_visible_devices"] == str(lane["gpu_id"])
        if lane["dataset"] in {"Mutagenicity", "AIDS"}:
            assert scientific["disallow_generation"] == "1"

    common4_gate = (
        spec.lane_state_root("bace_globalgce_common4")
        / "sentinels"
        / "bace_common4.SUCCESS.json"
    )
    comrecgc_gate = (
        spec.lane_state_root("bace_comrecgc")
        / "sentinels"
        / "bace_comrecgc_final.SUCCESS.json"
    )
    assert common4_gate.stat().st_mtime_ns >= comrecgc_gate.stat().st_mtime_ns
    registry = spec.registry_path.read_text(encoding="utf-8")
    assert '"backend": "autodl"' in registry
    assert '"slurm_job_id": null' in registry
    assert not list(spec.persistent_root.rglob("*.tmp"))


@requires_linux_procfs
def test_resume_uses_resume_command_after_a_failed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "_pid_matches_worker",
        lambda pid, _spec, _lane: orchestrator._pid_alive(pid),
    )
    spec = LoadedSpec.load(
        _write_spec(
            tmp_path,
            first_commands={"mut_recovery": [sys.executable, "-c", FAIL_CODE]},
        )
    )
    start(spec)
    blocked = _wait_for_status(spec, {"BLOCKED"})
    assert blocked["lanes"]["mut_recovery"]["status"] == "FAILED"
    resumed = resume(spec)
    assert "mut_recovery" in resumed["resumed_pids"]
    final = _wait_for_status(spec, {"LANES_COMPLETED"})
    assert final["lanes"]["mut_recovery"]["retry_count"] == 1


@requires_linux_procfs
def test_start_is_single_use_and_stop_terminates_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "_pid_matches_worker",
        lambda pid, _spec, _lane: orchestrator._pid_alive(pid),
    )
    spec = LoadedSpec.load(_write_spec(tmp_path, delay="10"))
    start(spec)
    with pytest.raises(OrchestratorError, match="use resume"):
        start(spec)
    stopped = stop(spec)
    assert set(stopped["stop_requests"]) == set(spec.lane_by_id)
    final = _wait_for_status(spec, {"STOPPED", "BLOCKED"})
    assert all(
        value["status"] in {"STOPPED", "FAILED"}
        for value in final["lanes"].values()
    )


def test_common4_must_depend_on_both_completion_sentinels(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    common4 = payload["lanes"][3]["stages"][1]
    common4["dependencies"] = ["bace_globalgce_common4:bace_globalgce_wnode"]
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OrchestratorError, match="must depend on both"):
        LoadedSpec.load(spec_path)


def test_spec_schema_version_is_exact(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OrchestratorError, match="schema_version must be exactly"):
        LoadedSpec.load(spec_path)


def test_placeholder_commands_fail_before_any_run_state_is_written(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["lanes"][0]["stages"][0]["command"] = [
        "__CONFIGURE_MUT_FREEZE_RECOVERY_COMMAND__"
    ]
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    spec = LoadedSpec.load(spec_path)
    with pytest.raises(OrchestratorError, match="not configured"):
        start(spec)
    assert not spec.run_state_path.exists()


def test_preserved_lane_cannot_invoke_generation_entrypoint(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["lanes"][0]["stages"][0]["command"] = [
        sys.executable,
        "scripts/baselines/comrecgc/run_generation.py",
    ]
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OrchestratorError, match="invokes generation"):
        LoadedSpec.load(spec_path)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("input_manifest", "outside-input.sha256", "input manifest escapes"),
        ("output_manifest", "outside-output.sha256", "output manifest escapes"),
    ],
)
def test_manifest_paths_must_remain_in_declared_roots(
    tmp_path: Path, field: str, replacement: str, message: str
) -> None:
    spec_path = _write_spec(tmp_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    lane = payload["lanes"][0]
    outside = tmp_path / replacement
    outside.write_text("placeholder\n", encoding="utf-8")
    if field == "input_manifest":
        lane[field] = str(outside)
    else:
        lane["stages"][0][field] = str(outside)
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OrchestratorError, match=message):
        LoadedSpec.load(spec_path)


def test_controller_rejects_managed_root_symlink_before_resolve(
    tmp_path: Path,
) -> None:
    spec_path = _write_spec(tmp_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    physical = tmp_path / "physical-project"
    physical.mkdir()
    alias = tmp_path / "project-alias"
    alias.symlink_to(physical, target_is_directory=True)
    payload["roots"]["project"] = str(alias)
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OrchestratorError, match="symbolic-link component"):
        LoadedSpec.load(spec_path)


def test_stage_environment_rejects_secret_without_echoing_value(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    secret = "never-print-this-secret"
    payload["lanes"][0]["stages"][0]["environment"]["API_TOKEN"] = secret
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OrchestratorError) as captured:
        LoadedSpec.load(spec_path)
    assert "credential key" in str(captured.value)
    assert secret not in str(captured.value)


def test_inherited_environment_scrubs_secret_keys_and_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAFE_THREE_LINES_VALUE", "visible")
    monkeypatch.setenv("HPC_PASSWORD", "must-not-reach-child")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-child-either")
    monkeypatch.setenv("ODDLY_NAMED_VALUE", "Authorization=must-not-reach-child")
    environment = orchestrator._sanitized_inherited_environment()
    assert environment["SAFE_THREE_LINES_VALUE"] == "visible"
    assert "HPC_PASSWORD" not in environment
    assert "OPENAI_API_KEY" not in environment
    assert "ODDLY_NAMED_VALUE" not in environment


def test_command_rejects_api_key_assignment_without_echoing_value() -> None:
    secret = "not-a-real-credential-value"
    with pytest.raises(OrchestratorError) as captured:
        orchestrator._assert_no_embedded_secrets(
            ["python", "script.py", "--set", f"openai_api_key={secret}"]
        )
    assert "credential values" in str(captured.value)
    assert secret not in str(captured.value)


@pytest.mark.parametrize(
    ("failure_site", "ignore_term"),
    (("save", False), ("heartbeat", True)),
)
def test_stage_metadata_failure_terminates_scientific_group_and_reaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
    ignore_term: bool,
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    spec.runtime["stop_grace_seconds"] = 0.25
    lane = spec.lane_by_id["mut_recovery"]
    stage_config = lane["stages"][0]
    ready = tmp_path / f"{failure_site}.ready"
    term_seen = tmp_path / f"{failure_site}.term"
    child_code = (
        "import os,pathlib,signal,sys,time; "
        "ready=pathlib.Path(sys.argv[1]); term=pathlib.Path(sys.argv[2]); "
        + (
            "signal.signal(signal.SIGTERM, lambda *_: term.write_text('TERM\\n')); "
            if ignore_term
            else ""
        )
        + "ready.write_text(str(os.getpid())); time.sleep(600)"
    )
    stage_config["command"] = [
        sys.executable,
        "-c",
        child_code,
        str(ready),
        str(term_seen),
    ]
    state = orchestrator._initial_lane_state(spec, lane)
    monkeypatch.setattr(
        orchestrator,
        "_capture_child_identity",
        lambda pid, *_args: {
            "pid": int(pid),
            "pgid": int(pid),
            "capture_status": "CAPTURED",
        },
    )

    class MetadataFailure(RuntimeError):
        pass

    observed_pid: list[int] = []

    def fail_when_ready(*_args: object, **_kwargs: object) -> None:
        deadline = time.monotonic() + 3
        while not ready.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not ready.is_file():
            raise AssertionError("scientific child did not become ready")
        pid = int(ready.read_text(encoding="utf-8"))
        observed_pid[:] = [pid]
        assert os.getpgid(pid) == pid
        raise MetadataFailure(f"primary-{failure_site}-failure")

    if failure_site == "save":
        monkeypatch.setattr(orchestrator, "_save_lane_state", fail_when_ready)
    else:
        monkeypatch.setattr(orchestrator, "_heartbeat", fail_when_ready)

    with pytest.raises(MetadataFailure, match=f"primary-{failure_site}-failure"):
        orchestrator._run_stage(
            spec,
            lane,
            stage_config,
            state,
            orchestrator.StopFlag(),
        )
    assert observed_pid
    if ignore_term:
        assert term_seen.read_text(encoding="utf-8") == "TERM\n"
    with pytest.raises(ChildProcessError):
        os.waitpid(observed_pid[0], os.WNOHANG)


def test_controller_failure_cannot_orphan_nested_term_resistant_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    # The stage runner's own bounded cleanup is two seconds; the controller
    # must allow that independent process group to finish before escalating its
    # immediate stage group.
    spec.runtime["stop_grace_seconds"] = 4
    lane = spec.lane_by_id["mut_recovery"]
    stage_config = lane["stages"][0]
    ready = tmp_path / "nested-science.ready"
    term_seen = tmp_path / "nested-science.term"
    repository_root = Path(orchestrator.__file__).resolve().parents[2]
    nested_runner = """
import pathlib
import sys
from scripts.autodl import run_three_lines_stage as stage

child_code = (
    "import os,pathlib,signal,sys,time; "
    "ready=pathlib.Path(sys.argv[1]); term=pathlib.Path(sys.argv[2]); "
    "signal.signal(signal.SIGTERM, lambda *_: term.write_text('TERM\\\\n')); "
    "ready.write_text(str(os.getpid())); time.sleep(600)"
)
stage._run_checked(
    [sys.executable, "-c", child_code, sys.argv[1], sys.argv[2]],
    cwd=pathlib.Path(sys.argv[3]),
)
"""
    stage_config["command"] = [
        sys.executable,
        "-c",
        nested_runner,
        str(ready),
        str(term_seen),
        str(repository_root),
    ]
    stage_config["environment"]["PYTHONPATH"] = str(repository_root)
    state = orchestrator._initial_lane_state(spec, lane)
    immediate_pid: list[int] = []
    monkeypatch.setattr(
        orchestrator,
        "_capture_child_identity",
        lambda pid, *_args: (
            immediate_pid.append(int(pid))
            or {
                "pid": int(pid),
                "pgid": int(pid),
                "capture_status": "CAPTURED",
            }
        ),
    )

    class HeartbeatFailure(RuntimeError):
        pass

    def fail_after_nested_science_is_ready(*_args: object, **_kwargs: object) -> None:
        deadline = time.monotonic() + 5
        while not ready.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not ready.is_file():
            raise AssertionError("nested scientific child did not become ready")
        scientific_pid = int(ready.read_text(encoding="utf-8"))
        assert os.getpgid(scientific_pid) == scientific_pid
        raise HeartbeatFailure("primary-nested-heartbeat-failure")

    monkeypatch.setattr(orchestrator, "_heartbeat", fail_after_nested_science_is_ready)
    with pytest.raises(HeartbeatFailure, match="primary-nested-heartbeat-failure"):
        orchestrator._run_stage(
            spec,
            lane,
            stage_config,
            state,
            orchestrator.StopFlag(),
        )

    scientific_pid = int(ready.read_text(encoding="utf-8"))
    assert term_seen.read_text(encoding="utf-8") == "TERM\n"
    with pytest.raises(ProcessLookupError):
        os.kill(scientific_pid, 0)
    assert immediate_pid
    with pytest.raises(ChildProcessError):
        os.waitpid(immediate_pid[0], os.WNOHANG)


def test_linux_proc_stat_parser_handles_parenthesized_command_name() -> None:
    # Fields following the final ``) `` start at Linux proc-stat field 3.
    suffix = ["S", "1", "4444", *("0" for _ in range(16)), "987654321"]
    payload = f"4321 (worker ) with spaces) {' '.join(suffix)}\n".encode()
    starttime, pgid = orchestrator._parse_linux_proc_stat(payload)
    assert starttime == "987654321"
    assert pgid == 4444


def test_linux_pidfd_syscall_compat_when_python_and_libc_wrappers_are_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeFunction:
        restype: object = None

        def __call__(self, *args: object) -> int:
            calls.append(tuple(args))
            if int(args[0]) == 434:
                return 91
            return 0

    class FakeLibc:
        syscall = FakeFunction()

    monkeypatch.delattr(orchestrator.os, "pidfd_open", raising=False)
    monkeypatch.delattr(orchestrator.signal, "pidfd_send_signal", raising=False)
    monkeypatch.setattr(orchestrator, "_load_linux_libc", lambda: FakeLibc())
    monkeypatch.setattr(orchestrator, "_linux_machine", lambda: "x86_64")

    descriptor = orchestrator._linux_pidfd_open(7001)
    orchestrator._linux_pidfd_send_signal(descriptor, int(signal.SIGTERM))

    assert descriptor == 91
    assert calls == [
        (434, 7001, 0),
        (424, 91, int(signal.SIGTERM), None, 0),
    ]


def test_linux_pidfd_syscall_is_fail_closed_for_unknown_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFunction:
        restype: object = None

        def __call__(self, *_args: object) -> int:
            raise AssertionError("unknown architecture must fail before syscall")

    class FakeLibc:
        syscall = FakeFunction()

    monkeypatch.delattr(orchestrator.os, "pidfd_open", raising=False)
    monkeypatch.setattr(orchestrator, "_load_linux_libc", lambda: FakeLibc())
    monkeypatch.setattr(orchestrator, "_linux_machine", lambda: "mips64")
    with pytest.raises(
        orchestrator.PidfdUnavailableError, match="unsupported on architecture"
    ):
        orchestrator._linux_pidfd_open(7001)


def test_linux_pidfd_syscall_enosys_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFunction:
        restype: object = None

        def __call__(self, *_args: object) -> int:
            orchestrator.ctypes.set_errno(errno.ENOSYS)
            return -1

    class FakeLibc:
        syscall = FakeFunction()

    monkeypatch.delattr(orchestrator.os, "pidfd_open", raising=False)
    monkeypatch.setattr(orchestrator, "_load_linux_libc", lambda: FakeLibc())
    monkeypatch.setattr(orchestrator, "_linux_machine", lambda: "x86_64")
    with pytest.raises(orchestrator.PidfdUnavailableError, match="ENOSYS"):
        orchestrator._linux_pidfd_open(7001)


def test_exact_worker_and_orphan_child_use_pidfd_on_linux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    descriptor = 91
    opened: list[int] = []
    sent: list[tuple[int, int]] = []
    closed: list[int] = []
    numeric_kills: list[tuple[int, int]] = []
    monkeypatch.setattr(orchestrator.sys, "platform", "linux")
    monkeypatch.setattr(orchestrator, "_linux_procfs_available", lambda: True)
    monkeypatch.setattr(
        orchestrator,
        "_linux_pidfd_open",
        lambda pid: opened.append(int(pid)) or descriptor,
    )
    monkeypatch.setattr(
        orchestrator,
        "_linux_pidfd_send_signal",
        lambda fd, sig: sent.append((int(fd), int(sig))),
    )
    monkeypatch.setattr(orchestrator.os, "close", lambda fd: closed.append(int(fd)))
    monkeypatch.setattr(
        orchestrator.os,
        "kill",
        lambda pid, sig: numeric_kills.append((int(pid), int(sig))),
    )
    monkeypatch.setattr(orchestrator, "_pid_matches_worker", lambda *_args: True)
    monkeypatch.setattr(orchestrator, "_state_child_matches", lambda *_args: True)

    assert orchestrator._signal_exact_worker(
        7001, spec, "mut_recovery", int(signal.SIGTERM)
    )
    assert orchestrator._signal_exact_child(
        7002,
        spec,
        "mut_recovery",
        {"child_pid": 7002},
        int(signal.SIGTERM),
    )

    assert opened == [7001, 7002]
    assert sent == [
        (descriptor, int(signal.SIGTERM)),
        (descriptor, int(signal.SIGTERM)),
    ]
    assert closed == [descriptor, descriptor]
    assert numeric_kills == []


def test_linux_worker_without_procfs_never_uses_numeric_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    numeric_kills: list[tuple[int, int]] = []
    monkeypatch.setattr(orchestrator.sys, "platform", "linux")
    monkeypatch.setattr(orchestrator, "_linux_procfs_available", lambda: False)
    monkeypatch.setattr(
        orchestrator.os,
        "kill",
        lambda pid, sig: numeric_kills.append((int(pid), int(sig))),
    )
    with pytest.raises(orchestrator.PidfdUnavailableError, match="procfs identity"):
        orchestrator._signal_exact_worker(
            7401, spec, "mut_recovery", int(signal.SIGTERM)
        )
    assert numeric_kills == []


def test_linux_orphan_child_without_procfs_never_uses_numeric_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    numeric_kills: list[tuple[str, int, int]] = []
    monkeypatch.setattr(orchestrator.sys, "platform", "linux")
    monkeypatch.setattr(orchestrator, "_linux_procfs_available", lambda: False)
    monkeypatch.setattr(
        orchestrator.os,
        "kill",
        lambda pid, sig: numeric_kills.append(("pid", int(pid), int(sig))),
    )
    monkeypatch.setattr(
        orchestrator.os,
        "killpg",
        lambda pgid, sig: numeric_kills.append(("pgid", int(pgid), int(sig))),
    )
    with pytest.raises(orchestrator.PidfdUnavailableError, match="procfs identity"):
        orchestrator._signal_exact_child(
            7402,
            spec,
            "mut_recovery",
            {"child_pid": 7402},
            int(signal.SIGTERM),
        )
    assert numeric_kills == []


def test_child_identity_binds_starttime_cmdline_pgid_and_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    current = {
        "identity_source": "procfs",
        "proc_starttime": "123456",
        "process_start_token": "procfs:123456",
        "cmdline_sha256": "a" * 64,
        "pgid": 7001,
    }
    monkeypatch.setattr(
        orchestrator, "_current_process_identity", lambda _pid: dict(current)
    )
    command = [sys.executable, "stage.py", "--dataset", "BACE"]
    identity = orchestrator._capture_child_identity(
        7002, spec, "bace_comrecgc", "bace_generation", command
    )
    assert identity["proc_starttime"] == "123456"
    assert identity["cmdline_sha256"] == "a" * 64
    assert identity["pgid"] == 7001
    assert orchestrator._child_identity_matches(
        identity,
        spec,
        "bace_comrecgc",
        "bace_generation",
        command,
    )

    reused = dict(current)
    reused["proc_starttime"] = "999999"
    reused["process_start_token"] = "procfs:999999"
    monkeypatch.setattr(orchestrator, "_current_process_identity", lambda _pid: reused)
    assert not orchestrator._child_identity_matches(
        identity,
        spec,
        "bace_comrecgc",
        "bace_generation",
        command,
    )
    monkeypatch.setattr(
        orchestrator, "_current_process_identity", lambda _pid: dict(current)
    )
    assert not orchestrator._child_identity_matches(
        identity,
        spec,
        "bace_comrecgc",
        "bace_comrecgc_final",
        command,
    )


def test_worker_identity_binds_starttime_cmdline_pgid_run_lane_and_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    lane_id = "mut_recovery"
    pid = 7001
    current = {
        "identity_source": "procfs",
        "proc_starttime": "123456",
        "process_start_token": "procfs:123456",
        "cmdline_sha256": "b" * 64,
        "pgid": pid,
    }
    monkeypatch.setattr(
        orchestrator, "_current_process_identity", lambda _pid: dict(current)
    )
    command = orchestrator._worker_command(spec, lane_id)
    identity = orchestrator._capture_worker_identity(pid, spec, lane_id, command)
    assert identity["capture_status"] == "CAPTURED"
    assert identity["spec_sha256"] == spec.spec_sha256
    assert orchestrator._worker_identity_matches(
        identity, spec, lane_id, command
    )
    mutations = {
        "cmdline_sha256": "c" * 64,
        "pgid": pid + 1,
        "run_id": "another-run",
        "lane": "aids_recovery",
        "spec_sha256": "d" * 64,
        "command_sha256": "e" * 64,
    }
    for field, value in mutations.items():
        changed = dict(identity)
        changed[field] = value
        assert not orchestrator._worker_identity_matches(
            changed, spec, lane_id, command
        ), field

    reused = dict(current)
    reused.update(
        {
            "proc_starttime": "999999",
            "process_start_token": "procfs:999999",
        }
    )
    monkeypatch.setattr(
        orchestrator, "_current_process_identity", lambda _pid: reused
    )
    assert not orchestrator._worker_identity_matches(
        identity, spec, lane_id, command
    )


def test_stop_never_signals_a_reused_worker_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    target_lane_id = "mut_recovery"
    target_pid = 7001
    command = orchestrator._worker_command(spec, target_lane_id)
    original_process = {
        "identity_source": "procfs",
        "proc_starttime": "123456",
        "process_start_token": "procfs:123456",
        "cmdline_sha256": "a" * 64,
        "pgid": target_pid,
    }
    identity = {
        "schema_version": orchestrator.SCHEMA_VERSION,
        "kind": "autodl_worker",
        "capture_status": "CAPTURED",
        "pid": target_pid,
        "run_id": spec.run_id,
        "lane": target_lane_id,
        "spec_sha256": orchestrator.sha256_file(spec.path),
        "command_sha256": orchestrator._command_sha256(command),
        "captured_at": orchestrator.utc_now(),
        **original_process,
    }
    for lane in spec.lanes:
        state = orchestrator._initial_lane_state(spec, lane)
        if lane["id"] == target_lane_id:
            state.update(
                {
                    "status": "RUNNING",
                    "worker_pid": target_pid,
                    "worker_identity": identity,
                    "started_at": orchestrator.utc_now(),
                }
            )
        orchestrator._save_lane_state(spec, state)
    _write_bound_run_state(spec, status_value="RUNNING")
    orchestrator.atomic_write_json(
        orchestrator._lane_paths(spec, target_lane_id)["pid"],
        {
            "schema_version": orchestrator.SCHEMA_VERSION,
            "state_schema_version": orchestrator.STATE_SCHEMA_VERSION,
            "backend": "autodl",
            "kind": "autodl_worker_pid",
            "run_id": spec.run_id,
            "lane": target_lane_id,
            "spec_sha256": orchestrator.sha256_file(spec.path),
            "pid": target_pid,
            "worker_identity": identity,
            "status": "RUNNING",
        },
    )
    reused_process = dict(original_process)
    reused_process.update(
        {
            "proc_starttime": "999999",
            "process_start_token": "procfs:999999",
        }
    )
    monkeypatch.setattr(orchestrator, "_pid_alive", lambda pid: int(pid) == target_pid)
    monkeypatch.setattr(
        orchestrator, "_current_process_identity", lambda _pid: reused_process
    )
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        orchestrator.os,
        "kill",
        lambda pid, sig: signals.append((int(pid), int(sig))),
    )

    result = stop(spec)

    assert signals == []
    assert result["stop_requests"][target_lane_id] == {
        "stale_worker_pid": target_pid,
        "signal": None,
        "target": "none_identity_mismatch",
    }
    repaired = orchestrator._load_lane_state(
        spec, spec.lane_by_id[target_lane_id]
    )
    assert repaired["stale_worker_references"][-1]["signal_sent"] is False


def test_stop_does_not_signal_when_worker_identity_changes_at_revalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    target_lane_id = "mut_recovery"
    target_pid = 7101
    for lane in spec.lanes:
        state = orchestrator._initial_lane_state(spec, lane)
        if lane["id"] == target_lane_id:
            state.update(
                {
                    "status": "RUNNING",
                    "worker_pid": target_pid,
                    "worker_identity": {"pid": target_pid},
                    "started_at": orchestrator.utc_now(),
                }
            )
        orchestrator._save_lane_state(spec, state)
    _write_bound_run_state(spec, status_value="RUNNING")
    orchestrator.atomic_write_json(
        orchestrator._lane_paths(spec, target_lane_id)["pid"],
        {
            "schema_version": orchestrator.SCHEMA_VERSION,
            "backend": "autodl",
            "run_id": spec.run_id,
            "lane": target_lane_id,
            "pid": target_pid,
        },
    )
    checks = 0

    def identity_matches(_pid: int, _spec: LoadedSpec, lane_id: str) -> bool:
        nonlocal checks
        if lane_id != target_lane_id:
            return False
        checks += 1
        return checks == 1

    monkeypatch.setattr(orchestrator, "_pid_matches_worker", identity_matches)
    monkeypatch.setattr(
        orchestrator, "_pid_alive", lambda pid: int(pid) == target_pid
    )
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        orchestrator.os,
        "kill",
        lambda pid, sig: signals.append((int(pid), int(sig))),
    )

    result = stop(spec)

    assert checks == 2
    assert signals == []
    assert result["stop_requests"][target_lane_id]["target"] == (
        "none_identity_mismatch"
    )


def test_stop_uses_cooperative_marker_when_live_worker_pidfd_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    target_lane_id = "mut_recovery"
    target_pid = 7201
    for lane in spec.lanes:
        state = orchestrator._initial_lane_state(spec, lane)
        if lane["id"] == target_lane_id:
            state.update(
                {
                    "status": "RUNNING",
                    "worker_pid": target_pid,
                    "worker_identity": {"pid": target_pid},
                    "started_at": orchestrator.utc_now(),
                }
            )
        orchestrator._save_lane_state(spec, state)
    _write_bound_run_state(spec, status_value="RUNNING")
    monkeypatch.setattr(
        orchestrator,
        "_pid_matches_worker",
        lambda pid, _spec, lane_id: int(pid) == target_pid
        and lane_id == target_lane_id,
    )
    monkeypatch.setattr(
        orchestrator,
        "_signal_exact_worker",
        lambda *_args: (_ for _ in ()).throw(
            orchestrator.PidfdUnavailableError("pidfd_open ENOSYS")
        ),
    )
    numeric_signals: list[tuple[str, int, int]] = []
    monkeypatch.setattr(
        orchestrator.os,
        "kill",
        lambda pid, sig: numeric_signals.append(("pid", int(pid), int(sig))),
    )
    monkeypatch.setattr(
        orchestrator.os,
        "killpg",
        lambda pgid, sig: numeric_signals.append(("pgid", int(pgid), int(sig))),
    )

    result = stop(spec)

    assert numeric_signals == []
    assert all(
        (spec.lane_state_root(str(lane["id"])) / "STOP_REQUESTED.json").is_file()
        for lane in spec.lanes
    )
    request = result["stop_requests"][target_lane_id]
    assert request["signal"] is None
    assert request["target"] == "cooperative_stop_marker"
    assert request["manual_stop_required"] is False
    state = orchestrator._load_lane_state(spec, spec.lane_by_id[target_lane_id])
    assert state["status"] == "STOPPING"
    assert state["stop_delivery"]["mode"] == "cooperative_stop_marker"
    assert state["stop_delivery"]["signal_sent"] is False


def test_stop_requires_manual_action_for_orphan_when_pidfd_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    target_lane_id = "mut_recovery"
    target_pid = 7301
    for lane in spec.lanes:
        state = orchestrator._initial_lane_state(spec, lane)
        if lane["id"] == target_lane_id:
            state.update(
                {
                    "status": "ORPHANED_CHILD",
                    "child_pid": target_pid,
                    "child_identity": {"pid": target_pid},
                    "current_stage": "mut_recovery_stage",
                    "current_command": [sys.executable, "stage.py"],
                }
            )
        orchestrator._save_lane_state(spec, state)
    _write_bound_run_state(spec, status_value="BLOCKED")
    monkeypatch.setattr(orchestrator, "_worker_pid", lambda *_args: None)
    monkeypatch.setattr(
        orchestrator,
        "_state_child_matches",
        lambda _spec, lane_id, _state: lane_id == target_lane_id,
    )
    monkeypatch.setattr(
        orchestrator,
        "_signal_exact_child",
        lambda *_args: (_ for _ in ()).throw(
            orchestrator.PidfdUnavailableError("pidfd_open ENOSYS")
        ),
    )
    numeric_signals: list[tuple[str, int, int]] = []
    monkeypatch.setattr(
        orchestrator.os,
        "kill",
        lambda pid, sig: numeric_signals.append(("pid", int(pid), int(sig))),
    )
    monkeypatch.setattr(
        orchestrator.os,
        "killpg",
        lambda pgid, sig: numeric_signals.append(("pgid", int(pgid), int(sig))),
    )

    result = stop(spec)

    assert numeric_signals == []
    request = result["stop_requests"][target_lane_id]
    assert request["signal"] is None
    assert request["target"] == "manual_stop_required"
    assert request["manual_stop_required"] is True
    state = orchestrator._load_lane_state(spec, spec.lane_by_id[target_lane_id])
    assert state["status"] == "ORPHANED_CHILD"
    assert state["manual_stop_required"]["mode"] == "manual_stop_required"
    assert state["manual_stop_required"]["signal_sent"] is False


def test_launch_discards_reused_child_pid_without_orphaning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    lane = spec.lane_by_id["mut_recovery"]
    state = orchestrator._initial_lane_state(spec, lane)
    state.update(
        {
            "status": "FAILED",
            "child_pid": 4242,
            "child_identity": {"pid": 4242, "capture_status": "CAPTURED"},
            "current_stage": "mut_recovery_stage",
            "current_command": [sys.executable, "stale.py"],
        }
    )
    orchestrator._save_lane_state(spec, state)
    monkeypatch.setattr(orchestrator, "_worker_pid", lambda *_args: None)
    monkeypatch.setattr(orchestrator, "_state_child_matches", lambda *_args: False)
    monkeypatch.setattr(orchestrator, "_pid_alive", lambda pid: int(pid) == 4242)

    class FakePopen:
        pid = 7777

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    monkeypatch.setattr(orchestrator.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(
        orchestrator,
        "_capture_worker_identity",
        lambda *_args: {
            "schema_version": orchestrator.SCHEMA_VERSION,
            "kind": "autodl_worker",
            "capture_status": "CAPTURED",
            "pid": 7777,
            "run_id": spec.run_id,
            "lane": "mut_recovery",
        },
    )
    launched = orchestrator._launch_worker(spec, lane, retry=True)
    repaired = orchestrator._load_lane_state(spec, lane)
    assert launched == 7777
    assert repaired["status"] == "STARTING"
    assert repaired["status"] != "ORPHANED_CHILD"
    assert repaired["child_pid"] is None
    assert repaired["stale_child_references"][-1]["pid"] == 4242
    assert repaired["stale_child_references"][-1]["signal_sent"] is False


def test_launch_blocks_only_an_exact_live_orphan_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    lane = spec.lane_by_id["mut_recovery"]
    state = orchestrator._initial_lane_state(spec, lane)
    state.update(
        {
            "status": "FAILED",
            "child_pid": 4242,
            "child_identity": {"pid": 4242, "capture_status": "CAPTURED"},
            "current_stage": "mut_recovery_stage",
            "current_command": [sys.executable, "stage.py"],
        }
    )
    orchestrator._save_lane_state(spec, state)
    monkeypatch.setattr(orchestrator, "_worker_pid", lambda *_args: None)
    monkeypatch.setattr(orchestrator, "_state_child_matches", lambda *_args: True)
    with pytest.raises(OrchestratorError, match="exact orphan child"):
        orchestrator._launch_worker(spec, lane, retry=True)
    blocked = orchestrator._load_lane_state(spec, lane)
    assert blocked["status"] == "ORPHANED_CHILD"


def test_stop_never_signals_a_reused_child_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    _write_bound_run_state(spec)
    target_lane = spec.lane_by_id["mut_recovery"]
    for lane in spec.lanes:
        state = orchestrator._initial_lane_state(spec, lane)
        if lane["id"] == target_lane["id"]:
            state.update(
                {
                    "status": "FAILED",
                    "child_pid": 5151,
                    "child_identity": {"pid": 5151, "capture_status": "CAPTURED"},
                    "current_stage": "mut_recovery_stage",
                    "current_command": [sys.executable, "stale.py"],
                }
            )
        orchestrator._save_lane_state(spec, state)
    monkeypatch.setattr(orchestrator, "_worker_pid", lambda *_args: None)
    monkeypatch.setattr(orchestrator, "_state_child_matches", lambda *_args: False)
    monkeypatch.setattr(orchestrator, "_pid_alive", lambda pid: int(pid) == 5151)
    signals: list[tuple[str, int, int]] = []
    monkeypatch.setattr(
        orchestrator.os,
        "kill",
        lambda pid, sig: signals.append(("pid", int(pid), int(sig))),
    )
    monkeypatch.setattr(
        orchestrator.os,
        "killpg",
        lambda pgid, sig: signals.append(("pgid", int(pgid), int(sig))),
    )
    result = stop(spec)
    assert signals == []
    request = result["stop_requests"]["mut_recovery"]
    assert request == {
        "stale_child_pid": 5151,
        "signal": None,
        "target": "none_identity_mismatch",
    }
    repaired = orchestrator._load_lane_state(spec, target_lane)
    assert repaired["status"] != "ORPHANED_CHILD"
    assert repaired["stale_child_references"][-1]["signal_sent"] is False


def test_stop_signals_only_the_exact_orphan_child_leader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    _write_bound_run_state(spec)
    target_lane = spec.lane_by_id["mut_recovery"]
    for lane in spec.lanes:
        state = orchestrator._initial_lane_state(spec, lane)
        if lane["id"] == target_lane["id"]:
            state.update(
                {
                    "status": "ORPHANED_CHILD",
                    "child_pid": 6162,
                    "child_identity": {
                        "pid": 6162,
                        "pgid": 6161,
                        "capture_status": "CAPTURED",
                    },
                    "current_stage": "mut_recovery_stage",
                    "current_command": [sys.executable, "stage.py"],
                }
            )
        orchestrator._save_lane_state(spec, state)
    monkeypatch.setattr(orchestrator, "_worker_pid", lambda *_args: None)
    monkeypatch.setattr(
        orchestrator,
        "_state_child_matches",
        lambda _spec, lane_id, _state: lane_id == "mut_recovery",
    )
    exact_signals: list[tuple[int, str, int]] = []
    numeric_signals: list[tuple[str, int, int]] = []
    monkeypatch.setattr(
        orchestrator,
        "_signal_exact_child",
        lambda pid, _spec, lane_id, _state, sig: (
            exact_signals.append((int(pid), str(lane_id), int(sig))) or True
        ),
    )
    monkeypatch.setattr(
        orchestrator.os,
        "kill",
        lambda pid, sig: numeric_signals.append(("pid", int(pid), int(sig))),
    )
    monkeypatch.setattr(
        orchestrator.os,
        "killpg",
        lambda pgid, sig: numeric_signals.append(("pgid", int(pgid), int(sig))),
    )
    result = stop(spec)
    assert exact_signals == [
        (6162, "mut_recovery", int(orchestrator.signal.SIGTERM))
    ]
    assert numeric_signals == []
    request = result["stop_requests"]["mut_recovery"]
    assert request["orphan_child_pid"] == 6162
    assert request["target"] == "validated_orphan_child_leader"


def test_paired_slurm_wrapper_arguments_are_parseable_and_fail_closed() -> None:
    parser = orchestrator.build_parser()
    accepted = parser.parse_args(
        [
            "status",
            "--spec",
            "ops/specs/autodl_three_lines_20260821.yaml",
            "--config",
            "configs/hpc.yaml",
            "--set",
            "inference.fallback_to_heuristic=false",
        ]
    )
    orchestrator._validate_paired_wrapper_arguments(accepted)

    wrong_config = parser.parse_args(["status", "--config", "configs/local.yaml"])
    with pytest.raises(OrchestratorError, match="wrapper parity only"):
        orchestrator._validate_paired_wrapper_arguments(wrong_config)
    wrong_override = parser.parse_args(["status", "--set", "training.seed=8"])
    with pytest.raises(OrchestratorError, match="unsupported override"):
        orchestrator._validate_paired_wrapper_arguments(wrong_override)


def test_formal_completion_verifier_is_mandatory_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    spec.policy["require_formal_stage_verifier"] = True
    lane = spec.lane_by_id["mut_recovery"]
    stage = lane["stages"][0]
    stage["command"] = [
        sys.executable,
        str(spec.project_root / "scripts/autodl/run_three_lines_stage.py"),
        "mut-freeze",
        "--project-root",
        str(spec.project_root),
    ]
    observed: list[list[str]] = []

    def accepted(command: list[str], **_kwargs: object) -> object:
        observed.append(command)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(orchestrator.subprocess, "run", accepted)
    orchestrator._run_completion_verifier(spec, lane, stage)
    assert observed == [[*stage["command"], "--verify-only"]]

    monkeypatch.setattr(
        orchestrator.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Result", (), {"returncode": 9})(),
    )
    with pytest.raises(OrchestratorError, match="rejected stage"):
        orchestrator._run_completion_verifier(spec, lane, stage)


def test_formal_verifier_policy_rejects_arbitrary_stage_commands(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["policy"]["require_formal_stage_verifier"] = True
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OrchestratorError, match="lacks the audited stage verifier"):
        LoadedSpec.load(spec_path)


def test_preflight_allows_only_bound_vendor_manifest_as_untracked_provenance(
    tmp_path: Path,
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    provenance = spec.external_root / "vendor_manifest.json"
    provenance.write_text('{"kind":"provenance"}\n', encoding="utf-8")
    passed = orchestrator._preflight(spec)
    assert passed["external_code_clean"] is True
    assert passed["external_provenance_sha256"] == orchestrator.sha256_file(provenance)
    unexpected = spec.external_root / "runtime_patch.py"
    unexpected.write_text("patched = True\n", encoding="utf-8")
    with pytest.raises(OrchestratorError, match="unapproved untracked"):
        orchestrator._preflight(spec)


@requires_linux_procfs
def test_nonstandard_sha256_output_manifest_is_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "_pid_matches_worker",
        lambda pid, _spec, _lane: orchestrator._pid_alive(pid),
    )
    spec_path = _write_spec(tmp_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    lane = payload["lanes"][0]
    stage = lane["stages"][0]
    output_root = Path(lane["output_root"])
    sentinel = Path(stage["required_success_sentinel"])
    manifest = output_root / "freeze_recovery.sha256"
    artifact = output_root / "artifact.bin"
    code = (
        "import json,pathlib; "
        f"r=pathlib.Path({str(output_root)!r}); r.mkdir(parents=True,exist_ok=True); "
        f"pathlib.Path({str(artifact)!r}).write_bytes(b'artifact'); "
        f"pathlib.Path({str(manifest)!r}).write_text('" + "0" * 64 + "  artifact.bin\\n'); "
        f"pathlib.Path({str(sentinel)!r}).write_text(json.dumps({{'status':'PASS'}})+'\\n')"
    )
    stage["command"] = [sys.executable, "-c", code]
    stage["resume_command"] = [sys.executable, "-c", code]
    stage["output_manifest"] = str(manifest)
    stage["output_manifest_root"] = str(output_root)
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    spec = LoadedSpec.load(spec_path)
    start(spec)
    blocked = _wait_for_status(spec, {"BLOCKED"})
    failure = blocked["lanes"]["mut_recovery"]
    assert failure["status"] == "FAILED"
    lane_state = json.loads(
        (spec.lane_state_root("mut_recovery") / "lane_state.json").read_text()
    )
    assert "SHA256 mismatch" in lane_state["failure"]["error"]


@requires_linux_procfs
def test_resume_revalidates_completed_scientific_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "_pid_matches_worker",
        lambda pid, _spec, _lane: orchestrator._pid_alive(pid),
    )
    spec = LoadedSpec.load(_write_spec(tmp_path))
    start(spec)
    _wait_for_status(spec, {"LANES_COMPLETED"})
    stage = spec.lane_by_id["mut_recovery"]["stages"][0]
    sentinel = Path(stage["required_success_sentinel"])
    payload = json.loads(sentinel.read_text(encoding="utf-8"))
    payload["status"] = "FAIL"
    sentinel.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    stale = status(spec)
    assert stale["status"] == "BLOCKED"
    assert stale["lanes"]["mut_recovery"]["status"] == "STALE_COMPLETION"
    assert stale["lanes"]["mut_recovery"]["completion_validation"]["verified"] is False
    with pytest.raises(OrchestratorError, match="field mismatch"):
        resume(spec)


@requires_linux_procfs
def test_completed_lane_rejects_unlisted_input_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "_pid_matches_worker",
        lambda pid, _spec, _lane: orchestrator._pid_alive(pid),
    )
    spec = LoadedSpec.load(_write_spec(tmp_path))
    start(spec)
    _wait_for_status(spec, {"LANES_COMPLETED"})
    lane = spec.lane_by_id["mut_recovery"]
    (Path(lane["input_root"]) / "unlisted.bin").write_bytes(b"unexpected")
    stale = status(spec)
    assert stale["status"] == "BLOCKED"
    assert stale["lanes"]["mut_recovery"]["status"] == "STALE_COMPLETION"


def test_dependency_success_sentinel_is_schema_validated(tmp_path: Path) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    dependent = spec.lane_by_id["bace_comrecgc"]["stages"][1]
    dependency = "bace_comrecgc:bace_generation"
    path = orchestrator._stage_ref_sentinel(spec, dependency)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "backend": "autodl",
                "run_id": "stale-run",
                "lane": "bace_comrecgc",
                "stage": "bace_generation",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(OrchestratorError, match="Stale or forged"):
        dependency_lane = "bace_comrecgc"
        state = orchestrator._initial_lane_state(
            spec, spec.lane_by_id[dependency_lane]
        )
        state["status"] = "RUNNING"
        state["stages"]["bace_generation"] = {
            "stage": "bace_generation",
            "status": "SUCCEEDED",
            "sentinel": str(path),
            "output_manifest_digest": "invalid",
        }
        orchestrator._save_lane_state(spec, state)
        orchestrator._dependencies_satisfied(spec, dependency_lane, dependent)


def test_cross_lane_dependency_waits_for_persisted_lane_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    dependency_ref = "bace_comrecgc:bace_comrecgc_final"
    dependency_lane = spec.lane_by_id["bace_comrecgc"]
    dependency_stage = dependency_lane["stages"][1]
    dependency_sentinel = orchestrator._stage_ref_sentinel(spec, dependency_ref)
    dependency_sentinel.parent.mkdir(parents=True, exist_ok=True)
    dependency_sentinel.write_text("{}\n", encoding="utf-8")
    # Model the sentinel-first/state-second publication crash window: the lane
    # sentinel exists, but the durable lane state still says RUNNING.
    orchestrator._lane_paths(spec, "bace_comrecgc")["lane_success"].write_text(
        "{}\n", encoding="utf-8"
    )
    dependency_state = orchestrator._initial_lane_state(spec, dependency_lane)
    dependency_state.update(
        {
            "status": "RUNNING",
            "started_at": orchestrator.utc_now(),
            "stages": {
                dependency_stage["id"]: {
                    "stage": dependency_stage["id"],
                    "status": "SUCCEEDED",
                    "sentinel": str(dependency_sentinel),
                    "output_manifest_digest": "bound-output",
                }
            },
        }
    )
    orchestrator._save_lane_state(spec, dependency_state)
    monkeypatch.setattr(
        orchestrator,
        "_validate_completed_stage",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("proof must not be adopted before lane SUCCEEDED")
        ),
    )

    ready, missing = orchestrator._dependencies_satisfied(
        spec,
        "bace_globalgce_common4",
        {"dependencies": [dependency_ref]},
    )

    assert ready is False
    assert missing == [dependency_ref]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stage", "wrong-stage"),
        ("sentinel", "/wrong/sentinel.json"),
        ("output_manifest_digest", "wrong-digest"),
    ],
)
def test_dependency_rejects_mismatched_persisted_stage_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    lane_id = "bace_comrecgc"
    lane = spec.lane_by_id[lane_id]
    dependency_stage = lane["stages"][0]
    dependency_ref = f"{lane_id}:{dependency_stage['id']}"
    sentinel = orchestrator._stage_ref_sentinel(spec, dependency_ref)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("{}\n", encoding="utf-8")
    stage_state = {
        "stage": dependency_stage["id"],
        "status": "SUCCEEDED",
        "sentinel": str(sentinel),
        "output_manifest_digest": "bound-digest",
    }
    stage_state[field] = value
    lane_state = orchestrator._initial_lane_state(spec, lane)
    lane_state.update(
        {
            "status": "RUNNING",
            "started_at": orchestrator.utc_now(),
            "stages": {dependency_stage["id"]: stage_state},
        }
    )
    orchestrator._save_lane_state(spec, lane_state)
    monkeypatch.setattr(
        orchestrator,
        "_validate_completed_stage",
        lambda *_args: {"output_manifest_digest": "bound-digest"},
    )

    with pytest.raises(OrchestratorError, match="stage binding mismatch"):
        orchestrator._dependencies_satisfied(
            spec,
            lane_id,
            {"dependencies": [dependency_ref]},
        )


@pytest.mark.parametrize("persisted_status", [None, "RUNNING", "FAILED"])
def test_dependency_waits_for_missing_or_non_success_stage_state(
    tmp_path: Path,
    persisted_status: str | None,
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    lane_id = "bace_comrecgc"
    lane = spec.lane_by_id[lane_id]
    dependency_stage = lane["stages"][0]
    dependency_ref = f"{lane_id}:{dependency_stage['id']}"
    sentinel = orchestrator._stage_ref_sentinel(spec, dependency_ref)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("{}\n", encoding="utf-8")
    lane_state = orchestrator._initial_lane_state(spec, lane)
    lane_state.update({"status": "RUNNING", "started_at": orchestrator.utc_now()})
    if persisted_status is not None:
        lane_state["stages"] = {
            dependency_stage["id"]: {
                "stage": dependency_stage["id"],
                "status": persisted_status,
            }
        }
    orchestrator._save_lane_state(spec, lane_state)

    ready, missing = orchestrator._dependencies_satisfied(
        spec,
        lane_id,
        {"dependencies": [dependency_ref]},
    )

    assert ready is False
    assert missing == [dependency_ref]


def test_same_lane_completed_stage_releases_while_lane_is_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    lane_id = "bace_comrecgc"
    lane = spec.lane_by_id[lane_id]
    dependency_stage = lane["stages"][0]
    dependency_ref = f"{lane_id}:{dependency_stage['id']}"
    sentinel = orchestrator._stage_ref_sentinel(spec, dependency_ref)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("{}\n", encoding="utf-8")
    lane_state = orchestrator._initial_lane_state(spec, lane)
    lane_state.update(
        {
            "status": "RUNNING",
            "started_at": orchestrator.utc_now(),
            "stages": {
                dependency_stage["id"]: {
                    "stage": dependency_stage["id"],
                    "status": "SUCCEEDED",
                    "sentinel": str(sentinel),
                    "output_manifest_digest": "bound-digest",
                }
            },
        }
    )
    orchestrator._save_lane_state(spec, lane_state)
    monkeypatch.setattr(
        orchestrator,
        "_validate_completed_stage",
        lambda *_args: {"output_manifest_digest": "bound-digest"},
    )

    ready, missing = orchestrator._dependencies_satisfied(
        spec,
        lane_id,
        {"dependencies": [dependency_ref]},
    )

    assert ready is True
    assert missing == []


def test_run_state_rejects_spec_drift_and_refresh_preserves_binding(
    tmp_path: Path,
) -> None:
    spec_path = _write_spec(tmp_path)
    spec = LoadedSpec.load(spec_path)
    for lane in spec.lanes:
        orchestrator._save_lane_state(
            spec, orchestrator._initial_lane_state(spec, lane)
        )
    _write_bound_run_state(spec, status_value="NOT_STARTED")
    published = orchestrator._refresh_run_state(spec)
    expected_binding = orchestrator._spec_state_binding(spec)
    assert {
        key: published[key] for key in expected_binding
    } == expected_binding

    changed = json.loads(spec_path.read_text(encoding="utf-8"))
    changed["runtime"]["dependency_poll_seconds"] = 999
    spec_path.write_text(json.dumps(changed, indent=2) + "\n", encoding="utf-8")
    drifted = LoadedSpec.load(spec_path)
    persisted_before = spec.run_state_path.read_bytes()
    with pytest.raises(OrchestratorError, match="run state binding mismatch"):
        status(drifted)
    with pytest.raises(OrchestratorError, match="run state binding mismatch"):
        resume(drifted, ["mut_recovery"])
    with pytest.raises(OrchestratorError, match="run state binding mismatch"):
        stop(drifted)
    assert spec.run_state_path.read_bytes() == persisted_before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 999),
        ("state_schema_version", 999),
        ("backend", "hpc"),
        ("run_id", "another-run"),
        ("spec_path", "/another/spec.yaml"),
        ("spec_sha256", "f" * 64),
        ("project_root", "/another/project"),
        ("external_root", "/another/vendor"),
        ("persistent_root", "/another/persistent"),
        ("fast_root", "/another/fast"),
        ("roots", {"project": "/another/root"}),
    ],
)
def test_run_state_binding_rejects_each_authority_field(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    payload = orchestrator._spec_state_binding(spec)
    payload[field] = value
    with pytest.raises(OrchestratorError, match="run state binding mismatch"):
        orchestrator._validate_run_state_binding(spec, payload)


def test_refresh_rejects_unbound_or_corrupted_state_publication(
    tmp_path: Path,
) -> None:
    spec = LoadedSpec.load(_write_spec(tmp_path))
    for lane in spec.lanes:
        orchestrator._save_lane_state(
            spec, orchestrator._initial_lane_state(spec, lane)
        )
    _write_bound_run_state(spec, status_value="NOT_STARTED")
    corrupted = json.loads(spec.run_state_path.read_text(encoding="utf-8"))
    corrupted["spec_sha256"] = "0" * 64
    spec.run_state_path.write_text(
        json.dumps(corrupted, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(OrchestratorError, match="run state binding mismatch"):
        orchestrator._refresh_run_state(spec)
