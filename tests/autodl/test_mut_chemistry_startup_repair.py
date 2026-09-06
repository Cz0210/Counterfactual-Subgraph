import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts.autodl import resume_mut_chemistry_startup as repair
from scripts.autodl import run_comrecgc_standardized_continuation as common
from scripts.autodl import run_mut_comrecgc_parity_standardization as mut
from src.baselines.comrecgc.contracts import (
    atomic_write_bytes, sha256_file, stable_json_sha256, write_json,
)
from src.utils import mut_chemistry_stage_boundary as boundary_module


def _commands(root):
    return [[stage, [sys.executable, stage, "--expected-project-commit", "a" * 40],
             str(root / stage / "_RUN_COMPLETE.json"), "run_complete"]
            for stage in ("chemistry", *repair.STAGES)]


def _fake_barrier(monkeypatch):
    launched = []
    class Process:
        pid = 4242
        def wait(self, timeout=None): return 0
        def poll(self): return 0
    class Barrier:
        launcher_argv = ["python", "-S", "barrier"]
        def launch(self, **kwargs): launched.append(kwargs); return Process()
        def release(self): pass
        def abort(self): pass
    def arm(**kwargs):
        write_json(kwargs["record_path"], {"binding": "test"})
        return Barrier()
    monkeypatch.setattr(common, "arm_exec_startup_barrier", arm)
    monkeypatch.setattr(common, "_read_proc_start_ticks", lambda *a, **k: 99)
    monkeypatch.setattr(common, "_proc_argv", lambda *a, **k: Barrier.launcher_argv)
    monkeypatch.setattr(common, "_wait_for_process_group_quiescence", lambda *a, **k: None)
    return launched


def test_exact_collision_and_three_real_bookkeeping_stages(tmp_path, monkeypatch):
    root, recovery = tmp_path / "science", tmp_path / "repair"
    commands = _commands(root)
    old = dict(schema_version=2, status="PASS", stage="chemistry",
               argv_sha256=stable_json_sha256(commands[0][1]), process_group_id=123)
    write_json(root / "stage_state.json", old)
    old_bytes = (root / "stage_state.json").read_bytes()
    with pytest.raises(ValueError, match="PREVIOUS_STAGE_STATE_INVALID:unified_eval"):
        common._next_stage_startup_generation(output_root=root, stage="unified_eval",
            argv=commands[1][1], checkpoint_path=None)
    launched = _fake_barrier(monkeypatch)
    runner = SimpleNamespace(_run_stage=common._run_stage, write_json=write_json)
    repair._install_control_adapter(runner, root=root, recovery=recovery,
        boundary={"contract": {"commands": commands}}, old_stage=old)
    for stage, argv, marker, field in commands[1:]:
        write_json(marker, {field: True})
        runner._run_stage(stage=stage, argv=argv, marker=Path(marker), required_field=field,
                          environment={"CUDA_VISIBLE_DEVICES": ""}, output_root=root)
        saved = json.loads((recovery / "stage_checkpoints" / (stage + ".json")).read_text())
        assert saved["stage"] == stage and saved["status"] == "PASS"
        assert saved["startup_barrier"]["generation"] == 0
        assert saved["argv_sha256"] == stable_json_sha256(argv)
    assert len(launched) == 3
    assert (root / "stage_state.json").read_bytes() == old_bytes
    assert not (root / "stage_checkpoints").exists()


def test_only_exact_diagnostic_path_redirected_and_changed_command_rejected(tmp_path):
    root, recovery = tmp_path / "science", tmp_path / "repair"
    commands = _commands(root)
    calls = []
    runner = SimpleNamespace(_run_stage=lambda **k: calls.append(k), write_json=write_json)
    write_json(root / "FAILED.json", {"message": repair.FAILURE})
    original = (root / "FAILED.json").read_bytes()
    repair._install_control_adapter(runner, root=root, recovery=recovery,
        boundary={"contract": {"commands": commands}}, old_stage={"process_group_id": 123})
    runner.write_json(root / "FAILED.json", {"message": "NEW_ERROR"})
    assert (root / "FAILED.json").read_bytes() == original
    assert json.loads((recovery / "runner_FAILED.json").read_text())["message"] == "NEW_ERROR"
    runner.write_json(root / "run_manifest.json", {"project_commit": "a" * 40})
    assert json.loads((root / "run_manifest.json").read_text())["project_commit"] == "a" * 40
    for stage in ("chemistry", "full_gate", "unified_eval"):
        with pytest.raises(ValueError, match="SCIENTIFIC_COMMAND_CHANGED"):
            runner._run_stage(stage=stage, argv=["altered"], marker=root / "marker",
                              required_field="run_complete", output_root=root)
    assert calls == []


def _invocation(tmp_path, monkeypatch):
    source, root, recovery = (tmp_path / n for n in ("source", "science", "repair"))
    source.mkdir()
    commands = _commands(root)
    for name in ("chemistry/_RUN_COMPLETE.json", "chemistry/run_manifest.json",
                 "chemistry/final_artifact_audit.json", "generation_adoption_manifest.json",
                 "historical_adoption_manifest.json", "upstream_checkout_audit.json"):
        write_json(root / name, {"run_complete": True})
    boundary_module.seal_boundary(root, {"project_commit": "a" * 40, "commands": commands})
    write_json(root / "stage_state.json", dict(schema_version=2, stage="chemistry", status="PASS",
        process_group_id=123, argv_sha256=stable_json_sha256(commands[0][1])))
    write_json(root / "FAILED.json", dict(message=repair.FAILURE, status="FAILED",
        error_class="ValueError", output_root=str(root)))
    config = tmp_path / "resource.json"
    write_json(config, {"stage_file_policy": {"immutable": True}})
    argv = [sys.executable, "-I", "-B", str(source / "scripts/autodl/run_mut_comrecgc_parity_standardization.py")]
    for flag in ("source-generation-root", "upstream-root", "dataset-dir", "distance-checkpoint",
                 "dataset-csv", "teacher-path", "molclr-root", "molclr-checkpoint", "thresholds-path",
                 "historical-adoption", "persistent-root"):
        argv.extend(["--" + flag, str(tmp_path / flag)])
    argv.extend(["--output-root", str(root), "--stage-resource-config", str(config),
                 "--resume-after-chemistry"])
    spec = dict(schema_version=repair.SPEC_SCHEMA, cwd=str(source), execution_commit="a" * 40,
        command=argv, argv_sha256=stable_json_sha256(argv), output_root=str(root),
        environment={"CUDA_VISIBLE_DEVICES": ""},
        resource_config={"path": str(config), "sha256": sha256_file(config)})
    spec["self_sha256"] = stable_json_sha256(spec)
    path = tmp_path / "spec.json"
    write_json(path, spec)
    calls = []
    runner = SimpleNamespace(PROJECT_ROOT=source, stable_json_sha256=stable_json_sha256,
        sha256_file=sha256_file, atomic_write_bytes=atomic_write_bytes, write_json=write_json,
        build_parser=mut.build_parser, _run_stage=lambda **k: None,
        main=lambda argv: calls.append(argv) or 0)
    monkeypatch.setattr(repair, "_load_runner", lambda p: runner)
    monkeypatch.setattr(repair, "_git", lambda tree, *args:
        ("a" * 40 if tree == source else "b" * 40) if args == ("rev-parse", "HEAD") else "")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    return path, spec, recovery, runner, calls


def test_receipt_distinguishes_actual_driver_from_unchanged_science(tmp_path, monkeypatch):
    path, spec, recovery, runner, calls = _invocation(tmp_path, monkeypatch)
    root = Path(spec["output_root"])
    original = {n: (root / n).read_bytes() for n in (
        "stage_state.json", "FAILED.json", "chemistry_stage_boundary.json")}
    assert repair.run(continuation_spec=path, recovery_root=recovery,
                      expected_driver_commit="b" * 40) == 0
    receipt = json.loads((recovery / "control_adapter_receipt.json").read_text())
    assert receipt["control_driver_commit"] == "b" * 40
    assert receipt["scientific_project_commit"] == "a" * 40
    assert receipt["self_sha256"] == stable_json_sha256({k: v for k, v in receipt.items() if k != "self_sha256"})
    assert calls == [spec["command"][4:]]
    for name, data in original.items():
        assert (root / name).read_bytes() == data
        assert (recovery / "preserved" / name).read_bytes() == data


@pytest.mark.parametrize("mutation", ["driver_commit", "command", "resource", "started_evaluation"])
def test_preflight_fails_closed_without_runner_dispatch(tmp_path, monkeypatch, mutation):
    path, spec, recovery, runner, calls = _invocation(tmp_path, monkeypatch)
    driver = "b" * 40
    if mutation == "driver_commit": driver = "c" * 40
    if mutation == "command":
        spec["command"].append("--through-stage")
        write_json(path, spec)
    if mutation == "resource": write_json(Path(spec["resource_config"]["path"]), {"changed": True})
    if mutation == "started_evaluation": (Path(spec["output_root"]) / "unified_eval").mkdir()
    with pytest.raises(ValueError):
        repair.run(continuation_spec=path, recovery_root=recovery, expected_driver_commit=driver)
    assert calls == [] and not recovery.exists()
