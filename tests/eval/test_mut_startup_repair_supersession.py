import json
from pathlib import Path

import pytest

from src.baselines.comrecgc.contracts import sha256_file, stable_json_sha256, write_json
from src.eval import mut_startup_repair_supersession as m


def _seal(path, value):
    value = {k: v for k, v in value.items() if k != "self_sha256"}
    write_json(path, {**value, "self_sha256": stable_json_sha256(value)})


def _time(second):
    return f"2026-09-07T00:00:{second:02d}+00:00"


def fixture(tmp_path, monkeypatch):
    root, control, source, proc = (tmp_path / n for n in ("science", "control", "source", "proc"))
    for path in (root, control, source, proc): path.mkdir()
    scripts = ("run_slot_unified_eval.py", "gate_recovery.py", "freeze_recovery_result.py")
    markers = ("unified_eval/_RUN_COMPLETE.json", "full_gate/gate_result.json", "standardized/_FINALIZED.json")
    fields = ("run_complete", "audit_passed", "finalized")
    commands = [["chemistry", ["python", "original-chemistry"], str(root / "chemistry/_RUN_COMPLETE.json"), "run_complete"]]
    for stage, script, marker, field in zip(m.STAGES, scripts, markers, fields):
        commands.append([stage, ["python", str(source / "scripts/baselines/comrecgc" / script)], str(root / marker), field])
    write_json(root / "FAILED.json", dict(schema_version="mut_comrecgc_fast_accurate_standardization_failure_v2",
        status="FAILED", dataset="mutagenicity", error_class="ValueError", message=m.FAILURE_MESSAGE,
        output_root=str(root), failed_at=_time(2)))
    write_json(root / "stage_state.json", dict(schema_version=2, status="PASS", stage="chemistry",
        argv_sha256=stable_json_sha256(commands[0][1]), completed_at=_time(0)))
    closure_names = ("chemistry/_RUN_COMPLETE.json", "chemistry/run_manifest.json",
        "chemistry/final_artifact_audit.json", "generation_adoption_manifest.json",
        "historical_adoption_manifest.json", "upstream_checkout_audit.json")
    for name in closure_names: write_json(root / name, {"run_complete": True})
    _seal(root / "chemistry_stage_boundary.json", dict(schema_version="mut_train_only_chemistry_stage_boundary_v1",
        state="SEALED_CHEMISTRY_WAITING_EVALUATION_ADMISSION", chemistry_complete=True,
        test_evaluation_started=False, completed_at=_time(1),
        sealed_files=[{"path": str(root / name), "sha256": sha256_file(root / name)} for name in closure_names],
        contract={"project_commit": m.SCIENCE_COMMIT, "commands": commands}))
    for index, (stage, argv, marker, field) in enumerate(commands[1:]):
        checkpoint_dir = control / "stage_checkpoints"
        record_path = checkpoint_dir / ("." + stage + ".exec-startup.00.json")
        lock_path = checkpoint_dir / ("." + stage + ".exec-startup.lock")
        launcher = ["python", "-S", "barrier", "--", *argv]
        write_json(record_path, dict(schema="autodl_exec_startup_barrier_v1", target_argv=argv,
            target_argv_sha256=stable_json_sha256(argv), launcher_argv=launcher,
            launcher_argv_sha256=stable_json_sha256(launcher), record_path=str(record_path), lock_path=str(lock_path)))
        write_json(Path(marker), {field: True})
        checkpoint = dict(schema_version=2, stage=stage, status="PASS", marker=marker, required_field=field,
            argv_sha256=stable_json_sha256(argv), marker_sha256=sha256_file(marker),
            process_group_contract="durable_barrier_dedicated_child_session_v2", runner_pid=901,
            child_pid=910 + index, child_start_ticks=100 + index, process_group_id=910 + index,
            started_at=_time(3 + index * 3), child_started_at=_time(4 + index * 3), completed_at=_time(5 + index * 3),
            startup_barrier=dict(schema_version="comrecgc_continuation_exec_startup_binding_v1", phase="BOUND",
                generation=0, stage=stage, target_argv_sha256=stable_json_sha256(argv), record_path=str(record_path),
                lock_path=str(lock_path), record_sha256=sha256_file(record_path), launcher_argv_sha256=stable_json_sha256(launcher)))
        write_json(checkpoint_dir / (stage + ".json"), checkpoint)
    write_json(control / "stage_state.json", checkpoint)
    final = dict(status="PASS", project_commit=m.SCIENCE_COMMIT, completed_at=_time(12),
        independent_scientific_adoption_authorized=True)
    for name in ("run_manifest.json", "final_gate.json"): write_json(root / name, final)
    write_json(root / "_RUN_COMPLETE.json", {**final, "run_complete": True})
    command = ["python", "original-runner", "--resume-after-chemistry"]
    spec_path = tmp_path / "continuation.json"
    _seal(spec_path, dict(execution_commit=m.SCIENCE_COMMIT, cwd=str(source), output_root=str(root),
        argv_sha256=stable_json_sha256(command), command=command))
    spec = json.loads(spec_path.read_text())
    names = ("FAILED.json", "stage_state.json", "chemistry_stage_boundary.json")
    for name in names:
        path = control / "preserved" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes((root / name).read_bytes())
    driver = Path(__file__).resolve().parents[2] / "scripts/autodl/resume_mut_chemistry_startup.py"
    receipt_path = control / "control_adapter_receipt.json"
    _seal(receipt_path, dict(schema_version="mut_chemistry_startup_control_repair_v1",
        state="CONTROL_ADAPTER_READY_NOT_SCIENCE_PASS", output_root=str(root), control_output_root=str(control),
        control_driver_commit=m.REPAIR_COMMIT, control_driver_path=str(driver), control_driver_sha256=m.REPAIR_DRIVER_SHA256,
        scientific_project_commit=m.SCIENCE_COMMIT, scientific_source_root=str(source),
        scientific_argv_changed=False, scientific_import_tree_changed=False, source_generation_rerun=False,
        common_recourse_rerun=False, chemistry_rerun=False, original_stage_state_rewritten=False, original_failure_rewritten=False,
        control_changes=["_run_stage.output_root", "_run_stage.checkpoint_path", "runner.write_json:exact_prior_FAILED.json_only"],
        preserved={name: sha256_file(root / name) for name in names}, continuation_spec_path=str(spec_path),
        continuation_spec_sha256=sha256_file(spec_path), continuation_spec_self_sha256=spec["self_sha256"],
        original_argv_sha256=stable_json_sha256(command),
        stage_checkpoint_paths={stage: str(control / "stage_checkpoints" / (stage + ".json")) for stage in m.STAGES}))
    monkeypatch.setattr(m.subprocess, "check_output", lambda argv, **k: m.SCIENCE_COMMIT if "rev-parse" in argv else "")
    return root, control, proc, receipt_path


def test_readonly_supersession_preserves_failure_and_binds_all_three_passes(tmp_path, monkeypatch):
    root, control, proc, receipt = fixture(tmp_path, monkeypatch)
    originals = {str(p): p.read_bytes() for p in root.rglob("*.json")}
    result = m.validate_startup_repair_supersession(root, receipt, proc_root=proc)
    assert result["status"] == "RESOLVED_PRE_SCIENCE_STARTUP_FAILURE"
    assert result["scientific_validation_bypassed"] is False
    assert result["repair_control_commit"] == m.REPAIR_COMMIT
    assert result["scientific_project_commit"] == m.SCIENCE_COMMIT
    assert all(Path(path).read_bytes() == value for path, value in originals.items())
    assert all(str(control / "stage_checkpoints" / (stage + ".json")) in result["inventory"] for stage in m.STAGES)


@pytest.mark.parametrize("change", ["failure", "extra_failure", "new_control_failure", "checkpoint_failure",
    "checkpoint_argv", "stage_marker", "stage_order", "active_owner", "driver", "final_commit", "missing_receipt"])
def test_unknown_or_incomplete_repair_stays_failed(tmp_path, monkeypatch, change):
    root, control, proc, receipt = fixture(tmp_path, monkeypatch)
    if change == "failure": write_json(root / "FAILED.json", {"message": "another failure"})
    if change == "extra_failure": (root / "FAILED").write_text("FAILED\n")
    if change == "new_control_failure": write_json(control / "runner_FAILED.json", {"status": "FAILED"})
    if change in {"checkpoint_failure", "checkpoint_argv", "stage_order"}:
        path = control / "stage_checkpoints/unified_eval.json"
        checkpoint = json.loads(path.read_text())
        checkpoint.update({"status": "FAILED"} if change == "checkpoint_failure" else
                          {"argv_sha256": "0" * 64} if change == "checkpoint_argv" else {"started_at": _time(0)})
        write_json(path, checkpoint)
    if change == "stage_marker": write_json(root / "full_gate/gate_result.json", {"audit_passed": False})
    if change == "active_owner": (proc / "901").mkdir()
    if change == "driver":
        value = json.loads(receipt.read_text()); value["control_driver_commit"] = "0" * 40; _seal(receipt, value)
    if change == "final_commit":
        value = json.loads((root / "final_gate.json").read_text()); value["project_commit"] = "0" * 40
        write_json(root / "final_gate.json", value)
    if change == "missing_receipt": receipt.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        m.validate_startup_repair_supersession(root, receipt, proc_root=proc)


def test_writable_fd_blocks_even_with_dead_recorded_pids(tmp_path, monkeypatch):
    root, control, proc, receipt = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(m, "scan_live_writers", lambda *a, **k:
        {"procfs_verified": True, "writable_fd_count": 1, "writers": [{"pid": 999}]})
    with pytest.raises(ValueError, match="live_writer"):
        m.validate_startup_repair_supersession(root, receipt, proc_root=proc)
