from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from scripts.ops import experimentctl
from scripts.ops.clear_phase_b_smoke_gate import SUCCESS_MARKERS, evaluate
from scripts.ops.spec import TaskSpec, load_task_spec
from scripts.ops.state import RunStore
from scripts.ops.subprocess_utils import CommandResult


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "ops/specs/clear_mutagenicity_phase_a_v2.yaml"
WRAPPER = ROOT / "scripts/slurm/clear_mutagenicity_phase_b_gpu_smoke.sh"
COMMIT = "a" * 40
JOB_ID = "2048123"


class QueueRunner:
    def __init__(self, *results: CommandResult) -> None:
        self.results = list(results)
        self.calls: list[list[str]] = []

    def run(self, argv, **kwargs):
        self.calls.append([str(value) for value in argv])
        if not self.results:
            raise AssertionError("unexpected external command")
        return self.results.pop(0)


class ExplodingRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def run(self, argv, **kwargs):
        self.calls.append([str(value) for value in argv])
        raise AssertionError("unexpected external command")


def _result(stdout: str = "", stderr: str = "", returncode: int = 0) -> CommandResult:
    return CommandResult(
        argv=["ssh"],
        cwd=str(ROOT),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _spec(tmp_path: Path) -> TaskSpec:
    loaded = load_task_spec(SPEC_PATH)
    payload = deepcopy(loaded.data)
    payload["project"]["local_root"] = str(tmp_path)
    return TaskSpec(path=loaded.path, data=payload)


def _store(tmp_path: Path, *, approved: bool = True) -> RunStore:
    store = RunStore.create(
        tmp_path / "reports",
        "clear_mutagenicity_phase_a_v2",
        run_id="phase_b_unit",
        spec_path=str(SPEC_PATH),
    )
    for stage_id in ("phase_a_prepare", "phase_a_probe", "phase_a_audit"):
        store.record_stage(stage_id, {"status": "ADOPTED_EXISTING", "attempt": 1})
    state = store.load()
    state["local_commit"] = COMMIT
    state["remote_commit"] = COMMIT
    store.save(state)
    if approved:
        store.approve(
            "phase_b_gpu_smoke",
            "Reviewed the 64-parent smoke boundary.",
            "researcher",
            git_commit=COMMIT,
        )
    return store


def _submit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    store: RunStore | None = None,
    result: CommandResult | None = None,
) -> tuple[TaskSpec, RunStore, QueueRunner]:
    spec = _spec(tmp_path)
    selected_store = store or _store(tmp_path)
    monkeypatch.setattr(experimentctl, "head_commit", lambda *args: COMMIT)
    runner = QueueRunner(
        result or _result("[EXP_SUBMIT_OK]\njob_id=" + JOB_ID + "\n")
    )
    experimentctl.submit(
        spec, dry_run=False, store=selected_store, runner=runner
    )
    return spec, selected_store, runner


def _gate_payload(**overrides) -> dict:
    payload = {
        "schema_version": 1,
        "audit_passed": True,
        "run_complete": True,
        "failed_hard_checks": [],
        "checks": {
            "model_train_rows": 2885,
            "model_val_rows": 355,
            "generation_source_rows": 1448,
            "selected_generation_parents": 64,
            "generation_profile": "smoke",
            "candidate_pool_rows": 2,
            "candidate_universe_rows": 2,
            "completed_chunk_count": 4,
            "calibration_loaded": False,
            "test_loaded": False,
            "model_training_performed": True,
            "git_commit": COMMIT,
        },
        "artifacts": {},
        "provenance": {
            "dataset": "Mutagenicity",
            "calibration_loaded": False,
            "test_loaded": False,
        },
    }
    for key, value in overrides.items():
        if key.startswith("checks__"):
            payload["checks"][key.split("__", 1)[1]] = value
        elif key.startswith("provenance__"):
            payload["provenance"][key.split("__", 1)[1]] = value
        else:
            payload[key] = value
    return payload


def _encoded_gate(payload: dict) -> str:
    encoded = base64.b64encode(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).decode("ascii")
    return f"[AUTOMATION_CLEAR_PHASE_B_GATE_EVIDENCE_B64] {encoded}\n"


def test_state_initializes_slurm_jobs_as_empty_list(tmp_path: Path) -> None:
    store = RunStore.create(tmp_path / "reports", "task", run_id="empty")
    assert store.load()["slurm_jobs"] == []


def test_approval_records_reason_stage_commit_and_time(tmp_path: Path) -> None:
    store = _store(tmp_path)
    approval = store.load()["approvals"]["phase_b_gpu_smoke"]
    assert approval["reason"] == "Reviewed the 64-parent smoke boundary."
    assert approval["stage_id"] == "phase_b_gpu_smoke"
    assert approval["git_commit"] == COMMIT
    assert approval["timestamp"]


def test_unapproved_submit_never_contacts_ssh(monkeypatch, tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    store = _store(tmp_path, approved=False)
    monkeypatch.setattr(experimentctl, "head_commit", lambda *args: COMMIT)
    runner = ExplodingRunner()
    with pytest.raises(experimentctl.AutomationBlocked, match="requires approval"):
        experimentctl.submit(spec, dry_run=False, store=store, runner=runner)
    assert runner.calls == []
    assert store.load()["slurm_jobs"] == []


@pytest.mark.parametrize(
    "permission,bad_value",
    (
        ("allow_remote_write", False),
        ("allow_sbatch", False),
        ("allow_gpu_smoke", False),
        ("allow_full", True),
        ("allow_calibration", True),
        ("allow_test", True),
        ("allow_finalization", True),
        ("allow_overwrite", True),
    ),
)
def test_every_smoke_permission_boundary_is_enforced(
    monkeypatch, tmp_path: Path, permission: str, bad_value: bool
) -> None:
    spec = _spec(tmp_path)
    spec.data["permissions"][permission] = bad_value
    monkeypatch.setattr(experimentctl, "head_commit", lambda *args: COMMIT)
    with pytest.raises(experimentctl.AutomationBlocked, match="permission contract"):
        experimentctl.submit(
            spec,
            dry_run=False,
            store=_store(tmp_path),
            runner=ExplodingRunner(),
        )


def test_submit_dry_run_has_no_remote_or_slurm_side_effect(monkeypatch, tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    monkeypatch.setattr(experimentctl, "head_commit", lambda *args: COMMIT)
    runner = ExplodingRunner()
    result = experimentctl.submit(spec, dry_run=True, runner=runner)
    assert runner.calls == []
    assert result["remote_write_performed"] is False
    assert result["slurm_jobs"] == []
    assert result["commands"][0][0] == "ssh"
    joined = " ".join(result["commands"][0])
    assert "scripts/exp_sbatch.sh" in joined
    assert "{run_id}" not in joined
    assert "ClearAllForwardings=yes" in result["commands"][0]
    assert "BatchMode=yes" in result["commands"][0]


def test_submit_persists_exactly_one_job_and_exp_sbatch(monkeypatch, tmp_path: Path) -> None:
    _, store, runner = _submit(monkeypatch, tmp_path)
    state = store.load()
    assert len(state["slurm_jobs"]) == 1
    job = state["slurm_jobs"][0]
    assert job["job_id"] == JOB_ID
    assert job["submit_argv"][0] == "scripts/exp_sbatch.sh"
    assert job["submitted_commit"] == COMMIT
    assert job["output_root"].endswith("/phase_b_unit")
    assert len(runner.calls) == 1
    assert "scripts/slurm/clear_mutagenicity_phase_b_gpu_smoke.sh" in job["submit_argv"]


def test_resume_never_resubmits_persisted_job(monkeypatch, tmp_path: Path) -> None:
    spec, store, _ = _submit(monkeypatch, tmp_path)
    experimentctl.dump_spec_snapshot(
        spec, experimentctl._execution_snapshot_path(store)
    )
    runner = QueueRunner(_result(f"{JOB_ID}|PENDING|N/A\n"))
    result = experimentctl.resume_run(store, runner=runner)
    assert result["state"]["status"] == "SUBMITTED"
    assert len(runner.calls) == 1
    assert "sacct" in runner.calls[0][-1]
    assert len(store.load()["slurm_jobs"]) == 1


def test_output_exists_guard_blocks_without_job_id(monkeypatch, tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(experimentctl.AutomationBlocked):
        _submit(
            monkeypatch,
            tmp_path,
            store=store,
            result=_result(stderr="output already exists", returncode=1),
        )
    assert store.load()["slurm_jobs"] == []
    assert json.loads((store.run_dir / "BLOCKED_REPORT.json").read_text())["failed_stage"] == (
        "phase_b_gpu_smoke_submit"
    )


def test_unparseable_job_id_blocks_without_retry(monkeypatch, tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(experimentctl.AutomationBlocked, match="marker"):
        _submit(monkeypatch, tmp_path, store=store, result=_result("job 123\n"))
    assert store.load()["slurm_jobs"] == []


@pytest.mark.parametrize("state", ("PENDING", "RUNNING"))
def test_active_job_is_resumable_and_does_not_run_gate(
    monkeypatch, tmp_path: Path, state: str
) -> None:
    spec, store, _ = _submit(monkeypatch, tmp_path)
    runner = QueueRunner(_result(f"{JOB_ID}|{state}|N/A\n"))
    result = experimentctl.refresh_status(store, spec, runner=runner)
    assert len(runner.calls) == 1
    assert store.load()["status"] in {"SUBMITTED", "RUNNING"}
    assert "phase_b_gpu_smoke_gate" not in store.load()["stages"]
    assert result["next_refresh_after_seconds"] == 120


@pytest.mark.parametrize(
    "state",
    ("FAILED", "CANCELLED", "TIMEOUT", "OOM", "OUT_OF_MEMORY", "NODE_FAIL"),
)
def test_terminal_slurm_failures_block_before_gate(
    monkeypatch, tmp_path: Path, state: str
) -> None:
    spec, store, _ = _submit(monkeypatch, tmp_path)
    runner = QueueRunner(_result(f"{JOB_ID}|{state}|1:0\n"))
    result = experimentctl.refresh_status(store, spec, runner=runner)
    assert result["status"] == "BLOCKED"
    assert len(runner.calls) == 1
    assert "phase_b_gpu_smoke_gate" not in store.load()["stages"]
    report = json.loads((store.run_dir / "BLOCKED_REPORT.json").read_text())
    assert report["details"]["slurm_state"] == state


def test_completed_with_nonzero_exit_blocks_before_gate(monkeypatch, tmp_path: Path) -> None:
    spec, store, _ = _submit(monkeypatch, tmp_path)
    runner = QueueRunner(_result(f"{JOB_ID}|COMPLETED|1:0\n"))
    result = experimentctl.refresh_status(store, spec, runner=runner)
    assert result["status"] == "BLOCKED"
    assert len(runner.calls) == 1
    assert "phase_b_gpu_smoke_gate" not in store.load()["stages"]


def test_completed_zero_runs_gate_and_stops_before_full(
    monkeypatch, tmp_path: Path
) -> None:
    spec, store, _ = _submit(monkeypatch, tmp_path)
    runner = QueueRunner(
        _result(f"{JOB_ID}|COMPLETED|0:0\n"),
        _result(_encoded_gate(_gate_payload())),
    )
    result = experimentctl.refresh_status(store, spec, runner=runner)
    assert len(runner.calls) == 2
    assert result["status"] == "STOPPED_BEFORE_APPROVAL"
    assert result["next_stage"] == "phase_c_full_run"
    state = store.load()
    assert state["stages"]["phase_b_gpu_smoke_gate"]["status"] == "PASSED"
    assert "phase_c_full_run" not in state["approvals"]
    assert state["slurm_jobs"][0]["gate_result"]
    events = [
        json.loads(line)
        for line in store.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        event.get("event_type") == "state_transition"
        and event.get("status") == "AUDITING"
        for event in events
    )
    evidence = json.loads(
        (store.run_dir / "evidence/phase_b_gpu_smoke_gate.json").read_text()
    )
    assert evidence["approved_stage"] == "phase_b_gpu_smoke"
    assert evidence["approval"]["reason"] == (
        "Reviewed the 64-parent smoke boundary."
    )
    assert evidence["remote_write_performed"] is True
    assert evidence["gate_remote_write_performed"] is False
    report = json.loads((store.run_dir / "FINAL_REPORT.json").read_text())
    assert report["next_allowed_stage"] == "phase_c_full_run"
    assert len(report["slurm_jobs"]) == 1


def test_completed_with_missing_marker_evidence_blocks(monkeypatch, tmp_path: Path) -> None:
    spec, store, _ = _submit(monkeypatch, tmp_path)
    runner = QueueRunner(
        _result(f"{JOB_ID}|COMPLETED|0:0\n"),
        _result("gate produced no evidence\n", returncode=1),
    )
    result = experimentctl.refresh_status(store, spec, runner=runner)
    assert result["status"] == "BLOCKED"
    evidence = json.loads(
        (store.run_dir / "evidence/phase_b_gpu_smoke_gate.json").read_text()
    )
    assert any("evidence_parse_failure" in item for item in evidence["failed_hard_checks"])


@pytest.mark.parametrize(
    "override",
    (
        {"checks__selected_generation_parents": 63},
        {"checks__calibration_loaded": True, "provenance__calibration_loaded": True},
        {"checks__test_loaded": True, "provenance__test_loaded": True},
    ),
)
def test_completed_with_bad_scientific_fields_blocks(
    monkeypatch, tmp_path: Path, override: dict
) -> None:
    spec, store, _ = _submit(monkeypatch, tmp_path)
    runner = QueueRunner(
        _result(f"{JOB_ID}|COMPLETED|0:0\n"),
        _result(_encoded_gate(_gate_payload(**override))),
    )
    result = experimentctl.refresh_status(store, spec, runner=runner)
    assert result["status"] == "BLOCKED"


def test_commit_change_blocks_resume_before_ssh(monkeypatch, tmp_path: Path) -> None:
    spec, store, _ = _submit(monkeypatch, tmp_path)
    monkeypatch.setattr(experimentctl, "head_commit", lambda *args: "b" * 40)
    runner = ExplodingRunner()
    result = experimentctl.refresh_status(store, spec, runner=runner)
    assert result["status"] == "BLOCKED"
    assert runner.calls == []


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _valid_gate_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    project = tmp_path / "project"
    output = project / "outputs/hpc/mutagenicity/baselines/clear/automation_phase_b_smoke/run"
    output.mkdir(parents=True)
    summary = {
        "model_train_rows": 2885,
        "model_val_rows": 355,
        "generation_source_parent_rows": 1448,
        "selected_generation_parents": 64,
        "parent_limit": 64,
        "generation_chunk_size": 16,
        "graphpred_epochs": 5,
        "cfe_epochs": 5,
        "batch_size": 8,
        "seed": 13,
        "generation_profile": "smoke",
        "generation_only": False,
        "model_training_performed": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "calibration_rows_loaded": 0,
        "test_rows_loaded": 0,
        "candidate_pool_rows": 1,
        "canonical_unique_candidates": 1,
        "run_complete": True,
    }
    manifest = {
        "parent_limit": 64,
        "generation_chunk_size": 16,
        "seed": 13,
        "generation_profile": "smoke",
        "generation_only": False,
        "model_training_performed": True,
        "generation_input_split": "train",
        "model_train_split": "train",
        "model_validation_split": "val",
        "candidate_selection_performed": False,
        "source_label": 1,
        "target_label": 0,
        "calibration_loaded": False,
        "test_loaded": False,
        "run_complete": True,
        "git_commit": COMMIT,
        "generation_parent_ids": [f"p{index}" for index in range(64)],
        "inputs": {
            "phase_a_root": "outputs/hpc/mutagenicity/final/clear_phase_a_dataset_codec_best",
            "generation_csv": "outputs/hpc/datasets/mutagenicity_v1_teacher_consistent/train_source_label1_teacher_correct.csv",
            "teacher_path": "outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl",
            "official_root": "baselines/clear_official",
        },
    }
    progress = {
        "selected_parent_count": 64,
        "completed_chunk_count": 4,
        "generation_profile": "smoke",
        "run_complete": True,
    }
    audit = {
        "model_train_rows": 2885,
        "model_val_rows": 355,
        "generation_source_rows": 1448,
        "selected_generation_parents": 64,
        "generation_profile": "smoke",
        "generation_only": False,
        "model_training_performed": True,
        "completed_chunk_count": 4,
        "candidate_pool_rows": 1,
        "candidate_universe_rows": 1,
        "calibration_rows_loaded": 0,
        "test_rows_loaded": 0,
        "chunk_resume_duplicate_rows": 0,
        "run_complete": True,
        "audit_passed": True,
    }
    marker = {
        "run_complete": True,
        "generation_profile": "smoke",
    }
    automation = {
        "run_complete": True,
        "git_commit": COMMIT,
        "job_id": "1",
        "output_dir": str(output.resolve()),
        "calibration_loaded": False,
        "test_loaded": False,
    }
    for name, payload in (
        ("summary.json", summary),
        ("run_manifest.json", manifest),
        ("generation_progress.json", progress),
        ("train_pool_audit.json", audit),
        ("_RUN_COMPLETE.json", marker),
        ("_AUTOMATION_PHASE_B_GPU_SMOKE_COMPLETE.json", automation),
    ):
        _write_json(output / name, payload)
    for name in (
        "raw_generated_candidates.jsonl",
        "candidate_pool.jsonl",
        "candidate_universe.jsonl",
    ):
        (output / name).write_text("{}\n", encoding="utf-8")
    stdout = project / "logs/clear_phase_b_smoke_1.out"
    stderr = project / "logs/clear_phase_b_smoke_1.err"
    stdout.parent.mkdir(parents=True)
    stdout.write_text("\n".join(SUCCESS_MARKERS) + "\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    return project, output, stdout, stderr


def test_read_only_scientific_gate_accepts_complete_smoke(tmp_path: Path) -> None:
    project, output, stdout, stderr = _valid_gate_tree(tmp_path)
    evidence = evaluate(
        project_root=project.resolve(),
        output_dir=output.resolve(),
        stdout_log=stdout.resolve(),
        stderr_log=stderr.resolve(),
        expected_commit=COMMIT,
        expected_job_id="1",
    )
    assert evidence["audit_passed"] is True
    assert evidence["failed_hard_checks"] == []


def test_read_only_scientific_gate_rejects_nonfinite_and_traceback(tmp_path: Path) -> None:
    project, output, stdout, stderr = _valid_gate_tree(tmp_path)
    summary = json.loads((output / "summary.json").read_text())
    summary["source_parent_coverage"] = float("nan")
    _write_json(output / "summary.json", summary)
    stderr.write_text("Traceback (most recent call last)\n", encoding="utf-8")
    evidence = evaluate(
        project_root=project.resolve(),
        output_dir=output.resolve(),
        stdout_log=stdout.resolve(),
        stderr_log=stderr.resolve(),
        expected_commit=COMMIT,
        expected_job_id="1",
    )
    assert evidence["audit_passed"] is False
    assert any("non_finite" in value for value in evidence["failed_hard_checks"])
    assert any("fatal_log_token" in value for value in evidence["failed_hard_checks"])


def test_read_only_scientific_gate_rejects_missing_completion_marker(
    tmp_path: Path,
) -> None:
    project, output, stdout, stderr = _valid_gate_tree(tmp_path)
    (output / "_AUTOMATION_PHASE_B_GPU_SMOKE_COMPLETE.json").unlink()
    evidence = evaluate(
        project_root=project.resolve(),
        output_dir=output.resolve(),
        stdout_log=stdout.resolve(),
        stderr_log=stderr.resolve(),
        expected_commit=COMMIT,
        expected_job_id="1",
    )
    assert evidence["audit_passed"] is False
    assert any(
        "_AUTOMATION_PHASE_B_GPU_SMOKE_COMPLETE.json" in value
        for value in evidence["failed_hard_checks"]
    )


def test_wrapper_reuses_existing_smoke_and_never_targets_full_or_splits() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert "bash scripts/slurm/clear_mutagenicity_train_pool.sh" in text
    assert "PARENT_LIMIT=64" in text
    assert "GRAPHPRED_EPOCHS=5" in text
    assert "CFE_EPOCHS=5" in text
    assert "GENERATION_CHUNK_SIZE=16" in text
    assert "RESUME=false" in text
    assert "parent_limit=1448" not in text
    assert "calibration_source" not in text
    assert "test_source" not in text
    assert "unset http_proxy" not in text
    assert "unset https_proxy" not in text
    assert "outputs/hpc/mutagenicity/baselines/clear/phase_a_dataset_codec_v2" not in text
    for destructive in ("rm -", "mv ", "git reset", "git clean", "git stash"):
        assert destructive not in text
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_wrapper_resources_match_validated_manual_smoke() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    for line in (
        "#SBATCH --partition=A800",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --gres=gpu:a800:1",
        "#SBATCH --mem=64G",
        "#SBATCH --time=04:00:00",
    ):
        assert line in text


def test_full_stage_remains_approval_only() -> None:
    spec = load_task_spec(SPEC_PATH)
    full = spec.stage_by_id["phase_c_full_run"]
    assert full["kind"] == "approval"
    assert full["script"] is None
    assert spec.data["permissions"]["allow_full"] is False
