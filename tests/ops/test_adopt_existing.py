from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from scripts.ops import experimentctl
from scripts.ops.adopt_existing import (
    build_remote_script,
    encode_evidence,
    verify_existing_artifacts,
)
from scripts.ops.spec import TaskSpec, load_task_spec
from scripts.ops.ssh_ops import SSHConfig
from scripts.ops.subprocess_utils import CommandResult


ROOT = Path(__file__).resolve().parents[2]
LEGACY_COMMIT = "f83f701a03306ba6ab0008ea61ce0cc34a2defca"
CURRENT_COMMIT = "d" * 40
OUTPUT_REL = Path(
    "outputs/hpc/mutagenicity/baselines/clear/phase_a_dataset_codec_v2"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def legacy_fixture(tmp_path: Path) -> tuple[Path, dict]:
    project = tmp_path / "project"
    output = project / OUTPUT_REL
    probe = output / "codec_probe_64"
    dataset_summary = output / "dataset_summary.json"
    probe_summary = probe / "codec_probe_summary.json"
    probe_rows = probe / "codec_probe_rows.jsonl"
    _write_json(
        dataset_summary,
        {
            "train_positive_rows": 1448,
            "train_negative_rows": 1437,
            "val_positive_rows": 260,
            "val_negative_rows": 95,
            "atom_sidecar_schema_version": (
                "clear_mutagenicity_atom_sidecar_v2"
            ),
            "calibration_loaded": False,
            "test_loaded": False,
        },
    )
    _write_json(
        probe_summary,
        {
            "probe_rows": 64,
            "probe_failed_rows": 0,
            "probe_passed": True,
            "atom_sidecar_schema_version": (
                "clear_mutagenicity_atom_sidecar_v2"
            ),
            "calibration_loaded": False,
            "test_loaded": False,
        },
    )
    probe_rows.parent.mkdir(parents=True, exist_ok=True)
    probe_rows.write_text(
        "".join(json.dumps({"row": index}) + "\n" for index in range(64)),
        encoding="utf-8",
    )
    artifacts = []
    for path in (dataset_summary, probe_summary, probe_rows):
        artifacts.append(
            {
                "path": path.relative_to(output).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha(path),
            }
        )
    _write_json(
        output / "phase_a_manifest.json",
        {"git_commit": LEGACY_COMMIT, "artifacts": artifacts},
    )
    _write_json(
        output / "_PHASE_A_COMPLETE.json",
        {
            "phase_a_complete": True,
            "manifest": "phase_a_manifest.json",
            "git_commit": LEGACY_COMMIT,
        },
    )
    current_probe = (OUTPUT_REL / "codec_probe").as_posix()
    legacy_probe = (OUTPUT_REL / "codec_probe_64").as_posix()
    config = {
        "mode": "legacy_manifest_sha256",
        "output_root": OUTPUT_REL.as_posix(),
        "completion_marker": (OUTPUT_REL / "_PHASE_A_COMPLETE.json").as_posix(),
        "manifest_path": (OUTPUT_REL / "phase_a_manifest.json").as_posix(),
        "finalized_marker": (OUTPUT_REL / "_FINALIZED.json").as_posix(),
        "expected_generation_commit": LEGACY_COMMIT,
        "artifact_aliases": {
            f"{current_probe}/codec_probe_summary.json": (
                f"{legacy_probe}/codec_probe_summary.json"
            ),
            f"{current_probe}/codec_probe_rows.jsonl": (
                f"{legacy_probe}/codec_probe_rows.jsonl"
            ),
        },
        "jsonl_row_counts": {
            f"{current_probe}/codec_probe_rows.jsonl": 64
        },
        "allow_missing_current_markers": True,
        "adopted_stages": [
            "phase_a_prepare",
            "phase_a_probe",
            "phase_a_audit",
        ],
        "next_stage": "phase_b_gpu_smoke",
        "current_local_commit": CURRENT_COMMIT,
        "stage_gates": [
            {
                "stage_id": "phase_a_prepare",
                "json_path": (OUTPUT_REL / "dataset_summary.json").as_posix(),
                "required_fields": {
                    "train_positive_rows": 1448,
                    "train_negative_rows": 1437,
                    "val_positive_rows": 260,
                    "val_negative_rows": 95,
                    "atom_sidecar_schema_version": (
                        "clear_mutagenicity_atom_sidecar_v2"
                    ),
                    "calibration_loaded": False,
                    "test_loaded": False,
                },
                "forbidden_fields": {},
            },
            {
                "stage_id": "phase_a_probe",
                "json_path": f"{current_probe}/codec_probe_summary.json",
                "required_fields": {
                    "probe_rows": 64,
                    "probe_failed_rows": 0,
                    "probe_passed": True,
                    "atom_sidecar_schema_version": (
                        "clear_mutagenicity_atom_sidecar_v2"
                    ),
                    "calibration_loaded": False,
                    "test_loaded": False,
                },
                "forbidden_fields": {},
            },
            {
                "stage_id": "phase_a_audit",
                "json_path": (OUTPUT_REL / "phase_a_gate.json").as_posix(),
                "required_fields": {"audit_passed": True},
                "forbidden_fields": {},
            },
        ],
    }
    return project, config


def verify(project: Path, config: dict, commit: str = CURRENT_COMMIT) -> dict:
    return verify_existing_artifacts(
        project, config, current_remote_commit=commit
    )


def test_successful_legacy_verification_is_complete_and_read_only(
    tmp_path: Path,
) -> None:
    project, config = legacy_fixture(tmp_path)
    before = sorted(
        (path.relative_to(project).as_posix(), path.read_bytes())
        for path in project.rglob("*")
        if path.is_file()
    )
    evidence = verify(project, config)
    after = sorted(
        (path.relative_to(project).as_posix(), path.read_bytes())
        for path in project.rglob("*")
        if path.is_file()
    )
    assert evidence["verification_passed"] is True
    assert evidence["artifact_count"] == 3
    assert evidence["current_required_marker_present"] is False
    assert evidence["accepted_via_legacy_manifest_integrity"] is True
    assert evidence["legacy_generation_commit"] == LEGACY_COMMIT
    assert before == after


@pytest.mark.parametrize(
    ("missing_name", "failure"),
    [
        ("_PHASE_A_COMPLETE.json", "completion_marker_missing"),
        ("phase_a_manifest.json", "manifest_missing"),
    ],
)
def test_required_legacy_control_file_missing_fails(
    tmp_path: Path, missing_name: str, failure: str
) -> None:
    project, config = legacy_fixture(tmp_path)
    (project / OUTPUT_REL / missing_name).unlink()
    assert failure in verify(project, config)["failed_hard_checks"]


def test_generation_commit_mismatch_fails(tmp_path: Path) -> None:
    project, config = legacy_fixture(tmp_path)
    config["expected_generation_commit"] = "e" * 40
    assert "generation_commit_mismatch" in verify(project, config)[
        "failed_hard_checks"
    ]


def test_artifact_missing_fails(tmp_path: Path) -> None:
    project, config = legacy_fixture(tmp_path)
    (project / OUTPUT_REL / "dataset_summary.json").unlink()
    failures = verify(project, config)["failed_hard_checks"]
    assert any(value.startswith("artifact_missing:") for value in failures)


@pytest.mark.parametrize("field", ["size", "sha256"])
def test_manifest_size_or_sha_mismatch_fails(
    tmp_path: Path, field: str
) -> None:
    project, config = legacy_fixture(tmp_path)
    manifest_path = project / OUTPUT_REL / "phase_a_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0][field] = (
        manifest["artifacts"][0][field] + 1
        if field == "size"
        else "0" * 64
    )
    _write_json(manifest_path, manifest)
    failures = verify(project, config)["failed_hard_checks"]
    expected = "artifact_size_mismatch:" if field == "size" else "artifact_sha256_mismatch:"
    assert any(value.startswith(expected) for value in failures)


@pytest.mark.parametrize("field", ["calibration_loaded", "test_loaded"])
def test_forbidden_split_loaded_fails(tmp_path: Path, field: str) -> None:
    project, config = legacy_fixture(tmp_path)
    summary = project / OUTPUT_REL / "dataset_summary.json"
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload[field] = True
    _write_json(summary, payload)
    failures = verify(project, config)["failed_hard_checks"]
    assert any("scientific_field_mismatch" in value for value in failures)


def test_finalized_marker_fails(tmp_path: Path) -> None:
    project, config = legacy_fixture(tmp_path)
    _write_json(project / OUTPUT_REL / "_FINALIZED.json", {})
    assert "finalized_marker_exists" in verify(project, config)[
        "failed_hard_checks"
    ]


def test_probe_row_count_mismatch_fails(tmp_path: Path) -> None:
    project, config = legacy_fixture(tmp_path)
    rows = project / OUTPUT_REL / "codec_probe_64/codec_probe_rows.jsonl"
    rows.write_text("{}\n", encoding="utf-8")
    failures = verify(project, config)["failed_hard_checks"]
    assert any(value.startswith("jsonl_row_count_mismatch:") for value in failures)


def test_current_local_remote_commit_mismatch_fails(tmp_path: Path) -> None:
    project, config = legacy_fixture(tmp_path)
    assert "current_local_remote_commit_mismatch" in verify(
        project, config, commit="c" * 40
    )["failed_hard_checks"]


def ssh_config() -> SSHConfig:
    return SSHConfig(
        host="logini.tongji.edu.cn",
        port=10022,
        user="u20526",
        remote_root="/share/home/u20526/czx/counterfactual-subgraph",
        conda_env="smiles_pip118",
        control_socket=None,
    )


def test_remote_adoption_script_is_read_only_and_bash_valid(tmp_path: Path) -> None:
    _, config = legacy_fixture(tmp_path)
    script = build_remote_script(ssh_config(), config)
    syntax = subprocess.run(
        ["bash", "-n"], input=script, text=True, capture_output=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    for command in (
        "mkdir", "rm", "mv", "cp", "ln", "touch", "chmod", "chown",
        "git fetch", "git pull", "git push", "git reset", "git clean",
        "git restore", "git stash", "sbatch", "srun", "scancel",
    ):
        assert not any(
            line.strip().startswith(command) for line in script.splitlines()
        )
    assert "export PYTHONDONTWRITEBYTECODE=1" in script


def test_cli_parser_and_help_support_adopt_existing(capsys) -> None:
    args = experimentctl._parser().parse_args(
        ["adopt-existing", "task.yaml", "--dry-run"]
    )
    assert args.command == "adopt-existing"
    assert args.dry_run is True
    with pytest.raises(SystemExit) as exit_info:
        experimentctl._parser().parse_args(["adopt-existing", "--help"])
    assert exit_info.value.code == 0
    assert "--dry-run" in capsys.readouterr().out


@pytest.mark.parametrize(
    "permission",
    [
        "allow_remote_write", "allow_sbatch", "allow_overwrite",
        "allow_gpu_smoke", "allow_full", "allow_calibration", "allow_test",
        "allow_finalization",
    ],
)
def test_any_execution_permission_rejects_adoption(permission: str) -> None:
    spec = load_task_spec(
        ROOT / "ops/specs/clear_mutagenicity_phase_a_v2.yaml"
    )
    payload = deepcopy(spec.data)
    payload["permissions"][permission] = True
    with pytest.raises(experimentctl.AutomationBlocked, match="permissions"):
        experimentctl._require_adoption_permissions(
            TaskSpec(path=spec.path, data=payload)
        )


def test_permission_boundary_writes_only_local_blocked_report(
    tmp_path: Path,
) -> None:
    spec = load_task_spec(
        ROOT / "ops/specs/clear_mutagenicity_phase_a_v2.yaml"
    )
    payload = deepcopy(spec.data)
    payload["permissions"]["allow_remote_write"] = True
    unsafe = TaskSpec(path=spec.path, data=payload)
    runner = ExplodingRunner()
    runner.calls = []
    result = experimentctl.adopt_existing(
        unsafe,
        dry_run=False,
        run_dir=tmp_path / "permission-blocked",
        runner=runner,
    )
    assert result["status"] == "BLOCKED"
    assert result["remote_write_performed"] is False
    assert result["slurm_jobs"] == []
    assert runner.calls == []
    assert (tmp_path / "permission-blocked/BLOCKED_REPORT.md").is_file()


def test_alias_path_traversal_is_rejected(tmp_path: Path) -> None:
    payload = yaml.safe_load(
        (ROOT / "ops/specs/clear_mutagenicity_phase_a_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    first = next(iter(payload["adopt_existing"]["artifact_aliases"]))
    payload["adopt_existing"]["artifact_aliases"][first] = "../escape.json"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="repository-relative|output_root"):
        load_task_spec(path)


class ExplodingRunner:
    calls: list[list[str]] = []

    def run(self, argv, **kwargs):
        self.calls.append(list(argv))
        raise AssertionError("dry-run contacted an external process")


class OneResultRunner:
    def __init__(self, result: CommandResult) -> None:
        self.result = result
        self.calls: list[list[str]] = []

    def run(self, argv, **kwargs):
        self.calls.append(list(argv))
        return self.result


def test_adopt_dry_run_has_contract_and_no_execution(
    monkeypatch, tmp_path: Path
) -> None:
    spec = load_task_spec(
        ROOT / "ops/specs/clear_mutagenicity_phase_a_v2.yaml"
    )
    monkeypatch.setattr(experimentctl, "head_commit", lambda *args: CURRENT_COMMIT)
    runner = ExplodingRunner()
    runner.calls = []
    result = experimentctl.adopt_existing(
        spec,
        dry_run=True,
        run_dir=tmp_path / "dry",
        runner=runner,
    )
    assert result["status"] == "ADOPT_EXISTING_DRY_RUN"
    assert result["adopted_stages"] == [
        "phase_a_prepare", "phase_a_probe", "phase_a_audit"
    ]
    assert result["next_stage"] == "phase_b_gpu_smoke"
    assert result["remote_write_performed"] is False
    assert result["slurm_jobs"] == []
    assert runner.calls == []
    syntax = subprocess.run(
        ["bash", "-n"],
        input=result["remote_script"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr


def test_successful_adoption_persists_state_report_and_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    project, config = legacy_fixture(tmp_path)
    evidence = verify(project, config)
    spec = load_task_spec(
        ROOT / "ops/specs/clear_mutagenicity_phase_a_v2.yaml"
    )
    monkeypatch.setattr(experimentctl, "head_commit", lambda *args: CURRENT_COMMIT)

    def pass_local(_spec, store, stage, _runner):
        store.record_stage(stage["id"], {"status": "PASSED", "attempt": 1})
        return True

    monkeypatch.setattr(experimentctl, "_run_local_stage", pass_local)
    result_obj = CommandResult(
        argv=["ssh"], cwd=str(tmp_path), returncode=0,
        stdout="[ADOPT_EXISTING_EVIDENCE_B64] " + encode_evidence(evidence),
        stderr="",
    )
    runner = OneResultRunner(result_obj)
    result = experimentctl.adopt_existing(
        spec,
        dry_run=False,
        run_dir=tmp_path / "adopted",
        runner=runner,
    )
    assert result["status"] == "STOPPED_BEFORE_APPROVAL"
    state = json.loads((tmp_path / "adopted/state.json").read_text())
    for stage_id in ("phase_a_prepare", "phase_a_probe", "phase_a_audit"):
        stage = state["stages"][stage_id]
        assert stage["status"] == "ADOPTED_EXISTING"
        assert stage["command_status"] == "NOT_EXECUTED"
        assert stage["gate_status"] == "PASSED"
        assert stage["job_id"] is None
    assert state["approvals"] == {}
    verification = state["stages"]["adopt_existing_verification"]
    assert verification["status"] == "PASSED"
    assert verification["command_status"] == "PASSED"
    assert verification["gate_status"] == "PASSED"
    saved = json.loads(
        (tmp_path / "adopted/evidence/adopt_existing_evidence.json").read_text()
    )
    assert saved["remote_write_performed"] is False
    report = (tmp_path / "adopted/FINAL_REPORT.md").read_text()
    assert "Phase A was adopted from verified legacy artifacts." in report
    assert "No Slurm job was submitted." in report
    assert "phase_b_gpu_smoke pending explicit approval" in report


def test_local_phase_a_failure_blocks_before_ssh(
    monkeypatch, tmp_path: Path
) -> None:
    spec = load_task_spec(
        ROOT / "ops/specs/clear_mutagenicity_phase_a_v2.yaml"
    )
    monkeypatch.setattr(experimentctl, "head_commit", lambda *args: CURRENT_COMMIT)

    def fail_local(_spec, store, stage, _runner):
        stderr_path = store.run_dir / "logs/local_phase_a_tests/failure.log"
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.write_text("local gate failed\n", encoding="utf-8")
        store.record_stage(
            stage["id"],
            {
                "status": "FAILED",
                "attempt": 1,
                "return_code": 1,
                "stderr_path": str(stderr_path),
                "gate_result": None,
            },
        )
        return False

    monkeypatch.setattr(experimentctl, "_run_local_stage", fail_local)
    runner = ExplodingRunner()
    runner.calls = []
    result = experimentctl.adopt_existing(
        spec,
        dry_run=False,
        run_dir=tmp_path / "local-failed",
        runner=runner,
    )
    assert result["status"] == "BLOCKED"
    assert runner.calls == []
    assert (tmp_path / "local-failed/BLOCKED_REPORT.md").is_file()


def test_failed_remote_evidence_does_not_adopt_stages(
    monkeypatch, tmp_path: Path
) -> None:
    project, config = legacy_fixture(tmp_path)
    evidence = verify(project, config)
    evidence["verification_passed"] = False
    evidence["failed_hard_checks"] = ["artifact_sha256_mismatch:x"]
    spec = load_task_spec(
        ROOT / "ops/specs/clear_mutagenicity_phase_a_v2.yaml"
    )
    monkeypatch.setattr(experimentctl, "head_commit", lambda *args: CURRENT_COMMIT)

    def pass_local(_spec, store, stage, _runner):
        store.record_stage(stage["id"], {"status": "PASSED", "attempt": 1})
        return True

    monkeypatch.setattr(experimentctl, "_run_local_stage", pass_local)
    runner = OneResultRunner(
        CommandResult(
            argv=["ssh"],
            cwd=str(tmp_path),
            returncode=3,
            stdout=(
                "[ADOPT_EXISTING_EVIDENCE_B64] "
                + encode_evidence(evidence)
            ),
            stderr="legacy integrity gate failed",
        )
    )
    result = experimentctl.adopt_existing(
        spec,
        dry_run=False,
        run_dir=tmp_path / "remote-failed",
        runner=runner,
    )
    state = json.loads((tmp_path / "remote-failed/state.json").read_text())
    assert result["status"] == "BLOCKED"
    assert not any(
        state["stages"].get(stage_id, {}).get("status") == "ADOPTED_EXISTING"
        for stage_id in ("phase_a_prepare", "phase_a_probe", "phase_a_audit")
    )
    assert (tmp_path / "remote-failed/BLOCKED_REPORT.md").is_file()
