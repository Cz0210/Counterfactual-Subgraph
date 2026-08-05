from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/automation/run_comrecgc_baseline.py"
SPEC = importlib.util.spec_from_file_location("run_comrecgc_baseline", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_state_job_submission_is_idempotent(tmp_path: Path) -> None:
    state = MODULE.RunState(tmp_path, run_id="run1", datasets=["aids"], mode="smoke")
    spec = MODULE.JobSpec(
        stage="smoke_generation",
        dataset="aids",
        script="scripts/slurm/comrecgc_project_generate.sh",
        output_root="out",
        dependency=None,
        environment={},
    )
    state.add_job(spec, "123", ["scripts/exp_sbatch.sh"])
    state.add_job(spec, "456", ["scripts/exp_sbatch.sh"])
    assert len(state.data["jobs"]) == 1
    assert state.data["jobs"][0]["job_id"] == "123"
    assert len((tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()) >= 2


def test_ssh_children_clear_forwardings() -> None:
    args = argparse.Namespace(
        control_socket="/tmp/tongji-codex.sock",
        remote_port=10022,
        remote_host="u20526@logini.tongji.edu.cn",
    )
    argv = MODULE.ssh_argv(args, "hostname")
    assert "BatchMode=yes" in argv
    assert "ClearAllForwardings=yes" in argv
    assert argv.count("bash") == 1
    assert argv.count("-lc") == 1


def test_run_state_has_machine_readable_jobs_list(tmp_path: Path) -> None:
    MODULE.RunState(tmp_path, run_id="run2", datasets=["mutagenicity"], mode="smoke")
    payload = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert payload["jobs"] == []


def test_submit_uses_registered_job_id_without_parsable(
    tmp_path: Path, monkeypatch
) -> None:
    state = MODULE.RunState(tmp_path, run_id="run3", datasets=["aids"], mode="smoke")
    spec = MODULE.JobSpec(
        stage="smoke_generation",
        dataset="aids",
        script="scripts/slurm/comrecgc_project_generate.sh",
        output_root="outputs/run_20260806",
        dependency="afterok:12345",
        environment={},
    )
    captured: dict[str, list[str]] = {}

    def fake_run(argv, *, cwd, timeout=600, check=True):
        captured["argv"] = list(argv)
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                "[EXP_SUBMIT_OK]\n"
                "job_id=2089235\n"
                "expected_output_root=outputs/run_20260806\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(MODULE, "run_command", fake_run)
    record = MODULE.submit_job(state, spec, dry_run=False)

    assert record["job_id"] == "2089235"
    assert "--parsable" not in captured["argv"]
    assert "--dependency=afterok:12345" in captured["argv"]


def test_submit_rejects_unknown_registered_job_id(tmp_path: Path, monkeypatch) -> None:
    state = MODULE.RunState(tmp_path, run_id="run4", datasets=["aids"], mode="smoke")
    spec = MODULE.JobSpec(
        stage="dataset_identity",
        dataset="shared",
        script="scripts/slurm/comrecgc_dataset_identity.sh",
        output_root="outputs/audits_20260806",
        dependency=None,
        environment={},
    )

    def fake_run(argv, *, cwd, timeout=600, check=True):
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                "[EXP_SUBMIT_OK]\n"
                "job_id=UNKNOWN\n"
                "expected_output_root=outputs/audits_20260806\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(MODULE, "run_command", fake_run)
    try:
        MODULE.submit_job(state, spec, dry_run=False)
    except MODULE.AutomationError as exc:
        assert "Could not parse Slurm job ID" in str(exc)
    else:  # pragma: no cover - guards against unsafe silent recovery.
        raise AssertionError("UNKNOWN job ID must block submission state")

    assert state.data["jobs"] == []
