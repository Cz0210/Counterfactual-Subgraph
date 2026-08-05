from __future__ import annotations

import argparse
import importlib.util
import json
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
