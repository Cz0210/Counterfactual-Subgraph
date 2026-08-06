from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts/automation/run_comrecgc_recovery.py"
)
SPEC = importlib.util.spec_from_file_location("run_comrecgc_recovery", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_recovery_ssh_child_is_batch_only_and_clears_forwardings() -> None:
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


def test_mut_full_submission_is_deferred_until_smoke_gate_passes(
    tmp_path: Path, monkeypatch
) -> None:
    state = MODULE.RecoveryState(
        tmp_path,
        "run1",
        requested_mode="all",
        datasets=["mutagenicity"],
    )
    stage_values = MODULE.stages("run1", {"mutagenicity"}, "all")

    def fake_submit(stage, run_state, *, dry_run):
        return run_state.add_job(stage, f"job_{stage.stage_id}", [stage.script])

    monkeypatch.setattr(MODULE, "submit", fake_submit)
    initially = MODULE.submit_ready_stages(stage_values, state, dry_run=False)
    assert "mut_chemrepair_smoke_gate" in initially
    assert "mut_full_generation" not in initially
    assert state.job("mut_full_generation") is None

    state.data["completed_stages"] = ["mut_chemrepair_smoke_gate"]
    after_gate = MODULE.submit_ready_stages(stage_values, state, dry_run=False)
    assert after_gate == [
        "mut_full_generation",
        "mut_full_common_recourse",
        "mut_full_chemistry",
        "mut_full_unified_eval",
        "mut_full_gate",
        "mut_freeze",
    ]


def test_recovery_submissions_use_experiment_registry() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"scripts/exp_sbatch.sh"' in source
    assert 'subprocess.run(["sbatch"' not in source
    assert 'command(["command", "-v"' not in source
    assert 'shutil.which("sbatch")' in source


def test_slot_preserving_eval_is_in_smoke_and_full_dag() -> None:
    values = MODULE.stages("run2", {"mutagenicity"}, "all")
    by_id = {stage.stage_id: stage for stage in values}
    assert by_id["mut_unified_eval_smoke"].dependency_stages == (
        "mut_chemistry_audit",
    )
    assert by_id["mut_chemrepair_smoke_gate"].dependency_stages == (
        "mut_unified_eval_smoke",
    )
    assert by_id["mut_full_unified_eval"].dependency_stages == (
        "mut_full_chemistry",
    )
    assert by_id["mut_full_gate"].dependency_type == "afterany"
    assert by_id["mut_freeze"].dependency_stages == ("mut_full_gate",)
