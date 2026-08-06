from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest


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


def test_aids_full_is_absent_from_smoke_and_deferred_in_all(
    tmp_path: Path, monkeypatch
) -> None:
    smoke = MODULE.stages("aids-smoke", {"aids"}, "smoke")
    assert [stage.stage_id for stage in smoke] == [
        "aids_existing_audit",
        "aids_density_retry",
        "aids_project_smoke_gate",
    ]

    state = MODULE.RecoveryState(
        tmp_path,
        "aids-all",
        requested_mode="all",
        datasets=["aids"],
    )
    all_stages = MODULE.stages("aids-all", {"aids"}, "all")

    def fake_submit(stage, run_state, *, dry_run):
        return run_state.add_job(stage, f"job_{stage.stage_id}", [stage.script])

    monkeypatch.setattr(MODULE, "submit", fake_submit)
    initially = MODULE.submit_ready_stages(all_stages, state, dry_run=False)
    assert initially == [
        "aids_existing_audit",
        "aids_density_retry",
        "aids_project_smoke_gate",
    ]
    assert state.job("aids_native_full") is None

    state.data["completed_stages"] = ["aids_project_smoke_gate"]
    after_density = MODULE.submit_ready_stages(all_stages, state, dry_run=False)
    assert after_density == [
        "aids_native_full",
        "aids_native_full_gate",
        "aids_project_full_generation",
        "aids_project_full_common_recourse",
        "aids_project_full_chemistry",
        "aids_project_full_unified_eval",
        "aids_project_full_gate",
        "aids_project_freeze",
    ]


def test_recovery_submissions_use_experiment_registry() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"scripts/exp_sbatch.sh"' in source
    assert 'subprocess.run(["sbatch"' not in source
    assert 'command(["command", "-v"' not in source
    assert 'shutil.which("sbatch")' in source


def test_recovery_dirty_allowlist_is_exact_and_only_for_generated_log() -> None:
    allowed, blocked = MODULE.partition_recovery_dirty(
        [
            " M docs/EXPERIMENT_LOG.md",
            " M docs/EXPERIMENT_LOG.md.evil",
            " M scripts/automation/run_comrecgc_recovery.py",
            "?? docs/",
        ]
    )
    assert allowed == [" M docs/EXPERIMENT_LOG.md"]
    assert blocked == [
        " M docs/EXPERIMENT_LOG.md.evil",
        " M scripts/automation/run_comrecgc_recovery.py",
        "?? docs/",
    ]


def test_slot_preserving_eval_is_in_smoke_and_full_dag() -> None:
    values = MODULE.stages("run2", {"mutagenicity"}, "all")
    by_id = {stage.stage_id: stage for stage in values}
    assert by_id["mut_trace_adopt"].script.endswith(
        "comrecgc_mut_trace_adopt.sh"
    )
    assert by_id["mut_trace_adopt"].dependency_stages == ()
    assert by_id["mut_chemistry_audit"].dependency_stages == (
        "mut_trace_adopt",
    )
    assert by_id["mut_unified_eval_smoke"].dependency_stages == (
        "mut_chemistry_audit",
    )
    assert by_id["mut_chemrepair_smoke_gate"].dependency_stages == (
        "mut_unified_eval_smoke",
    )
    assert by_id["mut_chemrepair_smoke_gate"].dependency_type == "afterok"
    assert by_id["mut_full_unified_eval"].dependency_stages == (
        "mut_full_chemistry",
    )
    assert by_id["mut_full_chemistry"].script.endswith(
        "comrecgc_project_chemistry.sh"
    )
    assert by_id["mut_full_unified_eval"].script.endswith(
        "comrecgc_project_slot_eval.sh"
    )
    assert by_id["mut_full_gate"].script.endswith(
        "comrecgc_project_full_gate.sh"
    )
    assert by_id["mut_freeze"].script.endswith("comrecgc_project_freeze.sh")
    assert by_id["mut_full_chemistry"].environment["TRACE_EVIDENCE_PATH"].endswith(
        "/generation/trace_parity.json"
    )
    assert by_id["mut_full_gate"].dependency_type == "afterok"
    assert by_id["mut_freeze"].dependency_stages == ("mut_full_gate",)


def test_retry3_requires_authorization_file_and_exact_project_commit(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="authorization is missing"):
        MODULE.load_retry3_authorization(tmp_path, project_commit="a" * 40)

    path, _digest = MODULE.initialize_retry3_authorization(
        tmp_path, project_commit="a" * 40
    )
    assert path.is_file()
    with pytest.raises(RuntimeError, match="project_commit"):
        MODULE.load_retry3_authorization(tmp_path, project_commit="b" * 40)


def _authorized_state(tmp_path: Path) -> object:
    authorization = MODULE.retry3_authorization_template(MODULE.git_commit(MODULE.PROJECT_ROOT))
    return MODULE.RecoveryState(
        tmp_path,
        "comrecgc_recovery_test_retry3",
        requested_mode="smoke",
        datasets=["aids", "mutagenicity"],
        authorization=authorization,
        authorization_sha256="f" * 64,
    )


def test_retry3_smoke_job_caps_are_two_four_and_six(tmp_path: Path) -> None:
    state = _authorized_state(tmp_path)
    state.data["jobs"] = [
        {"stage_id": "aids_existing_audit"},
        {"stage_id": "aids_density_retry"},
        {"stage_id": "mut_trace_adopt"},
        {"stage_id": "mut_chemistry_audit"},
        {"stage_id": "mut_unified_eval_smoke"},
        {"stage_id": "mut_chemrepair_smoke_gate"},
    ]
    MODULE.validate_job_caps(state)

    state.data["jobs"].append({"stage_id": "aids_existing_audit"})
    with pytest.raises(RuntimeError, match="AIDS job cap"):
        MODULE.validate_job_caps(state)

    state.data["jobs"] = [
        {"stage_id": "mut_trace_adopt"},
        {"stage_id": "mut_chemistry_audit"},
        {"stage_id": "mut_unified_eval_smoke"},
        {"stage_id": "mut_chemrepair_smoke_gate"},
        {"stage_id": "mut_trace_adopt"},
    ]
    with pytest.raises(RuntimeError, match="Mutagenicity job cap"):
        MODULE.validate_job_caps(state)

    state.data["authorization"]["max_aids_jobs"] = 7
    state.data["authorization"]["max_mutagenicity_jobs"] = 7
    state.data["jobs"] = [
        {"stage_id": "aids_existing_audit"},
        {"stage_id": "aids_density_retry"},
        {"stage_id": "mut_trace_adopt"},
        {"stage_id": "mut_chemistry_audit"},
        {"stage_id": "mut_unified_eval_smoke"},
        {"stage_id": "mut_chemrepair_smoke_gate"},
        {"stage_id": "mut_trace_adopt"},
    ]
    with pytest.raises(RuntimeError, match="total job cap"):
        MODULE.validate_job_caps(state)


def test_retry3_full_submission_is_blocked_before_registry_call(
    tmp_path: Path, monkeypatch
) -> None:
    state = _authorized_state(tmp_path)
    stage = MODULE.Stage(
        "aids_native_full",
        "aids",
        "scripts/slurm/comrecgc_aids_native_full.sh",
        "outputs/never-created",
        None,
        (),
        {},
    )

    def unexpected_command(*_args, **_kwargs):
        raise AssertionError("exp_sbatch must not be reached")

    monkeypatch.setattr(MODULE, "command", unexpected_command)
    with pytest.raises(RuntimeError, match="phase_c_full_approved"):
        MODULE.submit(stage, state, dry_run=False)


def test_retry3_aids_output_roots_are_versioned_by_run_id() -> None:
    run_id = "comrecgc_recovery_20260806_aids_retry3"
    values = MODULE.stages(run_id, {"aids"}, "smoke")
    assert all(run_id in stage.output_root for stage in values)
    assert values[0].environment["OUTPUT_DIR"] == values[0].output_root
    assert values[1].environment["EXISTING_AUDIT"].startswith(values[0].output_root)


def test_retry3_authorization_template_keeps_full_kill_switch_closed() -> None:
    value = MODULE.retry3_authorization_template("a" * 40)
    assert value["authorization_status"] == "AUTHORIZED"
    assert value["authorized_scope"] == "RETRY3_SMOKE_ONLY"
    assert value["phase_c_full_approved"] is False
    assert value["full_submission_allowed"] is False
    assert value["auto_promote_to_full"] is False
    assert value["candidate_regeneration_allowed"] is False
    assert value["random_walk_rerun_allowed"] is False
    assert (value["max_aids_jobs"], value["max_mutagenicity_jobs"]) == (2, 4)
    assert value["max_total_jobs"] == 6


def test_end_to_end_authorization_is_exact_and_enables_only_gated_full() -> None:
    value = MODULE.end_to_end_authorization_template("a" * 40)
    assert value == {
        "authorization_status": "AUTHORIZED",
        "authorized_scope": "COMRECGC_END_TO_END_AIDS_MUTAGENICITY",
        "project_commit": "a" * 40,
        "upstream_commit": MODULE.UPSTREAM_COMMIT,
        "phase_c_full_approved": True,
        "full_submission_allowed": True,
        "auto_promote_to_full": True,
        "candidate_regeneration_in_smoke": False,
        "candidate_generation_in_full": True,
        "random_walk_rerun_in_smoke": False,
        "random_walk_full_allowed": True,
        "scientific_parameter_sweep_allowed": False,
        "rank_backfill_allowed": False,
        "rf_guided_repair_allowed": False,
        "wnode_guided_repair_allowed": False,
    }
    MODULE.assert_full_authorized(value)


def test_end_to_end_dag_is_frozen_and_full_requires_dataset_smoke_gate(
    tmp_path: Path,
) -> None:
    authorization = MODULE.end_to_end_authorization_template(
        MODULE.git_commit(MODULE.PROJECT_ROOT)
    )
    state = MODULE.RecoveryState(
        tmp_path,
        "comrecgc_end_to_end_test",
        requested_mode="all",
        datasets=["aids", "mutagenicity"],
        authorization=authorization,
        authorization_sha256="e" * 64,
    )
    values = MODULE.stages(
        "comrecgc_end_to_end_test", {"aids", "mutagenicity"}, "all"
    )
    path, digest = MODULE.write_authorized_job_dag(state, values)
    assert path.is_file()
    assert len(digest) == 64
    by_id = {stage.stage_id: stage for stage in values}
    with pytest.raises(RuntimeError, match="smoke engineering Gate"):
        MODULE.assert_smoke_engineering_gate(state, "aids")
    with pytest.raises(RuntimeError, match="smoke engineering Gate"):
        MODULE.assert_smoke_engineering_gate(state, "mutagenicity")
    state.data["completed_stages"] = [
        "aids_project_smoke_gate",
        "mut_chemrepair_smoke_gate",
    ]
    MODULE.assert_smoke_engineering_gate(state, "aids")
    MODULE.assert_smoke_engineering_gate(state, "mutagenicity")
    MODULE.validate_stage_in_authorized_dag(
        by_id["aids_project_full_generation"], state
    )
    MODULE.validate_stage_in_authorized_dag(by_id["mut_full_generation"], state)


def test_end_to_end_dependencies_are_afterok_and_no_rank_backfill() -> None:
    values = MODULE.stages(
        "comrecgc_end_to_end_test", {"aids", "mutagenicity"}, "all"
    )
    assert all(
        stage.dependency_type in {None, "afterok"}
        for stage in values
    )
    authorization = MODULE.end_to_end_authorization_template("a" * 40)
    assert authorization["rank_backfill_allowed"] is False
    assert authorization["rf_guided_repair_allowed"] is False
    assert authorization["wnode_guided_repair_allowed"] is False


def test_frozen_blocker_artifacts_are_adopted_idempotently(tmp_path: Path) -> None:
    state = MODULE.RecoveryState(
        tmp_path,
        "run3",
        requested_mode="all",
        datasets=["aids", "mutagenicity"],
    )
    resolution = {
        "selected": {
            "aids_native": {
                "counterfactuals_path": "/aids/counterfactuals.pt",
                "counterfactuals_sha256": "a" * 64,
                "counterfactuals_bytes": 340685,
                "evidence_path": "/aids/failure.json",
            },
            "mutagenicity_generation": {
                "counterfactuals_path": "/mut/counterfactuals.pt",
                "counterfactuals_sha256": "b" * 64,
                "counterfactuals_bytes": 953049,
                "manifest_path": "/mut/run_manifest.json",
            },
            "mutagenicity_common_recourse": {
                "common_recourse_dir": "/mut/common_recourse",
                "selected_common_recourses_sha256": "c" * 64,
                "representative_counterfactuals_sha256": "d" * 64,
                "manifest_path": "/mut/common_recourse/run_manifest.json",
            },
        }
    }
    first = MODULE.adopt_resolved_artifacts(state, resolution)
    second = MODULE.adopt_resolved_artifacts(state, resolution)
    assert second == first
    assert [row["status"] for row in first] == ["ADOPT_EXISTING"] * 3
    assert all(row["algorithm_rerun"] is False for row in first)
    assert state.data["jobs"] == []
    events = state.events_path.read_text(encoding="utf-8")
    assert events.count('"event": "STAGES_ADOPTED"') == 1
