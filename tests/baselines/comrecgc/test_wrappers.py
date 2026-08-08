from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
WRAPPERS = sorted((ROOT / "scripts/slurm").glob("comrecgc_*.sh"))


@pytest.mark.parametrize("path", WRAPPERS, ids=lambda path: path.name)
def test_wrappers_use_safe_single_gpu_resources(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --partition=A800" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks-per-node=1" in text
    assert "#SBATCH --gres=gpu:a800:1" in text
    match = re.search(r"#SBATCH --cpus-per-task=(\d+)", text)
    assert match and int(match.group(1)) <= 7
    assert "unset http_proxy" not in text
    assert "unset https_proxy" not in text
    assert "export PYTHONPATH=" in text


def test_full_profile_is_not_an_arbitrary_budget() -> None:
    text = (ROOT / "scripts/slurm/comrecgc_project_generate.sh").read_text(encoding="utf-8")
    assert "EXPECTED_PARENT_LIMIT=1283" in text
    assert "EXPECTED_PARENT_LIMIT=1448" in text
    assert "PARENT_LIMIT=\"${PARENT_LIMIT:-$EXPECTED_PARENT_LIMIT}\"" in text


def test_all_submissions_route_through_registry() -> None:
    text = (ROOT / "scripts/automation/run_comrecgc_baseline.py").read_text(encoding="utf-8")
    assert '"scripts/exp_sbatch.sh"' in text
    assert "subprocess.run([\"sbatch\"" not in text


def test_registered_submission_precreates_relative_slurm_log_directory() -> None:
    wrapper = (ROOT / "scripts/exp_sbatch.sh").read_text(encoding="utf-8")
    mkdir_index = wrapper.index('mkdir -p "$PROJECT_ROOT/logs"')
    submit_index = wrapper.index('python scripts/exp_sbatch.py "$@"')
    assert mkdir_index < submit_index
    assert '[[ ! -d "$PROJECT_ROOT/logs" || ! -w "$PROJECT_ROOT/logs" ]]' in wrapper


def test_export_resume_is_explicit_and_defaults_off() -> None:
    text = (ROOT / "scripts/slurm/comrecgc_export.sh").read_text(encoding="utf-8")
    assert 'RESUME="${RESUME:-false}"' in text
    assert '[[ "$RESUME" == "true" || "$RESUME" == "false" ]]' in text
    assert 'RESUME_ARGS=(--resume)' in text
    assert '"${RESUME_ARGS[@]}"' in text


def test_unified_eval_supports_explicit_resume_and_smoke_audit_input() -> None:
    text = (ROOT / "scripts/slurm/comrecgc_unified_eval.sh").read_text(encoding="utf-8")
    assert 'RESUME="${RESUME:-false}"' in text
    assert "--candidate-filter-audit" in text
    assert '"${RESUME_ARGS[@]}"' in text


def test_official_checkout_is_not_vendored() -> None:
    assert not (ROOT / "baselines/comrecgc_official").exists()
    assert "external/COMRECGC/" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_native_smoke_requires_common_recourse_serialization() -> None:
    text = (ROOT / "scripts/slurm/comrecgc_native_smoke.sh").read_text(encoding="utf-8")
    assert 'NATIVE_PARENT_LIMIT="${NATIVE_PARENT_LIMIT:-64}"' in text
    assert '[[ "$NATIVE_PARENT_LIMIT" == "64" ]]' in text
    assert 'native_common_recourse.json' in text
    assert 'native_representative_counterfactuals.pt' in text


def test_aids_existing_audit_loads_trusted_upstream_pyg_cache() -> None:
    text = (ROOT / "scripts/slurm/comrecgc_aids_existing_audit.sh").read_text(
        encoding="utf-8"
    )
    assert "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1" not in text
    assert text.count("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1") == 1
    assert "audit_trusted_aids_cache.py" in text
    scoped_load = (
        "env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \\\n"
        "python scripts/baselines/comrecgc/materialize_trusted_aids_cache.py"
    )
    assert scoped_load in text
    assert "env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \\\npython scripts/baselines/comrecgc/audit_aids_native_dbscan.py" not in text
    assert "--trusted-dataset-payload" in text
    assert "--expected-cache-inventory-sha256" in text
    assert "--expected-inventory-sha256" in text

    density = (ROOT / "scripts/slurm/comrecgc_aids_density_retry.sh").read_text(
        encoding="utf-8"
    )
    assert "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1" not in density
    assert density.count("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1") == 1
    assert "audit_trusted_aids_cache.py" in density
    assert "materialize_trusted_aids_cache.py" in density
    assert "--trusted-dataset-payload" in density

    for name in ("comrecgc_aids_native_full.sh", "comrecgc_native_smoke.sh"):
        wrapper = (ROOT / "scripts/slurm" / name).read_text(encoding="utf-8")
        assert "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1" not in wrapper
        assert wrapper.count("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1") == 1
        assert "materialize_trusted_aids_cache.py" in wrapper
        assert "--trusted-dataset-payload" in wrapper

    generation = (
        ROOT / "scripts/baselines/comrecgc/run_generation.py"
    ).read_text(encoding="utf-8")
    assert 'parser.add_argument("--trusted-dataset-payload")' in generation
    assert 'parser.add_argument("--expected-cache-inventory-sha256")' in generation


def test_mut_chemistry_wrapper_accepts_only_explicit_complete_adoption() -> None:
    text = (ROOT / "scripts/slurm/comrecgc_mut_chemistry_audit.sh").read_text(
        encoding="utf-8"
    )
    assert 'ADOPT_EXISTING="${ADOPT_EXISTING:-false}"' in text
    assert "[COMRECGC_MUT_CHEMISTRY_ADOPT_EXISTING_SUCCESS]" in text
    assert text.count("[COMRECGC_PROJECT_CHEMISTRY_ENGINEERING_PASS]") == 2
    assert "[COMRECGC_MUT_CHEMISTRY_ENGINEERING_SMOKE_PASS]" not in text


def test_mut_trace_adoption_does_not_rerun_random_walk() -> None:
    text = (ROOT / "scripts/slurm/comrecgc_mut_trace_adopt.sh").read_text(
        encoding="utf-8"
    )
    assert "recover_mutagenicity_trace.py" in text
    assert "run_generation.py" not in text
    assert "--source-failed-generation-dir" in text
    assert "algorithm_rerun=false" in text


def test_project_full_wrappers_freeze_scientific_contracts() -> None:
    generation = (ROOT / "scripts/slurm/comrecgc_project_generate.sh").read_text(
        encoding="utf-8"
    )
    assert '[[ "$DATASET" == "aids" ]] && EXPECTED_PARENT_LIMIT=1283' in generation
    assert "EXPECTED_PARENT_LIMIT=1448" in generation
    assert "--trace-output-dir" in generation

    evaluation = (
        ROOT / "scripts/slurm/comrecgc_project_slot_eval.sh"
    ).read_text(encoding="utf-8")
    assert "wnode_figure4_redline_k10_figure4_wnode_coverage_vs_threshold.csv" in evaluation
    assert "--theta-star 0.05 --cost-cap 0.0535" in evaluation
    assert "test_source_label1_teacher_correct.csv" in evaluation
    assert "--max-k 20" in evaluation

    chemistry = (
        ROOT / "scripts/slurm/comrecgc_project_chemistry.sh"
    ).read_text(encoding="utf-8")
    assert "--trace-lineage-path" in chemistry
    assert "--trace-evidence-path" in chemistry
    assert "$TRACE_DIR/trace_summary.json" in chemistry
    assert "recovery_trace_v1" not in chemistry
    assert "--parent-limit \"$PARENT_LIMIT\"" in chemistry


def test_full_generation_jobs_use_qos_supported_seven_day_walltime() -> None:
    for name in (
        "comrecgc_aids_native_full.sh",
        "comrecgc_mut_full.sh",
        "comrecgc_project_generate.sh",
    ):
        text = (ROOT / "scripts/slurm" / name).read_text(encoding="utf-8")
        assert "#SBATCH --time=7-00:00:00" in text


def test_aids_project_full_uses_memory_headroom_and_rejects_fake_resume() -> None:
    text = (ROOT / "scripts/slurm/comrecgc_project_generate.sh").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --mem=192G" in text
    assert '[[ "$RESUME" == "false" ]]' in text
    assert "no proven cross-job RNG/state resume" in text
    assert "--resume" not in text
    assert "--cpus-per-task=7" in text
    assert "--gres=gpu:a800:1" in text


def test_full_runtime_enables_selected_transition_only_action_index() -> None:
    text = (ROOT / "src/baselines/comrecgc/runtime.py").read_text(encoding="utf-8")
    assert 'compact_enumeration=mode == "full"' in text
    assert '"full_selected_transition_weak_action_index_v1"' in text
    assert '"generation_resume_supported": False' in text
    assert 'preserve_active_transitions=mode == "full"' in text
    assert 'compact_transitions=mode == "full"' in text
    assert "transition_expanded_capacity=parameters.heads" in text
    assert '"active_move_transition_eviction_deferred_v1"' in text
    assert '"compact_transition_action_replay_lru_v1"' in (
        ROOT / "src/baselines/comrecgc/transition_cache.py"
    ).read_text(encoding="utf-8")


def test_mut_full_uses_bounded_transition_cache_with_memory_headroom() -> None:
    text = (ROOT / "scripts/slurm/comrecgc_mut_full.sh").read_text(encoding="utf-8")
    assert "#SBATCH --mem=128G" in text
    assert "#SBATCH --cpus-per-task=7" in text
    assert "#SBATCH --gres=gpu:a800:1" in text
    assert "exact_action_replay_with_bounded_expanded_lru_v1" in text
    assert "expanded_capacity=5" in text


def test_project_generation_wrapper_declares_move_scoped_transition_policy() -> None:
    text = (ROOT / "scripts/slurm/comrecgc_project_generate.sh").read_text(
        encoding="utf-8"
    )
    assert "pinned_upstream_active_move_deferred_eviction_v1" in text
    assert "transition_state_policy=$TRANSITION_STATE_POLICY" in text
    assert "#SBATCH --time=7-00:00:00" in text
    assert "#SBATCH --mem=192G" in text


def test_aids_project_smoke_is_read_only_adoption() -> None:
    text = (
        ROOT / "scripts/slurm/comrecgc_aids_project_smoke_adopt.sh"
    ).read_text(encoding="utf-8")
    assert "adopt_aids_project_smoke.py" in text
    assert "run_generation.py" not in text
    assert "algorithm_rerun=false" in text
