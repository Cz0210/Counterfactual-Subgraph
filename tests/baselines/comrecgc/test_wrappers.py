from __future__ import annotations

import os
import re
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
WRAPPERS = sorted((ROOT / "scripts/slurm").glob("comrecgc_*.sh"))
CPU_ONLY_WRAPPERS = {
    "comrecgc_aids_freeze_recovery_cpu_v7.sh",
    "comrecgc_bace_artifact_gate.sh",
    "comrecgc_bace_generation_integrity.sh",
    "comrecgc_bace_project_chemistry.sh",
    "comrecgc_checkpoint_audit.sh",
    "comrecgc_generation_integrity_gate.sh",
    "comrecgc_mut_freeze.sh",
    "comrecgc_mut_full_chemistry.sh",
    "comrecgc_mut_full_gate.sh",
    "comrecgc_project_chemistry.sh",
    "comrecgc_project_freeze.sh",
    "comrecgc_project_full_gate.sh",
    "comrecgc_storage_preflight_v6.sh",
}


@pytest.mark.parametrize("path", WRAPPERS, ids=lambda path: path.name)
def test_wrappers_never_request_more_than_one_gpu(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert (
        "#SBATCH --partition=A800" in text
        or "#SBATCH --partition=intel" in text
    )
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks-per-node=1" in text
    assert "--gres=gpu:2" not in text
    assert "--gres=gpu:a800:2" not in text
    assert "--gpus=2" not in text
    assert "--gpus-per-node=2" not in text
    if "#SBATCH --gres=" in text:
        assert "#SBATCH --partition=A800" in text
        assert "#SBATCH --gres=gpu:a800:1" in text
    match = re.search(r"#SBATCH --cpus-per-task=(\d+)", text)
    assert match and int(match.group(1)) <= 7
    assert "unset http_proxy" not in text
    assert "unset https_proxy" not in text
    assert "export PYTHONPATH=" in text


@pytest.mark.parametrize("name", sorted(CPU_ONLY_WRAPPERS))
def test_integrity_chemistry_gate_freeze_and_preflight_are_cpu_only(name: str) -> None:
    text = (ROOT / "scripts/slurm" / name).read_text(encoding="utf-8")
    assert "#SBATCH --gres=" not in text


@pytest.mark.parametrize(
    "name",
    [
        "comrecgc_mut_generation_storage_v6.sh",
        "comrecgc_bace_generation_storage_v6.sh",
        "comrecgc_common_recourse.sh",
        "comrecgc_bace_common_recourse.sh",
        "comrecgc_project_slot_eval.sh",
        "comrecgc_bace_slot_eval.sh",
    ],
)
def test_generation_recourse_and_wnode_eval_request_exactly_one_gpu(name: str) -> None:
    text = (ROOT / "scripts/slurm" / name).read_text(encoding="utf-8")
    assert text.count("#SBATCH --gres=gpu:a800:1") == 1


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


def test_common_recourse_wrapper_exposes_external_engine_without_changing_default() -> None:
    text = (ROOT / "scripts/slurm/comrecgc_common_recourse.sh").read_text(
        encoding="utf-8"
    )
    assert 'ENGINE="${ENGINE:-legacy_in_memory}"' in text
    assert "external_memory_exact_v1" in text
    assert "--external-max-rss-gb" in text
    assert "--external-dbscan-shortcut-mode" in text
    assert "--external-shortcut-failure-cap" in text
    assert "--external-summary-block-size" in text
    assert "--external-pair-store-source-manifest" in text
    assert "--external-pair-store-source-owner-root" in text
    assert 'EXTERNAL_PAIR_STORE_AUTO_ROOT="${EXTERNAL_PAIR_STORE_AUTO_ROOT:-}"' in text
    assert "[COMRECGC_PAIR_SOURCE_SELECTED] mode=promoted_final" in text
    assert 'DEVICE="${DEVICE:-cpu}"' in text
    assert "AIDS external engine is CPU-only" in text
    assert "--expected-sklearn-version" in text


def test_autodl_wrapper_forwards_exact_route_and_prefers_promoted_pair_store(
    tmp_path: Path,
) -> None:
    wrapper = ROOT / "scripts/autodl/run_comrecgc_standardized_continuation.sh"
    capture = tmp_path / "argv.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/bash\nset -euo pipefail\nprintf '%s\\n' \"$@\" > \"$CAPTURE\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    pair_root = tmp_path / "old/common_recourse/external_memory/pair_store"
    pair_root.mkdir(parents=True)
    (pair_root / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "fresh-output"
    values = {
        "DATASET": "aids",
        "SOURCE_GENERATION_ROOT": str(tmp_path / "generation"),
        "COMRECGC_UPSTREAM_ROOT": str(tmp_path / "upstream"),
        "DATASET_DIR": str(tmp_path / "dataset"),
        "SOURCE_CSV": str(tmp_path / "source.csv"),
        "DISTANCE_CHECKPOINT": str(tmp_path / "distance.pt"),
        "DATASET_CSV": str(tmp_path / "dataset.csv"),
        "TEACHER_PATH": str(tmp_path / "teacher.pkl"),
        "MOLCLR_ROOT": str(tmp_path / "molclr"),
        "MOLCLR_CHECKPOINT": str(tmp_path / "molclr.pt"),
        "THRESHOLDS_PATH": str(tmp_path / "thresholds.json"),
        "OUTPUT_ROOT": str(output),
        "AUTODL_PYTHON": str(fake_python),
        "CAPTURE": str(capture),
        "DEVICE": "cpu",
        "COMMON_RECOURSE_ENGINE": "external_memory_exact_v1",
        "COMRECGC_EXTERNAL_DBSCAN_SHORTCUT_MODE": (
            "all_core_one_component_adaptive_anchor_v1"
        ),
        "COMRECGC_EXTERNAL_SHORTCUT_SEED_COUNT": "3",
        "COMRECGC_EXTERNAL_SHORTCUT_FAILURE_CAP": "4096",
        "COMRECGC_EXTERNAL_SHORTCUT_QUERY_BLOCK_SIZE": "65536",
        "COMRECGC_EXTERNAL_EXACT_FALLBACK_MAX_SAMPLES": "0",
        "COMRECGC_EXTERNAL_SUMMARY_BLOCK_SIZE": "65536",
        "COMRECGC_EXTERNAL_PAIR_STORE_AUTO_ROOT": str(pair_root),
        "COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT": str(
            pair_root.parents[2]
        ),
        # A closed-chunk fallback is configured, but the promoted terminal
        # must be selected without forwarding cache allocation arguments.
        "COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_CHECKPOINT": str(
            pair_root / "checkpoint.json"
        ),
        "COMRECGC_EXTERNAL_VECTOR_CACHE_ROOT": str(tmp_path / "cache"),
        "COMRECGC_EXTERNAL_VECTOR_CACHE_LOCK": str(tmp_path / "cache.lock"),
        "COMRECGC_EXTERNAL_VECTOR_CACHE_ROUTE_LOCK": str(
            tmp_path / "route.lock"
        ),
    }
    completed = subprocess.run(
        ["bash", str(wrapper)],
        env={**os.environ, **values},
        check=True,
        capture_output=True,
        text=True,
    )
    argv = capture.read_text(encoding="utf-8").splitlines()
    assert "[COMRECGC_PAIR_SOURCE_SELECTED] mode=promoted_final" in completed.stdout
    assert argv[argv.index("--external-pair-store-source-manifest") + 1] == str(
        pair_root / "run_manifest.json"
    )
    assert argv[argv.index("--external-pair-store-source-owner-root") + 1] == str(
        pair_root.parents[2]
    )
    assert argv[argv.index("--external-dbscan-shortcut-mode") + 1] == (
        "all_core_one_component_adaptive_anchor_v1"
    )
    assert argv[argv.index("--external-exact-fallback-max-samples") + 1] == "0"
    assert "--external-pair-store-source-checkpoint" not in argv
    assert "--external-vector-cache-root" not in argv
    assert "--external-vector-cache-lock" not in argv
    assert "--external-vector-cache-route-lock" not in argv

    (pair_root / "run_manifest.json").unlink()
    fallback_capture = tmp_path / "fallback-argv.txt"
    fallback_values = {
        **values,
        "CAPTURE": str(fallback_capture),
        "OUTPUT_ROOT": str(tmp_path / "fresh-output-fallback"),
        "COMRECGC_EXTERNAL_VECTOR_CACHE_MIN_FREE_GB": "3",
        "COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT": str(tmp_path / "proc"),
    }
    fallback = subprocess.run(
        ["bash", str(wrapper)],
        env={**os.environ, **fallback_values},
        check=True,
        capture_output=True,
        text=True,
    )
    fallback_argv = fallback_capture.read_text(encoding="utf-8").splitlines()
    assert "[COMRECGC_PAIR_SOURCE_SELECTED] mode=closed_chunks" in fallback.stdout
    assert fallback_argv[
        fallback_argv.index("--external-pair-store-source-checkpoint") + 1
    ] == str(pair_root / "checkpoint.json")
    assert fallback_argv[
        fallback_argv.index("--external-vector-cache-root") + 1
    ] == str(tmp_path / "cache")
    assert fallback_argv[
        fallback_argv.index("--external-vector-cache-lock") + 1
    ] == str(tmp_path / "cache.lock")
    assert fallback_argv[
        fallback_argv.index("--external-vector-cache-route-lock") + 1
    ] == str(tmp_path / "route.lock")
    assert fallback_argv[
        fallback_argv.index("--external-vector-cache-min-free-gb") + 1
    ] == "3"
    assert fallback_argv[
        fallback_argv.index("--external-vector-cache-proc-root") + 1
    ] == str(tmp_path / "proc")

    (pair_root / "run_manifest.json").touch()
    invalid = subprocess.run(
        ["bash", str(wrapper)],
        env={
            **os.environ,
            **fallback_values,
            "OUTPUT_ROOT": str(tmp_path / "fresh-output-invalid-final"),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert invalid.returncode == 64
    assert "invalid promoted pair-store manifest" in invalid.stderr

    (pair_root / "run_manifest.json").unlink()
    required_final = subprocess.run(
        ["bash", str(wrapper)],
        env={
            **os.environ,
            **fallback_values,
            "OUTPUT_ROOT": str(tmp_path / "fresh-output-required-final"),
            "COMRECGC_EXTERNAL_REQUIRE_PROMOTED_FINAL": "1",
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert required_final.returncode == 75
    assert "required promoted pair-store manifest is absent" in required_final.stderr


def test_aids_exact_v5_supervisor_freezes_cpu_exact_and_storage_contracts() -> None:
    text = (
        ROOT / "scripts/autodl/run_aids_comrecgc_exact_route_v5_supervisor.sh"
    ).read_text(encoding="utf-8")
    required = {
        '"${DATASET:-}" == "aids"',
        '"${DEVICE:-}" == "cpu"',
        '"${GPU_REQUIRED:-}" == "0"',
        '"${COMMON_RECOURSE_ENGINE:-}" == "external_memory_exact_v1"',
        '"${COMRECGC_EXTERNAL_MAX_RSS_GB:-}" == "96"',
        '"${COMRECGC_EXTERNAL_QUERY_BLOCK_SIZE:-}" == "8"',
        '"${COMRECGC_EXTERNAL_CHECKPOINT_INTERVAL_BLOCKS:-}" == "1"',
        '"${COMRECGC_EXTERNAL_SHORTCUT_SEED_COUNT:-}" == "3"',
        '"${COMRECGC_EXTERNAL_SHORTCUT_FAILURE_CAP:-}" == "4096"',
        '"${COMRECGC_EXTERNAL_SHORTCUT_QUERY_BLOCK_SIZE:-}" == "65536"',
        '"${COMRECGC_EXTERNAL_EXACT_FALLBACK_MAX_SAMPLES:-}" == "0"',
        '"${COMRECGC_EXTERNAL_SUMMARY_BLOCK_SIZE:-}" == "65536"',
        '"${COMRECGC_EXTERNAL_VECTOR_CACHE_MIN_FREE_GB:-}" == "3"',
        "all_core_one_component_adaptive_anchor_v1",
        "COMRECGC_EXTERNAL_PAIR_STORE_AUTO_ROOT",
        "COMRECGC_EXTERNAL_REQUIRE_PROMOTED_FINAL",
        "COMRECGC_EXTERNAL_ROUTE_LOCK",
        "COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT",
        "terminal owner must be the exact pair-store root",
        "promoted-final route forbids chunk/cache fallback",
        "route/highmem locks must be distinct",
        "AIDS_COMRECGC_V5_MAX_SAME_ROOT_RESUMES",
        "AIDS_COMRECGC_V5_ALLOWED_OLD_PID",
        "AIDS_COMRECGC_V5_ALLOWED_OLD_START_TICKS",
        "AIDS_COMRECGC_V5_ALLOWED_OLD_CMDLINE_SHA256",
        "AIDS_COMRECGC_V5_ALLOWED_OLD_OUTPUT_ROOT",
        "AIDS_COMRECGC_V5_ALLOWED_OLD_PROJECT_ROOT",
        "verify_aids_comrecgc_v5_process_set.py",
        "common-recourse process set changed",
        "production test hooks are forbidden",
    }
    assert all(value in text for value in required)
    assert text.count("resume_count=$((resume_count + 1))") == 1
    assert "AIDS_COMRECGC_EXACT_ROUTE_V5_SUPERVISOR_PASS" in text
    paired = (
        ROOT / "scripts/slurm/run_aids_comrecgc_exact_route_v5_supervisor.sh"
    ).read_text(encoding="utf-8")
    assert "do not submit" in paired
    assert "exit 78" in paired
    assert "bash scripts/autodl/run_aids_comrecgc_exact_route_v5_supervisor.sh" in paired


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
    assert '"generation_resume_supported": mode == "full"' in text
    assert "run_generation_loop(" in text
    assert "save_generation_checkpoint(" in text
    assert 'preserve_active_transitions=mode == "full"' in text
    assert 'compact_transitions=mode == "full"' in text
    assert "transition_expanded_capacity=parameters.heads" in text
    assert '"active_move_transition_eviction_deferred_v1"' in text


def test_bace_generation_wrappers_require_exact_persistent_checkpoint_args() -> None:
    for name in (
        "comrecgc_bace_generation_storage_v6.sh",
        "comrecgc_bace_project_generate.sh",
    ):
        text = (ROOT / "scripts" / "slurm" / name).read_text(encoding="utf-8")
        assert "CHECKPOINT_MIRROR_ROOT must be an independent persistent path" in text
        assert "--checkpoint-root" in text
        assert "--checkpoint-mirror-root" in text
        assert "--checkpoint-interval-steps 500" in text
        assert "--checkpoint-keep-last 2" in text
        assert "--progress-interval-steps 25" in text
        assert "args+=(--resume)" in text
        assert "export PYTHONHASHSEED=0" in text
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
    assert "authoritative_backing_live_graph_resolution_v2" in text
    assert '--graph-state-dir "$GRAPH_STATE_DIR"' in text
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
