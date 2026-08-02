from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.baselines.gcfexplainer.run_mutagenicity_vrrw import (
    _resolve_profile_defaults,
    build_parser,
)


ROOT = Path(__file__).resolve().parents[1]
SLURM_ROOT = ROOT / "scripts/slurm/gcfexplainer"
WRAPPERS = (
    SLURM_ROOT / "prepare_mutagenicity_dataset.sh",
    SLURM_ROOT / "train_mutagenicity_gnn.sh",
    SLURM_ROOT / "reproduce_mutagenicity_vrrw.sh",
    SLURM_ROOT / "reproduce_mutagenicity_summary.sh",
    SLURM_ROOT / "reproduce_mutagenicity_all.sh",
)


@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_wrapper_bash_syntax_and_resources(wrapper: Path) -> None:
    result = subprocess.run(["bash", "-n", str(wrapper)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    text = wrapper.read_text(encoding="utf-8")
    assert "#SBATCH --partition=A800" in text
    assert "#SBATCH --gres=gpu:a800:1" in text
    match = re.search(r"#SBATCH --cpus-per-task=(\d+)", text)
    assert match and int(match.group(1)) <= 7
    assert "unset http_proxy" not in text
    assert "unset https_proxy" not in text
    assert "unset all_proxy" not in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"' in text
    assert "outputs/hpc/mutagenicity/baselines/gcfexplainer" in text or "RUN_ROOT" in text


def test_aids_wrappers_remain_dataset_specific_and_unchanged_in_contract() -> None:
    aids_train = (SLURM_ROOT / "train_aids_gnn.sh").read_text(encoding="utf-8")
    aids_vrrw = (SLURM_ROOT / "reproduce_aids_vrrw.sh").read_text(encoding="utf-8")
    aids_summary = (SLURM_ROOT / "reproduce_aids_summary.sh").read_text(encoding="utf-8")
    assert "gnn.py --dataset aids" in aids_train
    assert "python vrrw.py --dataset aids" in aids_vrrw
    assert "python summary.py --dataset aids" in aids_summary


def test_dataset_wrapper_uses_only_strict_train_and_val_files() -> None:
    text = WRAPPERS[0].read_text(encoding="utf-8")
    assert "train_source_label1_teacher_correct.csv" in text
    assert "train_target_label0_teacher_correct.csv" in text
    assert "val_source_label1_teacher_correct.csv" in text
    assert "val_target_label0_teacher_correct.csv" in text
    assert "calibration_source_label1_teacher_correct.csv" not in text
    assert "test_source_label1_teacher_correct.csv" not in text


def test_gnn_wrapper_profiles_and_checkpoint_isolation() -> None:
    text = WRAPPERS[1].read_text(encoding="utf-8")
    assert 'PROFILE="${PROFILE:-smoke}"' in text
    assert 'EPOCHS="${EPOCHS:-5}"' in text
    assert 'EPOCHS="${EPOCHS:-1000}"' in text
    assert "ALLOW_FULL" in text
    assert "$RUN_ROOT/gnn" in text
    assert "data/aids/gnn" not in text


def test_vrrw_wrapper_has_preregistered_mutagenicity_parameters() -> None:
    text = WRAPPERS[2].read_text(encoding="utf-8")
    assert 'VRRW_M="${VRRW_M:-500}"' in text
    assert 'VRRW_M="${VRRW_M:-50000}"' in text
    assert 'VRRW_ALPHA="${VRRW_ALPHA:-1.0}"' in text
    assert 'VRRW_THETA="${VRRW_THETA:-0.05}"' in text
    assert 'VRRW_SEED="${VRRW_SEED:-13}"' in text
    assert 'VRRW_PARENT_LIMIT="${VRRW_PARENT_LIMIT:-64}"' in text
    assert 'VRRW_PARENT_LIMIT="${VRRW_PARENT_LIMIT:-1448}"' in text
    assert '--parent-limit "$VRRW_PARENT_LIMIT"' in text
    assert '--m "$VRRW_M"' in text
    assert "[MUTAGENICITY_GCFEXPLAINER_VRRW_CONFIG]" in text
    assert 'echo "parent_limit=$VRRW_PARENT_LIMIT"' in text
    assert 'echo "M=$VRRW_M"' in text
    assert "--no-sample" in text
    assert "MAX_STEPS" not in text


def test_all_wrapper_explicitly_passes_profile_derived_vrrw_contract() -> None:
    text = WRAPPERS[4].read_text(encoding="utf-8")
    assert 'VRRW_PARENT_LIMIT="${VRRW_PARENT_LIMIT:-64}"' in text
    assert 'VRRW_M="${VRRW_M:-500}"' in text
    assert 'VRRW_PARENT_LIMIT="${VRRW_PARENT_LIMIT:-1448}"' in text
    assert 'VRRW_M="${VRRW_M:-50000}"' in text
    assert 'VRRW_ALPHA="${VRRW_ALPHA:-1.0}"' in text
    assert 'VRRW_THETA="${VRRW_THETA:-0.05}"' in text
    assert 'VRRW_SEED="${VRRW_SEED:-13}"' in text
    assert 'VRRW_PARENT_LIMIT="$VRRW_PARENT_LIMIT"' in text
    assert 'VRRW_M="$VRRW_M"' in text
    assert 'VRRW_ALPHA="$VRRW_ALPHA"' in text
    assert 'VRRW_THETA="$VRRW_THETA"' in text
    assert 'VRRW_SEED="$VRRW_SEED"' in text
    assert 'PARENT_LIMIT="$VRRW_PARENT_LIMIT"' in text
    assert "MAX_STEPS" not in text


def _parse_vrrw(*extra: str):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dataset-dir",
            "dataset",
            "--official-root",
            "official",
            "--gnn-checkpoint",
            "model_best.pth",
            "--neurosed-checkpoint",
            "best_model.pt",
            "--output-dir",
            "out",
            *extra,
        ]
    )
    _resolve_profile_defaults(args)
    return args


def test_vrrw_cli_defaults_are_profile_derived() -> None:
    smoke = _parse_vrrw()
    assert (smoke.profile, smoke.parent_limit, smoke.m) == ("smoke", 64, 500)
    assert (smoke.alpha, smoke.theta, smoke.seed) == (1.0, 0.05, 13)
    full = _parse_vrrw("--profile", "full")
    assert (full.parent_limit, full.m) == (1448, 50000)
    assert (full.alpha, full.theta, full.seed) == (1.0, 0.05, 13)


def test_shell_cli_and_runtime_use_one_to_one_vrrw_names() -> None:
    wrapper = WRAPPERS[2].read_text(encoding="utf-8")
    cli = (
        ROOT / "scripts/baselines/gcfexplainer/run_mutagenicity_vrrw.py"
    ).read_text(encoding="utf-8")
    runtime = (
        ROOT / "src/baselines/gcfexplainer_mutagenicity_runtime.py"
    ).read_text(encoding="utf-8")
    assert '--profile "$PROFILE"' in wrapper
    assert '--parent-limit "$VRRW_PARENT_LIMIT"' in wrapper
    assert '--m "$VRRW_M"' in wrapper
    assert 'parser.add_argument("--m", type=int)' in cli
    assert "m=args.m" in cli
    assert "m: int," in runtime
    assert "max_steps=int(m)" in runtime
    assert 'parser.add_argument("--max-steps"' not in cli


def test_invalid_vrrw_config_logs_actual_values_and_writes_failure_artifacts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "vrrw"
    command = [
        sys.executable,
        str(ROOT / "scripts/baselines/gcfexplainer/run_mutagenicity_vrrw.py"),
        "--dataset-dir",
        str(tmp_path / "dataset"),
        "--official-root",
        str(tmp_path / "official"),
        "--gnn-checkpoint",
        str(tmp_path / "model_best.pth"),
        "--neurosed-checkpoint",
        str(tmp_path / "best_model.pt"),
        "--output-dir",
        str(output_dir),
        "--profile",
        "smoke",
        "--parent-limit",
        "1448",
        "--m",
        "50000",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "[MUTAGENICITY_GCFEXPLAINER_VRRW_CONFIG]" in result.stdout
    assert "parent_limit=1448" in result.stdout
    assert "M=50000" in result.stdout
    assert "[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR]" in result.stderr
    assert "expected_parent_limit=64" in result.stderr
    assert "expected_M=500_or_1000" in result.stderr
    payload = json.loads(
        (output_dir / "failure_summary.json").read_text(encoding="utf-8")
    )
    assert payload["stage"] == "vrrw_config"
    assert payload["profile"] == "smoke"
    assert payload["parent_limit"] == 1448
    assert payload["M"] == 50000
    assert payload["expected_parent_limit"] == 64
    assert payload["expected_M"] == [500, 1000]
    assert payload["model_training_performed"] is False
    assert payload["calibration_loaded"] is False
    assert payload["test_loaded"] is False
    assert (output_dir / "_RUN_FAILED.json").is_file()


def test_all_resume_reuses_completed_gnn_before_vrrw() -> None:
    all_text = WRAPPERS[4].read_text(encoding="utf-8")
    gnn_text = WRAPPERS[1].read_text(encoding="utf-8")
    assert 'RESUME="${RESUME:-true}"' in all_text
    assert 'RESUME="$RESUME"' in all_text
    assert "train_mutagenicity_gnn.sh" in all_text
    assert "[MUTAGENICITY_GCFEXPLAINER_GNN_REUSED]" in gnn_text
    assert all_text.index("train_mutagenicity_gnn.sh") < all_text.index(
        "reproduce_mutagenicity_vrrw.sh"
    )


def test_vrrw_stage_never_trains_models_or_reads_held_out_splits() -> None:
    wrapper = WRAPPERS[2].read_text(encoding="utf-8")
    cli = (
        ROOT / "scripts/baselines/gcfexplainer/run_mutagenicity_vrrw.py"
    ).read_text(encoding="utf-8")
    assert "train_mutagenicity_gnn" not in wrapper
    assert "gnn.py" not in wrapper
    assert "calibration_source_label" not in wrapper
    assert "calibration_target_label" not in wrapper
    assert "test_source_label" not in wrapper
    assert "test_target_label" not in wrapper
    assert "--forbid-calibration-test" in wrapper
    assert '"model_training_performed": False' in cli


def test_mutagenicity_wrappers_do_not_unset_any_proxy_variable() -> None:
    forbidden = (
        "unset http_proxy",
        "unset https_proxy",
        "unset HTTP_PROXY",
        "unset HTTPS_PROXY",
        "unset all_proxy",
        "unset ALL_PROXY",
    )
    for wrapper in WRAPPERS:
        text = wrapper.read_text(encoding="utf-8")
        assert all(command not in text for command in forbidden)


def test_summary_wrapper_preserves_native_rank_then_rf_filters() -> None:
    text = WRAPPERS[3].read_text(encoding="utf-8")
    summary_position = text.index("run_mutagenicity_summary.py")
    export_position = text.index("export_mutagenicity_fullgraph_candidates.py")
    audit_position = text.index("audit_mutagenicity_run.py")
    assert summary_position < export_position < audit_position
    assert 'MINIMUM_NATIVE_EXPORT="${MINIMUM_NATIVE_EXPORT:-100}"' in text
    assert 'TOP_K="${TOP_K:-20}"' in text
    assert "mutagenicity_rf_v1/mutagenicity_rf_model.pkl" in text


def test_all_wrapper_defaults_to_smoke_and_requires_full_opt_in() -> None:
    text = WRAPPERS[4].read_text(encoding="utf-8")
    assert 'PROFILE="${PROFILE:-smoke}"' in text
    assert '"${ALLOW_FULL:-false}" == "true"' in text
    assert "full requires ALLOW_FULL=true" in text
    assert "prepare_mutagenicity_dataset.sh" in text
    assert "train_mutagenicity_gnn.sh" in text
    assert "reproduce_mutagenicity_vrrw.sh" in text
    assert "reproduce_mutagenicity_summary.sh" in text
    assert "[MUTAGENICITY_GCFEXPLAINER_SMOKE_OK]" in text


@pytest.mark.parametrize(
    "script",
    (
        ROOT / "scripts/baselines/gcfexplainer/prepare_mutagenicity_dataset.py",
        ROOT / "scripts/baselines/gcfexplainer/probe_mutagenicity_codec.py",
        ROOT / "scripts/baselines/gcfexplainer/train_mutagenicity_gnn.py",
        ROOT / "scripts/baselines/gcfexplainer/run_mutagenicity_vrrw.py",
        ROOT / "scripts/baselines/gcfexplainer/run_mutagenicity_summary.py",
        ROOT / "scripts/baselines/gcfexplainer/export_mutagenicity_fullgraph_candidates.py",
        ROOT / "scripts/baselines/gcfexplainer/audit_mutagenicity_run.py",
    ),
)
def test_python_entrypoints_forbid_calibration_test(script: Path) -> None:
    text = script.read_text(encoding="utf-8")
    assert "--forbid-calibration-test" in text


def test_runtime_reuses_official_algorithm_entrypoints() -> None:
    text = (ROOT / "src/baselines/gcfexplainer_mutagenicity_runtime.py").read_text(encoding="utf-8")
    assert "modules[\"gnn\"].GNN" in text
    assert "vrrw.counterfactual_summary_with_randomwalk" in text
    assert "greedy_counterfactual_summary_from_covering_sets" in text
    assert "native_rank_reordered\": False" in text


def test_mutagenicity_runtime_scopes_alpha_endpoint_patch() -> None:
    runtime = (
        ROOT / "src/baselines/gcfexplainer_mutagenicity_runtime.py"
    ).read_text(encoding="utf-8")
    official = (
        ROOT / "baselines/gcfexplainer_official/vrrw.py"
    ).read_text(encoding="utf-8")
    assert "vrrw_alpha_endpoint_none_safe_v1" in runtime
    assert "official_compatibility_patches" in runtime
    assert "alpha_endpoint_branch" in runtime
    assert "[GCFEXPLAINER_OFFICIAL_COMPAT_PATCH]" in runtime
    assert "with _official_vrrw_alpha_endpoint_patch(vrrw):" in runtime
    assert "vrrw.calculate_importance = original_calculate_importance" in runtime
    assert "vrrw_alpha_endpoint_none_safe_v1" not in official
