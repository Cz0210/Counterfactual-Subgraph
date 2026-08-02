from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


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
    assert 'MAX_STEPS="${MAX_STEPS:-500}"' in text
    assert 'MAX_STEPS="${MAX_STEPS:-50000}"' in text
    assert 'ALPHA="${ALPHA:-1.0}"' in text
    assert 'SEED="${SEED:-13}"' in text
    assert 'PARENT_LIMIT="${PARENT_LIMIT:-1448}"' in text
    assert "alpha=1.0 and seed=13 are fixed" in text
    assert "AIDS/HIV checkpoint is forbidden" in text
    assert "--no-sample" in text


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
    assert '"${ALLOW_FULL:-false}" != "true"' in text
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

