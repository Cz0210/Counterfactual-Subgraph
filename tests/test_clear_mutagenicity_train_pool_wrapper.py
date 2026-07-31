from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts/baselines/clear/build_mutagenicity_train_pool.py"
AUDIT = ROOT / "scripts/baselines/clear/audit_mutagenicity_train_pool.py"
WRAPPER = ROOT / "scripts/slurm/clear_mutagenicity_train_pool.sh"
REPLAY_WRAPPER = (
    ROOT / "scripts/slurm/clear_mutagenicity_generation_replay.sh"
)
APPLY = ROOT / "scripts/baselines/clear/apply_clear_patches.sh"
PATCH_ROOT = ROOT / "patches/clear_official"
PATCH_006 = (
    PATCH_ROOT / "006_mutagenicity_phase_b_run_local_checkpoints.patch"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_phase_b_cli_help() -> None:
    for script in (BUILD, AUDIT):
        result = subprocess.run(
            ["python3", str(script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "--forbid-calibration-test" in result.stdout
    build_help = subprocess.run(
        ["python3", str(BUILD), "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for option in (
        "--phase-a-root",
        "--generation-csv",
        "--teacher-path",
        "--official-root",
        "--parent-limit",
        "--graphpred-epochs",
        "--cfe-epochs",
        "--generation-chunk-size",
        "--generation-only",
        "--graphpred-checkpoint",
        "--graphcfe-checkpoint",
        "--source-run-root",
        "--resume",
    ):
        assert option in build_help


def test_wrapper_is_the_fixed_64_parent_smoke() -> None:
    text = _text(WRAPPER)
    assert 'PARENT_LIMIT="${PARENT_LIMIT:-64}"' in text
    assert 'GRAPHPRED_EPOCHS="${GRAPHPRED_EPOCHS:-5}"' in text
    assert 'CFE_EPOCHS="${CFE_EPOCHS:-5}"' in text
    assert 'GENERATION_CHUNK_SIZE="${GENERATION_CHUNK_SIZE:-16}"' in text
    assert 'BATCH_SIZE="${BATCH_SIZE:-8}"' in text
    assert 'NUM_WORKERS="${NUM_WORKERS:-0}"' in text
    assert 'SEED="${SEED:-13}"' in text
    assert 'if [[ "$PARENT_LIMIT" -ne 64 ]]' in text
    assert "full is forbidden" in text
    assert "parent-limit 1448" not in text


def test_wrapper_resources_and_project_root_contract() -> None:
    text = _text(WRAPPER)
    for required in (
        "#SBATCH --partition=A800",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --gres=gpu:a800:1",
        "#SBATCH --mem=64G",
        "#SBATCH --time=04:00:00",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
        'PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"',
        'export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"',
    ):
        assert required in text
    assert "/share/home" not in text
    assert "unset http_proxy" not in text
    assert "unset https_proxy" not in text
    assert "unset HTTP_PROXY" not in text
    assert "unset HTTPS_PROXY" not in text
    assert "scripts/exp_sbatch.sh" in text


def test_wrapper_nounset_safe_conda_activation() -> None:
    text = _text(WRAPPER)
    disable = text.index("set +u")
    source = text.index("source ~/.bashrc")
    activate = text.index("conda activate smiles_pip118")
    enable = text.index("set -u", activate)
    assert disable < source < activate < enable
    assert "set -eo pipefail" in text


def test_wrapper_uses_phase_a_v2_train_only_inputs_and_rf_teacher() -> None:
    text = _text(WRAPPER)
    assert "clear_phase_a_dataset_codec_best" in text
    assert "train_source_label1_teacher_correct.csv" in text
    assert "mutagenicity_rf_model.pkl" in text
    assert "--forbid-calibration-test" in text
    assert "--expected-model-train-rows 2885" in text
    assert "--expected-model-val-rows 355" in text
    assert "--expected-generation-parent-rows 1448" in text
    assert "--expected-selected-parents 64" in text
    for forbidden in (
        "calibration_source_label1_teacher_correct.csv",
        "test_source_label1_teacher_correct.csv",
        "aids_rf",
        "ogbg_molhiv",
    ):
        assert forbidden not in text


def test_wrapper_runs_build_audit_and_all_success_gates() -> None:
    text = _text(WRAPPER)
    assert "build_mutagenicity_train_pool.py" in text
    assert "audit_mutagenicity_train_pool.py" in text
    assert 'test -s "$OUTPUT_DIR/train_pool_audit.json"' in text
    assert 'test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"' in text
    assert "[MUTAGENICITY_CLEAR_TRAIN_POOL_SMOKE_OK]" in text
    build_text = _text(BUILD)
    for marker in (
        "[MUTAGENICITY_CLEAR_GRAPHPRED_SMOKE_OK]",
        "[MUTAGENICITY_CLEAR_GRAPHCFE_SMOKE_OK]",
        "[MUTAGENICITY_CLEAR_GENERATION_SMOKE_OK]",
    ):
        assert marker in build_text
    assert "[MUTAGENICITY_CLEAR_TRAIN_POOL_AUDIT_OK]" in _text(AUDIT)


def test_patch_006_is_run_local_and_preserves_official_objective() -> None:
    text = _text(PATCH_006)
    assert "CLEAR_WRAPPER_MUTAGENICITY_PHASE_B_RUNTIME" in text
    assert "--model_dir" in text
    assert "--num_experiments" in text
    assert "os.path.abspath(args.model_dir)" in text
    assert "weights_graphPred__" in text
    assert "weights_graphCFE_" in text
    for forbidden in (
        "decoder_edge",
        "bond_type_decoder",
        "loss_cfe =",
        "loss_sim =",
        "calibration",
        "test_source",
    ):
        assert forbidden not in text
    apply_text = _text(APPLY)
    assert PATCH_006.name in apply_text
    assert "CLEAR_WRAPPER_MUTAGENICITY_PHASE_B_RUNTIME" in apply_text


def test_patches_001_through_006_apply_in_order(tmp_path: Path) -> None:
    official = ROOT / "baselines/clear_official"
    checkout = tmp_path / "clear_official"
    real_before = subprocess.run(
        ["git", "-C", str(official), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    subprocess.run(
        ["git", "clone", "--shared", str(official), str(checkout)],
        check=True,
        capture_output=True,
        text=True,
    )
    names = (
        "001_save_cfe_checkpoints.patch",
        "002_export_test_counterfactuals.patch",
        "003_support_aids_dataset.patch",
        "004_aids_weighted_graphpred.patch",
        "005_support_mutagenicity_dataset.patch",
        PATCH_006.name,
    )
    for name in names:
        patch = PATCH_ROOT / name
        subprocess.run(
            ["git", "-C", str(checkout), "apply", "--check", str(patch)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "apply", str(patch)],
            check=True,
        )
    assert "CLEAR_WRAPPER_MUTAGENICITY_PHASE_B_RUNTIME" in _text(
        checkout / "src/main.py"
    )
    assert "CLEAR_WRAPPER_MUTAGENICITY_PHASE_B_RUNTIME" in _text(
        checkout / "src/train_pred.py"
    )
    real_after = subprocess.run(
        ["git", "-C", str(official), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert real_after == real_before


def test_wrapper_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_generation_replay_wrapper_uses_explicit_2021625_artifacts() -> None:
    text = _text(REPLAY_WRAPPER)
    for required in (
        ': "${SOURCE_RUN_ROOT:?',
        ': "${GRAPHPRED_CHECKPOINT:?',
        ': "${GRAPHCFE_CHECKPOINT:?',
        ': "${OUTPUT_DIR:?',
        "--generation-only",
        '--source-run-root "$SOURCE_RUN_ROOT"',
        '--graphpred-checkpoint "$GRAPHPRED_CHECKPOINT"',
        '--graphcfe-checkpoint "$GRAPHCFE_CHECKPOINT"',
        'PARENT_LIMIT="${PARENT_LIMIT:-64}"',
        'GENERATION_CHUNK_SIZE="${GENERATION_CHUNK_SIZE:-16}"',
        'SEED="${SEED:-13}"',
        "[MUTAGENICITY_CLEAR_GENERATION_REPLAY_OK]",
    ):
        assert required in text
    assert "graphpred-epochs" not in text
    assert "cfe-epochs" not in text
    assert 'if [[ "$OUTPUT_DIR" == "$SOURCE_RUN_ROOT" ]]' in text
    assert "scripts/exp_sbatch.sh" in text
    build_text = _text(BUILD)
    for manifest_field in (
        '"source_failed_run_root"',
        '"graphpred_checkpoint_path"',
        '"graphpred_checkpoint_sha256"',
        '"graphcfe_checkpoint_path"',
        '"graphcfe_checkpoint_sha256"',
        '"generation_parent_ids"',
        '"model_training_performed"',
        '"codec_version"',
    ):
        assert manifest_field in build_text


def test_generation_replay_wrapper_is_train_only_and_nounset_safe() -> None:
    text = _text(REPLAY_WRAPPER)
    assert "train_source_label1_teacher_correct.csv" in text
    assert "mutagenicity_rf_model.pkl" in text
    assert "--forbid-calibration-test" in text
    assert "calibration_source_label1_teacher_correct.csv" not in text
    assert "test_source_label1_teacher_correct.csv" not in text
    disable = text.index("set +u")
    source = text.index("source ~/.bashrc")
    activate = text.index("conda activate smiles_pip118")
    enable = text.index("set -u", activate)
    assert disable < source < activate < enable
    assert "/share/home" not in text
    for proxy in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
    ):
        assert f"unset {proxy}" not in text


def test_generation_replay_wrapper_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(REPLAY_WRAPPER)], check=True)
