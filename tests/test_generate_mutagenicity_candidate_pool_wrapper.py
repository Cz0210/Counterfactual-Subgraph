from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (
    REPO_ROOT
    / "scripts"
    / "slurm"
    / "generate_mutagenicity_candidate_pool.sh"
)


def wrapper_text() -> str:
    return WRAPPER.read_text(encoding="utf-8")


def test_wrapper_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_wrapper_uses_portable_project_root_and_requested_resources() -> None:
    text = wrapper_text()

    assert 'PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"' in text
    assert "git -C \"$PWD\" rev-parse --show-toplevel" in text
    assert "BASH_SOURCE" not in text
    assert "/share/home" not in text
    for directive in (
        "#SBATCH --partition=A800",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --cpus-per-task=7",
        "#SBATCH --gres=gpu:a800:1",
        "#SBATCH --mem=64G",
        "#SBATCH --time=48:00:00",
        "#SBATCH --job-name=mut_ppo_pool",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
    ):
        assert directive in text


def test_wrapper_uses_only_mutagenicity_fresh_v2_inputs() -> None:
    text = wrapper_text()
    lowered = text.lower()

    assert "mutagenicity_ppo_prompts_train_label1_v2.csv" in text
    assert "sft_fresh_strict_v2_best" in text
    assert "ppo_fresh_strict_v2_best" in text
    assert "mutagenicity_rf_v1/mutagenicity_rf_model.pkl" in text
    assert "aids_rf_model" not in lowered
    assert "hiv.csv" not in lowered
    assert "sft_v3_hiv_runs" not in lowered
    dataset_default = next(
        line for line in text.splitlines() if line.startswith("DATASET_PATH=")
    )
    assert "calibration" not in dataset_default.lower()
    assert "test" not in dataset_default.lower()


def test_wrapper_requires_new_explicit_output_directory() -> None:
    text = wrapper_text()

    assert 'if [[ -z "${OUTPUT_DIR:-}" ]]' in text
    assert "OUTPUT_DIR must be explicitly set by the caller" in text
    assert "OUTPUT_DIR already exists and is non-empty" in text
    assert 'find "$OUTPUT_DIR" -mindepth 1 -print -quit' in text
    assert "FORCE_REGEN" not in text


def test_generation_command_has_mutagenicity_schema_and_reward_semantics() -> None:
    text = wrapper_text()
    generation = text.split(
        "python scripts/generate_full_candidate_pool.py", maxsplit=1
    )[1].split("python scripts/audit_candidate_pool.py", maxsplit=1)[0]

    for expected in (
        "--config configs/hpc.yaml",
        "--set inference.fallback_to_heuristic=false",
        '--dataset-path "$DATASET_PATH"',
        '--base-model-path "$BASE_MODEL_PATH"',
        '--sft-lora-path "$SFT_LORA_PATH"',
        '--ppo-checkpoint-path "$PPO_CHECKPOINT_PATH"',
        '--teacher-path "$TEACHER_PATH"',
        "--label-col label",
        "--smiles-col parent_smiles",
        "--target-label 1",
        '--limit "$LIMIT"',
        "--enable-parent-projection",
        "--enable-projected-cf-reward",
        "--enable-substructure-distance-reward",
        "--substructure-distance-reward-weight 0.1923076923076923",
        "--projection-penalty 0.25",
        "--enable-minimal-syntax-repair",
        "--enable-component-salvage",
    ):
        assert expected in generation


def test_wrapper_runs_base_artifact_and_conditional_full_audits() -> None:
    text = wrapper_text()

    assert "python scripts/audit_candidate_pool.py" in text
    assert "python scripts/export_candidate_pool_audit_artifacts.py" in text
    assert "python scripts/audit_full_candidate_pool.py" in text
    assert 'if [[ "$LIMIT" -eq 0 ]]' in text
    assert 'elif [[ "$RUN_FULL_AUDIT_NORMALIZED"' in text
    assert '--coverage-parent-limit 0' in text
    assert 'FULL_AUDIT_DIR="$OUTPUT_DIR/full_audit"' in text


def test_wrapper_has_limit_row_parent_and_adapter_path_audits() -> None:
    text = wrapper_text()

    for expected in (
        "EXPECTED_PARENTS=1448",
        "EXPECTED_ROWS=$((EXPECTED_PARENTS * NUM_RETURN_SEQUENCES))",
        "Unique-parent mismatch",
        "Candidate-row mismatch",
        "parent_identifiers",
        "required_fields",
        '"parent_smiles"',
        '"final_fragment"',
        '"cf_flip"',
        '"parse_ok"',
        '"final_substructure"',
        '"ppo_checkpoint_path"',
        "expected_ppo_path",
        "PPO adapter-path mismatch",
        "input_file_sha256",
        "run_manifest.json",
    ):
        assert expected in text


def test_wrapper_rejects_calibration_test_inputs_and_marks_success() -> None:
    text = wrapper_text()

    assert "forbidden_input_patterns" in text
    assert "calibration_test_input_path_found" in text
    assert (
        're.search(r"(^|[/_.-])(calibration|test)([/_.-]|$)"' in text
    )
    assert "[MUTAGENICITY_CANDIDATE_POOL_GENERATION_OK]" in text
