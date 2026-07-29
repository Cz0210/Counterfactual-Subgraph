from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (
    REPO_ROOT
    / "scripts"
    / "slurm"
    / "build_mutagenicity_wnode_calibration_matrix.sh"
)


def _text() -> str:
    return WRAPPER.read_text(encoding="utf-8")


def test_wrapper_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_wrapper_has_requested_gpu_resources_and_portable_root() -> None:
    text = _text()

    assert 'PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"' in text
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
        "#SBATCH --job-name=mut_wnode_mat",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
    ):
        assert directive in text


def test_wrapper_requires_all_external_paths() -> None:
    text = _text()

    for variable in (
        "CANDIDATE_POOL",
        "CALIBRATION_CSV",
        "TEACHER_PATH",
        "MOLCLR_ROOT",
        "MOLCLR_CHECKPOINT",
        "OUTPUT_DIR",
        "WNODE_CACHE_DB",
    ):
        assert f': "${{{variable}:?' in text


def test_wrapper_uses_calibration_and_never_loads_a_fixed_test_input() -> None:
    text = _text()

    assert '--calibration-csv "$CALIBRATION_CSV"' in text
    assert "--cohort-name calibration" in text
    assert "test.csv" not in text.lower()
    assert "outputs/hpc" not in text


def test_wrapper_passes_wnode_resume_and_matrix_controls() -> None:
    text = _text()

    for expected in (
        'PARENT_LIMIT="${PARENT_LIMIT:-0}"',
        'CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-0}"',
        'EXPECTED_PARENT_COUNT="${EXPECTED_PARENT_COUNT:-235}"',
        'EXPECTED_CANDIDATE_COUNT="${EXPECTED_CANDIDATE_COUNT:-0}"',
        'CANDIDATE_ORDER="${CANDIDATE_ORDER:-source_support_desc}"',
        'FLUSH_EVERY="${FLUSH_EVERY:-100}"',
        'RESUME="${RESUME:-true}"',
        'WNODE_SIZE_PENALTY_BETA="${WNODE_SIZE_PENALTY_BETA:-0.0}"',
        '--wnode-cache-db "$WNODE_CACHE_DB"',
        '--wnode-size-penalty-beta "$WNODE_SIZE_PENALTY_BETA"',
        '"$RESUME_FLAG"',
    ):
        assert expected in text
    assert "Resume output is missing a valid" in text


def test_wrapper_runs_strict_cartesian_audit_and_marks_completion() -> None:
    text = _text()

    assert "python scripts/audit_mutagenicity_wnode_calibration_matrix.py" in text
    assert "--require-complete-cartesian" in text
    assert "--require-strict-flip-pair" in text
    assert "--forbid-test" in text
    assert '[[ -s "$OUTPUT_DIR/_RUN_COMPLETE.json" ]]' in text
    assert "[MUTAGENICITY_WNODE_CALIBRATION_MATRIX_OK]" in text
