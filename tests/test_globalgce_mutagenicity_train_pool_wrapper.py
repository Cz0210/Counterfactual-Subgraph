from __future__ import annotations

import subprocess
from pathlib import Path


WRAPPER = Path("scripts/slurm/globalgce_mutagenicity_train_pool.sh")


def _text() -> str:
    return WRAPPER.read_text(encoding="utf-8")


def test_wrapper_has_valid_bash_and_requested_resources() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    text = _text()
    for directive in (
        "#SBATCH --partition=A800",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --cpus-per-task=7",
        "#SBATCH --gres=gpu:a800:1",
        "#SBATCH --mem=128G",
        "#SBATCH --time=24:00:00",
        "#SBATCH --job-name=mut_globalgce",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
    ):
        assert directive in text


def test_wrapper_is_portable_and_requires_external_paths() -> None:
    text = _text()
    assert 'PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"' in text
    assert "BASH_SOURCE" not in text
    assert "/share/home" not in text
    for variable in ("TRAIN_CSV", "TEACHER_PATH", "OFFICIAL_ROOT", "OUTPUT_DIR"):
        assert f': "${{{variable}:?' in text


def test_wrapper_disables_nounset_during_shell_and_conda_initialization() -> None:
    lines = [line.strip() for line in _text().splitlines()]
    source_index = lines.index("source ~/.bashrc")
    conda_index = lines.index("conda activate smiles_pip118")
    restore_index = lines.index("set -u", conda_index + 1)
    assert lines[source_index - 1] == "set +u"
    assert conda_index == source_index + 1
    assert restore_index == conda_index + 1
    assert "set -eo pipefail" in lines
    assert "BASHRCSOURCED" not in _text()


def test_wrapper_is_train_only_and_runs_audit() -> None:
    text = _text()
    assert '--train-csv "$TRAIN_CSV"' in text
    assert '--native-train-csv "$NATIVE_TRAIN_CSV"' in text
    assert "--forbid-calibration-test" in text
    assert "calibration.csv" not in text.lower()
    assert "test.csv" not in text.lower()
    assert "audit_mutagenicity_train_pool.py" in text
    assert "--require-target-label-zero" in text
    assert "--require-unique-universe" in text
    assert "[MUTAGENICITY_GLOBALGCE_TRAIN_POOL_OK]" in text


def test_wrapper_has_deterministic_defaults_and_resume_guards() -> None:
    text = _text()
    for setting in (
        'PARENT_LIMIT="${PARENT_LIMIT:-0}"',
        'SEED="${SEED:-13}"',
        'EPOCHS="${EPOCHS:-100}"',
        'TOP_K_NATIVE="${TOP_K_NATIVE:-20}"',
        'LEARNING_RATE="${LEARNING_RATE:-0.1}"',
        'DROPOUT="${DROPOUT:-0.5}"',
        'RESUME="${RESUME:-true}"',
        'GENERATION_CHUNK_SIZE="${GENERATION_CHUNK_SIZE:-32}"',
        'GENERATION_NUM_WORKERS="${GENERATION_NUM_WORKERS:-0}"',
        'MEMORY_LOG_EVERY_CHUNKS="${MEMORY_LOG_EVERY_CHUNKS:-1}"',
    ):
        assert setting in text
    assert "Completed OUTPUT_DIR cannot be rerun" in text
    assert "Resume requires" in text


def test_wrapper_separates_input_and_selected_parent_expectations() -> None:
    text = _text()
    assert (
        'EXPECTED_INPUT_TRAIN_COUNT="${EXPECTED_INPUT_TRAIN_COUNT:-'
        '$EXPECTED_PARENT_COUNT}"'
    ) in text
    assert (
        'EXPECTED_SELECTED_PARENT_COUNT="${EXPECTED_SELECTED_PARENT_COUNT:-'
        '$PARENT_LIMIT}"'
    ) in text
    assert (
        'EXPECTED_SELECTED_PARENT_COUNT="${EXPECTED_SELECTED_PARENT_COUNT:-'
        '$EXPECTED_INPUT_TRAIN_COUNT}"'
    ) in text
    assert '--expected-parent-count "$EXPECTED_INPUT_TRAIN_COUNT"' in text
    assert '--expected-parent-count "$EXPECTED_SELECTED_PARENT_COUNT"' in text
    assert (
        '--expected-input-train-count "$EXPECTED_INPUT_TRAIN_COUNT"' in text
    )


def test_wrapper_passes_chunking_and_bounds_cpu_allocator_threads() -> None:
    text = _text()
    assert '--generation-chunk-size "$GENERATION_CHUNK_SIZE"' in text
    assert '--generation-num-workers "$GENERATION_NUM_WORKERS"' in text
    assert '--memory-log-every-chunks "$MEMORY_LOG_EVERY_CHUNKS"' in text
    assert 'export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"' in text
    assert 'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"' in text
    assert 'export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"' in text
    assert "unset http_proxy" not in text
    assert "unset https_proxy" not in text
    assert "unset HTTP_PROXY" not in text
    assert "unset HTTPS_PROXY" not in text
