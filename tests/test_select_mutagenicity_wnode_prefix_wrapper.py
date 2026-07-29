from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "slurm" / "select_mutagenicity_wnode_prefix.sh"


def _text() -> str:
    return WRAPPER.read_text(encoding="utf-8")


def test_wrapper_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_wrapper_uses_portable_root_and_requested_resources() -> None:
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
        "#SBATCH --time=12:00:00",
        "#SBATCH --job-name=mut_wnode_sel",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
    ):
        assert directive in text


def test_wrapper_requires_matrix_and_output_and_refuses_overwrite() -> None:
    text = _text()
    assert ': "${MATRIX_RUN_DIR:?' in text
    assert ': "${OUTPUT_DIR:?' in text
    assert "OUTPUT_DIR exists and is non-empty" in text


def test_wrapper_has_all_preregistered_defaults() -> None:
    text = _text()
    expected = (
        'TOP_K="${TOP_K:-20}"',
        'TABLE_K="${TABLE_K:-10}"',
        'THRESHOLD_QUANTILES="${THRESHOLD_QUANTILES:-0.05,0.10,0.20,0.30,0.50,0.70,0.90}"',
        'THRESHOLD_WEIGHTS="${THRESHOLD_WEIGHTS:-4,4,3,3,2,1,1}"',
        'THETA_STAR_QUANTILE="${THETA_STAR_QUANTILE:-0.30}"',
        'COST_CAP_QUANTILE="${COST_CAP_QUANTILE:-0.90}"',
        'PARENT_LIMIT="${PARENT_LIMIT:-0}"',
        'CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-0}"',
        'LOCAL_SWAP_PASSES="${LOCAL_SWAP_PASSES:-2}"',
        'SEED="${SEED:-13}"',
    )
    for value in expected:
        assert value in text
    assert "1,1,1,1,1,1,1,1,1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5" in text


def test_wrapper_runs_selector_and_strict_audit() -> None:
    text = _text()
    assert "python scripts/select_mutagenicity_wnode_prefix.py" in text
    assert "python scripts/audit_mutagenicity_wnode_prefix.py" in text
    for flag in (
        "--require-all-variants",
        "--require-nested-prefix",
        "--require-monotonic-coverage",
        "--require-nonincreasing-capped-cost",
        "--forbid-test",
    ):
        assert flag in text
    assert "[MUTAGENICITY_WNODE_PREFIX_SELECTOR_OK]" in text


def test_wrapper_uses_calibration_matrix_without_test_input() -> None:
    text = _text()
    assert "pair_matrix.jsonl" in text
    assert "selected_candidate_universe.jsonl" in text
    assert "test.csv" not in text.lower()
