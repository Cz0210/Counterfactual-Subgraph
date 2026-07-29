from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (
    REPO_ROOT
    / "scripts"
    / "slurm"
    / "merge_mutagenicity_ppo_candidate_pools.sh"
)


def _text() -> str:
    return WRAPPER.read_text(encoding="utf-8")


def test_wrapper_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_wrapper_is_portable_and_requires_all_paths() -> None:
    text = _text()

    assert 'PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"' in text
    assert "BASH_SOURCE" not in text
    assert "/share/home" not in text
    for variable in (
        "BASE_POOL",
        "HIGHTEMP_POOL",
        "OUTPUT_DIR",
        "DATASET_PATH",
        "TEACHER_PATH",
    ):
        assert f': "${{{variable}:?' in text
    assert "OUTPUT_DIR already exists and is non-empty" in text


def test_wrapper_uses_full_lossless_key_and_keep_metric() -> None:
    text = _text()

    assert "--dedup-key final_fragment,parent_smiles" in text
    assert "--keep-best-by reward_total" in text
    assert "eligible_input_keys == output_key_set" in text
    assert "missing_eligible_keys = eligible_input_keys - output_key_set" in text
    assert "unexpected_keys = output_key_set - eligible_input_keys" in text
    assert "remaining_duplicate_keys = len(output_keys) - len(output_key_set)" in text
    assert "0 < merged" not in text


def test_wrapper_locks_expected_mutagenicity_experiment_counts() -> None:
    text = _text()

    for expected in (
        "EXPECTED_BASE_ROWS=5792",
        "EXPECTED_HIGHTEMP_ROWS=5792",
        "EXPECTED_INPUT_ROWS=11584",
        "EXPECTED_ELIGIBLE_UNIQUE_KEYS=2773",
        "EXPECTED_MERGED_ROWS=2773",
        "EXPECTED_UNIQUE_PARENTS=1448",
        '"missing_eligible_keys": 0',
        '"unexpected_keys": 0',
        '"remaining_duplicate_keys": 0',
        "merge_semantic_audit.json",
    ):
        assert expected in text


def test_wrapper_runs_all_required_audits() -> None:
    text = _text()

    assert "python scripts/audit_candidate_pool.py" in text
    assert "python scripts/export_candidate_pool_audit_artifacts.py" in text
    assert "python scripts/audit_full_candidate_pool.py" in text
    assert "--smiles-col parent_smiles" in text
    assert "--target-label 1" in text
    assert "--coverage-parent-limit 0" in text


def test_wrapper_has_no_fixed_calibration_or_test_input() -> None:
    text = _text()

    assert "calibration" not in text.lower()
    assert "test" not in text.lower()
    assert "[MUTAGENICITY_PPO_POOL_MERGE_V2_OK]" in text
