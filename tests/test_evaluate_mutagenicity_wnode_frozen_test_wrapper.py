from __future__ import annotations

import subprocess
from pathlib import Path


WRAPPER = Path("scripts/slurm/evaluate_mutagenicity_wnode_frozen_test.sh")


def test_wrapper_is_portable_frozen_full_test_entrypoint():
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    text = WRAPPER.read_text(encoding="utf-8")
    assert "/share/home" not in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"' in text
    assert "BASH_SOURCE" not in text
    assert "#SBATCH --partition=A800" in text
    assert "#SBATCH --gres=gpu:a800:1" in text
    assert "#SBATCH --time=24:00:00" in text
    assert ": \"${FROZEN_SELECTOR_ROOT:?" in text
    assert ": \"${TEST_CSV:?" in text
    assert ": \"${OUTPUT_DIR:?" in text
    assert ": \"${WNODE_CACHE_DB:?" in text
    assert "EXPECTED_PARENT_COUNT=\"${EXPECTED_PARENT_COUNT:-217}\"" in text
    assert "EXPECTED_CANDIDATE_COUNT=\"${EXPECTED_CANDIDATE_COUNT:-20}\"" in text
    assert "EXPECTED_PAIR_COUNT=\"${EXPECTED_PAIR_COUNT:-4340}\"" in text
    assert "--cohort-name test" in text
    assert "--require-complete-cartesian" in text
    assert "--require-frozen-thresholds" in text
    assert "--require-frozen-candidate-order" in text
    assert "audit_mutagenicity_wnode_frozen_test.py" in text
    assert "[MUTAGENICITY_WNODE_FROZEN_TEST_EVAL_OK]" in text


def test_wrapper_has_no_sampling_or_selection_limits():
    text = WRAPPER.read_text(encoding="utf-8")
    assert "--parent-limit" not in text
    assert "--candidate-limit" not in text
    assert "numpy.quantile" not in text
    assert "select_mutagenicity_wnode_prefix.py" not in text


def test_wrapper_temporarily_disables_nounset_for_shell_and_conda_init():
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    text = WRAPPER.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines()]

    source_index = lines.index("source ~/.bashrc")
    conda_index = lines.index("conda activate smiles_pip118")
    restore_index = lines.index("set -u", conda_index + 1)

    assert lines[source_index - 1] == "set +u"
    assert conda_index == source_index + 1
    assert restore_index == conda_index + 1
    assert "set -euo pipefail" in lines
    assert "BASHRCSOURCED" not in text
