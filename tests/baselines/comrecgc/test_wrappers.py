from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
WRAPPERS = sorted((ROOT / "scripts/slurm").glob("comrecgc_*.sh"))


@pytest.mark.parametrize("path", WRAPPERS, ids=lambda path: path.name)
def test_wrappers_use_safe_single_gpu_resources(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --partition=A800" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks-per-node=1" in text
    assert "#SBATCH --gres=gpu:a800:1" in text
    match = re.search(r"#SBATCH --cpus-per-task=(\d+)", text)
    assert match and int(match.group(1)) <= 7
    assert "unset http_proxy" not in text
    assert "unset https_proxy" not in text
    assert "export PYTHONPATH=" in text


def test_full_profile_is_not_an_arbitrary_budget() -> None:
    text = (ROOT / "scripts/slurm/comrecgc_project_generate.sh").read_text(encoding="utf-8")
    assert "EXPECTED_PARENT_LIMIT=1283" in text
    assert "EXPECTED_PARENT_LIMIT=1448" in text
    assert "PARENT_LIMIT=\"${PARENT_LIMIT:-$EXPECTED_PARENT_LIMIT}\"" in text


def test_all_submissions_route_through_registry() -> None:
    text = (ROOT / "scripts/automation/run_comrecgc_baseline.py").read_text(encoding="utf-8")
    assert '"scripts/exp_sbatch.sh"' in text
    assert "subprocess.run([\"sbatch\"" not in text


def test_export_resume_is_explicit_and_defaults_off() -> None:
    text = (ROOT / "scripts/slurm/comrecgc_export.sh").read_text(encoding="utf-8")
    assert 'RESUME="${RESUME:-false}"' in text
    assert '[[ "$RESUME" == "true" || "$RESUME" == "false" ]]' in text
    assert 'RESUME_ARGS=(--resume)' in text
    assert '"${RESUME_ARGS[@]}"' in text


def test_official_checkout_is_not_vendored() -> None:
    assert not (ROOT / "baselines/comrecgc_official").exists()
    assert "external/COMRECGC/" in (ROOT / ".gitignore").read_text(encoding="utf-8")
