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


def test_unified_eval_supports_explicit_resume_and_smoke_audit_input() -> None:
    text = (ROOT / "scripts/slurm/comrecgc_unified_eval.sh").read_text(encoding="utf-8")
    assert 'RESUME="${RESUME:-false}"' in text
    assert "--candidate-filter-audit" in text
    assert '"${RESUME_ARGS[@]}"' in text


def test_official_checkout_is_not_vendored() -> None:
    assert not (ROOT / "baselines/comrecgc_official").exists()
    assert "external/COMRECGC/" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_native_smoke_requires_common_recourse_serialization() -> None:
    text = (ROOT / "scripts/slurm/comrecgc_native_smoke.sh").read_text(encoding="utf-8")
    assert 'NATIVE_PARENT_LIMIT="${NATIVE_PARENT_LIMIT:-64}"' in text
    assert '[[ "$NATIVE_PARENT_LIMIT" == "64" ]]' in text
    assert 'native_common_recourse.json' in text
    assert 'native_representative_counterfactuals.pt' in text


def test_aids_existing_audit_loads_trusted_upstream_pyg_cache() -> None:
    text = (ROOT / "scripts/slurm/comrecgc_aids_existing_audit.sh").read_text(
        encoding="utf-8"
    )
    assert "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1" not in text
    assert text.count("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1") == 1
    assert "audit_trusted_aids_cache.py" in text
    scoped_load = (
        "env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \\\n"
        "python scripts/baselines/comrecgc/audit_aids_native_dbscan.py"
    )
    assert scoped_load in text
    assert "--expected-inventory-sha256" in text

    density = (ROOT / "scripts/slurm/comrecgc_aids_density_retry.sh").read_text(
        encoding="utf-8"
    )
    assert "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1" not in density
    assert density.count("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1") == 1
    assert "audit_trusted_aids_cache.py" in density


def test_mut_trace_adoption_does_not_rerun_random_walk() -> None:
    text = (ROOT / "scripts/slurm/comrecgc_mut_trace_adopt.sh").read_text(
        encoding="utf-8"
    )
    assert "recover_mutagenicity_trace.py" in text
    assert "run_generation.py" not in text
    assert "--source-failed-generation-dir" in text
    assert "algorithm_rerun=false" in text
