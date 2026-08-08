from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "scripts/slurm"


def _directives(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("#SBATCH --"):
            continue
        key, value = line[len("#SBATCH --") :].split("=", 1)
        result[key] = value
    return result


def test_all_bbbp_stage_wrappers_use_safe_single_a800_contract() -> None:
    wrappers = sorted(SLURM.glob("bbbp_*.sh"))
    assert len(wrappers) >= 30
    for path in wrappers:
        if path.name == "bbbp_stage_common.sh":
            continue
        text = path.read_text(encoding="utf-8")
        directives = _directives(text)
        assert directives["partition"] == "A800", path
        assert directives["nodes"] == "1", path
        assert directives["ntasks-per-node"] == "1", path
        assert int(directives["cpus-per-task"]) <= 7, path
        assert directives["gres"] == "gpu:a800:1", path
        assert directives["output"] == "logs/%j.out", path
        assert directives["error"] == "logs/%j.err", path
        assert "source ~/.bashrc" in text
        assert "conda activate smiles_pip118" in text
        assert "export PYTHONPATH=$PWD" in text or "bbbp_stage_common.sh" in text
        assert not re.search(r"(^|\s)(sbatch|srun|salloc|scancel)(\s|$)", text, re.MULTILINE)
        assert "unset proxy" not in text.lower()


def test_common_wrapper_freezes_evaluation_semantics_and_dry_run() -> None:
    text = (SLURM / "bbbp_stage_common.sh").read_text(encoding="utf-8")
    assert "VALIDATE_ONLY" in text
    assert "DRY_RUN" in text
    assert "cf_mode=strict_flip" in text
    assert "distance_line=MolCLR-Node-Wasserstein" in text
    assert "threshold_source=calibration" in text
    assert "selection_performed_in_eval=false" in text
    assert "threshold_fitted_on_test=false" in text


def test_four_method_and_generalization_wrappers_exist() -> None:
    expected = {
        "bbbp_generate_ours_candidate_pool.sh",
        "bbbp_globalgce_native.sh",
        "bbbp_gcf_vrrw.sh",
        "bbbp_comrecgc_transition_gate.sh",
        "bbbp_cross_scaffold_ours.sh",
        "bbbp_cross_scaffold_globalgce.sh",
        "bbbp_cross_scaffold_gcfexplainer.sh",
        "bbbp_cross_scaffold_comrecgc.sh",
        "bbbp_heldout_ours.sh",
        "bbbp_candidate_source_ablation.sh",
        "bbbp_selector_ablation.sh",
        "bbbp_candidate_budget_ablation.sh",
    }
    assert not sorted(name for name in expected if not (SLURM / name).is_file())


def test_no_existing_aids_mut_or_bace_wrapper_was_edited() -> None:
    # New framework wrappers are BBBP-only; names make accidental reuse visible.
    for path in SLURM.glob("bbbp_*.sh"):
        text = path.read_text(encoding="utf-8")
        assert "outputs/hpc/eval/paper/bace_" not in text
        assert "outputs/hpc/mutagenicity" not in text
        assert "data/raw/AIDS" not in text
