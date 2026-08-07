from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "scripts/slurm"


def _text(name: str) -> str:
    return (SLURM / name).read_text(encoding="utf-8")


def test_requested_bace_wrappers_use_registered_cluster_contract() -> None:
    names = [
        "bace_train_teacher.sh",
        "bace_generate_ours_candidate_pool.sh",
        "bace_audit_ours_candidate_pool.sh",
        "bace_select_ours_top20.sh",
        "bace_eval_ours.sh",
        "bace_audit_paper_artifacts.sh",
    ]
    for name in names:
        content = _text(name)
        assert "#SBATCH --partition=A800" in content
        assert "#SBATCH --gres=gpu:a800:1" in content
        assert "#SBATCH --cpus-per-task=7" in content
        assert "#SBATCH --output=logs/%j.out" in content
        assert "#SBATCH --error=logs/%j.err" in content
        assert "unset http_proxy" not in content
        assert "unset https_proxy" not in content
        assert 'cd "$PROJECT_ROOT"' in content
        assert "export PYTHONPATH" in content


def test_bace_eval_common_reuses_shared_evaluator_without_selector() -> None:
    content = _text("bace_eval_method_common.sh")
    assert "scripts/evaluate_bace_method.py" in content
    assert "awk 'END {print (NR > 0 ? NR - 1 : 0)}'" in content
    assert "bace_teacher.pkl" in content
    assert "test_source_label1_teacher_correct.csv" in content
    assert "select_mutagenicity_wnode_prefix" not in content
    assert "selection_performed_in_eval" not in content


def test_bace_ours_generation_reuses_frozen_stable300_parameters() -> None:
    content = _text("bace_generate_ours_candidate_pool.sh")
    assert "decoded_chem_ppo_stable300_unified_sftv3" in content
    assert "--num-return-sequences \"$NUM_RETURN_SEQUENCES\"" in content
    assert "NUM_RETURN_SEQUENCES=4" in content
    assert "GEN_TEMPERATURE=0.5" in content
    assert "GEN_TOP_P=0.8" in content
    assert "--substructure-distance-reward-weight 0.3" in content
    assert "--projection-penalty 1.0" in content
    assert "calibration_loaded=false" in content
    assert "test_loaded=false" in content
    assert "awk 'END {print (NR > 0 ? NR - 1 : 0)}'" in content
    assert "RECOVER_GENERATION" in content
    assert "EXPECTED_RAW_POOL_SHA256" in content
    assert "[BACE_OURS_GENERATION_ADOPT_EXISTING] algorithm_rerun=false" in content
    assert "scripts/generate_full_candidate_pool.py" in content
    assert "scripts/baselines/bace/enrich_ours_candidate_pool.py" in content


def test_bace_ours_selector_preserves_existing_coverage_mmr_contract() -> None:
    content = _text("bace_select_ours_top20.sh")
    for token in (
        "--top-k 20",
        "--alpha-cf 1.0",
        "--beta-coverage 1.0",
        "--gamma-redundancy 0.7",
        "--eta-size 0.3",
        "--min-cf-drop 0.2",
        "--require-cf-flip",
        "--require-final-substructure",
        "--dedup-by-final-fragment",
        "--sim-metric morgan",
    ):
        assert token in content
    assert "selected_count=20" in content


def test_bace_ours_evaluation_uses_selector_and_direct_paper_root() -> None:
    content = _text("bace_eval_ours.sh")
    assert "outputs/hpc/selectors/bace_ours_top20" in content
    assert "outputs/hpc/eval/paper/bace_ours_wnode" in content
    assert "OUTPUT_DIR=${OUTPUT_DIR:-$PAPER_ROOT}" in content
    audit = _text("bace_audit_paper_artifacts.sh")
    assert "EXPECTED_METHODS=${EXPECTED_METHODS:-ours}" in audit
    assert "--thresholds-json" in audit
