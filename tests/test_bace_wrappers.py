from __future__ import annotations

from pathlib import Path

from src.eval.bace_paper_artifacts import METHODS


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


def test_bace_gcfexplainer_wrappers_preserve_official_full_contract() -> None:
    names = [
        "gcfexplainer/prepare_bace_dataset.sh",
        "gcfexplainer/train_bace_gnn.sh",
        "gcfexplainer/reproduce_bace_vrrw.sh",
        "gcfexplainer/reproduce_bace_summary.sh",
        "bace_eval_gcfexplainer.sh",
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
        assert "export PYTHONPATH=$PWD" in content

    prepare = _text("gcfexplainer/prepare_bace_dataset.sh")
    assert "train_source_label1_teacher_correct.csv" in prepare
    assert "train_target_label0_teacher_correct.csv" in prepare
    assert "val_source_label1_teacher_correct.csv" in prepare
    assert "val_target_label0_teacher_correct.csv" in prepare
    assert "calibration_source" not in prepare
    assert "test_source" not in prepare

    gnn = _text("gcfexplainer/train_bace_gnn.sh")
    for token in (
        "--profile full",
        "--epochs 1000",
        "--train-limit 869",
        "--val-limit 162",
        "--seed 13",
    ):
        assert token in gnn

    vrrw = _text("gcfexplainer/reproduce_bace_vrrw.sh")
    for token in (
        "--profile full",
        "--parent-limit 360",
        "--m 50000",
        "--alpha 1.0",
        "--theta 0.05",
        "--teleport 0.1",
        "--seed 13",
    ):
        assert token in vrrw
    assert "scripts/select_mutagenicity_wnode_prefix.py" not in vrrw

    summary = _text("gcfexplainer/reproduce_bace_summary.sh")
    assert "--native-candidate-limit \"$NATIVE_CANDIDATE_LIMIT\"" in summary
    assert "NATIVE_CANDIDATE_LIMIT=${NATIVE_CANDIDATE_LIMIT:-0}" in summary
    assert "--target-k \"$TARGET_K\"" in summary
    assert "--scan-limit \"$SCAN_LIMIT\"" in summary
    assert "--top-k 20" not in summary


def test_bace_native_candidate_audit_is_nonselective_and_registered_ready() -> None:
    content = _text("gcfexplainer/audit_bace_native_candidates.sh")
    assert "audit_bace_native_candidates.py" in content
    assert "--target-k 20" in content
    assert "--scan-limit 0" in content
    assert "calibration_source_label1_teacher_correct.csv" in content
    assert "test_source_label1_teacher_correct.csv" in content
    assert "select_mutagenicity_wnode_prefix" not in content


def test_bace_gcfexplainer_evaluation_reuses_frozen_wnode_contract() -> None:
    content = _text("bace_eval_gcfexplainer.sh")
    assert "BACE_METHOD=gcfexplainer" in content
    assert "full_v2/export/selected_top20.csv" in content
    assert "outputs/hpc/oracle/bace/bace_teacher.pkl" in content
    assert "bace_ours_wnode_work_v1/thresholds.json" in content
    assert "bace_common3_standardized_v1/gcfexplainer" in content
    assert "bace_eval_method_common.sh" in content
    assert "select_" not in content
    assert METHODS["gcfexplainer"] == {
        "display": "GCFExplainer",
        "candidate_kind": "fullgraph",
        "selection_method": "native_gcf_summary_rank_filtered_by_validity",
    }
