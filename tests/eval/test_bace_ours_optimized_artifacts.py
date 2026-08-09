from __future__ import annotations

from pathlib import Path


def test_v2_wrappers_keep_frozen_semantics() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/slurm/bace_ours_eval_wnode_prefix_v2.sh").read_text()
    assert "strict_flip" not in text or "evaluate_bace_method.py" in text
    assert "frozen_selection.json" in text
    assert "--test-evaluation-count 1" in text
    assert "bace_ours_wnode_prefix_v2" in text
    assert "OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_ours_wnode_prefix_v2}" in text


def test_all_v2_wrappers_support_dry_run_and_validation() -> None:
    root = Path(__file__).resolve().parents[2]
    names = (
        "bace_ours_precompute_calibration_matrix_v2.sh",
        "bace_ours_select_wnode_prefix_v2.sh",
        "bace_ours_generate_multiseed_pool_v2.sh",
        "bace_ours_merge_audit_pool_v2.sh",
        "bace_ours_eval_wnode_prefix_v2.sh",
        "bace_ours_audit_wnode_prefix_v2.sh",
    )
    for name in names:
        text = (root / "scripts/slurm" / name).read_text()
        assert "DRY_RUN" in text
        assert "VALIDATE_ONLY" in text
        assert "#SBATCH --partition=A800" in text
        assert "#SBATCH --gres=gpu:a800:1" in text
        assert "export PYTHONPATH=$PWD" in text
