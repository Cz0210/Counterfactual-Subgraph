from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTODL = PROJECT_ROOT / "scripts" / "autodl"


def test_tastemolnet_full_script_has_two_independent_heavy_guards() -> None:
    script = (AUTODL / "run_tastemolnet_gnn_full.sh").read_text(encoding="utf-8")
    assert 'RUN_TASTEMOLNET" != "1"' in script
    assert "--heavy" in script


def test_bace_scripts_use_stage_order_and_task_specific_gnn() -> None:
    smoke = (AUTODL / "run_bace_gnn_smoke.sh").read_text(encoding="utf-8")
    full = (AUTODL / "run_bace_gnn_full.sh").read_text(encoding="utf-8")
    assert "B2_GNN_SMOKE" in smoke
    assert "B3_GNN_FULL" in full
    for script in (smoke, full):
        assert "--dataset bace" in script
        assert "train_molecular_gnn.py" in script
        assert "configs/datasets/bace_gnn.yaml" in script
        assert "random_forest" not in script.lower()
        assert "sbatch" not in script.lower()


def test_waiting_launcher_never_starts_tastemolnet() -> None:
    script = (AUTODL / "launch_bace_when_idle.sh").read_text(encoding="utf-8")
    assert "export RUN_TASTEMOLNET=0" in script
    assert "run_bace_gnn_smoke.sh" in script
    assert "run_bace_gnn_full.sh" in script
    assert "run_tastemolnet" not in script


def test_every_python_entrypoint_has_paired_slurm_status_wrapper() -> None:
    for name in ("detect_runtime", "gpu_inventory", "gpu_lock", "exp_run", "status"):
        wrapper = PROJECT_ROOT / "scripts" / "slurm" / f"{name}.sh"
        assert wrapper.is_file(), name
        text = wrapper.read_text(encoding="utf-8")
        assert "#SBATCH --partition=A800" in text
        assert "#SBATCH --gres=gpu:a800:1" in text
        assert "--config configs/hpc.yaml" in text
