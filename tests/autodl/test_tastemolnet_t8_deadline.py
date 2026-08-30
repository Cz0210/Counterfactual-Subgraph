from pathlib import Path
import pytest
from scripts.autodl import run_tastemolnet_t8_deadline as deadline

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_attempt_id_requires_canonical_uuid4() -> None:
    value = "70db9c1b-3f28-4ae9-bc21-d81027b2e53d"
    assert deadline._uuid4(value) == value
    with pytest.raises(Exception, match="UUIDv4"):
        deadline._uuid4(value.upper())


def test_runner_reuses_frozen_two_target_science_without_rf_or_test_paths() -> None:
    text = (PROJECT_ROOT / "scripts/autodl/run_tastemolnet_t8_deadline.py").read_text()
    for expected in ("run_t8_science", "OfficialGlobalGCEMutagenicityGenerator", '"target_branches": [0, 2]',
                     "source_label=SOURCE_LABEL", "target_label=target_label", "num_classes=NUM_CLASSES",
                     '"rf_oracle_used": False', '"test_loaded": False'):
        assert expected in text
    assert "RandomForestClassifier" not in text
    assert "test.csv" not in text
    assert "calibration.csv" not in text


def test_slurm_wrapper_keeps_hpc_contract() -> None:
    text = (PROJECT_ROOT / "scripts/slurm/run_tastemolnet_t8_deadline.sh").read_text()
    for expected in ("#SBATCH --partition=A800", "#SBATCH --gres=gpu:a800:1", "#SBATCH --output=logs/%j.out",
                     "#SBATCH --error=logs/%j.err", "source ~/.bashrc", "conda activate smiles_pip118",
                     "cd /share/home/u20526/czx/counterfactual-subgraph", "export PYTHONPATH=$PWD",
                     "--config configs/hpc.yaml", "--set inference.fallback_to_heuristic=false"):
        assert expected in text
