from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import scripts.run_tastemolnet_comrecgc_smoke as cli_module
import src.utils.tastemolnet_t9_comrecgc_release as release_module

from src.baselines.comrecgc.held_upstream import OFFICIAL_SOURCE_SHA256
from src.utils.tastemolnet_t9_comrecgc_release import (
    TasteComRecGCReleaseDisabled,
    TasteComRecGCReleaseError,
    assert_t9_execution_released,
    load_t9_release_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _tracked_payload() -> dict[str, object]:
    return json.loads(release_module.RELEASE_CONFIG_PATH.read_text(encoding="utf-8"))


def _install_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> Path:
    path = tmp_path / "release.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(release_module, "RELEASE_CONFIG_PATH", path)
    return path


def test_tracked_t9_release_is_exact_and_disabled() -> None:
    value = load_t9_release_config()
    assert value["release_enabled"] is False
    assert value["gpu_index"] == 2
    assert value["managed_receipt_kind"] == "taste_t9_gpu2_v1"
    assert value["managed_task_id"] == "tastemolnet_t9_comrecgc_smoke"
    assert value["managed_validator"] == "taste_t9_v1"
    assert value["official_file_sha256"] == dict(OFFICIAL_SOURCE_SHA256)
    with pytest.raises(TasteComRecGCReleaseDisabled):
        assert_t9_execution_released()


@pytest.mark.parametrize(
    "mutator",
    (
        lambda value: value.update(release_enabled=1),
        lambda value: value.update(implementation_commit="a" * 40),
        lambda value: value["official_file_sha256"].update(
            {"comrecgc.py": "0" * 64}
        ),
        lambda value: value.update(gpu_index=True),
    ),
)
def test_t9_release_config_rejects_hostile_types_or_partial_pins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator,
) -> None:
    value = deepcopy(_tracked_payload())
    mutator(value)
    path = _install_config(tmp_path, monkeypatch, value)
    with pytest.raises(TasteComRecGCReleaseError):
        load_t9_release_config(path)


def test_t9_cli_refuses_before_output_or_science_import(
    tmp_path: Path,
) -> None:
    output = tmp_path / "never-created"
    arguments = [
        "--config",
        str(PROJECT_ROOT / "configs/hpc.yaml"),
        "--output-dir",
        str(output),
        "--t2-adoption-root",
        str(tmp_path / "t2"),
        "--t2-adoption-gate-sha256",
        "1" * 64,
        "--t2-adoption-receipt-sha256",
        "2" * 64,
        "--t2-source-evidence-sha256",
        "3" * 64,
        "--t3-output-root",
        str(tmp_path / "t3"),
        "--t4-output-root",
        str(tmp_path / "t4"),
        "--checkpoint-dir",
        str(tmp_path / "checkpoint"),
        "--train-csv",
        str(tmp_path / "train.csv"),
        "--official-root",
        str(tmp_path / "official"),
        "--set",
        "inference.fallback_to_heuristic=false",
    ]
    assert cli_module.main(arguments) == 78
    assert not output.exists()


def test_t9_wrappers_are_gpu2_managed_and_slurm_is_static_refusal() -> None:
    autodl = (
        PROJECT_ROOT / "scripts/autodl/run_tastemolnet_comrecgc_smoke.sh"
    ).read_text(encoding="utf-8")
    slurm = (
        PROJECT_ROOT / "scripts/slurm/run_tastemolnet_comrecgc_smoke.sh"
    ).read_text(encoding="utf-8")
    assert autodl.index("TASTE_T9_COMRECGC_WRAPPER_RELEASED=0") < autodl.index(
        'source "$SCRIPT_DIR/common.sh"'
    )
    for token in (
        "--gpu-index 2",
        "--gpu-lock-mode exclusive",
        "--execution-receipt-kind taste_t9_gpu2_v1",
        "--strict-result-validator taste_t9_v1",
        "--foreground",
    ):
        assert token in autodl
    for token in (
        "#SBATCH --partition=A800",
        "#SBATCH --gres=gpu:a800:1",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
        "source ~/.bashrc",
        "conda activate smiles_pip118",
        "cd /share/home/u20526/czx/counterfactual-subgraph",
        "export PYTHONPATH=$PWD",
        "--config configs/hpc.yaml",
        "--set inference.fallback_to_heuristic=false",
        "exit 64",
    ):
        assert token in slurm
