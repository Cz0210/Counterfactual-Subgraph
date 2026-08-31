from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autodl.run_fast_16of16_v2 import SCHEMA, TASKS, load_spec, run


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _spec(tmp_path: Path) -> dict[str, object]:
    tasks: dict[str, object] = {}
    for name in TASKS:
        root = tmp_path / name
        root.mkdir()
        tasks[name] = {
            "root": str(root),
            "pid": None,
            "start_ticks": None,
            "command_token": None,
            "progress_files": [str(root / "progress.json")],
            "terminal_files": [str(root / "PASS")],
        }
    return {
        "schema_version": SCHEMA,
        "controller_id": "fast-v2-test",
        "state_root": str(tmp_path / "state"),
        "execution_commit": "a" * 40,
        "execution_tree": "b" * 40,
        "poll_seconds": 60,
        "run_gnn_ablation": False,
        "tasks": tasks,
    }


def test_once_persists_all_fixed_queued_tasks_without_launch_authority(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec(tmp_path)))

    assert run(spec_path, once=True) == 0
    heartbeat = json.loads((tmp_path / "state/heartbeat.json").read_text())
    assert heartbeat["scope"] == "FIXED_EIGHT_STAGE_OBSERVER"
    assert heartbeat["science_launch_allowed"] is False
    assert heartbeat["process_termination_allowed"] is False
    assert heartbeat["tasks"]["mut_exact"]["process"]["state"] == "QUEUED"
    assert heartbeat["tasks"]["taste_t14"]["process"]["state"] == "QUEUED"
    assert heartbeat["run_gnn_ablation"] is False


def test_spec_rejects_task_reorder_and_ablation(tmp_path: Path) -> None:
    value = _spec(tmp_path)
    value["run_gnn_ablation"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="ablation"):
        load_spec(path)

    value["run_gnn_ablation"] = False
    tasks = value["tasks"]
    assert isinstance(tasks, dict)
    value["tasks"] = {name: tasks[name] for name in reversed(TASKS)}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="task order"):
        load_spec(path)


def test_launch_and_slurm_contracts_are_narrow() -> None:
    launch = (PROJECT_ROOT / "scripts/autodl/launch_fast_16of16_v2.sh").read_text()
    slurm = (PROJECT_ROOT / "scripts/slurm/run_fast_16of16_v2.sh").read_text()
    for token in (
        "FAST_16OF16_V2_SPEC",
        "RUN_GNN_ABLATION",
        "status_fast_16of16_v2.py",
        "run_fast_16of16_v2.py",
    ):
        assert token in launch
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
        "exit 64",
    ):
        assert token in slurm
