from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autodl.run_deadline_main_completion_v1 import load_spec, observe, run


def _spec(tmp_path: Path) -> dict:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    return {
        "schema_version": "deadline_main_completion_spec_v1",
        "controller_id": "deadline-test",
        "state_root": str(tmp_path / "state"),
        "execution_commit": "1" * 40,
        "execution_tree": "2" * 40,
        "poll_seconds": 60,
        "run_gnn_ablation": False,
        "observed_processes": {},
        "observed_artifacts": {"artifact": str(artifact)},
    }


def test_spec_disables_ablation_and_observes_artifact(tmp_path: Path) -> None:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(_spec(tmp_path)), encoding="utf-8")
    spec = load_spec(path)
    state = observe(spec, sequence=1)
    assert state["run_gnn_ablation"] is False
    assert state["artifacts"]["artifact"]["exists"] is True
    assert len(state["artifacts"]["artifact"]["sha256"]) == 64


def test_spec_rejects_gnn_ablation(tmp_path: Path) -> None:
    value = _spec(tmp_path)
    value["run_gnn_ablation"] = True
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="GNN ablation"):
        load_spec(path)


def test_once_writes_controller_receipt_and_heartbeat(tmp_path: Path) -> None:
    value = _spec(tmp_path)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert run(path, once=True) == 0
    state_root = Path(value["state_root"])
    receipt = json.loads(
        (state_root / "controller_receipt.json").read_text(encoding="utf-8")
    )
    heartbeat = json.loads(
        (state_root / "heartbeat.json").read_text(encoding="utf-8")
    )
    assert receipt["run_gnn_ablation"] is False
    assert heartbeat["sequence"] == 1
