from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from scripts.autodl import run_bace_globalgce_v6 as v6
from scripts.autodl.run_four_gpu_recovery_controller import load_controller_manifest


def _adoption_args(tmp_path: Path) -> list[str]:
    return [
        "--source-run-manifest",
        str(tmp_path / "v5" / "run_manifest.json"),
        "--source-task-state",
        str(tmp_path / "control" / "state.json"),
        "--source-checkpoint",
        str(tmp_path / "v5" / "checkpoint.json"),
        "--source-sqlite",
        str(tmp_path / "v5" / "frequent_patterns.sqlite3"),
        "--official-root",
        str(tmp_path / "official"),
        "--native-train-csv",
        str(tmp_path / "data" / "train.csv"),
        "--source-manifest",
        str(tmp_path / "source" / "source_manifest.jsonl"),
        "--gine-checkpoint",
        str(tmp_path / "gine"),
    ]


def test_mining_decision_explicitly_falls_back_to_fresh_exact_topk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject(**_kwargs):
        raise v6.GlobalGCEMiningAdoptionError("writer closure failed")

    monkeypatch.setattr(v6, "build_globalgce_gspan_adoption", reject)
    output = tmp_path / "decision"
    assert v6.main(
        ["mining-decision", *_adoption_args(tmp_path), "--output-dir", str(output)]
    ) == 0
    decision = json.loads((output / "decision.json").read_text())
    assert decision["route"] == "fresh_exact_top_k_v2"
    assert decision["fresh_remine_fallback"] is True
    assert (output / "FRESH_REMINE_REQUIRED").is_file()
    assert not (output / "adoption_proof.json").exists()
    assert (output / "PASS").is_file()


def test_v6_manifest_makes_preflight_decision_and_bridge_cpu_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.autodl import run_four_gpu_recovery_controller as controller

    # pytest's own parent directory contains the token "test"; keep the
    # controller's scientific split audit focused on explicit fixture names.
    monkeypatch.setattr(controller, "TEST_PATH", re.compile(r"/test\.jsonl$"))
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    fragment = runtime / "control" / "v6-fragment.json"
    manifest = runtime / "control" / "v6-manifest.json"
    args = [
        "build-manifest",
        "--controller-id",
        "bace-globalgce-v6-test",
        "--python",
        "/env/bin/python",
        "--project-root",
        str(tmp_path / "project"),
        "--runtime-root",
        str(runtime),
        "--output-root",
        str(runtime / "outputs" / "v6"),
        "--fragment-output",
        str(fragment),
        "--manifest-output",
        str(manifest),
        "--dataset-dir",
        str(runtime / "data"),
        "--calibration-split",
        str(runtime / "calibration.jsonl"),
        "--test-split",
        str(runtime / "test.jsonl"),
        "--molclr-root",
        str(runtime / "molclr"),
        "--molclr-checkpoint",
        str(runtime / "molclr.pt"),
        "--neurosed-checkpoint",
        str(runtime / "neurosed.pt"),
        *_adoption_args(runtime),
    ]
    assert v6.main(args) == 0
    loaded = load_controller_manifest(manifest)
    tasks = {task.task_id: task for task in loaded.tasks}
    decision = tasks[v6.DECISION_TASK_ID]
    bridge = tasks["bace_globalgce_bridge_smoke"]
    train = tasks["bace_globalgce_train_candidates"]
    assert decision.resource == "cpu"
    assert bridge.resource == "cpu"
    assert "cpu" in bridge.command
    assert set(train.depends_on) == {
        v6.DECISION_TASK_ID,
        "bace_globalgce_bridge_smoke",
    }
    assert "--gspan-mining-decision" in train.command
    assert train.resource == "gpu"
    assert fragment.is_file()
    assert manifest.is_file()
