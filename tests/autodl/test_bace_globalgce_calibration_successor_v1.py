from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.autodl import build_bace_globalgce_calibration_successor_v1 as successor


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[argparse.Namespace, dict[str, Path]]:
    project = tmp_path / "project"
    project.mkdir()
    python = tmp_path / "env/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    old = tmp_path / "control/old-controller"
    registry = tmp_path / "control/experiment_registry/run_state/candidate-run"
    candidate = tmp_path / "outputs/candidate/attempt-0"
    candidate.mkdir(parents=True)
    source_controller = tmp_path / "control/manifests/old.json"
    _write_json(source_controller, {"schema_version": 1})
    candidate_input = tmp_path / "inputs/source.json"
    _write_json(candidate_input, {"status": "FROZEN"})
    calibration = tmp_path / "data/calibration.csv"
    test = tmp_path / "data/test.csv"
    native = tmp_path / "data/train.csv"
    source = tmp_path / "data/source.jsonl"
    for path in (calibration, test, native, source):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")
    failed = tmp_path / "failed"
    round_root = failed / "rounds/round-1-seed-7"
    official = tmp_path / "official"
    checkpoint = tmp_path / "gine"
    molclr = tmp_path / "molclr"
    for path in (round_root, official, checkpoint, molclr):
        path.mkdir(parents=True)
    molclr_checkpoint = molclr / "model.pt"
    molclr_checkpoint.write_bytes(b"model")
    log = tmp_path / "logs/candidate.log"
    log.parent.mkdir()
    log.write_text("[CANDIDATE_PASS]\n", encoding="utf-8")
    (candidate / "candidate_universe.jsonl").write_text(
        "".join(json.dumps({"candidate_id": f"rule-{index}"}) + "\n" for index in range(20)),
        encoding="utf-8",
    )
    _write_json(
        candidate / "run_manifest.json",
        {"candidate_count": 20, "oracle_checkpoint_hash": "a" * 64},
    )
    _write_json(
        candidate / "summary.json",
        {"oracle_checkpoint_hash": "a" * 64},
    )
    recovery_command = [
        str(python),
        str(project / "scripts/autodl/recover_bace_globalgce_terminal.py"),
        "recover",
        "--failed-controller-root",
        str(failed),
        "--source-round-root",
        str(round_root),
        "--source-manifest",
        str(source),
        "--native-train-csv",
        str(native),
        "--official-root",
        str(official),
        "--gnn-checkpoint",
        str(checkpoint),
        "--output-dir",
        str(candidate),
    ]
    calibration_command = [
        str(python),
        str(project / "scripts/autodl/run_bace_baseline_gnn_route.py"),
        "verify-shard",
        "--split-path",
        str(calibration),
        "--molclr-root",
        str(molclr),
        "--molclr-checkpoint",
        str(molclr_checkpoint),
    ]
    test_command = [*calibration_command]
    test_command[test_command.index(str(calibration))] = str(test)
    candidate_task = {
        "id": successor.TASK_ID,
        "stage": "BACE_GLOBALGCE_AFFINE_EDGE_TERMINAL_RECOVERY",
        "depends_on": [],
        "command": recovery_command,
    }
    tasks = [
        candidate_task,
        {"id": successor.CALIBRATION_TASK, "command": calibration_command},
        {"id": successor.TEST_TASK, "command": test_command},
    ]
    _write_json(
        old / "controller_manifest.json",
        {
            "controller_id": "old-controller",
            "source_manifest": str(source_controller),
            "source_manifest_sha256": _sha(source_controller),
            "tasks": tasks,
        },
    )
    run_id = "candidate-run"
    _write_json(
        old / f"tasks/{successor.TASK_ID}/state.json",
        {
            "state": "PASS",
            "instances": {
                "main": {
                    "state": "PASS",
                    "run_id": run_id,
                    "expected_output": str(candidate),
                }
            },
        },
    )
    _write_json(
        old / f"tasks/{successor.TASK_ID}/gate.json",
        {"status": "PASS", "runs": [{"state": "PASS", "run_id": run_id}]},
    )
    _write_json(
        old / f"tasks/{successor.TASK_ID}/manifest.json",
        {"task_id": successor.TASK_ID, "status": "FROZEN"},
    )
    registry_spec = {
        "run_id": run_id,
        "dataset": "bace-baseline-globalgce",
        "stage": "BACE_GLOBALGCE_AFFINE_EDGE_TERMINAL_RECOVERY",
        "command": recovery_command,
        "input_manifest": str(candidate_input),
        "input_hash": _sha(candidate_input),
        "expected_output": str(candidate),
        "required_output_files": ["candidate_universe.jsonl", "run_manifest.json", "PASS"],
        "required_output_any": [],
        "required_absolute_output_files": [],
        "required_log_marker": "[CANDIDATE_PASS]",
        "python_executable": str(python),
        "project_root": str(project),
        "git_commit": "b" * 40,
        "max_gpus": 4,
        "heavy": False,
        "config_files": [],
        "config_hash": None,
        "environment": {
            "PYTHONPATH": str(project),
            "RUN_TASTEMOLNET": "0",
        },
        "gpu_index": None,
        "gpu_uuid": None,
    }
    _write_json(registry / "launch_spec.json", registry_spec)
    _write_json(
        registry / "state.json",
        {
            "run_id": run_id,
            "dataset": registry_spec["dataset"],
            "stage": registry_spec["stage"],
            "state": "PASS",
            "log_path": str(log),
        },
    )
    paths = {
        "project": project,
        "python": python,
        "old": old,
        "registry": registry,
        "candidate": candidate,
        "fragment": tmp_path / "fresh/fragment.json",
        "manifest": tmp_path / "fresh/manifest.json",
        "output": tmp_path / "outputs/fresh-successor",
    }
    args = argparse.Namespace(
        project_root=project,
        python=python,
        old_controller_root=old,
        old_controller_id="old-controller",
        registry_run_root=registry,
        run_id=run_id,
        candidate_output=candidate,
        controller_id="fresh-successor",
        output_root=paths["output"],
        fragment=paths["fragment"],
        manifest=paths["manifest"],
    )
    return args, paths


def _fresh_fragment(project: Path) -> dict[str, Any]:
    return {
        "schema_version": "bace_globalgce_terminal_recovery_fragment_v1",
        "root_task_ids": [successor.TASK_ID],
        "terminal_task_ids": ["bace_globalgce_standardized"],
        "tasks": [
            {
                "id": successor.TASK_ID,
                "stage": "RECOVERY",
                "runner_dataset": "bace-baseline-globalgce",
                "runner_stage": "RECOVERY",
                "depends_on": [],
                "resource": "cpu",
                "command": [str(project / "recover.py")],
                "input_manifest": "/input",
                "expected_output": "/output",
                "required_output_files": ["PASS"],
                "required_log_marker": "PASS",
                "environment": {},
            },
            {
                "id": successor.CALIBRATION_TASK,
                "depends_on": [successor.TASK_ID],
                "command": [str(project / "scripts/autodl/run_bace_baseline_gnn_route.py"), "verify-shard"],
            },
            {
                "id": successor.TEST_TASK,
                "depends_on": ["bace_globalgce_selection"],
                "command": [str(project / "scripts/autodl/run_bace_baseline_gnn_route.py"), "verify-shard"],
            },
            {
                "id": "bace_globalgce_standardized",
                "depends_on": [successor.TEST_TASK],
                "command": [str(project / "scripts/autodl/standardize_bace_frozen_cell.py")],
            },
        ],
    }


def _patch_build_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    args: argparse.Namespace,
    *,
    fragment: dict[str, Any] | None = None,
) -> None:
    monkeypatch.setattr(successor, "_require_current_fixed_worktree", lambda _path: "c" * 40)
    monkeypatch.setattr(successor, "validate_recovered_candidate_root", lambda *a, **k: {})
    monkeypatch.setattr(successor, "_open_writers", lambda _root: [])
    monkeypatch.setattr(
        successor,
        "build_recovery_controller_fragment",
        lambda **_kwargs: fragment or _fresh_fragment(args.project_root),
    )

    def compose(*, controller_id: str, fragments: list[Path], output: Path) -> dict[str, Any]:
        _write_json(output, {"controller_id": controller_id, "fragments": [str(p) for p in fragments]})
        return {"manifest_sha256": _sha(output)}

    monkeypatch.setattr(successor, "compose_manifest", compose)
    monkeypatch.setattr(
        successor,
        "load_controller_manifest",
        lambda _path: SimpleNamespace(
            tasks=tuple(_fresh_fragment(args.project_root)["tasks"]),
            by_id={
                successor.TASK_ID: SimpleNamespace(adopt_existing_run_id=args.run_id)
            },
        ),
    )


def test_builds_strict_candidate_adoption_without_generation_or_ablation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, paths = _fixture(tmp_path)
    _patch_build_dependencies(monkeypatch, args)
    result = successor.build(args)
    fragment = json.loads(paths["fragment"].read_text(encoding="utf-8"))
    tasks = {row["id"]: row for row in fragment["tasks"]}
    adopted = tasks[successor.TASK_ID]
    assert adopted["adopt_existing_run_id"] == args.run_id
    assert adopted["expected_output"] == str(paths["candidate"])
    assert adopted["depends_on"] == []
    assert adopted["read_only_adoption"] is True
    assert adopted["retraining_forbidden"] is True
    assert result["candidate_generation_replayed"] is False
    assert result["training_replayed"] is False
    assert result["gspan_replayed"] is False
    assert result["gnn_ablation_started"] is False
    assert result["candidate_rule_count"] == 20


def test_fails_closed_when_registry_state_is_not_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, paths = _fixture(tmp_path)
    state_path = paths["registry"] / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["state"] = "FAILED"
    _write_json(state_path, state)
    _patch_build_dependencies(monkeypatch, args)
    with pytest.raises(RuntimeError, match="not all PASS"):
        successor.build(args)
    assert not paths["fragment"].exists()


def test_rejects_generation_work_in_fresh_downstream_fragment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, paths = _fixture(tmp_path)
    fragment = _fresh_fragment(args.project_root)
    fragment["tasks"][1]["command"].append("globalgce-train-rules")
    _patch_build_dependencies(monkeypatch, args, fragment=fragment)
    with pytest.raises(RuntimeError, match="retained generation work"):
        successor.build(args)
    assert not paths["fragment"].exists()
