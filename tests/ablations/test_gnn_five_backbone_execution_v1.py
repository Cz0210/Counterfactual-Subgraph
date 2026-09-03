from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
import time
from uuid import uuid4

import pytest

from src.ablations.gnn.five_backbone_execution import (
    EXECUTED_BACKBONES,
    FiveBackboneExecutionError,
    FiveBackboneExecutionSpec,
    canonical_json_sha256,
    load_launch_evidence,
    run_five_backbone_execution,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _artifact(path: Path, role: str) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(role + "\n", encoding="utf-8")
    return {"role": role, "path": str(path), "sha256": sha256_file(path)}


def _queue(path: Path, tasks: list[dict[str, object]] | None = None) -> Path:
    return _write_json(
        path,
        {"status": "PASS", "ready_waiting_gpu": [] if tasks is None else tasks},
    )


def _status(tmp_path: Path, *, allowed: bool = True):
    authority = tmp_path / "authority"
    matrix = _write_json(authority / "matrix_status.json", {"cells": 16})
    combined = _write_json(authority / "combined_audit.json", {"status": "PASS"})
    payload = {
        "schema_version": "gnn_five_backbone_launch_decision_v1",
        "state": (
            "AUTHORIZED_TO_LAUNCH_FIVE_BACKBONE_PHASE1"
            if allowed
            else "BLOCKED_GNN_FIVE_BACKBONE_GATE"
        ),
        "science_launch_allowed": allowed,
        "blockers": [] if allowed else ["WAITING_HASH_CLOSED_MAIN_16_OF_16_AND_FINAL_EXPORTS"],
        "max_concurrent_gpus": 2,
        "phase1_seed": 7,
        "main_gate_pass": allowed,
        "user_authorized_after_16": True,
        "run_requested": True,
        "no_main_task_waiting_for_gpu": True,
        "proposal_fixed_manifest_pass": True,
        "gatedgcn_plus_runtime_pass": True,
        "graph_mamba_run_enabled": False,
        "backbones": ["gine", "gin", "gcn", "gatv2", "gatedgcn_plus"],
        "schedule": {
            "lane0": ["gine", "gin", "gatedgcn_plus"],
            "lane1": ["gcn", "gatv2"],
        },
        "main_matrix_modified": False,
        "main_gate": {
            "science_launch_allowed": allowed,
            "main_matrix_complete_cells": 16 if allowed else 15,
            "main_matrix_total_cells": 16,
            "authority_verified": True,
            "authority_root": str(authority),
            "matrix_status_sha256": sha256_file(matrix),
            "combined_audit_sha256": sha256_file(combined),
            "final_audit_pass": True,
            "figure3_pass": True,
            "figure4_pass": True,
            "table2_pass": True,
            "explicit_run_authorization": True,
        },
    }
    status = _write_json(tmp_path / "status.json", payload)
    return status, sha256_file(status), authority


def _science_script(project_root: Path) -> Path:
    script = project_root / "scripts" / "fake_science.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        r'''#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument("--config")
parser.add_argument("--set", action="append", default=[])
parser.add_argument("--backbone", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--checkpoint-request", type=Path, required=True)
parser.add_argument("--checkpoint-path", type=Path)
parser.add_argument("--event-log", type=Path, required=True)
parser.add_argument("--trigger-main-queue", type=Path)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()
args.output_root.mkdir(parents=True, exist_ok=True)
with args.event_log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"event":"START","backbone":args.backbone,"seed":args.seed,"time":time.time(),"gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),"resume":args.resume}) + "\n")
if args.trigger_main_queue is not None and not args.resume:
    args.trigger_main_queue.write_text(json.dumps({"status":"PASS","ready_waiting_gpu":[{"task_id":"main"}]}) + "\n")
deadline = time.time() + 0.12
while time.time() < deadline:
    if args.checkpoint_request.exists():
        checkpoint = {
            "schema_version":"gnn_five_backbone_task_checkpoint_v1",
            "status":"PAUSED_AT_SAFE_CHECKPOINT",
            "backbone":args.backbone,
            "seed":args.seed,
            "output_root":str(args.output_root),
            "checkpoint_resume_supported":True,
            "main_matrix_modified":False,
        }
        checkpoint["checkpoint_sha256"] = digest(checkpoint)
        target = args.checkpoint_path or args.output_root / "checkpoint.json"
        target.write_text(json.dumps(checkpoint, sort_keys=True) + "\n")
        raise SystemExit(75)
    time.sleep(0.005)
terminal = {
    "schema_version":"gnn_five_backbone_task_terminal_v1",
    "status":"PASS",
    "backbone":args.backbone,
    "seed":args.seed,
    "output_root":str(args.output_root),
    "checkpoint_resume_supported":True,
    "selector_frozen_before_test":True,
    "main_matrix_modified":False,
}
terminal["terminal_sha256"] = digest(terminal)
(args.output_root / "terminal.json").write_text(json.dumps(terminal, sort_keys=True) + "\n")
with args.event_log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"event":"END","backbone":args.backbone,"seed":args.seed,"time":time.time()}) + "\n")
''',
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _spec_payload(
    tmp_path: Path,
    *,
    seeds: list[int] | None = None,
    trigger_queue: Path | None = None,
) -> tuple[dict[str, object], Path]:
    project_root = tmp_path / "checkout"
    project_root.mkdir()
    script = _science_script(project_root)
    config = project_root / "configs" / "hpc.yaml"
    config.parent.mkdir()
    config.write_text("runtime: test\n", encoding="utf-8")
    event_log = tmp_path / "events.jsonl"
    run_id = str(uuid4())
    output_root = tmp_path / f"gnn-five-{run_id}"

    commands: dict[str, object] = {}
    lanes = {
        "gin": "lane0",
        "gatedgcn_plus": "lane0",
        "gcn": "lane1",
        "gatv2": "lane1",
    }
    for backbone in EXECUTED_BACKBONES:
        common = [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--set",
            "inference.fallback_to_heuristic=false",
            "--backbone",
            "{backbone}",
            "--seed",
            "{seed}",
            "--output-root",
            "{task_root}",
            "--checkpoint-request",
            "{checkpoint_request}",
            "--event-log",
            str(event_log),
        ]
        if trigger_queue is not None and backbone == "gin":
            common.extend(["--trigger-main-queue", str(trigger_queue)])
        commands[backbone] = {
            "lane": lanes[backbone],
            "argv_template": common,
            "resume_argv_template": common
            + ["--resume", "--checkpoint-path", "{checkpoint_path}"],
            "terminal_relpath": "terminal.json",
            "checkpoint_relpath": "checkpoint.json",
            "pause_exit_code": 75,
        }
    reference_root = tmp_path / "gine"
    payload: dict[str, object] = {
        "schema_version": "gnn_five_backbone_execution_spec_v1",
        "run_id": run_id,
        "execution_commit": "a" * 40,
        "project_root": str(project_root),
        "output_root": str(output_root),
        "max_concurrent_gpus": 2,
        "seeds": [7] if seeds is None else seeds,
        "lane_gpu_ids": {"lane0": "0", "lane1": "1"},
        "main_matrix_write_allowed": False,
        "gine_reference_artifacts": [
            _artifact(reference_root / "model.pt", "classifier_checkpoint"),
            _artifact(reference_root / "temperature.json", "temperature"),
            _artifact(reference_root / "run_manifest.json", "run_manifest"),
        ],
        "science_commands": commands,
    }
    payload["run_spec_sha256"] = canonical_json_sha256(payload)
    return payload, event_log


def _parsed_spec(tmp_path: Path, **kwargs) -> tuple[FiveBackboneExecutionSpec, Path]:
    payload, events = _spec_payload(tmp_path, **kwargs)
    return (
        FiveBackboneExecutionSpec.from_mapping(
            payload, expected_project_root=Path(str(payload["project_root"]))
        ),
        events,
    )


def test_incomplete_science_commands_fail_closed_before_output_creation(tmp_path: Path) -> None:
    payload, _ = _spec_payload(tmp_path)
    del payload["science_commands"]["gatv2"]  # type: ignore[index]
    payload["run_spec_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "run_spec_sha256"}
    )
    with pytest.raises(FiveBackboneExecutionError, match="all four"):
        FiveBackboneExecutionSpec.from_mapping(
            payload, expected_project_root=Path(str(payload["project_root"]))
        )
    assert not Path(str(payload["output_root"])).exists()


def test_false_status_cannot_launch_science(tmp_path: Path) -> None:
    status, digest, _ = _status(tmp_path, allowed=False)
    with pytest.raises(FiveBackboneExecutionError, match="does not authorize"):
        load_launch_evidence(status, digest)


def test_gine_is_adopted_and_two_lanes_execute_without_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, events = _parsed_spec(tmp_path)
    status, digest, authority = _status(tmp_path)
    launch = load_launch_evidence(status, digest)
    queue = _queue(tmp_path / "main-ready.json")
    from src.ablations.gnn import five_backbone_execution as execution

    real_popen = execution.subprocess.Popen
    popen_calls: list[dict[str, object]] = []

    def guarded_popen(*args, **kwargs):
        assert kwargs.get("shell") is False
        popen_calls.append(dict(kwargs))
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(execution.subprocess, "Popen", guarded_popen)
    result = run_five_backbone_execution(
        spec, launch, main_ready_gpu_tasks=queue, resume=False, poll_seconds=0.01
    )
    assert result["state"] == "PASS"
    assert len(popen_calls) == 4
    assert (Path(spec.output_root) / "gine_reference_adoption.json").is_file()
    assert not (Path(spec.output_root) / "gine" / "seed7").exists()
    assert sha256_file(authority / "matrix_status.json") == launch.matrix_status_sha256
    records = [json.loads(line) for line in events.read_text().splitlines()]
    starts = {(row["backbone"], row["seed"]): row for row in records if row["event"] == "START"}
    assert starts[("gin", 7)]["gpu"] == "0"
    assert starts[("gatedgcn_plus", 7)]["gpu"] == "0"
    assert starts[("gcn", 7)]["gpu"] == "1"
    assert starts[("gatv2", 7)]["gpu"] == "1"


def test_seed17_waits_until_every_seed7_lane_task_finishes(tmp_path: Path) -> None:
    spec, events = _parsed_spec(tmp_path, seeds=[7, 17])
    status, digest, _ = _status(tmp_path)
    queue = _queue(tmp_path / "main-ready.json")
    result = run_five_backbone_execution(
        spec,
        load_launch_evidence(status, digest),
        main_ready_gpu_tasks=queue,
        resume=False,
        poll_seconds=0.01,
    )
    assert result["state"] == "PASS"
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    seed7_end = max(row["time"] for row in rows if row["seed"] == 7 and row["event"] == "END")
    seed17_start = min(
        row["time"] for row in rows if row["seed"] == 17 and row["event"] == "START"
    )
    assert seed17_start >= seed7_end


def test_main_ready_during_science_requests_safe_checkpoint_and_resume(
    tmp_path: Path,
) -> None:
    queue = _queue(tmp_path / "main-ready.json")
    spec, events = _parsed_spec(tmp_path, trigger_queue=queue)
    status, digest, _ = _status(tmp_path)
    launch = load_launch_evidence(status, digest)
    paused = run_five_backbone_execution(
        spec, launch, main_ready_gpu_tasks=queue, resume=False, poll_seconds=0.005
    )
    assert paused["state"] == "PAUSED_MAIN_PRIORITY"
    assert not (Path(spec.output_root) / "gatedgcn_plus" / "seed7").exists()
    assert not (Path(spec.output_root) / "gatv2" / "seed7").exists()
    paused_tasks = [
        value for value in paused["tasks"].values() if value["state"] == "PAUSED_AT_SAFE_CHECKPOINT"
    ]
    assert paused_tasks
    for receipt in paused_tasks:
        assert Path(receipt["checkpoint_path"]).is_file()

    _queue(queue)
    resumed = run_five_backbone_execution(
        spec, launch, main_ready_gpu_tasks=queue, resume=True, poll_seconds=0.005
    )
    assert resumed["state"] == "PASS"
    assert len([row for row in events.read_text().splitlines() if row]) >= 8


def test_main_queue_missing_is_conservatively_blocked(tmp_path: Path) -> None:
    spec, _ = _parsed_spec(tmp_path)
    status, digest, _ = _status(tmp_path)
    with pytest.raises(FiveBackboneExecutionError, match="main priority blocks"):
        run_five_backbone_execution(
            spec,
            load_launch_evidence(status, digest),
            main_ready_gpu_tasks=tmp_path / "missing.json",
            resume=False,
            poll_seconds=0.01,
        )
    assert not Path(spec.output_root).exists()


def test_cli_and_two_gpu_slurm_are_real_and_policy_compliant() -> None:
    runner = REPO_ROOT / "scripts/autodl/run_gnn_five_backbone_ablation_v1.py"
    slurm = REPO_ROOT / "scripts/slurm/run_gnn_five_backbone_ablation_v1.sh"
    assert runner.is_file() and os.access(runner, os.X_OK)
    assert slurm.is_file() and os.access(slurm, os.X_OK)
    runner_source = runner.read_text(encoding="utf-8")
    assert "run_five_backbone_execution(" in runner_source
    slurm_source = slurm.read_text(encoding="utf-8")
    assert "#SBATCH --partition=A800" in slurm_source
    assert "#SBATCH --gres=gpu:a800:2" in slurm_source
    assert "#SBATCH --output=logs/%j.out" in slurm_source
    assert "#SBATCH --error=logs/%j.err" in slurm_source
    assert "source ~/.bashrc" in slurm_source
    assert "conda activate smiles_pip118" in slurm_source
    assert "cd /share/home/u20526/czx/counterfactual-subgraph" in slurm_source
    assert "export PYTHONPATH=$PWD" in slurm_source
    assert "--config configs/hpc.yaml" in slurm_source
    assert "--set inference.fallback_to_heuristic=false" in slurm_source
