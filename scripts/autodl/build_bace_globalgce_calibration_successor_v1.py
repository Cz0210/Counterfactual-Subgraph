#!/usr/bin/env python3
"""Build one strict-registry BACE GlobalGCE calibration-only successor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from scripts.autodl.build_four_by_four_manifest import compose_manifest
from scripts.autodl.run_four_gpu_recovery_controller import load_controller_manifest
from src.baselines.bace_globalgce_terminal_recovery import (
    build_recovery_controller_fragment,
    validate_recovered_candidate_root,
)
from src.baselines.bace_gnn_baseline_generic_adapter import (
    atomic_write_generic_fragment,
)
from src.utils.autodl_runtime import sha256_paths


TASK_ID = "bace_globalgce_train_candidates"
CALIBRATION_TASK = "bace_globalgce_calibration_shard_0"
TEST_TASK = "bace_globalgce_test_shard_0"
HEX40 = re.compile(r"[0-9a-f]{40}")
EXPECTED_EMPTY_FIX_COMMIT = "5364e68d5225304f1f9b28d06f8200b9f462e0ff"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required physical JSON file is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"required JSON object is malformed: {path}")
    return value


def _flag(command: list[str], name: str) -> str:
    if command.count(name) != 1:
        raise RuntimeError(f"expected exactly one {name} in command")
    index = command.index(name)
    if index + 1 >= len(command) or not command[index + 1]:
        raise RuntimeError(f"{name} has no value")
    return str(command[index + 1])


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError):
        return False
    return True


def _open_writers(root: Path) -> list[dict[str, Any]]:
    writers: list[dict[str, Any]] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        fd_root = proc / "fd"
        try:
            fds = list(fd_root.iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                target = Path(os.readlink(fd))
                if not target.is_absolute() or not _inside(target, root):
                    continue
                flags_line = next(
                    line
                    for line in (proc / "fdinfo" / fd.name).read_text().splitlines()
                    if line.startswith("flags:")
                )
                flags = int(flags_line.split()[1], 8)
                if flags & os.O_ACCMODE in {os.O_WRONLY, os.O_RDWR}:
                    writers.append(
                        {"pid": int(proc.name), "fd": int(fd.name), "path": str(target)}
                    )
            except (OSError, StopIteration, ValueError):
                continue
    return writers


def _require_current_fixed_worktree(project: Path) -> str:
    commit = subprocess.check_output(
        ["git", "-C", str(project), "rev-parse", "HEAD"], text=True
    ).strip()
    ancestry = subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "merge-base",
            "--is-ancestor",
            EXPECTED_EMPTY_FIX_COMMIT,
            commit,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ancestry.returncode != 0:
        raise RuntimeError(
            "execution worktree does not contain the expected-empty fix: "
            f"{commit}"
        )
    dirty = subprocess.check_output(
        ["git", "-C", str(project), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise RuntimeError("execution worktree is dirty")
    fixed = project / "src/eval/bace_native_baseline_gnn.py"
    oracle = project / "src/oracles/gnn_oracle.py"
    fixed_source = fixed.read_text(encoding="utf-8")
    oracle_source = oracle.read_text(encoding="utf-8")
    if (
        "EXPECTED_EMPTY_GRAPH_SEQUENCE" not in fixed_source
        or "UNEXPECTED_EMPTY_GRAPH_SEQUENCE" not in fixed_source
        or "NO_EVALUABLE_GRAPHS_AFTER_PRE_ORACLE_FILTERS" not in oracle_source
        or "UNEXPECTED_EMPTY_GRAPH_SEQUENCE" not in oracle_source
    ):
        raise RuntimeError("expected-empty implementation is absent")
    return commit


def build(args: argparse.Namespace) -> dict[str, Any]:
    project = args.project_root.resolve(strict=True)
    execution_commit = _require_current_fixed_worktree(project)
    old_root = args.old_controller_root.resolve(strict=True)
    registry = args.registry_run_root.resolve(strict=True)
    candidate_root = args.candidate_output.resolve(strict=True)
    old_snapshot = _json(old_root / "controller_manifest.json")
    if old_snapshot.get("controller_id") != args.old_controller_id:
        raise RuntimeError("old controller_id mismatch")
    source_manifest_raw = old_snapshot.get("source_manifest")
    source_manifest_sha = old_snapshot.get("source_manifest_sha256")
    if not isinstance(source_manifest_raw, str) or not source_manifest_raw:
        raise RuntimeError("old controller lacks its physical source manifest")
    source_controller_manifest = Path(source_manifest_raw).resolve(strict=True)
    if _sha256(source_controller_manifest) != source_manifest_sha:
        raise RuntimeError("old controller source manifest SHA mismatch")
    load_controller_manifest(source_controller_manifest)

    old_tasks = {
        str(row.get("id")): row
        for row in old_snapshot.get("tasks", [])
        if isinstance(row, dict)
    }
    if TASK_ID not in old_tasks or CALIBRATION_TASK not in old_tasks or TEST_TASK not in old_tasks:
        raise RuntimeError("old controller lacks candidate/calibration/test tasks")
    old_candidate = old_tasks[TASK_ID]
    if old_candidate.get("depends_on") not in ([], None):
        raise RuntimeError("candidate is not an independent recovery root")

    registry_spec = _json(registry / "launch_spec.json")
    registry_state = _json(registry / "state.json")
    old_state = _json(old_root / "tasks" / TASK_ID / "state.json")
    old_gate = _json(old_root / "tasks" / TASK_ID / "gate.json")
    frozen_task = _json(old_root / "tasks" / TASK_ID / "manifest.json")
    instance = (old_state.get("instances") or {}).get("main")
    runs = old_gate.get("runs") or []
    if not isinstance(instance, dict) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise RuntimeError("candidate task instance/gate topology changed")
    exact_run_values = {
        instance.get("run_id"),
        runs[0].get("run_id"),
        registry_spec.get("run_id"),
        registry_state.get("run_id"),
    }
    if exact_run_values != {args.run_id}:
        raise RuntimeError(f"candidate run_id closure failed: {exact_run_values}")
    if (
        old_state.get("state") != "PASS"
        or old_gate.get("status") != "PASS"
        or instance.get("state") != "PASS"
        or runs[0].get("state") != "PASS"
        or registry_state.get("state") != "PASS"
    ):
        raise RuntimeError("candidate registry/controller state is not all PASS")
    if frozen_task.get("task_id") != TASK_ID or frozen_task.get("status") != "FROZEN":
        raise RuntimeError("candidate frozen task manifest mismatch")
    if registry_spec.get("expected_output") != str(candidate_root):
        raise RuntimeError("registry expected_output differs from candidate root")
    if instance.get("expected_output") != str(candidate_root):
        raise RuntimeError("controller instance output differs from candidate root")
    if registry_spec.get("input_hash") != _sha256(
        Path(str(registry_spec.get("input_manifest"))).resolve(strict=True)
    ):
        raise RuntimeError("candidate registry input hash mismatch")
    if registry_spec.get("dataset") != registry_state.get("dataset") or registry_spec.get(
        "stage"
    ) != registry_state.get("stage"):
        raise RuntimeError("candidate registry spec/state identity mismatch")
    if Path(str(registry_spec.get("python_executable"))).resolve(strict=True) != args.python.resolve(
        strict=True
    ):
        raise RuntimeError("candidate registry interpreter differs from successor")
    config_files = [
        Path(str(value)).resolve(strict=True)
        for value in registry_spec.get("config_files") or []
    ]
    if registry_spec.get("config_hash") != sha256_paths(config_files):
        raise RuntimeError("candidate registry config hash mismatch")
    log_path = Path(str(registry_state.get("log_path") or ""))
    marker = str(registry_spec.get("required_log_marker") or "")
    if (
        not marker
        or log_path.is_symlink()
        or not log_path.is_file()
        or marker not in log_path.read_text(encoding="utf-8", errors="replace")
    ):
        raise RuntimeError("candidate registry PASS log marker is unavailable")
    candidate_manifest = _json(candidate_root / "run_manifest.json")
    candidate_summary = _json(candidate_root / "summary.json")
    checkpoint_hash = str(
        candidate_summary.get("oracle_checkpoint_hash")
        or candidate_manifest.get("oracle_checkpoint_hash")
        or ""
    )
    if len(checkpoint_hash) != 64:
        raise RuntimeError("candidate oracle checkpoint hash is absent")
    validate_recovered_candidate_root(
        candidate_root, checkpoint_hash=checkpoint_hash, require_pass=True
    )
    if candidate_manifest.get("candidate_count") != 20:
        raise RuntimeError("candidate adoption is not the existing 20-rule pool")
    writers = _open_writers(candidate_root)
    if writers:
        raise RuntimeError(f"candidate output has active write descriptors: {writers}")

    old_candidate_command = [str(value) for value in registry_spec.get("command") or []]
    calibration_command = [str(value) for value in old_tasks[CALIBRATION_TASK]["command"]]
    test_command = [str(value) for value in old_tasks[TEST_TASK]["command"]]
    gnn_checkpoint = Path(_flag(old_candidate_command, "--gnn-checkpoint"))
    calibration_split = Path(_flag(calibration_command, "--split-path"))
    test_split = Path(_flag(test_command, "--split-path"))
    fresh = build_recovery_controller_fragment(
        python=args.python,
        project_root=project,
        output_root=args.output_root,
        failed_controller_root=Path(
            _flag(old_candidate_command, "--failed-controller-root")
        ),
        source_round_root=Path(_flag(old_candidate_command, "--source-round-root")),
        source_manifest=Path(_flag(old_candidate_command, "--source-manifest")),
        native_train_csv=Path(_flag(old_candidate_command, "--native-train-csv")),
        official_root=Path(_flag(old_candidate_command, "--official-root")),
        gnn_checkpoint=gnn_checkpoint,
        dataset_dir=calibration_split.parent,
        calibration_split=calibration_split,
        test_split=test_split,
        molclr_root=Path(_flag(calibration_command, "--molclr-root")),
        molclr_checkpoint=Path(_flag(calibration_command, "--molclr-checkpoint")),
        # GlobalGCE never consumes NeuroSED; this required generic-builder slot
        # is removed with preflight/bridge before the fragment is published.
        neurosed_checkpoint=gnn_checkpoint,
    )
    by_id = {str(row["id"]): row for row in fresh["tasks"]}
    adoption = by_id[TASK_ID]
    adoption.update(
        {
            "stage": str(old_candidate["stage"]),
            "runner_dataset": str(registry_spec["dataset"]),
            "runner_stage": str(registry_spec["stage"]),
            "depends_on": [],
            "resource": (
                "cpu"
                if registry_spec.get("gpu_index") is None
                and registry_spec.get("gpu_uuid") is None
                else "gpu"
            ),
            "command": [str(value) for value in registry_spec["command"]],
            "input_manifest": str(registry_spec["input_manifest"]),
            "expected_output": str(registry_spec["expected_output"]),
            "required_output_files": list(registry_spec["required_output_files"]),
            "required_output_any": list(registry_spec.get("required_output_any") or []),
            "required_absolute_output_files": list(
                registry_spec.get("required_absolute_output_files") or []
            ),
            "required_log_marker": str(registry_spec["required_log_marker"]),
            "environment": dict(registry_spec["environment"]),
            "config_files": list(registry_spec.get("config_files") or []),
            "adopt_existing_run_id": args.run_id,
            "adopt_project_root": str(registry_spec["project_root"]),
            "adopt_git_commit": str(registry_spec["git_commit"]),
            "adopt_max_gpus": int(registry_spec["max_gpus"]),
            "adopt_heavy": bool(registry_spec["heavy"]),
            "read_only_adoption": True,
            "retraining_forbidden": True,
        }
    )
    if HEX40.fullmatch(str(adoption["adopt_git_commit"])) is None:
        raise RuntimeError("adopted run git commit is not a full SHA")
    if adoption["resource"] == "gpu":
        adoption["adopt_gpu_index"] = int(registry_spec["gpu_index"])
        adoption["adopt_gpu_uuid"] = str(registry_spec["gpu_uuid"])
    else:
        adoption.pop("adopt_gpu_index", None)
        adoption.pop("adopt_gpu_uuid", None)

    forbidden = ("globalgce-train-rules", "gspan", "bridge-smoke")
    for task_id, task in by_id.items():
        if task_id == TASK_ID:
            continue
        body = json.dumps(task, sort_keys=True).lower()
        if any(token in body for token in forbidden):
            raise RuntimeError(f"fresh downstream task retained generation work: {task_id}")
    if any("ablation" in task_id.lower() for task_id in by_id):
        raise RuntimeError("GNN ablation appeared in calibration successor")
    if fresh.get("root_task_ids") != [TASK_ID]:
        raise RuntimeError("fresh successor root is not candidate adoption only")

    fragment_path = atomic_write_generic_fragment(args.fragment, fresh)
    composed = compose_manifest(
        controller_id=args.controller_id,
        fragments=[fragment_path],
        output=args.manifest,
    )
    loaded = load_controller_manifest(args.manifest)
    adopted = loaded.by_id[TASK_ID]
    if adopted.adopt_existing_run_id != args.run_id:
        raise RuntimeError("composed manifest lost candidate adoption")
    audit = {
        "schema_version": "bace_globalgce_calibration_successor_build_v1",
        "status": "PASS",
        "controller_id": args.controller_id,
        "execution_commit": execution_commit,
        "old_controller_id": args.old_controller_id,
        "old_controller_manifest": str(old_root / "controller_manifest.json"),
        "old_controller_manifest_sha256": _sha256(
            old_root / "controller_manifest.json"
        ),
        "candidate_run_id": args.run_id,
        "candidate_output": str(candidate_root),
        "candidate_universe_sha256": _sha256(
            candidate_root / "candidate_universe.jsonl"
        ),
        "candidate_rule_count": 20,
        "checkpoint_hash": checkpoint_hash,
        "registry_spec_sha256": _sha256(registry / "launch_spec.json"),
        "registry_state_sha256": _sha256(registry / "state.json"),
        "candidate_active_write_descriptors": [],
        "candidate_generation_replayed": False,
        "training_replayed": False,
        "gspan_replayed": False,
        "gnn_ablation_started": False,
        "manifest": str(args.manifest),
        "manifest_sha256": composed["manifest_sha256"],
        "fresh_output_root": str(args.output_root),
    }
    audit_path = args.manifest.with_suffix(args.manifest.suffix + ".build-audit.json")
    if audit_path.exists():
        raise FileExistsError(audit_path)
    from src.utils.autodl_runtime import atomic_write_json

    atomic_write_json(audit_path, audit)
    return {**audit, "build_audit": str(audit_path), "task_count": len(loaded.tasks)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help=argparse.SUPPRESS)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--old-controller-root", type=Path, required=True)
    parser.add_argument("--old-controller-id", required=True)
    parser.add_argument("--registry-run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--controller-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    result = build(parser.parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    print("[BACE_GLOBALGCE_CALIBRATION_SUCCESSOR_MANIFEST_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
