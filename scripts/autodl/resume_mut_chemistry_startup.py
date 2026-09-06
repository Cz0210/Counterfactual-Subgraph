#!/usr/bin/env python3
"""One explicit control-only repair of the sealed Mut chemistry startup boundary.

Execute the original pinned runner and scientific commands, with independent
startup checkpoints and failure diagnostics in a fresh control root. This is
not a partial-evaluation resume or authorization to repeat any scientific stage.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC_SCHEMA = "mut_independent_evaluation_continuation_v1"
REPAIR_SCHEMA = "mut_chemistry_startup_control_repair_v1"
FAILURE = "CONTINUATION_PREVIOUS_STAGE_STATE_INVALID:unified_eval"
STAGES = ("unified_eval", "full_gate", "freeze")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, timeout=30
    ).strip()


def _physical(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.resolve(strict=False) != path:
        raise ValueError("MUT_STARTUP_REPAIR_PHYSICAL_ABSOLUTE_PATH_REQUIRED")
    return path


def _load_runner(source_root: Path):
    # The adapter imports no project packages before selecting the original
    # checkout. Never mix the new control-driver packages with old science.
    for name in ("src", "scripts"):
        loaded = sys.modules.get(name)
        if loaded is not None:
            locations = list(getattr(loaded, "__path__", ()))
            if any(not Path(p).resolve().is_relative_to(source_root) for p in locations):
                raise ValueError("MUT_STARTUP_REPAIR_MIXED_IMPORT_TREES")
    sys.path.insert(0, str(source_root))
    return importlib.import_module("scripts.autodl.run_mut_comrecgc_parity_standardization")


def _install_control_adapter(runner, *, root: Path, recovery: Path,
                             boundary: dict[str, Any], old_stage: dict[str, Any]):
    original_stage, original_write = runner._run_stage, runner.write_json
    pending = list(boundary["contract"]["commands"][1:])
    if [row[0] for row in pending] != list(STAGES):
        raise ValueError("MUT_STARTUP_REPAIR_STAGE_SEQUENCE_CHANGED")

    def stage(**kwargs):
        if not pending:
            raise ValueError("MUT_STARTUP_REPAIR_UNEXPECTED_STAGE")
        expected = pending[0]
        actual = [kwargs["stage"], list(kwargs["argv"]),
                  str(kwargs["marker"]), kwargs["required_field"]]
        if (actual != expected or kwargs["output_root"] != root
                or kwargs.get("checkpoint_path") is not None):
            raise ValueError("MUT_STARTUP_REPAIR_SCIENTIFIC_COMMAND_CHANGED")
        if kwargs["stage"] == "unified_eval":
            common = importlib.import_module(
                "scripts.autodl.run_comrecgc_standardized_continuation")
            common._wait_for_process_group_quiescence(
                int(old_stage["process_group_id"]), proc_root=common._PROC_ROOT,
                timeout_seconds=30.0)
        result = original_stage(
            **{**kwargs, "output_root": recovery,
               "checkpoint_path": recovery / "stage_checkpoints" / (kwargs["stage"] + ".json")})
        pending.pop(0)
        return result

    def write(path, payload):
        # Preserve the exact old startup failure even if a genuinely new
        # evaluation error occurs. No scientific output path is redirected.
        target = recovery / "runner_FAILED.json" if Path(path) == root / "FAILED.json" else path
        return original_write(target, payload)

    runner._run_stage, runner.write_json = stage, write
    return original_stage, original_write


def run(*, continuation_spec: Path, recovery_root: Path,
        expected_driver_commit: str) -> int:
    continuation_spec, recovery_root = map(_physical, (continuation_spec, recovery_root))
    spec = json.loads(continuation_spec.read_text())
    source_root = _physical(spec["cwd"])
    if (_git(PROJECT_ROOT, "rev-parse", "HEAD") != expected_driver_commit
            or _git(source_root, "rev-parse", "HEAD") != spec["execution_commit"]):
        raise ValueError("MUT_STARTUP_REPAIR_COMMIT_CHANGED")
    for tree in (PROJECT_ROOT, source_root):
        if _git(tree, "status", "--porcelain", "--untracked-files=all", "--", "scripts", "src", "configs"):
            raise ValueError("MUT_STARTUP_REPAIR_EXECUTION_TREE_DIRTY")
    runner = _load_runner(source_root)
    if runner.PROJECT_ROOT != source_root:
        raise ValueError("MUT_STARTUP_REPAIR_RUNNER_TREE_CHANGED")
    sha = runner.stable_json_sha256
    command = spec["command"]
    if (spec.get("schema_version") != SPEC_SCHEMA
            or spec.get("self_sha256") != sha({k: v for k, v in spec.items() if k != "self_sha256"})
            or spec.get("argv_sha256") != sha(command)
            or command[:4] != [str(Path(sys.executable)), "-I", "-B",
                str(source_root / "scripts/autodl/run_mut_comrecgc_parity_standardization.py")]
            or "--resume-after-chemistry" not in command[4:]):
        raise ValueError("MUT_STARTUP_REPAIR_SEALED_INVOCATION_CHANGED")
    args = runner.build_parser().parse_args(command[4:])
    root = _physical(spec["output_root"])
    if (args.output_root != root or args.through_stage != "all"
            or not args.resume_after_chemistry or args.device != "cpu"
            or args.historical_adoption is None or args.stage_resource_config is None
            or args.persistent_root is None or recovery_root.is_relative_to(root)
            or root.is_relative_to(recovery_root)):
        raise ValueError("MUT_STARTUP_REPAIR_SCOPE_CHANGED")
    resource = spec["resource_config"]
    if (str(args.stage_resource_config) != resource["path"]
            or runner.sha256_file(args.stage_resource_config) != resource["sha256"]):
        raise ValueError("MUT_STARTUP_REPAIR_RESOURCE_CONFIG_CHANGED")
    snapshots = {name: (root / name).read_bytes() for name in (
        "chemistry_stage_boundary.json", "stage_state.json", "FAILED.json")}
    boundary, old_stage, failure = [json.loads(data) for data in snapshots.values()]
    contract = boundary["contract"]
    if (contract["project_commit"] != spec["execution_commit"]
            or old_stage.get("schema_version") != 2 or old_stage.get("stage") != "chemistry"
            or old_stage.get("status") != "PASS" or int(old_stage.get("process_group_id", -1)) <= 1
            or old_stage.get("argv_sha256") != sha(contract["commands"][0][1])
            or failure.get("message") != FAILURE or failure.get("error_class") != "ValueError"
            or failure.get("status") != "FAILED" or failure.get("output_root") != str(root)):
        raise ValueError("MUT_STARTUP_REPAIR_NOT_EXACT_PRE_SCIENCE_FAILURE")
    boundary_module = importlib.import_module("src.utils.mut_chemistry_stage_boundary")
    boundary_module.validate_boundary(root, contract)
    recovery_root.mkdir(parents=True, exist_ok=False)
    for name, data in snapshots.items():
        runner.atomic_write_bytes(recovery_root / "preserved" / name, data)
    receipt = dict(
        schema_version=REPAIR_SCHEMA, state="CONTROL_ADAPTER_READY_NOT_SCIENCE_PASS",
        control_driver_commit=expected_driver_commit,
        control_driver_path=str(Path(__file__).resolve()),
        control_driver_sha256=runner.sha256_file(Path(__file__)),
        scientific_project_commit=spec["execution_commit"], scientific_source_root=str(source_root),
        continuation_spec_path=str(continuation_spec),
        continuation_spec_sha256=runner.sha256_file(continuation_spec),
        continuation_spec_self_sha256=spec["self_sha256"],
        original_argv_sha256=spec["argv_sha256"], output_root=str(root),
        control_output_root=str(recovery_root),
        preserved={name: runner.sha256_file(recovery_root / "preserved" / name) for name in snapshots},
        scientific_argv_changed=False, scientific_import_tree_changed=False,
        source_generation_rerun=False, common_recourse_rerun=False, chemistry_rerun=False,
        original_stage_state_rewritten=False, original_failure_rewritten=False,
        control_changes=["_run_stage.output_root", "_run_stage.checkpoint_path",
                         "runner.write_json:exact_prior_FAILED.json_only"],
        stage_checkpoint_paths={s: str(recovery_root / "stage_checkpoints" / (s + ".json")) for s in STAGES})
    receipt["self_sha256"] = sha(receipt)
    runner.write_json(recovery_root / "control_adapter_receipt.json", receipt)
    old = _install_control_adapter(runner, root=root, recovery=recovery_root,
                                   boundary=boundary, old_stage=old_stage)
    try:
        os.chdir(source_root)
        os.environ.update(spec["environment"])
        # Original main owns its original invocation lease, revalidates the
        # complete chemistry contract and makes fresh resource admission.
        return runner.main(command[4:])
    finally:
        runner._run_stage, runner.write_json = old


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--continuation-spec", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--expected-driver-commit", required=True)
    args = parser.parse_args(argv)
    if args.config != "configs/hpc.yaml" or args.set not in ([], ["inference.fallback_to_heuristic=false"]):
        raise ValueError("MUT_STARTUP_REPAIR_UNSUPPORTED_CONFIG_OVERRIDE")
    return run(continuation_spec=args.continuation_spec, recovery_root=args.recovery_root,
               expected_driver_commit=args.expected_driver_commit)


if __name__ == "__main__":
    raise SystemExit(main())
