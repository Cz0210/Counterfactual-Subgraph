#!/usr/bin/env python3
"""Resume a sealed CPU epoch with corrected orchestration and the old engine."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def allowed_driver_change(path: str) -> bool:
    return path.startswith(("docs/", "tests/")) or path in {
        "src/ablations/gnn/cpu_training.py",
        "scripts/hpc/gnn/resume_bace_gnn_cpu.py",
        "scripts/hpc/gnn/status_bace_gnn_seed7.py",
        "scripts/slurm/resume_bace_gnn_cpu.sh",
    }


def validate_resume_binding(*, driver_root: Path, engine: Path, bundle: Path,
                            output: Path, expected_driver_commit: str,
                            expected_checkpoint_sha256: str) -> dict:
    """Check actual immutable code and checkpoint bytes without altering them."""
    contract = json.loads((output / "training_state/training_contract.json").read_text())
    old_source = contract["contract"]["source_identity"]
    if git(driver_root, "rev-parse", "HEAD") != expected_driver_commit or git(driver_root, "status", "--porcelain"):
        raise ValueError("Resume driver must be its clean immutable commit")
    if git(engine, "rev-parse", "HEAD") != old_source["commit"] or git(engine, "rev-parse", "HEAD^{tree}") != old_source["tree"]:
        raise ValueError("Engine worktree does not match the sealed training source")
    if git(engine, "status", "--porcelain") or old_source["status_short"]:
        raise ValueError("The original scientific engine must remain clean")
    for item in old_source["tracked_source_files"]:
        if sha256(engine / item["path"]) != item["sha256"]:
            raise ValueError("An original trainer source file changed")
    changed = git(driver_root, "diff", "--name-only", old_source["commit"], expected_driver_commit).splitlines()
    if any(not allowed_driver_change(path) for path in changed):
        raise ValueError("Resume driver commit contains a scientific-engine change")
    latest_path = output / "training_state/latest_checkpoint.json"
    latest = json.loads(latest_path.read_text())
    checkpoint_name = latest["checkpoint_file"]
    if Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("Checkpoint path escapes its training state root")
    checkpoint = output / "training_state" / checkpoint_name
    if (latest["status"] != "CHECKPOINT_COMPLETE"
            or latest["contract_sha256"] != contract["contract_sha256"]
            or latest["checkpoint_sha256"] != expected_checkpoint_sha256
            or sha256(checkpoint) != expected_checkpoint_sha256
            or checkpoint.stat().st_size != latest["checkpoint_bytes"]):
        raise ValueError("Sealed epoch checkpoint differs from the requested resume boundary")
    cpu = json.loads((output / "cpu_contract.json").read_text())
    if sha256(output / "effective_config.yaml") != cpu["effective_config_sha256"]:
        raise ValueError("Sealed effective config bytes changed")
    if sha256(bundle / "bundle_manifest.json") != cpu["bundle_manifest_sha256"]:
        raise ValueError("The original input bundle changed")
    return {
        "schema_version": "bace_gnn_cpu_resume_driver_v1", "status": "VERIFIED",
        "driver_commit": expected_driver_commit,
        "driver_sha256": sha256(driver_root / "scripts/hpc/gnn/resume_bace_gnn_cpu.py"),
        "cpu_wrapper_sha256": sha256(driver_root / "src/ablations/gnn/cpu_training.py"),
        "scientific_engine_commit": old_source["commit"], "scientific_engine_root": str(engine),
        "scientific_engine_source_identity": old_source,
        "allowed_changed_paths": changed, "input_bundle_manifest_sha256": cpu["bundle_manifest_sha256"],
        "checkpoint_sha256": expected_checkpoint_sha256, "completed_epoch": latest["completed_epoch"],
        "checkpoint_path": str(checkpoint), "training_contract_sha256": contract["contract_sha256"],
        "effective_config_sha256": cpu["effective_config_sha256"],
        "sealed_input_bytes_modified": False, "benchmark_rerun": False,
        "main_matrix_write_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-worktree", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--backbone", required=True, choices=("gin", "gcn", "gatv2", "gatedgcn_plus"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--expected-driver-commit", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    driver_root = Path(__file__).resolve().parents[3]
    engine = Path(args.engine_worktree).resolve(strict=True)
    bundle = Path(args.bundle_root).resolve(strict=True)
    output = Path(args.output_root).resolve(strict=True)
    receipt_path = Path(args.receipt).resolve()
    if receipt_path.exists():
        raise FileExistsError("Resume driver receipt must be fresh")
    receipt = validate_resume_binding(driver_root=driver_root, engine=engine, bundle=bundle,
        output=output, expected_driver_commit=args.expected_driver_commit,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256)
    if any(name == "src" or name.startswith("src.") or name == "scripts" or name.startswith("scripts.") for name in sys.modules):
        raise RuntimeError("Resume driver requires a fresh process without preloaded project modules")
    # Imports and trainer.PROJECT_ROOT genuinely resolve to the old clean tree.
    # The corrected wrapper alone is loaded explicitly from the new commit.
    os.chdir(engine)
    sys.path.insert(0, str(engine))
    os.environ["PYTHONPATH"] = str(engine)
    spec = importlib.util.spec_from_file_location("bace_gnn_cpu_resume_implementation", driver_root / "src/ablations/gnn/cpu_training.py")
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    from scripts import train_molecular_gnn as trainer
    if Path(trainer.__file__).resolve() != engine / "scripts/train_molecular_gnn.py":
        raise RuntimeError("The original scientific trainer did not load from its pinned worktree")
    if trainer._git_state() != receipt["scientific_engine_source_identity"]:
        raise ValueError("Loaded original trainer does not reproduce its sealed source identity")
    implementation.atomic_json(receipt_path, receipt)
    result = implementation.run_cpu_training(bundle_root=bundle, backbone=args.backbone,
        phase="train", output_root=output, config_path=args.config,
        cpu_threads=args.cpu_threads, resume=True)
    implementation.atomic_json(receipt_path.with_name(receipt_path.stem + "-terminal.json"), {**receipt, "result": result})
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
