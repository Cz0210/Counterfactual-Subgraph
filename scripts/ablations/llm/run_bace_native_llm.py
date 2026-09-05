#!/usr/bin/env python3
"""Prepare BACE native LLM tasks or generate under an already-owned GPU lease.

This process never acquires a new GPU lock.  The existing resource owner passes
its held lock FD, keeps live resource evidence fresh, and retains T13 reservations.
SIGTERM/USR1 pauses only after a complete four-sequence call; no 120s bound is claimed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.llm.bace_native_runtime import run_generation, verified_file
from src.ablations.llm.bace_readiness import prepare_bace_llm
from src.ablations.llm.contracts import canonical_json_sha256
from src.ablations.gnn.early_policy import gpu_allowed
from src.ablations.gnn.scientific_verification import verify_package_archive


def resource_gate(evidence, held_fd):
    """Only validate the existing owner and inherited lock; no lock creation."""
    stamp = datetime.fromisoformat(evidence["observed_at"].replace("Z", "+00:00"))
    age = (datetime.now(timezone.utc) - stamp).total_seconds()
    if not 0 <= age <= 120:
        raise ValueError("Resource evidence must be refreshed by its existing owner within 120s")
    if evidence.get("gpu_lease_mode") != "EXCLUSIVE_IDLE":
        raise ValueError("Only normal idle exclusive leases are supported; borrowing is not implemented")
    lock_path = Path(evidence["gpu_lock_path"])
    if lock_path.is_symlink() or not lock_path.is_file():
        raise ValueError("Existing GPU lock disappeared")
    fd_stat, path_stat = os.fstat(held_fd), lock_path.stat()
    if (fd_stat.st_dev, fd_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino):
        raise ValueError("Inherited GPU lock FD differs from actual owner lock")
    # Opening the same inode via an independent file description must contend;
    # this proves the inherited descriptor corresponds to a held advisory lease.
    with lock_path.open("r+") as independent:
        try:
            fcntl.flock(independent, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(independent, fcntl.LOCK_UN)
            raise ValueError("The supplied GPU descriptor does not prove a held lock")
    metadata = json.loads(os.pread(held_fd, 65536, 0))
    if (metadata.get("state") != "LOCKED" or metadata.get("gpu_uuid") != evidence["gpu_uuid"]
            or metadata.get("gpu_index") != evidence["gpu_index"]
            or metadata.get("pid") != evidence["gpu_owner_pid"]
            or evidence["gpu_owner_pid"] not in (os.getpid(), os.getppid())):
        raise ValueError("Inherited GPU lock owner/UUID metadata differs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != evidence["gpu_uuid"]:
        raise ValueError("Exactly the leased physical GPU UUID must be visible")
    if evidence.get("target_gpu_uuid") != evidence["gpu_uuid"]:
        raise ValueError("Live arbiter target UUID differs")
    decision = gpu_allowed({**evidence, "gnn_core_seed7_audit": "PASS"}, family="llm")
    if not decision["allowed"]:
        raise ValueError("WAITING_RESOURCE: " + ",".join(decision["blockers"]))
    return True


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True)
    result.add_argument("--set", action="append", default=[], dest="overrides")
    commands = result.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--reference", required=True)
    prep.add_argument("--reference-sha256", required=True)
    prep.add_argument("--two-b-root", required=True)
    prep.add_argument("--brics-root", required=True)
    prep.add_argument("--output-root", required=True)
    prep.add_argument("--execution-commit", required=True)
    gen = commands.add_parser("generate")
    for name in ("task-spec", "task-spec-sha256", "output-root", "gnn-verified-archive",
                 "gnn-verified-archive-sha256", "resource-evidence", "resource-evidence-sha256"):
        gen.add_argument("--" + name, required=True)
    gen.add_argument("--held-gpu-lock-fd", required=True, type=int)
    gen.add_argument("--resume", action="store_true")
    gen.add_argument("--two-b-isolated-receipt")
    gen.add_argument("--two-b-isolated-receipt-sha256")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not Path(args.config).is_file() or "inference.fallback_to_heuristic=false" not in args.overrides:
        raise ValueError("Existing config and explicit no-heuristic setting required")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    if args.command == "prepare":
        if commit != args.execution_commit:
            raise ValueError("Preparation execution commit differs from actual checkout")
        result = prepare_bace_llm(reference_path=args.reference, reference_sha256=args.reference_sha256,
            two_b_root=args.two_b_root, brics_root=args.brics_root, output_root=args.output_root,
            execution_commit=commit)
    else:
        spec_path = verified_file({"path": args.task_spec, "sha256": args.task_spec_sha256})
        spec = json.loads(spec_path.read_text())
        body = {k: v for k, v in spec.items() if k != "task_spec_sha256"}
        if spec["task_spec_sha256"] != canonical_json_sha256(body) or spec["execution_commit"] != commit:
            raise ValueError("Task specification self hash/commit differs")
        archive = verified_file({"path": args.gnn_verified_archive, "sha256": args.gnn_verified_archive_sha256})
        if verify_package_archive(archive)["state"] != "PASS":
            raise ValueError("WAITING_GNN_CORE_SEED7")
        resource_path = verified_file({"path": args.resource_evidence, "sha256": args.resource_evidence_sha256})
        resource_gate(json.loads(resource_path.read_text()), args.held_gpu_lock_fd)
        if args.two_b_isolated_receipt or args.two_b_isolated_receipt_sha256:
            if not args.two_b_isolated_receipt or not args.two_b_isolated_receipt_sha256:
                raise ValueError("Both isolated receipt path and SHA are required")
            spec = {**spec, "isolated_cpu_load_receipt": {"path": args.two_b_isolated_receipt,
                    "sha256": args.two_b_isolated_receipt_sha256}}
        # Dynamic resource values are not scientific task fields or RNG inputs.
        # Stale/missing main evidence asks for a checkpoint pause, never a broad kill.
        def current_permission():
            try:
                return resource_gate(json.loads(resource_path.read_text()), args.held_gpu_lock_fd)
            except (ValueError, OSError, KeyError):
                return False
        result = run_generation(spec=spec, output_root=args.output_root, resume=args.resume,
                                continue_guard=current_permission)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if args.command == "prepare" or result["status"] == "CANDIDATE_POOL_PASS" else 75


if __name__ == "__main__":
    raise SystemExit(main())
