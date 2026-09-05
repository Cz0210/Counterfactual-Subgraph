#!/usr/bin/env python3
"""Execute the next L1 -> L2 -> L3 generator under an existing owner lease.

This is a one-shot exec adapter, not a scheduler. It does not acquire leases,
borrow a reserved GPU, or start a fourth variant. The owner's fresh evidence
and inherited FD are rechecked by the real native generator at each checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.ablations.llm.bace_native_runtime import VARIANTS, verified_file
from src.ablations.llm.contracts import canonical_json_sha256
from src.ablations.llm.corrected_core_gate import require_corrected_gnn_core
from src.eval.bace_frozen_gnn_contracts import sha256_file

ORDER = VARIANTS[1:]


def next_task(readiness_path, readiness_sha256, output_root):
    readiness = json.loads(verified_file({"path": str(readiness_path), "sha256": readiness_sha256}).read_text())
    if readiness.get("schema_version") != "bace_llm_native_readiness_v1":
        raise ValueError("READINESS_SCHEMA_MISMATCH")
    root = Path(output_root).absolute()
    for variant in ORDER:
        descriptor = readiness["variants"][variant]
        spec_file = verified_file(descriptor)
        spec = json.loads(spec_file.read_text())
        if (spec.get("variant") != variant or spec.get("task_spec_sha256") != canonical_json_sha256({
                k: v for k, v in spec.items() if k != "task_spec_sha256"})):
            raise ValueError("SUCCESSOR_VARIANT_SPEC_MISMATCH")
        destination = root / variant
        terminal = destination / "candidate_generation_receipt.json"
        if terminal.is_file():
            receipt = json.loads(terminal.read_text())
            if (receipt.get("status") != "CANDIDATE_POOL_PASS" or receipt.get("variant") != variant
                    or receipt.get("spec_sha256") != canonical_json_sha256(spec)
                    or receipt.get("candidate_pool_sha256") != sha256_file(destination / "candidate_pool.jsonl")
                    or receipt.get("next_call") != len(spec["calls"])):
                raise ValueError("SUCCESSOR_COMPLETED_VARIANT_BINDING_MISMATCH")
            continue
        if spec.get("generator_state", "").startswith("BLOCKED_"):
            return {"state": spec["generator_state"], "variant": variant, "blocker": spec.get("blocker")}
        resume = (destination / "latest_checkpoint.json").is_file()
        if destination.exists() and not resume:
            raise ValueError("SUCCESSOR_OUTPUT_EXISTS_WITHOUT_COMMITTED_CHECKPOINT")
        return {"state": "READY_WAITING_EXISTING_IDLE_GPU_OWNER", "variant": variant,
                "task_spec": str(spec_file), "task_spec_sha256": descriptor["sha256"],
                "output_root": str(destination), "resume": resume}
    return {"state": "ALL_THREE_GENERATION_POOLS_COMPLETE", "gpu_requested": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    for name in ("readiness", "readiness-sha256", "output-root", "gnn-verified-archive",
                 "gnn-verified-archive-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resource-evidence")
    parser.add_argument("--resource-evidence-sha256")
    parser.add_argument("--held-gpu-lock-fd", type=int)
    args = parser.parse_args(argv)
    if args.config != "configs/hpc.yaml" or set(args.set) - {"inference.fallback_to_heuristic=false"}:
        parser.error("Use configs/hpc.yaml without scientific overrides")
    require_corrected_gnn_core(args.gnn_verified_archive, args.gnn_verified_archive_sha256)
    result = next_task(args.readiness, args.readiness_sha256, args.output_root)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.plan_only or result["state"] != "READY_WAITING_EXISTING_IDLE_GPU_OWNER":
        return 0 if args.plan_only or result["state"] == "ALL_THREE_GENERATION_POOLS_COMPLETE" else 75
    if args.held_gpu_lock_fd is None or not args.resource_evidence or not args.resource_evidence_sha256:
        raise ValueError("WAITING_EXISTING_GPU_OWNER_NO_BORROW_OR_LEASE_CREATION")
    command = [sys.executable, "-I", "-B", str(ROOT / "scripts/ablations/llm/run_bace_native_llm.py"),
        "--config", "configs/hpc.yaml", "--set", "inference.fallback_to_heuristic=false", "generate",
        "--task-spec", result["task_spec"], "--task-spec-sha256", result["task_spec_sha256"],
        "--output-root", result["output_root"], "--gnn-verified-archive", args.gnn_verified_archive,
        "--gnn-verified-archive-sha256", args.gnn_verified_archive_sha256,
        "--resource-evidence", args.resource_evidence,
        "--resource-evidence-sha256", args.resource_evidence_sha256,
        "--held-gpu-lock-fd", str(args.held_gpu_lock_fd)]
    if result["resume"]: command.append("--resume")
    # Same process/parent identity reaches the existing lock-owner validator.
    os.set_inheritable(args.held_gpu_lock_fd, True)
    os.execv(command[0], command)


if __name__ == "__main__": raise SystemExit(main())
