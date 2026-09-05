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
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.ablations.llm.bace_native_runtime import VARIANTS, verified_file
from src.ablations.llm.contracts import canonical_json_sha256
from src.ablations.llm.corrected_core_gate import require_corrected_gnn_core, archive_identity, adopt_existing_acceptance
from src.eval.bace_frozen_gnn_contracts import sha256_file, atomic_json
from src.ablations.llm.existing_gpu_owner import DISPATCH_SCHEMA, receive_owner_binding

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
            if receipt.get("variant") != variant or receipt.get("spec_sha256") != canonical_json_sha256(spec):
                raise ValueError("SUCCESSOR_COMPLETED_VARIANT_BINDING_MISMATCH")
            if receipt.get("status") == "CANDIDATE_POOL_PASS":
                if (receipt.get("candidate_pool_sha256") != sha256_file(destination / "candidate_pool.jsonl")
                        or receipt.get("next_call") != len(spec["calls"])):
                    raise ValueError("SUCCESSOR_COMPLETED_VARIANT_BINDING_MISMATCH")
                continue
            if receipt.get("status") != "PAUSED_AT_CALL_CHECKPOINT" or not (destination / "latest_checkpoint.json").is_file():
                raise ValueError("SUCCESSOR_TERMINAL_NOT_COMPLETE_OR_RESUMABLE")
            checkpoint = json.loads((destination / "latest_checkpoint.json").read_text())
            if checkpoint.get("spec_sha256") != receipt["spec_sha256"] or checkpoint.get("next_call") != receipt.get("next_call"):
                raise ValueError("SUCCESSOR_PAUSED_CHECKPOINT_BINDING_MISMATCH")
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
    parser.add_argument("--held-project-slot-fd", type=int)
    parser.add_argument("--resource-live-evidence")
    parser.add_argument("--gnn-acceptance")
    parser.add_argument("--gnn-acceptance-sha256")
    parser.add_argument("--seal-dispatch-spec", help="CPU-only fresh file; does not acquire a GPU")
    parser.add_argument("--resource-config", help="Frozen paths/thresholds for direct live-source sampling")
    for name in ("adopt-corrective-overlay", "adopt-corrective-overlay-sha256", "adopt-corrective-audit", "adopt-corrective-audit-sha256"):
        parser.add_argument("--" + name)
    args = parser.parse_args(argv)
    if "AUTODL_LLM_OWNER_BOOTSTRAP_FD" in os.environ:
        # Consume/close the inherited bootstrap and set both lease FDs CLOEXEC
        # before importing model/verifier code or running any helper process.
        binding = receive_owner_binding()
        for field in ("held_gpu_lock_fd", "held_project_slot_fd", "resource_evidence", "resource_evidence_sha256", "resource_live_evidence"):
            if getattr(args, field) is not None:
                raise ValueError("OWNER_BOUND_FIELDS_MUST_NOT_BE_OVERRIDDEN")
            setattr(args, field, binding[field])
    if args.config != "configs/hpc.yaml" or set(args.set) - {"inference.fallback_to_heuristic=false"}:
        parser.error("Use configs/hpc.yaml without scientific overrides")
    acceptance = None
    if args.gnn_acceptance or args.gnn_acceptance_sha256:
        if not args.gnn_acceptance or not args.gnn_acceptance_sha256:
            raise ValueError("Corrected acceptance requires path and SHA")
        acceptance = {"path": args.gnn_acceptance, "sha256": args.gnn_acceptance_sha256}
    adopted_fields = (args.adopt_corrective_overlay, args.adopt_corrective_overlay_sha256,
                      args.adopt_corrective_audit, args.adopt_corrective_audit_sha256)
    if any(adopted_fields):
        if not all(adopted_fields) or not args.seal_dispatch_spec or acceptance is not None:
            raise ValueError("Import acceptance adoption requires all four fields and fresh dispatch sealing")
        target = Path(args.seal_dispatch_spec).absolute()
        if target.exists():
            raise ValueError("Dispatch destination must be fresh")
        acceptance = adopt_existing_acceptance(archive_path=args.gnn_verified_archive,
            archive_sha256=args.gnn_verified_archive_sha256,
            overlay={"path": args.adopt_corrective_overlay, "sha256": args.adopt_corrective_overlay_sha256},
            audit={"path": args.adopt_corrective_audit, "sha256": args.adopt_corrective_audit_sha256},
            output_path=target.with_name(target.stem + ".gnn_acceptance.json"))
    proof = require_corrected_gnn_core(args.gnn_verified_archive, args.gnn_verified_archive_sha256,
                                       acceptance=acceptance)
    result = next_task(args.readiness, args.readiness_sha256, args.output_root)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.seal_dispatch_spec:
        target = Path(args.seal_dispatch_spec).absolute()
        if target.exists() or not args.resource_config:
            raise ValueError("Fresh dispatch destination and real resource config required")
        from src.ablations.llm.existing_gpu_owner import validate_resource_config, read_small
        config, _ = read_small(args.resource_config)
        validate_resource_config(config)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        readiness = json.loads(verified_file({"path": args.readiness, "sha256": args.readiness_sha256}).read_text())
        for variant in ORDER:
            task = json.loads(verified_file(readiness["variants"][variant]).read_text())
            if task.get("execution_commit") != commit:
                raise ValueError("FRESH_PREPARE_REQUIRED_FOR_CURRENT_EXECUTION_COMMIT")
        if acceptance is None:
            accepted = target.with_name(target.stem + ".gnn_acceptance.json")
            if accepted.exists():
                raise ValueError("Acceptance destination must be fresh")
            atomic_json(accepted, {"schema_version": "bace_llm_corrected_core_acceptance_v1",
                        "archive_identity": archive_identity(args.gnn_verified_archive),
                        "archive_sha256": args.gnn_verified_archive_sha256, "independent_audit": proof})
            acceptance = {"path": str(accepted), "sha256": sha256_file(accepted)}
        command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--config", "configs/hpc.yaml",
                   "--set", "inference.fallback_to_heuristic=false", "--readiness", args.readiness,
                   "--readiness-sha256", args.readiness_sha256, "--output-root", args.output_root,
                   "--gnn-verified-archive", args.gnn_verified_archive,
                   "--gnn-verified-archive-sha256", args.gnn_verified_archive_sha256,
                   "--gnn-acceptance", acceptance["path"], "--gnn-acceptance-sha256", acceptance["sha256"]]
        payload = {"schema_version": DISPATCH_SCHEMA, "execution_commit": commit, "command": command,
                   "resource_config": {"path": str(Path(args.resource_config).absolute()), "sha256": sha256_file(args.resource_config)},
                   "readiness": {"path": args.readiness, "sha256": args.readiness_sha256},
                   "variant_order": list(ORDER), "max_llm_gpus": 1, "borrow_enabled": False,
                   "state": "DISPATCHABLE_WAITING_RESOURCE", "science_started": False,
                   "main_matrix_count_required": False, "secondary_seeds_required": False}
        payload["self_sha256"] = canonical_json_sha256(payload)
        atomic_json(target, payload)
        print(json.dumps({"dispatch_spec": str(target), "sha256": sha256_file(target), "state": payload["state"]}))
        return 0
    if args.plan_only or result["state"] != "READY_WAITING_EXISTING_IDLE_GPU_OWNER":
        return 0 if args.plan_only or result["state"] == "ALL_THREE_GENERATION_POOLS_COMPLETE" else 75
    if args.held_gpu_lock_fd is None or args.held_project_slot_fd is None or not args.resource_evidence or not args.resource_evidence_sha256:
        raise ValueError("WAITING_EXISTING_GPU_OWNER_NO_BORROW_OR_LEASE_CREATION")
    command = [sys.executable, "-I", "-B", str(ROOT / "scripts/ablations/llm/run_bace_native_llm.py"),
        "--config", "configs/hpc.yaml", "--set", "inference.fallback_to_heuristic=false", "generate",
        "--task-spec", result["task_spec"], "--task-spec-sha256", result["task_spec_sha256"],
        "--output-root", result["output_root"], "--gnn-verified-archive", args.gnn_verified_archive,
        "--gnn-verified-archive-sha256", args.gnn_verified_archive_sha256,
        "--resource-evidence", args.resource_evidence,
        "--resource-evidence-sha256", args.resource_evidence_sha256,
        "--held-gpu-lock-fd", str(args.held_gpu_lock_fd), "--held-project-slot-fd", str(args.held_project_slot_fd)]
    if args.resource_live_evidence:
        command.extend(["--resource-live-evidence", args.resource_live_evidence])
    if acceptance:
        command.extend(["--gnn-acceptance", acceptance["path"], "--gnn-acceptance-sha256", acceptance["sha256"]])
    if result["resume"]: command.append("--resume")
    # Same process/parent identity reaches the existing lock-owner validator.
    os.set_inheritable(args.held_gpu_lock_fd, True)
    os.set_inheritable(args.held_project_slot_fd, True)
    os.execv(command[0], command)


if __name__ == "__main__": raise SystemExit(main())
