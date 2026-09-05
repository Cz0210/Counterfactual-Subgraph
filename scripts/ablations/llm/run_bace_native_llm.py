#!/usr/bin/env python3
"""Prepare BACE native LLM tasks or generate under an already-owned GPU lease.

This process never acquires a new GPU lock.  The existing resource owner passes
its held lock FD, keeps live resource evidence fresh, and retains T13 reservations.
SIGTERM/USR1 pauses only after a complete four-sequence call; no 120s bound is claimed.
"""
from __future__ import annotations

import argparse
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
from src.ablations.llm.corrected_core_gate import require_corrected_gnn_core
from src.ablations.llm.existing_gpu_owner import validate_inherited_lease, seal_held_descriptors


def resource_gate(evidence, held_fd, slot_fd=None):
    """Only validate the existing owner and inherited lock; no lock creation."""
    return validate_inherited_lease(evidence, held_fd, slot_fd)


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
    prep.add_argument("--two-b-isolated-receipt")
    prep.add_argument("--two-b-isolated-receipt-sha256")
    gen = commands.add_parser("generate")
    for name in ("task-spec", "task-spec-sha256", "output-root", "gnn-verified-archive",
                 "gnn-verified-archive-sha256", "resource-evidence", "resource-evidence-sha256"):
        gen.add_argument("--" + name, required=True)
    gen.add_argument("--held-gpu-lock-fd", required=True, type=int)
    gen.add_argument("--held-project-slot-fd", required=True, type=int)
    gen.add_argument("--resource-live-evidence")
    gen.add_argument("--gnn-acceptance")
    gen.add_argument("--gnn-acceptance-sha256")
    gen.add_argument("--resume", action="store_true")
    gen.add_argument("--two-b-isolated-receipt")
    gen.add_argument("--two-b-isolated-receipt-sha256")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "generate":
        # Keep both leases for this process, but never leak them into an exec'd
        # tokenizer/helper/grandchild, even one using close_fds=False.
        seal_held_descriptors(args.held_gpu_lock_fd, args.held_project_slot_fd)
    if not Path(args.config).is_file() or "inference.fallback_to_heuristic=false" not in args.overrides:
        raise ValueError("Existing config and explicit no-heuristic setting required")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    if args.command == "prepare":
        if commit != args.execution_commit:
            raise ValueError("Preparation execution commit differs from actual checkout")
        proof = None
        if args.two_b_isolated_receipt or args.two_b_isolated_receipt_sha256:
            if not args.two_b_isolated_receipt or not args.two_b_isolated_receipt_sha256:
                raise ValueError("Both isolated receipt path and SHA are required")
            proof = {"path": args.two_b_isolated_receipt, "sha256": args.two_b_isolated_receipt_sha256}
        result = prepare_bace_llm(reference_path=args.reference, reference_sha256=args.reference_sha256,
            two_b_root=args.two_b_root, brics_root=args.brics_root, output_root=args.output_root,
            execution_commit=commit, two_b_isolated_receipt=proof)
    else:
        spec_path = verified_file({"path": args.task_spec, "sha256": args.task_spec_sha256})
        spec = json.loads(spec_path.read_text())
        body = {k: v for k, v in spec.items() if k != "task_spec_sha256"}
        if spec["task_spec_sha256"] != canonical_json_sha256(body) or spec["execution_commit"] != commit:
            raise ValueError("Task specification self hash/commit differs")
        acceptance = None
        if args.gnn_acceptance or args.gnn_acceptance_sha256:
            if not args.gnn_acceptance or not args.gnn_acceptance_sha256:
                raise ValueError("Corrective acceptance path and SHA are both required")
            acceptance = {"path": args.gnn_acceptance, "sha256": args.gnn_acceptance_sha256}
        require_corrected_gnn_core(args.gnn_verified_archive, args.gnn_verified_archive_sha256, acceptance=acceptance)
        resource_path = verified_file({"path": args.resource_evidence, "sha256": args.resource_evidence_sha256})
        initial_evidence = json.loads(resource_path.read_text())
        resource_gate(initial_evidence, args.held_gpu_lock_fd, args.held_project_slot_fd)
        if args.resource_live_evidence:
            resource_path = Path(args.resource_live_evidence)
            current = json.loads(resource_path.read_text())
            if current.get("owner_nonce") != initial_evidence["owner_nonce"]:
                raise ValueError("DYNAMIC_RESOURCE_OWNER_CHANGED")
            resource_gate(current, args.held_gpu_lock_fd, args.held_project_slot_fd)
        if args.two_b_isolated_receipt or args.two_b_isolated_receipt_sha256:
            if not args.two_b_isolated_receipt or not args.two_b_isolated_receipt_sha256:
                raise ValueError("Both isolated receipt path and SHA are required")
            proof = {"path": args.two_b_isolated_receipt, "sha256": args.two_b_isolated_receipt_sha256}
            if proof != spec.get("isolated_cpu_load_receipt"):
                raise ValueError("Isolated proof must be frozen by prepare, never injected into a sealed task")
        destination = Path(args.output_root).absolute()
        scope = Path(spec["output_scope_root"])
        if (not scope.is_absolute() or scope not in destination.parents
                or any(parent.is_symlink() for parent in (destination, *destination.parents))):
            raise ValueError("Native generation must stay within the dedicated physical LLM output scope")
        if args.resume:
            # Check before opening/creating the writer lock: an incorrect main
            # or unrelated ablation root must not be touched even on failure.
            latest = json.loads((destination / "latest_checkpoint.json").read_text())
            if latest["spec_sha256"] != canonical_json_sha256(spec):
                raise ValueError("Requested resume root belongs to another task")
        # Dynamic resource values are not scientific task fields or RNG inputs.
        # Stale/missing main evidence asks for a checkpoint pause, never a broad kill.
        def current_permission():
            try:
                current = json.loads(resource_path.read_text())
                if current.get("owner_nonce") != initial_evidence["owner_nonce"]:
                    return False
                return resource_gate(current, args.held_gpu_lock_fd, args.held_project_slot_fd)
            except (ValueError, OSError, KeyError):
                return False
        result = run_generation(spec=spec, output_root=args.output_root, resume=args.resume,
                                continue_guard=current_permission)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if args.command == "prepare" or result["status"] == "CANDIDATE_POOL_PASS" else 75


if __name__ == "__main__":
    raise SystemExit(main())
