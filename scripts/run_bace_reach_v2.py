#!/usr/bin/env python3
"""Explicit BACE Reach-v2 plan/search/calibration stages; no automatic test."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--action", required=True, choices=["plan", "canary", "train-search", "calibrate", "status", "owner", "resource-overlay"])
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--proposal-source", choices=["OURS_MAIN_PPO_66", "L0", "L1", "L2", "L3"], default="OURS_MAIN_PPO_66")
    p.add_argument("--proposal-path", type=Path)
    p.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    p.add_argument("--resource-config", type=Path)
    p.add_argument("--owner-root", type=Path)
    p.add_argument("--owned-action", choices=["canary", "train-search"], default="train-search")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--gpu-uuid")
    p.add_argument("--wait-seconds", type=int, default=86400)
    args = p.parse_args()
    if not args.config.is_file():
        raise ValueError("EXPLICIT_EXISTING_CONFIG_REQUIRED")
    if any(value != "inference.fallback_to_heuristic=false" for value in args.set):
        raise ValueError("REACH_SPEC_MUST_NOT_BE_OVERRIDDEN")
    from src.eval.bace_reach_v2 import plan, run_train, run_calibration, unseal
    if args.action == "plan":
        if not args.reference:
            raise ValueError("REFERENCE_REQUIRED")
        result = plan(args.reference, args.output_root, proposal_source=args.proposal_source, proposal_path=args.proposal_path)
    elif args.action == "status":
        result = {f: json.loads((args.output_root / f).read_text()) for f in ("progress.json", "candidate_freeze.json", "selector_freeze.json") if (args.output_root / f).is_file()}
    elif args.action == "resource-overlay":
        if not args.resource_config or not args.owner_root:
            raise ValueError("RESOURCE_OVERLAY_NEEDS_PRIOR_CONFIG_AND_FRESH_ROOT")
        from src.eval.bace_reach_resources import prepare_resources
        result = prepare_resources(args.resource_config, args.owner_root, args.output_root)
    elif args.action == "owner":
        if not args.resource_config or not args.owner_root or not args.gpu_uuid or args.gpu_index != 0:
            raise ValueError("EXISTING_OWNER_GPU0_RESOURCE_BINDING_REQUIRED")
        from src.ablations.llm.existing_gpu_owner import ResourceSampler, run_owned_child
        config = json.loads(args.resource_config.read_text())
        from src.eval.bace_frozen_gnn_contracts import sha256_file
        descriptor = {"path": str((args.output_root / "search_contract.json").resolve()),
                      "sha256": sha256_file(args.output_root / "search_contract.json")}
        sampler = ResourceSampler(config, args.gpu_index, args.gpu_uuid,
                                  task_family="ours_reach", reach_contract=descriptor)
        command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
            "--action", args.owned_action, "--output-root", str(args.output_root.resolve()), "--device", "cuda:0"]
        # Reuse the existing owner, UUID lock, single-slot FD and live watchdog;
        # no alternative locks or reservation authority are introduced.
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
        code = run_owned_child(command=command, environment=env, sampler=sampler,
            output_root=args.owner_root, lock_root=config["gpu_lock_root"],
            run_id="bace-ours-reach-v2-" + args.output_root.name, interval=60, max_wait_seconds=args.wait_seconds)
        if code == 0 and args.owned_action == "train-search":
            # Existing owner has waited for child exit and released both GPU
            # leases. The next executable stage is CPU-only calibration.
            from src.eval.bace_reach_resources import cpu_boundary
            boundary = lambda: cpu_boundary(config)
            boundary()
            result = run_calibration(args.output_root, device="cpu", boundary_check=boundary)
        else:
            return code
    else:
        from src.eval.bace_frozen_gnn_contracts import atomic_json, utc_now
        boundary = lambda: None
        if args.device == "cuda:0":
            from src.ablations.llm.existing_gpu_owner import receive_owner_binding, validate_inherited_lease
            binding = receive_owner_binding()
            def boundary():
                evidence = json.loads(Path(binding["resource_live_evidence"]).read_text())
                try:
                    validate_inherited_lease(evidence, binding["held_gpu_lock_fd"], binding["held_project_slot_fd"])
                except (ValueError, OSError) as error:
                    atomic_json(args.output_root / "paused.json", {"state": "PAUSED_AT_COMMITTED_PARENT_BOUNDARY", "reason": str(error), "updated_at": utc_now()})
                    raise SystemExit(75)
            boundary()
        if args.action in ("canary", "train-search"):
            if args.action == "train-search":
                canary = args.output_root / "canary" / "canary_receipt.json"
                proof = unseal(canary) if canary.exists() else run_train(args.output_root, device=args.device, boundary_check=boundary, canary_parents=1)
                if proof["state"] != "BOUNDED_TRAIN_CANARY_COMPLETE" or proof["budget_exceeded"]:
                    raise ValueError("TRAIN_CANARY_NOT_COMPLETE_OR_BUDGET_EXCEEDED")
            result = run_train(args.output_root, device=args.device, boundary_check=boundary,
                               canary_parents=1 if args.action == "canary" else 0)
        else:
            result = run_calibration(args.output_root, device=args.device, boundary_check=boundary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
