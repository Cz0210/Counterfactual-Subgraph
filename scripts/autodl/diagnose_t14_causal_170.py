#!/usr/bin/env python3
"""Record the two existing T14 checkpoint250 states before bounded replay."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--action", choices=("inspect-checkpoint", "compare", "replay-arm", "compare-replay"), required=True)
    parser.add_argument("--source-worktree", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--lowmemory", type=Path)
    parser.add_argument("--source-spec", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--campaign-contract", type=Path)
    parser.add_argument("--wait-for-terminals-seconds", type=int, default=0)
    parser.add_argument("--initial-comparison", type=Path)
    args = parser.parse_args()
    if not args.config.is_file() or not args.output_root.is_absolute():
        parser.error("existing config and absolute fresh output required")
    driver = Path(__file__).resolve().parents[2]
    if args.action in {"inspect-checkpoint", "replay-arm"}:
        if not args.source_worktree or not args.source_worktree.is_absolute():
            parser.error("source-worktree required and absolute")
        if args.action == "inspect-checkpoint" and (not args.source_root or not args.source_root.is_absolute()):
            parser.error("source-root required and absolute")
        sys.path.insert(0, str(args.source_worktree))
    else:
        sys.path.insert(0, str(driver))
    spec = importlib.util.spec_from_file_location("t14_diagnostic_overlay", driver / "src/baselines/t14_causal_diagnostic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output_existed_before = args.output_root.exists()
    try:
        if args.action == "inspect-checkpoint":
            resource_root = args.output_root.with_name(args.output_root.name + "-resources")
            resource_root.mkdir(parents=True, exist_ok=False)
            with module.ResourceSampler(resource_root):
                result = module.inspect_checkpoint(source_root=args.source_root, output_root=args.output_root)
        elif args.action == "replay-arm":
            if not args.source_spec or not args.source_spec.is_absolute() or not args.gpu_uuid or not args.campaign_contract or not args.campaign_contract.is_absolute():
                parser.error("replay requires exact source-spec, total170 campaign contract and GPU UUID")
            runtime_spec = importlib.util.spec_from_file_location("t14_causal_runtime_overlay", driver / "src/baselines/t14_causal_replay.py")
            runtime = importlib.util.module_from_spec(runtime_spec)
            runtime_spec.loader.exec_module(runtime)
            resource_root = args.output_root.with_name(args.output_root.name + "-resources")
            resource_root.mkdir(parents=True, exist_ok=False)
            with module.ResourceSampler(resource_root):
                result = runtime.execute_replay(source_worktree=args.source_worktree, source_spec_path=args.source_spec, output_root=args.output_root, gpu_uuid=args.gpu_uuid, campaign_path=args.campaign_contract, diagnostic=module)
        elif args.action == "compare-replay":
            from src.baselines.t14_causal_comparison import await_and_compare, compare_replays
            if args.wait_for_terminals_seconds:
                if not args.campaign_contract or not args.campaign_contract.is_absolute():
                    parser.error("waiting comparison requires absolute existing campaign contract")
                result = await_and_compare(args.campaign_contract, args.output_root, args.wait_for_terminals_seconds, initial_comparison=args.initial_comparison)
            else:
                if not args.reference or not args.lowmemory:
                    parser.error("sealed reference and lowmemory replay roots required")
                result = compare_replays(args.reference, args.lowmemory, args.output_root, args.initial_comparison)
        else:
            if not args.reference or not args.lowmemory:
                parser.error("both reference and lowmemory component roots required")
            args.output_root.mkdir(parents=True, exist_ok=True)
            result = module.compare_checkpoints(args.reference, args.lowmemory, args.output_root / "checkpoint250_comparison.json")
    except Exception as exc:
        if output_existed_before:
            # A repeated invocation must not mutate an earlier sealed attempt
            # by dropping a new FAILED marker into its output root.
            raise
        args.output_root.mkdir(parents=True, exist_ok=True)
        budget_path = args.output_root / "transition_budget.json"
        budget = json.loads(budget_path.read_text()) if budget_path.is_file() else {"started_new_transitions": 0, "completed_new_transitions": 0}
        module.atomic_json(args.output_root / "failed.json", {"status": "FAILED", "error": str(exc), "error_type": type(exc).__name__, "pid": os.getpid(), "transition_budget": budget, "formal_dispatch_allowed": False})
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
