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
    parser.add_argument("--action", choices=("inspect-checkpoint", "compare"), required=True)
    parser.add_argument("--source-worktree", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--lowmemory", type=Path)
    args = parser.parse_args()
    if not args.config.is_file() or not args.output_root.is_absolute():
        parser.error("existing config and absolute fresh output required")
    driver = Path(__file__).resolve().parents[2]
    if args.action == "inspect-checkpoint":
        if not args.source_worktree or not args.source_worktree.is_absolute() or not args.source_root or not args.source_root.is_absolute():
            parser.error("source-worktree/source-root required and absolute")
        sys.path.insert(0, str(args.source_worktree))
    else:
        sys.path.insert(0, str(driver))
    spec = importlib.util.spec_from_file_location("t14_diagnostic_overlay", driver / "src/baselines/t14_causal_diagnostic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        if args.action == "inspect-checkpoint":
            resource_root = args.output_root.with_name(args.output_root.name + "-resources")
            resource_root.mkdir(parents=True, exist_ok=False)
            with module.ResourceSampler(resource_root):
                result = module.inspect_checkpoint(source_root=args.source_root, output_root=args.output_root)
        else:
            if not args.reference or not args.lowmemory:
                parser.error("both reference and lowmemory component roots required")
            args.output_root.mkdir(parents=True, exist_ok=True)
            result = module.compare_checkpoints(args.reference, args.lowmemory, args.output_root / "checkpoint250_comparison.json")
    except Exception as exc:
        args.output_root.mkdir(parents=True, exist_ok=True)
        module.atomic_json(args.output_root / "failed.json", {"status": "FAILED", "error": str(exc), "error_type": type(exc).__name__, "pid": os.getpid(), "new_transitions": 0, "formal_dispatch_allowed": False})
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
