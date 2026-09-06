#!/usr/bin/env python3
"""Seal fresh Mut A replay, B, post-A/B and existing executor bindings only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.utils.autodl_mut_recovery_binding_v1 import read_json, resource_status, seal_recovery


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    commands = parser.add_subparsers(dest="action", required=True)
    seal = commands.add_parser("seal")
    for name in ("old-ab-spec", "old-executor-spec", "driver-root", "output-root", "gpu-lock-root"):
        seal.add_argument(f"--{name}", type=Path, required=True)
    seal.add_argument("--gpu-uuid", required=True)
    status = commands.add_parser("status")
    status.add_argument("--binding", type=Path, required=True)
    args = parser.parse_args()
    if args.config != "configs/hpc.yaml":
        raise ValueError("This binding uses configs/hpc.yaml only")
    if args.action == "seal":
        paths = (args.old_ab_spec, args.old_executor_spec, args.driver_root, args.output_root, args.gpu_lock_root)
        if any(not p.is_absolute() or p.is_symlink() for p in paths):
            raise ValueError("Physical absolute paths required")
        result = seal_recovery(old_ab_path=args.old_ab_spec, old_executor_path=args.old_executor_spec,
            driver_root=args.driver_root, output=args.output_root,
            gpu_lock_root=args.gpu_lock_root, gpu_uuid=args.gpu_uuid)
    else:
        result = read_json(args.binding)
        result["current_resource"] = resource_status(args.binding.parent)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
