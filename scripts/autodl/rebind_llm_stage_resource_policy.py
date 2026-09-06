#!/usr/bin/env python3
"""CPU-only seal/validation of a resource overlay; no owner activation."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ablations.llm.stage_dispatch_binding import (
    _read_json, seal_resource_dispatch, validate_dispatch_runtime,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--action", choices=("seal", "validate"), required=True)
    parser.add_argument("--original-dispatch")
    parser.add_argument("--original-dispatch-sha256")
    parser.add_argument("--resource-config")
    parser.add_argument("--resource-config-sha256")
    parser.add_argument("--output")
    parser.add_argument("--dispatch-spec")
    parser.add_argument("--dispatch-spec-sha256")
    args = parser.parse_args(argv)
    if args.config != "configs/hpc.yaml" or set(args.set) - {"inference.fallback_to_heuristic=false"}:
        parser.error("Only the frozen config and non-heuristic guard are accepted")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if args.action == "seal":
        if not all((args.original_dispatch, args.original_dispatch_sha256, args.resource_config,
                    args.resource_config_sha256, args.output)) or args.dispatch_spec or args.dispatch_spec_sha256:
            parser.error("seal requires original dispatch/SHA, fresh resource config/SHA, and output only")
        result = seal_resource_dispatch(
            original_dispatch={"path": args.original_dispatch, "sha256": args.original_dispatch_sha256},
            resource_config={"path": args.resource_config, "sha256": args.resource_config_sha256},
            owner_driver_commit=commit, project_root=ROOT, output_path=args.output)
    else:
        if not all((args.dispatch_spec, args.dispatch_spec_sha256)) or any((args.original_dispatch,
                args.original_dispatch_sha256, args.resource_config, args.resource_config_sha256, args.output)):
            parser.error("validate requires only dispatch spec and its SHA")
        spec = _read_json({"path": args.dispatch_spec, "sha256": args.dispatch_spec_sha256})
        result = {"state": "DISPATCH_BINDING_VALID", **validate_dispatch_runtime(spec, commit, ROOT),
                  "science_started": False, "owner_started": False}
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
