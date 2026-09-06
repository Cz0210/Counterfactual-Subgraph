#!/usr/bin/env python3
"""Seal/inspect one next-stage file-budget overlay; never start or stop science."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.stage_file_policy import canonical_sha, load_stage_policy, stage_file_admission
from src.eval.bace_frozen_gnn_contracts import atomic_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--action", choices=("seal", "status"), required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--stage-id", required=True)
    args = parser.parse_args()
    if not args.policy.is_absolute():
        parser.error("Absolute policy path required")
    if args.action == "seal":
        if not args.plan or args.policy.exists():
            parser.error("Fresh policy path and explicit bounded plan required")
        body = json.loads(args.plan.read_text())
        if "self_sha256" in body:
            raise ValueError("Plan must not pretend to be sealed")
        body["filesystem_device"] = os.stat(body["persistent_root"]).st_dev
        body["self_sha256"] = canonical_sha(body)
        # Validate complete small evidence first through a temporary sibling;
        # failed plans are retained and never called an admitted policy.
        temporary = args.policy.with_name(args.policy.name + ".pending")
        if temporary.exists():
            raise FileExistsError(temporary)
        atomic_json(temporary, body)
        identity = {"path": str(temporary), "sha256": sha256_file(temporary)}
        policy = load_stage_policy(identity, body["persistent_root"], args.stage_id)
        os.link(temporary, args.policy)
        temporary.unlink()
        directory = os.open(args.policy.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    else:
        if not args.sha256:
            parser.error("Sealed policy SHA required")
        raw = json.loads(args.policy.read_text())
        policy = load_stage_policy({"path": str(args.policy), "sha256": args.sha256},
                                   raw["persistent_root"], args.stage_id)
    fs = os.statvfs(policy["persistent_root"])
    result = stage_file_admission(policy, fs.f_favail, stage_id=args.stage_id)
    result.update(policy_path=str(args.policy), policy_file_sha256=sha256_file(args.policy),
                  actual_available_bytes=fs.f_bavail*fs.f_frsize, science_started=False,
                  platform_contact_required=False)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
