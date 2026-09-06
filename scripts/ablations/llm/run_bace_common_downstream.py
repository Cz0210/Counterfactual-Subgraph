#!/usr/bin/env python
"""Run one matched BACE proposal-pool downstream; never publish main matrix."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.ablations.llm.bace_common_downstream import run_downstream


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    for name in ("task-spec", "candidate-root", "gnn-input-bundle", "gnn-verified-archive",
                 "gnn-verified-sha256", "registry-root", "output-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--portable-input-bundle", help="Read original AutoDL manifest bytes through a SHA-bound L0 HPC mapping")
    parser.add_argument("--gnn-acceptance", help="Existing small corrected-GNN acceptance; avoids archive replay")
    parser.add_argument("--gnn-acceptance-sha256")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--stage-file-policy")
    parser.add_argument("--stage-file-policy-sha256")
    parser.add_argument("--compact-node-cache", action="store_true",
                        help="Future-only lossless embeddings in this evaluator's existing WNode DB")
    args = vars(parser.parse_args())
    # All science is bound to the real main reference, not arbitrary YAML overrides.
    args.pop("config")
    if set(args.pop("set")) - {"inference.fallback_to_heuristic=false"}:
        parser.error("Downstream science overrides are forbidden; use the frozen reference")
    if args["batch_size"] < 1 or args["cpu_threads"] < 1:
        parser.error("Batch size and CPU threads must be positive")
    policy_path, policy_sha = args.pop("stage_file_policy"), args.pop("stage_file_policy_sha256")
    if policy_path or policy_sha:
        if not policy_path or not policy_sha:
            parser.error("Stage file policy requires path and SHA")
        from src.utils.stage_file_policy import load_stage_policy, stage_file_admission
        # The policy binds the persistent resource domain, not this fresh child
        # output directory. Resolve the existing ancestor without creating files.
        output = Path(args["output_root"])
        ancestor = output
        while not ancestor.exists():
            ancestor = ancestor.parent
        raw = json.loads(Path(policy_path).read_text())
        policy = load_stage_policy({"path": policy_path, "sha256": policy_sha}, raw["persistent_root"])
        if os.stat(ancestor).st_dev != policy["filesystem_device"]:
            raise ValueError("CPU_EVALUATION_STAGE_RESOURCE_DOMAIN_CHANGED")
        decision = stage_file_admission(policy, os.statvfs(ancestor).f_favail,
                                        stage_id="llm_cpu_evaluation")
        if not decision["admitted"]:
            print(json.dumps({"state": "WAITING_CPU_RESOURCE", "file_admission": decision}))
            return 75
        args["stage_file_policy"] = policy
    result = run_downstream(**args)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["state"] == "PASS" else 75 if result["state"] == "PAUSED_AT_SAFE_PARENT_BOUNDARY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
