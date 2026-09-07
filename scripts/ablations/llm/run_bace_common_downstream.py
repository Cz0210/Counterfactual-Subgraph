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
    parser.add_argument("--reparse-saved-raw", action="store_true",
                        help="Fresh correction overlay using unchanged completed native raw generations; never invokes an LLM")
    parser.add_argument("--parser-correction-source-evaluation", help="Completed same-variant evaluation for exact match-level distance reuse")
    parser.add_argument("--parser-correction-source-audit-sha256")
    parser.add_argument("--stage-file-policy", help="Existing immutable stage file-policy JSON")
    parser.add_argument("--stage-file-policy-sha256")
    parser.add_argument("--portable-input-bundle", help="Read original AutoDL manifest bytes through a SHA-bound L0 HPC mapping")
    parser.add_argument("--gnn-acceptance", help="Existing small corrected-GNN acceptance; avoids archive replay")
    parser.add_argument("--gnn-acceptance-sha256")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = vars(parser.parse_args())
    # All science is bound to the real main reference, not arbitrary YAML overrides.
    args.pop("config")
    if set(args.pop("set")) - {"inference.fallback_to_heuristic=false"}:
        parser.error("Downstream science overrides are forbidden; use the frozen reference")
    if args["batch_size"] < 1 or args["cpu_threads"] < 1:
        parser.error("Batch size and CPU threads must be positive")
    policy_path = args.pop("stage_file_policy")
    policy_sha = args.pop("stage_file_policy_sha256")
    if (policy_path is None) != (policy_sha is None):
        parser.error("Stage policy path and SHA must be paired")
    if policy_path is not None:
        from src.utils.stage_file_policy import load_stage_policy
        policy = json.loads(Path(policy_path).read_text())
        args["stage_file_policy"] = load_stage_policy(
            {"path": policy_path, "sha256": policy_sha}, policy["persistent_root"],
            stage_id="llm_cpu_evaluation")
        from src.utils.stage_file_policy import stage_file_admission
        ancestor = Path(args["output_root"])
        while not ancestor.exists():
            ancestor = ancestor.parent
        if os.stat(ancestor).st_dev != policy["filesystem_device"]:
            raise ValueError("CPU_EVALUATION_STAGE_RESOURCE_DOMAIN_CHANGED")
        admission = stage_file_admission(args["stage_file_policy"], os.statvfs(ancestor).f_favail,
                                         stage_id="llm_cpu_evaluation")
        if not admission["admitted"]:
            print(json.dumps({"state": "WAITING_CPU_RESOURCE", "file_admission": admission}))
            return 75
    result = run_downstream(**args)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["state"] == "PASS" else 75 if result["state"] == "PAUSED_AT_SAFE_PARENT_BOUNDARY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
