#!/usr/bin/env python3
"""Audit/seal the explicitly nominated Mut trace-on artifact without parity claims."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.autodl_mut_independent_adoption import AdoptionError, STAGE_ID, audit, publish


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/hpc.yaml")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--action", choices=("audit", "seal"), required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--common-root", type=Path, required=True)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--authorization", type=Path)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--stage-policy", type=Path,
                   help="Existing resource-config JSON containing stage_file_policy descriptor")
    p.add_argument("--persistent-root", type=Path)
    p.add_argument("--proc-root", type=Path, default=Path("/proc"))
    a = p.parse_args()
    kwargs = dict(source_root=a.source_root, common_root=a.common_root,
                  inventory_path=a.inventory, proc_root=a.proc_root)
    try:
        if a.action == "audit":
            result = audit(**kwargs)
        else:
            if not all((a.authorization, a.output_root, a.stage_policy, a.persistent_root)):
                p.error("seal requires authorization/output-root/stage-policy/persistent-root")
            from src.utils.stage_file_policy import load_stage_policy, stage_file_admission
            import os
            config = json.loads(a.stage_policy.read_text())
            policy = load_stage_policy(config["stage_file_policy"], a.persistent_root, stage_id=STAGE_ID)
            available = os.statvfs(a.persistent_root).f_favail
            gate = stage_file_admission(policy, available, stage_id=STAGE_ID)
            if not gate["admitted"]:
                raise AdoptionError("METADATA_OR_PATH_GAP", "RESOURCE_ADMISSION:" + json.dumps(gate))
            result = publish(**kwargs, authorization_path=a.authorization, output_root=a.output_root)
        print(json.dumps(result, sort_keys=True))
        return 0
    except AdoptionError as e:
        print(json.dumps({"status": "BLOCKED", "classification": e.category,
                          "first_issue": e.field, "science_launched": False}), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
