#!/usr/bin/env python3
"""Mac-only scoped relay for the authorized first-temperature-fit correction."""
import argparse
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.gnn_seed7_mac_relay import CorrectiveRelayPlan, run_relay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--attempt-id", default=None)
    parser.add_argument("--hpc-job-id", action="append", required=True)
    parser.add_argument("--repair-driver-commit", required=True)
    parser.add_argument("--autodl-worktree", required=True)
    args = parser.parse_args(argv)
    plan = CorrectiveRelayPlan(args.attempt_id or str(uuid.uuid4()), tuple(args.hpc_job_id),
                               args.repair_driver_commit, Path(args.autodl_worktree))
    result = plan.to_dict() if args.action == "plan" else run_relay(plan)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
