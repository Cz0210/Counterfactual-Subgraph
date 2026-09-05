#!/usr/bin/env python3
"""Mac-only one-shot fd98c5f2 GNN result relay; no science launch."""
import argparse
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.gnn_seed7_mac_relay import RelayPlan, run_relay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--attempt-id", default=None)
    args = parser.parse_args(argv)
    try:
        plan = RelayPlan(args.attempt_id or str(uuid.uuid4()))
        result = plan.to_dict() if args.action == "plan" else run_relay(plan)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"[GNN_SEED7_RELAY_FAILED] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
