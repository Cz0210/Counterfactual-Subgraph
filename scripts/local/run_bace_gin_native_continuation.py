#!/usr/bin/env python3
"""This campaign's finite GCF/ComRec post-freeze continuation; never calibration."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.bace_gin_continuation import Continuation, locations, validate_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--action', choices=('plan','status','run'), required=True)
    args = parser.parse_args()
    plan = validate_plan(json.loads(args.plan.read_text()))
    if args.action == 'plan':
        result = dict(plan=plan, locations=locations(plan), executed=False)
    else:
        relay = Continuation(plan)
        result = relay.state if args.action == 'status' else relay.run()
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
