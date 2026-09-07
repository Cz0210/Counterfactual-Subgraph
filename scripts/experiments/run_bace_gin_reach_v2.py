#!/usr/bin/env python3
"""Dataset-specific A+ stage CLI; run model work on a compute node only."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--action", required=True, choices=("plan", "evaluate", "ceiling", "train_gate", "audit_calibration", "freeze", "aggregate", "audit", "status"))
    p.add_argument("--group", choices=("old66", "adopted2607"), default="old66")
    p.add_argument("--split", choices=("train", "calibration", "test"), default="train")
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    if not Path(args.config).is_file():
        p.error("Existing runtime config required")
    from src.experiments import bace_gin_reach_v2 as driver
    spec = json.loads(Path(args.spec).read_text())
    if args.action == "evaluate":
        result = driver.evaluate(spec, args.group, args.split, limit=args.limit)
    elif args.action == "ceiling":
        result = driver.ceiling(spec, args.group, args.split)
    else:
        result = getattr(driver, args.action)(spec)
    print(json.dumps(result, default=str, sort_keys=True))


if __name__ == "__main__":
    main()
