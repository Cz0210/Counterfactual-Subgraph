#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Migrate sealed train raw costs only; no inference, test or OT")
    p.add_argument("--config", required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--portable", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-commit", required=True)
    args = p.parse_args()
    if not Path(args.config).is_file():
        p.error("config absent")
    from src.experiments.bace_gin_reach_raw import migrate_train
    result = migrate_train(args.campaign, args.portable, args.output, args.source_commit)
    print(json.dumps({k:v for k,v in result.items() if k not in {"graph_costs", "source_spec"}}, sort_keys=True))
