#!/usr/bin/env python3
"""Verify a CM-only portable result into a fresh, authorized import root."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines.cm_crem_release import verify_import, finalize_interrupted_import

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--resume-staging", type=Path)
    args = parser.parse_args()
    result=(finalize_interrupted_import(args.package,args.manifest,args.destination,args.resume_staging)
            if args.resume_staging else verify_import(args.package,args.manifest,args.destination))
    print(json.dumps(result,sort_keys=True))
