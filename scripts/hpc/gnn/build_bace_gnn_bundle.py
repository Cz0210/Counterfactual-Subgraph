#!/usr/bin/env python3
"""Build on AutoDL, or verify/extract the scoped BACE bundle on HPC."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ablations.gnn.hpc_bundle import build_bundle, unpack_bundle, verify_bundle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hpc.yaml")
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build")
    for arg in ("reference-path", "matrix-path", "merged-pool-root", "molclr-source", "project-root", "output-root", "execution-commit"):
        build.add_argument("--" + arg, required=True)
    unpack = sub.add_parser("unpack")
    for arg in ("archive", "expected-sha", "output-root"):
        unpack.add_argument("--" + arg, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--root", required=True)
    args = vars(parser.parse_args())
    action = args.pop("action")
    args.pop("config")
    result = {"build": build_bundle, "unpack": unpack_bundle, "verify": verify_bundle}[action](**args)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
