#!/usr/bin/env python3
"""Import one complete independently verified GNN package into a fresh location."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ablations.gnn.scientific_verification import import_verified_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--archive-path", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    args = vars(parser.parse_args(argv))
    args.pop("config")
    print(json.dumps(import_verified_bundle(**args), sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
