#!/usr/bin/env python3
"""Independently replay completed row artifacts and create a fresh publication overlay."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ablations.gnn.scientific_verification import publish_overlay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--environment-manifest", required=True)
    parser.add_argument("--driver-commit", required=True)
    parser.add_argument("--calibration-prediction-root")
    args = vars(parser.parse_args(argv))
    args.pop("config")
    print(json.dumps(publish_overlay(**args), sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
