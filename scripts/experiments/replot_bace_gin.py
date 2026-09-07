#!/usr/bin/env python3
"""Offline exact BACE parent-reducer plots; no oracle, selector or training."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.experiments.bace_gin_reporting import render

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version-label", default="BACE-GIN-fixed-pool-v1")
    args = parser.parse_args()
    render(args.source_csv.resolve(strict=True), args.output.resolve(), version_label=args.version_label)
