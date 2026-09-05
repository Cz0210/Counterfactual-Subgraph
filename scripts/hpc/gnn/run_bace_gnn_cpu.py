#!/usr/bin/env python3
"""Run the frozen BACE alternative-backbone CPU benchmark or training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.ablations.gnn.cpu_training import run_cpu_auto, run_cpu_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--backbone", choices=("gin", "gcn", "gatv2", "gatedgcn_plus"), required=True)
    parser.add_argument("--phase", choices=("benchmark", "train", "auto"), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", dest="config_path", required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--benchmark-epochs", type=int, default=5)
    parser.add_argument("--benchmark-seconds", type=float, default=1200)
    parser.add_argument("--resume", action="store_true")
    values = vars(parser.parse_args(argv))
    phase = values.pop("phase")
    result = run_cpu_auto(**values) if phase == "auto" else run_cpu_training(**values, phase=phase)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
