#!/usr/bin/env python3
"""Run official GCFExplainer VRRW in an isolated work directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.gcf_official_adapter import build_run_config_from_args, run_official_vrrw  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--official-repo", default=None)
    parser.add_argument("--dataset", default="aids")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--train-theta", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--teleport", type=float, default=0.1)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--device1", default="0")
    parser.add_argument("--device2", default="0")
    parser.add_argument("--run-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metadata = run_official_vrrw(build_run_config_from_args(args))
    print("[GCF_OFFICIAL_RUN_DONE]", flush=True)
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

