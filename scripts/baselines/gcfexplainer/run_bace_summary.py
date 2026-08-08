#!/usr/bin/env python3
"""Build official greedy native summary ranks for BACE GCFExplainer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.gcfexplainer_bace_runtime import build_bace_native_summary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--vrrw-dir", required=True)
    parser.add_argument("--gnn-checkpoint", required=True)
    parser.add_argument("--neurosed-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--theta", type=float, default=0.1)
    parser.add_argument("--minimum-native-export", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    result = build_bace_native_summary(
        dataset_dir=args.dataset_dir,
        official_root=args.official_root,
        vrrw_dir=args.vrrw_dir,
        gnn_checkpoint=args.gnn_checkpoint,
        neurosed_checkpoint=args.neurosed_checkpoint,
        output_dir=args.output_dir,
        profile=args.profile,
        theta=args.theta,
        minimum_native_export=args.minimum_native_export,
        device=args.device,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[BACE_GCFEXPLAINER_NATIVE_SUMMARY_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
