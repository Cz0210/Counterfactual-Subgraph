#!/usr/bin/env python3
"""Run official GCFExplainer VRRW on the frozen BACE source cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.gcfexplainer_bace_runtime import run_bace_official_vrrw  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--gnn-checkpoint", required=True)
    parser.add_argument("--neurosed-checkpoint", required=True)
    parser.add_argument("--neurosed-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--parent-limit", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--theta", type=float, default=0.05)
    parser.add_argument("--teleport", type=float, default=0.1)
    parser.add_argument("--candidate-capacity", type=int, default=100000)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device1", default="cuda:0")
    parser.add_argument("--device2", default="cuda:0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--acceleration-mode", choices=("legacy", "ordered_v2"), default="legacy"
    )
    parser.add_argument("--gine-batch-size", type=int, default=256)
    parser.add_argument("--graph-cache-capacity", type=int, default=0)
    parser.add_argument("--cpu-neighbor-workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--acceleration-gate",
        help="Required PASS gate for a full ordered_v2 M=50000 run",
    )
    args = parser.parse_args(argv)
    result = run_bace_official_vrrw(
        dataset_dir=args.dataset_dir,
        official_root=args.official_root,
        gnn_checkpoint=args.gnn_checkpoint,
        neurosed_checkpoint=args.neurosed_checkpoint,
        neurosed_manifest=args.neurosed_manifest,
        output_dir=args.output_dir,
        profile=args.profile,
        parent_limit=args.parent_limit,
        m=args.m,
        alpha=args.alpha,
        theta=args.theta,
        teleport=args.teleport,
        candidate_capacity=args.candidate_capacity,
        sample=args.sample,
        sample_size=args.sample_size,
        seed=args.seed,
        device1=args.device1,
        device2=args.device2,
        resume=args.resume,
        acceleration_mode=args.acceleration_mode,
        gine_batch_size=args.gine_batch_size,
        graph_cache_capacity=args.graph_cache_capacity,
        cpu_neighbor_workers=args.cpu_neighbor_workers,
        progress_every=args.progress_every,
        acceleration_gate=args.acceleration_gate,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[BACE_GCFEXPLAINER_VRRW_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
