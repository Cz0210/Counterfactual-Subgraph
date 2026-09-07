#!/usr/bin/env python3
"""Bounded train-only frozen-GIN/original66 CPU timing; not model training."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if not args.config.is_file() or not os.environ.get("SLURM_JOB_ID"):
        parser.error("Existing config and HPC compute-node allocation required")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        parser.error("Timing is CPU-only; no exposed GPU")
    from src.eval.bace_frozen_gnn_contracts import read_json
    from src.experiments.bace_gin_ours import train_only_timing
    spec = read_json(args.spec)
    if spec.get("main_matrix_write") is not False or spec.get("max_concurrent_jobs") != 2:
        parser.error("Independent experiment and at most two CPU jobs required")
    import torch
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))
    if not 1 <= threads <= 8:
        parser.error("CPU thread allocation outside task bound")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    print(json.dumps(train_only_timing(spec, args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()
