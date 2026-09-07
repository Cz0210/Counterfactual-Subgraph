#!/usr/bin/env python3
"""CLI for the bounded future-only T13 bridge engineering benchmark."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    if not Path(args.config).is_file():
        parser.error("config path must exist; benchmark does not consume science config")
    if not 1 <= args.repeats <= 5:
        parser.error("bounded repeats must be 1..5")
    if args.device.startswith("cuda") and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        parser.error("CUDA benchmark requires child environment CUBLAS_WORKSPACE_CONFIG=:4096:8")
    import torch
    if args.device.startswith("cuda") and torch.__version__ != "2.7.1+cu118":
        parser.error("CUDA benchmark must match frozen Torch 2.7.1+cu118")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    from src.baselines.t13_bridge_materialization_benchmark import benchmark
    result = benchmark(ROOT, device=args.device, repeats=args.repeats)
    result["runtime_backend"] = {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "all_exact": result["all_exact"],
                      "speedups": [v["median_speedup"] for v in result["measurements"]]}))
    return 0 if result["all_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
