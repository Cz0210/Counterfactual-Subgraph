#!/usr/bin/env python3
"""Evaluate five frozen BACE classifiers over one immutable proposal universe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ablations.gnn.cpu_evaluation import benchmark_verification, run_evaluation, evaluate_with_cpu_admission


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-roots-json", help="JSON map of all five names to frozen bundle directories")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-cpu-admission", action="store_true")
    parser.add_argument("--benchmark-checkpoint", help="Only train-side timing; no calibration/test access")
    args = parser.parse_args(argv)
    if args.benchmark_checkpoint:
        if args.model_roots_json or args.resume:
            parser.error("Timing probe requires fresh output and one benchmark checkpoint")
        result = benchmark_verification(args.bundle_root, args.benchmark_checkpoint, args.output_root)
    else:
        if not args.model_roots_json:
            parser.error("Full proposal-fixed evaluation requires --model-roots-json")
        model_roots = json.loads(Path(args.model_roots_json).read_text())
        runner = evaluate_with_cpu_admission if args.require_cpu_admission else run_evaluation
        result = runner(bundle_root=args.bundle_root, model_roots=model_roots,
            output_root=args.output_root, resume=args.resume, cpu_threads=args.cpu_threads,
            batch_size=args.batch_size)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
