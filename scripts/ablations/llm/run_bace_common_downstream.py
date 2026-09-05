#!/usr/bin/env python
"""Run one matched BACE proposal-pool downstream; never publish main matrix."""
from __future__ import annotations

import argparse
import json

from src.ablations.llm.bace_common_downstream import run_downstream


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    for name in ("task-spec", "candidate-root", "gnn-input-bundle", "gnn-verified-archive",
                 "gnn-verified-sha256", "registry-root", "output-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = vars(parser.parse_args())
    # All science is bound to the real main reference, not arbitrary YAML overrides.
    args.pop("config")
    if set(args.pop("set")) - {"inference.fallback_to_heuristic=false"}:
        parser.error("Downstream science overrides are forbidden; use the frozen reference")
    if args["batch_size"] < 1 or args["cpu_threads"] < 1:
        parser.error("Batch size and CPU threads must be positive")
    result = run_downstream(**args)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["state"] in {"PASS", "PAUSED_AT_SAFE_PARENT_BOUNDARY"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
