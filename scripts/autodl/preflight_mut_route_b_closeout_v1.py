#!/usr/bin/env python3
"""Report the Mut Route-B scientific input gap; never launch backup science."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.autodl_mut_route_b_closeout_v1 import inspect_closeout  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--resource-path", type=Path, required=True)
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--vector-dimension", type=int)
    parser.add_argument("--vector-itemsize", type=int)
    args = parser.parse_args()
    if args.config != "configs/hpc.yaml" or args.set not in (
        [], ["inference.fallback_to_heuristic=false"]
    ):
        raise ValueError("unsupported config override")
    decision = json.loads(args.decision.read_text()) if args.decision else None
    result = inspect_closeout(
        repo_root=PROJECT_ROOT, resource_path=args.resource_path, decision=decision,
        vector_dimension=args.vector_dimension, vector_itemsize=args.vector_itemsize,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    # A successful audit is not a READY or scientific PASS terminal.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
