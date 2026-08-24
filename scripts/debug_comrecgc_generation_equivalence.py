#!/usr/bin/env python3
"""Write a safe JSON-only COMRECGC first-divergence report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.generation_divergence import (  # noqa: E402
    write_generation_divergence_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--optimized-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if "inference.fallback_to_heuristic=false" not in args.set:
        raise ValueError(
            "Divergence diagnostics require "
            "--set inference.fallback_to_heuristic=false."
        )
    report = write_generation_divergence_report(
        legacy_root=args.legacy_root,
        optimized_root=args.optimized_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
