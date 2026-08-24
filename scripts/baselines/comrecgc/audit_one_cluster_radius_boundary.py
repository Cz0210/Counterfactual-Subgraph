#!/usr/bin/env python3
"""Post-hoc audit of c766 one-cluster strict-radius scalar semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.one_cluster_radius_posthoc import (  # noqa: E402
    run_one_cluster_radius_posthoc_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--terminal-one-cluster-manifest", required=True)
    parser.add_argument("--expected-terminal-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_one_cluster_radius_posthoc_audit(
        terminal_manifest_path=args.terminal_one_cluster_manifest,
        expected_terminal_manifest_sha256=args.expected_terminal_manifest_sha256,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "old_vs_dtype_cast_diff_count": result[
                    "old_vs_dtype_cast_diff_count"
                ],
                "live_output_adoptable": result["live_output_adoptable"],
                "recommended_action": result["recommended_action"],
            },
            sort_keys=True,
        )
    )
    if result["status"] == "PASS":
        print("[COMRECGC_ONE_CLUSTER_RADIUS_POSTHOC_PASS]")
        return 0
    print("[COMRECGC_ONE_CLUSTER_RADIUS_POSTHOC_BLOCKED]")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
