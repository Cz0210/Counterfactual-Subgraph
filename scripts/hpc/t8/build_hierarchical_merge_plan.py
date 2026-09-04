#!/usr/bin/env python3
"""Adopt the completed T8 array and build a deterministic merge group plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_hpc_exact import validate_hpc_cli_contract  # noqa: E402
from src.baselines.globalgce_hpc_hierarchical import (  # noqa: E402
    adopt_completed_array,
    build_group_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--partition-manifest", required=True, type=Path)
    parser.add_argument("--shards-root", required=True, type=Path)
    parser.add_argument("--array-adoption", required=True, type=Path)
    parser.add_argument("--group-plan", required=True, type=Path)
    parser.add_argument("--group-count", type=int, default=4)
    args = parser.parse_args()
    validate_hpc_cli_contract(args.config, args.set)
    adoption = adopt_completed_array(
        partition_manifest=args.partition_manifest,
        shards_root=args.shards_root,
        output=args.array_adoption,
    )
    plan = build_group_plan(
        partition_manifest=args.partition_manifest,
        shards_root=args.shards_root,
        array_adoption=args.array_adoption,
        output=args.group_plan,
        group_count=args.group_count,
    )
    print(
        json.dumps(
            {
                "state": "PASS",
                "passed_shards": adoption["passed_shard_count"],
                "group_count": plan["group_count"],
                "group_plan_sha256": plan["group_plan_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
