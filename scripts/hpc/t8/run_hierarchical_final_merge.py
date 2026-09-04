#!/usr/bin/env python3
"""Finalize completed T8 groups into exact monolithic merge bytes."""

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
    finalize_hierarchical_merge,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--group-plan", required=True, type=Path)
    parser.add_argument("--groups-root", required=True, type=Path)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    validate_hpc_cli_contract(args.config, args.set)
    result = finalize_hierarchical_merge(
        group_plan=args.group_plan,
        groups_root=args.groups_root,
        state_root=args.state_root,
        scratch_root=args.scratch_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
