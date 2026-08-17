#!/usr/bin/env python3
"""Validate and render a frozen shared WNode threshold manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.frozen_threshold_manifest import load_shared_frozen_thresholds  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--thresholds-json", required=True)
    parser.add_argument("--format", choices=("csv", "json"), default="json")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    contract = load_shared_frozen_thresholds(args.thresholds_json)
    if args.validate_only:
        print("[FROZEN_WNODE_THRESHOLD_MANIFEST_VALID]")
    elif args.format == "csv":
        print(contract["threshold_csv"])
    else:
        print(json.dumps(contract, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
