#!/usr/bin/env python3
"""Audit whether pinned COMRECGC generation indices can be sharded."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.shardability import (  # noqa: E402
    audit_generation_shardability,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = audit_generation_shardability(
        upstream_root=args.upstream_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
