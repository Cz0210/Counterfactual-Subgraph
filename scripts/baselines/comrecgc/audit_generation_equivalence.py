#!/usr/bin/env python3
"""Audit fresh BACE COMRECGC legacy/optimized generation prefixes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.equivalence import (  # noqa: E402
    audit_generation_equivalence,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--optimized-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-steps", type=int, choices=(500, 1000), required=True)
    args = parser.parse_args()
    result = audit_generation_equivalence(
        legacy_root=args.legacy_root,
        optimized_root=args.optimized_root,
        output_dir=args.output_dir,
        expected_steps=args.expected_steps,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
