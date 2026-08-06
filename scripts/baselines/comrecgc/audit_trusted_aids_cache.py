#!/usr/bin/env python3
"""Audit the pinned COMRECGC AIDS PyG cache without loading it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.cache_trust import audit_aids_pyg_cache  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", default="external/COMRECGC")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-inventory-sha256")
    args = parser.parse_args()
    result = audit_aids_pyg_cache(
        upstream_root=args.upstream_root,
        output_path=args.output,
        expected_inventory_sha256=args.expected_inventory_sha256,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["cache_trust_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
