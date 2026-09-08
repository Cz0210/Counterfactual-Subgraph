#!/usr/bin/env python3
"""Offline review of the three existing T14 ledgers; never launches science."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # Explicit source bootstrap, including -I/-B.
from src.baselines.t14_semantic_review import review_three_ledgers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-ledger", type=Path, required=True)
    parser.add_argument("--continuous-ledger", type=Path, required=True)
    parser.add_argument("--reload-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error("config must be an existing project config (no inference is performed)")
    for path in (args.reference_ledger, args.continuous_ledger, args.reload_ledger, args.output_root):
        if not path.is_absolute():
            parser.error("all evidence/output paths must be absolute")
    result = review_three_ledgers(args.reference_ledger, args.continuous_ledger, args.reload_ledger, output_root=args.output_root)
    print(json.dumps(result, sort_keys=True))
    # Non-PASS is a successful audit of a failed/incomplete experiment, not an
    # excuse to submit duplicate diagnostics. Persisted status stays non-PASS.
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
