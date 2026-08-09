#!/usr/bin/env python3
"""Audit whether a COMRECGC generation checkpoint is safe to resume."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.checkpoint_audit import audit_generation_checkpoint  # noqa: E402
from src.baselines.comrecgc.contracts import write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit = audit_generation_checkpoint(args.generation_dir)
    write_json(args.output, audit)
    print(json.dumps(audit, sort_keys=True))
    print(f"RESUME_SAFE={str(bool(audit['RESUME_SAFE'])).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
