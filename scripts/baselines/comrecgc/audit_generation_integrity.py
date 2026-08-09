#!/usr/bin/env python3
"""Gate a completed COMRECGC generation before downstream processing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.contracts import require_empty_output, write_json  # noqa: E402
from src.baselines.comrecgc.generation_integrity import audit_generation_integrity  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-steps", type=int, default=50_000)
    args = parser.parse_args()
    output = require_empty_output(args.output_dir)
    audit = audit_generation_integrity(
        args.generation_dir, expected_steps=args.expected_steps
    )
    write_json(output / "generation_integrity_gate.json", audit)
    if not audit["generation_integrity_passed"]:
        write_json(output / "_RUN_FAILED.json", audit)
        print(json.dumps(audit, sort_keys=True))
        return 2
    write_json(output / "_RUN_COMPLETE.json", audit)
    print(json.dumps(audit, sort_keys=True))
    print("[COMRECGC_GENERATION_INTEGRITY_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
