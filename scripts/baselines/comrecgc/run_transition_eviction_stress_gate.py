#!/usr/bin/env python3
"""Run the CPU-only COMRECGC transition eviction stress gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.contracts import write_json  # noqa: E402
from src.baselines.comrecgc.stress_gate import run_transition_eviction_stress_gate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2_048)
    parser.add_argument("--cache-max-entries", type=int, default=64)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    result = run_transition_eviction_stress_gate(
        output_root=output.parent / "transition_eviction_stress_state",
        steps=args.steps,
        cache_max_entries=args.cache_max_entries,
    )
    write_json(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["stress_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
