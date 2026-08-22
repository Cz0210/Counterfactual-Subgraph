#!/usr/bin/env python3
"""Compare two frozen selector manifests for later ablation analysis."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.rule_stability import compare_frozen_rule_selections


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Output must be fresh: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--left-manifest", required=True)
    parser.add_argument("--right-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare_frozen_rule_selections(args.left_manifest, args.right_manifest)
    _atomic_json(Path(args.output), result)
    print(json.dumps(result, sort_keys=True))
    print("[SELECTED_RULE_STABILITY_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
