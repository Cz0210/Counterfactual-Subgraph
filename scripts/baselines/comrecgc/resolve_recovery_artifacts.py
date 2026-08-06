#!/usr/bin/env python3
"""Locate the exact frozen AIDS and Mutagenicity COMRECGC blockers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.artifact_resolution import resolve_recovery_artifacts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", default="outputs/hpc/baselines/comrecgc")
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    result = resolve_recovery_artifacts(
        outputs_root=args.outputs_root,
        output_path=args.output_path,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
