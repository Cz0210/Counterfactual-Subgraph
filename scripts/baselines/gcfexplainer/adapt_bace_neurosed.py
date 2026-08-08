#!/usr/bin/env python3
"""Create the audited nine-channel BACE NeuroSED transfer checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.gcfexplainer_bace_adapter import adapt_bace_neurosed_checkpoint  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--manifest-path", required=True)
    args = parser.parse_args(argv)
    result = adapt_bace_neurosed_checkpoint(
        source_checkpoint=args.source_checkpoint,
        output_checkpoint=args.output_checkpoint,
        manifest_path=args.manifest_path,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[BACE_GCFEXPLAINER_NEUROSED_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
