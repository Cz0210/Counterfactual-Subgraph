#!/usr/bin/env python3
"""Cluster COMRECGC graph recourses and export real graph medoids."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.contracts import RecourseParameters  # noqa: E402
from src.baselines.comrecgc.recourse import run_common_recourse  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("aids", "mutagenicity"), required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--upstream-root", default="external/COMRECGC")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--source-csv")
    parser.add_argument("--generation-dir", required=True)
    parser.add_argument("--distance-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-limit", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    upstream = Path(args.upstream_root)
    if not upstream.is_absolute():
        upstream = PROJECT_ROOT / upstream
    manifest = run_common_recourse(
        upstream_root=upstream,
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        source_csv=args.source_csv,
        generation_dir=args.generation_dir,
        distance_checkpoint=args.distance_checkpoint,
        output_dir=args.output_dir,
        mode=args.mode,
        parent_limit=args.parent_limit,
        parameters=RecourseParameters.for_mode(args.mode),
        device=args.device,
        batch_size=args.batch_size,
        resume=args.resume,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
