#!/usr/bin/env python3
"""Audit a strict Mutagenicity official GCFExplainer smoke/full run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.gcfexplainer_mutagenicity_runtime import audit_mutagenicity_run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--gnn-dir", required=True)
    parser.add_argument("--vrrw-dir", required=True)
    parser.add_argument("--summary-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--forbid-calibration-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.forbid_calibration_test:
        raise ValueError("Run audit requires --forbid-calibration-test.")
    result = audit_mutagenicity_run(
        dataset_dir=args.dataset_dir,
        gnn_dir=args.gnn_dir,
        vrrw_dir=args.vrrw_dir,
        summary_dir=args.summary_dir,
        export_dir=args.export_dir,
        profile=args.profile,
    )
    print("[MUTAGENICITY_GCFEXPLAINER_RUN_AUDIT_OK]", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
