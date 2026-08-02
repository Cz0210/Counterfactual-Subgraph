#!/usr/bin/env python3
"""Build the official greedy native summary for Mutagenicity VRRW output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.gcfexplainer_mutagenicity_adapter import write_failure_artifacts  # noqa: E402
from src.baselines.gcfexplainer_mutagenicity_runtime import (  # noqa: E402
    _SummaryConfigError,
    build_native_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--vrrw-dir", required=True)
    parser.add_argument("--gnn-checkpoint", required=True)
    parser.add_argument("--neurosed-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--theta", type=float, default=0.1)
    parser.add_argument("--minimum-native-export", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--forbid-calibration-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.forbid_calibration_test:
        raise ValueError("Native summary requires --forbid-calibration-test.")
    try:
        summary = build_native_summary(
            dataset_dir=args.dataset_dir,
            official_root=args.official_root,
            vrrw_dir=args.vrrw_dir,
            gnn_checkpoint=args.gnn_checkpoint,
            neurosed_checkpoint=args.neurosed_checkpoint,
            output_dir=args.output_dir,
            profile=args.profile,
            theta=args.theta,
            minimum_native_export=args.minimum_native_export,
            device=args.device,
        )
    except _SummaryConfigError as exc:
        resolved_config = {**vars(args), **exc.details}
        write_failure_artifacts(
            args.output_dir,
            error=exc,
            resolved_config=resolved_config,
            extra=exc.details,
        )
        print("[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR]", file=sys.stderr)
        for key in ("field", "actual", "expected", "count_source"):
            print(f"{key}={exc.details[key]}", file=sys.stderr)
        return 2
    except Exception as exc:
        write_failure_artifacts(
            args.output_dir,
            error=exc,
            resolved_config=vars(args),
            extra={"stage": "summary_runtime"},
        )
        raise
    print("[MUTAGENICITY_GCFEXPLAINER_NATIVE_SUMMARY_OK]", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
