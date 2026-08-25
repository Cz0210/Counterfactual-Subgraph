#!/usr/bin/env python3
"""Run one typed stage of the fresh AIDS disconnected-exact recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_aids_comrecgc_exact_recovery_stages_v1 import (  # noqa: E402
    run_downstream_stage,
    run_exact_stage,
    run_final_stage,
    run_subset_stage,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--controller-manifest", type=_absolute, required=True)
    parser.add_argument("--output-dir", type=_absolute, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # Kept for repository-wide CLI/Slurm parity.  Scientific settings are read
    # exclusively from the hash-bound controller manifest.
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="stage", required=True)

    subset = commands.add_parser("subset", help="run production subset preflight")
    _common(subset)
    subset.add_argument("--adoption-gate", type=_absolute, required=True)
    subset.add_argument("--resume", action="store_true")

    exact = commands.add_parser("exact", help="recover the exact full partition")
    _common(exact)
    exact.add_argument("--adoption-gate", type=_absolute, required=True)
    exact.add_argument("--subset-gate", type=_absolute, required=True)
    exact.add_argument("--resume", action="store_true")

    downstream = commands.add_parser(
        "downstream", help="stream centroid/radius/coverage/greedy"
    )
    _common(downstream)
    downstream.add_argument("--exact-gate", type=_absolute, required=True)
    downstream.add_argument("--resume", action="store_true")

    final = commands.add_parser(
        "final", help="resume standardization and publish the recovery binding"
    )
    _common(final)
    final.add_argument("--adoption-gate", type=_absolute, required=True)
    final.add_argument("--subset-gate", type=_absolute, required=True)
    final.add_argument("--exact-gate", type=_absolute, required=True)
    final.add_argument("--downstream-gate", type=_absolute, required=True)
    final.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = {
        "controller_manifest": args.controller_manifest,
        "output_dir": args.output_dir,
    }
    if args.stage == "subset":
        result = run_subset_stage(
            adoption_gate=args.adoption_gate, resume=args.resume, **common
        )
    elif args.stage == "exact":
        result = run_exact_stage(
            adoption_gate=args.adoption_gate,
            subset_gate=args.subset_gate,
            resume=args.resume,
            **common,
        )
    elif args.stage == "downstream":
        result = run_downstream_stage(
            exact_gate=args.exact_gate, resume=args.resume, **common
        )
    else:
        result = run_final_stage(
            adoption_gate=args.adoption_gate,
            subset_gate=args.subset_gate,
            exact_gate=args.exact_gate,
            downstream_gate=args.downstream_gate,
            resume=args.resume,
            **common,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"[AIDS_EXACT_RECOVERY_{args.stage.upper()}_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
