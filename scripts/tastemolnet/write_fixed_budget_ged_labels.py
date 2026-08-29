#!/usr/bin/env python3
"""Write the frozen 5000/1000 Taste NeuroSED GEDLIB interval labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.tastemolnet_neurosed_label_writer import (  # noqa: E402
    GED_LABEL_PASS_MARKER,
    write_fixed_budget_ged_labels,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--non-mip-selection-manifest", type=Path, required=True)
    parser.add_argument("--non-mip-verifier-receipt", type=Path, required=True)
    parser.add_argument("--train-pair-root", type=Path, required=True)
    parser.add_argument("--validation-pair-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, action="append", default=[])
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--pair-timeout-seconds", type=float, default=300.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    expected_config = REPO_ROOT / "configs" / "hpc.yaml"
    config = args.config if Path(args.config).is_absolute() else REPO_ROOT / args.config
    if config.resolve() != expected_config.resolve():
        raise ValueError("GED label writer requires configs/hpc.yaml")
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise ValueError(
            "GED label writer requires exactly --set "
            "inference.fallback_to_heuristic=false"
        )
    result = write_fixed_budget_ged_labels(
        build_manifest_path=args.build_manifest,
        selection_manifest_path=args.non_mip_selection_manifest,
        selection_verifier_receipt_path=args.non_mip_verifier_receipt,
        train_pair_root=args.train_pair_root,
        validation_pair_root=args.validation_pair_root,
        output_root=args.output_root,
        workers=args.workers,
        pair_timeout_seconds=args.pair_timeout_seconds,
        cache_roots=args.cache_root,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    print(GED_LABEL_PASS_MARKER, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
