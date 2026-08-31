#!/usr/bin/env python3
"""Run/resume TasteMolNet T12 GCF calibration-freeze and held-out test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.tastemolnet_gcf_full_postprocess import (  # noqa: E402
    load_input_authority,
    run_t12_postprocess,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.absolute() != path:
        raise argparse.ArgumentTypeError("path must be normalized and absolute")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--generation-root", type=_absolute, required=True)
    parser.add_argument(
        "--generation-verification-root", type=_absolute, required=True
    )
    parser.add_argument("--train-csv", type=_absolute, required=True)
    parser.add_argument("--calibration-csv", type=_absolute, required=True)
    parser.add_argument("--test-csv", type=_absolute, required=True)
    parser.add_argument("--gnn-checkpoint", type=_absolute, required=True)
    parser.add_argument("--molclr-root", type=_absolute, required=True)
    parser.add_argument("--molclr-checkpoint", type=_absolute, required=True)
    parser.add_argument("--threshold-contract", type=_absolute, required=True)
    parser.add_argument("--output-root", type=_absolute, required=True)
    parser.add_argument("--wnode-cache-db", type=_absolute, required=True)
    parser.add_argument("--node-embedding-cache-dir", type=_absolute, required=True)
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.config.resolve(strict=True) != (REPO_ROOT / "configs/hpc.yaml").resolve(
        strict=True
    ):
        raise RuntimeError("T12 paper postprocess requires configs/hpc.yaml")
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise RuntimeError("T12 paper postprocess requires fail-closed inference")
    authority = load_input_authority(
        generation_root=args.generation_root,
        generation_verification_root=args.generation_verification_root,
        train_csv=args.train_csv,
        calibration_csv=args.calibration_csv,
        test_csv=args.test_csv,
        gnn_checkpoint=args.gnn_checkpoint,
        molclr_root=args.molclr_root,
        molclr_checkpoint=args.molclr_checkpoint,
        threshold_contract=args.threshold_contract,
    )
    result = run_t12_postprocess(
        authority=authority,
        output_dir=args.output_root,
        resume=args.resume,
        device=args.device,
        wnode_cache_db=args.wnode_cache_db,
        node_embedding_cache_dir=args.node_embedding_cache_dir,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
