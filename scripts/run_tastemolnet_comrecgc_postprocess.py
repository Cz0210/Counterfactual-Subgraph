#!/usr/bin/env python3
"""Run or independently publish the TasteMolNet T14 paper-cell continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.tastemolnet_comrecgc_postprocess import (  # noqa: E402
    PASS_MARKER,
    load_postprocess_authority,
    run_t14_postprocess,
    verify_and_publish_t14,
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("postprocess", "verify"), required=True)
    parser.add_argument("--config", type=_absolute, required=True)
    parser.add_argument("--generation-root", type=_absolute, required=True)
    parser.add_argument("--science-root", type=_absolute, required=True)
    parser.add_argument("--final-root", type=_absolute)
    parser.add_argument("--calibration-csv", type=_absolute, required=True)
    parser.add_argument("--test-csv", type=_absolute, required=True)
    parser.add_argument("--gnn-checkpoint", type=_absolute, required=True)
    parser.add_argument("--molclr-root", type=_absolute, required=True)
    parser.add_argument("--molclr-checkpoint", type=_absolute, required=True)
    parser.add_argument("--threshold-contract", type=_absolute, required=True)
    parser.add_argument("--wnode-cache-db", type=_absolute)
    parser.add_argument("--node-embedding-cache-dir", type=_absolute)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise ValueError("T14 postprocess requires the fail-closed inference override")
    authority = load_postprocess_authority(
        generation_root=args.generation_root,
        calibration_csv=args.calibration_csv,
        test_csv=args.test_csv,
        gnn_checkpoint=args.gnn_checkpoint,
        molclr_root=args.molclr_root,
        molclr_checkpoint=args.molclr_checkpoint,
        threshold_contract=args.threshold_contract,
    )
    if args.mode == "postprocess":
        if args.final_root is not None:
            raise ValueError("postprocess mode does not accept --final-root")
        if args.wnode_cache_db is None or args.node_embedding_cache_dir is None:
            raise ValueError("postprocess mode requires both WNode cache paths")
        result = run_t14_postprocess(
            authority=authority,
            science_root=args.science_root,
            resume=args.resume,
            device=args.device,
            wnode_cache_db=args.wnode_cache_db,
            node_embedding_cache_dir=args.node_embedding_cache_dir,
        )
        marker = "[TASTE_T14_COMRECGC_POSTPROCESS_SEALED]"
    else:
        if args.resume:
            raise ValueError("verify mode does not accept --resume")
        if args.final_root is None:
            raise ValueError("verify mode requires --final-root")
        result = verify_and_publish_t14(
            authority=authority,
            science_root=args.science_root,
            final_root=args.final_root,
        )
        marker = PASS_MARKER
    print(json.dumps(result, sort_keys=True, ensure_ascii=True), flush=True)
    print(marker, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
