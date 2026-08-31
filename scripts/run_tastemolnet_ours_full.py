#!/usr/bin/env python3
"""Run T11 Ours full generation/evaluation or its independent verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.tastemolnet_ours_full import (  # noqa: E402
    PASS_MARKER,
    TasteOursFullError,
    load_authority,
    run_science,
    verify_and_publish,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--science-root", type=Path, required=True)
    parser.add_argument("--final-root", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ppo-root", type=Path)
    parser.add_argument("--gnn-checkpoint", type=Path)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--calibration-csv", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--molclr-root", type=Path)
    parser.add_argument("--molclr-checkpoint", type=Path)
    parser.add_argument("--threshold-contract", type=Path, required=True)
    parser.add_argument("--wnode-cache-db", type=Path)
    parser.add_argument("--node-embedding-cache-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _required(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    missing = [name.replace("_", "-") for name in names if getattr(args, name) is None]
    if missing:
        raise TasteOursFullError("T11 is missing required arguments: " + ", ".join(missing))


def run(args: argparse.Namespace) -> int:
    if args.config.resolve(strict=True) != (REPO_ROOT / "configs/hpc.yaml").resolve(strict=True):
        raise TasteOursFullError("T11 requires configs/hpc.yaml")
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise TasteOursFullError("T11 requires fail-closed inference override")
    if args.verify_only:
        _required(args, ("final_root",))
        manifest = verify_and_publish(
            science_root=args.science_root,
            final_root=args.final_root,
            threshold_contract=args.threshold_contract,
        )
        print(json.dumps(manifest, sort_keys=True), flush=True)
        print(PASS_MARKER, flush=True)
        return 0
    _required(
        args,
        (
            "ppo_root", "gnn_checkpoint", "train_csv", "calibration_csv",
            "test_csv", "molclr_root", "molclr_checkpoint", "wnode_cache_db",
            "node_embedding_cache_dir",
        ),
    )
    authority = load_authority(
        ppo_root=args.ppo_root,
        gnn_checkpoint=args.gnn_checkpoint,
        train_csv=args.train_csv,
        calibration_csv=args.calibration_csv,
        test_csv=args.test_csv,
        molclr_root=args.molclr_root,
        molclr_checkpoint=args.molclr_checkpoint,
        threshold_contract=args.threshold_contract,
    )
    manifest = run_science(
        authority=authority,
        output_dir=args.science_root,
        resume=args.resume,
        device=args.device,
        wnode_cache_db=args.wnode_cache_db,
        node_embedding_cache_dir=args.node_embedding_cache_dir,
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)
    print("[TASTE_T11_OURS_FULL_SEALED]", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
