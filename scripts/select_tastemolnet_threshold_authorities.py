#!/usr/bin/env python3
"""Select frozen TasteMolNet NeuroSED and shared WNode thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.tastemolnet_threshold_authorities import (  # noqa: E402
    run_tastemolnet_threshold_authority_selector,
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--t3-root", type=_absolute, required=True)
    parser.add_argument("--t4-root", type=_absolute, required=True)
    parser.add_argument("--graph-cache-root", type=_absolute, required=True)
    parser.add_argument("--managed-neurosed-root", type=_absolute, required=True)
    parser.add_argument("--official-gcf-root", type=_absolute, required=True)
    parser.add_argument("--molclr-root", type=_absolute, required=True)
    parser.add_argument("--molclr-checkpoint", type=_absolute, required=True)
    parser.add_argument("--output-root", type=_absolute, required=True)
    parser.add_argument("--wnode-cache-db", type=_absolute, required=True)
    parser.add_argument("--node-embedding-cache-dir", type=_absolute, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config.is_file():
        raise ValueError("--config is unavailable")
    if "inference.fallback_to_heuristic=false" not in args.set:
        raise ValueError("selector requires inference.fallback_to_heuristic=false")
    result = run_tastemolnet_threshold_authority_selector(
        t3_root=args.t3_root,
        t4_root=args.t4_root,
        graph_cache_root=args.graph_cache_root,
        managed_neurosed_root=args.managed_neurosed_root,
        official_gcf_root=args.official_gcf_root,
        molclr_root=args.molclr_root,
        molclr_checkpoint=args.molclr_checkpoint,
        output_root=args.output_root,
        wnode_cache_db=args.wnode_cache_db,
        node_embedding_cache_dir=args.node_embedding_cache_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, sort_keys=True, ensure_ascii=True), flush=True)
    print(result["marker"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
