#!/usr/bin/env python3
"""Evaluate official GCFExplainer selected fullgraphs in native official space."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.gcf_official_adapter import resolve_official_repo  # noqa: E402
from src.eval.gcf_native_fullgraph_eval import evaluate_native_fullgraph  # noqa: E402


def _floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--official-repo", default=None)
    parser.add_argument("--selected-graphs-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", default="aids")
    parser.add_argument(
        "--direction",
        choices=["official_default_label0_to_label1", "mirrored_label1_to_label0"],
        default="official_default_label0_to_label1",
    )
    parser.add_argument("--thresholds", default="0.05,0.10,0.20")
    parser.add_argument("--top-k-list", default="1,5,10,20,50,100")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = evaluate_native_fullgraph(
        official_repo=resolve_official_repo(args.official_repo),
        selected_graphs_path=args.selected_graphs_path,
        out_dir=args.out_dir,
        dataset=args.dataset,
        direction=args.direction,
        thresholds=_floats(args.thresholds),
        top_k_list=_ints(args.top_k_list),
        device=args.device,
        batch_size=args.batch_size,
    )
    print("[GCF_NATIVE_EVAL_DONE]", flush=True)
    print(json.dumps(result["config"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

