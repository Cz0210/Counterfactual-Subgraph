#!/usr/bin/env python3
"""Label HIV GREED pairs with normalized GED targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.greed_distance.ged_labeling import label_pairs_csv  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--pairs-dir", default="outputs/hpc/greed_hiv/pairs")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--allow-networkx-debug", action="store_true")
    parser.add_argument("--networkx-timeout", type=float, default=1.0)
    parser.add_argument("--fullgraph-label-mode", choices=["bounded_approx", "networkx_debug", "fail"], default="bounded_approx")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pair_dir = Path(args.pairs_dir).expanduser().resolve()
    summaries = {}
    for split in [part.strip() for part in args.splits.split(",") if part.strip()]:
        summary = label_pairs_csv(
            input_csv=pair_dir / f"{split}_pairs.csv",
            output_csv=pair_dir / f"{split}_pairs_labeled.csv",
            allow_networkx_debug=bool(args.allow_networkx_debug),
            networkx_timeout=float(args.networkx_timeout),
            fullgraph_label_mode=str(args.fullgraph_label_mode),
        )
        summaries[split] = summary
        print(f"[GED_LABEL] split={split} ok={summary['num_label_ok']}/{summary['num_pairs']} output={summary['output_csv']}", flush=True)
    (pair_dir / "ged_labeling_summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
