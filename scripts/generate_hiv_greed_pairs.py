#!/usr/bin/env python3
"""Generate train/val/test HIV graph pairs for GREED-style GED training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.greed_distance.pair_generation import generate_pair_rows, write_pairs_csv  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--graphs-jsonl", default="outputs/hpc/greed_hiv/dataset/graphs.jsonl")
    parser.add_argument("--out-dir", default="outputs/hpc/greed_hiv/pairs")
    parser.add_argument("--num-train-pairs", type=int, default=5000)
    parser.add_argument("--num-val-pairs", type=int, default=1000)
    parser.add_argument("--num-test-pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--include-deletion-pairs", action="store_true", default=True)
    parser.add_argument("--include-fullgraph-pairs", action="store_true", default=True)
    parser.add_argument("--include-random-pairs", action="store_true", default=True)
    parser.add_argument("--no-deletion-pairs", action="store_false", dest="include_deletion_pairs")
    parser.add_argument("--no-fullgraph-pairs", action="store_false", dest="include_fullgraph_pairs")
    parser.add_argument("--no-random-pairs", action="store_false", dest="include_random_pairs")
    parser.add_argument("--ours-selected-path", default="")
    parser.add_argument("--gt-fullgraph-candidates-path", default="")
    parser.add_argument("--max-parents", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"train": args.num_train_pairs, "val": args.num_val_pairs, "test": args.num_test_pairs}
    summary = {}
    for offset, (split, count) in enumerate(counts.items()):
        rows = generate_pair_rows(
            graphs_jsonl=args.graphs_jsonl,
            split=split,
            num_pairs=int(count),
            seed=int(args.seed) + offset,
            include_deletion_pairs=bool(args.include_deletion_pairs),
            include_fullgraph_pairs=bool(args.include_fullgraph_pairs),
            include_random_pairs=bool(args.include_random_pairs),
            ours_selected_path=args.ours_selected_path or None,
            gt_fullgraph_candidates_path=args.gt_fullgraph_candidates_path or None,
            max_parents=args.max_parents,
            max_candidates=args.max_candidates,
        )
        output_csv = out_dir / f"{split}_pairs.csv"
        write_pairs_csv(output_csv, rows)
        summary[split] = {"output_csv": str(output_csv), "num_pairs": len(rows)}
        print(f"[PAIR_GENERATION] split={split} num_pairs={len(rows)} output={output_csv}", flush=True)
    (out_dir / "pair_generation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
