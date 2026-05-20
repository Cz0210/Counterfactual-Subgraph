#!/usr/bin/env python3
"""Check label balance and parent-molecule mix for one unified PPO prompt CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.unified_ppo_prompts import check_unified_prompt_balance  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Accepted for HPC wrapper parity.")
    parser.add_argument("--dataset-path", required=True, help="Unified PPO prompt CSV path.")
    parser.add_argument("--out-json", required=True, help="Output summary JSON.")
    parser.add_argument("--block-size", type=int, default=50, help="Rows per summary block.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = check_unified_prompt_balance(
        args.dataset_path,
        block_size=int(args.block_size),
    )
    out_json = Path(args.out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"dataset_path: {Path(args.dataset_path).expanduser().resolve()}")
    print(f"out_json: {out_json}")
    print(f"total_rows: {summary['total_rows']}")
    print(f"label_counts: {summary['label_counts']}")


if __name__ == "__main__":
    main()
