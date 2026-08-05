#!/usr/bin/env python3
"""Decode and RF-filter COMRECGC common-recourse graph medoids."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.exporter import export_representatives  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("aids", "mutagenicity"), required=True)
    parser.add_argument("--common-recourse-dir", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--atom-vocabulary-json")
    parser.add_argument("--dataset-summary-json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--require-top-k", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    vocabulary_path = args.atom_vocabulary_json or args.dataset_summary_json
    if not vocabulary_path:
        raise ValueError("Provide --atom-vocabulary-json or --dataset-summary-json.")
    vocabulary_payload = json.loads(Path(vocabulary_path).read_text(encoding="utf-8"))
    if args.dataset_summary_json:
        vocabulary_payload = vocabulary_payload.get("feature_atomic_numbers")
    if isinstance(vocabulary_payload, dict):
        atom_vocabulary = [
            key for key, _value in sorted(vocabulary_payload.items(), key=lambda item: int(item[1]))
        ]
    elif isinstance(vocabulary_payload, list):
        atom_vocabulary = vocabulary_payload
    else:
        raise ValueError("Atom vocabulary JSON must be a mapping or list.")
    summary = export_representatives(
        dataset=args.dataset,
        common_recourse_dir=args.common_recourse_dir,
        teacher_path=args.teacher_path,
        atom_vocabulary=atom_vocabulary,
        output_dir=args.output_dir,
        top_k=args.top_k,
        require_top_k=args.require_top_k,
        resume=args.resume,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
