#!/usr/bin/env python3
"""Build a minimal label-specific PPO prompt CSV from a shared source dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ppo_prompt_dataset import load_ppo_prompt_records  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Accepted for HPC wrapper parity.")
    parser.add_argument("--source-path", required=True, help="Source CSV/JSONL containing both labels.")
    parser.add_argument("--out-csv", required=True, help="Output minimal PPO prompt CSV.")
    parser.add_argument("--out-json", required=True, help="Output summary JSON.")
    parser.add_argument("--target-label", type=int, choices=(0, 1), required=True)
    parser.add_argument("--label-col", default="label", help="Preferred label column.")
    parser.add_argument("--smiles-col", default="smiles", help="Preferred SMILES column.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows to keep. 0 keeps all.")
    return parser


def build_label_prompt_csv(
    *,
    source_path: str | Path,
    out_csv: str | Path,
    out_json: str | Path,
    target_label: int,
    label_col: str,
    smiles_col: str,
    limit: int = 0,
) -> dict[str, object]:
    records, metadata = load_ppo_prompt_records(
        source_path,
        label_col=label_col,
        smiles_col=smiles_col,
        target_label=int(target_label),
        limit=int(limit),
    )
    out_csv_path = Path(out_csv).expanduser().resolve()
    out_json_path = Path(out_json).expanduser().resolve()
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    label_counts: Counter[int] = Counter()
    with out_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["smiles", "label"])
        writer.writeheader()
        for record in records:
            label = int(record.label)
            label_counts[label] += 1
            writer.writerow({"smiles": record.parent_smiles, "label": label})

    summary: dict[str, object] = {
        "source_path": str(Path(source_path).expanduser().resolve()),
        "out_csv": str(out_csv_path),
        "target_label": int(target_label),
        "num_rows": len(records),
        "label_counts": {str(label): int(count) for label, count in sorted(label_counts.items())},
        "output_schema": ["smiles", "label"],
        "requested_label_col": str(label_col),
        "requested_smiles_col": str(smiles_col),
        "limit": int(limit),
        "source_metadata": metadata,
    }
    out_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = build_label_prompt_csv(
        source_path=args.source_path,
        out_csv=args.out_csv,
        out_json=args.out_json,
        target_label=int(args.target_label),
        label_col=str(args.label_col),
        smiles_col=str(args.smiles_col),
        limit=int(args.limit),
    )
    metadata = summary["source_metadata"]
    assert isinstance(metadata, dict)
    print(
        "[BUILD_LABEL_PROMPTS_CONFIG] "
        f"target_label={summary['target_label']} source_path={summary['source_path']} "
        f"label_col={summary['requested_label_col']} smiles_col={summary['requested_smiles_col']} "
        f"limit={summary['limit']}"
    )
    print(
        "[BUILD_LABEL_PROMPTS_SOURCE] "
        f"dataset_format={metadata.get('dataset_format')} "
        f"resolved_label_col={metadata.get('resolved_label_col')} "
        f"resolved_smiles_col={metadata.get('resolved_smiles_col')} "
        f"input_row_count={metadata.get('input_row_count')}"
    )
    print(
        "[BUILD_LABEL_PROMPTS_COUNTS] "
        f"target_label={summary['target_label']} num_rows={summary['num_rows']} "
        f"label_counts={summary['label_counts']} dropped_counts={metadata.get('dropped_counts')}"
    )
    print(
        "[BUILD_LABEL_PROMPTS_OUTPUT] "
        f"out_csv={summary['out_csv']} out_json={Path(args.out_json).expanduser().resolve()} "
        f"output_schema={summary['output_schema']}"
    )


if __name__ == "__main__":
    main()
