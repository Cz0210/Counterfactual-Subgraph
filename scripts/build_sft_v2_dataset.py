#!/usr/bin/env python3
"""Filter and rebalance weak SFT data into a PPO-friendly SFT v2 dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.fragment_quality import (
    audit_fragment_records,
    bucket_name_for_atom_ratio,
    build_training_row_from_record,
    load_fragment_source_records,
)
from src.utils.io import write_jsonl


DEFAULT_INPUT = REPO_ROOT / "data" / "sft_train.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "sft_train_v2.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input SFT jsonl path. The script auto-adapts current repository SFT row formats.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output SFT v2 jsonl path compatible with scripts/train_sft.py.",
    )
    parser.add_argument(
        "--summary-json",
        help="Optional output path for the summary JSON. Defaults to <output>.summary.json.",
    )
    parser.add_argument(
        "--report-path",
        help="Optional output path for the filter report. Defaults to <output>.report.txt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for bucket rebalancing.",
    )
    parser.add_argument(
        "--min-atom-ratio",
        type=float,
        default=0.08,
        help="Minimum allowed fragment atom_ratio.",
    )
    parser.add_argument(
        "--max-atom-ratio",
        type=float,
        default=0.6,
        help="Maximum allowed fragment atom_ratio.",
    )
    parser.add_argument(
        "--max-near-parent-ratio",
        type=float,
        default=0.8,
        help="Fragments at or above this atom_ratio are treated as near-parent.",
    )
    drop_full_parent_group = parser.add_mutually_exclusive_group()
    drop_full_parent_group.add_argument(
        "--drop-full-parent",
        dest="drop_full_parent",
        action="store_true",
        default=True,
        help="Drop fragments whose canonical core equals the canonical parent.",
    )
    drop_full_parent_group.add_argument(
        "--keep-full-parent",
        dest="drop_full_parent",
        action="store_false",
        help="Keep full-parent fragments.",
    )
    require_valid_group = parser.add_mutually_exclusive_group()
    require_valid_group.add_argument(
        "--require-valid",
        dest="require_valid",
        action="store_true",
        default=True,
        help="Require chemically valid fragments.",
    )
    require_valid_group.add_argument(
        "--allow-invalid",
        dest="require_valid",
        action="store_false",
        help="Keep chemically invalid fragments.",
    )
    require_substructure_group = parser.add_mutually_exclusive_group()
    require_substructure_group.add_argument(
        "--require-substructure",
        dest="require_substructure",
        action="store_true",
        default=True,
        help="Require fragments to match the parent molecule.",
    )
    require_substructure_group.add_argument(
        "--allow-non-substructure",
        dest="require_substructure",
        action="store_false",
        help="Keep non-substructure fragments.",
    )
    require_residual_group = parser.add_mutually_exclusive_group()
    require_residual_group.add_argument(
        "--require-nonempty-residual",
        dest="require_nonempty_residual",
        action="store_true",
        default=True,
        help="Require deletion to produce a non-empty residual molecule.",
    )
    require_residual_group.add_argument(
        "--allow-empty-residual",
        dest="require_nonempty_residual",
        action="store_false",
        help="Keep fragments whose deletion removes the whole parent.",
    )
    parser.add_argument(
        "--rebalance-by-atom-ratio",
        action="store_true",
        help="Downsample small/medium/large buckets toward a near-balanced mix.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    summary_path = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else output_path.with_suffix(".summary.json")
    )
    report_path = (
        Path(args.report_path).expanduser().resolve()
        if args.report_path
        else output_path.with_suffix(".report.txt")
    )

    records = load_fragment_source_records([input_path])
    audited_records = audit_fragment_records(records)

    kept_items: list[dict[str, Any]] = []
    dropped_items: list[dict[str, Any]] = []
    pre_filter_bucket_counts = Counter()
    post_filter_bucket_counts = Counter()
    drop_reason_counts = Counter()
    fragment_source_counts = Counter()

    for audited in audited_records:
        inspection, fragment_source = _pick_training_fragment(audited)
        if inspection is None:
            drop_reason_counts["missing_reference_or_generated_fragment"] += 1
            dropped_items.append(
                {
                    "sample_id": audited.record.sample_id,
                    "fragment_source": None,
                    "drop_reasons": ["missing_reference_or_generated_fragment"],
                }
            )
            continue

        fragment_source_counts[fragment_source] += 1
        bucket_name = bucket_name_for_atom_ratio(inspection.atom_ratio)
        if bucket_name:
            pre_filter_bucket_counts[bucket_name] += 1

        drop_reasons = _collect_drop_reasons(inspection, args)
        if drop_reasons:
            for reason in drop_reasons:
                drop_reason_counts[reason] += 1
            dropped_items.append(
                {
                    "sample_id": audited.record.sample_id,
                    "fragment_source": fragment_source,
                    "drop_reasons": drop_reasons,
                    "atom_ratio": inspection.atom_ratio,
                }
            )
            continue

        meta = {
            "raw_fragment": inspection.raw_fragment,
            "core_fragment": inspection.core_fragment,
            "atom_ratio": inspection.atom_ratio,
            "is_full_parent": inspection.is_full_parent,
            "residual_nonempty": inspection.residual_nonempty,
            "fragment_source": fragment_source,
        }
        output_row = build_training_row_from_record(
            audited.record,
            output_fragment=inspection.raw_fragment,
            meta=meta,
        )
        kept_items.append(
            {
                "row": output_row,
                "sample_id": audited.record.sample_id,
                "bucket_name": bucket_name,
                "atom_ratio": inspection.atom_ratio,
            }
        )
        if bucket_name:
            post_filter_bucket_counts[bucket_name] += 1

    rebalance_summary = {
        "enabled": bool(args.rebalance_by_atom_ratio),
        "before_counts": dict(post_filter_bucket_counts),
        "after_counts": dict(post_filter_bucket_counts),
        "target_per_bucket": None,
    }
    if args.rebalance_by_atom_ratio and kept_items:
        kept_items, rebalance_summary = _rebalance_items(
            kept_items,
            seed=args.seed,
        )

    if not kept_items:
        raise SystemExit("No rows remained after SFT v2 filtering.")

    write_jsonl(output_path, (item["row"] for item in kept_items))

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_input_records": len(audited_records),
        "kept_records": len(kept_items),
        "dropped_records": len(dropped_items),
        "fragment_source_counts": dict(fragment_source_counts),
        "filter_thresholds": {
            "min_atom_ratio": args.min_atom_ratio,
            "max_atom_ratio": args.max_atom_ratio,
            "max_near_parent_ratio": args.max_near_parent_ratio,
            "drop_full_parent": args.drop_full_parent,
            "require_valid": args.require_valid,
            "require_substructure": args.require_substructure,
            "require_nonempty_residual": args.require_nonempty_residual,
        },
        "drop_reason_counts": dict(drop_reason_counts),
        "rebalance": rebalance_summary,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_report(
            input_path=input_path,
            output_path=output_path,
            summary=summary,
        ),
        encoding="utf-8",
    )

    print("SFT v2 dataset build completed.")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(
        f"Kept {summary['kept_records']} / {summary['total_input_records']} "
        f"records after filtering."
    )
    print(f"Drop reasons: {summary['drop_reason_counts']}")
    print(f"Rebalance: {summary['rebalance']}")
    print(f"Summary JSON: {summary_path}")
    print(f"Filter report: {report_path}")


def _pick_training_fragment(audited) -> tuple[Any | None, str]:
    if audited.reference is not None:
        return audited.reference, "reference"
    if audited.generated is not None:
        return audited.generated, "generated"
    return None, "missing"


def _collect_drop_reasons(inspection, args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    if args.require_valid and not inspection.chemically_valid:
        reasons.append("invalid_fragment")
    if args.require_substructure and not inspection.substructure_ok:
        reasons.append("not_substructure")
    if args.require_nonempty_residual and not inspection.residual_nonempty:
        reasons.append("empty_or_missing_residual")
    if inspection.atom_ratio is None:
        reasons.append("missing_atom_ratio")
        return reasons
    if inspection.atom_ratio < args.min_atom_ratio:
        reasons.append("below_min_atom_ratio")
    if inspection.atom_ratio > args.max_atom_ratio:
        reasons.append("above_max_atom_ratio")
    if inspection.atom_ratio >= args.max_near_parent_ratio:
        reasons.append("near_parent_ratio")
    if args.drop_full_parent and inspection.is_full_parent:
        reasons.append("full_parent_fragment")
    return reasons


def _rebalance_items(
    items: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unbucketed: list[dict[str, Any]] = []
    for item in items:
        bucket_name = item.get("bucket_name")
        if bucket_name:
            grouped[bucket_name].append(item)
        else:
            unbucketed.append(item)

    non_empty_counts = [len(rows) for rows in grouped.values() if rows]
    if not non_empty_counts:
        return items, {
            "enabled": True,
            "before_counts": {},
            "after_counts": {},
            "target_per_bucket": None,
            "note": "No small/medium/large buckets were available for rebalancing.",
        }

    target_per_bucket = min(non_empty_counts)
    kept: list[dict[str, Any]] = []
    after_counts: dict[str, int] = {}
    before_counts = {bucket: len(rows) for bucket, rows in grouped.items()}
    for bucket, rows in grouped.items():
        bucket_rows = list(rows)
        rng.shuffle(bucket_rows)
        selected = bucket_rows[:target_per_bucket]
        kept.extend(selected)
        after_counts[bucket] = len(selected)

    kept.extend(unbucketed)
    kept.sort(key=lambda item: str(item["sample_id"]))
    return kept, {
        "enabled": True,
        "before_counts": before_counts,
        "after_counts": after_counts,
        "target_per_bucket": target_per_bucket,
        "kept_unbucketed": len(unbucketed),
    }


def _render_report(
    *,
    input_path: Path,
    output_path: Path,
    summary: dict[str, Any],
) -> str:
    lines = [
        "SFT v2 Filter Report",
        "====================",
        f"Input: {input_path}",
        f"Output: {output_path}",
        f"Total input records: {summary['total_input_records']}",
        f"Kept records: {summary['kept_records']}",
        f"Dropped records: {summary['dropped_records']}",
        f"Fragment source counts: {summary['fragment_source_counts']}",
        "",
        "Thresholds",
        "----------",
    ]
    for key, value in summary["filter_thresholds"].items():
        lines.append(f"{key}: {value}")
    lines.extend(
        [
            "",
            "Drop Reasons",
            "------------",
        ]
    )
    if summary["drop_reason_counts"]:
        for key, value in sorted(summary["drop_reason_counts"].items()):
            lines.append(f"{key}: {value}")
    else:
        lines.append("None")
    lines.extend(
        [
            "",
            "Rebalance",
            "---------",
            json.dumps(summary["rebalance"], ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Before/After",
            "------------",
            f"input={summary['total_input_records']} kept={summary['kept_records']} dropped={summary['dropped_records']}",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
