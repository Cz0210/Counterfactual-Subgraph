#!/usr/bin/env python3
"""Audit fragment size/chemistry distributions in SFT data or inference outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.fragment_quality import (
    AuditOptions,
    audit_fragment_records,
    build_detail_rows,
    build_summary_payload,
    format_role_summary_lines,
    load_fragment_source_records,
    select_top_k_by_atom_ratio,
    write_detail_csv,
    write_jsonl_rows,
)


DEFAULT_SUMMARY_JSON = REPO_ROOT / "outputs" / "hpc" / "logs" / "sft_fragment_distribution_summary.json"
DEFAULT_DETAILS_CSV = REPO_ROOT / "outputs" / "hpc" / "logs" / "sft_fragment_distribution_details.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Accepted for HPC wrapper compatibility. This audit script uses explicit CLI overrides only.",
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input path. Supports current SFT jsonl datasets and SFT inference txt/jsonl outputs.",
    )
    parser.add_argument(
        "--summary-json",
        default=str(DEFAULT_SUMMARY_JSON),
        help="Path to the output summary JSON.",
    )
    parser.add_argument(
        "--details-csv",
        default=str(DEFAULT_DETAILS_CSV),
        help="Path to the per-sample detail CSV.",
    )
    parser.add_argument(
        "--near-parent-threshold",
        type=float,
        default=0.8,
        help="Atom-ratio threshold used for near_parent_rate.",
    )
    parser.add_argument(
        "--tiny-fragment-threshold",
        type=float,
        default=0.08,
        help="Atom-ratio threshold used for tiny_fragment_rate.",
    )
    parser.add_argument(
        "--mid-size-min",
        type=float,
        default=0.1,
        help="Lower atom-ratio bound used for mid_size_rate.",
    )
    parser.add_argument(
        "--mid-size-max",
        type=float,
        default=0.6,
        help="Upper atom-ratio bound used for mid_size_rate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many largest/smallest generated-fragment samples to include.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Optional starting record index after input adaptation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional max number of adapted records to process. 0 means all records after --start-index.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print one progress log every N processed records. Set 0 to disable.",
    )
    parser.add_argument(
        "--skip-deleteability-check",
        action="store_true",
        help="Skip fragment deletion checks to avoid slow samples.",
    )
    parser.add_argument(
        "--skip-substructure-check",
        action="store_true",
        help="Skip substructure existence checks to avoid slow samples.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Raise on the first per-sample audit exception instead of recording and continuing.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_paths = [Path(path).expanduser().resolve() for path in args.input]
    records = load_fragment_source_records(input_paths)
    total_loaded_records = len(records)
    start_index = max(args.start_index, 0)
    if args.max_samples and args.max_samples > 0:
        end_index = start_index + args.max_samples
        records = records[start_index:end_index]
    else:
        end_index = total_loaded_records
        records = records[start_index:]
    slow_events = []
    audit_options = AuditOptions(
        skip_deleteability_check=bool(args.skip_deleteability_check),
        skip_substructure_check=bool(args.skip_substructure_check),
        fail_fast=bool(args.fail_fast),
    )
    audited_records = audit_fragment_records(
        records,
        options=audit_options,
        progress_every=max(args.progress_every, 0),
        slow_events=slow_events,
    )
    summary = build_summary_payload(
        audited_records,
        input_paths=input_paths,
        near_parent_threshold=args.near_parent_threshold,
        tiny_fragment_threshold=args.tiny_fragment_threshold,
        mid_size_min=args.mid_size_min,
        mid_size_max=args.mid_size_max,
    )
    summary["total_loaded_records"] = total_loaded_records
    summary["selected_window"] = {
        "start_index": start_index,
        "max_samples": args.max_samples,
        "selected_count": len(records),
        "end_index_exclusive": min(end_index, total_loaded_records),
    }
    summary["audit_options"] = {
        "skip_deleteability_check": audit_options.skip_deleteability_check,
        "skip_substructure_check": audit_options.skip_substructure_check,
        "fail_fast": audit_options.fail_fast,
        "progress_every": max(args.progress_every, 0),
    }
    summary["slow_event_count"] = len(slow_events)
    summary["generated_top_k_largest_atom_ratio"] = select_top_k_by_atom_ratio(
        audited_records,
        role="generated",
        k=args.top_k,
        largest=True,
    )
    summary["generated_top_k_smallest_atom_ratio"] = select_top_k_by_atom_ratio(
        audited_records,
        role="generated",
        k=args.top_k,
        largest=False,
    )

    detail_rows = build_detail_rows(audited_records)
    details_csv = Path(args.details_csv).expanduser().resolve()
    write_detail_csv(details_csv, detail_rows)
    slow_log_jsonl = details_csv.with_suffix(".slow.jsonl")
    write_jsonl_rows(slow_log_jsonl, [event.to_dict() for event in slow_events])
    summary["slow_event_jsonl"] = str(slow_log_jsonl)

    summary_json = Path(args.summary_json).expanduser().resolve()
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    print("SFT fragment distribution audit completed.")
    print(f"Inputs: {', '.join(str(path) for path in input_paths)}")
    print(
        f"Selected records: {summary['selected_window']['selected_count']} "
        f"(loaded={summary['total_loaded_records']} "
        f"window={summary['selected_window']['start_index']}:"
        f"{summary['selected_window']['end_index_exclusive']})"
    )
    print(f"Source kinds: {summary['source_kinds']}")
    for role_name in ("reference", "generated"):
        for line in format_role_summary_lines(summary[role_name]):
            print(line)
    print(f"Summary JSON: {summary_json}")
    print(f"Detail CSV: {details_csv}")
    print(f"Slow-event JSONL: {slow_log_jsonl}")

    if slow_events:
        print(
            f"Slow-stage events recorded: {len(slow_events)} "
            f"(threshold={audit_options.slow_stage_threshold_sec:.2f}s)"
        )
        for event in slow_events:
            sys.stderr.write(
                json.dumps(
                    {
                        "kind": "slow_audit_event",
                        **event.to_dict(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    if summary["generated_top_k_largest_atom_ratio"]:
        print("Top generated fragments by atom_ratio (largest):")
        for entry in summary["generated_top_k_largest_atom_ratio"]:
            print(
                f"  record_index={entry['record_index']} sample_id={entry['sample_id']} "
                f"atom_ratio={entry['atom_ratio']} "
                f"generated={entry['generated_fragment']} parent={entry['parent_smiles']}"
            )
    if summary["generated_top_k_smallest_atom_ratio"]:
        print("Top generated fragments by atom_ratio (smallest):")
        for entry in summary["generated_top_k_smallest_atom_ratio"]:
            print(
                f"  record_index={entry['record_index']} sample_id={entry['sample_id']} "
                f"atom_ratio={entry['atom_ratio']} "
                f"generated={entry['generated_fragment']} parent={entry['parent_smiles']}"
            )


if __name__ == "__main__":
    main()
