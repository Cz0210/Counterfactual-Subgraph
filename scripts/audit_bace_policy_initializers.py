#!/usr/bin/env python3
"""Classify BACE policy initializers and freeze one clean selection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train.bace_policy_init import (  # noqa: E402
    atomic_text,
    audit_policy_initializer,
    select_policy_initializer,
    sha256_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Accepted for paired Slurm compatibility; audit inputs stay explicit.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Initializer path or KIND=path (KIND may be raw_base). Repeatable.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, default=None)
    parser.add_argument("--pass-path", type=Path, default=None)
    return parser


def _candidate(value: str) -> tuple[str | None, Path]:
    kind, separator, raw_path = str(value).partition("=")
    if separator and kind in {"raw_base", "chemllm_base", "adapter", "sft"}:
        return kind, Path(raw_path).expanduser()
    return None, Path(value).expanduser()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    targets = [args.output_csv, args.selection_json]
    if args.audit_manifest is not None:
        targets.append(args.audit_manifest)
    if args.pass_path is not None:
        targets.append(args.pass_path)
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "BACE initializer audit outputs must be fresh: " + ", ".join(existing)
        )
    rows = [
        audit_policy_initializer(path, kind_hint=kind).to_dict()
        for kind, path in (_candidate(value) for value in args.candidate)
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    selected = select_policy_initializer(rows)
    args.selection_json.parent.mkdir(parents=True, exist_ok=True)
    args.selection_json.write_text(
        json.dumps(
            {
                "schema_version": "bace_policy_initializer_selection_v1",
                "selection_rule": (
                    "provenance_clean_then_parse_direct_then_oracle_evaluable_"
                    "then_strict_flip_then_diversity"
                ),
                "calibration_loaded": False,
                "test_loaded": False,
                "selected": selected,
                "candidates": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.audit_manifest is not None:
        args.audit_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.audit_manifest.write_text(
            json.dumps(
                {
                    "schema_version": "bace_policy_initializer_audit_v1",
                    "status": "PASS",
                    "candidate_count": len(rows),
                    "eligible_count": sum(1 for row in rows if row["eligible"]),
                    "output_csv": str(args.output_csv.resolve()),
                    "output_csv_sha256": sha256_file(args.output_csv),
                    "selection_json": str(args.selection_json.resolve()),
                    "selection_json_sha256": sha256_file(args.selection_json),
                    "selection_rule": (
                        "provenance_clean_then_parse_direct_then_oracle_evaluable_"
                        "then_strict_flip_then_diversity"
                    ),
                    "rf_oracle_used": False,
                    "calibration_loaded": False,
                    "test_loaded": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.pass_path is not None:
        args.pass_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_text(args.pass_path, "[BACE_POLICY_PROVENANCE_AUDIT_PASS]\n")
    print(json.dumps(selected, sort_keys=True), flush=True)
    print("[BACE_POLICY_PROVENANCE_AUDIT_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
