#!/usr/bin/env python3
"""Build or audit the inductive held-out molecule evaluation protocol."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.molecular_split import SPLIT_NAMES, file_sha256  # noqa: E402
from src.eval.heldout_molecule_protocol import (  # noqa: E402
    build_heldout_protocol_manifest,
    transductive_vs_heldout_schema,
)
from src.eval.split_leakage_audit import audit_split_files  # noqa: E402


PROTOCOLS = ("standard", "heldout_molecule", "cross_scaffold")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("protocol", "combine"), required=True)
    parser.add_argument("--dataset", default="BBBP")
    parser.add_argument("--method", default="Ours")
    parser.add_argument("--train-csv")
    parser.add_argument("--val-csv")
    parser.add_argument("--calibration-csv")
    parser.add_argument("--test-csv")
    parser.add_argument("--split-manifest")
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        help="Combine mode: protocol=path/to/summary.json (repeatable).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Held-out split is empty: {path}")
    return rows


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def build_protocol(
    *,
    dataset: str,
    method: str,
    split_paths: Mapping[str, Path],
    split_manifest: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    leakage = audit_split_files(
        split_paths,
        protocol="inductive_heldout_molecule_v1",
        require_scaffold_disjoint=False,
        candidate_source_splits=("train", "val"),
        selector_source_splits=("calibration",),
        threshold_source_split="calibration",
    )
    parent_ids: dict[str, Sequence[str]] = {}
    for split, path in split_paths.items():
        values = [str(row.get("molecule_id") or "") for row in _rows(path)]
        if any(not value for value in values):
            raise ValueError(f"Held-out split lacks molecule_id: {path}")
        parent_ids[split] = values
    manifest = build_heldout_protocol_manifest(
        dataset=dataset,
        method=method,
        split_manifest_sha256=file_sha256(split_manifest),
        parent_ids_by_split=parent_ids,
    )
    return manifest, leakage


def _summary_row(protocol: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown comparison protocol: {protocol}")
    return {
        "dataset": payload.get("dataset"),
        "method": payload.get("method"),
        "protocol": protocol,
        "coverage": payload.get("coverage"),
        "cost": payload.get("cost"),
        "cf_drop": payload.get("cf_drop"),
        "flip_rate": payload.get("flip_rate"),
        "valid_rate": payload.get("valid_rate"),
        "structural_redundancy": payload.get("structural_redundancy"),
        "coverage_redundancy": payload.get("coverage_redundancy"),
        "status": payload.get("status", "NOT_RUN"),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    if args.mode == "protocol":
        supplied = {
            "train": args.train_csv,
            "val": args.val_csv,
            "calibration": args.calibration_csv,
            "test": args.test_csv,
        }
        missing = sorted(key for key, value in supplied.items() if not value)
        if missing or not args.split_manifest:
            raise ValueError(
                f"Held-out protocol requires four split CSVs and --split-manifest; missing={missing}."
            )
        split_paths = {key: Path(value).expanduser().resolve() for key, value in supplied.items()}
        split_manifest = Path(args.split_manifest).expanduser().resolve()
        manifest, leakage = build_protocol(
            dataset=args.dataset,
            method=args.method,
            split_paths=split_paths,
            split_manifest=split_manifest,
        )
        if args.validate_only or args.dry_run:
            print(json.dumps({**manifest, "status": "VALIDATED_NOT_RUN", "formal_output_written": False}, sort_keys=True))
            return 0
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Held-out protocol output is non-empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        _atomic(output / "protocol_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _atomic(output / "split_leakage_audit.json", json.dumps(leakage, indent=2, sort_keys=True) + "\n")
        print("[HELDOUT_MOLECULE_PROTOCOL_PASS]")
        return 0
    parsed: dict[str, Path] = {}
    for value in args.summary:
        if "=" not in value:
            raise ValueError("--summary must use protocol=path syntax.")
        protocol, raw_path = value.split("=", 1)
        if protocol in parsed:
            raise ValueError(f"Duplicate held-out comparison protocol: {protocol}")
        parsed[protocol] = Path(raw_path).expanduser().resolve()
    if set(parsed) != set(PROTOCOLS):
        raise ValueError(f"Combined held-out report requires exactly {PROTOCOLS}.")
    rows = [_summary_row(protocol, json.loads(parsed[protocol].read_text(encoding="utf-8"))) for protocol in PROTOCOLS]
    if args.validate_only or args.dry_run:
        print(json.dumps({"status": "VALIDATED_NOT_RUN", "protocols": list(PROTOCOLS), "formal_output_written": False}, sort_keys=True))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    fields = transductive_vs_heldout_schema()
    csv_path = output / "transductive_vs_heldout_summary.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, csv_path)
    _atomic(output / "transductive_vs_heldout_summary.json", json.dumps({"schema_version":"transductive_vs_heldout_v1","rows":rows}, indent=2, sort_keys=True) + "\n")
    print("[HELDOUT_COMBINED_REPORT_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
