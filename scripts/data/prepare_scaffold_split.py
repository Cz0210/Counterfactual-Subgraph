#!/usr/bin/env python3
"""Create a deterministic four-way Bemis-Murcko split from standardized CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.molecular_split import (  # noqa: E402
    DEFAULT_SPLIT_RATIOS,
    SPLIT_NAMES,
    file_sha256,
    stable_json_sha256,
)
from src.data.scaffold_split import assign_scaffold_splits  # noqa: E402


def _ratios(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(values) != 4:
        raise argparse.ArgumentTypeError("Expected train,val,calibration,test ratios.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="BBBP")
    parser.add_argument("--smiles-field", default="canonical_smiles")
    parser.add_argument("--molecule-id-field", default="molecule_id")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--split-ratios", type=_ratios, default=DEFAULT_SPLIT_RATIOS)
    parser.add_argument(
        "--acyclic-policy",
        choices=("canonical-smiles", "group"),
        default="canonical-smiles",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Scaffold split input is empty: {path}")
    return fields, rows


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _write_csv(path: Path, fields: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> None:
    field_list = list(fields)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_list, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare_scaffold_split(
    *,
    input_csv: str | Path,
    output_dir: str | Path,
    dataset: str,
    smiles_field: str,
    molecule_id_field: str,
    label_field: str,
    seed: int,
    ratios: tuple[float, ...],
    acyclic_policy: str,
) -> dict[str, Any]:
    source = Path(input_csv).expanduser().resolve()
    fields, rows = _read(source)
    required = {smiles_field, molecule_id_field, label_field}
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"Scaffold split input is missing fields {missing}: {source}")
    assigned, audit = assign_scaffold_splits(
        rows,
        smiles_field=smiles_field,
        seed=seed,
        ratios=ratios,
        acyclic_policy=acyclic_policy,
    )
    by_split = {
        split: [row for row in assigned if row["split"] == split]
        for split in SPLIT_NAMES
    }
    if any(not values for values in by_split.values()):
        raise ValueError(
            "Scaffold protocol produced an empty split; choose a preregistered seed "
            "before freezing, not after viewing test metrics."
        )
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Scaffold split output is non-empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    output_fields = list(fields)
    for field in ("canonical_smiles", "scaffold_smiles", "split"):
        if field not in output_fields:
            output_fields.append(field)
    _write_csv(destination / "all.csv", output_fields, assigned)
    for split in SPLIT_NAMES:
        _write_csv(destination / f"{split}.csv", output_fields, by_split[split])
    label_counts = {
        split: dict(sorted(Counter(str(row[label_field]) for row in by_split[split]).items()))
        for split in SPLIT_NAMES
    }
    manifest = {
        "schema_version": "cross_scaffold_split_manifest_v1",
        "dataset": dataset,
        "protocol": "cross_scaffold_generalization_v1",
        "raw_path": str(source),
        "raw_sha256": file_sha256(source),
        "split_seed": int(seed),
        "split_strategy": "sha256_bemis_murcko_group_v1",
        "split_ratios": list(ratios),
        "acyclic_policy": acyclic_policy,
        "split_sizes": {key: len(value) for key, value in by_split.items()},
        "label_counts": label_counts,
        "molecule_ids_hash": stable_json_sha256(
            sorted(str(row[molecule_id_field]) for row in assigned)
        ),
        "canonical_smiles_hash": stable_json_sha256(
            sorted(str(row["canonical_smiles"]) for row in assigned)
        ),
        "scaffold_hash": audit["scaffold_hash"],
        "candidate_source_splits": ["train", "val"],
        "selector_source_splits": ["calibration"],
        "threshold_source": "calibration",
        "test_usage": "final_evaluation_only",
        "test_scaffolds_used_for_split_tuning": False,
        "status": "PREPARED",
    }
    _atomic_text(
        destination / "split_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    leakage = {
        **audit["leakage_audit"],
        "schema_version": "cross_scaffold_leakage_audit_v1",
        "dataset": dataset,
        "protocol": "cross_scaffold_generalization_v1",
        "scaffold_overlap_count": 0,
        "test_usage": "final_evaluation_only",
    }
    _atomic_text(
        destination / "split_leakage_audit.json",
        json.dumps(leakage, indent=2, sort_keys=True) + "\n",
    )
    summary = {
        **audit,
        "dataset": dataset,
        "input_sha256": file_sha256(source),
        "manifest_sha256": file_sha256(destination / "split_manifest.json"),
        "leakage_audit_sha256": file_sha256(destination / "split_leakage_audit.json"),
    }
    _atomic_text(
        destination / "scaffold_split_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.input_csv).expanduser().resolve()
    fields, rows = _read(source)
    required = {args.smiles_field, args.molecule_id_field, args.label_field}
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"Scaffold split input is missing fields {missing}: {source}")
    if args.validate_only or args.dry_run:
        print(
            json.dumps(
                {
                    "status": "VALIDATED_NOT_RUN",
                    "dataset": args.dataset,
                    "input_csv": str(source),
                    "input_rows": len(rows),
                    "input_sha256": file_sha256(source),
                    "planned_output_dir": str(Path(args.output_dir).expanduser()),
                    "seed": args.seed,
                    "acyclic_policy": args.acyclic_policy,
                    "formal_output_written": False,
                },
                sort_keys=True,
            )
        )
        return 0
    result = prepare_scaffold_split(
        input_csv=source,
        output_dir=args.output_dir,
        dataset=args.dataset,
        smiles_field=args.smiles_field,
        molecule_id_field=args.molecule_id_field,
        label_field=args.label_field,
        seed=args.seed,
        ratios=tuple(args.split_ratios),
        acyclic_policy=args.acyclic_policy,
    )
    print(json.dumps(result, sort_keys=True))
    print("[SCAFFOLD_SPLIT_PREPARE_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
