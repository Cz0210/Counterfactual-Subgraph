#!/usr/bin/env python3
"""Audit Mutagenicity CSV chemistry and its alignment with TU graph labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.mutagenicity import (
    RDKIT_AUDIT_FIELDS,
    audit_csv_source,
    evaluate_tu_label_mappings,
    load_csv_source,
    normalize_binary_label,
    read_integer_lines,
    sha256_file,
    summarize_rdkit_audit,
    write_csv,
    write_json,
)


def _counts(values: Sequence[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items())}


def _manifest_entry(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument(
        "--raw-csv", default="data/raw/Mutagenicity/smiles/smiles_mutagenicity_raw.csv"
    )
    parser.add_argument(
        "--curated-csv",
        default="data/raw/Mutagenicity/smiles/smiles_mutagenicity_curated.csv",
    )
    parser.add_argument(
        "--removed-csv",
        default="data/raw/Mutagenicity/smiles/smiles_mutagenicity_removed.csv",
    )
    parser.add_argument(
        "--tu-dir", default="data/raw/Mutagenicity/tudataset/Mutagenicity"
    )
    parser.add_argument(
        "--output-dir", default="outputs/hpc/datasets/mutagenicity_v1_full/source_audit"
    )
    return parser


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Mutagenicity Source Audit",
        "",
        f"- Raw rows: {summary['raw_rows']}",
        f"- Curated rows: {summary['curated_rows']}",
        f"- Removed rows: {summary['removed_rows']}",
        f"- TU graphs: {summary['tu_num_graphs']}",
        f"- Project label semantics: `1=mutagenic`, `0=non_mutagenic`",
        f"- TU label mapping: `{summary['tu_label_mapping']}`",
        f"- TU label match rate: {summary['tu_label_match_rate']:.8f}",
        "",
        "## RDKit audit",
        "",
        f"- Raw parse rate: {summary['raw_parse_rate']:.8f}",
        f"- Raw sanitize rate: {summary['raw_sanitize_rate']:.8f}",
        f"- Raw multicomponent rate: {summary['raw_multicomponent_rate']:.8f}",
        f"- Curated parse rate: {summary['curated_parse_rate']:.8f}",
        f"- Curated sanitize rate: {summary['curated_sanitize_rate']:.8f}",
        f"- Curated multicomponent rate: {summary['curated_multicomponent_rate']:.8f}",
        "",
        f"Audit passed: **{summary['audit_passed']}**",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_csv_source(args.raw_csv)
    curated = load_csv_source(args.curated_csv)
    removed = load_csv_source(args.removed_csv)
    raw_labels = [normalize_binary_label(row[raw.label_col]) for row in raw.rows]
    curated_labels = [normalize_binary_label(row[curated.label_col]) for row in curated.rows]
    tu_dir = Path(args.tu_dir)
    tu_label_path = tu_dir / "Mutagenicity_graph_labels.txt"
    tu_labels = read_integer_lines(tu_label_path)
    mapping_rows, selected_mapping, mismatch_rows = evaluate_tu_label_mappings(
        raw_labels, tu_labels
    )
    raw_audit = audit_csv_source(raw)
    curated_audit = audit_csv_source(curated)
    raw_stats = summarize_rdkit_audit(raw_audit)
    curated_stats = summarize_rdkit_audit(curated_audit)
    selected_mapping_row = next(row for row in mapping_rows if row["mapping"] == selected_mapping)
    audit_passed = bool(
        len(raw.rows) == 4337
        and len(curated.rows) == 4247
        and len(removed.rows) == len(raw.rows) - len(curated.rows)
        and len(tu_labels) == 4337
        and selected_mapping == "inverted_0_1"
        and not mismatch_rows
        and math_is_one(curated_stats["parse_rate"])
        and math_is_one(curated_stats["sanitize_rate"])
    )
    summary = {
        "raw_rows": len(raw.rows),
        "curated_rows": len(curated.rows),
        "removed_rows": len(removed.rows),
        "tu_num_graphs": len(tu_labels),
        "raw_smiles_col": raw.smiles_col,
        "raw_label_col": raw.label_col,
        "curated_smiles_col": curated.smiles_col,
        "curated_label_col": curated.label_col,
        "raw_label_counts": _counts(raw_labels),
        "curated_label_counts": _counts(curated_labels),
        "raw_parse_rate": raw_stats["parse_rate"],
        "raw_sanitize_rate": raw_stats["sanitize_rate"],
        "raw_multicomponent_rate": raw_stats["multicomponent_rate"],
        "curated_parse_rate": curated_stats["parse_rate"],
        "curated_sanitize_rate": curated_stats["sanitize_rate"],
        "curated_multicomponent_rate": curated_stats["multicomponent_rate"],
        "tu_label_mapping": selected_mapping,
        "tu_label_match_rate": selected_mapping_row["match_rate"],
        "audit_passed": audit_passed,
    }
    write_csv(output_dir / "raw_rdkit_audit.csv", raw_audit, RDKIT_AUDIT_FIELDS)
    write_csv(output_dir / "curated_rdkit_audit.csv", curated_audit, RDKIT_AUDIT_FIELDS)
    write_csv(
        output_dir / "tu_label_mapping.csv",
        mapping_rows,
        (
            "mapping",
            "mapping_json",
            "num_rows",
            "num_mappable",
            "num_matches",
            "num_mismatches",
            "match_rate",
        ),
    )
    write_csv(
        output_dir / "tu_label_mismatches.csv",
        mismatch_rows,
        ("graph_id", "csv_label", "tu_label", "mapped_tu_label"),
    )
    manifest_paths = [
        Path(args.raw_csv),
        Path(args.curated_csv),
        Path(args.removed_csv),
        *(sorted(tu_dir.glob("Mutagenicity_*.txt"))),
    ]
    row_counts = {
        str(Path(args.raw_csv)): len(raw.rows),
        str(Path(args.curated_csv)): len(curated.rows),
        str(Path(args.removed_csv)): len(removed.rows),
        str(tu_label_path): len(tu_labels),
    }
    write_json(
        output_dir / "source_file_manifest.json",
        {
            "files": [
                _manifest_entry(path, rows=row_counts.get(str(path))) for path in manifest_paths
            ]
        },
    )
    write_json(output_dir / "source_audit_summary.json", summary)
    (output_dir / "source_audit_report.md").write_text(
        _render_report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    if not audit_passed:
        print("[MUTAGENICITY_SOURCE_AUDIT_FAILED]", flush=True)
        return 1
    print("[MUTAGENICITY_SOURCE_AUDIT_OK]", flush=True)
    return 0


def math_is_one(value: float) -> bool:
    return abs(float(value) - 1.0) <= 1e-12


if __name__ == "__main__":
    raise SystemExit(main())
