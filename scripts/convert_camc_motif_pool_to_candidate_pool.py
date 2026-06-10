#!/usr/bin/env python3
"""Convert a CAMC action-motif pool CSV into selector-readable candidate_pool JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chem import is_connected_fragment, is_rdkit_available, parse_smiles  # noqa: E402


FRAGMENT_FIELD_PRIORITY = (
    "final_fragment",
    "motif_smiles",
    "canonical_motif_smiles",
    "deleted_motif_smiles",
    "action_motif",
    "action_motif_smiles",
    "fragment",
    "fragment_smiles",
    "core_fragment",
    "subgraph",
    "subgraph_smiles",
    "smiles",
)
PARENT_FIELD_PRIORITY = (
    "parent_smiles",
    "input_smiles",
    "original_smiles",
    "molecule_smiles",
    "full_parent",
    "source_smiles",
)
CF_DROP_FIELD_PRIORITY = (
    "cf_drop",
    "drop",
    "camc_drop",
    "score_drop",
    "confidence_drop",
)
CF_FLIP_FIELD_PRIORITY = (
    "cf_flip",
    "fullgraph_cf_flip",
    "flip",
    "success",
    "is_success",
)
SUPPORT_FIELD_PRIORITY = (
    "support",
    "support_count",
    "coverage",
)
FINAL_SUBSTRUCTURE_FIELD_PRIORITY = (
    "final_substructure",
    "substructure",
    "substructure_ok",
    "valid_motif",
)
ORACLE_OK_FIELD_PRIORITY = (
    "oracle_ok",
    "valid_motif",
)
FAILURE_FIELD_PRIORITY = (
    "failure_reason",
    "failure_tag",
    "invalid_detail",
    "error",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config path for Slurm parity. The converter uses explicit CLI paths.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Ignored dotted overrides kept for Slurm wrapper parity.",
    )
    parser.add_argument("--input-csv", required=True, help="Input camc_gt_fullgraph_motif_pool.csv.")
    parser.add_argument("--output-jsonl", required=True, help="Output selector candidate_pool JSONL.")
    parser.add_argument("--label", type=int, default=1, help="Target class label written to each row.")
    parser.add_argument("--method", default="gt_fullgraph_greedy_proxy", help="Method name written to each row.")
    parser.add_argument(
        "--failed-rows-jsonl",
        default="",
        help="Optional failed-row JSONL path. Defaults to failed_rows.jsonl beside output.",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional summary path. Defaults to candidate_pool_conversion_summary.json beside output.",
    )
    return parser


def _normalize_header(name: str) -> str:
    return str(name).strip().lower()


def _build_header_map(fieldnames: list[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for field in fieldnames or []:
        normalized = _normalize_header(field)
        if normalized and normalized not in mapping:
            mapping[normalized] = field
    return mapping


def _select_field(header_map: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        matched = header_map.get(_normalize_header(name))
        if matched is not None:
            return matched
    return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"none", "null", "nan"}:
        return None
    return text


def _as_float(value: Any) -> float | None:
    text = _text(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text or text in {"none", "null", "nan"}:
        return None
    if text in {"1", "true", "t", "yes", "y", "on", "ok", "success"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off", "fail", "failed"}:
        return False
    numeric = _as_float(value)
    if numeric is not None:
        return bool(numeric)
    return None


def _is_failure_free(value: Any) -> bool:
    text = _text(value)
    if text is None:
        return True
    return text.lower() in {"ok", "none", "null", "nan", "false", "0", "no_failure"}


def _get(row: dict[str, Any], field: str | None) -> Any:
    if field is None:
        return None
    return row.get(field)


def _parse_fragment(smiles: str) -> tuple[bool, bool, str | None, int | None, str | None]:
    parsed_raw = parse_smiles(smiles, sanitize=False, canonicalize=True)
    if not parsed_raw.parseable:
        return (
            False,
            False,
            parsed_raw.canonical_smiles,
            parsed_raw.atom_count,
            parsed_raw.failure_reason,
        )
    parsed = parse_smiles(smiles, sanitize=True, canonicalize=True)
    return (
        True,
        bool(parsed.sanitized),
        parsed.canonical_smiles or parsed_raw.canonical_smiles,
        parsed.atom_count or parsed_raw.atom_count,
        parsed.failure_reason,
    )


def _parent_atom_count(parent_smiles: str | None) -> int | None:
    if not parent_smiles:
        return None
    parsed = parse_smiles(parent_smiles, sanitize=True, canonicalize=True)
    if not parsed.parseable:
        return None
    return parsed.atom_count


def _resolve_atom_ratio(
    row: dict[str, Any],
    *,
    fragment_atom_count: int | None,
    parent_smiles: str | None,
) -> float | None:
    explicit = _as_float(row.get("atom_ratio"))
    if explicit is not None:
        return explicit
    parent_atoms = _parent_atom_count(parent_smiles)
    if parent_atoms is None or parent_atoms <= 0 or fragment_atom_count is None:
        return None
    return float(fragment_atom_count) / float(parent_atoms)


def _copy_optional_support_fields(row: dict[str, Any], payload: dict[str, Any]) -> None:
    for field_name in SUPPORT_FIELD_PRIORITY:
        if field_name in row and row[field_name] not in {None, ""}:
            payload[field_name] = row[field_name]


def _failed_payload(
    *,
    source_csv: Path,
    source_row_index: int,
    reason: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_csv": str(source_csv),
        "source_row_index": int(source_row_index),
        "failure_reason": reason,
        "row": row,
    }


def convert_csv_to_jsonl(
    *,
    input_csv: str | Path,
    output_jsonl: str | Path,
    label: int,
    method: str,
    failed_rows_jsonl: str | Path | None = None,
    summary_json: str | Path | None = None,
) -> dict[str, Any]:
    input_path = Path(input_csv).expanduser().resolve()
    output_path = Path(output_jsonl).expanduser().resolve()
    failed_path = (
        Path(failed_rows_jsonl).expanduser().resolve()
        if failed_rows_jsonl
        else output_path.parent / "failed_rows.jsonl"
    )
    summary_path = (
        Path(summary_json).expanduser().resolve()
        if summary_json
        else output_path.parent / "candidate_pool_conversion_summary.json"
    )

    if not input_path.exists():
        raise FileNotFoundError(f"input CSV not found: {input_path}")
    if not is_rdkit_available():
        raise RuntimeError(
            "RDKit is required to validate and size CAMC motif fragments. "
            "Run this converter inside the smiles_pip118 conda environment."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    converted_rows = 0
    skipped_rows = 0
    missing_fragment_rows = 0
    missing_parent_rows = 0
    missing_cf_drop_rows = 0
    missing_cf_flip_rows = 0
    failed_rows: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        header_map = _build_header_map(reader.fieldnames)
        fragment_field = _select_field(header_map, FRAGMENT_FIELD_PRIORITY)
        parent_field = _select_field(header_map, PARENT_FIELD_PRIORITY)
        cf_drop_field = _select_field(header_map, CF_DROP_FIELD_PRIORITY)
        cf_flip_field = _select_field(header_map, CF_FLIP_FIELD_PRIORITY)
        final_substructure_field = _select_field(header_map, FINAL_SUBSTRUCTURE_FIELD_PRIORITY)
        oracle_ok_field = _select_field(header_map, ORACLE_OK_FIELD_PRIORITY)
        failure_field = _select_field(header_map, FAILURE_FIELD_PRIORITY)
        field_mapping_used = {
            "fragment": fragment_field,
            "parent_smiles": parent_field,
            "cf_drop": cf_drop_field,
            "cf_flip": cf_flip_field,
            "final_substructure": final_substructure_field,
            "oracle_ok": oracle_ok_field,
            "failure": failure_field,
            "support_fields": [field for field in SUPPORT_FIELD_PRIORITY if field in header_map],
        }

        with output_path.open("w", encoding="utf-8") as output_handle:
            for row_index, row in enumerate(reader):
                total_rows += 1
                fragment = _text(_get(row, fragment_field))
                parent_smiles = _text(_get(row, parent_field))
                failure_value = _get(row, failure_field)
                failure_free = _is_failure_free(failure_value)

                if fragment is None:
                    missing_fragment_rows += 1
                    skipped_rows += 1
                    failed_rows.append(
                        _failed_payload(
                            source_csv=input_path,
                            source_row_index=row_index,
                            reason=(
                                "missing fragment/motif field. "
                                f"Tried fields={list(FRAGMENT_FIELD_PRIORITY)}"
                            ),
                            row=row,
                        )
                    )
                    continue

                parse_ok, sanitized, canonical_fragment, fragment_atom_count, parse_failure = _parse_fragment(fragment)
                if not parse_ok:
                    skipped_rows += 1
                    failed_rows.append(
                        _failed_payload(
                            source_csv=input_path,
                            source_row_index=row_index,
                            reason=f"fragment could not be parsed: {parse_failure}",
                            row=row,
                        )
                    )
                    continue

                if parent_smiles is None:
                    missing_parent_rows += 1

                cf_drop = _as_float(_get(row, cf_drop_field))
                cf_drop_missing = cf_drop is None
                if cf_drop_missing:
                    missing_cf_drop_rows += 1
                    cf_drop = 0.0

                cf_flip = _as_bool(_get(row, cf_flip_field))
                cf_flip_missing = cf_flip is None
                if cf_flip is None:
                    missing_cf_flip_rows += 1
                    cf_flip = True

                final_substructure = _as_bool(_get(row, final_substructure_field))
                if final_substructure is None:
                    final_substructure = True

                oracle_ok = _as_bool(_get(row, oracle_ok_field))
                if oracle_ok is None:
                    oracle_ok = True

                connected = False
                try:
                    connected = bool(is_connected_fragment(fragment))
                except Exception:
                    connected = bool(parse_ok and sanitized)

                atom_ratio = _resolve_atom_ratio(
                    row,
                    fragment_atom_count=fragment_atom_count,
                    parent_smiles=parent_smiles,
                )
                payload: dict[str, Any] = {
                    "id": f"{method}:{row_index}",
                    "method": method,
                    "label": int(label),
                    "parent_smiles": parent_smiles,
                    "final_fragment": fragment,
                    "core_fragment": fragment,
                    "canonical_fragment": canonical_fragment,
                    "parse_ok": bool(parse_ok),
                    "valid": bool(parse_ok),
                    "sanitize_ok": bool(sanitized),
                    "connected": bool(connected),
                    "final_substructure": bool(final_substructure),
                    "oracle_ok": bool(oracle_ok),
                    "cf_drop": cf_drop,
                    "cf_drop_missing": bool(cf_drop_missing),
                    "cf_flip": bool(cf_flip),
                    "cf_flip_missing": bool(cf_flip_missing),
                    "reward_total": float(cf_drop) if cf_drop is not None else 0.0,
                    "atom_count": int(fragment_atom_count) if fragment_atom_count is not None else None,
                    "atom_ratio": atom_ratio,
                    "failure_tag": None if failure_free else _text(failure_value),
                    "source_csv": str(input_path),
                    "source_row_index": int(row_index),
                    "source_fragment_field": fragment_field,
                    "source_parent_field": parent_field,
                    "source_cf_drop_field": cf_drop_field,
                    "source_cf_flip_field": cf_flip_field,
                }
                _copy_optional_support_fields(row, payload)
                output_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                converted_rows += 1

    if failed_rows:
        with failed_path.open("w", encoding="utf-8") as failed_handle:
            for row in failed_rows:
                failed_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif failed_path.exists():
        failed_path.unlink()

    summary = {
        "input_csv": str(input_path),
        "output_jsonl": str(output_path),
        "total_rows": int(total_rows),
        "converted_rows": int(converted_rows),
        "skipped_rows": int(skipped_rows),
        "missing_fragment_rows": int(missing_fragment_rows),
        "missing_parent_rows": int(missing_parent_rows),
        "missing_cf_drop_rows": int(missing_cf_drop_rows),
        "missing_cf_flip_rows": int(missing_cf_flip_rows),
        "failed_rows_path": str(failed_path) if failed_rows else None,
        "field_mapping_used": field_mapping_used,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = build_parser().parse_args()
    summary = convert_csv_to_jsonl(
        input_csv=args.input_csv,
        output_jsonl=args.output_jsonl,
        label=int(args.label),
        method=str(args.method),
        failed_rows_jsonl=args.failed_rows_jsonl or None,
        summary_json=args.summary_json or None,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
