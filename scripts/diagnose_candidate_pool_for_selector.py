#!/usr/bin/env python3
"""Diagnose whether a candidate pool can pass selector filtering."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.candidate_pool_audit import (  # noqa: E402
    _as_bool,
    _as_float,
    _canonical_fragment_key,
    _normalize_row,
    _normalize_text,
)
from src.eval.subgraph_similarity import parse_embedding  # noqa: E402
from src.utils.io import read_jsonl  # noqa: E402


OK_FAILURE_TAGS = {"", "ok", "none", "null", "nan", "false", "0", "no_failure"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-jsonl", required=True, help="Candidate pool JSONL to diagnose.")
    parser.add_argument("--label", type=int, default=1, help="Target label used by selector.")
    parser.add_argument("--embedding-field", default="final_fragment_embedding")
    parser.add_argument("--out-json", required=True, help="Output JSON diagnosis path.")
    parser.add_argument("--out-txt", required=True, help="Output text diagnosis path.")
    parser.add_argument("--sample-count", type=int, default=5, help="Number of example rows to print.")
    return parser


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        text = value.strip()
        return bool(text) and text.lower() not in {"none", "null", "nan"}
    return True


def _failure_ok(value: Any) -> bool:
    text = _normalize_text(value)
    if text is None:
        return True
    return text.strip().lower() in OK_FAILURE_TAGS


def _raw_final_fragment(row: dict[str, Any]) -> Any:
    return row.get("final_fragment") or row.get("final_fragment_smiles")


def _raw_parent_smiles(row: dict[str, Any]) -> Any:
    return row.get("parent_smiles") or row.get("smiles")


def _raw_cf_drop(row: dict[str, Any]) -> Any:
    return row.get("cf_drop") if "cf_drop" in row else row.get("counterfactual_drop")


def _raw_cf_flip(row: dict[str, Any]) -> Any:
    return row.get("cf_flip") if "cf_flip" in row else row.get("counterfactual_flip")


def _raw_final_substructure(row: dict[str, Any]) -> Any:
    return row.get("final_substructure")


def _raw_oracle_ok(row: dict[str, Any]) -> Any:
    return row.get("oracle_ok")


def _embedding_dimension(row: dict[str, Any], embedding_field: str) -> tuple[int | None, str | None]:
    value = row.get(embedding_field)
    if not _has_value(value):
        return None, "missing"
    try:
        embedding = parse_embedding(value)
    except ValueError as exc:
        return None, str(exc)
    return int(embedding.size), None


def _strict_selector_row_ok(normalized: Any, has_embedding: bool, label: int) -> bool:
    fragment_key = _canonical_fragment_key(normalized.final_fragment)
    return bool(
        normalized.label == label
        and normalized.parent_smiles
        and normalized.final_fragment
        and fragment_key is not None
        and normalized.parse_ok
        and normalized.valid
        and normalized.connected
        and normalized.oracle_ok
        and normalized.final_substructure
        and normalized.cf_flip
        and normalized.cf_drop is not None
        and float(normalized.cf_drop) >= 0.2
        and _failure_ok(normalized.failure_tag)
        and not normalized.full_parent
        and not normalized.near_parent
        and not normalized.too_small
        and has_embedding
    )


def _relaxed_gt_row_ok(normalized: Any, has_embedding: bool, label: int) -> bool:
    fragment_key = _canonical_fragment_key(normalized.final_fragment)
    cf_drop = normalized.cf_drop if normalized.cf_drop is not None else 0.0
    return bool(
        normalized.label == label
        and normalized.parent_smiles
        and normalized.final_fragment
        and fragment_key is not None
        and normalized.parse_ok
        and normalized.valid
        and normalized.connected
        and normalized.oracle_ok
        and normalized.final_substructure
        and float(cf_drop) >= -999.0
        and _failure_ok(normalized.failure_tag)
        and not normalized.full_parent
        and not normalized.near_parent
        and not normalized.too_small
        and has_embedding
    )


def diagnose_candidate_pool(
    *,
    pool_jsonl: str | Path,
    label: int,
    embedding_field: str,
    sample_count: int = 5,
) -> dict[str, Any]:
    pool_path = Path(pool_jsonl).expanduser().resolve()
    rows = read_jsonl(pool_path)

    label_distribution: Counter[str] = Counter()
    embedding_dimensions: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    possible_reasons: list[str] = []

    counts = Counter()
    raw_missing = Counter()
    strict_filter_reasons = Counter()
    relaxed_filter_reasons = Counter()

    for index, row in enumerate(rows):
        normalized = _normalize_row(row, record_index=index)
        label_distribution[str(normalized.label)] += 1

        has_final_fragment = _has_value(_raw_final_fragment(row)) or bool(normalized.final_fragment)
        has_core_fragment = _has_value(row.get("core_fragment")) or bool(normalized.core_fragment)
        has_parent = _has_value(_raw_parent_smiles(row)) or bool(normalized.parent_smiles)
        has_cf_drop = _has_value(_raw_cf_drop(row))
        has_cf_flip = _has_value(_raw_cf_flip(row))
        has_oracle_ok = _has_value(_raw_oracle_ok(row))
        has_final_substructure = _has_value(_raw_final_substructure(row))

        embedding_dim, embedding_error = _embedding_dimension(row, embedding_field)
        has_embedding = embedding_dim is not None
        if embedding_dim is not None:
            embedding_dimensions[str(embedding_dim)] += 1
        elif embedding_error:
            embedding_dimensions[f"invalid:{embedding_error}"] += 1

        if normalized.label == label:
            counts["rows_label_target"] += 1
        if has_final_fragment:
            counts["rows_with_final_fragment"] += 1
        else:
            raw_missing["missing_final_fragment"] += 1
        if has_core_fragment:
            counts["rows_with_core_fragment"] += 1
        if has_parent:
            counts["rows_with_parent_smiles"] += 1
        else:
            raw_missing["missing_parent_smiles"] += 1
        if has_embedding:
            counts["rows_with_embedding"] += 1
        else:
            raw_missing["missing_embedding"] += 1
        if bool(normalized.oracle_ok):
            counts["rows_oracle_ok_true"] += 1
        if not has_oracle_ok:
            raw_missing["missing_oracle_ok"] += 1
        if bool(normalized.final_substructure):
            counts["rows_final_substructure_true"] += 1
        if not has_final_substructure:
            raw_missing["missing_final_substructure"] += 1
        if bool(normalized.cf_flip):
            counts["rows_cf_flip_true"] += 1
        if not has_cf_flip:
            raw_missing["missing_cf_flip"] += 1
        if normalized.cf_drop is not None:
            counts["rows_cf_drop_not_null"] += 1
            if float(normalized.cf_drop) >= 0.0:
                counts["rows_cf_drop_ge_0"] += 1
            if float(normalized.cf_drop) >= 0.2:
                counts["rows_cf_drop_ge_0p2"] += 1
        if not has_cf_drop:
            raw_missing["missing_cf_drop"] += 1
        if _failure_ok(normalized.failure_tag):
            counts["rows_failure_tag_empty_or_ok"] += 1
        if bool(normalized.parse_ok):
            counts["rows_parse_ok_true"] += 1
        if bool(normalized.valid):
            counts["rows_valid_true"] += 1
        if bool(normalized.connected):
            counts["rows_connected_true"] += 1

        if _strict_selector_row_ok(normalized, has_embedding, label):
            counts["rows_valid_for_default_selector"] += 1
        else:
            if normalized.label != label:
                strict_filter_reasons["label_mismatch"] += 1
            if not normalized.parent_smiles:
                strict_filter_reasons["missing_parent_smiles"] += 1
            if not normalized.final_fragment:
                strict_filter_reasons["missing_final_fragment"] += 1
            if normalized.final_fragment and _canonical_fragment_key(normalized.final_fragment) is None:
                strict_filter_reasons["invalid_fragment_key"] += 1
            if not normalized.parse_ok:
                strict_filter_reasons["parse_ok_false"] += 1
            if not normalized.valid:
                strict_filter_reasons["valid_false"] += 1
            if not normalized.connected:
                strict_filter_reasons["connected_false"] += 1
            if not normalized.oracle_ok:
                strict_filter_reasons["oracle_ok_false"] += 1
            if not normalized.final_substructure:
                strict_filter_reasons["final_substructure_false"] += 1
            if not normalized.cf_flip:
                strict_filter_reasons["cf_flip_false"] += 1
            if normalized.cf_drop is None:
                strict_filter_reasons["cf_drop_null"] += 1
            elif float(normalized.cf_drop) < 0.2:
                strict_filter_reasons["cf_drop_lt_0p2"] += 1
            if not _failure_ok(normalized.failure_tag):
                strict_filter_reasons["failure_tag_not_ok"] += 1
            if normalized.full_parent:
                strict_filter_reasons["full_parent_true"] += 1
            if normalized.near_parent:
                strict_filter_reasons["near_parent_true"] += 1
            if normalized.too_small:
                strict_filter_reasons["too_small_true"] += 1
            if not has_embedding:
                strict_filter_reasons["embedding_missing_or_invalid"] += 1

        if _relaxed_gt_row_ok(normalized, has_embedding, label):
            counts["rows_valid_for_relaxed_gt_selector"] += 1
        else:
            if normalized.label != label:
                relaxed_filter_reasons["label_mismatch"] += 1
            if not normalized.parent_smiles:
                relaxed_filter_reasons["missing_parent_smiles"] += 1
            if not normalized.final_fragment:
                relaxed_filter_reasons["missing_final_fragment"] += 1
            if not normalized.parse_ok:
                relaxed_filter_reasons["parse_ok_false"] += 1
            if not normalized.valid:
                relaxed_filter_reasons["valid_false"] += 1
            if not normalized.connected:
                relaxed_filter_reasons["connected_false"] += 1
            if not normalized.oracle_ok:
                relaxed_filter_reasons["oracle_ok_false"] += 1
            if not normalized.final_substructure:
                relaxed_filter_reasons["final_substructure_false"] += 1
            if not _failure_ok(normalized.failure_tag):
                relaxed_filter_reasons["failure_tag_not_ok"] += 1
            if normalized.full_parent:
                relaxed_filter_reasons["full_parent_true"] += 1
            if normalized.near_parent:
                relaxed_filter_reasons["near_parent_true"] += 1
            if normalized.too_small:
                relaxed_filter_reasons["too_small_true"] += 1
            if not has_embedding:
                relaxed_filter_reasons["embedding_missing_or_invalid"] += 1

        if len(samples) < max(1, int(sample_count)):
            samples.append(
                {
                    "final_fragment": normalized.final_fragment,
                    "parent_smiles": normalized.parent_smiles,
                    "label": normalized.label,
                    "cf_drop": normalized.cf_drop,
                    "cf_flip": normalized.cf_flip,
                    "oracle_ok": normalized.oracle_ok,
                    "final_substructure": normalized.final_substructure,
                    "failure_tag": normalized.failure_tag,
                    "has_embedding": bool(has_embedding),
                    "embedding_dim": embedding_dim,
                }
            )

    total = len(rows)
    if counts["rows_valid_for_default_selector"] == 0:
        if counts["rows_label_target"] == 0:
            possible_reasons.append("label 不匹配")
        if counts["rows_cf_drop_not_null"] == 0:
            possible_reasons.append("cf_drop 全为空")
        elif counts["rows_cf_drop_ge_0p2"] == 0:
            possible_reasons.append("cf_drop 全部小于 0.2")
        if counts["rows_cf_flip_true"] == 0:
            possible_reasons.append("cf_flip 全为空或 false")
        if counts["rows_final_substructure_true"] == 0:
            possible_reasons.append("final_substructure false")
        if counts["rows_oracle_ok_true"] == 0:
            possible_reasons.append("oracle_ok false")
        if counts["rows_with_embedding"] == 0:
            possible_reasons.append("embedding 缺失")
        if counts["rows_with_parent_smiles"] == 0:
            possible_reasons.append("parent_smiles 缺失")
        if counts["rows_with_final_fragment"] == 0:
            possible_reasons.append("final_fragment 缺失")

    return {
        "pool_jsonl": str(pool_path),
        "label": int(label),
        "embedding_field": str(embedding_field),
        "total_rows": total,
        "label_distribution": dict(sorted(label_distribution.items())),
        "rows_label_1": int(counts["rows_label_target"]) if int(label) == 1 else None,
        "rows_label_target": int(counts["rows_label_target"]),
        "rows_with_final_fragment": int(counts["rows_with_final_fragment"]),
        "rows_with_core_fragment": int(counts["rows_with_core_fragment"]),
        "rows_with_parent_smiles": int(counts["rows_with_parent_smiles"]),
        "rows_with_embedding": int(counts["rows_with_embedding"]),
        "embedding_dimension_distribution": dict(sorted(embedding_dimensions.items())),
        "rows_oracle_ok_true": int(counts["rows_oracle_ok_true"]),
        "rows_final_substructure_true": int(counts["rows_final_substructure_true"]),
        "rows_cf_flip_true": int(counts["rows_cf_flip_true"]),
        "rows_cf_drop_not_null": int(counts["rows_cf_drop_not_null"]),
        "rows_cf_drop_ge_0": int(counts["rows_cf_drop_ge_0"]),
        "rows_cf_drop_ge_0p2": int(counts["rows_cf_drop_ge_0p2"]),
        "rows_failure_tag_empty_or_ok": int(counts["rows_failure_tag_empty_or_ok"]),
        "rows_parse_ok_true": int(counts["rows_parse_ok_true"]),
        "rows_valid_true": int(counts["rows_valid_true"]),
        "rows_connected_true": int(counts["rows_connected_true"]),
        "rows_valid_for_default_selector": int(counts["rows_valid_for_default_selector"]),
        "rows_valid_for_relaxed_gt_selector": int(counts["rows_valid_for_relaxed_gt_selector"]),
        "missing_final_fragment": int(raw_missing["missing_final_fragment"]),
        "missing_parent_smiles": int(raw_missing["missing_parent_smiles"]),
        "missing_cf_drop": int(raw_missing["missing_cf_drop"]),
        "missing_cf_flip": int(raw_missing["missing_cf_flip"]),
        "missing_oracle_ok": int(raw_missing["missing_oracle_ok"]),
        "missing_final_substructure": int(raw_missing["missing_final_substructure"]),
        "missing_embedding": int(raw_missing["missing_embedding"]),
        "strict_selector_filter_reasons": dict(sorted(strict_filter_reasons.items())),
        "relaxed_gt_selector_filter_reasons": dict(sorted(relaxed_filter_reasons.items())),
        "samples": samples,
        "possible_zero_valid_reasons": possible_reasons,
    }


def render_text(summary: dict[str, Any]) -> str:
    lines = [
        "Candidate Pool Selector Diagnosis",
        f"pool_jsonl: {summary['pool_jsonl']}",
        f"label: {summary['label']}",
        f"embedding_field: {summary['embedding_field']}",
        "",
        "Basic statistics:",
        f"- total_rows: {summary['total_rows']}",
        f"- label_distribution: {summary['label_distribution']}",
        f"- rows_label_target: {summary['rows_label_target']}",
        f"- rows_with_final_fragment: {summary['rows_with_final_fragment']}",
        f"- rows_with_core_fragment: {summary['rows_with_core_fragment']}",
        f"- rows_with_parent_smiles: {summary['rows_with_parent_smiles']}",
        f"- rows_with_embedding: {summary['rows_with_embedding']}",
        f"- embedding_dimension_distribution: {summary['embedding_dimension_distribution']}",
        "",
        "Selector-critical fields:",
        f"- rows_oracle_ok_true: {summary['rows_oracle_ok_true']}",
        f"- rows_final_substructure_true: {summary['rows_final_substructure_true']}",
        f"- rows_cf_flip_true: {summary['rows_cf_flip_true']}",
        f"- rows_cf_drop_not_null: {summary['rows_cf_drop_not_null']}",
        f"- rows_cf_drop_ge_0: {summary['rows_cf_drop_ge_0']}",
        f"- rows_cf_drop_ge_0p2: {summary['rows_cf_drop_ge_0p2']}",
        f"- rows_failure_tag_empty_or_ok: {summary['rows_failure_tag_empty_or_ok']}",
        f"- rows_parse_ok_true: {summary['rows_parse_ok_true']}",
        f"- rows_valid_for_default_selector: {summary['rows_valid_for_default_selector']}",
        f"- rows_valid_for_relaxed_gt_selector: {summary['rows_valid_for_relaxed_gt_selector']}",
        "",
        "Missing fields:",
        f"- missing_final_fragment: {summary['missing_final_fragment']}",
        f"- missing_parent_smiles: {summary['missing_parent_smiles']}",
        f"- missing_cf_drop: {summary['missing_cf_drop']}",
        f"- missing_cf_flip: {summary['missing_cf_flip']}",
        f"- missing_oracle_ok: {summary['missing_oracle_ok']}",
        f"- missing_final_substructure: {summary['missing_final_substructure']}",
        f"- missing_embedding: {summary['missing_embedding']}",
        "",
        "Strict selector filter reasons:",
    ]
    for key, value in summary["strict_selector_filter_reasons"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Relaxed GT selector filter reasons:")
    for key, value in summary["relaxed_gt_selector_filter_reasons"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    if summary["rows_valid_for_default_selector"] == 0:
        lines.append("rows_valid_for_default_selector = 0 possible reasons:")
        if summary["possible_zero_valid_reasons"]:
            for reason in summary["possible_zero_valid_reasons"]:
                lines.append(f"- {reason}")
        else:
            lines.append("- no single obvious global reason; inspect strict_selector_filter_reasons")
        lines.append("")
    lines.append("Samples:")
    for sample in summary["samples"]:
        lines.append(json.dumps(sample, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_parser().parse_args()
    summary = diagnose_candidate_pool(
        pool_jsonl=args.pool_jsonl,
        label=int(args.label),
        embedding_field=str(args.embedding_field),
        sample_count=int(args.sample_count),
    )
    out_json = Path(args.out_json).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_txt.write_text(render_text(summary), encoding="utf-8")
    print(render_text(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
