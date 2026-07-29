"""Losslessly merge candidate pools with deterministic key-only deduplication."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.io import ensure_directory, read_jsonl, write_jsonl


@dataclass(frozen=True, slots=True)
class MergeConfig:
    """Execution knobs for candidate-pool merging."""

    dedup_key: tuple[str, ...] = ("final_fragment", "parent_smiles")
    keep_best_by: str = "reward_total"


def _dedup_tuple(
    payload: dict[str, Any],
    dedup_key: tuple[str, ...],
) -> tuple[str, ...]:
    """Build the exact configured key without chemistry or eligibility filtering."""

    return tuple(str(payload.get(key_name) or "").strip() for key_name in dedup_key)


def _finite_keep_metric(
    payload: dict[str, Any],
    metric_name: str,
) -> float | None:
    aliases = {
        "reward_total": ("reward_total", "total_reward"),
        "cf_drop": ("cf_drop", "counterfactual_drop", "teacher_cf_drop"),
    }
    raw_value: Any = None
    for field_name in aliases.get(metric_name, (metric_name,)):
        value = payload.get(field_name)
        if value is not None:
            raw_value = value
            break
    try:
        numeric = float(raw_value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _format_key_examples(keys: set[tuple[str, ...]], limit: int = 5) -> list[list[str]]:
    return [list(key) for key in sorted(keys)[:limit]]


def merge_candidate_pools(
    pool_jsonls: list[str | Path],
    *,
    out_jsonl: str | Path,
    out_summary_json: str | Path | None = None,
    config: MergeConfig | None = None,
) -> dict[str, Any]:
    """Merge multiple candidate pools and deduplicate by the configured key."""

    if not pool_jsonls:
        raise ValueError("At least one pool_jsonl path is required.")

    resolved_config = config or MergeConfig()
    if not resolved_config.dedup_key:
        raise ValueError("dedup_key must contain at least one field.")
    if any(not str(field).strip() for field in resolved_config.dedup_key):
        raise ValueError("dedup_key fields must be non-empty.")

    input_counts: list[dict[str, Any]] = []
    merged_rows_before_dedup: list[dict[str, Any]] = []
    best_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    best_metric_by_key: dict[tuple[str, ...], float | None] = {}
    raw_input_key_set: set[tuple[str, ...]] = set()
    eligible_input_key_set: set[tuple[str, ...]] = set()
    skipped_empty_key_rows = 0
    non_numeric_keep_best_rows = 0
    eligible_input_rows = 0

    for path_like in pool_jsonls:
        pool_path = Path(path_like).expanduser().resolve()
        rows = read_jsonl(pool_path)
        input_counts.append(
            {
                "path": str(pool_path),
                "count": len(rows),
            }
        )
        for payload in rows:
            merged_rows_before_dedup.append(payload)
            dedup_tuple = _dedup_tuple(payload, resolved_config.dedup_key)
            raw_input_key_set.add(dedup_tuple)
            metric = _finite_keep_metric(payload, resolved_config.keep_best_by)
            if metric is None:
                non_numeric_keep_best_rows += 1
            if any(not component for component in dedup_tuple):
                skipped_empty_key_rows += 1
                continue

            eligible_input_rows += 1
            eligible_input_key_set.add(dedup_tuple)
            if dedup_tuple not in best_by_key:
                best_by_key[dedup_tuple] = payload
                best_metric_by_key[dedup_tuple] = metric
                continue

            current_metric = best_metric_by_key[dedup_tuple]
            if metric is not None and (
                current_metric is None or metric > current_metric
            ):
                best_by_key[dedup_tuple] = payload
                best_metric_by_key[dedup_tuple] = metric

    merged_rows = list(best_by_key.values())
    output_path = Path(out_jsonl).expanduser().resolve()
    summary_path = (
        Path(out_summary_json).expanduser().resolve()
        if out_summary_json is not None
        else output_path.with_name("merge_summary.json")
    )

    write_jsonl(output_path, merged_rows)
    ensure_directory(summary_path.parent)

    written_rows = read_jsonl(output_path)
    output_key_set: set[tuple[str, ...]] = set()
    for payload in written_rows:
        dedup_tuple = _dedup_tuple(payload, resolved_config.dedup_key)
        if any(not component for component in dedup_tuple):
            raise RuntimeError(
                "Merged output contains an empty dedup key: "
                f"{list(dedup_tuple)}"
            )
        output_key_set.add(dedup_tuple)

    missing_eligible_keys = eligible_input_key_set - output_key_set
    unexpected_keys = output_key_set - eligible_input_key_set
    if missing_eligible_keys or unexpected_keys:
        raise RuntimeError(
            "Candidate-pool merge violated key-set conservation: "
            f"missing_count={len(missing_eligible_keys)} "
            f"unexpected_count={len(unexpected_keys)} "
            f"missing_examples={_format_key_examples(missing_eligible_keys)} "
            f"unexpected_examples={_format_key_examples(unexpected_keys)}"
        )

    unique_parent_smiles = {
        str(row.get("parent_smiles") or "").strip()
        for row in written_rows
        if str(row.get("parent_smiles") or "").strip()
    }
    unique_final_fragments = {
        str(row.get("final_fragment") or "").strip()
        for row in written_rows
        if str(row.get("final_fragment") or "").strip()
    }

    summary = {
        "input_counts": input_counts,
        "input_rows": len(merged_rows_before_dedup),
        "raw_unique_key_count": len(raw_input_key_set),
        "eligible_unique_key_count": len(eligible_input_key_set),
        "eligible_input_rows": eligible_input_rows,
        "skipped_empty_key_rows": skipped_empty_key_rows,
        "non_numeric_keep_best_rows": non_numeric_keep_best_rows,
        "merged_count_before_dedup": len(merged_rows_before_dedup),
        "merged_count_after_dedup": len(written_rows),
        "dedup_removed_count": eligible_input_rows - len(written_rows),
        "missing_eligible_key_count": len(missing_eligible_keys),
        "unexpected_key_count": len(unexpected_keys),
        "unique_parent_count": len(unique_parent_smiles),
        "unique_final_fragment_count": len(unique_final_fragments),
        "dedup_key": list(resolved_config.dedup_key),
        "keep_best_by": resolved_config.keep_best_by,
        "out_jsonl": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


__all__ = [
    "MergeConfig",
    "merge_candidate_pools",
]
