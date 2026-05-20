"""Merge one or more candidate pools with deterministic per-parent deduplication."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.eval.candidate_pool_audit import (
    _as_float,
    _canonical_fragment_key,
    _coalesce,
    _normalize_row,
    _normalize_text,
)
from src.utils.io import ensure_directory, read_jsonl, write_jsonl


@dataclass(frozen=True, slots=True)
class MergeConfig:
    """Execution knobs for candidate-pool merging."""

    dedup_key: tuple[str, ...] = ("final_fragment", "parent_smiles")
    keep_best_by: str = "reward_total"


def _keep_metric_value(payload: dict[str, Any], metric_name: str) -> float | None:
    if metric_name == "reward_total":
        return _as_float(_coalesce(payload, "reward_total", "total_reward"))
    if metric_name == "cf_drop":
        return _as_float(_coalesce(payload, "cf_drop", "counterfactual_drop", "teacher_cf_drop"))
    return _as_float(_coalesce(payload, metric_name))


def _resolve_dedup_component(payload: dict[str, Any], key_name: str, record_index: int) -> str:
    normalized = _normalize_row(payload, record_index)
    if key_name == "final_fragment":
        fragment_key = _canonical_fragment_key(normalized.final_fragment)
        return fragment_key or ""
    if key_name == "parent_smiles":
        return normalized.parent_smiles or ""
    value = _normalize_text(_coalesce(payload, key_name))
    return value or ""


def _candidate_rank_tuple(payload: dict[str, Any], keep_best_by: str) -> tuple[float, float]:
    primary = _keep_metric_value(payload, keep_best_by)
    secondary = _keep_metric_value(payload, "cf_drop")
    return (
        float(primary) if primary is not None else float("-inf"),
        float(secondary) if secondary is not None else float("-inf"),
    )


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
    input_counts: list[dict[str, Any]] = []
    merged_rows_before_dedup: list[dict[str, Any]] = []
    best_by_key: dict[tuple[str, ...], dict[str, Any]] = {}

    for path_like in pool_jsonls:
        pool_path = Path(path_like).expanduser().resolve()
        rows = read_jsonl(pool_path)
        input_counts.append(
            {
                "path": str(pool_path),
                "count": len(rows),
            }
        )
        for index, payload in enumerate(rows):
            merged_rows_before_dedup.append(payload)
            dedup_tuple = tuple(
                _resolve_dedup_component(payload, key_name, index)
                for key_name in resolved_config.dedup_key
            )
            current = best_by_key.get(dedup_tuple)
            if current is None or _candidate_rank_tuple(payload, resolved_config.keep_best_by) > _candidate_rank_tuple(
                current,
                resolved_config.keep_best_by,
            ):
                best_by_key[dedup_tuple] = payload

    merged_rows = list(best_by_key.values())
    output_path = Path(out_jsonl).expanduser().resolve()
    summary_path = (
        Path(out_summary_json).expanduser().resolve()
        if out_summary_json is not None
        else output_path.with_name("merge_summary.json")
    )

    write_jsonl(output_path, merged_rows)
    ensure_directory(summary_path.parent)

    unique_parent_smiles = {
        normalized.parent_smiles
        for normalized in (_normalize_row(row, index) for index, row in enumerate(merged_rows))
        if normalized.parent_smiles
    }
    unique_final_fragments = {
        _canonical_fragment_key(_normalize_row(row, index).final_fragment)
        for index, row in enumerate(merged_rows)
    }
    unique_final_fragments.discard(None)

    summary = {
        "input_counts": input_counts,
        "merged_count_before_dedup": len(merged_rows_before_dedup),
        "merged_count_after_dedup": len(merged_rows),
        "dedup_removed_count": len(merged_rows_before_dedup) - len(merged_rows),
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
