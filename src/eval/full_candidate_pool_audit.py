"""Full candidate-pool audit helpers for selector preparation."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict, OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any

from src.chem import is_parent_substructure
from src.data.ppo_prompt_dataset import PPOPromptRecord, load_ppo_prompt_records
from src.eval.candidate_pool_audit import (
    ATOM_RATIO_BUCKETS,
    NormalizedCandidateRow,
    _build_similarity_stats,
    _bucketize_atom_ratio,
    _canonical_fragment_key,
    _normalize_row,
    _safe_mean,
    _safe_median,
    _safe_quantile,
    _safe_rate,
)
from src.utils.io import ensure_directory, read_jsonl, write_jsonl


@dataclass(frozen=True, slots=True)
class FullPoolAuditConfig:
    """Execution knobs for full candidate-pool auditing."""

    label_col: str = "label"
    smiles_col: str = "parent_smiles"
    target_label: int = 1
    sim_sample_size: int = 5000
    topk_show: int = 10
    coverage_parent_limit: int = 0


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"none", "null", "nan"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _row_parent_key(row: NormalizedCandidateRow) -> str:
    parent_index = row.raw_payload.get("parent_index")
    if parent_index is not None:
        return f"index:{parent_index}"
    if row.parent_smiles:
        return f"smiles:{row.parent_smiles}"
    return f"row:{row.record_index}"


def _row_label_key(row: NormalizedCandidateRow) -> str:
    return str(row.label) if row.label is not None else "missing"


def _projection_method_bucket(row: NormalizedCandidateRow) -> str:
    projection_method = str(row.raw_payload.get("projection_method") or "").strip()
    if projection_method == "direct_match":
        return "direct_match"
    if row.direct_substructure:
        return "direct_match"
    if projection_method in {"nearest_parent_subgraph", "retrieval", "identity"}:
        return "nearest_parent_subgraph"
    if row.projection_attempted and not row.projection_success:
        return "failed"
    return "none"


def _counterfactual_reason(row: NormalizedCandidateRow) -> str:
    return str(
        row.raw_payload.get("counterfactual_reason")
        or row.raw_payload.get("cf_reward_skipped_reason")
        or "missing"
    )


def _cf_oracle_called(row: NormalizedCandidateRow) -> bool:
    return bool(
        _as_bool(row.raw_payload.get("cf_oracle_called"))
        or _as_bool(row.raw_payload.get("counterfactual_called"))
        or _as_bool(row.raw_payload.get("counterfactual_teacher_called"))
    )


def _residual_sanitize_failed(row: NormalizedCandidateRow) -> bool:
    if bool(_as_bool(row.raw_payload.get("residual_sanitize_failed"))):
        return True
    reason = _counterfactual_reason(row).lower()
    invalid_detail = str(row.raw_payload.get("invalid_detail") or "").lower()
    return "residual_sanitize_failed" in reason or (
        "residual" in reason and "sanitize" in reason
    ) or ("residual" in invalid_detail and "sanitize" in invalid_detail)


def _projection_failed(row: NormalizedCandidateRow) -> bool:
    if row.projection_attempted and not row.projection_success:
        return True
    reason = str(row.raw_payload.get("projection_reason") or "").lower()
    return "failed" in reason


def _reward_total(row: NormalizedCandidateRow) -> float | None:
    return _as_float(row.raw_payload.get("reward_total") or row.raw_payload.get("total"))


def _num_candidates_by_parent_distribution(rows: list[NormalizedCandidateRow]) -> dict[str, int]:
    parent_counter: Counter[str] = Counter(_row_parent_key(row) for row in rows)
    distribution: Counter[str] = Counter(str(count) for count in parent_counter.values())
    return dict(sorted(distribution.items(), key=lambda item: int(item[0])))


def _basic_stat_block(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": _safe_mean(values),
        "median": _safe_median(values),
        "p25": _safe_quantile(values, 0.25),
        "p75": _safe_quantile(values, 0.75),
    }


def _atom_ratio_histogram(rows: list[NormalizedCandidateRow]) -> dict[str, dict[str, float]]:
    total = len(rows)
    histogram = OrderedDict((label, 0) for label, *_rest in ATOM_RATIO_BUCKETS)
    histogram["full-parent"] = 0
    histogram["missing"] = 0
    for row in rows:
        bucket = _bucketize_atom_ratio(row)
        if bucket is None:
            histogram["missing"] += 1
        else:
            histogram[bucket] += 1
    return {
        bucket: {"count": int(count), "rate": _safe_rate(int(count), total)}
        for bucket, count in histogram.items()
    }


@lru_cache(maxsize=500000)
def _fragment_covers_parent(parent_smiles: str, fragment_smiles: str) -> bool:
    try:
        return bool(is_parent_substructure(parent_smiles, fragment_smiles))
    except Exception:
        return False


def _compute_parent_coverage(
    rows: list[NormalizedCandidateRow],
    dataset_records: list[PPOPromptRecord],
    *,
    config: FullPoolAuditConfig,
) -> dict[str, Any]:
    fragment_to_rows: dict[str, list[NormalizedCandidateRow]] = defaultdict(list)
    for row in rows:
        fragment_key = _canonical_fragment_key(row.final_fragment)
        if fragment_key:
            fragment_to_rows[fragment_key].append(row)

    coverage_records: list[dict[str, Any]] = []
    parents_for_coverage = list(dataset_records)
    if config.coverage_parent_limit > 0:
        parents_for_coverage = parents_for_coverage[: int(config.coverage_parent_limit)]

    for fragment_key, fragment_rows in fragment_to_rows.items():
        covered_parent_indices: list[int] = []
        for record in parents_for_coverage:
            if _fragment_covers_parent(record.parent_smiles, fragment_key):
                covered_parent_indices.append(int(record.parent_index))

        cf_drop_values = [
            value
            for value in (_as_float(row.raw_payload.get("cf_drop")) for row in fragment_rows)
            if value is not None
        ]
        cf_flip_rate = _safe_rate(
            sum(1 for row in fragment_rows if bool(_as_bool(row.raw_payload.get("cf_flip")))),
            len(fragment_rows),
        )
        coverage_records.append(
            {
                "final_fragment": fragment_key,
                "coverage_count": len(covered_parent_indices),
                "coverage_rate": _safe_rate(
                    len(covered_parent_indices),
                    len(parents_for_coverage),
                ),
                "covered_parent_indices": covered_parent_indices,
                "avg_cf_drop_on_generated_parents": _safe_mean(cf_drop_values),
                "cf_flip_rate_on_generated_parents": cf_flip_rate,
                "candidate_frequency": len(fragment_rows),
            }
        )

    coverage_records.sort(
        key=lambda item: (
            -int(item["coverage_count"]),
            -int(item["candidate_frequency"]),
            str(item["final_fragment"]),
        )
    )
    return {
        "num_dataset_parents": len(dataset_records),
        "num_parents_used_for_coverage": len(parents_for_coverage),
        "coverage_parent_limit": int(config.coverage_parent_limit),
        "num_unique_final_fragments": len(fragment_to_rows),
        "top_fragments": coverage_records[: max(1, int(config.topk_show))],
        "all_fragments": coverage_records,
    }


def _build_failure_cases(rows: list[NormalizedCandidateRow]) -> list[dict[str, Any]]:
    failure_rows: list[dict[str, Any]] = []
    for row in rows:
        failure_reasons: list[str] = []
        if not row.parse_ok:
            failure_reasons.append("parse_ok=false")
        if not row.final_substructure:
            failure_reasons.append("final_substructure=false")
        if row.core_unusable:
            failure_reasons.append("core_unusable=true")
        reward_total = _reward_total(row)
        if reward_total is not None and reward_total <= -5.0:
            failure_reasons.append("reward_total<=-5")
        if _projection_failed(row):
            failure_reasons.append("projection_failed")
        if _residual_sanitize_failed(row):
            failure_reasons.append("residual_sanitize_failed")
        if not failure_reasons:
            continue
        payload = dict(row.raw_payload)
        payload.update(
            {
                "_failure_reasons": failure_reasons,
                "parent_smiles": row.parent_smiles,
                "label": row.label,
                "final_fragment": row.final_fragment,
                "parse_ok": row.parse_ok,
                "final_substructure": row.final_substructure,
                "core_unusable": row.core_unusable,
                "reward_total": reward_total,
            }
        )
        failure_rows.append(payload)
    return failure_rows


def _selector_gate(overall: dict[str, Any]) -> dict[str, Any]:
    checks = OrderedDict(
        (
            ("final_substructure_rate>=0.9", (overall["final_substructure_rate"] or 0.0) >= 0.9),
            ("cf_flip_rate>=0.5", (overall["cf_flip_rate"] or 0.0) >= 0.5),
            ("unique_final_fragment_rate>=0.3", (overall["unique_final_fragment_rate"] or 0.0) >= 0.3),
            ("top5_final_fragment_ratio<=0.5", (overall["top5_final_fragment_ratio"] or 1.0) <= 0.5),
            (
                "mean_pairwise_tanimoto<=0.7",
                overall["mean_pairwise_tanimoto"] is not None
                and float(overall["mean_pairwise_tanimoto"] or 0.0) <= 0.7,
            ),
            ("projection_used_rate<=0.4", (overall["projection_used_rate"] or 1.0) <= 0.4),
            (
                "0.15<=atom_ratio_mean<=0.55",
                overall["atom_ratio_mean"] is not None
                and 0.15 <= float(overall["atom_ratio_mean"] or 0.0) <= 0.55,
            ),
        )
    )
    passed_count = sum(1 for passed in checks.values() if passed)
    ready = passed_count >= 5 and bool(checks["final_substructure_rate>=0.9"])
    return {
        "checks": checks,
        "passed_count": passed_count,
        "ready_for_selector": ready,
        "recommendation": (
            "ready_to_start_selector"
            if ready
            else "improve_pool_before_selector"
        ),
    }


def audit_full_candidate_pool(
    *,
    pool_jsonl: str | Path,
    dataset_path: str | Path,
    teacher_path: str | Path,
    out_dir: str | Path,
    config: FullPoolAuditConfig,
) -> dict[str, Any]:
    """Audit one full candidate pool and persist selector-facing artifacts."""

    pool_path = Path(pool_jsonl).expanduser().resolve()
    dataset_records, dataset_metadata = load_ppo_prompt_records(
        dataset_path,
        label_col=config.label_col,
        smiles_col=config.smiles_col,
        target_label=config.target_label,
        limit=0,
    )
    rows = [_normalize_row(row, record_index=index) for index, row in enumerate(read_jsonl(pool_path))]
    total = len(rows)

    label_distribution = Counter(_row_label_key(row) for row in rows)
    parent_counter = Counter(_row_parent_key(row) for row in rows)
    projection_method_counter = Counter(_projection_method_bucket(row) for row in rows)
    counterfactual_reason_counter = Counter(_counterfactual_reason(row) for row in rows)

    cf_drop_values = [row.cf_drop for row in rows if row.cf_drop is not None]
    p_before_values = [row.p_before for row in rows if row.p_before is not None]
    p_after_values = [row.p_after for row in rows if row.p_after is not None]
    fragment_atom_counts = [
        float(row.atom_count) for row in rows if row.atom_count is not None
    ]
    atom_ratio_values = [
        float(row.atom_ratio) for row in rows if row.atom_ratio is not None
    ]

    raw_counter = Counter(
        key for key in (_canonical_fragment_key(row.raw_fragment) for row in rows) if key
    )
    core_counter = Counter(
        key for key in (_canonical_fragment_key(row.core_fragment) for row in rows) if key
    )
    projected_counter = Counter(
        key for key in (_canonical_fragment_key(row.projected_fragment) for row in rows) if key
    )
    final_counter = Counter(
        key for key in (_canonical_fragment_key(row.final_fragment) for row in rows) if key
    )

    similarity_summary = _build_similarity_stats(
        list(final_counter.keys()),
        config.sim_sample_size,
    )

    coverage_summary = _compute_parent_coverage(
        rows,
        dataset_records,
        config=config,
    )
    failure_rows = _build_failure_cases(rows)

    top1_count = final_counter.most_common(1)[0][1] if final_counter else 0
    top5_count = sum(count for _fragment, count in final_counter.most_common(5))
    top10_count = sum(count for _fragment, count in final_counter.most_common(10))

    overall = {
        "num_rows": total,
        "num_unique_parents": len(parent_counter),
        "avg_candidates_per_parent": (
            float(total) / float(len(parent_counter)) if parent_counter else 0.0
        ),
        "num_candidates_by_parent_distribution": _num_candidates_by_parent_distribution(rows),
        "label_distribution": dict(sorted(label_distribution.items())),
        "parse_ok_rate": _safe_rate(sum(1 for row in rows if row.parse_ok), total),
        "valid_rate": _safe_rate(sum(1 for row in rows if row.valid), total),
        "connected_rate": _safe_rate(sum(1 for row in rows if row.connected), total),
        "direct_substructure_rate": _safe_rate(
            sum(1 for row in rows if row.direct_substructure),
            total,
        ),
        "final_substructure_rate": _safe_rate(
            sum(1 for row in rows if row.final_substructure),
            total,
        ),
        "projection_used_rate": _safe_rate(
            sum(1 for row in rows if row.projection_used),
            total,
        ),
        "projection_direct_match_rate": _safe_rate(
            projection_method_counter.get("direct_match", 0),
            total,
        ),
        "projection_nearest_parent_subgraph_rate": _safe_rate(
            projection_method_counter.get("nearest_parent_subgraph", 0),
            total,
        ),
        "projection_failed_rate": _safe_rate(
            projection_method_counter.get("failed", 0),
            total,
        ),
        "core_unusable_rate": _safe_rate(sum(1 for row in rows if row.core_unusable), total),
        "parse_failed_rate": _safe_rate(sum(1 for row in rows if not row.parse_ok), total),
        "near_parent_rate": _safe_rate(sum(1 for row in rows if row.near_parent), total),
        "too_small_rate": _safe_rate(sum(1 for row in rows if row.too_small), total),
        "oracle_ok_rate": _safe_rate(sum(1 for row in rows if row.oracle_ok), total),
        "cf_oracle_called_rate": _safe_rate(sum(1 for row in rows if _cf_oracle_called(row)), total),
        "cf_oracle_skipped_rate": _safe_rate(sum(1 for row in rows if not _cf_oracle_called(row)), total),
        "cf_flip_rate": _safe_rate(sum(1 for row in rows if row.cf_flip), total),
        "cf_drop_mean": _safe_mean(cf_drop_values),
        "cf_drop_median": _safe_median(cf_drop_values),
        "cf_drop_p25": _safe_quantile(cf_drop_values, 0.25),
        "cf_drop_p75": _safe_quantile(cf_drop_values, 0.75),
        "p_before_mean": _safe_mean(p_before_values),
        "p_before_median": _safe_median(p_before_values),
        "p_after_mean": _safe_mean(p_after_values),
        "p_after_median": _safe_median(p_after_values),
        "residual_sanitize_failed_rate": _safe_rate(
            sum(1 for row in rows if _residual_sanitize_failed(row)),
            total,
        ),
        "counterfactual_reason_distribution": dict(
            sorted(counterfactual_reason_counter.items(), key=lambda item: (-item[1], item[0]))
        ),
        "fragment_atom_count_mean": _safe_mean(fragment_atom_counts),
        "fragment_atom_count_median": _safe_median(fragment_atom_counts),
        "atom_ratio_mean": _safe_mean(atom_ratio_values),
        "atom_ratio_median": _safe_median(atom_ratio_values),
        "atom_ratio_p25": _safe_quantile(atom_ratio_values, 0.25),
        "atom_ratio_p75": _safe_quantile(atom_ratio_values, 0.75),
        "atom_ratio_histogram": _atom_ratio_histogram(rows),
        "unique_raw_fragment_count": len(raw_counter),
        "unique_core_fragment_count": len(core_counter),
        "unique_projected_fragment_count": len(projected_counter),
        "unique_final_fragment_count": len(final_counter),
        "unique_final_fragment_rate": _safe_rate(len(final_counter), total),
        "top1_final_fragment_ratio": _safe_rate(top1_count, total),
        "top5_final_fragment_ratio": _safe_rate(top5_count, total),
        "top10_final_fragment_ratio": _safe_rate(top10_count, total),
        "mean_pairwise_tanimoto": similarity_summary["mean_pairwise_tanimoto"],
        "median_pairwise_tanimoto": similarity_summary["median_pairwise_tanimoto"],
        "skipped_similarity_count": similarity_summary["skipped_similarity_count"],
    }

    selector_gate = _selector_gate(overall)
    diversity_summary = {
        "unique_raw_fragment_count": len(raw_counter),
        "unique_core_fragment_count": len(core_counter),
        "unique_projected_fragment_count": len(projected_counter),
        "unique_final_fragment_count": len(final_counter),
        "unique_final_fragment_rate": overall["unique_final_fragment_rate"],
        "top1_final_fragment_ratio": overall["top1_final_fragment_ratio"],
        "top5_final_fragment_ratio": overall["top5_final_fragment_ratio"],
        "top10_final_fragment_ratio": overall["top10_final_fragment_ratio"],
        "mean_pairwise_tanimoto": overall["mean_pairwise_tanimoto"],
        "median_pairwise_tanimoto": overall["median_pairwise_tanimoto"],
        "skipped_similarity_count": overall["skipped_similarity_count"],
    }

    summary = {
        "metadata": {
            "pool_jsonl": str(pool_path),
            "dataset_path": str(Path(dataset_path).expanduser().resolve()),
            "teacher_path": str(Path(teacher_path).expanduser().resolve()),
            "out_dir": str(Path(out_dir).expanduser().resolve()),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
            "dataset": dataset_metadata,
        },
        "overall": overall,
        "selector_gate": selector_gate,
    }

    _write_full_audit_outputs(
        summary=summary,
        diversity_summary=diversity_summary,
        coverage_summary=coverage_summary,
        failure_rows=failure_rows,
        final_counter=final_counter,
        out_dir=out_dir,
        topk_show=config.topk_show,
    )
    return summary


def _render_report(
    summary: dict[str, Any],
    *,
    coverage_summary: dict[str, Any],
    diversity_summary: dict[str, Any],
) -> str:
    overall = summary["overall"]
    selector_gate = summary["selector_gate"]
    lines = [
        "Full Candidate Pool Audit",
        f"pool_jsonl: {summary['metadata']['pool_jsonl']}",
        f"dataset_path: {summary['metadata']['dataset_path']}",
        f"teacher_path: {summary['metadata']['teacher_path']}",
        f"generated_at_utc: {summary['metadata']['generated_at_utc']}",
        "",
        "Scale:",
        f"- num_rows: {overall['num_rows']}",
        f"- num_unique_parents: {overall['num_unique_parents']}",
        f"- avg_candidates_per_parent: {overall['avg_candidates_per_parent']:.4f}",
        f"- label_distribution: {overall['label_distribution']}",
        "",
        "Legality:",
        f"- parse_ok_rate: {overall['parse_ok_rate']:.4f}",
        f"- valid_rate: {overall['valid_rate']:.4f}",
        f"- connected_rate: {overall['connected_rate']:.4f}",
        f"- direct_substructure_rate: {overall['direct_substructure_rate']:.4f}",
        f"- final_substructure_rate: {overall['final_substructure_rate']:.4f}",
        f"- projection_used_rate: {overall['projection_used_rate']:.4f}",
        f"- projection_direct_match_rate: {overall['projection_direct_match_rate']:.4f}",
        f"- projection_nearest_parent_subgraph_rate: {overall['projection_nearest_parent_subgraph_rate']:.4f}",
        f"- projection_failed_rate: {overall['projection_failed_rate']:.4f}",
        "",
        "Counterfactuality:",
        f"- oracle_ok_rate: {overall['oracle_ok_rate']:.4f}",
        f"- cf_oracle_called_rate: {overall['cf_oracle_called_rate']:.4f}",
        f"- cf_oracle_skipped_rate: {overall['cf_oracle_skipped_rate']:.4f}",
        f"- cf_flip_rate: {overall['cf_flip_rate']:.4f}",
        f"- cf_drop_mean: {overall['cf_drop_mean'] if overall['cf_drop_mean'] is not None else 'n/a'}",
        f"- cf_drop_median: {overall['cf_drop_median'] if overall['cf_drop_median'] is not None else 'n/a'}",
        f"- p_before_mean: {overall['p_before_mean'] if overall['p_before_mean'] is not None else 'n/a'}",
        f"- p_after_mean: {overall['p_after_mean'] if overall['p_after_mean'] is not None else 'n/a'}",
        "",
        "Diversity:",
        f"- unique_final_fragment_rate: {diversity_summary['unique_final_fragment_rate']:.4f}",
        f"- top5_final_fragment_ratio: {diversity_summary['top5_final_fragment_ratio']:.4f}",
        f"- mean_pairwise_tanimoto: {diversity_summary['mean_pairwise_tanimoto'] if diversity_summary['mean_pairwise_tanimoto'] is not None else 'n/a'}",
        "",
        "Coverage:",
        f"- num_parents_used_for_coverage: {coverage_summary['num_parents_used_for_coverage']}",
        f"- num_unique_final_fragments: {coverage_summary['num_unique_final_fragments']}",
        "",
        "Selector gate:",
    ]
    for check_name, passed in selector_gate["checks"].items():
        lines.append(f"- {check_name}: {'pass' if passed else 'fail'}")
    lines.extend(
        [
            f"- passed_count: {selector_gate['passed_count']}",
            f"- ready_for_selector: {'yes' if selector_gate['ready_for_selector'] else 'not_yet'}",
            f"- recommendation: {selector_gate['recommendation']}",
            "",
            "Top fragment coverage:",
        ]
    )
    for item in coverage_summary["top_fragments"]:
        lines.append(
            f"- {item['final_fragment']}: freq={item['candidate_frequency']} coverage_count={item['coverage_count']} coverage_rate={item['coverage_rate']:.4f} avg_cf_drop={item['avg_cf_drop_on_generated_parents'] if item['avg_cf_drop_on_generated_parents'] is not None else 'n/a'} cf_flip_rate={item['cf_flip_rate_on_generated_parents']:.4f}"
        )
    return "\n".join(lines) + "\n"


def _write_fragment_frequency_csv(
    final_counter: Counter[str],
    coverage_summary: dict[str, Any],
    *,
    destination: Path,
    topk_show: int,
) -> None:
    coverage_by_fragment = {
        str(item["final_fragment"]): item
        for item in coverage_summary["all_fragments"]
    }
    ensure_directory(destination.parent)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "final_fragment",
                "count",
                "ratio",
                "coverage_count",
                "coverage_rate",
                "avg_cf_drop_on_generated_parents",
                "cf_flip_rate_on_generated_parents",
            ],
        )
        writer.writeheader()
        total = sum(final_counter.values())
        for fragment, count in final_counter.most_common(max(1, int(topk_show))):
            coverage_item = coverage_by_fragment.get(fragment, {})
            writer.writerow(
                {
                    "final_fragment": fragment,
                    "count": int(count),
                    "ratio": _safe_rate(int(count), total),
                    "coverage_count": coverage_item.get("coverage_count"),
                    "coverage_rate": coverage_item.get("coverage_rate"),
                    "avg_cf_drop_on_generated_parents": coverage_item.get(
                        "avg_cf_drop_on_generated_parents"
                    ),
                    "cf_flip_rate_on_generated_parents": coverage_item.get(
                        "cf_flip_rate_on_generated_parents"
                    ),
                }
            )


def _write_full_audit_outputs(
    *,
    summary: dict[str, Any],
    diversity_summary: dict[str, Any],
    coverage_summary: dict[str, Any],
    failure_rows: list[dict[str, Any]],
    final_counter: Counter[str],
    out_dir: str | Path,
    topk_show: int,
) -> None:
    destination = Path(out_dir).expanduser().resolve()
    ensure_directory(destination)

    (destination / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (destination / "diversity_summary.json").write_text(
        json.dumps(diversity_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (destination / "parent_coverage_summary.json").write_text(
        json.dumps(coverage_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_jsonl(destination / "failure_cases.jsonl", failure_rows)
    _write_fragment_frequency_csv(
        final_counter,
        coverage_summary,
        destination=destination / "fragment_frequency_topk.csv",
        topk_show=topk_show,
    )
    (destination / "audit_report.txt").write_text(
        _render_report(
            summary,
            coverage_summary=coverage_summary,
            diversity_summary=diversity_summary,
        ),
        encoding="utf-8",
    )


__all__ = ["FullPoolAuditConfig", "audit_full_candidate_pool"]
