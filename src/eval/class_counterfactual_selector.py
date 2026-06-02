"""Class-level greedy MMR selector for counterfactual fragment pools."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any

from src.chem import parse_smiles
from src.eval.candidate_pool_audit import (
    NormalizedCandidateRow,
    _as_float,
    _canonical_fragment_key,
    _coalesce,
    _normalize_row,
    _normalize_text,
)
from src.eval.subgraph_similarity import (
    DEFAULT_EMBEDDING_FIELD,
    cosine_embedding_similarity,
    get_candidate_embedding,
)
from src.utils.io import ensure_directory, read_jsonl

try:  # pragma: no cover - depends on runtime env
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
except ImportError:  # pragma: no cover - depends on runtime env
    DataStructs = None
    rdFingerprintGenerator = None


@dataclass(frozen=True, slots=True)
class SelectorConfig:
    """Execution knobs for class-level counterfactual selection."""

    label: int
    top_k: int = 20
    alpha_cf: float = 1.0
    beta_coverage: float = 1.0
    gamma_redundancy: float = 0.7
    eta_size: float = 0.3
    min_cf_drop: float = 0.2
    require_cf_flip: bool = False
    require_final_substructure: bool = False
    max_projection_used_rate: float = 1.0
    sim_metric: str = "morgan"
    embedding_field: str = DEFAULT_EMBEDDING_FIELD
    embedding_missing_policy: str = "error"
    top_candidates_per_fragment: int = 3
    dedup_by_final_fragment: bool = False


@dataclass(frozen=True, slots=True)
class SelectorCandidate:
    """One normalized candidate row enriched with selector-specific metadata."""

    record_index: int
    parent_key: str
    parent_id: str
    parent_smiles: str
    label: int
    raw_fragment: str | None
    final_fragment: str
    final_fragment_key: str
    parse_ok: bool
    valid: bool
    connected: bool
    final_substructure: bool
    projection_used: bool
    oracle_ok: bool
    cf_flip: bool
    cf_drop: float
    reward_total: float | None
    atom_ratio: float | None
    failure_tag: str | None
    full_parent: bool
    near_parent: bool
    too_small: bool
    embedding: tuple[float, ...] | None
    embedding_field_used: str | None
    normalized_row: NormalizedCandidateRow
    raw_payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FragmentAggregate:
    """Aggregated statistics for one shared fragment."""

    fragment: str
    parent_keys: tuple[str, ...]
    parent_ids: tuple[str, ...]
    parent_smiles: tuple[str, ...]
    example_raw_fragments: tuple[str, ...]
    support_count: int
    support_rate: float
    mean_cf_drop: float
    median_cf_drop: float
    cf_flip_rate: float
    mean_reward: float | None
    mean_atom_ratio: float | None
    projection_used_rate: float
    representative_embedding: tuple[float, ...] | None
    representative_embedding_field: str | None
    candidate_rows: tuple[SelectorCandidate, ...]


def _safe_mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _parse_bool_string(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in {"", "none", "null"}


def _extract_parent_id(payload: dict[str, Any], normalized_row: NormalizedCandidateRow) -> str:
    explicit = _normalize_text(
        _coalesce(
            payload,
            "parent_id",
            "example_id",
            "source_id",
            "dataset_row_id",
        )
    )
    if explicit:
        return explicit
    parent_index = _coalesce(payload, "parent_index")
    if parent_index is not None:
        return str(parent_index)
    if normalized_row.parent_smiles:
        return normalized_row.parent_smiles
    return f"row-{normalized_row.record_index}"


def _extract_reward_total(payload: dict[str, Any]) -> float | None:
    return _as_float(_coalesce(payload, "reward_total", "total_reward"))


def _is_failure_free(failure_tag: str | None) -> bool:
    normalized = _normalize_text(failure_tag)
    if normalized is None:
        return True
    return normalized.lower() in {"none", "null"}


def _size_penalty(atom_ratio: float | None) -> float:
    if atom_ratio is None:
        return 0.30
    penalty = abs(float(atom_ratio) - 0.30)
    if atom_ratio > 0.55:
        penalty += (float(atom_ratio) - 0.55) * 2.0
    return penalty


@lru_cache(maxsize=1)
def _get_morgan_fp_generator() -> Any | None:
    if rdFingerprintGenerator is None:
        return None
    try:
        return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    except Exception:
        return None


@lru_cache(maxsize=4096)
def _fingerprint_for_fragment(fragment: str, sim_metric: str) -> Any | None:
    if sim_metric != "morgan" or DataStructs is None:
        return None
    generator = _get_morgan_fp_generator()
    if generator is None:
        return None
    parsed = parse_smiles(fragment, sanitize=True, canonicalize=True)
    if not parsed.parseable or not parsed.sanitized or parsed.mol is None:
        return None
    try:
        return generator.GetFingerprint(parsed.mol)
    except Exception:
        return None


def _fragment_similarity(left: str, right: str, sim_metric: str) -> float:
    if left == right:
        return 1.0
    left_fp = _fingerprint_for_fragment(left, sim_metric)
    right_fp = _fingerprint_for_fragment(right, sim_metric)
    if left_fp is None or right_fp is None or DataStructs is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(left_fp, right_fp))


def _embedding_similarity(
    left_embedding: tuple[float, ...] | None,
    right_embedding: tuple[float, ...] | None,
) -> float:
    if left_embedding is None or right_embedding is None:
        return 0.0
    try:
        return cosine_embedding_similarity(left_embedding, right_embedding)
    except ValueError:
        return 0.0


def _aggregate_similarity(
    left: FragmentAggregate,
    right: FragmentAggregate,
    config: SelectorConfig,
) -> float:
    if config.sim_metric == "morgan":
        return _fragment_similarity(left.fragment, right.fragment, "morgan")
    if config.sim_metric == "embedding":
        return _embedding_similarity(
            left.representative_embedding,
            right.representative_embedding,
        )
    raise ValueError(f"Unsupported sim_metric: {config.sim_metric!r}")


def _selected_pairwise_similarity(
    fragments: list[str],
    sim_metric: str,
) -> tuple[float | None, float | None]:
    if len(fragments) < 2:
        return None, None
    values: list[float] = []
    for left_index in range(len(fragments)):
        for right_index in range(left_index + 1, len(fragments)):
            values.append(
                _fragment_similarity(
                    fragments[left_index],
                    fragments[right_index],
                    sim_metric,
                )
            )
    if not values:
        return None, None
    return _safe_mean(values), max(values)


def _selected_pairwise_embedding_similarity(
    selected_rows: list[dict[str, Any]],
) -> dict[str, float | None]:
    embeddings = [
        tuple(float(value) for value in row["representative_embedding"])
        for row in selected_rows
        if row.get("representative_embedding") is not None
    ]
    values: list[float] = []
    for left_index in range(len(embeddings)):
        for right_index in range(left_index + 1, len(embeddings)):
            values.append(
                _embedding_similarity(
                    embeddings[left_index],
                    embeddings[right_index],
                )
            )
    if not values:
        return {
            "selected_pairwise_embedding_cosine_mean": None,
            "selected_pairwise_embedding_cosine_max": None,
            "selected_pairwise_embedding_cosine_min": None,
            "selected_pairwise_embedding_cosine_std": None,
        }
    mean_value = _safe_mean(values)
    variance = (
        sum((value - float(mean_value)) ** 2 for value in values) / len(values)
        if mean_value is not None
        else 0.0
    )
    return {
        "selected_pairwise_embedding_cosine_mean": mean_value,
        "selected_pairwise_embedding_cosine_max": max(values),
        "selected_pairwise_embedding_cosine_min": min(values),
        "selected_pairwise_embedding_cosine_std": variance ** 0.5,
    }


def _candidate_embedding_for_config(
    payload: dict[str, Any],
    config: SelectorConfig,
) -> tuple[tuple[float, ...] | None, str | None, str | None]:
    if config.sim_metric != "embedding":
        return None, None, None
    try:
        parsed = get_candidate_embedding(payload, config.embedding_field)
    except ValueError as exc:
        if config.embedding_missing_policy == "skip":
            return None, None, str(exc)
        raise ValueError(
            f"--sim-metric embedding requires usable embeddings in candidate_pool.jsonl: {exc}"
        ) from exc
    return tuple(float(value) for value in parsed.vector.tolist()), parsed.field_name, None


def _build_selector_candidates(
    rows: list[dict[str, Any]],
    config: SelectorConfig,
) -> tuple[list[SelectorCandidate], dict[str, int], int]:
    filter_counts: dict[str, int] = defaultdict(int)
    accepted: list[SelectorCandidate] = []
    total_label_parent_keys: set[str] = set()

    for index, payload in enumerate(rows):
        normalized = _normalize_row(payload, record_index=index)
        if normalized.label != config.label:
            filter_counts["label_mismatch"] += 1
            continue

        parent_smiles = normalized.parent_smiles
        if parent_smiles:
            parent_key = _extract_parent_id(payload, normalized)
            total_label_parent_keys.add(parent_key)
        else:
            parent_key = _extract_parent_id(payload, normalized)

        final_fragment = normalized.final_fragment
        fragment_key = _canonical_fragment_key(final_fragment)
        if fragment_key is None or final_fragment is None:
            filter_counts["missing_final_fragment"] += 1
            continue
        if config.require_final_substructure and not normalized.final_substructure:
            filter_counts["final_substructure_fail"] += 1
            continue
        if not normalized.parse_ok:
            filter_counts["parse_fail"] += 1
            continue
        if not normalized.valid:
            filter_counts["valid_fail"] += 1
            continue
        if not normalized.connected:
            filter_counts["connected_fail"] += 1
            continue
        if not normalized.oracle_ok:
            filter_counts["oracle_fail"] += 1
            continue
        if normalized.cf_drop is None or float(normalized.cf_drop) < float(config.min_cf_drop):
            filter_counts["cf_drop_fail"] += 1
            continue
        if config.require_cf_flip and not normalized.cf_flip:
            filter_counts["cf_flip_fail"] += 1
            continue
        if not _is_failure_free(normalized.failure_tag):
            filter_counts["failure_tag_fail"] += 1
            continue
        if normalized.full_parent:
            filter_counts["full_parent_fail"] += 1
            continue
        if normalized.near_parent:
            filter_counts["near_parent_fail"] += 1
            continue
        if normalized.too_small:
            filter_counts["too_small_fail"] += 1
            continue
        if not parent_smiles:
            filter_counts["missing_parent_smiles"] += 1
            continue
        embedding, embedding_field_used, embedding_error = _candidate_embedding_for_config(payload, config)
        if embedding_error is not None:
            filter_counts["embedding_missing_or_invalid"] += 1
            continue

        accepted.append(
            SelectorCandidate(
                record_index=index,
                parent_key=parent_key,
                parent_id=parent_key,
                parent_smiles=parent_smiles,
                label=int(normalized.label),
                raw_fragment=normalized.raw_fragment,
                final_fragment=final_fragment,
                final_fragment_key=fragment_key,
                parse_ok=normalized.parse_ok,
                valid=normalized.valid,
                connected=normalized.connected,
                final_substructure=normalized.final_substructure,
                projection_used=normalized.projection_used,
                oracle_ok=normalized.oracle_ok,
                cf_flip=normalized.cf_flip,
                cf_drop=float(normalized.cf_drop),
                reward_total=_extract_reward_total(payload),
                atom_ratio=normalized.atom_ratio,
                failure_tag=normalized.failure_tag,
                full_parent=normalized.full_parent,
                near_parent=normalized.near_parent,
                too_small=normalized.too_small,
                embedding=embedding,
                embedding_field_used=embedding_field_used,
                normalized_row=normalized,
                raw_payload=payload,
            )
        )

    return accepted, dict(filter_counts), len(total_label_parent_keys)


def _candidate_rank_key(candidate: SelectorCandidate) -> tuple[float, float, int]:
    return (
        float(candidate.cf_drop),
        float(candidate.reward_total) if candidate.reward_total is not None else float("-inf"),
        -candidate.record_index,
    )


def _aggregate_fragments(
    candidates: list[SelectorCandidate],
    *,
    total_parent_count: int,
    config: SelectorConfig,
) -> list[FragmentAggregate]:
    grouped: dict[str, list[SelectorCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.final_fragment_key].append(candidate)

    aggregates: list[FragmentAggregate] = []
    for fragment_key, group_rows in grouped.items():
        if config.dedup_by_final_fragment:
            best_by_parent: dict[str, SelectorCandidate] = {}
            for candidate in group_rows:
                current = best_by_parent.get(candidate.parent_key)
                if current is None or _candidate_rank_key(candidate) > _candidate_rank_key(current):
                    best_by_parent[candidate.parent_key] = candidate
            selected_rows = list(best_by_parent.values())
        else:
            selected_rows = list(group_rows)

        selected_rows.sort(key=_candidate_rank_key, reverse=True)
        parent_keys = sorted({candidate.parent_key for candidate in selected_rows})
        parent_ids = sorted({candidate.parent_id for candidate in selected_rows})
        parent_smiles = sorted({candidate.parent_smiles for candidate in selected_rows})
        cf_drops = [candidate.cf_drop for candidate in selected_rows]
        cf_flip_rate = _safe_rate(sum(1 for candidate in selected_rows if candidate.cf_flip), len(selected_rows))
        rewards = [
            float(candidate.reward_total)
            for candidate in selected_rows
            if candidate.reward_total is not None
        ]
        atom_ratios = [
            float(candidate.atom_ratio)
            for candidate in selected_rows
            if candidate.atom_ratio is not None
        ]
        projection_used_rate = _safe_rate(
            sum(1 for candidate in selected_rows if candidate.projection_used),
            len(selected_rows),
        )
        if projection_used_rate > float(config.max_projection_used_rate):
            continue
        representative = selected_rows[0]
        aggregates.append(
            FragmentAggregate(
                fragment=fragment_key,
                parent_keys=tuple(parent_keys),
                parent_ids=tuple(parent_ids),
                parent_smiles=tuple(parent_smiles),
                example_raw_fragments=tuple(
                    fragment
                    for fragment in dict.fromkeys(
                        candidate.raw_fragment or candidate.final_fragment
                        for candidate in selected_rows[: max(1, int(config.top_candidates_per_fragment))]
                    )
                    if fragment
                ),
                support_count=len(parent_keys),
                support_rate=_safe_rate(len(parent_keys), total_parent_count),
                mean_cf_drop=sum(cf_drops) / len(cf_drops),
                median_cf_drop=float(median(cf_drops)),
                cf_flip_rate=cf_flip_rate,
                mean_reward=_safe_mean(rewards),
                mean_atom_ratio=_safe_mean(atom_ratios),
                projection_used_rate=projection_used_rate,
                representative_embedding=representative.embedding,
                representative_embedding_field=representative.embedding_field_used,
                candidate_rows=tuple(selected_rows),
            )
        )
    aggregates.sort(
        key=lambda item: (
            float(item.mean_cf_drop),
            float(item.cf_flip_rate),
            float(item.support_count),
        ),
        reverse=True,
    )
    return aggregates


def _cf_score(aggregate: FragmentAggregate) -> float:
    return float(aggregate.mean_cf_drop) + 0.2 * float(aggregate.cf_flip_rate)


def _select_fragments(
    aggregates: list[FragmentAggregate],
    *,
    total_parent_count: int,
    config: SelectorConfig,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_aggregates: list[FragmentAggregate] = []
    covered_parents: set[str] = set()

    while len(selected) < int(config.top_k):
        best_choice: dict[str, Any] | None = None
        best_score: float | None = None
        selected_fragments = [item["fragment"] for item in selected]

        for aggregate in aggregates:
            if aggregate.fragment in selected_fragments:
                continue
            new_parents = set(aggregate.parent_keys) - covered_parents
            coverage_gain = _safe_rate(len(new_parents), total_parent_count)
            max_similarity = 0.0
            if selected_aggregates:
                max_similarity = max(
                    _aggregate_similarity(aggregate, selected_aggregate, config)
                    for selected_aggregate in selected_aggregates
                )
            size_penalty = _size_penalty(aggregate.mean_atom_ratio)
            cf_score = _cf_score(aggregate)
            score = (
                float(config.alpha_cf) * cf_score
                + float(config.beta_coverage) * coverage_gain
                - float(config.gamma_redundancy) * max_similarity
                - float(config.eta_size) * size_penalty
            )
            choice = {
                "fragment": aggregate.fragment,
                "score": score,
                "coverage_gain": coverage_gain,
                "new_parent_count": len(new_parents),
                "max_similarity_to_previous": max_similarity,
                "cf_score": cf_score,
                "size_penalty": size_penalty,
                "aggregate": aggregate,
            }
            if best_score is None or score > best_score:
                best_score = score
                best_choice = choice

        if best_choice is None:
            break

        aggregate = best_choice["aggregate"]
        covered_parents.update(aggregate.parent_keys)
        rank = len(selected) + 1
        selected_aggregates.append(aggregate)
        selected.append(
            {
                "rank": rank,
                "selected_step": rank,
                "fragment": aggregate.fragment,
                "score": float(best_choice["score"]),
                "mmr_score": float(best_choice["score"]),
                "support_count": int(aggregate.support_count),
                "support_rate": float(aggregate.support_rate),
                "coverage_gain": float(best_choice["coverage_gain"]),
                "cumulative_coverage": _safe_rate(len(covered_parents), total_parent_count),
                "mean_cf_drop": float(aggregate.mean_cf_drop),
                "median_cf_drop": float(aggregate.median_cf_drop),
                "cf_flip_rate": float(aggregate.cf_flip_rate),
                "mean_reward": aggregate.mean_reward,
                "mean_atom_ratio": aggregate.mean_atom_ratio,
                "projection_used_rate": float(aggregate.projection_used_rate),
                "max_similarity_to_previous": float(best_choice["max_similarity_to_previous"]),
                "max_redundancy_sim_at_selection": float(best_choice["max_similarity_to_previous"]),
                "redundancy_sim_metric": str(config.sim_metric),
                "cf_score": float(best_choice["cf_score"]),
                "size_penalty": float(best_choice["size_penalty"]),
                "embedding_field": (
                    str(config.embedding_field) if config.sim_metric == "embedding" else None
                ),
                "representative_embedding": (
                    list(aggregate.representative_embedding)
                    if aggregate.representative_embedding is not None
                    else None
                ),
                "representative_embedding_field": aggregate.representative_embedding_field,
                "representative_parent_ids": list(
                    aggregate.parent_ids[: max(1, int(config.top_candidates_per_fragment))]
                ),
                "representative_parent_smiles": list(
                    aggregate.parent_smiles[: max(1, int(config.top_candidates_per_fragment))]
                ),
                "representative_raw_fragments": list(aggregate.example_raw_fragments),
            }
        )

    return selected


def render_selector_report(summary: dict[str, Any], selected_rows: list[dict[str, Any]]) -> str:
    """Render a concise selector report."""

    lines = [
        "Class Counterfactual Selector",
        f"pool_jsonl: {summary['metadata']['pool_jsonl']}",
        f"generated_at_utc: {summary['metadata']['generated_at_utc']}",
        "",
        "Summary:",
        f"- input_candidate_count: {summary['input_candidate_count']}",
        f"- valid_candidate_count_after_filter: {summary['valid_candidate_count_after_filter']}",
        f"- unique_fragment_count_after_filter: {summary['unique_fragment_count_after_filter']}",
        f"- selected_count: {summary['selected_count']}",
        f"- total_parent_count: {summary['total_parent_count']}",
        f"- filtered_parent_count: {summary['filtered_parent_count']}",
        f"- final_cumulative_coverage: {summary['final_cumulative_coverage']:.4f}",
        f"- selected_mean_cf_drop: {summary['selected_mean_cf_drop'] if summary['selected_mean_cf_drop'] is not None else 'n/a'}",
        f"- selected_cf_flip_rate: {summary['selected_cf_flip_rate'] if summary['selected_cf_flip_rate'] is not None else 'n/a'}",
        f"- selected_mean_atom_ratio: {summary['selected_mean_atom_ratio'] if summary['selected_mean_atom_ratio'] is not None else 'n/a'}",
        f"- selected_pairwise_tanimoto_mean: {summary['selected_pairwise_tanimoto_mean'] if summary['selected_pairwise_tanimoto_mean'] is not None else 'n/a'}",
        f"- selected_pairwise_tanimoto_max: {summary['selected_pairwise_tanimoto_max'] if summary['selected_pairwise_tanimoto_max'] is not None else 'n/a'}",
        "",
        "Redundancy similarity metric:",
        f"- sim_metric: {summary['sim_metric']}",
        f"- embedding_field: {summary['embedding_field'] if summary['sim_metric'] == 'embedding' else 'n/a'}",
        "- embedding cosine mapping: max(0, cosine)",
        f"- selected_pairwise_embedding_cosine_mean: {summary['selected_pairwise_embedding_cosine_mean'] if summary['selected_pairwise_embedding_cosine_mean'] is not None else 'n/a'}",
        f"- selected_pairwise_embedding_cosine_max: {summary['selected_pairwise_embedding_cosine_max'] if summary['selected_pairwise_embedding_cosine_max'] is not None else 'n/a'}",
        f"- selected_pairwise_embedding_cosine_min: {summary['selected_pairwise_embedding_cosine_min'] if summary['selected_pairwise_embedding_cosine_min'] is not None else 'n/a'}",
        f"- selected_pairwise_embedding_cosine_std: {summary['selected_pairwise_embedding_cosine_std'] if summary['selected_pairwise_embedding_cosine_std'] is not None else 'n/a'}",
        "",
        "Filter counts:",
    ]
    for key, value in sorted(summary["filter_counts"].items()):
        lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("Selected fragments:")
    if not selected_rows:
        lines.append("- none")
    else:
        for row in selected_rows:
            lines.append(
                f"- rank={row['rank']} fragment={row['fragment']} score={row['score']:.4f} support={row['support_count']} coverage={row['cumulative_coverage']:.4f} cf_drop={row['mean_cf_drop']:.4f} cf_flip_rate={row['cf_flip_rate']:.4f} atom_ratio={row['mean_atom_ratio'] if row['mean_atom_ratio'] is not None else 'n/a'} redundancy={row['max_similarity_to_previous']:.4f}"
            )
    return "\n".join(lines) + "\n"


def write_selector_outputs(
    *,
    out_dir: str | Path,
    selected_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, str]:
    """Persist selector artifacts to disk."""

    output_dir = Path(out_dir).expanduser().resolve()
    ensure_directory(output_dir)
    public_selected_rows = [
        {key: value for key, value in row.items() if key != "representative_embedding"}
        for row in selected_rows
    ]
    json_path = output_dir / "selected_subgraphs.json"
    csv_path = output_dir / "selected_subgraphs.csv"
    summary_path = output_dir / "selector_summary.json"
    report_path = output_dir / "selector_report.txt"

    json_path.write_text(
        json.dumps(public_selected_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_selector_report(summary, public_selected_rows), encoding="utf-8")

    fieldnames = [
        "rank",
        "selected_step",
        "fragment",
        "score",
        "mmr_score",
        "support_count",
        "support_rate",
        "coverage_gain",
        "cumulative_coverage",
        "mean_cf_drop",
        "median_cf_drop",
        "cf_flip_rate",
        "cf_score",
        "mean_reward",
        "mean_atom_ratio",
        "size_penalty",
        "projection_used_rate",
        "max_similarity_to_previous",
        "max_redundancy_sim_at_selection",
        "redundancy_sim_metric",
        "embedding_field",
        "representative_embedding_field",
        "representative_parent_ids",
        "representative_parent_smiles",
        "representative_raw_fragments",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in public_selected_rows:
            payload = dict(row)
            payload["representative_parent_ids"] = "|".join(payload["representative_parent_ids"])
            payload["representative_parent_smiles"] = "|".join(payload["representative_parent_smiles"])
            payload["representative_raw_fragments"] = "|".join(payload["representative_raw_fragments"])
            writer.writerow(payload)

    return {
        "selected_json": str(json_path),
        "selected_csv": str(csv_path),
        "summary_json": str(summary_path),
        "report_txt": str(report_path),
    }


def select_class_counterfactual_subgraphs(
    pool_jsonl: str | Path,
    *,
    out_dir: str | Path,
    config: SelectorConfig,
) -> dict[str, Any]:
    """Select a low-redundancy class-level fragment set from a candidate pool."""

    if config.sim_metric not in {"morgan", "embedding"}:
        raise ValueError(f"Unsupported sim_metric: {config.sim_metric!r}")
    if config.embedding_missing_policy not in {"error", "skip"}:
        raise ValueError(
            "embedding_missing_policy must be one of {'error', 'skip'}, "
            f"got {config.embedding_missing_policy!r}"
        )

    pool_path = Path(pool_jsonl).expanduser().resolve()
    rows = read_jsonl(pool_path)
    filtered_candidates, filter_counts, total_parent_count = _build_selector_candidates(rows, config)
    filtered_parent_count = len({candidate.parent_key for candidate in filtered_candidates})
    aggregates = _aggregate_fragments(
        filtered_candidates,
        total_parent_count=max(1, total_parent_count),
        config=config,
    )
    selected_rows = _select_fragments(
        aggregates,
        total_parent_count=max(1, total_parent_count),
        config=config,
    )

    selected_fragments = [row["fragment"] for row in selected_rows]
    selected_mean_tanimoto, selected_max_tanimoto = _selected_pairwise_similarity(
        selected_fragments,
        "morgan",
    )
    embedding_stats = (
        _selected_pairwise_embedding_similarity(selected_rows)
        if config.sim_metric == "embedding"
        else {
            "selected_pairwise_embedding_cosine_mean": None,
            "selected_pairwise_embedding_cosine_max": None,
            "selected_pairwise_embedding_cosine_min": None,
            "selected_pairwise_embedding_cosine_std": None,
        }
    )
    summary = {
        "metadata": {
            "pool_jsonl": str(pool_path),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "label": int(config.label),
            "top_k": int(config.top_k),
            "alpha_cf": float(config.alpha_cf),
            "beta_coverage": float(config.beta_coverage),
            "gamma_redundancy": float(config.gamma_redundancy),
            "eta_size": float(config.eta_size),
            "min_cf_drop": float(config.min_cf_drop),
            "require_cf_flip": bool(config.require_cf_flip),
            "require_final_substructure": bool(config.require_final_substructure),
            "max_projection_used_rate": float(config.max_projection_used_rate),
            "sim_metric": str(config.sim_metric),
            "embedding_field": str(config.embedding_field),
            "embedding_missing_policy": str(config.embedding_missing_policy),
            "top_candidates_per_fragment": int(config.top_candidates_per_fragment),
            "dedup_by_final_fragment": bool(config.dedup_by_final_fragment),
            "rdkit_available": bool(DataStructs is not None and _get_morgan_fp_generator() is not None),
        },
        "sim_metric": str(config.sim_metric),
        "embedding_field": str(config.embedding_field),
        "embedding_missing_policy": str(config.embedding_missing_policy),
        "input_candidate_count": len(rows),
        "valid_candidate_count_after_filter": len(filtered_candidates),
        "unique_fragment_count_after_filter": len(aggregates),
        "selected_count": len(selected_rows),
        "total_parent_count": int(total_parent_count),
        "filtered_parent_count": int(filtered_parent_count),
        "final_cumulative_coverage": (
            float(selected_rows[-1]["cumulative_coverage"]) if selected_rows else 0.0
        ),
        "selected_mean_cf_drop": _safe_mean([row["mean_cf_drop"] for row in selected_rows]),
        "selected_cf_flip_rate": _safe_mean([row["cf_flip_rate"] for row in selected_rows]),
        "selected_mean_atom_ratio": _safe_mean(
            [
                float(row["mean_atom_ratio"])
                for row in selected_rows
                if row["mean_atom_ratio"] is not None
            ]
        ),
        "selected_pairwise_tanimoto_mean": selected_mean_tanimoto,
        "selected_pairwise_tanimoto_max": selected_max_tanimoto,
        "selected_pairwise_embedding_cosine_mean": embedding_stats["selected_pairwise_embedding_cosine_mean"],
        "selected_pairwise_embedding_cosine_max": embedding_stats["selected_pairwise_embedding_cosine_max"],
        "selected_pairwise_embedding_cosine_min": embedding_stats["selected_pairwise_embedding_cosine_min"],
        "selected_pairwise_embedding_cosine_std": embedding_stats["selected_pairwise_embedding_cosine_std"],
        "selected_fragments": selected_fragments,
        "filter_counts": filter_counts,
    }
    outputs = write_selector_outputs(
        out_dir=out_dir,
        selected_rows=selected_rows,
        summary=summary,
    )
    return {
        "summary": summary,
        "selected_rows": [
            {key: value for key, value in row.items() if key != "representative_embedding"}
            for row in selected_rows
        ],
        "outputs": outputs,
    }


__all__ = [
    "SelectorConfig",
    "render_selector_report",
    "select_class_counterfactual_subgraphs",
    "write_selector_outputs",
]
