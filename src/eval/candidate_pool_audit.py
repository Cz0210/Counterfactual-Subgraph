"""Candidate-pool audit helpers for PPO-generated counterfactual fragments."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any

from src.chem import is_connected_fragment, is_parent_substructure, parse_smiles
from src.utils.io import ensure_directory, read_jsonl

try:  # pragma: no cover - depends on runtime env
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
except ImportError:  # pragma: no cover - depends on runtime env
    DataStructs = None
    AllChem = None


ATOM_RATIO_BUCKETS = (
    ("0-0.05", 0.0, 0.05),
    ("0.05-0.1", 0.05, 0.1),
    ("0.1-0.2", 0.1, 0.2),
    ("0.2-0.4", 0.2, 0.4),
    ("0.4-0.6", 0.4, 0.6),
    ("0.6-0.8", 0.6, 0.8),
    ("0.8-1.0", 0.8, 1.0000001),
)


@dataclass(frozen=True, slots=True)
class AuditConfig:
    """Execution knobs for candidate pool analysis."""

    group_by_label: bool = False
    sim_sample_size: int = 5000
    topk_show: int = 10


@dataclass(frozen=True, slots=True)
class NormalizedCandidateRow:
    """One candidate pool row normalized into the audit contract."""

    record_index: int
    parent_smiles: str | None
    label: int | None
    raw_fragment: str | None
    core_fragment: str | None
    projected_fragment: str | None
    final_fragment: str | None
    valid: bool
    parse_ok: bool
    connected: bool
    direct_substructure: bool
    final_substructure: bool
    projection_attempted: bool
    projection_success: bool
    projection_used: bool
    projection_identity: bool
    projection_retrieval: bool
    core_unusable: bool
    full_parent: bool
    near_parent: bool
    too_small: bool
    oracle_ok: bool
    cf_flip: bool
    cf_drop: float | None
    p_before: float | None
    p_after: float | None
    atom_count: int | None
    atom_ratio: float | None
    failure_tag: str | None
    invalid_detail: str | None
    raw_payload: dict[str, Any]


def _coalesce(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
    if lowered in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _as_int(value: Any) -> int | None:
    numeric = _as_float(value)
    if numeric is None:
        return None
    try:
        return int(numeric)
    except (TypeError, ValueError):
        return None


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
    numeric = _as_float(value)
    if numeric is not None:
        return bool(numeric)
    return None


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _safe_mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _safe_median(values: list[float]) -> float | None:
    return median(values) if values else None


def _safe_quantile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@lru_cache(maxsize=4096)
def _parse_fragment(smiles: str) -> tuple[bool, bool, str | None, int | None]:
    parsed = parse_smiles(smiles, sanitize=True, canonicalize=True)
    return (
        bool(parsed.parseable),
        bool(parsed.sanitized),
        parsed.canonical_smiles,
        int(parsed.atom_count) if parsed.atom_count is not None else None,
    )


@lru_cache(maxsize=4096)
def _parse_fragment_unsanitized(smiles: str) -> tuple[bool, str | None, int | None]:
    parsed = parse_smiles(smiles, sanitize=False, canonicalize=True)
    return (
        bool(parsed.parseable),
        parsed.canonical_smiles,
        int(parsed.atom_count) if parsed.atom_count is not None else None,
    )


@lru_cache(maxsize=4096)
def _canonical_fragment_key(smiles: str | None) -> str | None:
    normalized = _normalize_text(smiles)
    if normalized is None:
        return None
    parse_ok, sanitized, canonical, _atom_count = _parse_fragment(normalized)
    if parse_ok and sanitized and canonical:
        return canonical
    parse_ok_raw, canonical_raw, _atom_count_raw = _parse_fragment_unsanitized(normalized)
    if parse_ok_raw and canonical_raw:
        return canonical_raw
    return normalized


@lru_cache(maxsize=4096)
def _parent_atom_count(parent_smiles: str) -> int | None:
    _parse_ok, _sanitized, _canonical, atom_count = _parse_fragment(parent_smiles)
    return atom_count


def _infer_valid(row: dict[str, Any], final_fragment: str | None) -> bool:
    explicit = _as_bool(_coalesce(row, "valid", "valid_smiles", "sanitize_ok"))
    if explicit is not None:
        return explicit
    if final_fragment is None:
        return False
    parse_ok, sanitized, _canonical, _atom_count = _parse_fragment(final_fragment)
    return bool(parse_ok and sanitized)


def _infer_parse_ok(
    row: dict[str, Any],
    raw_fragment: str | None,
    final_fragment: str | None,
) -> bool:
    explicit = _as_bool(_coalesce(row, "parse_ok", "raw_parse_ok", "core_parse_ok"))
    if explicit is not None:
        return explicit
    candidate = raw_fragment or final_fragment
    if candidate is None:
        return False
    parse_ok, _canonical, _atom_count = _parse_fragment_unsanitized(candidate)
    return bool(parse_ok)


def _infer_connected(row: dict[str, Any], final_fragment: str | None) -> bool:
    explicit = _as_bool(_coalesce(row, "connected_ok", "connected_fragment", "connected"))
    if explicit is not None:
        return explicit
    if final_fragment is None:
        return False
    try:
        return bool(is_connected_fragment(final_fragment))
    except Exception:
        return False


def _infer_direct_substructure(
    row: dict[str, Any],
    parent_smiles: str | None,
    core_fragment: str | None,
) -> bool:
    explicit = _as_bool(
        _coalesce(
            row,
            "direct_substructure",
            "direct_substructure_success",
        )
    )
    if explicit is not None:
        return explicit
    if parent_smiles and core_fragment:
        try:
            return bool(is_parent_substructure(parent_smiles, core_fragment))
        except Exception:
            return False
    fallback = _as_bool(_coalesce(row, "is_substructure", "substructure", "substructure_ok"))
    return bool(fallback) if fallback is not None else False


def _infer_final_substructure(
    row: dict[str, Any],
    parent_smiles: str | None,
    final_fragment: str | None,
    direct_substructure: bool,
    projection_success: bool,
    projection_used: bool,
) -> bool:
    explicit = _as_bool(_coalesce(row, "final_substructure"))
    if explicit is not None:
        return explicit
    if parent_smiles and final_fragment:
        try:
            return bool(is_parent_substructure(parent_smiles, final_fragment))
        except Exception:
            return False
    if direct_substructure:
        return True
    return bool(
        projection_used
        or projection_success
        or _as_bool(_coalesce(row, "is_substructure", "substructure", "substructure_ok"))
    )


def _resolve_final_fragment(
    row: dict[str, Any],
    raw_fragment: str | None,
    core_fragment: str | None,
    projected_fragment: str | None,
) -> str | None:
    explicit_final = _normalize_text(_coalesce(row, "final_fragment", "final_fragment_smiles"))
    if explicit_final:
        return explicit_final

    projection_used = bool(_as_bool(_coalesce(row, "used_projected_subgraph_for_reward")))
    projection_success = bool(_as_bool(_coalesce(row, "projection_success")))
    if (projection_used or projection_success) and projected_fragment:
        return projected_fragment
    if core_fragment:
        return core_fragment
    if raw_fragment:
        return raw_fragment
    if projected_fragment:
        return projected_fragment
    return None


def _infer_atom_count(
    row: dict[str, Any],
    final_fragment: str | None,
) -> int | None:
    explicit = _as_int(
        _coalesce(
            row,
            "atom_count",
            "final_fragment_atom_count",
            "fragment_atom_count",
            "projection_atom_count",
            "core_atom_count",
        )
    )
    if explicit is not None:
        return explicit
    if final_fragment is None:
        return None
    _parse_ok, _sanitized, _canonical, atom_count = _parse_fragment(final_fragment)
    return atom_count


def _infer_atom_ratio(
    row: dict[str, Any],
    parent_smiles: str | None,
    atom_count: int | None,
) -> float | None:
    explicit = _as_float(
        _coalesce(
            row,
            "atom_ratio",
            "final_fragment_atom_ratio",
            "projection_atom_ratio",
        )
    )
    if explicit is not None:
        return explicit
    if parent_smiles is None or atom_count is None:
        return None
    parent_atoms = _parent_atom_count(parent_smiles)
    if parent_atoms is None or parent_atoms <= 0:
        return None
    return float(atom_count) / float(parent_atoms)


def _infer_core_unusable(row: dict[str, Any]) -> bool:
    if bool(_as_bool(_coalesce(row, "core_unusable"))):
        return True
    failure_tag = str(_coalesce(row, "failure_tag") or "")
    invalid_detail = str(_coalesce(row, "invalid_detail") or "")
    parse_failed_reason = str(_coalesce(row, "parse_failed_reason") or "")
    joined = " ".join((failure_tag, invalid_detail, parse_failed_reason)).lower()
    return "core_unusable" in joined


def _infer_full_parent(row: dict[str, Any], atom_ratio: float | None) -> bool:
    if bool(_as_bool(_coalesce(row, "full_parent"))):
        return True
    failure_tag = str(_coalesce(row, "failure_tag") or "")
    if failure_tag == "full_parent_fragment":
        return True
    return bool(atom_ratio is not None and atom_ratio >= 0.999999)


def _infer_near_parent(row: dict[str, Any], atom_ratio: float | None) -> bool:
    if bool(_as_bool(_coalesce(row, "near_parent_hard_fail"))):
        return True
    failure_tag = str(_coalesce(row, "failure_tag") or "")
    invalid_detail = str(_coalesce(row, "invalid_detail") or "")
    joined = " ".join((failure_tag, invalid_detail)).lower()
    if "near_parent" in joined:
        return True
    return bool(atom_ratio is not None and atom_ratio >= 0.85 and atom_ratio < 0.999999)


def _infer_too_small(row: dict[str, Any], atom_count: int | None) -> bool:
    if bool(_as_bool(_coalesce(row, "tiny_fragment_hard_fail", "too_small"))):
        return True
    failure_tag = str(_coalesce(row, "failure_tag") or "")
    invalid_detail = str(_coalesce(row, "invalid_detail") or "")
    joined = " ".join((failure_tag, invalid_detail)).lower()
    if "too_small" in joined or "tiny_fragment_hard_fail" in joined:
        return True
    return False


def _normalize_row(row: dict[str, Any], record_index: int) -> NormalizedCandidateRow:
    parent_smiles = _normalize_text(_coalesce(row, "parent_smiles", "smiles"))
    label = _as_int(_coalesce(row, "original_label", "label"))

    raw_fragment = _normalize_text(
        _coalesce(
            row,
            "raw_fragment",
            "fragment",
            "fragment_smiles",
            "generated_fragment",
            "raw_output",
        )
    )
    core_fragment = _normalize_text(_coalesce(row, "core_fragment", "core_fragment_smiles"))
    projected_fragment = _normalize_text(
        _coalesce(
            row,
            "projected_fragment",
            "projected_fragment_smiles",
            "nearest_parent_subgraph_smiles",
        )
    )
    final_fragment = _resolve_final_fragment(row, raw_fragment, core_fragment, projected_fragment)

    projection_attempted = bool(_as_bool(_coalesce(row, "projection_attempted")))
    projection_success = bool(_as_bool(_coalesce(row, "projection_success")))
    explicit_projection_used = _as_bool(_coalesce(row, "used_projected_subgraph_for_reward"))
    projection_used = bool(explicit_projection_used) if explicit_projection_used is not None else bool(
        projection_success and projected_fragment and final_fragment == projected_fragment
    )

    direct_substructure = _infer_direct_substructure(row, parent_smiles, core_fragment)
    final_substructure = _infer_final_substructure(
        row=row,
        parent_smiles=parent_smiles,
        final_fragment=final_fragment,
        direct_substructure=direct_substructure,
        projection_success=projection_success,
        projection_used=projection_used,
    )

    valid = _infer_valid(row, final_fragment)
    parse_ok = _infer_parse_ok(row, raw_fragment, final_fragment)
    connected = _infer_connected(row, final_fragment)
    atom_count = _infer_atom_count(row, final_fragment)
    atom_ratio = _infer_atom_ratio(row, parent_smiles, atom_count)

    base_fragment_key = _canonical_fragment_key(core_fragment or raw_fragment)
    projected_fragment_key = _canonical_fragment_key(projected_fragment)
    projection_identity = bool(
        projection_success
        and projected_fragment_key is not None
        and base_fragment_key is not None
        and projected_fragment_key == base_fragment_key
    )
    projection_retrieval = bool(projection_success and not projection_identity)

    return NormalizedCandidateRow(
        record_index=record_index,
        parent_smiles=parent_smiles,
        label=label,
        raw_fragment=raw_fragment,
        core_fragment=core_fragment,
        projected_fragment=projected_fragment,
        final_fragment=final_fragment,
        valid=valid,
        parse_ok=parse_ok,
        connected=connected,
        direct_substructure=direct_substructure,
        final_substructure=final_substructure,
        projection_attempted=projection_attempted,
        projection_success=projection_success,
        projection_used=projection_used,
        projection_identity=projection_identity,
        projection_retrieval=projection_retrieval,
        core_unusable=_infer_core_unusable(row),
        full_parent=_infer_full_parent(row, atom_ratio),
        near_parent=_infer_near_parent(row, atom_ratio),
        too_small=_infer_too_small(row, atom_count),
        oracle_ok=bool(_as_bool(_coalesce(row, "oracle_ok"))),
        cf_flip=bool(_as_bool(_coalesce(row, "cf_flip", "counterfactual_flip"))),
        cf_drop=_as_float(_coalesce(row, "cf_drop", "counterfactual_drop", "teacher_cf_drop")),
        p_before=_as_float(_coalesce(row, "p_before", "teacher_p_before")),
        p_after=_as_float(_coalesce(row, "p_after", "teacher_p_after")),
        atom_count=atom_count,
        atom_ratio=atom_ratio,
        failure_tag=_normalize_text(_coalesce(row, "failure_tag")),
        invalid_detail=_normalize_text(_coalesce(row, "invalid_detail")),
        raw_payload=row,
    )


def _bucketize_atom_ratio(row: NormalizedCandidateRow) -> str | None:
    if row.full_parent:
        return "full-parent"
    if row.atom_ratio is None:
        return None
    ratio = float(row.atom_ratio)
    for label, low, high in ATOM_RATIO_BUCKETS:
        if ratio >= low and ratio < high:
            return label
    return None


def _top_fragment_rows(counter: Counter[str], total: int, topk_show: int) -> list[dict[str, Any]]:
    top_rows: list[dict[str, Any]] = []
    for fragment, count in counter.most_common(topk_show):
        top_rows.append(
            {
                "fragment": fragment,
                "count": int(count),
                "ratio": _safe_rate(int(count), total),
            }
        )
    return top_rows


def _build_similarity_stats(
    fragment_keys: list[str],
    sim_sample_size: int,
) -> dict[str, Any]:
    unique_fragment_keys = sorted(set(fragment_keys))
    result: dict[str, Any] = {
        "rdkit_available": bool(DataStructs is not None and AllChem is not None),
        "similarity_pool_basis": "unique_final_fragments",
        "similarity_fragment_count": len(unique_fragment_keys),
        "skipped_similarity_count": 0,
        "similarity_pair_count": 0,
        "similarity_pairs_evaluated": 0,
        "mean_pairwise_tanimoto": None,
        "median_pairwise_tanimoto": None,
    }
    if DataStructs is None or AllChem is None:
        result["skipped_similarity_count"] = len(unique_fragment_keys)
        return result

    fingerprints: list[Any] = []
    skipped = 0
    for fragment in unique_fragment_keys:
        parse_ok, sanitized, _canonical, _atom_count = _parse_fragment(fragment)
        if not parse_ok or not sanitized:
            skipped += 1
            continue
        mol = parse_smiles(fragment, sanitize=True, canonicalize=True).mol
        if mol is None:
            skipped += 1
            continue
        fingerprints.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    result["skipped_similarity_count"] = skipped
    if len(fingerprints) < 2:
        return result

    all_pairs = [
        (left_index, right_index)
        for left_index in range(len(fingerprints))
        for right_index in range(left_index + 1, len(fingerprints))
    ]
    result["similarity_pair_count"] = len(all_pairs)
    if sim_sample_size > 0 and len(all_pairs) > sim_sample_size:
        rng = random.Random(13)
        sampled_pairs = rng.sample(all_pairs, sim_sample_size)
    else:
        sampled_pairs = all_pairs
    result["similarity_pairs_evaluated"] = len(sampled_pairs)

    similarities = [
        float(DataStructs.TanimotoSimilarity(fingerprints[left], fingerprints[right]))
        for left, right in sampled_pairs
    ]
    result["mean_pairwise_tanimoto"] = _safe_mean(similarities)
    result["median_pairwise_tanimoto"] = _safe_median(similarities)
    return result


def _compute_summary(rows: list[NormalizedCandidateRow], config: AuditConfig) -> dict[str, Any]:
    total = len(rows)
    parents = sorted({row.parent_smiles for row in rows if row.parent_smiles})
    label_counter = Counter(
        str(row.label) for row in rows if row.label is not None
    )

    cf_drop_values = [row.cf_drop for row in rows if row.cf_drop is not None]
    p_before_values = [row.p_before for row in rows if row.p_before is not None]
    p_after_values = [row.p_after for row in rows if row.p_after is not None]
    atom_count_values = [float(row.atom_count) for row in rows if row.atom_count is not None]
    atom_ratio_values = [float(row.atom_ratio) for row in rows if row.atom_ratio is not None]

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

    atom_ratio_histogram = OrderedDict((label, 0) for label, *_rest in ATOM_RATIO_BUCKETS)
    atom_ratio_histogram["full-parent"] = 0
    atom_ratio_missing_count = 0
    for row in rows:
        bucket = _bucketize_atom_ratio(row)
        if bucket is None:
            atom_ratio_missing_count += 1
            continue
        atom_ratio_histogram[bucket] += 1

    similarity_stats = _build_similarity_stats(
        fragment_keys=list(final_counter.keys()),
        sim_sample_size=config.sim_sample_size,
    )

    projection_failed_count = sum(
        1 for row in rows if row.projection_attempted and not row.projection_success
    )
    top5_count = sum(count for _fragment, count in final_counter.most_common(5))
    top10_count = sum(count for _fragment, count in final_counter.most_common(10))
    top1_count = final_counter.most_common(1)[0][1] if final_counter else 0

    summary = {
        "num_total": total,
        "num_by_label": {label: int(count) for label, count in sorted(label_counter.items())},
        "num_unique_parent": len(parents),
        "avg_candidates_per_parent": (
            float(total) / float(len(parents)) if parents else 0.0
        ),
        "valid_rate": _safe_rate(sum(1 for row in rows if row.valid), total),
        "parse_ok_rate": _safe_rate(sum(1 for row in rows if row.parse_ok), total),
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
        "projection_identity_rate": _safe_rate(
            sum(1 for row in rows if row.projection_identity),
            total,
        ),
        "projection_retrieval_rate": _safe_rate(
            sum(1 for row in rows if row.projection_retrieval),
            total,
        ),
        "projection_failed_rate": _safe_rate(projection_failed_count, total),
        "core_unusable_rate": _safe_rate(
            sum(1 for row in rows if row.core_unusable),
            total,
        ),
        "full_parent_rate": _safe_rate(sum(1 for row in rows if row.full_parent), total),
        "near_parent_rate": _safe_rate(sum(1 for row in rows if row.near_parent), total),
        "too_small_rate": _safe_rate(sum(1 for row in rows if row.too_small), total),
        "oracle_ok_rate": _safe_rate(sum(1 for row in rows if row.oracle_ok), total),
        "cf_flip_rate": _safe_rate(sum(1 for row in rows if row.cf_flip), total),
        "cf_drop_mean": _safe_mean(cf_drop_values),
        "cf_drop_median": _safe_median(cf_drop_values),
        "cf_drop_p25": _safe_quantile(cf_drop_values, 0.25),
        "cf_drop_p75": _safe_quantile(cf_drop_values, 0.75),
        "p_before_mean": _safe_mean(p_before_values),
        "p_after_mean": _safe_mean(p_after_values),
        "atom_count_mean": _safe_mean(atom_count_values),
        "atom_count_median": _safe_median(atom_count_values),
        "atom_ratio_mean": _safe_mean(atom_ratio_values),
        "atom_ratio_median": _safe_median(atom_ratio_values),
        "atom_ratio_histogram": {
            label: {
                "count": int(count),
                "rate": _safe_rate(int(count), total),
            }
            for label, count in atom_ratio_histogram.items()
        },
        "atom_ratio_missing_count": atom_ratio_missing_count,
        "unique_raw_fragment_count": len(raw_counter),
        "unique_core_fragment_count": len(core_counter),
        "unique_projected_fragment_count": len(projected_counter),
        "unique_final_fragment_count": len(final_counter),
        "unique_final_fragment_rate": _safe_rate(len(final_counter), total),
        "top1_final_fragment_ratio": _safe_rate(top1_count, total),
        "top5_final_fragment_ratio": _safe_rate(top5_count, total),
        "top10_final_fragment_ratio": _safe_rate(top10_count, total),
        "top_final_fragments": _top_fragment_rows(final_counter, total, config.topk_show),
        **similarity_stats,
    }
    return summary


def _build_judgment(summary: dict[str, Any]) -> dict[str, Any]:
    checks = OrderedDict(
        (
            ("final_substructure_rate>=0.9", (summary.get("final_substructure_rate") or 0.0) >= 0.9),
            ("cf_flip_rate>=0.5", (summary.get("cf_flip_rate") or 0.0) >= 0.5),
            ("unique_final_fragment_rate>=0.3", (summary.get("unique_final_fragment_rate") or 0.0) >= 0.3),
            ("top5_final_fragment_ratio<=0.5", (summary.get("top5_final_fragment_ratio") or 1.0) <= 0.5),
            (
                "mean_pairwise_tanimoto<=0.7",
                summary.get("mean_pairwise_tanimoto") is not None
                and float(summary.get("mean_pairwise_tanimoto") or 0.0) <= 0.7,
            ),
            ("projection_used_rate<=0.4", (summary.get("projection_used_rate") or 1.0) <= 0.4),
            (
                "0.15<=atom_ratio_mean<=0.55",
                summary.get("atom_ratio_mean") is not None
                and 0.15 <= float(summary.get("atom_ratio_mean") or 0.0) <= 0.55,
            ),
        )
    )
    passed_count = sum(1 for passed in checks.values() if passed)
    mode_collapse = (
        (summary.get("top5_final_fragment_ratio") or 0.0) > 0.5
        or (summary.get("unique_final_fragment_rate") or 1.0) < 0.3
        or (
            summary.get("mean_pairwise_tanimoto") is not None
            and float(summary.get("mean_pairwise_tanimoto") or 0.0) > 0.7
        )
    )
    projection_dependency_high = (summary.get("projection_used_rate") or 0.0) > 0.4
    atom_ratio_mean = summary.get("atom_ratio_mean")
    atom_ratio_out_of_range = (
        atom_ratio_mean is not None
        and not (0.15 <= float(atom_ratio_mean) <= 0.55)
    )
    selector_ready = passed_count >= 5 and (summary.get("final_substructure_rate") or 0.0) >= 0.9
    strong_cf_but_low_diversity = (
        (summary.get("cf_flip_rate") or 0.0) >= 0.5
        and (
            (summary.get("unique_final_fragment_rate") or 1.0) < 0.3
            or (summary.get("top5_final_fragment_ratio") or 0.0) > 0.5
        )
    )
    continue_long_ppo = not selector_ready and not strong_cf_but_low_diversity and (
        (summary.get("cf_flip_rate") or 0.0) < 0.5
        or (summary.get("final_substructure_rate") or 0.0) < 0.9
    )
    return {
        "heuristic_checks": checks,
        "passed_check_count": passed_count,
        "selector_ready": selector_ready,
        "mode_collapse_risk": mode_collapse,
        "projection_dependency_high": projection_dependency_high,
        "strong_cf_but_low_diversity": strong_cf_but_low_diversity,
        "atom_ratio_out_of_range": atom_ratio_out_of_range,
        "recommend_continue_long_ppo": continue_long_ppo,
        "recommend_sampling_tuning": strong_cf_but_low_diversity or mode_collapse,
        "recommend_start_selector": selector_ready,
    }


def audit_candidate_pool(
    pool_jsonl: str | Path,
    *,
    config: AuditConfig | None = None,
) -> dict[str, Any]:
    """Read a candidate pool JSONL file and compute selector-facing metrics."""

    resolved_config = config or AuditConfig()
    pool_path = Path(pool_jsonl).expanduser().resolve()
    rows = read_jsonl(pool_path)
    normalized_rows = [
        _normalize_row(row, record_index=index)
        for index, row in enumerate(rows)
    ]
    overall = _compute_summary(normalized_rows, resolved_config)
    judgment = _build_judgment(overall)

    by_label: dict[str, Any] = {}
    if resolved_config.group_by_label:
        for label in sorted({row.label for row in normalized_rows if row.label is not None}):
            if label is None:
                continue
            subset = [row for row in normalized_rows if row.label == label]
            by_label[str(label)] = _compute_summary(subset, resolved_config)

    return {
        "metadata": {
            "pool_jsonl": str(pool_path),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "group_by_label": bool(resolved_config.group_by_label),
            "sim_sample_size": int(resolved_config.sim_sample_size),
            "topk_show": int(resolved_config.topk_show),
            "rdkit_available": bool(DataStructs is not None and AllChem is not None),
        },
        "overall": overall,
        "judgment": judgment,
        "by_label": by_label,
    }


def render_audit_report(summary: dict[str, Any]) -> str:
    """Render a concise human-readable audit report."""

    overall = summary["overall"]
    judgment = summary["judgment"]
    lines = [
        "Candidate Pool Audit",
        f"pool_jsonl: {summary['metadata']['pool_jsonl']}",
        f"generated_at_utc: {summary['metadata']['generated_at_utc']}",
        "",
        "High-level judgment:",
        f"- suitable_for_selector: {'yes' if judgment['recommend_start_selector'] else 'not_yet'}",
        f"- mode_collapse_risk: {'high' if judgment['mode_collapse_risk'] else 'acceptable'}",
        f"- projection_dependency: {'high' if judgment['projection_dependency_high'] else 'acceptable'}",
        f"- strong_cf_but_low_diversity: {'yes' if judgment['strong_cf_but_low_diversity'] else 'no'}",
        f"- atom_ratio_scale: {'out_of_range' if judgment['atom_ratio_out_of_range'] else 'reasonable'}",
        f"- recommend_continue_long_ppo: {'yes' if judgment['recommend_continue_long_ppo'] else 'no'}",
        f"- recommend_sampling_tuning: {'yes' if judgment['recommend_sampling_tuning'] else 'no'}",
        "",
        "Core metrics:",
        f"- num_total: {overall['num_total']}",
        f"- num_by_label: {overall['num_by_label']}",
        f"- num_unique_parent: {overall['num_unique_parent']}",
        f"- avg_candidates_per_parent: {overall['avg_candidates_per_parent']:.4f}",
        f"- valid_rate: {overall['valid_rate']:.4f}",
        f"- parse_ok_rate: {overall['parse_ok_rate']:.4f}",
        f"- connected_rate: {overall['connected_rate']:.4f}",
        f"- direct_substructure_rate: {overall['direct_substructure_rate']:.4f}",
        f"- final_substructure_rate: {overall['final_substructure_rate']:.4f}",
        f"- projection_used_rate: {overall['projection_used_rate']:.4f}",
        f"- projection_identity_rate: {overall['projection_identity_rate']:.4f}",
        f"- projection_retrieval_rate: {overall['projection_retrieval_rate']:.4f}",
        f"- projection_failed_rate: {overall['projection_failed_rate']:.4f}",
        f"- oracle_ok_rate: {overall['oracle_ok_rate']:.4f}",
        f"- cf_flip_rate: {overall['cf_flip_rate']:.4f}",
        f"- cf_drop_mean: {overall['cf_drop_mean'] if overall['cf_drop_mean'] is not None else 'n/a'}",
        f"- cf_drop_median: {overall['cf_drop_median'] if overall['cf_drop_median'] is not None else 'n/a'}",
        f"- atom_ratio_mean: {overall['atom_ratio_mean'] if overall['atom_ratio_mean'] is not None else 'n/a'}",
        f"- atom_ratio_median: {overall['atom_ratio_median'] if overall['atom_ratio_median'] is not None else 'n/a'}",
        f"- unique_final_fragment_rate: {overall['unique_final_fragment_rate']:.4f}",
        f"- top5_final_fragment_ratio: {overall['top5_final_fragment_ratio']:.4f}",
        f"- mean_pairwise_tanimoto: {overall['mean_pairwise_tanimoto'] if overall['mean_pairwise_tanimoto'] is not None else 'n/a'}",
        f"- median_pairwise_tanimoto: {overall['median_pairwise_tanimoto'] if overall['median_pairwise_tanimoto'] is not None else 'n/a'}",
        "",
        "Heuristic checks:",
    ]
    for check_name, passed in judgment["heuristic_checks"].items():
        lines.append(f"- {check_name}: {'pass' if passed else 'fail'}")

    lines.append("")
    lines.append("Top final fragments:")
    top_fragments = overall.get("top_final_fragments") or []
    if not top_fragments:
        lines.append("- none")
    else:
        for item in top_fragments:
            lines.append(
                f"- {item['fragment']}: count={item['count']} ratio={item['ratio']:.4f}"
            )

    lines.append("")
    lines.append("Atom ratio histogram:")
    for bucket, payload in overall["atom_ratio_histogram"].items():
        lines.append(
            f"- {bucket}: count={payload['count']} rate={payload['rate']:.4f}"
        )
    lines.append(f"- atom_ratio_missing_count: {overall['atom_ratio_missing_count']}")

    if summary.get("by_label"):
        lines.append("")
        lines.append("By label:")
        for label, label_summary in summary["by_label"].items():
            lines.append(
                f"- label={label}: num_total={label_summary['num_total']} unique_parent={label_summary['num_unique_parent']} final_substructure_rate={label_summary['final_substructure_rate']:.4f} projection_used_rate={label_summary['projection_used_rate']:.4f} cf_flip_rate={label_summary['cf_flip_rate']:.4f} cf_drop_mean={label_summary['cf_drop_mean'] if label_summary['cf_drop_mean'] is not None else 'n/a'} atom_ratio_mean={label_summary['atom_ratio_mean'] if label_summary['atom_ratio_mean'] is not None else 'n/a'} unique_final_fragment_rate={label_summary['unique_final_fragment_rate']:.4f} top5_final_fragment_ratio={label_summary['top5_final_fragment_ratio']:.4f} mean_pairwise_tanimoto={label_summary['mean_pairwise_tanimoto'] if label_summary['mean_pairwise_tanimoto'] is not None else 'n/a'}"
            )
    return "\n".join(lines) + "\n"


def write_audit_outputs(
    summary: dict[str, Any],
    *,
    out_json: str | Path,
    out_txt: str | Path,
) -> None:
    """Persist both machine-readable and human-readable audit artifacts."""

    json_path = Path(out_json).expanduser().resolve()
    txt_path = Path(out_txt).expanduser().resolve()
    ensure_directory(json_path.parent)
    ensure_directory(txt_path.parent)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    txt_path.write_text(render_audit_report(summary), encoding="utf-8")


__all__ = [
    "AuditConfig",
    "audit_candidate_pool",
    "render_audit_report",
    "write_audit_outputs",
]
