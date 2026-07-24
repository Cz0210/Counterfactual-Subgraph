"""Pure aggregation and checkpoint selection for Mutagenicity generators."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


TASK_SELECTION_RULE = (
    "lexicographic(-strict_cf_flip_rate,-cf_drop_mean,"
    "-final_substructure_rate,-parse_ok_rate,"
    "atom_ratio_target_deviation,duplicate_rate,checkpoint)"
)


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validation_cohort_hash(parent_ids: Iterable[str]) -> str:
    normalized = "\n".join(sorted(str(value) for value in parent_ids))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _finite(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            result.append(parsed)
    return result


def _entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum(
        (count / total) * math.log(count / total)
        for count in counter.values()
        if count > 0
    )


def aggregate_generator_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    num_parents: int,
    num_candidates_per_parent: int,
) -> dict[str, Any]:
    """Aggregate one model/Hit@N view over the complete parent cohort."""

    if num_parents <= 0:
        raise ValueError("num_parents must be positive")
    by_parent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        parent_id = str(row.get("molecule_id") or row.get("parent_id") or "")
        if not parent_id:
            raise ValueError("Generator detail row is missing molecule_id")
        by_parent[parent_id].append(row)
    if len(by_parent) != int(num_parents):
        raise ValueError(
            f"Generator parent cohort mismatch: expected={num_parents} "
            f"actual={len(by_parent)}"
        )
    expected_rows = int(num_parents) * int(num_candidates_per_parent)
    if len(rows) != expected_rows:
        raise ValueError(
            f"Generator row count mismatch: expected={expected_rows} actual={len(rows)}"
        )
    bad_counts = {
        parent: len(parent_rows)
        for parent, parent_rows in by_parent.items()
        if len(parent_rows) != int(num_candidates_per_parent)
    }
    if bad_counts:
        raise ValueError(f"Per-parent candidate count mismatch: {dict(list(bad_counts.items())[:10])}")

    def rate(field: str) -> float:
        return sum(bool(row.get(field)) for row in rows) / len(rows)

    cf_drops = _finite(row.get("cf_drop") for row in rows)
    atom_ratios = _finite(row.get("atom_ratio") for row in rows)
    fragments = [
        str(row.get("final_fragment") or "").strip()
        for row in rows
        if str(row.get("final_fragment") or "").strip()
    ]
    fragment_counts = Counter(fragments)
    strict_fragments = Counter(
        str(row.get("final_fragment") or "").strip()
        for row in rows
        if bool(row.get("cf_flip"))
        and str(row.get("final_fragment") or "").strip()
    )
    parent_hits = {
        parent
        for parent, parent_rows in by_parent.items()
        if any(bool(row.get("cf_flip")) for row in parent_rows)
    }
    parent_conditioned_unique = [
        len(
            {
                str(row.get("final_fragment") or "").strip()
                for row in parent_rows
                if str(row.get("final_fragment") or "").strip()
            }
        )
        for parent_rows in by_parent.values()
    ]
    top_counts = sorted(fragment_counts.values(), reverse=True)
    strategies = {
        str(
            row.get("candidate_strategy")
            or row.get("strategy")
            or ("projected" if bool(row.get("projection_used")) else "direct_or_invalid")
        )
        for row in rows
    }
    target_atom_ratio = 0.35
    return {
        "num_parents": int(num_parents),
        "num_candidates_per_parent": int(num_candidates_per_parent),
        "num_candidate_rows": len(rows),
        "parse_ok_rate": rate("parse_ok"),
        "valid_rate": rate("valid"),
        "direct_substructure_rate": rate("direct_substructure"),
        "final_substructure_rate": rate("final_substructure"),
        "projection_used_rate": rate("projection_used"),
        "oracle_ok_rate": rate("oracle_ok"),
        "strict_cf_flip_rate": rate("cf_flip"),
        "candidate_level_strict_flip_rate": rate("cf_flip"),
        "parent_hit_rate": len(parent_hits) / num_parents,
        f"parent_hit_at_{num_candidates_per_parent}": len(parent_hits) / num_parents,
        "num_parents_hit": len(parent_hits),
        "cf_drop_mean": sum(cf_drops) / len(cf_drops) if cf_drops else 0.0,
        "cf_drop_median": median(cf_drops) if cf_drops else 0.0,
        "atom_ratio_mean": sum(atom_ratios) / len(atom_ratios) if atom_ratios else 0.0,
        "atom_ratio_median": median(atom_ratios) if atom_ratios else 0.0,
        "atom_ratio_target": target_atom_ratio,
        "atom_ratio_target_deviation": (
            abs(sum(atom_ratios) / len(atom_ratios) - target_atom_ratio)
            if atom_ratios
            else float("inf")
        ),
        "unique_generated_fragments": len(fragment_counts),
        "unique_strict_fragments": len(strict_fragments),
        "duplicate_rate": (
            1.0 - len(fragment_counts) / len(fragments) if fragments else 1.0
        ),
        "fragment_entropy": _entropy(fragment_counts),
        "top1_fragment_frequency": top_counts[0] / len(fragments) if top_counts else 0.0,
        "top10_fragment_frequency": (
            sum(top_counts[:10]) / len(fragments) if top_counts else 0.0
        ),
        "mean_parent_conditioned_unique_fragments": (
            sum(parent_conditioned_unique) / len(parent_conditioned_unique)
            if parent_conditioned_unique
            else 0.0
        ),
        "strategy_diversity": len(strategies),
        "core_unusable_rate": sum(
            "core_unusable" in str(row.get("failure_tag") or "")
            or "core_unusable" in str(row.get("invalid_detail") or "")
            for row in rows
        )
        / len(rows),
        "invalid_substructure_rate": 1.0 - rate("final_substructure"),
    }


def fragment_frequency_strict_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Cross fragment frequency with strict success and parent coverage potential."""

    model_parent_counts: Counter[str] = Counter()
    model_parents: dict[str, set[str]] = defaultdict(set)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        model = str(row.get("model_name") or "")
        parent = str(row.get("molecule_id") or row.get("parent_id") or "")
        fragment = str(row.get("final_fragment") or "").strip()
        if model and parent:
            model_parents[model].add(parent)
        if model and fragment:
            grouped[(model, fragment)].append(row)
    model_parent_counts.update(
        {model: len(parents) for model, parents in model_parents.items()}
    )
    output: list[dict[str, Any]] = []
    for (model, fragment), members in grouped.items():
        parents = {
            str(row.get("molecule_id") or row.get("parent_id") or "")
            for row in members
        }
        strict_rows = [row for row in members if bool(row.get("cf_flip"))]
        strict_parents = {
            str(row.get("molecule_id") or row.get("parent_id") or "")
            for row in strict_rows
        }
        denominator = int(model_parent_counts[model])
        output.append(
            {
                "model_name": model,
                "fragment": fragment,
                "candidate_frequency": len(members),
                "parent_frequency": len(parents),
                "strict_candidate_count": len(strict_rows),
                "strict_candidate_rate": len(strict_rows) / len(members),
                "strict_parent_coverage_potential": len(strict_parents),
                "strict_parent_coverage_potential_rate": (
                    len(strict_parents) / denominator if denominator else 0.0
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            str(row["model_name"]),
            -int(row["candidate_frequency"]),
            str(row["fragment"]),
        ),
    )


def task_checkpoint_sort_key(
    row: Mapping[str, Any],
    *,
    atom_ratio_target: float = 0.35,
) -> tuple[Any, ...]:
    """Lower key is better; never use token loss ahead of task metrics."""

    ratio = float(row.get("atom_ratio_mean", float("inf")))
    ratio_deviation = (
        abs(ratio - float(atom_ratio_target)) if math.isfinite(ratio) else float("inf")
    )
    return (
        -float(row.get("strict_cf_flip_rate", 0.0)),
        -float(row.get("cf_drop_mean", 0.0)),
        -float(row.get("final_substructure_rate", 0.0)),
        -float(row.get("parse_ok_rate", 0.0)),
        ratio_deviation,
        float(row.get("duplicate_rate", 1.0)),
        str(row.get("checkpoint") or row.get("model_path") or ""),
    )


def rank_checkpoints(
    rows: Sequence[Mapping[str, Any]],
    *,
    atom_ratio_target: float = 0.35,
) -> list[dict[str, Any]]:
    ranked = sorted(
        (dict(row) for row in rows),
        key=lambda row: task_checkpoint_sort_key(
            row, atom_ratio_target=atom_ratio_target
        ),
    )
    for index, row in enumerate(ranked, start=1):
        row["task_rank"] = index
        row["selection_rule"] = TASK_SELECTION_RULE
    return ranked


def best_task_checkpoint_payload(
    ranked_rows: Sequence[Mapping[str, Any]],
    *,
    cohort_hash: str,
    decoding_config_hash: str,
) -> dict[str, Any]:
    if not ranked_rows:
        raise ValueError("No checkpoint metrics were supplied")
    best = dict(ranked_rows[0])
    return {
        "checkpoint": best.get("checkpoint") or best.get("model_path"),
        "step": best.get("step"),
        "strict_cf_flip_rate": best.get("strict_cf_flip_rate"),
        "cf_drop_mean": best.get("cf_drop_mean"),
        "final_substructure_rate": best.get("final_substructure_rate"),
        "parse_ok_rate": best.get("parse_ok_rate"),
        "atom_ratio_mean": best.get("atom_ratio_mean"),
        "duplicate_rate": best.get("duplicate_rate"),
        "selection_rule": TASK_SELECTION_RULE,
        "validation_cohort_hash": cohort_hash,
        "decoding_config_hash": decoding_config_hash,
        "selection_split": "val",
        "calibration_or_test_used": False,
    }


def compare_parent_difficulty(
    sft_rows: Sequence[Mapping[str, Any]],
    ppo_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Classify Hit@1 parents for one SFT/PPO pair."""

    def one_per_parent(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            parent = str(row.get("molecule_id") or "")
            if not parent:
                raise ValueError("Difficulty row is missing molecule_id")
            if parent in result:
                raise ValueError("Difficulty analysis requires one candidate per parent")
            result[parent] = row
        return result

    sft = one_per_parent(sft_rows)
    ppo = one_per_parent(ppo_rows)
    if set(sft) != set(ppo):
        raise ValueError("SFT/PPO difficulty cohorts do not match")
    output: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for parent in sorted(sft):
        before = sft[parent]
        after = ppo[parent]
        sft_valid = bool(before.get("final_substructure"))
        ppo_valid = bool(after.get("final_substructure"))
        if not sft_valid and not ppo_valid:
            category = "invalid"
        elif bool(before.get("cf_flip")) and bool(after.get("cf_flip")):
            category = "easy"
        elif not bool(before.get("cf_flip")) and bool(after.get("cf_flip")):
            category = "improved"
        elif bool(before.get("cf_flip")) and not bool(after.get("cf_flip")):
            category = "regressed"
        else:
            category = "hard"
        counts[category] += 1
        output.append(
            {
                "parent_id": parent,
                "parent_smiles": before.get("parent_smiles"),
                "teacher_p1_before": before.get("prob_before_1"),
                "sft_fragment": before.get("final_fragment"),
                "sft_pred_after": before.get("pred_after"),
                "sft_cf_drop": before.get("cf_drop"),
                "sft_strict_flip": bool(before.get("cf_flip")),
                "sft_strategy": before.get("candidate_strategy")
                or before.get("strategy")
                or "generated",
                "ppo_fragment": after.get("final_fragment"),
                "ppo_pred_after": after.get("pred_after"),
                "ppo_cf_drop": after.get("cf_drop"),
                "ppo_strict_flip": bool(after.get("cf_flip")),
                "ppo_strategy": after.get("candidate_strategy")
                or after.get("strategy")
                or "generated",
                "projection_used": bool(after.get("projection_used")),
                "atom_ratio": after.get("atom_ratio"),
                "difficulty": category,
                "failure_reason": after.get("failure_tag") or after.get("invalid_detail"),
            }
        )
    return output, {
        "num_parents": len(output),
        "difficulty_counts": dict(sorted(counts.items())),
    }


def summarize_strategy_failures(
    difficulty_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize failure modes without changing or oversampling the cohort."""

    grouped: Counter[tuple[str, str, str, str]] = Counter()
    for row in difficulty_rows:
        failure_reason = str(row.get("failure_reason") or "none")
        projection = "projected" if bool(row.get("projection_used")) else "not_projected"
        for stage in ("sft", "ppo"):
            strategy = str(row.get(f"{stage}_strategy") or "unknown")
            strict = bool(row.get(f"{stage}_strict_flip"))
            outcome = "strict_flip" if strict else str(row.get("difficulty") or "unknown")
            grouped[(stage, strategy, outcome, f"{projection}:{failure_reason}")] += 1
    return [
        {
            "stage": stage,
            "strategy": strategy,
            "outcome": outcome,
            "projection_and_failure_reason": reason,
            "count": count,
        }
        for (stage, strategy, outcome, reason), count in sorted(grouped.items())
    ]


def checkpoint_step(path: str | Path) -> int | None:
    name = Path(path).name
    if name == "best_val_checkpoint":
        return None
    if not name.startswith("checkpoint-"):
        return None
    suffix = name.removeprefix("checkpoint-")
    return int(suffix) if suffix.isdigit() else None


__all__ = [
    "TASK_SELECTION_RULE",
    "aggregate_generator_rows",
    "best_task_checkpoint_payload",
    "checkpoint_step",
    "compare_parent_difficulty",
    "fragment_frequency_strict_summary",
    "rank_checkpoints",
    "summarize_strategy_failures",
    "stable_json_hash",
    "task_checkpoint_sort_key",
    "validation_cohort_hash",
]
