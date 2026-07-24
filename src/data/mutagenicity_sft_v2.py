"""Strict-first Mutagenicity SFT/PPO dataset construction and audits."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from src.chem import parse_smiles
from src.data.mutagenicity import write_csv, write_json
from src.data.mutagenicity_sft_ppo import (
    BASE_OUTPUT_FIELDS,
    MutagenicityParent,
    MutagenicitySFTPPOConfig,
    SOURCE_LABEL,
    TARGET_LABEL,
    load_teacher_consistent_parents,
    validate_source_isolation,
)
from src.data.prompts import build_counterfactual_prompt
from src.data.schemas import MoleculeRecord
from src.data.sft_column_compat import normalize_completion_text
from src.data.sft_v3_builder import (
    SFTV3BuilderConfig,
    SFTV3ReferenceCandidate,
    enumerate_reference_candidates_for_parent,
)
from src.rewards.counterfactual_oracle import CounterfactualTeacherScorer


TARGET_QUALITY_STRICT = "strict"
TARGET_QUALITY_FALLBACK = "soft_fallback"
_STRATEGY_ORDER = {
    strategy: index
    for index, strategy in enumerate(
        (
            "murcko_scaffold",
            "murcko_scaffold_r1",
            "ring_system_r0",
            "ring_system_r1",
            "fg_carboxyl",
            "fg_amide",
            "fg_sulfonic_acid",
            "fg_sulfonamide",
            "fg_azo",
            "fg_nitro",
            "fg_disulfide",
            "fg_halogen",
            "fg_aromatic_oh",
            "atom_k1",
            "atom_k2",
            "atom_k3",
            "bond_k1",
            "bond_k2",
            "brics_component",
        )
    )
}

V2_SFT_FIELDS = BASE_OUTPUT_FIELDS + (
    "prompt",
    "completion",
    "instruction",
    "output",
    "response",
    "reference_fragment",
    "raw_fragment",
    "core_fragment",
    "candidate_strategy",
    "selected_strategy",
    "atom_ratio",
    "selected_atom_ratio",
    "residual_smiles",
    "cf_drop",
    "selected_cf_drop",
    "cf_flip",
    "strict_flip",
    "oracle_ok",
    "pred_before",
    "pred_after",
    "p1_before",
    "p1_after",
    "target_quality",
    "strict_candidate_count",
    "fallback_reason",
    "completion_global_frequency",
    "inverse_frequency_weight",
    "candidate_rank_within_parent",
)
V2_PPO_FIELDS = BASE_OUTPUT_FIELDS + ("prompt",)
CANDIDATE_FIELDS = (
    "split",
    "parent_id",
    "parent_smiles",
    "candidate_index",
    "candidate_strategy",
    "core_fragment",
    "canonical_fragment",
    "raw_fragment",
    "oracle_ok",
    "pred_before",
    "pred_after",
    "p1_before",
    "p1_after",
    "cf_drop",
    "strict_flip",
    "positive_cf_drop",
    "atom_ratio",
    "residual_smiles",
    "completion_global_frequency",
    "ranking_score",
)


@dataclass(frozen=True, slots=True)
class ParentCandidateSet:
    parent: MutagenicityParent
    candidates: tuple[SFTV3ReferenceCandidate, ...]
    proposal_count: int
    drop_reason: str | None


@dataclass(frozen=True, slots=True)
class StrictFirstSelection:
    selected: SFTV3ReferenceCandidate | None
    target_quality: str | None
    strict_candidate_count: int
    fallback_reason: str | None


def canonicalize_fragment(fragment: str) -> str | None:
    parsed = parse_smiles(
        str(fragment or "").strip(),
        sanitize=True,
        canonicalize=True,
        allow_capped_fragments=False,
    )
    if not parsed.sanitized or parsed.mol is None or not parsed.canonical_smiles:
        return None
    return str(parsed.canonical_smiles)


def is_teacher_strict_candidate(candidate: SFTV3ReferenceCandidate) -> bool:
    return bool(
        candidate.oracle_ok
        and candidate.pred_before == SOURCE_LABEL
        and candidate.pred_after == TARGET_LABEL
    )


def is_positive_fallback_candidate(candidate: SFTV3ReferenceCandidate) -> bool:
    return bool(
        candidate.oracle_ok
        and candidate.pred_before == SOURCE_LABEL
        and candidate.cf_drop is not None
        and math.isfinite(float(candidate.cf_drop))
        and float(candidate.cf_drop) > 0.0
    )


def _candidate_sort_key(
    candidate: SFTV3ReferenceCandidate,
    *,
    completion_frequency: Mapping[str, int],
) -> tuple[Any, ...]:
    canonical = canonicalize_fragment(candidate.core_fragment) or candidate.core_fragment
    cf_drop = float(candidate.cf_drop) if candidate.cf_drop is not None else -math.inf
    return (
        -cf_drop,
        float(candidate.atom_ratio),
        int(completion_frequency.get(canonical, 0)),
        int(_STRATEGY_ORDER.get(candidate.candidate_strategy, 10_000)),
        canonical,
    )


def select_strict_first_candidate(
    candidates: Sequence[SFTV3ReferenceCandidate],
    *,
    completion_frequency: Mapping[str, int] | None = None,
    allow_soft_fallback: bool,
) -> StrictFirstSelection:
    """Apply the hard strict-first rule before any soft ranking."""

    frequencies = completion_frequency or {}
    strict = [candidate for candidate in candidates if is_teacher_strict_candidate(candidate)]
    if strict:
        selected = min(
            strict,
            key=lambda candidate: _candidate_sort_key(
                candidate,
                completion_frequency=frequencies,
            ),
        )
        return StrictFirstSelection(
            selected=selected,
            target_quality=TARGET_QUALITY_STRICT,
            strict_candidate_count=len(strict),
            fallback_reason=None,
        )

    fallback = [
        candidate
        for candidate in candidates
        if is_positive_fallback_candidate(candidate)
    ]
    if allow_soft_fallback and fallback:
        selected = min(
            fallback,
            key=lambda candidate: _candidate_sort_key(
                candidate,
                completion_frequency=frequencies,
            ),
        )
        return StrictFirstSelection(
            selected=selected,
            target_quality=TARGET_QUALITY_FALLBACK,
            strict_candidate_count=0,
            fallback_reason="no_strict_candidate_positive_cf_drop_fallback",
        )
    return StrictFirstSelection(
        selected=None,
        target_quality=None,
        strict_candidate_count=0,
        fallback_reason=(
            "strict_only_no_strict_candidate"
            if not allow_soft_fallback
            else "no_strict_or_positive_cf_drop_candidate"
        ),
    )


def build_teacher_scorer(path: str | Path) -> CounterfactualTeacherScorer:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"Mutagenicity RF teacher is missing: {resolved}")
    scorer = CounterfactualTeacherScorer(resolved, device="cpu")
    if not scorer.available:
        raise RuntimeError(
            "Mutagenicity RF teacher is unavailable: "
            f"{scorer.availability_reason}"
        )
    return scorer


def enumerate_split_candidates(
    parents: Sequence[MutagenicityParent],
    *,
    config: SFTV3BuilderConfig,
    scorer: CounterfactualTeacherScorer,
) -> list[ParentCandidateSet]:
    output: list[ParentCandidateSet] = []
    for parent in parents:
        enumeration = enumerate_reference_candidates_for_parent(
            parent.to_sft_v3_parent(),
            config=config,
            oracle_scorer=scorer,
        )
        output.append(
            ParentCandidateSet(
                parent=parent,
                candidates=enumeration.candidates,
                proposal_count=enumeration.candidate_count,
                drop_reason=enumeration.drop_reason,
            )
        )
    return output


def completion_frequency(
    candidate_sets: Sequence[ParentCandidateSet],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for candidate_set in candidate_sets:
        for candidate in candidate_set.candidates:
            canonical = canonicalize_fragment(candidate.core_fragment)
            if canonical:
                counts[canonical] += 1
    return counts


def _candidate_score_text(
    candidate: SFTV3ReferenceCandidate,
    frequencies: Mapping[str, int],
) -> str:
    return json.dumps(
        list(_candidate_sort_key(candidate, completion_frequency=frequencies)),
        ensure_ascii=False,
    )


def candidate_inventory_rows(
    candidate_sets: Sequence[ParentCandidateSet],
    *,
    split: str,
    frequencies: Mapping[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_set in candidate_sets:
        for index, candidate in enumerate(candidate_set.candidates):
            canonical = canonicalize_fragment(candidate.core_fragment) or ""
            strict = is_teacher_strict_candidate(candidate)
            rows.append(
                {
                    "split": split,
                    "parent_id": candidate_set.parent.molecule_id,
                    "parent_smiles": candidate_set.parent.parent_smiles,
                    "candidate_index": index,
                    "candidate_strategy": candidate.candidate_strategy,
                    "core_fragment": candidate.core_fragment,
                    "canonical_fragment": canonical,
                    "raw_fragment": candidate.raw_fragment,
                    "oracle_ok": candidate.oracle_ok,
                    "pred_before": candidate.pred_before,
                    "pred_after": candidate.pred_after,
                    "p1_before": candidate.p_before,
                    "p1_after": candidate.p_after,
                    "cf_drop": candidate.cf_drop,
                    "strict_flip": strict,
                    "positive_cf_drop": is_positive_fallback_candidate(candidate),
                    "atom_ratio": candidate.atom_ratio,
                    "residual_smiles": candidate.residual_smiles,
                    "completion_global_frequency": int(
                        frequencies.get(canonical, 0)
                    ),
                    "ranking_score": _candidate_score_text(candidate, frequencies),
                }
            )
    return rows


def _sft_row(
    parent: MutagenicityParent,
    candidate: SFTV3ReferenceCandidate,
    *,
    target_quality: str,
    strict_candidate_count: int,
    fallback_reason: str | None,
    frequencies: Mapping[str, int],
    rank: int = 1,
) -> dict[str, Any]:
    prompt = build_counterfactual_prompt(
        MoleculeRecord(
            record_id=parent.molecule_id,
            smiles=parent.parent_smiles,
            label=parent.label,
        ),
        include_label=False,
    )
    canonical = canonicalize_fragment(candidate.core_fragment) or candidate.core_fragment
    frequency = int(frequencies.get(canonical, 1))
    completion = normalize_completion_text(candidate.core_fragment)
    return {
        **parent.base_output_row(),
        "prompt": prompt,
        "completion": completion,
        "instruction": prompt,
        "output": candidate.core_fragment,
        "response": candidate.core_fragment,
        "reference_fragment": candidate.core_fragment,
        "raw_fragment": candidate.raw_fragment,
        "core_fragment": candidate.core_fragment,
        "candidate_strategy": candidate.candidate_strategy,
        "selected_strategy": candidate.candidate_strategy,
        "atom_ratio": candidate.atom_ratio,
        "selected_atom_ratio": candidate.atom_ratio,
        "residual_smiles": candidate.residual_smiles,
        "cf_drop": candidate.cf_drop,
        "selected_cf_drop": candidate.cf_drop,
        "cf_flip": is_teacher_strict_candidate(candidate),
        "strict_flip": is_teacher_strict_candidate(candidate),
        "oracle_ok": candidate.oracle_ok,
        "pred_before": candidate.pred_before,
        "pred_after": candidate.pred_after,
        "p1_before": candidate.p_before,
        "p1_after": candidate.p_after,
        "target_quality": target_quality,
        "strict_candidate_count": strict_candidate_count,
        "fallback_reason": fallback_reason or "",
        "completion_global_frequency": frequency,
        "inverse_frequency_weight": 1.0 / float(max(1, frequency)),
        "candidate_rank_within_parent": int(rank),
    }


def _ppo_row(parent: MutagenicityParent) -> dict[str, Any]:
    prompt = build_counterfactual_prompt(
        MoleculeRecord(
            record_id=parent.molecule_id,
            smiles=parent.parent_smiles,
            label=parent.label,
        ),
        include_label=True,
    )
    return {**parent.base_output_row(), "prompt": prompt}


def build_strict_multitarget_rows(
    candidate_sets: Sequence[ParentCandidateSet],
    *,
    frequencies: Mapping[str, int],
    max_targets_per_parent: int,
    max_completion_frequency: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_set in candidate_sets:
        deduplicated: dict[str, SFTV3ReferenceCandidate] = {}
        for candidate in candidate_set.candidates:
            if not is_teacher_strict_candidate(candidate):
                continue
            canonical = canonicalize_fragment(candidate.core_fragment)
            if not canonical:
                continue
            incumbent = deduplicated.get(canonical)
            if incumbent is None or _candidate_sort_key(
                candidate, completion_frequency=frequencies
            ) < _candidate_sort_key(incumbent, completion_frequency=frequencies):
                deduplicated[canonical] = candidate
        ordered = sorted(
            deduplicated.values(),
            key=lambda candidate: _candidate_sort_key(
                candidate, completion_frequency=frequencies
            ),
        )
        if max_completion_frequency is not None:
            ordered = [
                candidate
                for candidate in ordered
                if frequencies.get(
                    canonicalize_fragment(candidate.core_fragment) or "", 0
                )
                <= max_completion_frequency
            ]
        strict_count = len(deduplicated)
        for rank, candidate in enumerate(
            ordered[: max(1, int(max_targets_per_parent))],
            start=1,
        ):
            rows.append(
                _sft_row(
                    candidate_set.parent,
                    candidate,
                    target_quality=TARGET_QUALITY_STRICT,
                    strict_candidate_count=strict_count,
                    fallback_reason=None,
                    frequencies=frequencies,
                    rank=rank,
                )
            )
    return rows


def _gini(values: Sequence[int]) -> float:
    positive = sorted(int(value) for value in values if int(value) > 0)
    if not positive:
        return 0.0
    total = sum(positive)
    n = len(positive)
    weighted = sum((index + 1) * value for index, value in enumerate(positive))
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def _distribution_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completions = [str(row["core_fragment"]) for row in rows]
    counts = Counter(completions)
    total = len(completions)
    return {
        "rows": total,
        "unique_completions": len(counts),
        "duplicate_completion_rate": (
            1.0 - len(counts) / total if total else 0.0
        ),
        "completion_frequency_gini": _gini(list(counts.values())),
        "strategy_counts": dict(
            sorted(Counter(str(row["selected_strategy"]) for row in rows).items())
        ),
        "target_quality_counts": dict(
            sorted(Counter(str(row["target_quality"]) for row in rows).items())
        ),
    }


def _top_completion_rows(
    inventory: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_completion: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    selected_count = Counter(
        canonicalize_fragment(str(row.get("core_fragment") or ""))
        or str(row.get("core_fragment") or "")
        for row in selected_rows
    )
    for row in inventory:
        by_completion[str(row.get("canonical_fragment") or "")].append(row)
    output: list[dict[str, Any]] = []
    for completion, rows in by_completion.items():
        drops = [
            float(row["cf_drop"])
            for row in rows
            if row.get("cf_drop") not in (None, "")
        ]
        strict_count = sum(bool(row.get("strict_flip")) for row in rows)
        output.append(
            {
                "completion": completion,
                "candidate_frequency": len(rows),
                "selected_frequency": int(selected_count.get(completion, 0)),
                "strict_count": strict_count,
                "strict_rate": strict_count / len(rows) if rows else 0.0,
                "cf_drop_mean": mean(drops) if drops else None,
                "strategies": "|".join(
                    sorted({str(row["candidate_strategy"]) for row in rows})
                ),
            }
        )
    output.sort(
        key=lambda row: (
            -int(row["candidate_frequency"]),
            str(row["completion"]),
        )
    )
    return output[:100]


def build_mutagenicity_sft_ppo_v2(
    *,
    train_input: str | Path,
    val_input: str | Path,
    calibration_exclusion_input: str | Path,
    test_exclusion_input: str | Path,
    teacher_path: str | Path,
    output_dir: str | Path,
    config: MutagenicitySFTPPOConfig,
    expected_counts: Mapping[str, int | None],
    max_targets_per_parent: int = 3,
    max_completion_frequency: int | None = None,
) -> dict[str, Any]:
    out = Path(output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty v2 output: {out}")

    train = load_teacher_consistent_parents(
        train_input,
        expected_split="train",
        expected_count=expected_counts.get("train"),
    )
    val = load_teacher_consistent_parents(
        val_input,
        expected_split="val",
        expected_count=expected_counts.get("val"),
    )
    calibration = load_teacher_consistent_parents(
        calibration_exclusion_input,
        expected_split="calibration",
        expected_count=expected_counts.get("calibration"),
    )
    test = load_teacher_consistent_parents(
        test_exclusion_input,
        expected_split="test",
        expected_count=expected_counts.get("test"),
    )
    leakage = validate_source_isolation(train, val, calibration, test)
    scorer = build_teacher_scorer(teacher_path)
    sft_config = config.sft_v3_config(teacher_path)
    train_sets = enumerate_split_candidates(train, config=sft_config, scorer=scorer)
    val_sets = enumerate_split_candidates(val, config=sft_config, scorer=scorer)
    train_frequency = completion_frequency(train_sets)
    val_frequency = completion_frequency(val_sets)

    train_inventory = candidate_inventory_rows(
        train_sets, split="train", frequencies=train_frequency
    )
    val_inventory = candidate_inventory_rows(
        val_sets, split="val", frequencies=val_frequency
    )
    strict_train: list[dict[str, Any]] = []
    strict_val: list[dict[str, Any]] = []
    fallback_train: list[dict[str, Any]] = []
    fallback_val: list[dict[str, Any]] = []
    parent_candidate_rows: list[dict[str, Any]] = []
    for split, candidate_sets, frequencies, strict_rows, fallback_rows in (
        ("train", train_sets, train_frequency, strict_train, fallback_train),
        ("val", val_sets, val_frequency, strict_val, fallback_val),
    ):
        for candidate_set in candidate_sets:
            strict_selection = select_strict_first_candidate(
                candidate_set.candidates,
                completion_frequency=frequencies,
                allow_soft_fallback=False,
            )
            fallback_selection = select_strict_first_candidate(
                candidate_set.candidates,
                completion_frequency=frequencies,
                allow_soft_fallback=True,
            )
            if strict_selection.selected is not None:
                strict_rows.append(
                    _sft_row(
                        candidate_set.parent,
                        strict_selection.selected,
                        target_quality=TARGET_QUALITY_STRICT,
                        strict_candidate_count=strict_selection.strict_candidate_count,
                        fallback_reason=None,
                        frequencies=frequencies,
                    )
                )
            if fallback_selection.selected is not None:
                fallback_rows.append(
                    _sft_row(
                        candidate_set.parent,
                        fallback_selection.selected,
                        target_quality=str(fallback_selection.target_quality),
                        strict_candidate_count=fallback_selection.strict_candidate_count,
                        fallback_reason=fallback_selection.fallback_reason,
                        frequencies=frequencies,
                    )
                )
            parent_candidate_rows.append(
                {
                    "split": split,
                    "parent_id": candidate_set.parent.molecule_id,
                    "parent_smiles": candidate_set.parent.parent_smiles,
                    "proposal_count": candidate_set.proposal_count,
                    "valid_candidate_count": len(candidate_set.candidates),
                    "oracle_ok_candidate_count": sum(
                        candidate.oracle_ok for candidate in candidate_set.candidates
                    ),
                    "positive_cf_drop_candidate_count": sum(
                        is_positive_fallback_candidate(candidate)
                        for candidate in candidate_set.candidates
                    ),
                    "strict_candidate_count": sum(
                        is_teacher_strict_candidate(candidate)
                        for candidate in candidate_set.candidates
                    ),
                    "drop_reason": candidate_set.drop_reason or "",
                }
            )

    multitarget_train = build_strict_multitarget_rows(
        train_sets,
        frequencies=train_frequency,
        max_targets_per_parent=max_targets_per_parent,
        max_completion_frequency=max_completion_frequency,
    )
    ppo_train = [_ppo_row(parent) for parent in train]
    ppo_val = [_ppo_row(parent) for parent in val]

    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "strict_train": out / "mutagenicity_sft_train_strict_v2.csv",
        "strict_val": out / "mutagenicity_sft_val_strict_v2.csv",
        "fallback_train": out / "mutagenicity_sft_train_fallback_v2.csv",
        "fallback_val": out / "mutagenicity_sft_val_fallback_v2.csv",
        "multitarget_train": out
        / "mutagenicity_sft_train_strict_multitarget_v2.csv",
        "ppo_train": out / "mutagenicity_ppo_prompts_train_label1_v2.csv",
        "ppo_val": out / "mutagenicity_ppo_prompts_val_label1_v2.csv",
        "candidate_train": out / "candidate_inventory_train.csv",
        "candidate_val": out / "candidate_inventory_val.csv",
        "parent_counts": out / "parent_candidate_counts.csv",
        "top_completions": out / "completion_frequency_top100.csv",
    }
    for key in ("strict_train", "strict_val", "fallback_train", "fallback_val"):
        rows = {
            "strict_train": strict_train,
            "strict_val": strict_val,
            "fallback_train": fallback_train,
            "fallback_val": fallback_val,
        }[key]
        write_csv(paths[key], rows, V2_SFT_FIELDS)
    write_csv(paths["multitarget_train"], multitarget_train, V2_SFT_FIELDS)
    write_csv(paths["ppo_train"], ppo_train, V2_PPO_FIELDS)
    write_csv(paths["ppo_val"], ppo_val, V2_PPO_FIELDS)
    write_csv(paths["candidate_train"], train_inventory, CANDIDATE_FIELDS)
    write_csv(paths["candidate_val"], val_inventory, CANDIDATE_FIELDS)
    write_csv(
        paths["parent_counts"],
        parent_candidate_rows,
        (
            "split",
            "parent_id",
            "parent_smiles",
            "proposal_count",
            "valid_candidate_count",
            "oracle_ok_candidate_count",
            "positive_cf_drop_candidate_count",
            "strict_candidate_count",
            "drop_reason",
        ),
    )
    top_rows = _top_completion_rows(
        [*train_inventory, *val_inventory],
        [*strict_train, *strict_val],
    )
    write_csv(
        paths["top_completions"],
        top_rows,
        (
            "completion",
            "candidate_frequency",
            "selected_frequency",
            "strict_count",
            "strict_rate",
            "cf_drop_mean",
            "strategies",
        ),
    )

    summary = {
        "dataset": "Mutagenicity",
        "dataset_version": "sft_ppo_v2",
        "source_label": SOURCE_LABEL,
        "target_label": TARGET_LABEL,
        "selection_rule": "strict_first_then_positive_cf_drop_fallback",
        "strict_definition": "oracle_ok and pred_before == 1 and pred_after == 0",
        "train_parent_count": len(train),
        "val_parent_count": len(val),
        "strict_train_rows": len(strict_train),
        "strict_val_rows": len(strict_val),
        "fallback_train_rows": len(fallback_train),
        "fallback_val_rows": len(fallback_val),
        "multitarget_train_rows": len(multitarget_train),
        "ppo_train_rows": len(ppo_train),
        "ppo_val_rows": len(ppo_val),
        "train_candidate_count": len(train_inventory),
        "val_candidate_count": len(val_inventory),
        "strict_train_distribution": _distribution_summary(strict_train),
        "strict_val_distribution": _distribution_summary(strict_val),
        "fallback_train_distribution": _distribution_summary(fallback_train),
        "fallback_val_distribution": _distribution_summary(fallback_val),
        "leakage_audit": leakage,
        "calibration_test_used_for_training": False,
        "max_targets_per_parent": int(max_targets_per_parent),
        "max_completion_frequency": max_completion_frequency,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    write_json(out / "dataset_summary.json", summary)
    write_json(
        out / "leakage_audit.json",
        {
            **leakage,
            "calibration_test_loaded_for_exclusion_only": True,
            "calibration_test_used_for_training": False,
        },
    )
    report = "\n".join(
        [
            "# Mutagenicity strict-first SFT/PPO v2",
            "",
            f"- Train/val parents: {len(train)} / {len(val)}",
            f"- Strict-only train/val rows: {len(strict_train)} / {len(strict_val)}",
            f"- Fallback train/val rows: {len(fallback_train)} / {len(fallback_val)}",
            f"- PPO train/val parents: {len(ppo_train)} / {len(ppo_val)}",
            "- PPO prompts retain every teacher-correct source parent.",
            "- Calibration/test are read only for leakage exclusion.",
            "- Existing v1 artifacts are not modified.",
        ]
    )
    (out / "dataset_report.md").write_text(report + "\n", encoding="utf-8")
    print("[MUTAGENICITY_SFT_STRICT_V2_BUILD_OK]", flush=True)
    return summary


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def cohort_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    parent_ids = sorted(str(row.get("molecule_id") or row.get("parent_id") or "") for row in rows)
    return hashlib.sha256("\n".join(parent_ids).encode("utf-8")).hexdigest()


__all__ = [
    "CANDIDATE_FIELDS",
    "ParentCandidateSet",
    "StrictFirstSelection",
    "TARGET_QUALITY_FALLBACK",
    "TARGET_QUALITY_STRICT",
    "V2_PPO_FIELDS",
    "V2_SFT_FIELDS",
    "build_mutagenicity_sft_ppo_v2",
    "build_teacher_scorer",
    "candidate_inventory_rows",
    "canonicalize_fragment",
    "cohort_hash",
    "completion_frequency",
    "enumerate_split_candidates",
    "is_positive_fallback_candidate",
    "is_teacher_strict_candidate",
    "read_csv_rows",
    "select_strict_first_candidate",
]
