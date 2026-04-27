"""Build a higher-quality counterfactual SFT v3 dataset from raw HIV.csv."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import json
import math
import random
from pathlib import Path
from typing import Any

from src.chem import (
    build_parent_projection_candidates,
    delete_fragment_from_parent,
    match_core_fragment_to_parent,
    normalize_core_fragment,
    parse_smiles,
)
from src.data.hiv_dataset_utils import (
    HIVParentRecord,
    load_hiv_dataframe,
    normalize_hiv_records,
    sample_records_by_strata,
    stratified_round_robin_order,
)
from src.data.prompts import build_counterfactual_prompt
from src.data.schemas import MoleculeRecord
from src.rewards.counterfactual_oracle import CounterfactualTeacherScorer
from src.utils.io import ensure_directory, write_jsonl

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:  # pragma: no cover - depends on runtime environment
    Chem = None
    MurckoScaffold = None


_CANDIDATE_SOURCE_PRIORITY = {
    "murcko_scaffold": 96,
    "murcko_scaffold_r1": 94,
    "ring_system_r0": 90,
    "ring_system_r1": 86,
    "fg_carboxyl": 82,
    "fg_amide": 82,
    "fg_sulfonic_acid": 82,
    "fg_sulfonamide": 82,
    "fg_azo": 82,
    "fg_nitro": 82,
    "fg_disulfide": 82,
    "fg_halogen": 78,
    "fg_aromatic_oh": 78,
    "atom_k1": 60,
    "atom_k2": 56,
    "atom_k3": 52,
    "bond_k1": 48,
    "bond_k2": 44,
    "brics_component": 40,
}


@dataclass(frozen=True, slots=True)
class SFTV3BuilderConfig:
    """Configurable controls for the HIV -> SFT v3 build pipeline."""

    positive_label: Any = 1
    neg_pos_ratio: float = 2.0
    seed: int = 7
    val_ratio: float = 0.1
    max_parents: int | None = None
    min_atom_ratio: float = 0.10
    max_atom_ratio: float = 0.55
    min_frag_atoms: int = 3
    max_frag_atoms: int = 30
    oracle_path: str | Path | None = None
    use_oracle_ranking: bool = True
    max_candidates_per_parent: int = 160
    include_label_in_prompt: bool = False

    def validate(self) -> None:
        if self.neg_pos_ratio < 0.0:
            raise ValueError("neg_pos_ratio must be non-negative.")
        if not (0.0 < self.val_ratio < 1.0):
            raise ValueError("val_ratio must be between 0 and 1.")
        if self.max_parents is not None and int(self.max_parents) <= 0:
            raise ValueError("max_parents must be positive when provided.")
        if self.min_frag_atoms <= 0:
            raise ValueError("min_frag_atoms must be positive.")
        if self.max_frag_atoms < self.min_frag_atoms:
            raise ValueError("max_frag_atoms must be >= min_frag_atoms.")
        if not (0.0 < self.min_atom_ratio < 1.0):
            raise ValueError("min_atom_ratio must be between 0 and 1.")
        if not (0.0 < self.max_atom_ratio < 1.0):
            raise ValueError("max_atom_ratio must be between 0 and 1.")
        if self.min_atom_ratio >= self.max_atom_ratio:
            raise ValueError("min_atom_ratio must be smaller than max_atom_ratio.")
        if self.max_candidates_per_parent <= 0:
            raise ValueError("max_candidates_per_parent must be positive.")

    @property
    def target_atom_ratio(self) -> float:
        return (self.min_atom_ratio + self.max_atom_ratio) / 2.0


@dataclass(frozen=True, slots=True)
class SFTV3ReferenceCandidate:
    """One structurally valid SFT reference candidate for a single parent molecule."""

    core_fragment: str
    raw_fragment: str
    candidate_strategy: str
    atom_count: int
    atom_ratio: float
    residual_smiles: str
    residual_nonempty: bool
    is_full_parent: bool
    cf_drop: float | None = None
    cf_flip: bool | None = None
    oracle_ok: bool = False


@dataclass(frozen=True, slots=True)
class SFTV3Example:
    """One SFT training example compatible with scripts/train_sft.py."""

    sample_id: str
    graph_id: str
    parent_smiles: str
    label: int
    parent_atom_count: int
    scaffold_smiles: str
    instruction: str
    output: str
    meta: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        payload = {
            "id": self.sample_id,
            "graph_id": self.graph_id,
            "smiles": self.parent_smiles,
            "parent_smiles": self.parent_smiles,
            "label": int(self.label),
            "prompt": self.instruction,
            "response": self.output,
            "instruction": self.instruction,
            "output": self.output,
            "reference_fragment": self.output,
            "task_type": "counterfactual_smiles_core_sft_v3",
            "meta": dict(self.meta),
        }
        return payload


@dataclass(frozen=True, slots=True)
class ParentBuildResult:
    """Per-parent candidate-generation outcome used for reporting."""

    record: HIVParentRecord
    selected_candidate: SFTV3ReferenceCandidate | None
    candidate_count: int
    valid_candidate_count: int
    candidate_strategy_counts: dict[str, int]
    candidate_drop_counts: dict[str, int]
    drop_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SFTV3BuildArtifacts:
    """All output paths and summaries produced by one dataset build."""

    train_output: Path
    val_output: Path
    train_summary_path: Path
    val_summary_path: Path
    report_path: Path
    dropped_summary_path: Path
    train_count: int
    val_count: int
    total_count: int
    build_summary: dict[str, Any]


def build_and_write_sft_v3_dataset(
    *,
    input_csv: str | Path,
    train_output: str | Path,
    val_output: str | Path,
    config: SFTV3BuilderConfig,
) -> SFTV3BuildArtifacts:
    """Build the SFT v3 dataset, write JSONL outputs, and emit summaries."""

    config.validate()
    input_csv_path = Path(input_csv).expanduser().resolve()
    train_output_path = Path(train_output).expanduser().resolve()
    val_output_path = Path(val_output).expanduser().resolve()
    ensure_directory(train_output_path.parent)
    ensure_directory(val_output_path.parent)

    dataframe = load_hiv_dataframe(input_csv_path)
    normalized_records, normalization_summary = normalize_hiv_records(
        dataframe,
        positive_label=config.positive_label,
    )
    if not normalized_records:
        raise ValueError("No valid parent molecules remained after HIV.csv normalization.")

    positives = [record for record in normalized_records if record.label == 1]
    negatives = [record for record in normalized_records if record.label == 0]
    if not positives:
        raise ValueError("No positive molecules remained after HIV.csv normalization.")
    if not negatives:
        raise ValueError("No negative molecules remained after HIV.csv normalization.")

    selection = _select_parent_queues(
        positives=positives,
        negatives=negatives,
        config=config,
    )
    oracle_scorer, oracle_summary = _build_oracle_scorer(config)

    build_results, build_summary = _build_parent_results(
        positive_records=selection["positive_records"],
        negative_records=selection["negative_records"],
        negative_target=selection["negative_target"],
        config=config,
        oracle_scorer=oracle_scorer,
        oracle_summary=oracle_summary,
    )
    examples = _materialize_examples(
        build_results,
        include_label_in_prompt=config.include_label_in_prompt,
    )
    if not examples:
        raise ValueError("No SFT examples were built after candidate filtering.")

    train_examples, val_examples, split_summary = split_examples_scaffold_aware(
        examples,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    if not train_examples:
        raise ValueError("Scaffold split produced an empty train set.")
    if not val_examples:
        raise ValueError("Scaffold split produced an empty validation set.")

    write_jsonl(train_output_path, (example.to_json() for example in train_examples))
    write_jsonl(val_output_path, (example.to_json() for example in val_examples))

    train_summary = _dataset_summary(
        train_examples,
        split_name="train",
        source_path=train_output_path,
    )
    val_summary = _dataset_summary(
        val_examples,
        split_name="val",
        source_path=val_output_path,
    )
    train_summary_path = train_output_path.with_suffix(".summary.json")
    val_summary_path = val_output_path.with_suffix(".summary.json")
    train_summary_path.write_text(
        json.dumps(train_summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    val_summary_path.write_text(
        json.dumps(val_summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    dropped_summary = _dropped_summary(
        normalization_summary=normalization_summary,
        build_results=build_results,
    )
    dropped_summary_path = train_output_path.with_name(
        f"{train_output_path.stem}.dropped_summary.json"
    )
    dropped_summary_path.write_text(
        json.dumps(dropped_summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    build_report = _render_report(
        input_csv_path=input_csv_path,
        config=config,
        normalization_summary=normalization_summary,
        selection_summary=selection["summary"],
        build_summary=build_summary,
        split_summary=split_summary,
        train_summary=train_summary,
        val_summary=val_summary,
        dropped_summary_path=dropped_summary_path,
    )
    report_path = train_output_path.with_name(f"{train_output_path.stem}.report.txt")
    report_path.write_text(build_report, encoding="utf-8")

    overall_summary = {
        "input_csv": str(input_csv_path),
        "train_output": str(train_output_path),
        "val_output": str(val_output_path),
        "normalization": normalization_summary,
        "selection": selection["summary"],
        "build": build_summary,
        "split": split_summary,
        "train": train_summary,
        "val": val_summary,
        "oracle": oracle_summary,
        "dropped_summary_path": str(dropped_summary_path),
        "report_path": str(report_path),
    }
    return SFTV3BuildArtifacts(
        train_output=train_output_path,
        val_output=val_output_path,
        train_summary_path=train_summary_path,
        val_summary_path=val_summary_path,
        report_path=report_path,
        dropped_summary_path=dropped_summary_path,
        train_count=len(train_examples),
        val_count=len(val_examples),
        total_count=len(examples),
        build_summary=overall_summary,
    )


def select_reference_candidate_for_parent(
    record: HIVParentRecord,
    *,
    config: SFTV3BuilderConfig,
    oracle_scorer: CounterfactualTeacherScorer | None = None,
) -> ParentBuildResult:
    """Select the best filtered reference candidate for one parent molecule."""

    parent = parse_smiles(
        record.parent_smiles,
        sanitize=True,
        canonicalize=True,
        allow_capped_fragments=False,
    )
    if not parent.sanitized or parent.mol is None:
        return ParentBuildResult(
            record=record,
            selected_candidate=None,
            candidate_count=0,
            valid_candidate_count=0,
            candidate_strategy_counts={},
            candidate_drop_counts={},
            drop_reason="parent_parse_failed",
        )

    proposals = _collect_candidate_proposals(
        parent_mol=parent.mol,
        parent_smiles=record.parent_smiles,
        config=config,
    )
    if not proposals:
        return ParentBuildResult(
            record=record,
            selected_candidate=None,
            candidate_count=0,
            valid_candidate_count=0,
            candidate_strategy_counts={},
            candidate_drop_counts={},
            drop_reason="no_candidate_proposals",
        )

    candidate_strategy_counts: Counter[str] = Counter()
    candidate_drop_counts: Counter[str] = Counter()
    valid_candidates: list[SFTV3ReferenceCandidate] = []
    for core_fragment, strategy in proposals.items():
        candidate_strategy_counts[strategy] += 1
        candidate, drop_reason = _build_candidate(
            record=record,
            core_fragment=core_fragment,
            strategy=strategy,
            config=config,
            oracle_scorer=oracle_scorer,
        )
        if candidate is None:
            candidate_drop_counts[drop_reason or "candidate_filter_failed"] += 1
            continue
        valid_candidates.append(candidate)

    if not valid_candidates:
        return ParentBuildResult(
            record=record,
            selected_candidate=None,
            candidate_count=len(proposals),
            valid_candidate_count=0,
            candidate_strategy_counts=dict(sorted(candidate_strategy_counts.items())),
            candidate_drop_counts=dict(sorted(candidate_drop_counts.items())),
            drop_reason="no_candidates_after_filter",
        )

    best_candidate = max(
        valid_candidates,
        key=lambda candidate: _candidate_ranking_key(candidate, config=config),
    )
    return ParentBuildResult(
        record=record,
        selected_candidate=best_candidate,
        candidate_count=len(proposals),
        valid_candidate_count=len(valid_candidates),
        candidate_strategy_counts=dict(sorted(candidate_strategy_counts.items())),
        candidate_drop_counts=dict(sorted(candidate_drop_counts.items())),
        drop_reason=None,
    )


def split_examples_scaffold_aware(
    examples: list[SFTV3Example],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[SFTV3Example], list[SFTV3Example], dict[str, Any]]:
    """Split examples with an approximate scaffold-level holdout."""

    if len(examples) <= 1:
        return list(examples), [], {
            "method": "degenerate_all_train",
            "val_ratio": val_ratio,
            "train_label_counts": _example_label_counts(examples),
            "val_label_counts": {"0": 0, "1": 0},
        }

    target_val_total = max(1, min(len(examples) - 1, int(round(len(examples) * val_ratio))))
    scaffold_groups: dict[str, list[SFTV3Example]] = defaultdict(list)
    for example in examples:
        scaffold_groups[str(example.scaffold_smiles or "ACYCLIC")].append(example)
    if len(scaffold_groups) <= 1:
        train_examples, val_examples = _fallback_label_stratified_split(
            examples,
            target_val_total=target_val_total,
            seed=seed,
        )
        return train_examples, val_examples, _split_summary(
            method="label_stratified_fallback",
            train_examples=train_examples,
            val_examples=val_examples,
            target_val_total=target_val_total,
        )

    groups = list(scaffold_groups.items())
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)

    total_pos = sum(1 for example in examples if example.label == 1)
    total_neg = len(examples) - total_pos
    target_val_pos = int(round(total_pos * val_ratio))
    target_val_neg = int(round(total_neg * val_ratio))
    target_val_pos = min(target_val_total, max(0, target_val_pos))
    target_val_neg = min(target_val_total, max(0, target_val_neg))

    selected_scaffolds: set[str] = set()
    current_val: list[SFTV3Example] = []
    current_total = 0
    current_pos = 0
    current_neg = 0
    remaining_groups = list(groups)

    while remaining_groups and current_total < target_val_total:
        best_index = 0
        best_score: tuple[float, float, float, int] | None = None
        for index, (_scaffold, scaffold_examples) in enumerate(remaining_groups):
            group_total = len(scaffold_examples)
            group_pos = sum(1 for example in scaffold_examples if example.label == 1)
            group_neg = group_total - group_pos
            new_total = current_total + group_total
            new_pos = current_pos + group_pos
            new_neg = current_neg + group_neg
            overshoot = max(0, new_total - target_val_total)
            score = (
                abs(target_val_total - new_total) + (4.0 * overshoot),
                abs(target_val_pos - new_pos),
                abs(target_val_neg - new_neg),
                group_total,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_index = index

        scaffold, scaffold_examples = remaining_groups.pop(best_index)
        selected_scaffolds.add(scaffold)
        current_val.extend(scaffold_examples)
        current_total += len(scaffold_examples)
        current_pos += sum(1 for example in scaffold_examples if example.label == 1)
        current_neg += sum(1 for example in scaffold_examples if example.label == 0)

    val_examples = list(current_val)
    train_examples = [
        example
        for scaffold, scaffold_examples in groups
        if scaffold not in selected_scaffolds
        for example in scaffold_examples
    ]
    if not train_examples or not val_examples:
        train_examples, val_examples = _fallback_label_stratified_split(
            examples,
            target_val_total=target_val_total,
            seed=seed,
        )
        return train_examples, val_examples, _split_summary(
            method="label_stratified_fallback",
            train_examples=train_examples,
            val_examples=val_examples,
            target_val_total=target_val_total,
        )
    return train_examples, val_examples, _split_summary(
        method="scaffold_group_greedy",
        train_examples=train_examples,
        val_examples=val_examples,
        target_val_total=target_val_total,
    )


def _select_parent_queues(
    *,
    positives: list[HIVParentRecord],
    negatives: list[HIVParentRecord],
    config: SFTV3BuilderConfig,
) -> dict[str, Any]:
    positive_records = stratified_round_robin_order(
        list(positives),
        seed=config.seed,
    )
    negative_records = stratified_round_robin_order(
        list(negatives),
        seed=config.seed + 17,
    )
    raw_positive_count = len(positive_records)
    desired_negative_target = min(
        len(negative_records),
        int(math.ceil(raw_positive_count * config.neg_pos_ratio)),
    )

    summary = {
        "raw_positive_count": raw_positive_count,
        "raw_negative_count": len(negative_records),
        "desired_negative_target": desired_negative_target,
        "neg_pos_ratio": config.neg_pos_ratio,
        "max_parents": config.max_parents,
    }

    if config.max_parents is None:
        summary.update(
            {
                "selection_mode": "keep_all_positives_negatives_by_target",
                "selected_positive_count": raw_positive_count,
                "selected_negative_count": len(negative_records),
                "negative_success_target": desired_negative_target,
            }
        )
        return {
            "positive_records": positive_records,
            "negative_records": negative_records,
            "negative_target": desired_negative_target,
            "summary": summary,
        }

    cap = int(config.max_parents)
    if raw_positive_count >= cap:
        selected_positives = sample_records_by_strata(
            positive_records,
            sample_size=cap,
            seed=config.seed,
        )
        summary.update(
            {
                "selection_mode": "positive_only_due_to_parent_cap",
                "selected_positive_count": len(selected_positives),
                "selected_negative_count": 0,
                "negative_success_target": 0,
            }
        )
        return {
            "positive_records": selected_positives,
            "negative_records": [],
            "negative_target": 0,
            "summary": summary,
        }

    remaining_budget = max(0, cap - raw_positive_count)
    negative_target = min(remaining_budget, desired_negative_target, len(negative_records))
    selected_negatives = sample_records_by_strata(
        negative_records,
        sample_size=negative_target,
        seed=config.seed + 17,
    )
    summary.update(
        {
            "selection_mode": "capped_parent_pool",
            "selected_positive_count": raw_positive_count,
            "selected_negative_count": len(selected_negatives),
            "negative_success_target": len(selected_negatives),
        }
    )
    return {
        "positive_records": positive_records,
        "negative_records": selected_negatives,
        "negative_target": len(selected_negatives),
        "summary": summary,
    }


def _build_oracle_scorer(
    config: SFTV3BuilderConfig,
) -> tuple[CounterfactualTeacherScorer | None, dict[str, Any]]:
    if not config.oracle_path:
        return None, {
            "requested": False,
            "ranking_mode": "heuristic",
            "teacher_available": False,
            "availability_reason": "teacher_path_not_provided",
        }
    scorer = CounterfactualTeacherScorer(config.oracle_path, device="cpu")
    ranking_mode = "oracle" if config.use_oracle_ranking and scorer.available else "heuristic"
    return scorer, {
        "requested": True,
        "teacher_available": bool(scorer.available),
        "availability_reason": scorer.availability_reason,
        "ranking_mode": ranking_mode,
        "teacher_path": str(Path(config.oracle_path).expanduser().resolve()),
    }


def _build_parent_results(
    *,
    positive_records: list[HIVParentRecord],
    negative_records: list[HIVParentRecord],
    negative_target: int,
    config: SFTV3BuilderConfig,
    oracle_scorer: CounterfactualTeacherScorer | None,
    oracle_summary: dict[str, Any],
) -> tuple[list[ParentBuildResult], dict[str, Any]]:
    results: list[ParentBuildResult] = []
    selected_strategy_counts: Counter[str] = Counter()
    candidate_strategy_counts: Counter[str] = Counter()
    candidate_drop_counts: Counter[str] = Counter()
    parent_drop_counts: Counter[str] = Counter()
    label_success_counts: Counter[int] = Counter()
    label_attempt_counts: Counter[int] = Counter()
    oracle_scored_candidates = 0
    oracle_success_candidates = 0
    oracle_flip_selected = 0
    total_candidates = 0
    total_valid_candidates = 0

    for record in positive_records:
        label_attempt_counts[record.label] += 1
        result = select_reference_candidate_for_parent(
            record,
            config=config,
            oracle_scorer=oracle_scorer if oracle_summary["ranking_mode"] == "oracle" else None,
        )
        results.append(result)
        total_candidates += result.candidate_count
        total_valid_candidates += result.valid_candidate_count
        candidate_strategy_counts.update(result.candidate_strategy_counts)
        candidate_drop_counts.update(result.candidate_drop_counts)
        if result.selected_candidate is None:
            parent_drop_counts[result.drop_reason or "parent_build_failed"] += 1
            continue
        label_success_counts[record.label] += 1
        selected_strategy_counts[result.selected_candidate.candidate_strategy] += 1
        oracle_scored_candidates += int(result.selected_candidate.cf_drop is not None)
        oracle_success_candidates += int(result.selected_candidate.oracle_ok)
        oracle_flip_selected += int(bool(result.selected_candidate.cf_flip))

    effective_negative_target = min(
        negative_target,
        int(math.ceil(label_success_counts.get(1, 0) * config.neg_pos_ratio)),
    )
    negative_success_count = 0
    for record in negative_records:
        if negative_success_count >= effective_negative_target:
            break
        label_attempt_counts[record.label] += 1
        result = select_reference_candidate_for_parent(
            record,
            config=config,
            oracle_scorer=oracle_scorer if oracle_summary["ranking_mode"] == "oracle" else None,
        )
        results.append(result)
        total_candidates += result.candidate_count
        total_valid_candidates += result.valid_candidate_count
        candidate_strategy_counts.update(result.candidate_strategy_counts)
        candidate_drop_counts.update(result.candidate_drop_counts)
        if result.selected_candidate is None:
            parent_drop_counts[result.drop_reason or "parent_build_failed"] += 1
            continue
        negative_success_count += 1
        label_success_counts[record.label] += 1
        selected_strategy_counts[result.selected_candidate.candidate_strategy] += 1
        oracle_scored_candidates += int(result.selected_candidate.cf_drop is not None)
        oracle_success_candidates += int(result.selected_candidate.oracle_ok)
        oracle_flip_selected += int(bool(result.selected_candidate.cf_flip))

    examples_built = sum(result.selected_candidate is not None for result in results)
    build_summary = {
        "attempted_parent_count": len(results),
        "built_example_count": examples_built,
        "failed_parent_count": len(results) - examples_built,
        "positive_attempt_count": int(label_attempt_counts.get(1, 0)),
        "negative_attempt_count": int(label_attempt_counts.get(0, 0)),
        "positive_success_count": int(label_success_counts.get(1, 0)),
        "negative_success_count": int(label_success_counts.get(0, 0)),
        "requested_negative_target": int(negative_target),
        "effective_negative_target": int(effective_negative_target),
        "achieved_negative_target": int(label_success_counts.get(0, 0)),
        "total_candidate_count": int(total_candidates),
        "valid_candidate_count": int(total_valid_candidates),
        "candidate_strategy_counts": dict(sorted(candidate_strategy_counts.items())),
        "candidate_drop_counts": dict(sorted(candidate_drop_counts.items())),
        "selected_strategy_counts": dict(sorted(selected_strategy_counts.items())),
        "parent_drop_counts": dict(sorted(parent_drop_counts.items())),
        "oracle_scored_selected_count": int(oracle_scored_candidates),
        "oracle_ok_selected_count": int(oracle_success_candidates),
        "oracle_flip_selected_count": int(oracle_flip_selected),
    }
    return results, build_summary


def _materialize_examples(
    build_results: list[ParentBuildResult],
    *,
    include_label_in_prompt: bool,
) -> list[SFTV3Example]:
    examples: list[SFTV3Example] = []
    for result in build_results:
        candidate = result.selected_candidate
        if candidate is None:
            continue
        record = result.record
        prompt = build_counterfactual_prompt(
            MoleculeRecord(
                record_id=record.sample_id,
                smiles=record.parent_smiles,
                label=record.label,
            ),
            include_label=include_label_in_prompt,
        )
        meta = {
            "parent_smiles": record.parent_smiles,
            "label": int(record.label),
            "parent_atom_count": int(record.parent_atom_count),
            "parent_scaffold": record.scaffold_smiles,
            "raw_fragment": candidate.raw_fragment,
            "core_fragment": candidate.core_fragment,
            "atom_ratio": candidate.atom_ratio,
            "candidate_strategy": candidate.candidate_strategy,
            "residual_nonempty": candidate.residual_nonempty,
            "residual_smiles": candidate.residual_smiles,
            "is_full_parent": candidate.is_full_parent,
            "cf_drop": candidate.cf_drop,
            "cf_flip": candidate.cf_flip,
            "oracle_ok": candidate.oracle_ok,
            "source_row_index": record.source_row_index,
            "source_smiles": record.source_smiles,
            "reference_build": "sft_v3_from_hiv",
            "target_format": "core_no_dummy",
        }
        examples.append(
            SFTV3Example(
                sample_id=record.sample_id,
                graph_id=record.sample_id,
                parent_smiles=record.parent_smiles,
                label=record.label,
                parent_atom_count=record.parent_atom_count,
                scaffold_smiles=record.scaffold_smiles,
                instruction=prompt,
                output=candidate.core_fragment,
                meta=meta,
            )
        )
    return examples


def _collect_candidate_proposals(
    *,
    parent_mol: object,
    parent_smiles: str,
    config: SFTV3BuilderConfig,
) -> dict[str, str]:
    proposals: dict[str, str] = {}
    projection_candidates = build_parent_projection_candidates(
        parent_mol,
        parent_smiles=parent_smiles,
        max_candidates=config.max_candidates_per_parent,
        min_atoms=max(1, config.min_frag_atoms),
        max_atom_ratio=config.max_atom_ratio,
        enable_khop3=False,
    )
    for candidate in projection_candidates:
        proposals.setdefault(candidate.smiles, candidate.source)
    for smiles, strategy in _murcko_like_candidates(
        parent_mol=parent_mol,
        parent_smiles=parent_smiles,
    ):
        proposals.setdefault(smiles, strategy)
    return proposals


def _murcko_like_candidates(
    *,
    parent_mol: object,
    parent_smiles: str,
) -> list[tuple[str, str]]:
    if Chem is None or MurckoScaffold is None or parent_mol is None:
        return []
    candidates: list[tuple[str, str]] = []
    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(parent_mol)
    except Exception:  # pragma: no cover - depends on RDKit internals
        scaffold_mol = None
    if scaffold_mol is None or scaffold_mol.GetNumAtoms() == 0:
        return []

    scaffold_smiles = _mol_to_smiles(scaffold_mol)
    if scaffold_smiles and scaffold_smiles != parent_smiles:
        candidates.append((scaffold_smiles, "murcko_scaffold"))

    match = match_core_fragment_to_parent(parent_smiles, scaffold_smiles or "")
    if match.matched and match.match_atom_indices:
        expanded = _expand_parent_atom_indices(
            parent_mol,
            match.match_atom_indices,
            radius=1,
        )
        expanded_smiles = _fragment_smiles_from_atom_indices(parent_mol, expanded)
        if expanded_smiles and expanded_smiles != parent_smiles:
            candidates.append((expanded_smiles, "murcko_scaffold_r1"))
    return candidates


def _build_candidate(
    *,
    record: HIVParentRecord,
    core_fragment: str,
    strategy: str,
    config: SFTV3BuilderConfig,
    oracle_scorer: CounterfactualTeacherScorer | None,
) -> tuple[SFTV3ReferenceCandidate | None, str | None]:
    normalized = normalize_core_fragment(core_fragment, keep_largest_component=True)
    if not normalized.core_parse_ok or not normalized.core_fragment_smiles:
        return None, normalized.failure_tag or "normalize_core_failed"

    strict_core = str(normalized.core_fragment_smiles).strip()
    match = match_core_fragment_to_parent(record.parent_smiles, strict_core)
    if not match.matched:
        return None, match.reason or "not_parent_substructure"
    if match.full_parent:
        return None, "full_parent_fragment"
    if match.atom_ratio is None:
        return None, "missing_atom_ratio"

    atom_ratio = float(match.atom_ratio)
    atom_count = int(match.atom_count)
    if atom_ratio < config.min_atom_ratio:
        return None, "below_min_atom_ratio"
    if atom_ratio > config.max_atom_ratio:
        return None, "above_max_atom_ratio"
    if atom_count < config.min_frag_atoms:
        return None, "below_min_frag_atoms"
    if atom_count > config.max_frag_atoms:
        return None, "above_max_frag_atoms"

    deletion = delete_fragment_from_parent(record.parent_smiles, strict_core, max_matches=1)
    if not deletion.success:
        return None, deletion.failure_reason or "deletion_failed"
    residual_smiles = str(deletion.residual_smiles or "")
    if not residual_smiles:
        return None, "empty_residual"

    candidate = SFTV3ReferenceCandidate(
        core_fragment=strict_core,
        raw_fragment=match.explanation_fragment_with_dummy or strict_core,
        candidate_strategy=strategy,
        atom_count=atom_count,
        atom_ratio=atom_ratio,
        residual_smiles=residual_smiles,
        residual_nonempty=True,
        is_full_parent=False,
        cf_drop=None,
        cf_flip=None,
        oracle_ok=False,
    )
    if oracle_scorer is None:
        return candidate, None

    oracle_result = oracle_scorer.score_counterfactual(
        record.parent_smiles,
        strict_core,
        label=record.label,
    )
    cf_drop = oracle_result.get("cf_drop")
    candidate = replace(
        candidate,
        cf_drop=float(cf_drop) if cf_drop is not None else None,
        cf_flip=bool(oracle_result.get("cf_flip"))
        if oracle_result.get("teacher_result_ok")
        else None,
        oracle_ok=bool(oracle_result.get("teacher_result_ok")),
    )
    return candidate, None


def _candidate_ranking_key(
    candidate: SFTV3ReferenceCandidate,
    *,
    config: SFTV3BuilderConfig,
) -> tuple[float, ...]:
    strategy_priority = float(_CANDIDATE_SOURCE_PRIORITY.get(candidate.candidate_strategy, 0))
    ratio_distance = abs(candidate.atom_ratio - config.target_atom_ratio)
    heuristic_key = (
        -ratio_distance,
        strategy_priority,
        -abs(candidate.atom_count - _target_atom_count(candidate, config)),
        -candidate.atom_count,
    )
    if candidate.cf_drop is None or candidate.cf_flip is None:
        return (0.0, 0.0, -1.0, *heuristic_key)
    return (
        1.0,
        1.0 if candidate.oracle_ok else 0.0,
        1.0 if candidate.cf_flip else 0.0,
        float(candidate.cf_drop),
        *heuristic_key,
    )


def _target_atom_count(
    candidate: SFTV3ReferenceCandidate,
    config: SFTV3BuilderConfig,
) -> float:
    del candidate
    return (config.min_frag_atoms + config.max_frag_atoms) / 2.0


def _fallback_label_stratified_split(
    examples: list[SFTV3Example],
    *,
    target_val_total: int,
    seed: int,
) -> tuple[list[SFTV3Example], list[SFTV3Example]]:
    rng = random.Random(seed)
    positives = [example for example in examples if example.label == 1]
    negatives = [example for example in examples if example.label == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    pos_val = min(len(positives), int(round(len(positives) * target_val_total / len(examples))))
    neg_val = min(len(negatives), max(0, target_val_total - pos_val))
    val_examples = positives[:pos_val] + negatives[:neg_val]
    train_examples = positives[pos_val:] + negatives[neg_val:]
    if not val_examples and train_examples:
        val_examples = [train_examples.pop()]
    if not train_examples and val_examples:
        train_examples = [val_examples.pop()]
    rng.shuffle(train_examples)
    rng.shuffle(val_examples)
    return train_examples, val_examples


def _split_summary(
    *,
    method: str,
    train_examples: list[SFTV3Example],
    val_examples: list[SFTV3Example],
    target_val_total: int,
) -> dict[str, Any]:
    train_scaffolds = {example.scaffold_smiles for example in train_examples}
    val_scaffolds = {example.scaffold_smiles for example in val_examples}
    return {
        "method": method,
        "target_val_total": int(target_val_total),
        "train_count": len(train_examples),
        "val_count": len(val_examples),
        "train_label_counts": _example_label_counts(train_examples),
        "val_label_counts": _example_label_counts(val_examples),
        "train_unique_scaffolds": len(train_scaffolds),
        "val_unique_scaffolds": len(val_scaffolds),
        "scaffold_overlap_count": len(train_scaffolds & val_scaffolds),
    }


def _dataset_summary(
    examples: list[SFTV3Example],
    *,
    split_name: str,
    source_path: Path,
) -> dict[str, Any]:
    atom_ratios = [float(example.meta["atom_ratio"]) for example in examples]
    parent_atom_counts = [int(example.parent_atom_count) for example in examples]
    strategy_counts = Counter(
        str(example.meta.get("candidate_strategy") or "unknown")
        for example in examples
    )
    cf_drop_values = [
        float(example.meta["cf_drop"])
        for example in examples
        if example.meta.get("cf_drop") is not None
    ]
    cf_flip_values = [
        bool(example.meta["cf_flip"])
        for example in examples
        if example.meta.get("cf_flip") is not None
    ]
    scaffolds = {example.scaffold_smiles for example in examples}
    return {
        "split": split_name,
        "path": str(source_path),
        "count": len(examples),
        "label_counts": _example_label_counts(examples),
        "unique_scaffolds": len(scaffolds),
        "strategy_counts": dict(sorted(strategy_counts.items())),
        "atom_ratio": _numeric_summary(atom_ratios),
        "parent_atom_count": _numeric_summary(parent_atom_counts),
        "cf_drop": _numeric_summary(cf_drop_values) if cf_drop_values else None,
        "cf_flip_rate": (
            sum(1 for value in cf_flip_values if value) / len(cf_flip_values)
            if cf_flip_values
            else None
        ),
    }


def _dropped_summary(
    *,
    normalization_summary: dict[str, Any],
    build_results: list[ParentBuildResult],
) -> dict[str, Any]:
    parent_drop_counts = Counter(
        result.drop_reason
        for result in build_results
        if result.selected_candidate is None and result.drop_reason
    )
    preview = [
        {
            "sample_id": result.record.sample_id,
            "label": result.record.label,
            "parent_smiles": result.record.parent_smiles,
            "drop_reason": result.drop_reason,
            "candidate_count": result.candidate_count,
            "valid_candidate_count": result.valid_candidate_count,
            "candidate_drop_counts": result.candidate_drop_counts,
        }
        for result in build_results
        if result.selected_candidate is None
    ][:50]
    return {
        "normalization_dropped_counts": normalization_summary.get("dropped_counts", {}),
        "parent_build_drop_counts": dict(sorted(parent_drop_counts.items())),
        "failed_parent_preview": preview,
    }


def _render_report(
    *,
    input_csv_path: Path,
    config: SFTV3BuilderConfig,
    normalization_summary: dict[str, Any],
    selection_summary: dict[str, Any],
    build_summary: dict[str, Any],
    split_summary: dict[str, Any],
    train_summary: dict[str, Any],
    val_summary: dict[str, Any],
    dropped_summary_path: Path,
) -> str:
    lines = [
        "SFT v3 HIV Build Report",
        "=======================",
        "",
        f"Input CSV: {input_csv_path}",
        f"positive_label: {config.positive_label}",
        f"neg_pos_ratio: {config.neg_pos_ratio}",
        f"val_ratio: {config.val_ratio}",
        f"max_parents: {config.max_parents}",
        f"atom_ratio_window: [{config.min_atom_ratio:.2f}, {config.max_atom_ratio:.2f}]",
        f"frag_atom_window: [{config.min_frag_atoms}, {config.max_frag_atoms}]",
        f"use_oracle_ranking: {config.use_oracle_ranking}",
        f"oracle_path: {config.oracle_path}",
        "",
        "Normalization",
        "-------------",
        f"valid_rows: {normalization_summary['valid_rows']}",
        f"dropped_rows: {normalization_summary['dropped_rows']}",
        f"label_counts: {normalization_summary['valid_label_counts']}",
        f"smiles_column: {normalization_summary['smiles_column']}",
        f"label_column: {normalization_summary['label_column']}",
        "",
        "Selection",
        "---------",
        f"selection_mode: {selection_summary['selection_mode']}",
        f"selected_positive_count: {selection_summary['selected_positive_count']}",
        f"selected_negative_count: {selection_summary['selected_negative_count']}",
        f"negative_success_target: {selection_summary['negative_success_target']}",
        "",
        "Candidate Build",
        "---------------",
        f"attempted_parent_count: {build_summary['attempted_parent_count']}",
        f"built_example_count: {build_summary['built_example_count']}",
        f"failed_parent_count: {build_summary['failed_parent_count']}",
        f"selected_strategy_counts: {build_summary['selected_strategy_counts']}",
        f"parent_drop_counts: {build_summary['parent_drop_counts']}",
        f"candidate_drop_counts: {build_summary['candidate_drop_counts']}",
        "",
        "Split",
        "-----",
        f"method: {split_summary['method']}",
        f"train_count: {split_summary['train_count']}",
        f"val_count: {split_summary['val_count']}",
        f"train_label_counts: {split_summary['train_label_counts']}",
        f"val_label_counts: {split_summary['val_label_counts']}",
        f"scaffold_overlap_count: {split_summary['scaffold_overlap_count']}",
        "",
        "Train Summary",
        "-------------",
        f"count: {train_summary['count']}",
        f"label_counts: {train_summary['label_counts']}",
        f"unique_scaffolds: {train_summary['unique_scaffolds']}",
        f"strategy_counts: {train_summary['strategy_counts']}",
        f"atom_ratio: {train_summary['atom_ratio']}",
        "",
        "Val Summary",
        "-----------",
        f"count: {val_summary['count']}",
        f"label_counts: {val_summary['label_counts']}",
        f"unique_scaffolds: {val_summary['unique_scaffolds']}",
        f"strategy_counts: {val_summary['strategy_counts']}",
        f"atom_ratio: {val_summary['atom_ratio']}",
        "",
        f"Dropped summary JSON: {dropped_summary_path}",
        "",
    ]
    return "\n".join(lines)


def _example_label_counts(examples: list[SFTV3Example]) -> dict[str, int]:
    counts = Counter(example.label for example in examples)
    return {
        "0": int(counts.get(0, 0)),
        "1": int(counts.get(1, 0)),
    }


def _numeric_summary(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
        }
    sorted_values = sorted(float(value) for value in values)
    return {
        "count": len(sorted_values),
        "min": _round(sorted_values[0]),
        "max": _round(sorted_values[-1]),
        "mean": _round(sum(sorted_values) / len(sorted_values)),
        "median": _round(_percentile(sorted_values, 0.50)),
        "p25": _round(_percentile(sorted_values, 0.25)),
        "p75": _round(_percentile(sorted_values, 0.75)),
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = max(0.0, min(1.0, fraction)) * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _expand_parent_atom_indices(
    parent_mol: object,
    atom_indices: tuple[int, ...],
    *,
    radius: int,
) -> tuple[int, ...]:
    visited = {int(index) for index in atom_indices}
    frontier = set(visited)
    for _ in range(max(0, int(radius))):
        next_frontier: set[int] = set()
        for atom_index in frontier:
            atom = parent_mol.GetAtomWithIdx(int(atom_index))
            for neighbor in atom.GetNeighbors():
                neighbor_index = int(neighbor.GetIdx())
                if neighbor_index not in visited:
                    visited.add(neighbor_index)
                    next_frontier.add(neighbor_index)
        frontier = next_frontier
        if not frontier:
            break
    return tuple(sorted(visited))


def _fragment_smiles_from_atom_indices(
    parent_mol: object,
    atom_indices: tuple[int, ...],
) -> str | None:
    if Chem is None or not atom_indices:
        return None
    try:
        smiles = Chem.MolFragmentToSmiles(
            parent_mol,
            atomsToUse=list(atom_indices),
            canonical=True,
            isomericSmiles=True,
        )
    except Exception:  # pragma: no cover - depends on RDKit internals
        return None
    normalized = str(smiles or "").strip()
    return normalized or None


def _mol_to_smiles(mol: object | None) -> str | None:
    if Chem is None or mol is None:
        return None
    try:
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:  # pragma: no cover - depends on RDKit internals
        return None
    normalized = str(smiles or "").strip()
    return normalized or None


__all__ = [
    "SFTV3BuildArtifacts",
    "SFTV3BuilderConfig",
    "SFTV3Example",
    "SFTV3ReferenceCandidate",
    "build_and_write_sft_v3_dataset",
    "select_reference_candidate_for_parent",
    "split_examples_scaffold_aware",
]
