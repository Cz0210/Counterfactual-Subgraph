from __future__ import annotations

import math

from src.eval.mutagenicity_generator import (
    aggregate_generator_rows,
    best_task_checkpoint_payload,
    compare_parent_difficulty,
    fragment_frequency_strict_summary,
    rank_checkpoints,
    stable_json_hash,
    summarize_strategy_failures,
    validation_cohort_hash,
)


def _row(parent: str, candidate: int, *, flip: bool, distance: float) -> dict:
    return {
        "molecule_id": parent,
        "candidate_index": candidate,
        "parse_ok": True,
        "valid": True,
        "direct_substructure": True,
        "final_substructure": True,
        "projection_used": False,
        "oracle_ok": True,
        "cf_flip": flip,
        "cf_drop": 0.4 if flip else 0.1,
        "atom_ratio": distance,
        "final_fragment": f"C{candidate}{parent}",
    }


def test_full_parent_hit_n_aggregation_and_diversity() -> None:
    rows = [
        _row(parent, candidate, flip=(parent == "p1" and candidate == 2), distance=0.3)
        for parent in ("p1", "p2")
        for candidate in (1, 2)
    ]
    summary = aggregate_generator_rows(
        rows, num_parents=2, num_candidates_per_parent=2
    )
    assert summary["candidate_level_strict_flip_rate"] == 0.25
    assert summary["parent_hit_at_2"] == 0.5
    assert summary["unique_generated_fragments"] == 4
    frequency = fragment_frequency_strict_summary(
        [{**row, "model_name": "model"} for row in rows]
    )
    assert len(frequency) == 4
    assert sum(row["strict_candidate_count"] for row in frequency) == 1


def test_task_checkpoint_selection_is_lexicographic() -> None:
    rows = [
        {
            "checkpoint": "checkpoint-50",
            "strict_cf_flip_rate": 0.6,
            "cf_drop_mean": 0.2,
            "final_substructure_rate": 1.0,
            "parse_ok_rate": 1.0,
            "atom_ratio_mean": 0.35,
            "duplicate_rate": 0.1,
        },
        {
            "checkpoint": "checkpoint-100",
            "strict_cf_flip_rate": 0.6,
            "cf_drop_mean": 0.3,
            "final_substructure_rate": 0.9,
            "parse_ok_rate": 1.0,
            "atom_ratio_mean": 0.35,
            "duplicate_rate": 0.0,
        },
    ]
    ranked = rank_checkpoints(rows)
    assert ranked[0]["checkpoint"] == "checkpoint-100"
    best = best_task_checkpoint_payload(
        ranked,
        cohort_hash="cohort",
        decoding_config_hash="decode",
    )
    assert best["checkpoint"] == "checkpoint-100"
    assert best["selection_split"] == "val"


def test_cohort_and_decoding_hashes_are_deterministic() -> None:
    assert validation_cohort_hash(["b", "a"]) == validation_cohort_hash(["a", "b"])
    assert stable_json_hash({"b": 2, "a": 1}) == stable_json_hash({"a": 1, "b": 2})


def test_parent_difficulty_categories() -> None:
    sft = [
        {**_row("easy", 1, flip=True, distance=0.3), "parent_smiles": "CC"},
        {**_row("improved", 1, flip=False, distance=0.3), "parent_smiles": "CO"},
        {**_row("regressed", 1, flip=True, distance=0.3), "parent_smiles": "CN"},
        {**_row("hard", 1, flip=False, distance=0.3), "parent_smiles": "CF"},
    ]
    ppo = [
        _row("easy", 1, flip=True, distance=0.3),
        _row("improved", 1, flip=True, distance=0.3),
        _row("regressed", 1, flip=False, distance=0.3),
        _row("hard", 1, flip=False, distance=0.3),
    ]
    details, summary = compare_parent_difficulty(sft, ppo)
    assert summary["difficulty_counts"] == {
        "easy": 1,
        "hard": 1,
        "improved": 1,
        "regressed": 1,
    }
    assert len(details) == 4
    strategy_rows = summarize_strategy_failures(details)
    assert sum(row["count"] for row in strategy_rows) == 8
    assert {row["stage"] for row in strategy_rows} == {"sft", "ppo"}
