from __future__ import annotations

import csv
from pathlib import Path

from src.baselines.comrecgc.slot_evaluation import (
    build_internal_valid_candidates,
    compute_slot_metrics,
    expand_pair_rows,
    load_official_slots,
    table_row,
)


def _write_slots(path: Path) -> None:
    fields = [
        "official_cluster_rank",
        "cluster_id",
        "candidate_id",
        "repair_success",
        "repaired_smiles",
        "invalid_slot_backfill",
        "rank_compaction",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "official_cluster_rank": 1,
                    "cluster_id": "cluster_1",
                    "candidate_id": "invalid_rank_1",
                    "repair_success": False,
                    "repaired_smiles": "",
                    "invalid_slot_backfill": False,
                    "rank_compaction": False,
                },
                {
                    "official_cluster_rank": 2,
                    "cluster_id": "cluster_2",
                    "candidate_id": "valid_rank_2",
                    "repair_success": True,
                    "repaired_smiles": "CC",
                    "invalid_slot_backfill": False,
                    "rank_compaction": False,
                },
            ]
        )


def _evaluated_rows(
    candidate_slot_id: str = "COMRECGC_OFFICIAL_SLOT_000002",
) -> list[dict[str, object]]:
    return [
        {
            "parent_id": "p1",
            "candidate_id": candidate_slot_id,
            "distance": 0.1,
            "match": True,
            "delete_valid": True,
            "pred_before": 1,
            "pred_after": 0,
            "teacher_strict_flip": True,
            "cf_drop": 0.8,
        },
        {
            "parent_id": "p2",
            "candidate_id": candidate_slot_id,
            "distance": 0.3,
            "match": True,
            "delete_valid": True,
            "pred_before": 1,
            "pred_after": 0,
            "teacher_strict_flip": True,
            "cf_drop": 0.7,
        },
    ]


def test_invalid_slot_is_not_compacted_or_sent_to_shared_evaluator(tmp_path: Path) -> None:
    path = tmp_path / "medoid_validity.csv"
    _write_slots(path)
    slots = load_official_slots(path)
    internal = build_internal_valid_candidates(slots)

    assert [row["candidate_id"] for row in internal] == [
        "COMRECGC_OFFICIAL_SLOT_000002"
    ]
    assert [row["source_candidate_id"] for row in internal] == ["valid_rank_2"]
    assert internal[0]["rank"] == 1
    assert internal[0]["native_rank"] == 2
    assert [row["official_cluster_rank"] for row in slots] == [1, 2]


def test_no_cross_rank_backfill_in_prefix_metrics(tmp_path: Path) -> None:
    path = tmp_path / "medoid_validity.csv"
    _write_slots(path)
    slots = load_official_slots(path)
    pairs = expand_pair_rows(
        parent_ids=["p1", "p2"],
        slots=slots,
        evaluated_rows=_evaluated_rows(),
    )
    invalid = [row for row in pairs if row["source_candidate_id"] == "invalid_rank_1"]
    assert len(invalid) == 2
    assert all(row["error"] == "candidate_not_sent_to_rf_or_wnode" for row in invalid)

    prefixes, thresholds, parent_best = compute_slot_metrics(
        pair_rows=pairs,
        slots=slots,
        parent_ids=["p1", "p2"],
        thresholds=[0.05, 0.2, 0.4],
        theta_star=0.2,
        cost_cap=0.4,
        max_k=20,
    )
    assert len(prefixes) == 20
    assert prefixes[0]["k"] == 1
    assert prefixes[0]["valid_k"] == 0
    assert prefixes[0]["close_cf_coverage"] == 0.0
    assert prefixes[0]["conditional_median_cost"] is None
    assert prefixes[1]["k"] == 2
    assert prefixes[1]["valid_k"] == 1
    assert prefixes[1]["close_cf_coverage"] == 0.5
    assert prefixes[1]["conditional_median_cost"] == 0.2
    assert all(row["close_cf_coverage"] == 0.5 for row in prefixes[1:])
    assert len([row for row in thresholds if row["k"] == 20]) == 3
    assert len(parent_best) == 40
    table = table_row(prefixes[9], theta_star=0.2)
    assert table["requested_k"] == 10
    assert table["valid_k"] == 1
    assert table["invalid_slot_backfill"] is False
    assert table["rank_compaction"] is False


def test_empty_scientific_output_keeps_cost_na() -> None:
    slots = [
        {
            "official_cluster_rank": 1,
            "cluster_id": "cluster_1",
            "candidate_id": "invalid",
            "candidate_slot_id": "COMRECGC_OFFICIAL_SLOT_000001",
            "source_candidate_id": "invalid",
            "candidate_slot_valid": False,
            "slot_rejection_reason": "repair_invalid",
        }
    ]
    pairs = expand_pair_rows(parent_ids=["p1"], slots=slots, evaluated_rows=[])
    prefixes, _thresholds, _parents = compute_slot_metrics(
        pair_rows=pairs,
        slots=slots,
        parent_ids=["p1"],
        thresholds=[0.1],
        theta_star=0.1,
        cost_cap=0.2,
        max_k=20,
    )
    assert prefixes[-1]["close_cf_coverage"] == 0.0
    assert prefixes[-1]["conditional_mean_cost"] is None
    assert prefixes[-1]["conditional_median_cost"] is None


def test_slot_adapter_does_not_implement_teacher_or_distance() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "src/baselines/comrecgc/slot_evaluation.py"
    ).read_text(encoding="utf-8")
    assert "TeacherSemanticScorer" not in source
    assert "MolCLRNodeWassersteinDistance" not in source
    assert "summarize_wnode_thresholds" in source


def test_reused_source_medoid_is_preserved_as_distinct_official_slots(
    tmp_path: Path,
) -> None:
    path = tmp_path / "medoid_validity.csv"
    fields = [
        "official_cluster_rank",
        "cluster_id",
        "candidate_id",
        "repair_success",
        "repaired_smiles",
        "invalid_slot_backfill",
        "rank_compaction",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank in range(1, 4):
            writer.writerow(
                {
                    "official_cluster_rank": rank,
                    "cluster_id": f"cluster_{rank}",
                    "candidate_id": "shared_medoid",
                    "repair_success": True,
                    "repaired_smiles": "CC",
                    "invalid_slot_backfill": False,
                    "rank_compaction": False,
                }
            )

    slots = load_official_slots(path)
    internal = build_internal_valid_candidates(slots)

    assert [row["candidate_id"] for row in slots] == ["shared_medoid"] * 3
    assert [row["source_candidate_id"] for row in slots] == ["shared_medoid"] * 3
    assert [row["official_cluster_rank"] for row in slots] == [1, 2, 3]
    assert len({row["candidate_slot_id"] for row in slots}) == 3
    assert len(internal) == 1
    assert internal[0]["source_candidate_id"] == "shared_medoid"
    assert internal[0]["official_rank_slots"] == [1, 2, 3]
    assert internal[0]["evaluation_compute_reuse_count"] == 3
    assert all(row["source_candidate_reused_across_slots"] for row in slots)


def test_reused_source_medoid_pair_rows_do_not_overwrite_rank_slots(
    tmp_path: Path,
) -> None:
    path = tmp_path / "medoid_validity.csv"
    fields = [
        "official_cluster_rank",
        "cluster_id",
        "candidate_id",
        "repair_success",
        "repaired_smiles",
        "invalid_slot_backfill",
        "rank_compaction",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank in range(1, 4):
            writer.writerow(
                {
                    "official_cluster_rank": rank,
                    "cluster_id": f"cluster_{rank}",
                    "candidate_id": "shared_medoid",
                    "repair_success": True,
                    "repaired_smiles": "CC",
                    "invalid_slot_backfill": False,
                    "rank_compaction": False,
                }
            )
    slots = load_official_slots(path)
    evaluated = [
        {
            "parent_id": "p1",
            "candidate_id": slots[0]["evaluation_candidate_id"],
            "distance": 0.1,
            "pred_before": 1,
            "pred_after": 0,
        }
    ]
    pairs = expand_pair_rows(parent_ids=["p1"], slots=slots, evaluated_rows=evaluated)
    prefixes, _thresholds, _parents = compute_slot_metrics(
        pair_rows=pairs,
        slots=slots,
        parent_ids=["p1"],
        thresholds=[0.2],
        theta_star=0.2,
        cost_cap=0.2,
        max_k=20,
    )

    assert len(pairs) == 3
    assert len({row["candidate_id"] for row in pairs}) == 3
    assert [row["source_candidate_id"] for row in pairs] == ["shared_medoid"] * 3
    assert len({row["evaluation_candidate_id"] for row in pairs}) == 1
    assert [row["official_cluster_rank"] for row in pairs] == [1, 2, 3]
    assert prefixes[0]["valid_k"] == 1
    assert prefixes[1]["valid_k"] == 2
    assert prefixes[2]["valid_k"] == 3


def test_duplicate_cluster_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "medoid_validity.csv"
    fields = [
        "official_cluster_rank",
        "cluster_id",
        "candidate_id",
        "repair_success",
        "repaired_smiles",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "official_cluster_rank": 1,
                    "cluster_id": "same_cluster",
                    "candidate_id": "candidate_1",
                    "repair_success": True,
                    "repaired_smiles": "CC",
                },
                {
                    "official_cluster_rank": 2,
                    "cluster_id": "same_cluster",
                    "candidate_id": "candidate_2",
                    "repair_success": True,
                    "repaired_smiles": "CN",
                },
            ]
        )

    try:
        load_official_slots(path)
    except ValueError as exc:
        assert "cluster IDs must be unique" in str(exc)
    else:
        raise AssertionError("duplicate cluster IDs must be rejected")
