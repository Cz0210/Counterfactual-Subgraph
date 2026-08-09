from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.audit_bace_hard_deletion_semantics import (
    audit_existing_ours_connectivity,
    audit_gcf_candidates,
)


def test_gcf_top20_connected_candidate_audit_preserves_native_rank(
    tmp_path: Path,
) -> None:
    candidates = tmp_path / "selected_top20.csv"
    with candidates.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "candidate_id", "candidate_smiles", "rf_pred"],
        )
        writer.writeheader()
        for rank in range(1, 21):
            writer.writerow(
                {
                    "rank": rank,
                    "candidate_id": f"g{rank}",
                    "candidate_smiles": "C" * rank,
                    "rf_pred": 0,
                }
            )
    audit = audit_gcf_candidates(
        candidates, tmp_path / "audit.json", expected_target_label=0
    )
    assert audit["passed"] is True
    assert audit["candidate_count"] == 20
    assert audit["all_candidates_connected"] is True
    assert audit["native_rank_preserved"] is True
    assert audit["all_candidates_teacher_counterfactual"] is True


def test_gcf_candidate_audit_rejects_wrong_teacher_target(tmp_path: Path) -> None:
    candidates = tmp_path / "selected_top20.csv"
    with candidates.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "candidate_id", "candidate_smiles", "rf_pred"],
        )
        writer.writeheader()
        for rank in range(1, 21):
            writer.writerow(
                {
                    "rank": rank,
                    "candidate_id": f"g{rank}",
                    "candidate_smiles": "C" * rank,
                    "rf_pred": int(rank == 20),
                }
            )

    try:
        audit_gcf_candidates(
            candidates, tmp_path / "audit.json", expected_target_label=0
        )
    except ValueError as error:
        assert "failed closed" in str(error)
    else:
        raise AssertionError("A non-counterfactual GCF candidate must fail the audit")


def test_existing_connectivity_audit_reads_frozen_jsonl(tmp_path: Path) -> None:
    run_dir = tmp_path / "mut-run"
    run_dir.mkdir()
    rows = [
        {
            "parent_id": "p1",
            "candidate_id": "c1",
            "match_index": 0,
            "residual_smiles": "CC.C",
            "teacher_strict_flip": True,
            "wnode_distance": 0.1,
            "cf_drop": 0.2,
        },
        {
            "parent_id": "p1",
            "candidate_id": "c1",
            "match_index": 1,
            "residual_smiles": "CC",
            "teacher_strict_flip": True,
            "wnode_distance": 0.2,
            "cf_drop": 0.3,
        },
    ]
    (run_dir / "match_instances.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    audit = audit_existing_ours_connectivity(run_dir, tmp_path / "impact.json")
    assert audit["winning_row_count"] == 1
    assert audit["disconnected_winning_count"] == 1
    assert audit["reevaluation_required"] is True
