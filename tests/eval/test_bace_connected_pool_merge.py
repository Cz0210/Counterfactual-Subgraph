from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.merge_bace_ours_candidate_pools_v2 import (
    connected_source_residual_status,
    main,
)


def test_connected_source_residual_status_rejects_disconnected() -> None:
    valid, reason = connected_source_residual_status(
        {"parent_without_fragment_smiles": "CC.C"}
    )
    assert valid is False
    assert reason == "source_residual_disconnected"
    assert connected_source_residual_status(
        {"parent_without_fragment_smiles": "CCO"}
    ) == (True, "connected_sanitized_residual")


def test_merge_filters_disconnected_source_residual(
    tmp_path: Path, monkeypatch
) -> None:
    base = tmp_path / "base.jsonl"
    rows = [
        {
            "molecule_id": "p1",
            "final_fragment": "C",
            "parent_without_fragment_smiles": "CC.C",
            "cf_flip": True,
            "cf_drop": 0.9,
            "atom_ratio": 0.1,
        },
        {
            "molecule_id": "p1",
            "final_fragment": "N",
            "parent_without_fragment_smiles": "CCO",
            "cf_flip": True,
            "cf_drop": 0.8,
            "atom_ratio": 0.2,
        },
    ]
    base.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    train_ids = tmp_path / "train.txt"
    test_ids = tmp_path / "test.txt"
    train_ids.write_text("p1\n", encoding="utf-8")
    test_ids.write_text("p2\n", encoding="utf-8")
    output = tmp_path / "merged"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge",
            "--base-pool",
            str(base),
            "--train-parent-ids",
            str(train_ids),
            "--test-parent-ids",
            str(test_ids),
            "--output-dir",
            str(output),
            "--require-connected-source-residual",
        ],
    )
    assert main() == 0
    retained = [
        json.loads(line)
        for line in (output / "candidate_pool.jsonl").read_text().splitlines()
    ]
    audit = json.loads((output / "candidate_pool_audit.json").read_text())
    assert [row["final_fragment"] for row in retained] == ["N"]
    assert audit["source_residual_filtered_count"] == 1
    assert audit["source_residual_failure_counts"] == {
        "source_residual_disconnected": 1
    }
    assert audit["all_retained_source_residuals_connected"] is True
