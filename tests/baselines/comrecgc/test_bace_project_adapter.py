from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.baselines.comrecgc.run_slot_unified_eval import (
    _apply_bace_connected_candidate_gate,
    _bace_paper_rows,
    _bace_table_row,
)
from src.baselines.comrecgc.project_dataset import load_bace_generation_bundle
from src.chem.hard_deletion import CONNECTED_ACTION_SEMANTICS


def test_bace_generation_bundle_uses_only_frozen_generation_cohort(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "dataset_summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "generation_source_graphs.pt").write_bytes(b"generation")
    schema = SimpleNamespace(
        node_feature_dim=2,
        feature_atomic_numbers=(6, 8),
    )
    records = [
        {
            "molecule_id": "bace_train_2",
            "canonical_smiles": "CO",
            "source_graph_hash": "hash-2",
        },
        {
            "molecule_id": "bace_train_1",
            "canonical_smiles": "CC",
            "source_graph_hash": "hash-1",
        },
    ]
    summary = {"generation_source_cohort_hash": "cohort"}

    import src.baselines.gcfexplainer_bace_adapter as bace_adapter
    import src.baselines.gcfexplainer_mutagenicity_adapter as mut_adapter

    monkeypatch.setattr(
        bace_adapter,
        "load_bace_gcf_dataset",
        lambda _root: (schema, [], [], records, summary),
    )

    def fake_record_to_pyg(record, *, origin_index):
        return SimpleNamespace(gcf_node_origin=[origin_index])

    monkeypatch.setattr(mut_adapter, "record_to_pyg", fake_record_to_pyg)
    bundle = load_bace_generation_bundle(dataset_dir=tmp_path, parent_limit=2)

    assert bundle.dataset == "BACE"
    assert bundle.parent_ids == ["bace_train_1", "bace_train_2"]
    assert bundle.source_label == 1
    assert bundle.target_label == 0
    assert bundle.generation_source_parent_rows == 2
    assert [graph.comrecgc_source_smiles for graph in bundle.graphs] == ["CC", "CO"]


def test_bace_connected_gate_invalidates_without_rank_compaction() -> None:
    slots = [
        {
            "official_cluster_rank": 1,
            "candidate_slot_valid": True,
            "repaired_smiles": "CC",
        },
        {
            "official_cluster_rank": 2,
            "candidate_slot_valid": True,
            "repaired_smiles": "C.C",
        },
        {
            "official_cluster_rank": 3,
            "candidate_slot_valid": False,
            "repaired_smiles": "",
        },
    ]
    result = _apply_bace_connected_candidate_gate(slots)

    assert [row["official_cluster_rank"] for row in slots] == [1, 2, 3]
    assert [row["candidate_slot_valid"] for row in slots] == [True, False, False]
    assert slots[1]["slot_status"] == "CONNECTED_PROTOCOL_INVALID"
    assert result["connected_repaired_slot_count"] == 1
    assert result["disconnected_repaired_slot_count"] == 1
    assert result["disconnected_output_used_count"] == 0
    assert all(row["connected_action_semantics"] == CONNECTED_ACTION_SEMANTICS for row in slots)


def test_bace_comrecgc_rows_keep_common_paper_schema() -> None:
    prefixes = [
        {
            "k": k,
            "close_cf_coverage": k / 20,
            "conditional_median_cost": 0.01,
            "num_close_cf_covered": k,
            "avg_cf_drop_among_covered": 0.4,
        }
        for k in range(1, 21)
    ]
    thresholds = [
        {"threshold": value, "close_cf_coverage": value}
        for value in (0.01, 0.02, 0.03)
    ]
    figure3, figure4, by_k = _bace_paper_rows(
        prefixes=prefixes,
        figure4=thresholds,
    )
    table = _bace_table_row(by_k[10])

    assert tuple(figure3[0]) == ("method", "k", "coverage", "cost")
    assert tuple(figure4[0]) == ("method", "threshold", "coverage")
    assert tuple(table) == ("method", "k", "coverage", "cost", "flip_rate", "cf_drop")
    assert [row["k"] for row in figure3] == list(range(1, 21))
