from __future__ import annotations

from pathlib import Path

import pytest

from scripts.baselines.comrecgc.gate_run import run_gate
from src.baselines.comrecgc.contracts import ordered_ids_sha256, sha256_file
from src.baselines.comrecgc.exporter import (
    _sync_generated_node_lineage,
    build_frozen_candidate_manifest,
    export_gate_failure,
)


def test_frozen_manifest_records_file_and_order_without_reranking(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_top20.csv"
    csv_path.write_text("rank,candidate_id,smiles\n1,z,CC\n2,a,CN\n", encoding="utf-8")
    selected = [{"candidate_id": "z"}, {"candidate_id": "a"}]
    payload = build_frozen_candidate_manifest(
        dataset="aids",
        selected=selected,
        csv_path=csv_path,
    )
    assert payload["selected_candidate_ids"] == ["z", "a"]
    assert payload["selected_candidate_order_sha256"] == ordered_ids_sha256(["z", "a"])
    assert payload["file_inventory"]["selected_top20.csv"]["sha256"] == sha256_file(csv_path)
    assert payload["selection_performed_in_eval"] is False


def test_smoke_export_requires_at_least_one_rf_scored_candidate() -> None:
    failure = export_gate_failure(
        {"rf_scored_count": 0, "selected_count": 0},
        require_top_k=False,
        top_k=20,
    )
    assert failure is not None
    assert failure[0] == "NoRFScoredCandidates"
    assert export_gate_failure(
        {"rf_scored_count": 1, "selected_count": 0},
        require_top_k=False,
        top_k=20,
    ) is None


def test_full_export_still_requires_top_k() -> None:
    failure = export_gate_failure(
        {"rf_scored_count": 19, "selected_count": 19},
        require_top_k=True,
        top_k=20,
    )
    assert failure is not None
    assert failure[0] == "InsufficientStrictFlipCandidates"


def test_generated_lineage_replaces_stale_source_codec_lineage() -> None:
    class GeneratedGraph:
        gcf_node_origin = [0, 1, 2]
        comrecgc_node_origin = [0, 2]

    graph = GeneratedGraph()
    _sync_generated_node_lineage(graph)
    assert graph.gcf_node_origin == [0, 2]


def test_generated_lineage_is_required() -> None:
    class GeneratedGraph:
        gcf_node_origin = [0]

    with pytest.raises(ValueError, match="generated_missing_source_lineage"):
        _sync_generated_node_lineage(GeneratedGraph())


def test_gate_writes_structured_failure_when_upstream_artifacts_are_missing(
    tmp_path: Path,
) -> None:
    gate = run_gate(root=tmp_path, dataset="mutagenicity", mode="smoke")
    assert gate["audit_passed"] is False
    assert gate["missing_artifacts"]
    assert "evaluation_manifest_present" in gate["failed_hard_checks"]
    assert (tmp_path / "gate.json").is_file()
    assert (tmp_path / "_GATE_FAILED.json").is_file()
    assert not (tmp_path / "_GATE_PASS.json").exists()
