from __future__ import annotations

import json

import pytest

import src.baselines.gcfexplainer_bace_runtime as runtime
from src.baselines.gcfexplainer_mutagenicity_adapter import (
    GCFExplainerEmptyCandidateSetError,
)

from conftest import FakeTeacher, decode_result, ranked_candidates


def test_full_pool_below_twenty_fails_closed_with_complete_audit(
    tmp_path,
    monkeypatch,
) -> None:
    parents = [{"molecule_id": f"BACE_{index:03d}"} for index in range(360)]
    parent_ids = [str(row["molecule_id"]) for row in parents]
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir()
    runtime.write_json(
        summary_dir / "run_manifest.json",
        {"run_complete": True, "generation_parent_ids": parent_ids},
    )
    monkeypatch.setattr(
        runtime,
        "load_bace_gcf_dataset",
        lambda _path: (object(), [], [], parents, {}),
    )
    monkeypatch.setattr(
        runtime,
        "_load_ranked_summary_graphs",
        lambda *_args: ranked_candidates(25),
    )
    monkeypatch.setattr(
        runtime,
        "decode_generated_fullgraph",
        lambda graph, **_kwargs: decode_result(graph.candidate_test_index),
    )
    teacher_path = tmp_path / "bace_teacher.pkl"
    teacher_path.write_bytes(b"teacher")
    output = tmp_path / "export"
    with pytest.raises(
        GCFExplainerEmptyCandidateSetError,
        match="INSUFFICIENT_VALID_NATIVE_CANDIDATES",
    ):
        runtime.export_bace_rf_valid_top20(
            dataset_dir=tmp_path / "dataset",
            summary_dir=summary_dir,
            teacher=FakeTeacher({f"C{index}" for index in range(19)}),
            teacher_path=teacher_path,
            output_dir=output,
            profile="full",
            parent_limit=360,
            top_k=20,
            scan_limit=0,
        )
    audit = json.loads((output / "candidate_attrition_audit.json").read_text())
    assert audit["num_retained"] == 19
    assert audit["scan_exhausted"] is True
    assert (output / "_RUN_FAILED.json").is_file()
    assert not (output / "selected_top20.csv").exists()
