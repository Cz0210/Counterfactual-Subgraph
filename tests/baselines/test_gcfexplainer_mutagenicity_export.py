from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.baselines.gcfexplainer_mutagenicity_runtime as runtime
from src.baselines.gcfexplainer_mutagenicity_adapter import (
    GeneratedDecodeResult,
    GCFExplainerEmptyCandidateSetError,
    GCFExplainerMutagenicityCodecError,
)


class _FakeTeacher:
    available = True

    def __init__(self, target_smiles: set[str]) -> None:
        self.target_smiles = set(target_smiles)

    def score_smiles(self, smiles: str, label: int | None = None, **_kwargs):
        pred = 0 if smiles in self.target_smiles else 1
        return {
            "teacher_result_ok": True,
            "teacher_label": pred,
            "teacher_prob": 0.9 if label == pred else 0.1,
        }


def _decode_result(index: int, *, valid: bool) -> GeneratedDecodeResult:
    if not valid:
        return GeneratedDecodeResult(
            decode_ok=False,
            canonical_smiles="",
            raw_smiles="",
            failure_reason="generated_valence_sanitize_failed",
            projected_new_edge_count=0,
            retained_edge_count=0,
            removed_source_edge_count=0,
            inherited_atom_state_count=0,
            reset_atom_state_count=0,
            source_parent_id="P0000",
        )
    return GeneratedDecodeResult(
        decode_ok=True,
        canonical_smiles=f"candidate_{index:04d}",
        raw_smiles=f"candidate_{index:04d}",
        failure_reason="",
        projected_new_edge_count=1,
        retained_edge_count=2,
        removed_source_edge_count=0,
        inherited_atom_state_count=1,
        reset_atom_state_count=1,
        source_parent_id="P0000",
    )


def _prepare_export_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile: str,
    candidate_count: int,
    invalid_count: int,
    target_indices: set[int],
) -> tuple[dict[str, object], _FakeTeacher, Path, Path]:
    parent_count = 64 if profile == "smoke" else 1448
    source_records = [{"molecule_id": f"P{index:04d}"} for index in range(parent_count)]
    parent_ids = [str(row["molecule_id"]) for row in source_records]
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir()
    summary_manifest = {
        "run_complete": True,
        "profile": profile,
        "generation_parent_ids": parent_ids,
        "generation_source_cohort_hash": runtime.cohort_hash(source_records),
        "generation_source_parent_rows": 1448,
        "summary_parent_count": parent_count,
    }
    runtime.write_json(summary_dir / "run_manifest.json", summary_manifest)
    graphs = [
        SimpleNamespace(
            export_test_index=index,
            num_nodes=2,
            x=[[1.0], [1.0]],
            edge_index=[[0, 1], [1, 0]],
            gcf_origin_index=[0],
        )
        for index in range(candidate_count)
    ]
    native_rows = [
        {
            "candidate_id": f"NATIVE_{index + 1:04d}",
            "native_rank": index + 1,
            "source_graph_index": 0,
        }
        for index in range(candidate_count)
    ]
    payload = {"selected_graphs": graphs, "selected_records": native_rows}
    monkeypatch.setattr(
        runtime,
        "load_dataset_artifacts",
        lambda _path: (object(), [], [], source_records, {}),
    )
    monkeypatch.setattr(runtime, "_torch_load_compat", lambda _path: payload)
    monkeypatch.setattr(
        runtime,
        "decode_generated_fullgraph",
        lambda graph, **_kwargs: _decode_result(
            graph.export_test_index,
            valid=graph.export_test_index >= invalid_count,
        ),
    )
    teacher = _FakeTeacher(
        {f"candidate_{index:04d}" for index in target_indices}
    )
    teacher_path = tmp_path / "mutagenicity_rf_model.pkl"
    teacher_path.write_bytes(b"fake-teacher")
    output_dir = tmp_path / "export"
    return summary_manifest, teacher, teacher_path, output_dir


def _run_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile: str,
    candidate_count: int,
    invalid_count: int,
    target_indices: set[int],
) -> tuple[dict[str, object], Path]:
    _manifest, teacher, teacher_path, output_dir = _prepare_export_case(
        tmp_path,
        monkeypatch,
        profile=profile,
        candidate_count=candidate_count,
        invalid_count=invalid_count,
        target_indices=target_indices,
    )
    result = runtime.export_rf_valid_native_top20(
        dataset_dir=tmp_path / "dataset",
        summary_dir=tmp_path / "summary",
        teacher=teacher,
        teacher_path=teacher_path,
        output_dir=output_dir,
        profile=profile,
        parent_limit=64 if profile == "smoke" else 1448,
        top_k=20,
    )
    return result, output_dir


def test_smoke_audits_zero_yield_after_rf_without_requiring_top20(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, output_dir = _run_export(
        tmp_path,
        monkeypatch,
        profile="smoke",
        candidate_count=100,
        invalid_count=80,
        target_indices=set(),
    )
    summary = json.loads((output_dir / "filter_summary.json").read_text())
    audit = [json.loads(line) for line in (output_dir / "candidate_filter_audit.jsonl").read_text().splitlines()]
    assert len(audit) == 100
    assert summary["native_rank_input_count"] == 100
    assert summary["rdkit_sanitize_failed_count"] == 80
    assert summary["rf_scored_count"] == 20
    assert summary["rf_target_count"] == 0
    assert summary["selected_count"] == 0
    assert result["candidate_yield_gate_passed"] is False
    assert result["full_result_ready"] is False
    assert (output_dir / "_SMOKE_AUDIT_COMPLETE.json").is_file()
    assert not (output_dir / "_RUN_COMPLETE.json").exists()
    assert not (output_dir / "selected_top20.csv").exists()
    assert audit[0]["native_rank"] == 1
    assert audit[0]["rejection_stage"] == "rdkit_sanitize"
    assert audit[0]["sanitize_error_type"] == "generated_valence_sanitize_failed"
    assert audit[80]["native_rank"] == 81
    assert audit[80]["rf_inference_ok"] is True
    assert audit[80]["rejection_stage"] == "rf_target_filter"


def test_smoke_all_rdkit_invalid_writes_audit_then_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest, teacher, teacher_path, output_dir = _prepare_export_case(
        tmp_path,
        monkeypatch,
        profile="smoke",
        candidate_count=100,
        invalid_count=100,
        target_indices=set(),
    )
    with pytest.raises(GCFExplainerMutagenicityCodecError, match="no RF-scorable"):
        runtime.export_rf_valid_native_top20(
            dataset_dir=tmp_path / "dataset",
            summary_dir=tmp_path / "summary",
            teacher=teacher,
            teacher_path=teacher_path,
            output_dir=output_dir,
            profile="smoke",
            parent_limit=64,
        )
    summary = json.loads((output_dir / "filter_summary.json").read_text())
    failure = json.loads((output_dir / "failure_summary.json").read_text())
    assert summary["audit_row_count"] == 100
    assert summary["rdkit_sanitize_ok_count"] == 0
    assert summary["rf_scored_count"] == 0
    assert failure["field"] == "rf_scored_count"
    assert failure["actual"] == 0
    assert failure["expected_min"] == 1
    assert (output_dir / "_RUN_FAILED.json").is_file()
    assert not (output_dir / "_SMOKE_AUDIT_COMPLETE.json").exists()


def test_full_with_19_target_candidates_writes_audit_then_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest, teacher, teacher_path, output_dir = _prepare_export_case(
        tmp_path,
        monkeypatch,
        profile="full",
        candidate_count=20,
        invalid_count=0,
        target_indices=set(range(19)),
    )
    with pytest.raises(GCFExplainerEmptyCandidateSetError, match="selected_count=19"):
        runtime.export_rf_valid_native_top20(
            dataset_dir=tmp_path / "dataset",
            summary_dir=tmp_path / "summary",
            teacher=teacher,
            teacher_path=teacher_path,
            output_dir=output_dir,
            profile="full",
            parent_limit=1448,
        )
    summary = json.loads((output_dir / "filter_summary.json").read_text())
    assert summary["selected_count"] == 19
    assert summary["native_order_preserved"] is True
    assert (output_dir / "_RUN_FAILED.json").is_file()
    assert not (output_dir / "selected_top20.csv").exists()


def test_full_with_20_target_candidates_freezes_native_rank_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, output_dir = _run_export(
        tmp_path,
        monkeypatch,
        profile="full",
        candidate_count=25,
        invalid_count=0,
        target_indices=set(range(25)),
    )
    with (output_dir / "selected_top20.csv").open(newline="") as handle:
        selected = list(csv.DictReader(handle))
    audit = [json.loads(line) for line in (output_dir / "candidate_filter_audit.jsonl").read_text().splitlines()]
    assert [int(row["native_rank"]) for row in selected] == list(range(1, 21))
    assert len(audit) == 25
    assert sum(bool(row["selected"]) for row in audit) == 20
    assert audit[20]["rejection_stage"] == "selected"
    assert audit[20]["rejection_reason"] == "beyond_requested_top_k"
    assert result["rf_reranking_performed"] is False
    assert result["wnode_reranking_performed"] is False
    assert result["full_result_ready"] is True
    assert (output_dir / "_RUN_COMPLETE.json").is_file()
    assert not (output_dir / "_SMOKE_AUDIT_COMPLETE.json").exists()


def test_filter_audit_preserves_native_rank_and_rejection_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _result, output_dir = _run_export(
        tmp_path,
        monkeypatch,
        profile="smoke",
        candidate_count=3,
        invalid_count=1,
        target_indices={2},
    )
    audit = [json.loads(line) for line in (output_dir / "candidate_filter_audit.jsonl").read_text().splitlines()]
    assert [row["native_rank"] for row in audit] == [1, 2, 3]
    assert audit[0]["rejection_reason"] == "generated_valence_sanitize_failed"
    assert audit[1]["rejection_reason"] == "rf_not_target_label_0"
    assert audit[2]["selected"] is True
    assert all(row["rejection_stage"] for row in audit)

