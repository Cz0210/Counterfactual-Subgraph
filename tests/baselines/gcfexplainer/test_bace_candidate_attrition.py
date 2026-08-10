from __future__ import annotations

import src.baselines.gcfexplainer_bace_runtime as runtime

from conftest import FakeTeacher, decode_result, ranked_candidates


def test_attrition_records_sanitize_dedup_and_teacher_rejections(
    monkeypatch,
    source_records,
) -> None:
    ranked = ranked_candidates(5)

    def decode(graph, **_kwargs):
        index = graph.candidate_test_index
        if index == 0:
            return decode_result(index, valid=False)
        if index in {1, 2}:
            return decode_result(index, smiles="CC")
        return decode_result(index, smiles=f"C{index}")

    monkeypatch.setattr(runtime, "decode_generated_fullgraph", decode)
    audit, selected, summary = runtime._audit_bace_ranked_candidates(
        ranked=ranked,
        source_records=source_records,
        schema=object(),
        teacher=FakeTeacher({"C4"}),
        target_k=2,
        scan_limit=0,
    )
    assert len(selected) == 1
    assert audit[0]["rejection_reason"] == "generated_valence_sanitize_failed"
    assert audit[2]["rejection_reason"] == "duplicate_canonical_smiles"
    assert audit[3]["rejection_reason"] == "rf_not_target_label_0"
    assert summary["num_sanitize_failed"] == 1
    assert summary["num_canonical_unique"] == 3
    assert summary["num_teacher_counterfactual"] == 1
    assert summary["scan_exhausted"] is True


def test_scan_all_audits_tail_without_changing_frozen_top_k(
    monkeypatch,
    source_records,
) -> None:
    ranked = ranked_candidates(8)
    monkeypatch.setattr(
        runtime,
        "decode_generated_fullgraph",
        lambda graph, **_kwargs: decode_result(
            graph.candidate_test_index,
            smiles="C" * (graph.candidate_test_index + 1),
        ),
    )
    audit, selected, summary = runtime._audit_bace_ranked_candidates(
        ranked=ranked,
        source_records=source_records,
        schema=object(),
        teacher=FakeTeacher({"C" * (index + 1) for index in range(8)}),
        target_k=2,
        scan_limit=0,
        scan_all=True,
        require_connected=True,
    )

    assert len(audit) == 8
    assert [int(row["native_rank"]) for row in selected] == [1, 2]
    assert summary["num_teacher_counterfactual"] == 8
    assert summary["num_retained"] == 2
    assert summary["scan_exhausted"] is True
    assert summary["scan_all"] is True
