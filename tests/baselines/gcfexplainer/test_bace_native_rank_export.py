from __future__ import annotations

from types import SimpleNamespace

import src.baselines.gcfexplainer_bace_runtime as runtime

from conftest import FakeTeacher, decode_result, ranked_candidates


def _reference_greedy(covering, covered_by):
    covering = {key: set(value) for key, value in covering.items()}
    covered = set()
    result = []
    for _rank in range(len(covering)):
        candidate, new = max(covering.items(), key=lambda item: len(item[1]))
        covered.update(new)
        covering.pop(candidate)
        for parent in new:
            for other in covered_by[parent] - {candidate}:
                if other in covering:
                    covering[other].remove(parent)
        result.append((candidate, len(covered)))
    return result


def test_zero_gain_fast_forward_is_exactly_official_greedy() -> None:
    covering = {
        0: {0, 1},
        1: {1, 2},
        2: {2},
        3: set(),
        4: set(),
    }
    covered_by = {
        0: {0},
        1: {0, 1},
        2: {1, 2},
    }
    assert runtime.official_greedy_coverage_order(covering, covered_by) == (
        _reference_greedy(covering, covered_by)
    )


def test_zero_candidate_limit_keeps_complete_model_cf_frequency_sequence() -> None:
    graphs = {
        index: SimpleNamespace(
            x=[[1.0, 0.0]],
            edge_index=[[], []],
            num_nodes=1,
        )
        for index in range(5)
    }
    candidates = [
        {
            "graph_hash": index,
            "frequency": 10 - index,
            "importance_parts": [0.6 if index != 2 else 0.4],
        }
        for index in range(5)
    ]
    selected, metadata, available = runtime._model_counterfactual_candidates(
        candidates,
        graphs,
        limit=None,
    )
    assert len(selected) == 4
    assert available == 4
    assert [row["graph_hash"] for row in metadata] == ["0", "1", "3", "4"]
    assert [row["frequency"] for row in metadata] == [10, 9, 7, 6]


def test_export_scans_past_invalid_prefix_without_reordering(
    monkeypatch,
    source_records,
) -> None:
    ranked = ranked_candidates(60)
    target_indices = set(range(40, 60))
    teacher = FakeTeacher({f"C{index}" for index in target_indices})
    monkeypatch.setattr(
        runtime,
        "decode_generated_fullgraph",
        lambda graph, **_kwargs: decode_result(
            graph.candidate_test_index,
            valid=graph.candidate_test_index >= 20,
        ),
    )
    audit, selected, summary = runtime._audit_bace_ranked_candidates(
        ranked=ranked,
        source_records=source_records,
        schema=object(),
        teacher=teacher,
        target_k=20,
        scan_limit=0,
    )
    assert [int(row["native_rank"]) for row in selected] == list(range(41, 61))
    assert [int(row["native_rank"]) for row in audit] == list(range(1, 61))
    assert summary["num_retained"] == 20
    assert summary["native_order_preserved"] is True
    assert summary["candidate_repair_performed"] is False
    assert summary["rank_backfill_performed"] is False
