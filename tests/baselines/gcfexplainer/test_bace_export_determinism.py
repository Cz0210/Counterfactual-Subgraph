from __future__ import annotations

import src.baselines.gcfexplainer_bace_runtime as runtime

from conftest import FakeTeacher, decode_result, fake_graph, ranked_candidates


def test_candidate_export_is_deterministic_and_keeps_native_slots(
    monkeypatch,
    source_records,
) -> None:
    ranked = ranked_candidates(30)
    monkeypatch.setattr(
        runtime,
        "decode_generated_fullgraph",
        lambda graph, **_kwargs: decode_result(graph.candidate_test_index),
    )
    teacher = FakeTeacher({f"C{index}" for index in range(5, 30)})
    first = runtime._audit_bace_ranked_candidates(
        ranked=ranked,
        source_records=source_records,
        schema=object(),
        teacher=teacher,
        target_k=20,
        scan_limit=0,
    )
    second = runtime._audit_bace_ranked_candidates(
        ranked=ranked,
        source_records=source_records,
        schema=object(),
        teacher=teacher,
        target_k=20,
        scan_limit=0,
    )
    assert first[1] == second[1]
    assert [int(row["native_rank"]) for row in first[1]] == list(range(6, 26))
    assert first[2]["candidate_copy_performed"] is False
    assert first[2]["rf_reranking_performed"] is False
    assert first[2]["wnode_reranking_performed"] is False


def test_large_summary_graph_hash_references_rehydrate_in_native_order(
    tmp_path,
    monkeypatch,
) -> None:
    summary = tmp_path / "summary"
    summary.mkdir()
    source = tmp_path / "counterfactuals.pt"
    source.write_bytes(b"immutable-vrrw")
    rows = [
        {"candidate_id": f"N{rank}", "native_rank": rank}
        for rank in range(1, 4)
    ]
    selected_payload = {
        "selected_graphs": [],
        "selected_graph_hashes": ["11", "22", "33"],
        "selected_records": rows,
        "source_counterfactuals_path": str(source),
        "source_counterfactuals_sha256": runtime.sha256_file(source),
    }
    graphs = {11: fake_graph(1), 22: fake_graph(2), 33: fake_graph(3)}

    def load(path):
        return (
            selected_payload
            if path.name == "selected_counterfactual_graphs.pt"
            else {"graph_map": graphs}
        )

    monkeypatch.setattr(runtime, "_torch_load_compat", load)
    resolved = runtime._load_ranked_summary_graphs(
        summary,
        {
            "counterfactuals_path": str(source),
            "counterfactuals_sha256": runtime.sha256_file(source),
        },
    )
    assert [int(row["native_rank"]) for row, _graph in resolved] == [1, 2, 3]
    assert [graph.candidate_test_index for _row, graph in resolved] == [1, 2, 3]
