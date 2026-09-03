from __future__ import annotations

import json

import pytest

from src.baselines.globalgce_root_sharding_canary import (
    RootShardingCanaryError,
    load_graph_jsonl,
    merge_ranked_shard_rows,
    plan_disjoint_root_shards,
)


def _row(support: int, root: int, local: int, token: str) -> dict[str, object]:
    return {
        "support": support,
        "root_index": root,
        "local_index": local,
        "pattern_sha256": token * 64,
    }


def test_root_plan_is_an_exact_disjoint_partition() -> None:
    shards = plan_disjoint_root_shards((0, 2, 5, 9), shard_count=3)
    assert shards == ((0, 9), (2,), (5,))
    flattened = [value for shard in shards for value in shard]
    assert sorted(flattened) == [0, 2, 5, 9]
    assert len(flattened) == len(set(flattened))


def test_union_of_per_shard_topk_recovers_stable_global_topk() -> None:
    shard_zero = [_row(10, 0, 0, "a"), _row(8, 0, 1, "b"), _row(1, 0, 2, "c")]
    shard_one = [_row(10, 1, 0, "d"), _row(9, 1, 1, "e"), _row(2, 1, 2, "f")]
    observed = merge_ranked_shard_rows((shard_zero, shard_one), top_k=3)
    assert [(row["support"], row["root_index"]) for row in observed] == [
        (10, 0),
        (10, 1),
        (9, 1),
    ]


def test_shard_merge_fails_closed_on_overlap() -> None:
    duplicate = _row(10, 0, 0, "a")
    with pytest.raises(RootShardingCanaryError, match="duplicate"):
        merge_ranked_shard_rows(([duplicate], [duplicate]), top_k=1)


def test_real_graph_jsonl_loader_preserves_native_order_and_rejects_drift(tmp_path) -> None:
    source = tmp_path / "graphs.jsonl"
    source.write_text(
        json.dumps(
            {
                "graph_id": 7,
                "nodes": [{"id": 0, "label": 6}, {"id": 1, "label": 8}],
                "edges": [{"source": 0, "target": 1, "label": 1}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    graphs, digest = load_graph_jsonl(source)
    assert list(graphs[0].nodes) == [0, 1]
    assert list(graphs[0].edges) == [(0, 1)]
    assert len(digest) == 64

    source.write_text(
        json.dumps(
            {
                "graph_id": 7,
                "nodes": [{"id": 1, "label": 6}, {"id": 0, "label": 8}],
                "edges": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RootShardingCanaryError, match="native insertion order"):
        load_graph_jsonl(source)
