from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import pytest

from src.baselines import tastemolnet_gcf_full_resume as t12
from src.baselines import tastemolnet_gcf_production_state as production
from src.baselines.tastemolnet_comrecgc_smoke import _identity_graph_sha256
from tests.baselines.test_tastemolnet_gcf_full_resume import (
    _Adapter,
    _bridge,
    _native_graph,
)


MODEL_SHA = "c" * 64
FEATURE_SCHEMA_SHA = "d" * 64


def _history(tmp_path: Path, *, snapshot=None, index_name: str = "index"):
    return production.T12CompactHistoryJournal(
        root=(tmp_path / "history").resolve(),
        index_root=(tmp_path / index_name).resolve(),
        bounds=production.T12ProductionBounds.pinned(parent_count=2),
        contract_sha256="e" * 64,
        attempt_id=str(uuid.uuid4()) if snapshot is None else snapshot["attempt_id"],
        generation_token="f" * 64,
        resume_snapshot=snapshot,
    )


def _append_compact(history, *, graph_hash: str, embedding_sha256: str) -> None:
    history.append_observation(
        graph_identity_sha256=graph_hash,
        probabilities=(0.1, 0.2, 0.7),
        prediction=2,
        candidate=True,
        valid_fullgraph=True,
        coverage_vector=(1, 0),
        embedding_sha256=embedding_sha256,
        failure_reason="",
        lineage_sha256="a" * 64,
        neurosed_query_sha256="b" * 64,
    )


def test_first_embedding_raw_bytes_and_authorities_are_checkpoint_bound(tmp_path):
    history = _history(tmp_path)
    history.bind_first_seen_embedding_authority(
        model_sha256=MODEL_SHA,
        feature_schema_sha256=FEATURE_SCHEMA_SHA,
    )
    row = np.asarray([1.25, -0.0, 3.5], dtype=np.float32)
    first = history.append_first_embedding(
        graph_identity_sha256="1" * 64,
        dtype=row.dtype.str,
        shape=row.shape,
        raw_bytes=row.tobytes(order="C"),
    )
    _append_compact(
        history,
        graph_hash="1" * 64,
        embedding_sha256=first.embedding_sha256,
    )
    snapshot = history.checkpoint_state()
    store = snapshot["first_seen_embedding_store"]
    assert snapshot["historical_embedding_values_retained"] is True
    assert store["model_sha256"] == MODEL_SHA
    assert store["feature_schema_sha256"] == FEATURE_SCHEMA_SHA
    assert store["raw_bytes_authoritative"] is True
    assert store["record_count"] == 1
    # The checkpoint binds an authenticated external prefix; it never embeds
    # the historical tensor payload itself.
    encoded_snapshot = json.dumps(snapshot, sort_keys=True)
    assert '"raw_bytes":' not in encoded_snapshot
    assert row.tobytes(order="C").hex() not in encoded_snapshot
    history.close()

    reopened = _history(tmp_path, snapshot=snapshot, index_name="reopened-index")
    reopened.bind_first_seen_embedding_authority(
        model_sha256=MODEL_SHA,
        feature_schema_sha256=FEATURE_SCHEMA_SHA,
    )
    restored = reopened.lookup_first_embedding("1" * 64)
    assert restored is not None
    assert restored.dtype == row.dtype.str
    assert restored.shape == row.shape
    assert restored.raw_bytes == row.tobytes(order="C")
    assert restored.raw_sha256 == hashlib.sha256(restored.raw_bytes).hexdigest()
    assert restored.embedding_sha256 == first.embedding_sha256
    with pytest.raises(
        production.TasteT12ProductionStateError,
        match="model/feature authority changed",
    ):
        reopened.bind_first_seen_embedding_authority(
            model_sha256="9" * 64,
            feature_schema_sha256=FEATURE_SCHEMA_SHA,
        )
    reopened.close()


def test_first_embedding_committed_raw_tamper_fails_reload(tmp_path):
    history = _history(tmp_path)
    history.bind_first_seen_embedding_authority(
        model_sha256=MODEL_SHA,
        feature_schema_sha256=FEATURE_SCHEMA_SHA,
    )
    row = np.asarray([1.0, 2.0], dtype=np.float32)
    first = history.append_first_embedding(
        graph_identity_sha256="2" * 64,
        dtype=row.dtype.str,
        shape=row.shape,
        raw_bytes=row.tobytes(order="C"),
    )
    _append_compact(
        history,
        graph_hash="2" * 64,
        embedding_sha256=first.embedding_sha256,
    )
    snapshot = history.checkpoint_state()
    history.close()
    store_snapshot = snapshot["first_seen_embedding_store"]
    path = (
        tmp_path
        / "history"
        / "first-seen-embeddings"
        / store_snapshot["segments"][0]["segment_file"]
    )
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(
        production.TasteT12ProductionStateError,
        match="first embedding (hash chain|committed prefix digest)",
    ):
        _history(tmp_path, snapshot=snapshot, index_name="tamper-index")


def test_evicted_reloaded_bridge_uses_first_raw_bytes_not_gpu_recomputation(
    monkeypatch, tmp_path
):
    collisions = {
        "a": {"canonical_graph": "[C]", "num_nodes": 1, "num_edges": 0},
        "b": {"canonical_graph": "[C][N]", "num_nodes": 2, "num_edges": 1},
    }
    identities = {
        key: SimpleNamespace(
            graph_identity_sha256=_identity_graph_sha256(value),
            collision_payload=lambda value=value: value,
        )
        for key, value in collisions.items()
    }
    monkeypatch.setattr(
        t12,
        "canonical_attributed_graph",
        lambda graph, **_kwargs: identities[graph.token],
    )
    history = _history(tmp_path)
    adapter = _Adapter()
    adapter.metadata = {
        "checkpoint_id": MODEL_SHA,
        "feature_schema": {"schema_sha256": FEATURE_SCHEMA_SHA},
    }
    embedding = {"a": [1.0, 2.0], "b": [3.0, 4.0]}

    def score(values):
        return SimpleNamespace(
            probabilities=np.asarray([[0.1, 0.2, 0.7] for _ in values]),
            graph_embeddings=np.asarray(
                [embedding[value.token] for value in values], dtype=np.float32
            ),
            valid_fullgraphs=tuple(True for _ in values),
            failure_reasons=tuple("" for _ in values),
            identity_graph_payloads=tuple(
                collisions[value.token] for value in values
            ),
            model_graph_payloads=tuple(
                {
                    "schema_version": "test_gine_model_graph_v1",
                    "token": value.token,
                }
                for value in values
            ),
        )

    adapter.score = score
    bridge = _bridge(
        adapter=adapter,
        production_history=history,
        feature_atomic_numbers=(6, 7),
    )
    graphs = {
        "a": _native_graph(
            x=np.asarray([[1.0, 0.0]], dtype=np.float32),
            edge_index=np.empty((2, 0), dtype=np.int64),
            num_nodes=1,
            num_edges=0,
            token="a",
            gcf_origin_index=[0],
            gcf_node_origin=[0],
        ),
        "b": _native_graph(
            x=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
            num_nodes=2,
            num_edges=2,
            token="b",
            gcf_origin_index=[0],
            gcf_node_origin=[0, 1],
        ),
    }
    graph_hashes = {}
    for token in ("a", "b"):
        _parts, rows, _coverage = bridge.call([graphs[token]], {})
        graph_hashes[token] = bridge.calculate_hash(rows[0])
    bridge.vrrw.graph_map = {graph_hashes["a"]: graphs["a"]}
    bridge.vrrw.graph_index_map = {graph_hashes["a"]: 0}
    bridge.vrrw.counterfactual_candidates = [
        {"graph_hash": graph_hashes["a"]}
    ]
    bridge.vrrw.transitions = {}
    audit = bridge.retain_official_live_domain(
        vrrw=bridge.vrrw,
        current_graph_identity=graph_hashes["a"],
    )
    assert audit["evicted_this_boundary"] == 1
    state = bridge.checkpoint_state()
    history.close()

    # Deliberately make the next GPU observation bytewise and numerically
    # different.  Scientific probabilities/coverage remain unchanged.
    embedding["b"] = [30.0, 40.0]
    reopened = _history(tmp_path, snapshot=state["history"], index_name="resume-index")
    restored_bridge = _bridge(
        adapter=adapter,
        production_history=reopened,
        feature_atomic_numbers=(6, 7),
    )
    restored_bridge.restore_checkpoint_state(state)
    _parts, rows, _coverage = restored_bridge.call([graphs["b"]], {})
    assert rows.dtype == np.dtype("<f4")
    assert rows.tolist() == [[3.0, 4.0]]
    assert restored_bridge.calculate_hash(rows[0]) == graph_hashes["b"]
    reopened.close()
