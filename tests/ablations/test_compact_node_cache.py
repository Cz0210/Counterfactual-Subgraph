from pathlib import Path
import sqlite3

import numpy as np
import pytest

from src.ablations.llm.compact_node_cache import CompactNodeCache
from src.eval.molclr_node_embeddings import MolCLRNodeEmbedder, MolCLRNodeEmbeddingStats, atom_numbers_for_smiles
from src.eval.node_wasserstein_distance import MolCLRNodeWassersteinConfig, MolCLRNodeWassersteinDistance
from src.ablations.llm.compact_node_cache import install_compact_node_cache


def embedder(path):
    obj = object.__new__(MolCLRNodeEmbedder)
    obj.node_emb_cache_dir = path
    path.mkdir()
    obj.stats = MolCLRNodeEmbeddingStats()
    obj.checkpoint_identity = "tiny-frozen-weight-identity"
    obj.encoder_type = "gin"
    obj.architecture_identity = "tiny-architecture"
    obj.legacy_cache_path = lambda smiles: path / "unused-legacy-cache.npz"
    obj.cache_payload = lambda smiles: {"canonical": smiles, "encoder": "tiny-frozen", "schema": "same"}
    calls = []
    def compute(smiles):
        calls.append(smiles)
        return np.arange(len(atom_numbers_for_smiles(smiles)) * 4, dtype=np.float32).reshape(-1, 4) / 7
    obj._compute_node_embeddings = compute
    return obj, calls


def test_original_npz_and_compact_blob_are_exact_and_resume_without_recompute(tmp_path):
    # Calls the original node cache and the new adapter with the identical tiny
    # encoder. No model weights, GPU, real OT campaign, or live DB is accessed.
    original, old_calls = embedder(tmp_path / "old")
    new, new_calls = embedder(tmp_path / "new")
    conn = sqlite3.connect(tmp_path / "test-only.sqlite")
    compact = CompactNodeCache(new, conn)
    for smiles in ("CCO", "C(C)O", "CCN"):
        expected, actual = original.get(smiles), compact.get(smiles)
        assert actual.canonical_smiles == expected.canonical_smiles
        assert np.array_equal(actual.H, expected.H)
        assert np.array_equal(actual.atom_numbers, expected.atom_numbers)
    assert old_calls == new_calls == ["CCO", "CCN"]
    assert not list(new.node_emb_cache_dir.iterdir())
    conn.close()
    compact = CompactNodeCache(new, sqlite3.connect(tmp_path / "test-only.sqlite"))
    assert np.array_equal(compact.get("CCO").H, original.get("CCO").H)
    assert new_calls == ["CCO", "CCN"]


def test_compact_refuses_unbound_old_cache_and_corruption(tmp_path):
    obj, _ = embedder(tmp_path / "nodes")
    conn = sqlite3.connect(":memory:")
    cache = CompactNodeCache(obj, conn)
    cache.get("CC")
    conn.execute("UPDATE llm_node_embeddings_v1 SET content_sha256='wrong'")
    with pytest.raises(ValueError, match="CONTENT_CONFLICT"):
        cache.get("CC")
    (obj.node_emb_cache_dir / "previous.npz").write_bytes(b"old")
    with pytest.raises(ValueError, match="IGNORE_EXISTING_NPZ"):
        CompactNodeCache(obj, conn)


def test_real_distance_caller_preserves_keys_costs_and_action_context(tmp_path):
    old, _ = embedder(tmp_path / "old")
    new, _ = embedder(tmp_path / "new")
    distances = []
    # Actual transport caller; tiny exact LP solves the identical non-self graph
    # pair on both paths. No real encoder/model is loaded.
    from scipy.optimize import linprog
    def emd(a, b, cost):
        rows, cols = cost.shape
        A = np.zeros((rows + cols, rows * cols))
        for i in range(rows): A[i, i*cols:(i+1)*cols] = 1
        for j in range(cols): A[rows+j, j::cols] = 1
        result = linprog(cost.ravel(), A_eq=A, b_eq=np.r_[a, b], bounds=(0, None), method="highs")
        assert result.success
        return result.fun
    for name, item in (("old", old), ("new", new)):
        config = MolCLRNodeWassersteinConfig("unused", "unused", cache_db=tmp_path / f"{name}.sqlite")
        distance = MolCLRNodeWassersteinDistance(config, embedder=item, emd2_fn=emd)
        if name == "new": install_compact_node_cache(distance)
        distances.append(distance)
    context = {"candidate_id": "tiny-rule", "match_atom_indices": [0],
        "teacher_sha256": "frozen-teacher", "action_semantics_version": "delete_v1",
        "match_selection_policy": "frozen", "distance_implementation_version": "wnode_v1"}
    left = distances[0].distance_for_action("CCO", "CC", action_context=context)
    right = distances[1].distance_for_action("CCO", "CC", action_context=context)
    assert left["ok"] and right["ok"]
    assert left["distance"] == right["distance"] and left["metadata"] == right["metadata"]
    assert distances[1].distance_for_action("CCO", "CC", action_context=context)["cache_hit"]
    assert not list(new.node_emb_cache_dir.iterdir())
