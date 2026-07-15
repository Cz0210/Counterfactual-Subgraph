from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest

from src.eval.molclr_node_embeddings import (
    MolCLRNodeEmbedder,
    node_embedding_cache_payload,
)


def test_node_embedding_key_payload_has_no_structure_or_distance_fields() -> None:
    first = node_embedding_cache_payload(
        canonical_smiles="CCO", checkpoint_identity_value="ckpt", encoder_type="gin",
        architecture_identity="GIN:5:300",
    )
    second = node_embedding_cache_payload(
        canonical_smiles="CCO", checkpoint_identity_value="ckpt", encoder_type="gin",
        architecture_identity="GIN:5:300",
    )
    assert first == second
    forbidden = {"structure_mode", "fgw_lambda", "feature_cost", "size_penalty_beta", "distance_type"}
    assert forbidden.isdisjoint(first)


def test_legacy_shortest_path_cache_fallback_migrates_without_deleting(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    model = SimpleNamespace(num_layer=5, emb_dim=2)
    loaded = SimpleNamespace(
        checkpoint_path=checkpoint.resolve(), model=model, model_class="GINet", device="cpu"
    )
    embedder = MolCLRNodeEmbedder(
        molclr_root=tmp_path, molclr_ckpt=checkpoint, node_emb_cache_dir=tmp_path / "cache",
        structure_mode="shortest_path_unweighted", loaded_model=loaded,
    )
    legacy = embedder.legacy_cache_path("CCO")
    expected_payload = {
        "canonical_smiles": "CCO",
        "molclr_ckpt": str(checkpoint.resolve()),
        "version": "molclr_node_fgw_v1",
        "structure_mode": "shortest_path_unweighted",
        "encoder_type": "gin",
    }
    expected_name = hashlib.sha256(json.dumps(expected_payload, sort_keys=True).encode("utf-8")).hexdigest() + ".npz"
    assert legacy.name == expected_name
    legacy.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        legacy,
        H=np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        D=np.zeros((3, 3), dtype=np.float32),
        atom_numbers=np.asarray([6, 6, 8], dtype=np.int64),
        canonical_smiles=np.asarray("CCO"),
        n_atoms=np.asarray(3),
    )
    embedder._compute_node_embeddings = lambda _smiles: (_ for _ in ()).throw(AssertionError("must not compute"))  # type: ignore[method-assign]
    data = embedder.get("CCO")
    assert data.H.shape == (3, 2)
    assert legacy.is_file()
    assert embedder.cache_path("CCO").is_file()
    assert embedder.stats.node_embedding_cache_legacy_hits == 1
    assert embedder.stats.node_embedding_cache_migrations == 1
