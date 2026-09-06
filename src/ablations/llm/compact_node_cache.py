"""Future BACE LLM-only lossless node cache in its existing WNode database.

The original MolCLR encoder, features, cache identity and exact OT are unchanged.
This adapter creates no per-graph files and never adopts another run's live DB.
"""
from __future__ import annotations

import hashlib
import io

import numpy as np

from src.eval.molclr_node_embeddings import (
    MoleculeNodeEmbedding, atom_numbers_for_smiles, canonicalize_smiles, _sha256_payload,
)


class CompactNodeCache:
    def __init__(self, original, connection):
        if any(original.node_emb_cache_dir.glob("*.npz")):
            raise ValueError("COMPACT_NODE_CACHE_CANNOT_SILENTLY_IGNORE_EXISTING_NPZ")
        self.original = original
        self.conn = connection
        self.stats = original.stats
        self.checkpoint_identity = original.checkpoint_identity
        self.conn.execute("CREATE TABLE IF NOT EXISTS llm_node_embeddings_v1 "
                          "(key TEXT PRIMARY KEY, canonical_smiles TEXT NOT NULL, "
                          "content_sha256 TEXT NOT NULL, npz BLOB NOT NULL)")
        self.conn.commit()

    def get(self, smiles):
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            self.stats.num_invalid_smiles += 1
            raise ValueError("Invalid SMILES for compact MolCLR node cache")
        key = _sha256_payload(self.original.cache_payload(canonical))
        row = self.conn.execute("SELECT canonical_smiles, content_sha256, npz "
                                "FROM llm_node_embeddings_v1 WHERE key=?", (key,)).fetchone()
        if row is not None:
            stored, digest, raw = row
            if stored != canonical or hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("COMPACT_NODE_CACHE_CONTENT_CONFLICT")
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                H = np.asarray(data["H"], dtype=np.float32)
                atoms = np.asarray(data["atom_numbers"], dtype=np.int64)
                if (str(data["canonical_smiles"].item()) != canonical or H.ndim != 2
                        or H.shape[0] <= 0 or H.shape[0] != atoms.shape[0]
                        or not np.all(np.isfinite(H))):
                    raise ValueError("COMPACT_NODE_CACHE_SCHEMA_CONFLICT")
            self.stats.node_embedding_cache_hits += 1
            return MoleculeNodeEmbedding(canonical, H, atoms)
        self.stats.node_embedding_cache_misses += 1
        H = self.original._compute_node_embeddings(canonical)
        atoms = atom_numbers_for_smiles(canonical)
        if H.shape[0] != atoms.shape[0]:
            raise ValueError("COMPACT_NODE_CACHE_ATOM_ORDER_COUNT_MISMATCH")
        result = MoleculeNodeEmbedding(canonical, H, atoms)
        stream = io.BytesIO()
        np.savez_compressed(stream, canonical_smiles=np.asarray(canonical),
                            H=np.asarray(H, dtype=np.float32), atom_numbers=np.asarray(atoms, dtype=np.int64))
        raw = stream.getvalue()
        self.conn.execute("INSERT INTO llm_node_embeddings_v1 VALUES (?,?,?,?)",
                          (key, canonical, hashlib.sha256(raw).hexdigest(), raw))
        self.conn.commit()
        return result


def install_compact_node_cache(distance):
    """Call only for a fresh future LLM evaluator, or its same-mode resume."""
    distance.embedder = CompactNodeCache(distance.embedder, distance.cache.conn)
    return distance
