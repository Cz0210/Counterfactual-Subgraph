"""Exact MolCLR node-level Wasserstein distance for unified CCRCov evaluation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.eval.distance_cache import SQLiteDistanceCache, canonical_symmetric_distance_key
from src.eval.molclr_node_embeddings import (
    DEFAULT_NODE_EMB_CACHE_DIR,
    NODE_EMBEDDING_CACHE_SCHEMA_VERSION,
    MolCLRNodeEmbedder,
    canonicalize_smiles,
)


WNODE_DISTANCE_NAMESPACE = "molclr_node_wasserstein_v1"
DEFAULT_WNODE_CACHE_DB = "outputs/hpc/cache/distance_cache/molclr_node_wasserstein_v1.sqlite"


@dataclass(frozen=True)
class MolCLRNodeWassersteinConfig:
    molclr_root: str | Path
    molclr_ckpt: str | Path
    cache_db: str | Path = DEFAULT_WNODE_CACHE_DB
    node_emb_cache_dir: str | Path = DEFAULT_NODE_EMB_CACHE_DIR
    feature_cost: str = "cosine"
    node_mass: str = "uniform"
    size_penalty_beta: float = 0.0
    device: str = "cuda"
    encoder_type: str = "gin"


def cosine_node_cost_matrix(H1: np.ndarray, H2: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    """Return float64 ``1 - cosine`` node costs with stable row normalization."""

    left = np.asarray(H1, dtype=np.float64)
    right = np.asarray(H2, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] <= 0 or right.shape[0] <= 0:
        raise ValueError("Node Wasserstein requires non-empty rank-2 node embedding matrices.")
    if left.shape[1] != right.shape[1]:
        raise ValueError(f"Node embedding dimensions differ: {left.shape} vs {right.shape}")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("Node embeddings contain NaN or Inf.")
    left /= np.maximum(np.linalg.norm(left, axis=1, keepdims=True), float(epsilon))
    right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), float(epsilon))
    similarity = np.clip(left @ right.T, -1.0, 1.0)
    return np.asarray(1.0 - similarity, dtype=np.float64)


def uniform_node_mass(num_nodes: int) -> np.ndarray:
    if int(num_nodes) <= 0:
        raise ValueError("Uniform node mass requires at least one node.")
    return np.full(int(num_nodes), 1.0 / float(num_nodes), dtype=np.float64)


def compute_node_wasserstein_distance(
    H1: np.ndarray,
    H2: np.ndarray,
    *,
    feature_cost: str = "cosine",
    node_mass: str = "uniform",
    size_penalty_beta: float = 0.0,
    emd2_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Compute exact EMD over MolCLR node embeddings, without graph structure costs."""

    if feature_cost != "cosine":
        raise ValueError(f"Unsupported feature_cost={feature_cost!r}; expected 'cosine'.")
    if node_mass != "uniform":
        raise ValueError(f"Unsupported node_mass={node_mass!r}; expected 'uniform'.")
    if float(size_penalty_beta) < 0.0:
        raise ValueError("size_penalty_beta must be non-negative.")
    M = cosine_node_cost_matrix(H1, H2)
    n_a, n_b = int(M.shape[0]), int(M.shape[1])
    a = uniform_node_mass(n_a)
    b = uniform_node_mass(n_b)
    if emd2_fn is None:
        try:
            import ot
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "POT is required for MolCLR-Node-Wasserstein exact OT. Install with "
                "`pip install POT` or `conda install -c conda-forge pot`."
            ) from exc
        emd2_fn = ot.emd2
    base_distance = max(float(emd2_fn(a, b, np.asarray(M, dtype=np.float64))), 0.0)
    if not math.isfinite(base_distance):
        raise ValueError("ot.emd2 returned a non-finite distance.")
    size_penalty = float(size_penalty_beta) * abs(n_a - n_b) / float(max(n_a, n_b))
    total = max(base_distance + size_penalty, 0.0)
    return total, {
        "base_ot_distance": base_distance,
        "size_penalty": size_penalty,
        "size_penalty_beta": float(size_penalty_beta),
        "num_nodes_a": n_a,
        "num_nodes_b": n_b,
        "feature_cost": feature_cost,
        "node_mass": node_mass,
        "solver": "exact_emd2",
        "mass_sum_a": float(a.sum()),
        "mass_sum_b": float(b.sum()),
    }


def node_wasserstein_pair_key(
    *,
    canonical_smiles_a: str,
    canonical_smiles_b: str,
    checkpoint_identity: str,
    feature_cost: str,
    node_mass: str,
    size_penalty_beta: float,
    node_embedding_schema_version: str = NODE_EMBEDDING_CACHE_SCHEMA_VERSION,
) -> str:
    return canonical_symmetric_distance_key(
        distance_namespace=WNODE_DISTANCE_NAMESPACE,
        canonical_smiles_a=canonical_smiles_a,
        canonical_smiles_b=canonical_smiles_b,
        parameters={
            "feature_cost": str(feature_cost),
            "node_mass": str(node_mass),
            "size_penalty_beta": float(size_penalty_beta),
            "molclr_checkpoint_identity": str(checkpoint_identity),
            "node_embedding_schema_version": str(node_embedding_schema_version),
            "solver": "exact_emd2",
        },
    )


class MolCLRNodeWassersteinDistance:
    """DistanceProvider-compatible exact node OT with independent pair cache."""

    def __init__(
        self,
        config: MolCLRNodeWassersteinConfig,
        *,
        embedder: MolCLRNodeEmbedder | None = None,
        cache: SQLiteDistanceCache | None = None,
        emd2_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], Any] | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder or MolCLRNodeEmbedder(
            molclr_root=config.molclr_root,
            molclr_ckpt=config.molclr_ckpt,
            node_emb_cache_dir=config.node_emb_cache_dir,
            encoder_type=config.encoder_type,
            device=config.device,
        )
        self.cache = cache or SQLiteDistanceCache(config.cache_db)
        self._emd2_fn = emd2_fn
        self.started_at = time.time()

    def distance(self, smiles_a: str, smiles_b: str) -> dict[str, Any]:
        canonical_a = canonicalize_smiles(smiles_a)
        canonical_b = canonicalize_smiles(smiles_b)
        if canonical_a is None or canonical_b is None:
            self.embedder.stats.num_invalid_smiles += 1
            self.embedder.stats.num_nan_distances += 1
            return {
                "distance": float("nan"),
                "cosine_similarity": None,
                "ok": False,
                "cache_hit": False,
                "error": "invalid_or_empty_smiles",
                "metadata": None,
            }
        key = node_wasserstein_pair_key(
            canonical_smiles_a=canonical_a,
            canonical_smiles_b=canonical_b,
            checkpoint_identity=self.embedder.checkpoint_identity,
            feature_cost=self.config.feature_cost,
            node_mass=self.config.node_mass,
            size_penalty_beta=float(self.config.size_penalty_beta),
        )
        cached, metadata = self.cache.get_distance(key)
        if cached is not None:
            return {
                "distance": max(float(cached), 0.0),
                "cosine_similarity": None,
                "ok": True,
                "cache_hit": True,
                "error": None,
                "metadata": metadata,
            }
        try:
            left = self.embedder.get(canonical_a)
            right = self.embedder.get(canonical_b)
            value, metadata = compute_node_wasserstein_distance(
                left.H,
                right.H,
                feature_cost=self.config.feature_cost,
                node_mass=self.config.node_mass,
                size_penalty_beta=float(self.config.size_penalty_beta),
                emd2_fn=self._emd2_fn,
            )
        except Exception as exc:
            self.embedder.stats.num_nan_distances += 1
            return {
                "distance": float("nan"),
                "cosine_similarity": None,
                "ok": False,
                "cache_hit": False,
                "error": str(exc),
                "metadata": None,
            }
        metadata.update(
            {
                "canonical_smiles_a": canonical_a,
                "canonical_smiles_b": canonical_b,
                "distance_namespace": WNODE_DISTANCE_NAMESPACE,
                "node_embedding_schema_version": NODE_EMBEDDING_CACHE_SCHEMA_VERSION,
            }
        )
        self.cache.set_distance(key, value, metadata, distance_type="molclr_node_wasserstein")
        return {
            "distance": value,
            "cosine_similarity": None,
            "ok": True,
            "cache_hit": False,
            "error": None,
            "metadata": metadata,
        }

    def stats_dict(self) -> dict[str, Any]:
        node = self.embedder.stats
        return {
            **self.cache.stats_dict(),
            "node_embedding_cache_hits": node.node_embedding_cache_hits,
            "node_embedding_cache_misses": node.node_embedding_cache_misses,
            "node_embedding_cache_hit_rate": node.node_embedding_cache_hit_rate,
            "node_embedding_cache_legacy_hits": node.node_embedding_cache_legacy_hits,
            "node_embedding_cache_migrations": node.node_embedding_cache_migrations,
            "num_nan_distances": node.num_nan_distances,
            "num_invalid_smiles": node.num_invalid_smiles,
            "runtime_seconds": float(time.time() - self.started_at),
            "distance_type": "node_wasserstein",
            "distance_line": "MolCLR-Node-Wasserstein",
            "feature_cost": self.config.feature_cost,
            "node_mass": self.config.node_mass,
            "size_penalty_beta": float(self.config.size_penalty_beta),
            "solver": "exact_emd2",
            "node_emb_cache_dir": str(Path(self.config.node_emb_cache_dir).expanduser()),
        }

    def close(self) -> None:
        self.cache.close()


__all__ = [
    "DEFAULT_WNODE_CACHE_DB",
    "MolCLRNodeWassersteinConfig",
    "MolCLRNodeWassersteinDistance",
    "WNODE_DISTANCE_NAMESPACE",
    "compute_node_wasserstein_distance",
    "cosine_node_cost_matrix",
    "node_wasserstein_pair_key",
    "uniform_node_mass",
]
