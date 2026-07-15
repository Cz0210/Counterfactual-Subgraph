"""MolCLR node-embedding Fused Gromov-Wasserstein distance for CCRCov."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.embeddings.molclr_gnn_embedding import MolCLREmbeddingError
from src.eval.distance_cache import DEFAULT_DISTANCE_CACHE_PATH, SQLiteDistanceCache, canonical_pair_key
from src.eval.molclr_node_embeddings import (
    DEFAULT_NODE_EMB_CACHE_DIR,
    MolCLRNodeEmbedder as _SharedMolCLRNodeEmbedder,
    MolCLRNodeEmbeddingStats,
    atom_numbers_for_smiles,
    canonicalize_smiles,
)


NODE_FGW_VERSION = "molclr_node_fgw_v1"


@dataclass(frozen=True)
class NodeFGWConfig:
    molclr_root: str | Path
    molclr_ckpt: str | Path
    fgw_lambda: float = 0.5
    structure_mode: str = "shortest_path_unweighted"
    feature_cost: str = "cosine"
    atom_penalty: float = 0.0
    max_iter: int = 100
    tol: float = 1e-7
    device: str = "cuda"
    encoder_type: str = "gin"
    cache_db: str | Path = DEFAULT_DISTANCE_CACHE_PATH
    node_emb_cache_dir: str | Path = DEFAULT_NODE_EMB_CACHE_DIR


NodeFGWStats = MolCLRNodeEmbeddingStats


@dataclass
class MoleculeNodeData:
    canonical_smiles: str
    H: np.ndarray
    D: np.ndarray
    atom_numbers: np.ndarray

    @property
    def n_atoms(self) -> int:
        return int(self.H.shape[0])


def mol_to_structure_matrix(smiles: str, structure_mode: str = "shortest_path_unweighted") -> np.ndarray:
    """Return a normalized topological shortest-path distance matrix."""

    if structure_mode != "shortest_path_unweighted":
        raise ValueError(f"Unsupported structure_mode={structure_mode!r}")
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover
        raise MolCLREmbeddingError("RDKit is required to build structure distance matrices.") from exc
    mol = Chem.MolFromSmiles(str(smiles or "").strip())
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    n_atoms = int(mol.GetNumAtoms())
    if n_atoms <= 0:
        raise ValueError("Molecule has no atoms.")
    adjacency: list[list[int]] = [[] for _ in range(n_atoms)]
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        adjacency[begin].append(end)
        adjacency[end].append(begin)

    matrix = np.full((n_atoms, n_atoms), np.inf, dtype=np.float32)
    for start in range(n_atoms):
        matrix[start, start] = 0.0
        queue = [start]
        cursor = 0
        while cursor < len(queue):
            current = queue[cursor]
            cursor += 1
            for neighbor in adjacency[current]:
                if math.isfinite(float(matrix[start, neighbor])):
                    continue
                matrix[start, neighbor] = matrix[start, current] + 1.0
                queue.append(neighbor)

    finite = matrix[np.isfinite(matrix)]
    max_finite = float(np.max(finite)) if finite.size else 0.0
    disconnected_value = max_finite + 1.0
    matrix[~np.isfinite(matrix)] = disconnected_value
    normalizer = float(np.max(matrix))
    if normalizer > 0.0:
        matrix = matrix / normalizer
    np.fill_diagonal(matrix, 0.0)
    return matrix.astype(np.float32, copy=False)


class MolCLRNodeEmbedder(_SharedMolCLRNodeEmbedder):
    """Backward-compatible FGW view that adds ``D`` to shared node embeddings."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._fgw_structure_cache: dict[str, np.ndarray] = {}

    def get(self, smiles: str) -> MoleculeNodeData:
        embedding = super().get(smiles)
        matrix = self._fgw_structure_cache.get(embedding.canonical_smiles)
        if matrix is None:
            matrix = mol_to_structure_matrix(embedding.canonical_smiles, self.legacy_structure_mode)
            self._fgw_structure_cache[embedding.canonical_smiles] = matrix
        return MoleculeNodeData(
            canonical_smiles=embedding.canonical_smiles,
            H=embedding.H,
            D=matrix,
            atom_numbers=embedding.atom_numbers,
        )


def compute_feature_cost_matrix(
    H1: np.ndarray,
    H2: np.ndarray,
    atoms1: np.ndarray,
    atoms2: np.ndarray,
    *,
    feature_cost: str = "cosine",
    atom_penalty: float = 0.0,
) -> np.ndarray:
    if feature_cost != "cosine":
        raise ValueError(f"Unsupported feature_cost={feature_cost!r}")
    left = np.asarray(H1, dtype=np.float32)
    right = np.asarray(H2, dtype=np.float32)
    left_norm = left / np.clip(np.linalg.norm(left, axis=1, keepdims=True), 1e-12, None)
    right_norm = right / np.clip(np.linalg.norm(right, axis=1, keepdims=True), 1e-12, None)
    cost = 1.0 - np.matmul(left_norm, right_norm.T)
    cost = np.clip(cost, 0.0, 2.0)
    if float(atom_penalty) > 0.0:
        mismatch = np.asarray(atoms1)[:, None] != np.asarray(atoms2)[None, :]
        cost = cost + float(atom_penalty) * mismatch.astype(np.float32)
    return cost.astype(np.float64, copy=False)


def compute_fgw_distance(
    *,
    H1: np.ndarray,
    D1: np.ndarray,
    atoms1: np.ndarray,
    H2: np.ndarray,
    D2: np.ndarray,
    atoms2: np.ndarray,
    fgw_lambda: float = 0.5,
    feature_cost: str = "cosine",
    atom_penalty: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> float:
    try:
        import ot
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "POT is required for MolCLR Node-FGW distance. Install with "
            "`pip install POT` or `conda install -c conda-forge pot`."
        ) from exc
    n = int(np.asarray(H1).shape[0])
    m = int(np.asarray(H2).shape[0])
    if n <= 0 or m <= 0:
        raise ValueError("FGW requires non-empty node embeddings.")
    C = compute_feature_cost_matrix(H1, H2, atoms1, atoms2, feature_cost=feature_cost, atom_penalty=atom_penalty)
    C1 = np.asarray(D1, dtype=np.float64)
    C2 = np.asarray(D2, dtype=np.float64)
    p = np.ones(n, dtype=np.float64) / float(n)
    q = np.ones(m, dtype=np.float64) / float(m)
    kwargs = {
        "M": C,
        "C1": C1,
        "C2": C2,
        "p": p,
        "q": q,
        "loss_fun": "square_loss",
        "alpha": float(fgw_lambda),
        "max_iter": int(max_iter),
        "tol_rel": float(tol),
        "tol_abs": float(tol),
    }
    try:
        value = ot.gromov.fused_gromov_wasserstein2(**kwargs, armijo=False)
    except TypeError:
        kwargs.pop("tol_rel", None)
        kwargs.pop("tol_abs", None)
        try:
            value = ot.gromov.fused_gromov_wasserstein2(**kwargs)
        except TypeError:
            kwargs.pop("max_iter", None)
            value = ot.gromov.fused_gromov_wasserstein2(**kwargs)
    distance = float(value)
    if not math.isfinite(distance):
        raise ValueError("FGW returned non-finite distance.")
    return distance


class MolCLRNodeFGWDistanceProvider:
    """DistanceProvider-compatible MolCLR Node-FGW implementation."""

    def __init__(self, config: NodeFGWConfig) -> None:
        self.config = config
        self.embedder = _SharedMolCLRNodeEmbedder(
            molclr_root=config.molclr_root,
            molclr_ckpt=config.molclr_ckpt,
            node_emb_cache_dir=config.node_emb_cache_dir,
            structure_mode=config.structure_mode,
            encoder_type=config.encoder_type,
            device=config.device,
        )
        self.cache = SQLiteDistanceCache(config.cache_db)
        self._structure_cache: dict[str, np.ndarray] = {}
        self.started_at = time.time()

    def _structure_matrix(self, canonical_smiles: str) -> np.ndarray:
        matrix = self._structure_cache.get(canonical_smiles)
        if matrix is None:
            matrix = mol_to_structure_matrix(canonical_smiles, self.config.structure_mode)
            self._structure_cache[canonical_smiles] = matrix
        return matrix

    def distance(self, smiles_a: str, smiles_b: str) -> dict[str, Any]:
        result = self.compute_cached_fgw_distance(smiles_a, smiles_b)
        return {
            "distance": result.get("distance"),
            "cosine_similarity": None,
            "ok": bool(result.get("ok")),
            "error": result.get("error"),
            "cache_hit": result.get("cache_hit"),
        }

    def compute_cached_fgw_distance(self, smiles_a: str, smiles_b: str) -> dict[str, Any]:
        canonical_a = canonicalize_smiles(smiles_a)
        canonical_b = canonicalize_smiles(smiles_b)
        if canonical_a is None or canonical_b is None:
            self.embedder.stats.num_invalid_smiles += 1
            self.embedder.stats.num_nan_distances += 1
            return {"distance": np.nan, "ok": False, "cache_hit": False, "error": "invalid_smiles"}
        key = canonical_pair_key(
            distance_type="molclr_node_fgw",
            version=NODE_FGW_VERSION,
            canonical_smiles_a=canonical_a,
            canonical_smiles_b=canonical_b,
            molclr_ckpt=str(Path(self.config.molclr_ckpt).expanduser()),
            fgw_lambda=float(self.config.fgw_lambda),
            structure_mode=self.config.structure_mode,
            feature_cost=self.config.feature_cost,
            atom_penalty=float(self.config.atom_penalty),
        )
        cached, metadata = self.cache.get_distance(key)
        if cached is not None:
            return {"distance": float(cached), "ok": True, "cache_hit": True, "metadata": metadata, "error": None}
        try:
            left_embedding = self.embedder.get(canonical_a)
            right_embedding = self.embedder.get(canonical_b)
            left = MoleculeNodeData(
                canonical_smiles=canonical_a,
                H=left_embedding.H,
                D=self._structure_matrix(canonical_a),
                atom_numbers=left_embedding.atom_numbers,
            )
            right = MoleculeNodeData(
                canonical_smiles=canonical_b,
                H=right_embedding.H,
                D=self._structure_matrix(canonical_b),
                atom_numbers=right_embedding.atom_numbers,
            )
            distance = compute_fgw_distance(
                H1=left.H,
                D1=left.D,
                atoms1=left.atom_numbers,
                H2=right.H,
                D2=right.D,
                atoms2=right.atom_numbers,
                fgw_lambda=float(self.config.fgw_lambda),
                feature_cost=self.config.feature_cost,
                atom_penalty=float(self.config.atom_penalty),
                max_iter=int(self.config.max_iter),
                tol=float(self.config.tol),
            )
        except Exception as exc:
            self.embedder.stats.num_nan_distances += 1
            return {"distance": np.nan, "ok": False, "cache_hit": False, "error": str(exc)}
        metadata = {
            "n_atoms_a": left.n_atoms,
            "n_atoms_b": right.n_atoms,
            "fgw_lambda": float(self.config.fgw_lambda),
            "structure_mode": self.config.structure_mode,
            "feature_cost": self.config.feature_cost,
            "atom_penalty": float(self.config.atom_penalty),
        }
        self.cache.set_distance(key, distance, metadata, distance_type="molclr_node_fgw")
        return {"distance": distance, "ok": True, "cache_hit": False, "metadata": metadata, "error": None}

    def close(self) -> None:
        self.cache.close()

    def stats_dict(self) -> dict[str, Any]:
        pair = self.cache.stats_dict()
        node = self.embedder.stats
        return {
            **pair,
            "node_embedding_cache_hits": node.node_embedding_cache_hits,
            "node_embedding_cache_misses": node.node_embedding_cache_misses,
            "node_embedding_cache_hit_rate": node.node_embedding_cache_hit_rate,
            "node_embedding_cache_legacy_hits": node.node_embedding_cache_legacy_hits,
            "node_embedding_cache_migrations": node.node_embedding_cache_migrations,
            "num_nan_distances": node.num_nan_distances,
            "num_invalid_smiles": node.num_invalid_smiles,
            "runtime_seconds": float(time.time() - self.started_at),
            "fgw_lambda": float(self.config.fgw_lambda),
            "structure_mode": self.config.structure_mode,
            "feature_cost": self.config.feature_cost,
            "atom_penalty": float(self.config.atom_penalty),
            "node_emb_cache_dir": str(Path(self.config.node_emb_cache_dir).expanduser()),
        }


def compute_cached_fgw_distance(smiles_a: str, smiles_b: str, config: NodeFGWConfig) -> dict[str, Any]:
    provider = MolCLRNodeFGWDistanceProvider(config)
    return provider.compute_cached_fgw_distance(smiles_a, smiles_b)


__all__ = [
    "DEFAULT_NODE_EMB_CACHE_DIR",
    "NODE_FGW_VERSION",
    "MolCLRNodeEmbedder",
    "MolCLRNodeEmbeddingStats",
    "MolCLRNodeFGWDistanceProvider",
    "MoleculeNodeData",
    "NodeFGWConfig",
    "canonicalize_smiles",
    "compute_cached_fgw_distance",
    "compute_feature_cost_matrix",
    "compute_fgw_distance",
    "mol_to_structure_matrix",
]
