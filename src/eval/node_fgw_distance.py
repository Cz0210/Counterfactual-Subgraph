"""MolCLR node-embedding Fused Gromov-Wasserstein distance for CCRCov."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.embeddings.molclr_gnn_embedding import (
    MolCLREmbeddingError,
    load_molclr_model,
    smiles_to_molclr_data,
)
from src.eval.distance_cache import DEFAULT_DISTANCE_CACHE_PATH, SQLiteDistanceCache, canonical_pair_key

try:  # pragma: no cover - runtime dependency
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


NODE_FGW_VERSION = "molclr_node_fgw_v1"
DEFAULT_NODE_EMB_CACHE_DIR = "outputs/hpc/cache/molclr_node_embeddings"


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


@dataclass
class NodeFGWStats:
    node_embedding_cache_hits: int = 0
    node_embedding_cache_misses: int = 0
    num_nan_distances: int = 0
    num_invalid_smiles: int = 0

    @property
    def node_embedding_cache_hit_rate(self) -> float:
        total = self.node_embedding_cache_hits + self.node_embedding_cache_misses
        return float(self.node_embedding_cache_hits / total) if total else 0.0


@dataclass
class MoleculeNodeData:
    canonical_smiles: str
    H: np.ndarray
    D: np.ndarray
    atom_numbers: np.ndarray

    @property
    def n_atoms(self) -> int:
        return int(self.H.shape[0])


def canonicalize_smiles(smiles: str) -> str | None:
    if Chem is None:
        return None
    text = str(smiles or "").strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None or mol.GetNumAtoms() <= 0:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def atom_numbers_for_smiles(smiles: str) -> np.ndarray:
    if Chem is None:
        raise MolCLREmbeddingError("RDKit is required for atom-number extraction.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return np.asarray([int(atom.GetAtomicNum()) for atom in mol.GetAtoms()], dtype=np.int64)


def mol_to_structure_matrix(smiles: str, structure_mode: str = "shortest_path_unweighted") -> np.ndarray:
    """Return a normalized topological shortest-path distance matrix."""

    if structure_mode != "shortest_path_unweighted":
        raise ValueError(f"Unsupported structure_mode={structure_mode!r}")
    if Chem is None:
        raise MolCLREmbeddingError("RDKit is required to build structure distance matrices.")
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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


class MolCLRNodeEmbedder:
    """MolCLR GIN node embedding extractor with per-molecule NPZ cache."""

    def __init__(
        self,
        *,
        molclr_root: str | Path,
        molclr_ckpt: str | Path,
        node_emb_cache_dir: str | Path = DEFAULT_NODE_EMB_CACHE_DIR,
        structure_mode: str = "shortest_path_unweighted",
        encoder_type: str = "gin",
        device: str = "cuda",
    ) -> None:
        self.molclr_root = Path(molclr_root).expanduser()
        self.molclr_ckpt = Path(molclr_ckpt).expanduser()
        self.node_emb_cache_dir = Path(node_emb_cache_dir).expanduser()
        self.node_emb_cache_dir.mkdir(parents=True, exist_ok=True)
        self.structure_mode = str(structure_mode)
        self.encoder_type = str(encoder_type)
        self.loaded = load_molclr_model(
            molclr_root=self.molclr_root,
            molclr_ckpt=self.molclr_ckpt,
            encoder_type=self.encoder_type,
            device=device,
        )
        self.stats = NodeFGWStats()

    def cache_path(self, canonical_smiles: str) -> Path:
        payload = json.dumps(
            {
                "canonical_smiles": canonical_smiles,
                "molclr_ckpt": str(self.loaded.checkpoint_path),
                "version": NODE_FGW_VERSION,
                "structure_mode": self.structure_mode,
                "encoder_type": self.encoder_type,
            },
            sort_keys=True,
        )
        return self.node_emb_cache_dir / f"{_sha256_text(payload)}.npz"

    def get(self, smiles: str) -> MoleculeNodeData:
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            self.stats.num_invalid_smiles += 1
            raise ValueError(f"Invalid SMILES for MolCLR node FGW: {smiles!r}")
        path = self.cache_path(canonical)
        if path.exists():
            try:
                loaded = np.load(path, allow_pickle=False)
                self.stats.node_embedding_cache_hits += 1
                return MoleculeNodeData(
                    canonical_smiles=str(loaded["canonical_smiles"].item()),
                    H=np.asarray(loaded["H"], dtype=np.float32),
                    D=np.asarray(loaded["D"], dtype=np.float32),
                    atom_numbers=np.asarray(loaded["atom_numbers"], dtype=np.int64),
                )
            except Exception:
                path.unlink(missing_ok=True)

        self.stats.node_embedding_cache_misses += 1
        H = self._compute_node_embeddings(canonical)
        D = mol_to_structure_matrix(canonical, self.structure_mode)
        atoms = atom_numbers_for_smiles(canonical)
        if H.shape[0] != atoms.shape[0] or D.shape != (atoms.shape[0], atoms.shape[0]):
            raise MolCLREmbeddingError(
                "MolCLR node embedding order/count does not align with RDKit atom order: "
                f"H={H.shape}, D={D.shape}, atoms={atoms.shape}"
            )
        np.savez_compressed(
            path,
            H=H.astype(np.float32, copy=False),
            D=D.astype(np.float32, copy=False),
            atom_numbers=atoms.astype(np.int64, copy=False),
            canonical_smiles=np.asarray(canonical),
            n_atoms=np.asarray(atoms.shape[0]),
        )
        return MoleculeNodeData(canonical_smiles=canonical, H=H, D=D, atom_numbers=atoms)

    def _compute_node_embeddings(self, canonical_smiles: str) -> np.ndarray:
        torch = _require_torch()
        Batch = _require_batch()
        data = smiles_to_molclr_data(canonical_smiles)
        batch = Batch.from_data_list([data]).to(self.loaded.device)
        model = self.loaded.model
        if not all(hasattr(model, name) for name in ("x_embedding1", "x_embedding2", "gnns", "batch_norms")):
            raise MolCLREmbeddingError(
                "Node-level MolCLR extraction currently requires a GIN/GINet-style model "
                "with x_embedding1/x_embedding2/gnns/batch_norms. Graph-level embedding "
                "fallback is intentionally not used for molclr_node_fgw."
            )
        with torch.inference_mode():
            x = batch.x
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr
            h = model.x_embedding1(x[:, 0]) + model.x_embedding2(x[:, 1])
            num_layer = int(getattr(model, "num_layer", len(model.gnns)))
            for layer in range(num_layer):
                h = model.gnns[layer](h, edge_index, edge_attr)
                h = model.batch_norms[layer](h)
                if layer != num_layer - 1:
                    h = torch.relu(h)
            h = h.detach().to(torch.float32)
            if not torch.all(torch.isfinite(h)):
                raise MolCLREmbeddingError("MolCLR node embeddings contained NaN or Inf.")
            return h.cpu().numpy().astype(np.float32, copy=False)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise MolCLREmbeddingError("MolCLR node FGW requires PyTorch.") from exc
    return torch


def _require_batch() -> Any:
    try:
        from torch_geometric.data import Batch
    except ImportError as exc:  # pragma: no cover
        raise MolCLREmbeddingError("MolCLR node FGW requires torch_geometric.") from exc
    return Batch


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
        self.embedder = MolCLRNodeEmbedder(
            molclr_root=config.molclr_root,
            molclr_ckpt=config.molclr_ckpt,
            node_emb_cache_dir=config.node_emb_cache_dir,
            structure_mode=config.structure_mode,
            encoder_type=config.encoder_type,
            device=config.device,
        )
        self.cache = SQLiteDistanceCache(config.cache_db)
        self.started_at = time.time()

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
            left = self.embedder.get(canonical_a)
            right = self.embedder.get(canonical_b)
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

    def stats_dict(self) -> dict[str, Any]:
        pair = self.cache.stats_dict()
        node = self.embedder.stats
        return {
            **pair,
            "node_embedding_cache_hits": node.node_embedding_cache_hits,
            "node_embedding_cache_misses": node.node_embedding_cache_misses,
            "node_embedding_cache_hit_rate": node.node_embedding_cache_hit_rate,
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
    "MolCLRNodeFGWDistanceProvider",
    "MoleculeNodeData",
    "NodeFGWConfig",
    "canonicalize_smiles",
    "compute_cached_fgw_distance",
    "compute_feature_cost_matrix",
    "compute_fgw_distance",
    "mol_to_structure_matrix",
]
