"""Shared MolCLR node embeddings with a structure-independent NPZ cache."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.embeddings.molclr_gnn_embedding import (
    MolCLREmbeddingError,
    load_molclr_model,
    smiles_to_molclr_data,
)

try:  # pragma: no cover - runtime dependency
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


DEFAULT_NODE_EMB_CACHE_DIR = "outputs/hpc/cache/molclr_node_embeddings"
NODE_EMBEDDING_CACHE_SCHEMA_VERSION = "molclr_node_embedding_v2"
NODE_EXTRACTION_VERSION = "molclr_gin_nodes_v2"
LEGACY_NODE_FGW_VERSION = "molclr_node_fgw_v1"
LEGACY_STRUCTURE_MODE = "shortest_path_unweighted"


@dataclass
class MolCLRNodeEmbeddingStats:
    node_embedding_cache_hits: int = 0
    node_embedding_cache_misses: int = 0
    node_embedding_cache_legacy_hits: int = 0
    node_embedding_cache_migrations: int = 0
    num_nan_distances: int = 0
    num_invalid_smiles: int = 0

    @property
    def node_embedding_cache_hit_rate(self) -> float:
        total = self.node_embedding_cache_hits + self.node_embedding_cache_misses
        return float(self.node_embedding_cache_hits / total) if total else 0.0


@dataclass(frozen=True)
class MoleculeNodeEmbedding:
    canonical_smiles: str
    H: np.ndarray
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
    return str(Chem.MolToSmiles(mol, canonical=True))


def atom_numbers_for_smiles(smiles: str) -> np.ndarray:
    if Chem is None:
        raise MolCLREmbeddingError("RDKit is required for atom-number extraction.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() <= 0:
        raise ValueError(f"RDKit could not parse a non-empty molecule: {smiles!r}")
    return np.asarray([int(atom.GetAtomicNum()) for atom in mol.GetAtoms()], dtype=np.int64)


def _sha256_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _legacy_sha256_payload(payload: dict[str, Any]) -> str:
    # v1 used json.dumps(..., sort_keys=True) with the default separators.
    encoded = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def checkpoint_identity(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        stat = resolved.stat()
        payload = {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except OSError:
        payload = {"path": str(resolved), "size": None, "mtime_ns": None}
    return _sha256_payload(payload)


def node_embedding_cache_payload(
    *,
    canonical_smiles: str,
    checkpoint_identity_value: str,
    encoder_type: str,
    architecture_identity: str,
    feature_schema_identity: str = "molclr_atom_index_chirality_v1",
) -> dict[str, Any]:
    """Return the v2 key payload; distance/structure configuration is excluded."""

    return {
        "canonical_smiles": str(canonical_smiles),
        "checkpoint_identity": str(checkpoint_identity_value),
        "encoder_type": str(encoder_type),
        "node_extraction_version": NODE_EXTRACTION_VERSION,
        "cache_schema_version": NODE_EMBEDDING_CACHE_SCHEMA_VERSION,
        "architecture_identity": str(architecture_identity),
        "feature_schema_identity": str(feature_schema_identity),
    }


def legacy_node_cache_payload(
    *,
    canonical_smiles: str,
    checkpoint_path: str | Path,
    encoder_type: str,
    structure_mode: str = LEGACY_STRUCTURE_MODE,
) -> dict[str, Any]:
    """Exact payload used by the historical Node-FGW NPZ cache."""

    return {
        "canonical_smiles": str(canonical_smiles),
        "molclr_ckpt": str(Path(checkpoint_path)),
        "version": LEGACY_NODE_FGW_VERSION,
        "structure_mode": str(structure_mode),
        "encoder_type": str(encoder_type),
    }


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise MolCLREmbeddingError("MolCLR node embeddings require PyTorch.") from exc
    return torch


def _require_batch() -> Any:
    try:
        from torch_geometric.data import Batch
    except ImportError as exc:  # pragma: no cover
        raise MolCLREmbeddingError("MolCLR node embeddings require torch_geometric.") from exc
    return Batch


class MolCLRNodeEmbedder:
    """Extract GIN node states and cache them independently of a distance line."""

    def __init__(
        self,
        *,
        molclr_root: str | Path,
        molclr_ckpt: str | Path,
        node_emb_cache_dir: str | Path = DEFAULT_NODE_EMB_CACHE_DIR,
        structure_mode: str | None = None,
        encoder_type: str = "gin",
        device: str = "cuda",
        loaded_model: Any | None = None,
    ) -> None:
        self.molclr_root = Path(molclr_root).expanduser()
        self.molclr_ckpt = Path(molclr_ckpt).expanduser()
        self.node_emb_cache_dir = Path(node_emb_cache_dir).expanduser()
        self.node_emb_cache_dir.mkdir(parents=True, exist_ok=True)
        # Kept only to find the historical v1 cache. It is not part of the v2 key.
        self.legacy_structure_mode = str(structure_mode or LEGACY_STRUCTURE_MODE)
        self.encoder_type = str(encoder_type)
        self.loaded = loaded_model or load_molclr_model(
            molclr_root=self.molclr_root,
            molclr_ckpt=self.molclr_ckpt,
            encoder_type=self.encoder_type,
            device=device,
        )
        model = self.loaded.model
        self.architecture_identity = ":".join(
            str(value)
            for value in (
                getattr(self.loaded, "model_class", model.__class__.__name__),
                getattr(model, "num_layer", "unknown_layers"),
                getattr(model, "emb_dim", getattr(model, "feat_dim", "unknown_dim")),
            )
        )
        self.checkpoint_identity = checkpoint_identity(self.loaded.checkpoint_path)
        self.stats = MolCLRNodeEmbeddingStats()

    def cache_payload(self, canonical_smiles: str) -> dict[str, Any]:
        return node_embedding_cache_payload(
            canonical_smiles=canonical_smiles,
            checkpoint_identity_value=self.checkpoint_identity,
            encoder_type=self.encoder_type,
            architecture_identity=self.architecture_identity,
        )

    def cache_path(self, canonical_smiles: str) -> Path:
        return self.node_emb_cache_dir / f"{_sha256_payload(self.cache_payload(canonical_smiles))}.npz"

    def legacy_cache_path(self, canonical_smiles: str) -> Path:
        payload = legacy_node_cache_payload(
            canonical_smiles=canonical_smiles,
            checkpoint_path=self.loaded.checkpoint_path,
            encoder_type=self.encoder_type,
            structure_mode=self.legacy_structure_mode,
        )
        return self.node_emb_cache_dir / f"{_legacy_sha256_payload(payload)}.npz"

    @staticmethod
    def _load_npz(path: Path, canonical_smiles: str) -> MoleculeNodeEmbedding:
        with np.load(path, allow_pickle=False) as loaded:
            H = np.asarray(loaded["H"], dtype=np.float32)
            atoms = np.asarray(loaded["atom_numbers"], dtype=np.int64)
            stored = str(loaded["canonical_smiles"].item())
        if stored != canonical_smiles or H.ndim != 2 or H.shape[0] <= 0 or H.shape[0] != atoms.shape[0]:
            raise ValueError(f"Invalid MolCLR node cache schema/content: {path}")
        if not np.all(np.isfinite(H)):
            raise ValueError(f"MolCLR node cache contains non-finite embeddings: {path}")
        return MoleculeNodeEmbedding(stored, H, atoms)

    def _atomic_write(self, path: Path, data: MoleculeNodeEmbedding) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz.tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(
                    handle,
                    H=data.H.astype(np.float32, copy=False),
                    atom_numbers=data.atom_numbers.astype(np.int64, copy=False),
                    canonical_smiles=np.asarray(data.canonical_smiles),
                    n_atoms=np.asarray(data.n_atoms),
                    cache_schema_version=np.asarray(NODE_EMBEDDING_CACHE_SCHEMA_VERSION),
                    checkpoint_identity=np.asarray(self.checkpoint_identity),
                    encoder_type=np.asarray(self.encoder_type),
                    node_extraction_version=np.asarray(NODE_EXTRACTION_VERSION),
                    architecture_identity=np.asarray(self.architecture_identity),
                    feature_schema_identity=np.asarray("molclr_atom_index_chirality_v1"),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    def get(self, smiles: str) -> MoleculeNodeEmbedding:
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            self.stats.num_invalid_smiles += 1
            raise ValueError(f"Invalid SMILES for MolCLR node embeddings: {smiles!r}")
        path = self.cache_path(canonical)
        if path.is_file():
            try:
                data = self._load_npz(path, canonical)
                self.stats.node_embedding_cache_hits += 1
                return data
            except Exception:
                # A corrupt shared cache is ignored, but never deleted here.
                pass

        legacy_path = self.legacy_cache_path(canonical)
        if legacy_path.is_file():
            try:
                data = self._load_npz(legacy_path, canonical)
                self.stats.node_embedding_cache_hits += 1
                self.stats.node_embedding_cache_legacy_hits += 1
                self._atomic_write(path, data)
                self.stats.node_embedding_cache_migrations += 1
                return data
            except Exception:
                pass

        self.stats.node_embedding_cache_misses += 1
        H = self._compute_node_embeddings(canonical)
        atoms = atom_numbers_for_smiles(canonical)
        if H.shape[0] != atoms.shape[0]:
            raise MolCLREmbeddingError(
                "MolCLR node embedding order/count does not align with RDKit atom order: "
                f"H={H.shape}, atoms={atoms.shape}"
            )
        data = MoleculeNodeEmbedding(canonical, H, atoms)
        self._atomic_write(path, data)
        return data

    def _compute_node_embeddings(self, canonical_smiles: str) -> np.ndarray:
        torch = _require_torch()
        Batch = _require_batch()
        data = smiles_to_molclr_data(canonical_smiles)
        batch = Batch.from_data_list([data]).to(self.loaded.device)
        model = self.loaded.model
        if not all(hasattr(model, name) for name in ("x_embedding1", "x_embedding2", "gnns", "batch_norms")):
            raise MolCLREmbeddingError(
                "Node-level MolCLR extraction requires a GIN/GINet-style model with "
                "x_embedding1/x_embedding2/gnns/batch_norms; graph pooling is not used."
            )
        with torch.inference_mode():
            x = batch.x
            h = model.x_embedding1(x[:, 0]) + model.x_embedding2(x[:, 1])
            num_layer = int(getattr(model, "num_layer", len(model.gnns)))
            for layer in range(num_layer):
                h = model.gnns[layer](h, batch.edge_index, batch.edge_attr)
                h = model.batch_norms[layer](h)
                if layer != num_layer - 1:
                    h = torch.relu(h)
            h = h.detach().to(torch.float32)
            if not torch.all(torch.isfinite(h)):
                raise MolCLREmbeddingError("MolCLR node embeddings contained NaN or Inf.")
            return h.cpu().numpy().astype(np.float32, copy=False)


__all__ = [
    "DEFAULT_NODE_EMB_CACHE_DIR",
    "LEGACY_NODE_FGW_VERSION",
    "LEGACY_STRUCTURE_MODE",
    "MolCLRNodeEmbedder",
    "MolCLRNodeEmbeddingStats",
    "MoleculeNodeEmbedding",
    "NODE_EMBEDDING_CACHE_SCHEMA_VERSION",
    "NODE_EXTRACTION_VERSION",
    "atom_numbers_for_smiles",
    "canonicalize_smiles",
    "checkpoint_identity",
    "legacy_node_cache_payload",
    "node_embedding_cache_payload",
]
