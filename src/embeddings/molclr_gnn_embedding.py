"""MolCLR pretrained GNN fragment embedding helpers.

MolCLR itself is intentionally not vendored in this repository.  This module
loads a runtime MolCLR checkout and checkpoint from explicit user-supplied
paths, converts fragment SMILES into the common MolCLR/PyG molecular graph
format, and returns L2-normalized graph embeddings that can be used by the
class-level selector's existing ``--sim-metric embedding`` mode.
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


InvalidPolicy = Literal["error", "skip", "zero"]


class MolCLREmbeddingError(RuntimeError):
    """Raised when MolCLR embeddings cannot be produced."""


@dataclass(frozen=True, slots=True)
class MolCLRFailedSmiles:
    """One SMILES that could not be converted into a MolCLR graph."""

    smiles: str
    error: str
    failure_reason: str = "invalid_smiles"


class MolCLRInvalidSmilesError(ValueError):
    """Raised for invalid SMILES when invalid_policy='error'."""

    def __init__(self, failures: list[MolCLRFailedSmiles]) -> None:
        self.failures = tuple(failures)
        examples = "; ".join(f"{failure.smiles}: {failure.error}" for failure in failures[:8])
        super().__init__(
            "Invalid fragment SMILES encountered while building MolCLR graphs: "
            + examples
        )


@dataclass(frozen=True, slots=True)
class ParsedGraph:
    """A parsed molecular graph plus its original SMILES key."""

    smiles: str
    data: Any


@dataclass(frozen=True, slots=True)
class LoadedMolCLRModel:
    """Runtime MolCLR model bundle."""

    model: Any
    device: Any
    checkpoint_path: Path
    encoder_type: str
    model_class: str
    matched_state_keys: int | None
    missing_state_keys: int | None
    unexpected_state_keys: int | None


@dataclass(frozen=True, slots=True)
class MolCLREncodeResult:
    """MolCLR embedding result plus invalid-SMILES diagnostics."""

    embeddings: dict[str, list[float]]
    failed_smiles: tuple[MolCLRFailedSmiles, ...]
    zero_embedding_smiles: tuple[str, ...]
    embedding_dim: int | None


ATOM_NUM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_OTHER",
]
BOND_DIR_LIST = [
    "NONE",
    "ENDUPRIGHT",
    "ENDDOWNRIGHT",
]


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MolCLREmbeddingError(
            "MolCLR GNN embeddings require PyTorch. Install/use the same HPC "
            "environment that contains MolCLR dependencies."
        ) from exc
    return torch


def _require_rdkit() -> tuple[Any, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem import rdchem
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MolCLREmbeddingError(
            "MolCLR GNN embeddings require RDKit to parse fragment SMILES."
        ) from exc
    return Chem, rdchem


def _require_pyg() -> tuple[Any, Any]:
    try:
        from torch_geometric.data import Batch, Data
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MolCLREmbeddingError(
            "MolCLR GNN embeddings require torch_geometric. "
            "Please run inside the MolCLR/PyG-enabled conda environment; the "
            "project will not download PyG at runtime."
        ) from exc
    return Batch, Data


def _normalize_smiles_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_index(values: list[Any], value: Any) -> int:
    try:
        return values.index(value)
    except ValueError:
        return len(values) - 1


def _bond_type_list() -> list[Any]:
    _Chem, rdchem = _require_rdkit()
    return [
        rdchem.BondType.SINGLE,
        rdchem.BondType.DOUBLE,
        rdchem.BondType.TRIPLE,
        rdchem.BondType.AROMATIC,
    ]


def smiles_to_molclr_data(smiles: str) -> Any:
    """Convert one fragment SMILES into a common MolCLR ``torch_geometric`` Data.

    The feature layout follows the widely used MolCLR preprocessing convention:
    atom features are ``[atomic_num_index, chirality_index]`` and directed bond
    features are ``[bond_type_index, bond_direction_index]``.
    """

    torch = _require_torch()
    Chem, _rdchem = _require_rdkit()
    _Batch, Data = _require_pyg()

    normalized = _normalize_smiles_text(smiles)
    if normalized is None:
        raise ValueError("SMILES is empty")

    mol = Chem.MolFromSmiles(normalized)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {normalized!r}")
    if mol.GetNumAtoms() <= 0:
        raise ValueError(f"SMILES has no atoms: {normalized!r}")

    atom_features: list[list[int]] = []
    for atom in mol.GetAtoms():
        atomic_num_index = _safe_index(ATOM_NUM_LIST, int(atom.GetAtomicNum()))
        chirality_index = _safe_index(CHIRALITY_LIST, str(atom.GetChiralTag()))
        atom_features.append([atomic_num_index, chirality_index])

    edge_indices: list[tuple[int, int]] = []
    edge_features: list[list[int]] = []
    bond_type_values = _bond_type_list()
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        bond_type_index = _safe_index(bond_type_values, bond.GetBondType())
        bond_dir_index = _safe_index(BOND_DIR_LIST, str(bond.GetBondDir()))
        feature = [bond_type_index, bond_dir_index]
        edge_indices.extend([(begin, end), (end, begin)])
        edge_features.extend([feature, feature])

    x = torch.tensor(atom_features, dtype=torch.long)
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=normalized)


def _resolve_device(device: str) -> str:
    torch = _require_torch()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise MolCLREmbeddingError("--device cuda was requested, but CUDA is not available.")
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported device: {device!r}. Expected auto/cuda/cpu.")
    return device


def _prepare_molclr_import_path(molclr_root: str | Path) -> Path:
    root = Path(molclr_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"MolCLR root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"MolCLR root is not a directory: {root}")

    candidates = [root]
    if (root / "MolCLR").is_dir():
        candidates.append(root / "MolCLR")
    for candidate in reversed(candidates):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)
    return root


def _load_json_or_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError:
                return None
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
    except Exception:
        return None
    return None


def _load_model_config(molclr_root: Path, ckpt_path: Path) -> dict[str, Any]:
    config_paths = [
        ckpt_path.with_suffix(".json"),
        ckpt_path.with_suffix(".yaml"),
        ckpt_path.with_suffix(".yml"),
        ckpt_path.parent / "config.json",
        ckpt_path.parent / "config.yaml",
        ckpt_path.parent / "config.yml",
        molclr_root / "config.json",
        molclr_root / "config.yaml",
        molclr_root / "config.yml",
        molclr_root / "config_finetune.yaml",
        molclr_root / "config_pretrain.yaml",
    ]
    for path in config_paths:
        payload = _load_json_or_yaml(path)
        if payload:
            model_cfg = payload.get("model", payload)
            return model_cfg if isinstance(model_cfg, dict) else {}
    return {}


def _import_model_class(encoder_type: str) -> type[Any]:
    encoder = encoder_type.lower()
    candidates_by_encoder: dict[str, list[tuple[str, str]]] = {
        "gin": [
            ("models.ginet_finetune", "GINet"),
            ("models.ginet_finetune", "Ginet"),
            ("models.ginet", "GINet"),
            ("models.ginet", "Ginet"),
            ("models.gin", "GINet"),
            ("models.gin", "GIN"),
        ],
        "gcn": [
            ("models.gcn_finetune", "GCN"),
            ("models.gcn_finetune", "GCNNet"),
            ("models.gcn", "GCN"),
            ("models.gcn", "GCNNet"),
        ],
    }
    candidates = candidates_by_encoder.get(encoder)
    if not candidates:
        raise ValueError(f"Unsupported encoder_type={encoder_type!r}; expected 'gin' or 'gcn'.")

    errors: list[str] = []
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: import failed: {exc}")
            continue
        model_class = getattr(module, class_name, None)
        if model_class is not None:
            return model_class
        errors.append(f"{module_name}.{class_name}: class not found")

    raise MolCLREmbeddingError(
        "Could not import a MolCLR model class from --molclr-root. "
        "Tried: "
        + "; ".join(f"{module}.{cls}" for module, cls in candidates)
        + ". Import errors: "
        + " | ".join(errors[:8])
    )


def _extract_state_dict(checkpoint: Any) -> tuple[dict[str, Any] | None, Any | None]:
    torch = _require_torch()
    if isinstance(checkpoint, torch.nn.Module):
        return None, checkpoint
    if not isinstance(checkpoint, dict):
        return None, None

    for key in ("model", "encoder", "net"):
        value = checkpoint.get(key)
        if isinstance(value, torch.nn.Module):
            return None, value

    tensor_like = [value for value in checkpoint.values() if hasattr(value, "shape")]
    if tensor_like:
        return checkpoint, None

    for key in (
        "state_dict",
        "model_state_dict",
        "encoder_state_dict",
        "gnn_state_dict",
        "online_encoder",
        "net",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value, None
    return None, None


def _strip_state_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    prefix_with_dot = prefix + "."
    return {
        key[len(prefix_with_dot) :] if key.startswith(prefix_with_dot) else key: value
        for key, value in state_dict.items()
    }


def _build_default_model_kwargs(model_config: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "num_layer": 5,
        "emb_dim": 300,
        "feat_dim": 512,
        "drop_ratio": 0.0,
        "pool": "mean",
    }
    defaults.update({key: value for key, value in model_config.items() if value is not None})
    return defaults


def _filter_kwargs_for_constructor(model_class: type[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(model_class)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _instantiate_model(model_class: type[Any], model_config: dict[str, Any]) -> Any:
    kwargs = _build_default_model_kwargs(model_config)
    filtered = _filter_kwargs_for_constructor(model_class, kwargs)
    attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
        ((), kwargs),
        ((), filtered),
        ((1,), kwargs),
        ((1,), filtered),
        ((2,), kwargs),
        ((2,), filtered),
    ]
    errors: list[str] = []
    seen: set[str] = set()
    for args, call_kwargs in attempts:
        key = repr((args, sorted(call_kwargs.items())))
        if key in seen:
            continue
        seen.add(key)
        try:
            return model_class(*args, **call_kwargs)
        except Exception as exc:
            errors.append(f"args={args}, kwargs={call_kwargs}: {exc}")
    raise MolCLREmbeddingError(
        f"Could not instantiate MolCLR model class {model_class!r}. "
        "If your MolCLR fork uses different constructor arguments, place a "
        "config.yaml/json next to the checkpoint or use a compatible checkpoint. "
        "Attempts: "
        + " | ".join(errors[:8])
    )


def _load_state_dict_flexibly(model: Any, state_dict: dict[str, Any]) -> tuple[int, int, int]:
    model_keys = set(model.state_dict().keys())
    variants = [state_dict]
    for prefix in ("module", "model", "encoder", "gnn", "online_encoder"):
        stripped = _strip_state_prefix(state_dict, prefix)
        if stripped is not state_dict:
            variants.append(stripped)

    best_error: Exception | None = None
    for candidate in variants:
        matched = len(model_keys.intersection(candidate.keys()))
        if matched <= 0:
            continue
        try:
            incompatible = model.load_state_dict(candidate, strict=False)
        except Exception as exc:
            best_error = exc
            continue
        missing = len(getattr(incompatible, "missing_keys", []))
        unexpected = len(getattr(incompatible, "unexpected_keys", []))
        return matched, missing, unexpected

    if best_error is not None:
        raise MolCLREmbeddingError(f"MolCLR checkpoint state_dict could not be loaded: {best_error}")
    raise MolCLREmbeddingError(
        "MolCLR checkpoint state_dict had no keys matching the instantiated model. "
        "Check --encoder-type and whether --molclr-root matches the checkpoint code."
    )


def load_molclr_model(
    *,
    molclr_root: str | Path,
    molclr_ckpt: str | Path,
    encoder_type: str = "gin",
    device: str = "cuda",
) -> LoadedMolCLRModel:
    """Load a MolCLR model from runtime code and checkpoint paths."""

    torch = _require_torch()
    root = _prepare_molclr_import_path(molclr_root)
    ckpt_path = Path(molclr_ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"MolCLR checkpoint not found: {ckpt_path}")

    resolved_device = torch.device(_resolve_device(device))
    try:
        checkpoint = torch.load(str(ckpt_path), map_location=resolved_device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(ckpt_path), map_location=resolved_device)
    state_dict, checkpoint_model = _extract_state_dict(checkpoint)

    matched_keys: int | None = None
    missing_keys: int | None = None
    unexpected_keys: int | None = None
    if checkpoint_model is not None:
        model = checkpoint_model
        model_class_name = model.__class__.__module__ + "." + model.__class__.__name__
    else:
        model_config = _load_model_config(root, ckpt_path)
        model_class = _import_model_class(encoder_type)
        model = _instantiate_model(model_class, model_config)
        model_class_name = model_class.__module__ + "." + model_class.__name__
        if state_dict is None:
            raise MolCLREmbeddingError(
                f"Could not find a state_dict or serialized model inside checkpoint: {ckpt_path}"
            )
        matched_keys, missing_keys, unexpected_keys = _load_state_dict_flexibly(model, state_dict)

    model = model.to(resolved_device)
    model.eval()
    return LoadedMolCLRModel(
        model=model,
        device=resolved_device,
        checkpoint_path=ckpt_path,
        encoder_type=str(encoder_type),
        model_class=model_class_name,
        matched_state_keys=matched_keys,
        missing_state_keys=missing_keys,
        unexpected_state_keys=unexpected_keys,
    )


def _extract_tensor_from_output(output: Any, batch: Any) -> Any | None:
    torch = _require_torch()
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        preferred_keys = (
            "embedding",
            "embeddings",
            "graph_embedding",
            "graph_embeddings",
            "graph_representation",
            "representation",
            "features",
            "z",
            "h",
            "out",
        )
        for key in preferred_keys:
            if key in output:
                tensor = _extract_tensor_from_output(output[key], batch)
                if tensor is not None:
                    return tensor
        for value in output.values():
            tensor = _extract_tensor_from_output(value, batch)
            if tensor is not None:
                return tensor
    if isinstance(output, (tuple, list)):
        for value in output:
            tensor = _extract_tensor_from_output(value, batch)
            if tensor is not None:
                return tensor
    return None


def _graph_pool_node_tensor(tensor: Any, batch: Any) -> Any | None:
    torch = _require_torch()
    if tensor.ndim != 2 or tensor.shape[0] != batch.x.shape[0]:
        return None
    num_graphs = int(batch.num_graphs)
    pooled = torch.zeros((num_graphs, tensor.shape[1]), dtype=tensor.dtype, device=tensor.device)
    counts = torch.zeros((num_graphs, 1), dtype=tensor.dtype, device=tensor.device)
    graph_index = batch.batch.to(device=tensor.device)
    pooled.index_add_(0, graph_index, tensor)
    ones = torch.ones((tensor.shape[0], 1), dtype=tensor.dtype, device=tensor.device)
    counts.index_add_(0, graph_index, ones)
    return pooled / counts.clamp(min=1.0)


def _tensor_to_graph_embeddings(tensor: Any, batch: Any) -> Any:
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise MolCLREmbeddingError(f"MolCLR embedding tensor must be 2D, got shape={tuple(tensor.shape)}")
    if tensor.shape[0] == int(batch.num_graphs):
        return tensor
    pooled = _graph_pool_node_tensor(tensor, batch)
    if pooled is not None:
        return pooled
    raise MolCLREmbeddingError(
        "MolCLR output tensor does not align with graph batch size or node count: "
        f"tensor_shape={tuple(tensor.shape)}, num_graphs={int(batch.num_graphs)}, "
        f"num_nodes={int(batch.x.shape[0])}"
    )


def _method_call_attempts(method: Callable[..., Any], batch: Any) -> list[Callable[[], Any]]:
    return [
        lambda: method(batch),
        lambda: method(batch.x, batch.edge_index, batch.edge_attr, batch.batch),
        lambda: method(batch.x, batch.edge_index, batch.batch),
    ]


def _encode_batch_with_model(model: Any, batch: Any) -> Any:
    torch = _require_torch()
    method_names = [
        "forward_cl",
        "encode",
        "get_graph_embedding",
        "get_graph_embeddings",
        "get_embedding",
        "get_embeddings",
        "encoder",
        "forward",
    ]
    errors: list[str] = []
    with torch.inference_mode():
        for method_name in method_names:
            method = getattr(model, method_name, None)
            if method is None:
                continue
            for attempt in _method_call_attempts(method, batch):
                try:
                    output = attempt()
                except Exception as exc:
                    errors.append(f"{method_name}: {exc}")
                    continue
                tensor = _extract_tensor_from_output(output, batch)
                if tensor is None:
                    errors.append(f"{method_name}: no tensor-like embedding found")
                    continue
                return _tensor_to_graph_embeddings(tensor, batch)

    raise MolCLREmbeddingError(
        "Could not extract graph embeddings from the MolCLR model. "
        "Tried methods forward_cl/encode/get_graph_embedding/get_embeddings/encoder/forward. "
        "Recent errors: "
        + " | ".join(errors[-8:])
    )


def _l2_normalized_float_lists(tensor: Any) -> list[list[float]]:
    torch = _require_torch()
    tensor = tensor.detach()
    if not torch.is_floating_point(tensor):
        tensor = tensor.to(torch.float32)
    tensor = tensor.to(torch.float32)
    norms = torch.linalg.vector_norm(tensor, ord=2, dim=1, keepdim=True).clamp(min=1e-12)
    normalized = tensor / norms
    if not torch.all(torch.isfinite(normalized)):
        raise MolCLREmbeddingError("MolCLR embeddings contained NaN or Inf after normalization.")
    return normalized.cpu().tolist()


def _batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _unique_normalized_smiles(smiles_list: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in smiles_list:
        smiles = _normalize_smiles_text(value)
        if smiles is None:
            continue
        if smiles in seen:
            continue
        seen.add(smiles)
        ordered.append(smiles)
    return ordered


def _encode_valid_graphs(
    parsed_graphs: list[ParsedGraph],
    *,
    loaded: LoadedMolCLRModel,
    batch_size: int,
) -> dict[str, list[float]]:
    if not parsed_graphs:
        return {}
    _Batch, _Data = _require_pyg()
    embeddings: dict[str, list[float]] = {}
    for graph_batch in _batched(parsed_graphs, batch_size):
        pyg_batch = _Batch.from_data_list([item.data for item in graph_batch]).to(loaded.device)
        raw_embeddings = _encode_batch_with_model(loaded.model, pyg_batch)
        normalized_embeddings = _l2_normalized_float_lists(raw_embeddings)
        if len(normalized_embeddings) != len(graph_batch):
            raise MolCLREmbeddingError(
                "MolCLR returned a different number of embeddings than input graphs: "
                f"{len(normalized_embeddings)} vs {len(graph_batch)}"
            )
        for graph, embedding in zip(graph_batch, normalized_embeddings):
            embeddings[graph.smiles] = [float(value) for value in embedding]
    return embeddings


def encode_smiles_list_with_failures(
    smiles_list: Iterable[str],
    molclr_root: str | Path,
    molclr_ckpt: str | Path,
    encoder_type: str = "gin",
    batch_size: int = 64,
    device: str = "cuda",
    invalid_policy: InvalidPolicy = "error",
) -> MolCLREncodeResult:
    """Encode fragment SMILES with a MolCLR pretrained GNN encoder.

    Returns normalized embeddings and invalid-SMILES diagnostics. Invalid SMILES
    are handled according to ``invalid_policy``: ``error`` raises
    :class:`MolCLRInvalidSmilesError`, ``skip`` omits them from embeddings, and
    ``zero`` writes zero vectors with the same dimension as valid embeddings.
    """

    if invalid_policy not in {"error", "skip", "zero"}:
        raise ValueError("--invalid-policy must be one of: error, skip, zero")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")

    unique_smiles = _unique_normalized_smiles(smiles_list)
    if not unique_smiles:
        return MolCLREncodeResult(
            embeddings={},
            failed_smiles=(),
            zero_embedding_smiles=(),
            embedding_dim=None,
        )

    loaded = load_molclr_model(
        molclr_root=molclr_root,
        molclr_ckpt=molclr_ckpt,
        encoder_type=encoder_type,
        device=device,
    )

    parsed_graphs: list[ParsedGraph] = []
    invalids: list[MolCLRFailedSmiles] = []
    for smiles in unique_smiles:
        try:
            parsed_graphs.append(ParsedGraph(smiles=smiles, data=smiles_to_molclr_data(smiles)))
        except Exception as exc:
            invalids.append(MolCLRFailedSmiles(smiles=smiles, error=str(exc)))

    if invalids and invalid_policy == "error":
        raise MolCLRInvalidSmilesError(invalids)

    embeddings = _encode_valid_graphs(parsed_graphs, loaded=loaded, batch_size=int(batch_size))
    embedding_dim = len(next(iter(embeddings.values()))) if embeddings else None

    zero_embedding_smiles: list[str] = []
    if invalids and invalid_policy == "zero":
        if embedding_dim is None:
            dummy_embeddings = _encode_valid_graphs(
                [ParsedGraph(smiles="C", data=smiles_to_molclr_data("C"))],
                loaded=loaded,
                batch_size=1,
            )
            embedding_dim = len(dummy_embeddings["C"])
        zero = [0.0] * int(embedding_dim)
        for failure in invalids:
            embeddings[failure.smiles] = list(zero)
            zero_embedding_smiles.append(failure.smiles)

    return MolCLREncodeResult(
        embeddings=embeddings,
        failed_smiles=tuple(invalids),
        zero_embedding_smiles=tuple(zero_embedding_smiles),
        embedding_dim=int(embedding_dim) if embedding_dim is not None else None,
    )


def encode_smiles_list(
    smiles_list: Iterable[str],
    molclr_root: str | Path,
    molclr_ckpt: str | Path,
    encoder_type: str = "gin",
    batch_size: int = 64,
    device: str = "cuda",
    invalid_policy: InvalidPolicy = "error",
) -> dict[str, list[float]]:
    """Encode fragment SMILES and return only the SMILES-to-embedding mapping."""

    result = encode_smiles_list_with_failures(
        smiles_list,
        molclr_root=molclr_root,
        molclr_ckpt=molclr_ckpt,
        encoder_type=encoder_type,
        batch_size=batch_size,
        device=device,
        invalid_policy=invalid_policy,
    )
    return result.embeddings


__all__ = [
    "InvalidPolicy",
    "LoadedMolCLRModel",
    "MolCLREncodeResult",
    "MolCLREmbeddingError",
    "MolCLRFailedSmiles",
    "MolCLRInvalidSmilesError",
    "encode_smiles_list",
    "encode_smiles_list_with_failures",
    "load_molclr_model",
    "smiles_to_molclr_data",
]
