"""Official-semantics NeuroSED model helpers for TasteMolNet.

The training forward follows GREED's ``NormSEDModel`` while the exported
checkpoint intentionally remains a plain embedding-model ``state_dict``.  The
official GCFExplainer fork loads that same state dictionary into its
``NormGEDModel``.  Keeping the two forwards explicit prevents an accidental
change to either the training objective or the downstream runner contract.

Upstream authority (pinned by the execution config and model card):

* idea-iitd/greed ``1c756f49625abb62c9f6de5b0059876a4c7499c1``;
* idea-iitd/greed-expts ``e85423dc943fda1979811e7449846efffec2a1e1``.

The upstream GREED implementation is MIT licensed.  This module reuses the
project's already bundled GCFExplainer ``EmbedModel`` implementation, whose
parameter names are the runner's checkpoint schema.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Any, Mapping


GREED_COMMIT = "1c756f49625abb62c9f6de5b0059876a4c7499c1"
GREED_EXPERIMENTS_COMMIT = "e85423dc943fda1979811e7449846efffec2a1e1"
GREED_MODELS_SHA256 = "c5653dd9eeec1add8d6ae6253c30908df5ab8962ea0d9f9a6f25d32c393e0e70"
GREED_TRAIN_SHA256 = "8e4d425d9d63e0aa56d5a1e6e25738f511ca7b52b08ac297fcf2c1678bdf9e28"
GCF_FORK_MODELS_SHA256 = "8025f0cdc187625fb9d469a9ec0791694f3e923ee94e3d9084cb74a066397a60"
GCF_FORK_DISTANCE_SHA256 = "d81182ccb31ef0fc5aef6a95a7debc6c17e3b495596e4ee3ff1642adf29745c3"

DEFAULT_NUM_LAYERS = 8
DEFAULT_HIDDEN_DIM = 64
DEFAULT_OUTPUT_DIM = 64
DEFAULT_MAX_GRAD_NORM = 0.1


class TasteNeuroSEDDependencyError(RuntimeError):
    """Raised when the AutoDL NeuroSED runtime is unavailable."""


def runtime_stack() -> tuple[Any, Any, Any]:
    """Import torch, PyG, and the bundled runner model only at runtime."""

    try:
        import torch
        import torch_geometric as tg
        from baselines.gcfexplainer_official.neurosed import models as gcf_models
    except ImportError as exc:  # pragma: no cover - runtime environment check.
        raise TasteNeuroSEDDependencyError(
            "Taste NeuroSED requires torch, torch_geometric, and the bundled "
            "GCFExplainer NeuroSED model."
        ) from exc
    return torch, tg, gcf_models


def build_training_model(
    *,
    input_dim: int,
    device: str | Any,
    num_layers: int = DEFAULT_NUM_LAYERS,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    output_dim: int = DEFAULT_OUTPUT_DIM,
) -> Any:
    """Build the official GREED ``NormSEDModel`` training forward.

    GCFExplainer's bundled fork only retains ``NormGEDModel``.  Both models
    contain exactly one identically named ``embed_model``.  Defining the
    directional SED forward around that same class therefore preserves the
    official state-dictionary schema without modifying the upstream subtree.
    """

    torch, _tg, gcf_models = runtime_stack()

    class NormSEDModel(gcf_models.SiameseModel):
        def __init__(self) -> None:
            super().__init__(device)
            self.embed_model = gcf_models.EmbedModel(
                int(num_layers),
                int(input_dim),
                int(hidden_dim),
                int(output_dim),
                conv="gin",
                pool="add",
            )

        def forward_emb(self, gx: Any, hx: Any) -> Any:
            return torch.norm(torch.nn.functional.relu(gx - hx), dim=-1)

    return NormSEDModel().to(device)


def build_runner_model(
    *,
    input_dim: int,
    device: str | Any,
    num_layers: int = DEFAULT_NUM_LAYERS,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    output_dim: int = DEFAULT_OUTPUT_DIM,
) -> Any:
    """Build the exact ``NormGEDModel`` used by GCFExplainer's loader."""

    _torch, _tg, gcf_models = runtime_stack()
    return gcf_models.NormGEDModel(
        int(num_layers),
        int(input_dim),
        int(hidden_dim),
        int(output_dim),
        device=device,
        conv="gin",
        pool="add",
    ).to(device)


def interval_loss(lb: Any, ub: Any, prediction: Any) -> Any:
    """GREED interval criterion, unchanged from the pinned implementation."""

    torch, _tg, _gcf_models = runtime_stack()
    return torch.mean(
        torch.nn.functional.relu(lb - prediction) ** 2
        + torch.nn.functional.relu(prediction - ub) ** 2
    )


def load_state_dict_bytes(data: bytes, *, map_location: str | Any = "cpu") -> Mapping[str, Any]:
    """Load one plain runner-compatible state dictionary from held bytes."""

    import io

    torch, _tg, _gcf_models = runtime_stack()
    try:
        payload = torch.load(io.BytesIO(data), map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch before ``weights_only``.
        payload = torch.load(io.BytesIO(data), map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("NeuroSED checkpoint is not a plain state dictionary")
    if "embed_model.pre.weight" not in payload:
        raise ValueError("NeuroSED checkpoint lacks the runner input projection")
    return payload


def load_runner_checkpoint(
    checkpoint: str | Path,
    *,
    input_dim: int,
    device: str | Any,
) -> Any:
    """Load a checkpoint with the exact model/schema used by ``distance.py``."""

    torch, _tg, _gcf_models = runtime_stack()
    path = Path(checkpoint)
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - older AutoDL torch.
        state = torch.load(path, map_location=device)
    model = build_runner_model(input_dim=input_dim, device=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def model_contract(input_dim: int) -> dict[str, Any]:
    """Return the immutable architecture and upstream-authority contract."""

    return {
        "training_model": "NormSEDModel",
        "runner_model": "NormGEDModel",
        "checkpoint_schema": "plain_torch_state_dict",
        "state_dict_isomorphic": True,
        "num_layers": DEFAULT_NUM_LAYERS,
        "input_dim": int(input_dim),
        "hidden_dim": DEFAULT_HIDDEN_DIM,
        "output_dim": DEFAULT_OUTPUT_DIM,
        "convolution": "GINConv",
        "pooling": "global_add_pool",
        "training_forward": "l2_norm(relu(query_embedding-parent_embedding))",
        "runner_forward": "l2_norm(query_embedding-parent_embedding)",
        "interval_loss": "mean(relu(lb-pred)^2+relu(pred-ub)^2)",
        "distance_normalization": "divide_by_sum_graph_element_counts",
        "graph_element_count": "num_nodes+num_directed_edges/2",
        "official_greed_commit": GREED_COMMIT,
        "official_greed_experiments_commit": GREED_EXPERIMENTS_COMMIT,
        "official_greed_models_sha256": GREED_MODELS_SHA256,
        "official_greed_train_sha256": GREED_TRAIN_SHA256,
        "bundled_gcf_models_sha256": GCF_FORK_MODELS_SHA256,
        "bundled_gcf_distance_sha256": GCF_FORK_DISTANCE_SHA256,
    }


def verify_bundled_runner_sources() -> dict[str, Any]:
    """Rehash the exact bundled GCF model and loader used at runtime."""

    _torch, _tg, gcf_models = runtime_stack()
    models_path = Path(gcf_models.__file__).resolve(strict=True)
    distance_path = models_path.parents[1] / "distance.py"
    if not distance_path.is_file():
        raise ValueError("bundled GCF distance.py is unavailable")

    def digest(path: Path) -> str:
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        return value

    models_sha256 = digest(models_path)
    distance_sha256 = digest(distance_path)
    if models_sha256 != GCF_FORK_MODELS_SHA256:
        raise ValueError("bundled GCF NeuroSED model source SHA256 changed")
    if distance_sha256 != GCF_FORK_DISTANCE_SHA256:
        raise ValueError("bundled GCF distance loader source SHA256 changed")
    return {
        "models_path": str(models_path),
        "models_sha256": models_sha256,
        "distance_path": str(distance_path),
        "distance_sha256": distance_sha256,
    }


__all__ = [
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_MAX_GRAD_NORM",
    "DEFAULT_NUM_LAYERS",
    "DEFAULT_OUTPUT_DIM",
    "GCF_FORK_DISTANCE_SHA256",
    "GCF_FORK_MODELS_SHA256",
    "GREED_COMMIT",
    "GREED_EXPERIMENTS_COMMIT",
    "GREED_MODELS_SHA256",
    "GREED_TRAIN_SHA256",
    "TasteNeuroSEDDependencyError",
    "build_runner_model",
    "build_training_model",
    "interval_loss",
    "load_runner_checkpoint",
    "load_state_dict_bytes",
    "model_contract",
    "runtime_stack",
    "verify_bundled_runner_sources",
]
