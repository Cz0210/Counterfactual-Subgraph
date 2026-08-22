"""Frozen-BACE-GINE adapter for native graph-edit baseline runtimes.

GCFExplainer and COMRECGC represent an edited molecule as a PyG graph with a
one-hot atom vocabulary plus project-owned source-lineage attributes.  The
paper-facing BACE classifier instead consumes the categorical molecular graph
schema frozen in the GINE checkpoint bundle.  This module is the only bridge
between those representations.

The adapter deliberately reconstructs a connected, sanitized molecule before
calling GINE.  A graph that cannot be reconstructed is assigned source-class
probability one, so it can never become a counterfactual candidate.  It is not
repaired, silently dropped, or scored by a fallback classifier.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.baselines.gcfexplainer_mutagenicity_adapter import (
    decode_generated_fullgraph,
)
from src.data.molecular_graph_dataset import MolecularGraphData, collate_molecular_graphs
from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer
from src.oracles.gnn_oracle import load_gnn_checkpoint_bundle, sha256_file


INVALID_GRAPH_POLICY = "source_class_reject_no_fallback_v1"
ADAPTER_VERSION = "bace_frozen_gine_native_graph_adapter_v1"


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency.
        raise RuntimeError("The BACE native GINE adapter requires PyTorch.") from exc
    return torch


def _scalar_int(value: Any) -> int:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("Native graph source index must contain one value.")
        value = value[0]
    return int(value)


def _to_data_list(batch: Any) -> list[Any]:
    if hasattr(batch, "to_data_list"):
        return list(batch.to_data_list())
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes, bytearray)):
        return list(batch)
    return [batch]


class BACEFrozenGINENativeGraphAdapter:
    """Expose the frozen calibrated BACE GINE through the official GNN API.

    Official GCFExplainer/COMRECGC callers expect ``(node_hidden,
    graph_hidden, log_probs)`` and interpret internal class ``0`` as the source
    class and internal class ``1`` as the counterfactual target.  Project BACE
    labels are the opposite ordering (target=0, source=1), hence the explicit
    ``[1, 0]`` reorder below.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        source_records: Sequence[Mapping[str, Any]],
        graph_schema: Any,
        device: str | Any,
    ) -> None:
        torch = _require_torch()
        root = Path(checkpoint_dir).expanduser().resolve()
        model, metadata = load_gnn_checkpoint_bundle(root, device=device)
        card = dict(metadata["model_card"])
        if str(card.get("dataset", "")).strip().lower() != "bace":
            raise ValueError("Native BACE adapter requires a BACE checkpoint bundle.")
        if str(card.get("backbone", "")).strip().lower() != "gine":
            raise ValueError("Native BACE adapter requires backbone=gine.")
        if int(card.get("num_classes", -1)) != 2 or int(card.get("source_label", -1)) != 1:
            raise ValueError("Native BACE adapter requires num_classes=2/source_label=1.")
        if card.get("rf_oracle_used") is not False:
            raise ValueError("RF provenance is forbidden in the BACE native adapter.")
        if not source_records:
            raise ValueError("Native BACE adapter requires source-lineage records.")
        self.model = model
        self.metadata = metadata
        self.card = card
        self.checkpoint_dir = root
        self.checkpoint_id = str(metadata["checkpoint_id"])
        self.temperature = float(metadata["temperature_scaling"].get("temperature", 1.0))
        if not self.temperature > 0.0:
            raise ValueError("Frozen BACE temperature must be positive.")
        self.source_records = tuple(dict(record) for record in source_records)
        self.graph_schema = graph_schema
        self.device = device
        self.featurizer = MolecularGraphFeaturizer(metadata["feature_schema"])
        self.hidden_dim = int(getattr(model.config, "hidden_dim"))
        self.edge_feature_dim = len(metadata["feature_schema"].edge_fields)
        self.decode_failures: Counter[str] = Counter()
        self.decode_success_count = 0
        self.call_count = 0
        self.model.to(device)
        self.model.eval()
        # Keep one tensor anchor for dtype/device creation without guessing the
        # runtime's current CUDA index.
        self._parameter = next(self.model.parameters())
        self._torch = torch

    def eval(self) -> "BACEFrozenGINENativeGraphAdapter":
        self.model.eval()
        return self

    def to(self, device: str | Any) -> "BACEFrozenGINENativeGraphAdapter":
        self.device = device
        self.model.to(device)
        self._parameter = next(self.model.parameters())
        return self

    def _decode(self, graph: Any) -> tuple[str | None, str | None]:
        try:
            source_identity = getattr(graph, "gcf_origin_index", None)
            if source_identity is None:
                source_identity = getattr(graph, "comrecgc_source_index")
            source_index = _scalar_int(source_identity)
        except Exception as exc:
            return None, f"missing_source_index:{type(exc).__name__}"
        if not 0 <= source_index < len(self.source_records):
            return None, "source_index_out_of_range"
        # Official COMRECGC edits copy the project-owned lineage fields, while
        # GCFExplainer uses gcf_node_origin directly.  The shared codec expects
        # the latter name, so expose the authoritative COMRECGC lineage without
        # changing graph tensors or parent identity.
        comrecgc_lineage = getattr(graph, "comrecgc_node_origin", None)
        if comrecgc_lineage is not None:
            graph.gcf_node_origin = comrecgc_lineage
        decoded = decode_generated_fullgraph(
            graph,
            source_record=self.source_records[source_index],
            schema=self.graph_schema,
        )
        if not decoded.decode_ok or not str(decoded.canonical_smiles).strip():
            return None, str(decoded.failure_reason or "native_graph_decode_failed")
        return str(decoded.canonical_smiles), None

    def _portable_graph(self, smiles: str, *, row_index: int) -> MolecularGraphData:
        features = self.featurizer.featurize(smiles)
        return MolecularGraphData(
            x=features.node_features,
            edge_index=features.edge_index,
            edge_attr=features.edge_features,
            y=1,
            molecule_id=f"native-bace-{row_index}",
            smiles=features.canonical_smiles,
            split="train_native_graph_edit",
            graph_sha256=features.graph_sha256,
        )

    def __call__(self, data: Any, edge_weight: Any = None) -> tuple[Any, Any, Any]:
        del edge_weight
        torch = self._torch
        graphs = _to_data_list(data)
        if not graphs:
            raise ValueError("Native BACE GINE adapter received an empty batch.")
        self.call_count += 1
        valid_positions: list[int] = []
        valid_graphs: list[MolecularGraphData] = []
        for index, graph in enumerate(graphs):
            smiles, failure = self._decode(graph)
            if failure is not None or smiles is None:
                self.decode_failures[str(failure or "unknown_decode_failure")] += 1
                continue
            try:
                valid_graphs.append(self._portable_graph(smiles, row_index=index))
            except Exception as exc:
                self.decode_failures[f"gine_featurize_failed:{type(exc).__name__}"] += 1
                continue
            valid_positions.append(index)

        graph_hidden = torch.zeros(
            (len(graphs), self.hidden_dim),
            dtype=self._parameter.dtype,
            device=self._parameter.device,
        )
        # Project-label order is [target=0, source=1].  Invalid graphs remain
        # source by construction and cannot satisfy the native CF gate.
        project_logits = torch.empty(
            (len(graphs), 2),
            dtype=self._parameter.dtype,
            device=self._parameter.device,
        )
        project_logits[:, 0] = -20.0
        project_logits[:, 1] = 20.0
        if valid_graphs:
            batch = collate_molecular_graphs(
                valid_graphs, edge_feature_dim=self.edge_feature_dim
            ).to(self.device)
            valid_hidden = self.model.encode_graph(batch)
            valid_logits = self.model.classifier(valid_hidden) / self.temperature
            positions = torch.tensor(
                valid_positions, dtype=torch.long, device=valid_logits.device
            )
            graph_hidden.index_copy_(0, positions, valid_hidden)
            project_logits.index_copy_(0, positions, valid_logits)
            self.decode_success_count += len(valid_graphs)

        internal_logits = project_logits[:, torch.tensor([1, 0], device=project_logits.device)]
        internal_log_probs = torch.log_softmax(internal_logits, dim=-1)
        node_hidden = torch.zeros(
            (sum(int(getattr(graph, "num_nodes", 0)) for graph in graphs), self.hidden_dim),
            dtype=self._parameter.dtype,
            device=self._parameter.device,
        )
        return node_hidden, graph_hidden, internal_log_probs

    def provenance(self) -> dict[str, Any]:
        return {
            "adapter_version": ADAPTER_VERSION,
            "checkpoint_path": str(self.checkpoint_dir),
            "checkpoint_sha256": sha256_file(self.checkpoint_dir / "model.pt"),
            "oracle_checkpoint_hash": self.checkpoint_id,
            "oracle_backend": "gnn",
            "classifier_family": "gine",
            "rf_oracle_used": False,
            "source_label": 1,
            "num_classes": 2,
            "temperature": self.temperature,
            "label_mapping": "project[target0,source1]_to_native[source0,target1]",
            "invalid_graph_policy": INVALID_GRAPH_POLICY,
            "model_retrained": False,
            "call_count": self.call_count,
            "decode_success_count": self.decode_success_count,
            "decode_failure_count": int(sum(self.decode_failures.values())),
            "decode_failure_reasons": dict(sorted(self.decode_failures.items())),
        }


__all__ = [
    "ADAPTER_VERSION",
    "BACEFrozenGINENativeGraphAdapter",
    "INVALID_GRAPH_POLICY",
]
