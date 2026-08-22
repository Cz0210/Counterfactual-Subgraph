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

from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from src.baselines.gcfexplainer_mutagenicity_adapter import (
    decode_generated_fullgraph,
)
from src.data.molecular_graph_dataset import MolecularGraphData, collate_molecular_graphs
from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer
from src.oracles.gnn_oracle import load_gnn_checkpoint_bundle, sha256_file
from src.baselines.comrecgc.bace_preprocessing import (
    NativeGraphPreprocessRequest,
    NativeGraphPreprocessResult,
    PREPROCESS_ENGINE,
    initialize_preprocess_worker,
    ordered_bounded_submit,
    preprocess_native_graph,
    preprocess_native_graph_worker,
)
from src.baselines.comrecgc.contracts import stable_json_sha256


INVALID_GRAPH_POLICY = "source_class_reject_no_fallback_v1"
ADAPTER_VERSION = "bace_frozen_gine_native_graph_adapter_v1"
LEGACY_PREPROCESS_ENGINE = "legacy_sequential_rdkit_v1"


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
        preprocess_engine: str = LEGACY_PREPROCESS_ENGINE,
        preprocess_workers: int = 0,
        preprocess_max_inflight: int = 64,
        source_cache_capacity: int = 0,
        candidate_cache_capacity: int = 0,
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
        self.preprocess_engine = str(preprocess_engine)
        if self.preprocess_engine not in {
            LEGACY_PREPROCESS_ENGINE,
            PREPROCESS_ENGINE,
        }:
            raise ValueError(
                "Unknown BACE native preprocessing engine: "
                f"{self.preprocess_engine!r}."
            )
        self.preprocess_workers = int(preprocess_workers)
        self.preprocess_max_inflight = int(preprocess_max_inflight)
        self.source_cache_capacity = int(source_cache_capacity)
        self.candidate_cache_capacity = int(candidate_cache_capacity)
        if self.preprocess_workers < 0:
            raise ValueError("preprocess_workers cannot be negative.")
        if self.preprocess_max_inflight <= 0:
            raise ValueError("preprocess_max_inflight must be positive.")
        if self.source_cache_capacity < 0 or self.candidate_cache_capacity < 0:
            raise ValueError("BACE preprocessing cache capacities cannot be negative.")
        if self.preprocess_engine == LEGACY_PREPROCESS_ENGINE and any(
            (
                self.preprocess_workers,
                self.source_cache_capacity,
                self.candidate_cache_capacity,
            )
        ):
            raise ValueError(
                "Legacy BACE preprocessing rejects worker/cache settings; "
                "select the equivalent process-pool engine explicitly."
            )
        self._preprocess_executor: ProcessPoolExecutor | None = None
        self._source_cache: OrderedDict[str, NativeGraphPreprocessResult] = (
            OrderedDict()
        )
        self._candidate_cache: OrderedDict[str, NativeGraphPreprocessResult] = (
            OrderedDict()
        )
        self.preprocess_stats: Counter[str] = Counter()
        self.preprocess_wall_seconds = 0.0
        feature_schema_payload = metadata["feature_schema"].to_dict()
        self.feature_schema_sha256 = str(feature_schema_payload["schema_sha256"])
        self._source_content_sha256: tuple[str | None, ...] = (
            tuple(
                self._record_content_sha256(record)
                for record in self.source_records
            )
            if self.preprocess_engine == PREPROCESS_ENGINE
            else tuple(None for _record in self.source_records)
        )
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

    @staticmethod
    def _plain_rows(value: Any, *, cast: Any) -> tuple[tuple[Any, ...], ...]:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        return tuple(tuple(cast(item) for item in row) for row in value)

    @classmethod
    def _content_sha256(
        cls,
        *,
        x: Sequence[Sequence[Any]],
        edge_index: Sequence[Sequence[Any]],
        num_nodes: int,
    ) -> str:
        if len(edge_index) != 2:
            raise ValueError("Native graph edge_index must contain two rows.")
        directed_edges = sorted(
            (int(source), int(target))
            for source, target in zip(edge_index[0], edge_index[1], strict=True)
        )
        return stable_json_sha256(
            {
                "num_nodes": int(num_nodes),
                "x": [[float(item) for item in row] for row in x],
                "directed_edges": [list(edge) for edge in directed_edges],
            }
        )

    @classmethod
    def _record_content_sha256(cls, record: Mapping[str, Any]) -> str:
        return cls._content_sha256(
            x=record["x"],
            edge_index=record["edge_index"],
            num_nodes=int(record["num_nodes"]),
        )

    def _request(self, graph: Any) -> NativeGraphPreprocessRequest:
        source_identity = getattr(graph, "gcf_origin_index", None)
        if source_identity is None:
            source_identity = getattr(graph, "comrecgc_source_index", None)
        if source_identity is None:
            raise ValueError("missing_source_index")
        source_index = _scalar_int(source_identity)
        x = self._plain_rows(getattr(graph, "x"), cast=float)
        edge_index = self._plain_rows(getattr(graph, "edge_index"), cast=int)
        if len(edge_index) != 2:
            raise ValueError("Native graph edge_index must contain two rows.")
        lineage = getattr(graph, "comrecgc_node_origin", None)
        if lineage is None:
            lineage = getattr(graph, "gcf_node_origin", None)
        if lineage is None:
            raise ValueError("generated_missing_source_lineage")
        if hasattr(lineage, "detach"):
            lineage = lineage.detach().cpu()
        if hasattr(lineage, "tolist"):
            lineage = lineage.tolist()
        node_origin = tuple(int(value) for value in lineage)
        num_nodes = int(getattr(graph, "num_nodes", len(x)))
        content_sha256 = self._content_sha256(
            x=x,
            edge_index=edge_index,
            num_nodes=num_nodes,
        )
        is_source = bool(
            0 <= source_index < len(self._source_content_sha256)
            and content_sha256 == self._source_content_sha256[source_index]
            and node_origin == tuple(range(num_nodes))
        )
        cache_kind = "source" if is_source else "candidate"
        cache_key = stable_json_sha256(
            {
                "schema_version": "bace_native_preprocess_cache_key_v1",
                "global_graph_content_sha256": content_sha256,
                # Parent/source metadata is provenance, not graph identity, but
                # it changes chemical sidecar reconstruction and must therefore
                # bind a cached preprocessing result.
                "source_index": source_index,
                "node_origin": list(node_origin),
                "feature_schema_sha256": self.feature_schema_sha256,
                "checkpoint_id": self.checkpoint_id,
            }
        )
        return NativeGraphPreprocessRequest(
            cache_key=cache_key,
            cache_kind=cache_kind,
            source_index=source_index,
            num_nodes=num_nodes,
            x=x,
            edge_index=(edge_index[0], edge_index[1]),
            node_origin=node_origin,
        )

    def _cache_for(
        self, kind: str
    ) -> tuple[OrderedDict[str, NativeGraphPreprocessResult], int]:
        if kind == "source":
            return self._source_cache, self.source_cache_capacity
        if kind == "candidate":
            return self._candidate_cache, self.candidate_cache_capacity
        raise ValueError(f"Unknown native preprocessing cache kind: {kind!r}")

    def _cache_get(
        self, request: NativeGraphPreprocessRequest
    ) -> NativeGraphPreprocessResult | None:
        cache, capacity = self._cache_for(request.cache_kind)
        if capacity <= 0 or request.cache_key not in cache:
            self.preprocess_stats[f"{request.cache_kind}_cache_miss_count"] += 1
            return None
        result = cache.pop(request.cache_key)
        cache[request.cache_key] = result
        self.preprocess_stats[f"{request.cache_kind}_cache_hit_count"] += 1
        return result

    def _cache_put(self, result: NativeGraphPreprocessResult) -> None:
        cache, capacity = self._cache_for(result.cache_kind)
        if capacity <= 0:
            return
        cache.pop(result.cache_key, None)
        cache[result.cache_key] = result
        while len(cache) > capacity:
            cache.popitem(last=False)
            self.preprocess_stats[f"{result.cache_kind}_cache_eviction_count"] += 1

    def _executor(self) -> ProcessPoolExecutor:
        if self.preprocess_workers <= 0:
            raise RuntimeError("Inline preprocessing does not create an executor.")
        if self._preprocess_executor is None:
            self._preprocess_executor = ProcessPoolExecutor(
                max_workers=self.preprocess_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_preprocess_worker,
                initargs=(
                    self.source_records,
                    self.graph_schema,
                    self.metadata["feature_schema"],
                ),
            )
        return self._preprocess_executor

    def close(self) -> None:
        """Release optional CPU workers without touching scientific artifacts."""

        if self._preprocess_executor is not None:
            self._preprocess_executor.shutdown(wait=True, cancel_futures=False)
            self._preprocess_executor = None

    def _optimized_preprocess(
        self, graphs: Sequence[Any]
    ) -> list[NativeGraphPreprocessResult]:
        started = time.monotonic()
        requests: list[NativeGraphPreprocessRequest | None] = []
        resolved: dict[str, NativeGraphPreprocessResult] = {}
        uncached: OrderedDict[str, NativeGraphPreprocessRequest] = OrderedDict()
        invalid_results: list[NativeGraphPreprocessResult | None] = []
        for index, graph in enumerate(graphs):
            try:
                request = self._request(graph)
            except Exception as exc:
                reason = str(exc).strip()
                if reason not in {
                    "missing_source_index",
                    "generated_missing_source_lineage",
                }:
                    reason = f"missing_source_index:{type(exc).__name__}"
                requests.append(None)
                invalid_results.append(
                    NativeGraphPreprocessResult(
                        cache_key=f"uncacheable-{index}",
                        cache_kind="candidate",
                        features=None,
                        failure_reason=reason,
                    )
                )
                continue
            requests.append(request)
            invalid_results.append(None)
            cached = self._cache_get(request)
            if cached is not None:
                resolved[request.cache_key] = cached
            elif request.cache_key not in uncached:
                uncached[request.cache_key] = request
            else:
                self.preprocess_stats["within_batch_coalesced_count"] += 1

        if uncached:
            values = list(uncached.values())
            self.preprocess_stats["unique_preprocess_count"] += len(values)
            if self.preprocess_workers > 0:
                produced = ordered_bounded_submit(
                    self._executor(),
                    preprocess_native_graph_worker,
                    values,
                    max_inflight=self.preprocess_max_inflight,
                )
                self.preprocess_stats["process_pool_submitted_count"] += len(values)
            else:
                produced = (
                    preprocess_native_graph(
                        request,
                        source_records=self.source_records,
                        graph_schema=self.graph_schema,
                        featurizer=self.featurizer,
                    )
                    for request in values
                )
                self.preprocess_stats["inline_preprocess_count"] += len(values)
            for result in produced:
                expected = uncached[result.cache_key]
                if result.cache_kind != expected.cache_kind:
                    raise RuntimeError(
                        "BACE preprocessing worker changed cache-kind provenance."
                    )
                resolved[result.cache_key] = result
                self._cache_put(result)

        output: list[NativeGraphPreprocessResult] = []
        for request, invalid in zip(requests, invalid_results, strict=True):
            if invalid is not None:
                output.append(invalid)
            else:
                assert request is not None
                output.append(resolved[request.cache_key])
        self.preprocess_stats["request_count"] += len(graphs)
        self.preprocess_wall_seconds += time.monotonic() - started
        return output

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

    @staticmethod
    def _portable_features(
        features: Any, *, row_index: int
    ) -> MolecularGraphData:
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
        if self.preprocess_engine == LEGACY_PREPROCESS_ENGINE:
            for index, graph in enumerate(graphs):
                smiles, failure = self._decode(graph)
                if failure is not None or smiles is None:
                    self.decode_failures[
                        str(failure or "unknown_decode_failure")
                    ] += 1
                    continue
                try:
                    valid_graphs.append(
                        self._portable_graph(smiles, row_index=index)
                    )
                except Exception as exc:
                    self.decode_failures[
                        f"gine_featurize_failed:{type(exc).__name__}"
                    ] += 1
                    continue
                valid_positions.append(index)
        else:
            for index, result in enumerate(self._optimized_preprocess(graphs)):
                if result.failure_reason is not None or result.features is None:
                    self.decode_failures[
                        str(result.failure_reason or "unknown_decode_failure")
                    ] += 1
                    continue
                valid_graphs.append(
                    self._portable_features(result.features, row_index=index)
                )
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
            "preprocess_engine": self.preprocess_engine,
            "preprocess_workers": self.preprocess_workers,
            "preprocess_max_inflight": self.preprocess_max_inflight,
            "source_cache_capacity": self.source_cache_capacity,
            "candidate_cache_capacity": self.candidate_cache_capacity,
            "source_cache_entry_count": len(self._source_cache),
            "candidate_cache_entry_count": len(self._candidate_cache),
            "preprocess_stats": dict(sorted(self.preprocess_stats.items())),
            "preprocess_wall_seconds": self.preprocess_wall_seconds,
            "preprocess_order_preserved": True,
            "preprocess_rng_calls_added": 0,
            "preprocess_cuda_context_in_workers": False,
        }


__all__ = [
    "ADAPTER_VERSION",
    "BACEFrozenGINENativeGraphAdapter",
    "INVALID_GRAPH_POLICY",
    "LEGACY_PREPROCESS_ENGINE",
]
