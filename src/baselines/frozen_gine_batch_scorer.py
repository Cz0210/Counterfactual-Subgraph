"""Order-preserving batched scoring for a frozen project GINE oracle.

The scorer deliberately caches only an *entire ordered batch*.  A partial hit
must never remove rows or change the batch shape because the native
GCFExplainer graph identity is built from the exact GINE embedding bytes.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence


def _tensor_snapshot(value: Any, *, row_digests: bool = True) -> dict[str, Any]:
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        if getattr(tensor, "is_sparse", False):
            tensor = tensor.to_dense()
        array = tensor.contiguous().numpy()
    else:
        import numpy as np

        array = np.ascontiguousarray(value)
    identity = hashlib.sha256()
    identity.update(str(array.dtype).encode("ascii"))
    identity.update(json.dumps(list(array.shape)).encode("ascii"))
    identity.update(array.tobytes(order="C"))
    payload: dict[str, Any] = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": identity.hexdigest(),
    }
    if row_digests and array.ndim > 0:
        if array.shape[0] == 0:
            payload["row_sha256"] = []
        else:
            rows = array.reshape(array.shape[0], -1)
            payload["row_sha256"] = [
                hashlib.sha256(row.tobytes(order="C")).hexdigest() for row in rows
            ]
    return payload


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _graph_feature_identity(graph: Any) -> str:
    return _stable_sha256(
        {
            "x": _tensor_snapshot(graph.x, row_digests=False),
            "edge_index": _tensor_snapshot(graph.edge_index, row_digests=False),
            "edge_attr": _tensor_snapshot(graph.edge_attr, row_digests=False),
            "graph_sha256": str(getattr(graph, "graph_sha256", "")),
        }
    )


@dataclass(frozen=True)
class FrozenGINEBatchScore:
    """One complete ordered-batch result."""

    graph_hidden: Any
    project_logits: Any
    diagnostic_trace: Mapping[str, Any] | None


class FrozenGINEBatchScorer:
    """Score a full ordered graph batch without deduplication or chunking."""

    def __init__(
        self,
        *,
        model: Any,
        device: Any,
        temperature: float,
        checkpoint_id: str,
        collate_fn: Callable[[Sequence[Any]], Any],
        cache_capacity: int = 0,
        diagnostic_trace: bool = False,
    ) -> None:
        if float(temperature) <= 0:
            raise ValueError("Frozen GINE temperature must be positive.")
        if isinstance(cache_capacity, bool) or int(cache_capacity) < 0:
            raise ValueError("Frozen GINE batch cache capacity cannot be negative.")
        self.model = model
        self.device = device
        self.temperature = float(temperature)
        self.checkpoint_id = str(checkpoint_id)
        self.collate_fn = collate_fn
        self.cache_capacity = int(cache_capacity)
        self.diagnostic_trace_enabled = bool(diagnostic_trace)
        self._cache: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self.calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.scored_rows = 0
        self.last_trace: Mapping[str, Any] | None = None

    def _key(
        self, graphs: Sequence[Any], context: Mapping[str, Any] | None
    ) -> str:
        return _stable_sha256(
            {
                "schema_version": "frozen_gine_exact_ordered_batch_key_v1",
                "checkpoint_id": self.checkpoint_id,
                "temperature": self.temperature,
                "ordered_graphs": [_graph_feature_identity(row) for row in graphs],
                "context": dict(context or {}),
            }
        )

    def _put(self, key: str, graph_hidden: Any, logits: Any) -> None:
        if self.cache_capacity <= 0:
            return
        self._cache.pop(key, None)
        self._cache[key] = (
            graph_hidden.detach().cpu().clone(),
            logits.detach().cpu().clone(),
        )
        while len(self._cache) > self.cache_capacity:
            self._cache.popitem(last=False)

    def score(
        self,
        graphs: Sequence[Any],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> FrozenGINEBatchScore:
        if not graphs:
            raise ValueError("Frozen GINE scorer received an empty batch.")
        self.calls += 1
        cache_key = self._key(graphs, context) if self.cache_capacity > 0 else None
        if cache_key is not None and cache_key in self._cache:
            hidden_cpu, logits_cpu = self._cache.pop(cache_key)
            self._cache[cache_key] = (hidden_cpu, logits_cpu)
            parameter = next(self.model.parameters())
            graph_hidden = hidden_cpu.to(parameter.device)
            logits = logits_cpu.to(parameter.device)
            self.cache_hits += len(graphs)
            batch = None
        else:
            batch = self.collate_fn(graphs).to(self.device)
            graph_hidden = self.model.encode_graph(batch)
            logits = self.model.classifier(graph_hidden) / self.temperature
            self.cache_misses += len(graphs)
            self.scored_rows += len(graphs)
            if cache_key is not None:
                self._put(cache_key, graph_hidden, logits)

        trace: Mapping[str, Any] | None = None
        if self.diagnostic_trace_enabled:
            batch_payload: dict[str, Any] | None = None
            if batch is not None:
                batch_payload = {
                    name: _tensor_snapshot(getattr(batch, name))
                    for name in ("x", "edge_index", "edge_attr", "batch")
                }
            trace = {
                "schema_version": 1,
                "ordered_graph_count": len(graphs),
                "ordered_graph_sha256": [
                    _graph_feature_identity(row) for row in graphs
                ],
                "batch": batch_payload,
                "graph_hidden": _tensor_snapshot(graph_hidden),
                "project_logits": {
                    **_tensor_snapshot(logits),
                    "values": logits.detach().cpu().tolist(),
                },
            }
        self.last_trace = trace
        return FrozenGINEBatchScore(graph_hidden, logits, trace)

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "calls": self.calls,
            "cache_capacity": self.cache_capacity,
            "cache_entries": len(self._cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "scored_rows": self.scored_rows,
            "cache_scope": "exact_complete_ordered_batch_v1",
            "partial_row_reuse": False,
            "deduplication": False,
            "chunking": False,
        }


__all__ = ["FrozenGINEBatchScore", "FrozenGINEBatchScorer"]
