"""Deterministic CPU preprocessing for the BACE native GINE bridge.

COMRECGC's random walk is stateful and cannot be split across independent step
ranges.  This module therefore accelerates only the pure graph-to-molecule
preprocessing below that state machine.  Requests are consumed in the original
order, worker completion order is never exposed, and no worker owns an RNG or
CUDA context.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Mapping, Sequence

from src.baselines.gcfexplainer_mutagenicity_adapter import (
    decode_generated_fullgraph,
)
from src.data.molecular_graph_featurizer import (
    MolecularFeatureSchema,
    MolecularGraphFeatures,
    MolecularGraphFeaturizer,
)


PREPROCESS_ENGINE = "ordered_bounded_rdkit_process_pool_v1"


@dataclass(frozen=True, slots=True)
class NativeGraphPreprocessRequest:
    """Pickle-safe immutable input for one pure preprocessing operation."""

    cache_key: str
    cache_kind: str
    source_index: int
    num_nodes: int
    x: tuple[tuple[float, ...], ...]
    edge_index: tuple[tuple[int, ...], tuple[int, ...]]
    node_origin: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NativeGraphPreprocessResult:
    """One decoded/featurized graph or a fail-closed reason."""

    cache_key: str
    cache_kind: str
    features: MolecularGraphFeatures | None
    failure_reason: str | None


_WORKER_SOURCE_RECORDS: tuple[Mapping[str, Any], ...] | None = None
_WORKER_GRAPH_SCHEMA: Any | None = None
_WORKER_FEATURIZER: MolecularGraphFeaturizer | None = None


def initialize_preprocess_worker(
    source_records: Sequence[Mapping[str, Any]],
    graph_schema: Any,
    feature_schema: MolecularFeatureSchema,
) -> None:
    """Initialize one spawn worker without loading CUDA or model state."""

    global _WORKER_SOURCE_RECORDS, _WORKER_GRAPH_SCHEMA, _WORKER_FEATURIZER
    _WORKER_SOURCE_RECORDS = tuple(dict(record) for record in source_records)
    _WORKER_GRAPH_SCHEMA = graph_schema
    _WORKER_FEATURIZER = MolecularGraphFeaturizer(feature_schema)


def preprocess_native_graph(
    request: NativeGraphPreprocessRequest,
    *,
    source_records: Sequence[Mapping[str, Any]],
    graph_schema: Any,
    featurizer: MolecularGraphFeaturizer,
) -> NativeGraphPreprocessResult:
    """Decode and featurize one request without mutating caller-owned graphs."""

    if not 0 <= request.source_index < len(source_records):
        return NativeGraphPreprocessResult(
            cache_key=request.cache_key,
            cache_kind=request.cache_kind,
            features=None,
            failure_reason="source_index_out_of_range",
        )
    graph = SimpleNamespace(
        x=request.x,
        edge_index=request.edge_index,
        num_nodes=request.num_nodes,
        gcf_node_origin=request.node_origin,
    )
    decoded = decode_generated_fullgraph(
        graph,
        source_record=source_records[request.source_index],
        schema=graph_schema,
    )
    canonical = str(decoded.canonical_smiles or "").strip()
    if not decoded.decode_ok or not canonical:
        return NativeGraphPreprocessResult(
            cache_key=request.cache_key,
            cache_kind=request.cache_kind,
            features=None,
            failure_reason=str(
                decoded.failure_reason or "native_graph_decode_failed"
            ),
        )
    try:
        features = featurizer.featurize(canonical)
    except Exception as exc:
        return NativeGraphPreprocessResult(
            cache_key=request.cache_key,
            cache_kind=request.cache_kind,
            features=None,
            failure_reason=f"gine_featurize_failed:{type(exc).__name__}",
        )
    return NativeGraphPreprocessResult(
        cache_key=request.cache_key,
        cache_kind=request.cache_kind,
        features=features,
        failure_reason=None,
    )


def preprocess_native_graph_worker(
    request: NativeGraphPreprocessRequest,
) -> NativeGraphPreprocessResult:
    """Process-pool target initialized by :func:`initialize_preprocess_worker`."""

    if (
        _WORKER_SOURCE_RECORDS is None
        or _WORKER_GRAPH_SCHEMA is None
        or _WORKER_FEATURIZER is None
    ):
        raise RuntimeError("BACE native preprocessing worker was not initialized.")
    return preprocess_native_graph(
        request,
        source_records=_WORKER_SOURCE_RECORDS,
        graph_schema=_WORKER_GRAPH_SCHEMA,
        featurizer=_WORKER_FEATURIZER,
    )


def ordered_bounded_submit(
    executor: Executor,
    function: Any,
    values: Iterable[Any],
    *,
    max_inflight: int,
) -> Iterator[Any]:
    """Yield executor results in input order with a strict submission bound."""

    limit = int(max_inflight)
    if limit <= 0:
        raise ValueError("max_inflight must be positive.")
    iterator = iter(values)
    pending: deque[Future[Any]] = deque()
    exhausted = False
    while pending or not exhausted:
        while len(pending) < limit and not exhausted:
            try:
                value = next(iterator)
            except StopIteration:
                exhausted = True
                break
            pending.append(executor.submit(function, value))
        if pending:
            # Waiting on the oldest future hides worker completion order from
            # the scientific caller and provides deterministic backpressure.
            yield pending.popleft().result()


__all__ = [
    "NativeGraphPreprocessRequest",
    "NativeGraphPreprocessResult",
    "PREPROCESS_ENGINE",
    "initialize_preprocess_worker",
    "ordered_bounded_submit",
    "preprocess_native_graph",
    "preprocess_native_graph_worker",
]
