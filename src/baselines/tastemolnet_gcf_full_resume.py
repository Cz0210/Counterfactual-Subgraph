"""TasteMolNet T12 deterministic VRRW identity and restart substrate.

This module is deliberately dataset-specific.  It does not replace the
official GCFExplainer walk, edit operators, transition weights, NeuroSED
coverage, or three-class GINE scorer.  It changes only the process-local
``hash(embedding.tobytes())`` registry key at the project boundary: the T12
registry key is the SHA-256 of the canonical, parent-free attributed graph
already used by the TasteMolNet ComRecGC route.

The second responsibility is a persistent, reopenable checkpoint for the
complete official mutable walk and every RNG that can affect it.  Production
checkpoints are permitted only at 10k and 20k.  A separate bounded canary
purpose is available solely to prove exact uninterrupted-versus-new-process
replay on the real GPU before production is released.

Importing this module performs no scientific I/O and does not require CUDA.
"""

from __future__ import annotations

from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterator, Mapping, MutableMapping, Sequence
import uuid

from src.baselines.tastemolnet_comrecgc_smoke import canonical_attributed_graph
from src.baselines.tastemolnet_gcf_smoke import (
    _VRRW_PROGRESS_STATE_FIELDS,
    _capture_rng_state,
    _embedding_sha256,
    _restore_rng_state,
    _semantic_sha256,
    score_and_candidate,
)


STAGE = "T12_GCF_FULL"
DATASET = "tastemolnet"
METHOD = "GCFExplainer"
SOURCE_LABEL = 1
NUM_CLASSES = 3
PRODUCTION_TOTAL_STEPS = 20_000
PRODUCTION_CHECKPOINT_CURSORS = frozenset({10_000, 20_000})
GRAPH_IDENTITY_CONTRACT = "canonical_parent_free_attributed_graph_sha256_v1"
CHECKPOINT_SCHEMA = "tastemolnet_t12_vrrw_checkpoint_v1"
CHECKPOINT_MANIFEST_SCHEMA = "tastemolnet_t12_vrrw_checkpoint_manifest_v1"
BRIDGE_SCHEMA = "tastemolnet_t12_gcf_stable_bridge_v1"
CANARY_OBSERVATION_SCHEMA = "tastemolnet_t12_gpu_replay_observation_v1"
CANARY_GATE_SCHEMA = "tastemolnet_t12_gpu_replay_gate_v1"
CANARY_PASS_MARKER = "[TASTE_T12_GPU_CROSS_PROCESS_REPLAY_CANARY_PASS]"

GINE_CANONICAL_REUSE_RTOL = 1e-5
GINE_CANONICAL_REUSE_ATOL = 1e-7

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_GPU_UUID = re.compile(r"^GPU-[A-Za-z0-9-]+$")
_CHECKPOINT_FILE = re.compile(r"^checkpoint-(\d{8})\.pt$")
_CANARY_PROCESS_FIELDS = frozenset(
    {"pid", "start_ticks", "command_sha256", "executable_sha256", "cwd_sha256"}
)
_CANARY_SCIENCE_FIELDS = frozenset(
    {
        "completed_steps",
        "traversed_graph_identities",
        "candidate_frequency_order",
        "graph_map_sha256",
        "graph_index_map_sha256",
        "transitions_sha256",
        "current_graph_identity",
        "bridge_state_sha256",
        "adapter_state_sha256",
        "action_counts_sha256",
        "rng_state_sha256",
        "generated_to_original_coverage_sha256",
        "official_native_result_semantic_sha256",
    }
)


class TasteGCFFullResumeError(RuntimeError):
    """T12 identity, checkpoint, or exact-replay evidence is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _pretty_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise TasteGCFFullResumeError(f"{field} must be lowercase SHA-256")
    return value


def _require_uuid4(value: Any, *, field: str) -> str:
    if type(value) is not str:
        raise TasteGCFFullResumeError(f"{field} must be canonical UUIDv4")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise TasteGCFFullResumeError(f"{field} must be canonical UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise TasteGCFFullResumeError(f"{field} must be canonical UUIDv4")
    return value


def _native_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TasteGCFFullResumeError(f"{field} must be an integer >= {minimum}")
    return value


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TasteGCFFullResumeError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise TasteGCFFullResumeError(f"{field} must be finite")
    return result


def _stable_graph_hash(value: Any, *, field: str) -> str:
    return _require_sha256(value, field=field)


def _lineage_sha256(graph: Any, *, graph_identity_sha256: str) -> str:
    origin = getattr(graph, "gcf_origin_index", None)
    node_origin = getattr(graph, "gcf_node_origin", None)
    if origin is None or node_origin is None:
        raise TasteGCFFullResumeError("T12 graph is missing GCF lineage")
    if hasattr(origin, "detach"):
        origin = origin.detach().cpu().reshape(-1).tolist()
    if hasattr(node_origin, "detach"):
        node_origin = node_origin.detach().cpu().reshape(-1).tolist()
    if not isinstance(origin, (list, tuple)):
        origin = [origin]
    if not isinstance(node_origin, (list, tuple)):
        node_origin = [node_origin]
    if (
        len(origin) != 1
        or type(origin[0]) is not int
        or origin[0] < 0
        or len(node_origin) != int(getattr(graph, "num_nodes", -1))
        or any(type(value) is not int or value < -1 for value in node_origin)
    ):
        raise TasteGCFFullResumeError("T12 GCF lineage is malformed")
    return _sha256_bytes(
        _canonical_bytes(
            {
                "schema_version": "tastemolnet_t12_gcf_lineage_v1",
                "graph_identity_sha256": graph_identity_sha256,
                "source_cohort_index": origin[0],
                "node_origin": list(node_origin),
            }
        )
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalized_absolute(path: str | Path, *, field: str) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute() or Path(os.path.abspath(value)) != value:
        raise TasteGCFFullResumeError(f"{field} must be normalized and absolute")
    return value


@dataclass(frozen=True, slots=True)
class T12StableGraphRecord:
    graph_identity_sha256: str
    collision_payload: Mapping[str, Any]
    probabilities: tuple[float, float, float]
    prediction: int
    score: float
    candidate: bool
    valid_fullgraph: bool
    failure_reason: str
    coverage_vector: tuple[int, ...]
    coverage_ratio: float
    canonical_embedding_dtype: str
    canonical_embedding_values: tuple[float, ...]
    canonical_embedding_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_identity_sha256": self.graph_identity_sha256,
            "collision_payload": dict(self.collision_payload),
            "probabilities": list(self.probabilities),
            "prediction": self.prediction,
            "score": self.score,
            "candidate": self.candidate,
            "valid_fullgraph": self.valid_fullgraph,
            "failure_reason": self.failure_reason,
            "coverage_vector": list(self.coverage_vector),
            "coverage_ratio": self.coverage_ratio,
            "canonical_embedding_dtype": self.canonical_embedding_dtype,
            "canonical_embedding_values": list(self.canonical_embedding_values),
            "canonical_embedding_sha256": self.canonical_embedding_sha256,
        }


class T12StableGCFBridge:
    """Narrow stable-key patch around official GCF importance calls.

    The official module still requests a hash for each returned GINE embedding.
    ``call`` queues the structural graph identities in that exact batch order;
    ``calculate_hash`` consumes one identity and verifies the corresponding
    embedding bytes.  Thus no parent metadata or Python hash seed enters the
    registry while call-order drift remains fail-closed.
    """

    def __init__(
        self,
        *,
        adapter: Any,
        vrrw: Any,
        importance: Any,
        neurosed_model: Any,
        original_graph_element_counts: Any,
        distance_threshold: float,
        parent_count: int,
        feature_atomic_numbers: Sequence[Any],
    ) -> None:
        self.adapter = adapter
        self.vrrw = vrrw
        self.importance = importance
        self.neurosed_model = neurosed_model
        self.original_graph_element_counts = original_graph_element_counts
        self.distance_threshold = _finite(
            distance_threshold, field="T12 NeuroSED distance threshold"
        )
        if self.distance_threshold < 0.0:
            raise TasteGCFFullResumeError(
                "T12 NeuroSED distance threshold must be nonnegative"
            )
        self.parent_count = _native_int(parent_count, field="T12 parent_count", minimum=1)
        self.feature_atomic_numbers = tuple(
            _native_int(value, field="T12 feature atomic number", minimum=1)
            for value in feature_atomic_numbers
        )
        if (
            not self.feature_atomic_numbers
            or len(set(self.feature_atomic_numbers)) != len(self.feature_atomic_numbers)
        ):
            raise TasteGCFFullResumeError(
                "T12 feature atomic-number vocabulary must be unique"
            )
        self.records: dict[str, T12StableGraphRecord] = {}
        self.lineage_occurrences: dict[str, Counter[str]] = {}
        self._pending_hashes: deque[tuple[str, str]] = deque()
        self.call_count = 0
        self.evaluated_graph_count = 0
        self.calculate_hash_count = 0
        self.distance_call_count = 0
        self.distance_evaluated_graph_count = 0
        self.canonical_row_reuse_count = 0

    def _assert_idle(self) -> None:
        if self._pending_hashes:
            raise TasteGCFFullResumeError(
                "official GCF did not consume every queued structural identity"
            )

    def call(
        self,
        graphs: Sequence[Any],
        _importance_args: Mapping[str, Any],
    ) -> tuple[Any, Any, Any]:
        import numpy as np

        self._assert_idle()
        values = list(graphs)
        if not values:
            raise TasteGCFFullResumeError("T12 GCF received an empty graph batch")
        batch = self.adapter.score(values)
        probabilities = np.asarray(batch.probabilities)
        embeddings = np.asarray(batch.graph_embeddings)
        if probabilities.shape != (len(values), NUM_CLASSES):
            raise TasteGCFFullResumeError("T12 GINE probabilities are not three-class")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(values):
            raise TasteGCFFullResumeError("T12 GINE embeddings are unaligned")
        coverage = self.importance.neurosed_threshold_coverage_estimation(
            self.neurosed_model,
            values,
            self.original_graph_element_counts,
            self.distance_threshold,
        )
        self.distance_call_count += 1
        self.distance_evaluated_graph_count += len(values)
        if (
            tuple(coverage.shape) != (len(values), self.parent_count)
            or not bool(self.vrrw.torch.isfinite(coverage).all().item())
            or not bool(((coverage == 0) | (coverage == 1)).all().item())
        ):
            raise TasteGCFFullResumeError(
                "T12 official generated-to-original NeuroSED coverage changed"
            )
        dense_coverage = coverage.to_dense() if getattr(coverage, "is_sparse", False) else coverage
        dense_coverage = dense_coverage.cpu()
        valid = tuple(batch.valid_fullgraphs)
        failures = tuple(batch.failure_reasons)
        if len(valid) != len(values) or len(failures) != len(values):
            raise TasteGCFFullResumeError("T12 adapter evidence is unaligned")

        canonical_parts: list[tuple[float, float]] = []
        for index, graph in enumerate(values):
            identity = canonical_attributed_graph(
                graph, feature_atomic_numbers=self.feature_atomic_numbers
            )
            graph_hash = _stable_graph_hash(
                identity.graph_identity_sha256, field="T12 graph identity"
            )
            collision = identity.collision_payload()
            row = tuple(float(value) for value in probabilities[index].tolist())
            score, prediction, candidate_condition = score_and_candidate(row)
            coverage_vector = tuple(
                int(value) for value in dense_coverage[index].tolist()
            )
            if (
                len(coverage_vector) != self.parent_count
                or any(value not in (0, 1) for value in coverage_vector)
            ):
                raise TasteGCFFullResumeError("T12 NeuroSED coverage is not binary")
            coverage_ratio = sum(coverage_vector) / float(self.parent_count)
            raw_embedding = np.ascontiguousarray(embeddings[index])
            if raw_embedding.ndim != 1 or raw_embedding.dtype.kind != "f":
                raise TasteGCFFullResumeError("T12 GINE embedding is not one float row")
            observed = T12StableGraphRecord(
                graph_identity_sha256=graph_hash,
                collision_payload=collision,
                probabilities=row,
                prediction=prediction,
                score=score,
                candidate=bool(valid[index] and candidate_condition),
                valid_fullgraph=bool(valid[index]),
                failure_reason=str(failures[index]),
                coverage_vector=coverage_vector,
                coverage_ratio=coverage_ratio,
                canonical_embedding_dtype=raw_embedding.dtype.str,
                canonical_embedding_values=tuple(
                    float(value) for value in raw_embedding.tolist()
                ),
                canonical_embedding_sha256=_embedding_sha256(raw_embedding),
            )
            previous = self.records.get(graph_hash)
            if previous is None:
                record = observed
                self.records[graph_hash] = record
            else:
                previous_embedding = np.asarray(
                    previous.canonical_embedding_values,
                    dtype=np.dtype(previous.canonical_embedding_dtype),
                )
                if (
                    dict(previous.collision_payload) != collision
                    or previous.prediction != observed.prediction
                    or previous.candidate is not observed.candidate
                    or previous.valid_fullgraph is not observed.valid_fullgraph
                    or previous.failure_reason != observed.failure_reason
                    or previous.coverage_vector != observed.coverage_vector
                    or previous.canonical_embedding_dtype
                    != observed.canonical_embedding_dtype
                    or not np.allclose(
                        np.asarray(previous.probabilities, dtype=np.float64),
                        np.asarray(observed.probabilities, dtype=np.float64),
                        rtol=GINE_CANONICAL_REUSE_RTOL,
                        atol=GINE_CANONICAL_REUSE_ATOL,
                    )
                    or not np.allclose(
                        previous_embedding,
                        raw_embedding,
                        rtol=GINE_CANONICAL_REUSE_RTOL,
                        atol=GINE_CANONICAL_REUSE_ATOL,
                    )
                ):
                    raise TasteGCFFullResumeError(
                        "one T12 structural identity changed GINE/NeuroSED semantics"
                    )
                record = previous
                self.canonical_row_reuse_count += 1
            self._pending_hashes.append(
                (graph_hash, _embedding_sha256(raw_embedding))
            )
            lineage = _lineage_sha256(graph, graph_identity_sha256=graph_hash)
            self.lineage_occurrences.setdefault(graph_hash, Counter())[lineage] += 1
            canonical_parts.append((record.score, record.coverage_ratio))
        self.call_count += 1
        self.evaluated_graph_count += len(values)
        return np.asarray(canonical_parts, dtype=float), embeddings, coverage

    def calculate_hash(self, graph_embedding: Any) -> str:
        if not self._pending_hashes:
            raise TasteGCFFullResumeError(
                "official GCF requested a hash without a scored graph"
            )
        graph_hash, expected_embedding_sha256 = self._pending_hashes.popleft()
        if _embedding_sha256(graph_embedding) != expected_embedding_sha256:
            raise TasteGCFFullResumeError(
                "official GCF graph/embedding call order changed"
            )
        self.calculate_hash_count += 1
        return graph_hash

    def is_graph_counterfactual(self, graph_hash: Any) -> bool:
        if type(graph_hash) is not str or graph_hash not in self.records:
            raise TasteGCFFullResumeError(
                "official GCF queried an unknown structural identity"
            )
        return self.records[graph_hash].candidate

    def checkpoint_state(self) -> dict[str, Any]:
        self._assert_idle()
        return {
            "schema_version": BRIDGE_SCHEMA,
            "records": {
                key: value.to_dict() for key, value in sorted(self.records.items())
            },
            "lineage_occurrences": {
                key: dict(sorted(values.items()))
                for key, values in sorted(self.lineage_occurrences.items())
            },
            "call_count": self.call_count,
            "evaluated_graph_count": self.evaluated_graph_count,
            "calculate_hash_count": self.calculate_hash_count,
            "distance_call_count": self.distance_call_count,
            "distance_evaluated_graph_count": self.distance_evaluated_graph_count,
            "canonical_row_reuse_count": self.canonical_row_reuse_count,
            "parent_count": self.parent_count,
            "distance_threshold_hex": self.distance_threshold.hex(),
            "feature_atomic_numbers": list(self.feature_atomic_numbers),
            "graph_identity_contract": GRAPH_IDENTITY_CONTRACT,
            "generated_to_original_neurosed": True,
            "python_builtin_hash_used": False,
            "embedding_identity_used": False,
        }

    def restore_checkpoint_state(self, payload: Mapping[str, Any]) -> None:
        import numpy as np

        self._assert_idle()
        expected = {
            "schema_version", "records", "lineage_occurrences", "call_count", "evaluated_graph_count",
            "calculate_hash_count", "distance_call_count",
            "distance_evaluated_graph_count", "canonical_row_reuse_count",
            "parent_count", "distance_threshold_hex", "feature_atomic_numbers",
            "graph_identity_contract", "generated_to_original_neurosed",
            "python_builtin_hash_used", "embedding_identity_used",
        }
        if type(payload) is not dict or set(payload) != expected:
            raise TasteGCFFullResumeError("T12 bridge checkpoint keys changed")
        if (
            payload.get("schema_version") != BRIDGE_SCHEMA
            or payload.get("parent_count") != self.parent_count
            or payload.get("distance_threshold_hex") != self.distance_threshold.hex()
            or payload.get("feature_atomic_numbers") != list(self.feature_atomic_numbers)
            or payload.get("graph_identity_contract") != GRAPH_IDENTITY_CONTRACT
            or payload.get("generated_to_original_neurosed") is not True
            or payload.get("python_builtin_hash_used") is not False
            or payload.get("embedding_identity_used") is not False
            or type(payload.get("records")) is not dict
            or type(payload.get("lineage_occurrences")) is not dict
        ):
            raise TasteGCFFullResumeError("T12 bridge checkpoint semantics changed")
        records: dict[str, T12StableGraphRecord] = {}
        for key, raw in payload["records"].items():
            key = _stable_graph_hash(key, field="T12 checkpoint graph identity")
            if type(raw) is not dict or set(raw) != set(T12StableGraphRecord(
                graph_identity_sha256=key,
                collision_payload={},
                probabilities=(0.0, 0.0, 1.0),
                prediction=2,
                score=1.0,
                candidate=True,
                valid_fullgraph=True,
                failure_reason="",
                coverage_vector=(),
                coverage_ratio=0.0,
                canonical_embedding_dtype="<f4",
                canonical_embedding_values=(),
                canonical_embedding_sha256="0" * 64,
            ).to_dict()):
                raise TasteGCFFullResumeError("T12 bridge record schema changed")
            collision = raw.get("collision_payload")
            if type(collision) is not dict:
                raise TasteGCFFullResumeError("T12 bridge collision evidence is invalid")
            # Reuse the already-reviewed identity payload contract directly;
            # reconstruction of a PyG graph is neither required nor allowed.
            from src.baselines.tastemolnet_comrecgc_smoke import _identity_graph_sha256

            if _identity_graph_sha256(collision) != key:
                raise TasteGCFFullResumeError("T12 bridge graph identity drifted")
            probabilities = raw.get("probabilities")
            score, prediction, candidate_condition = score_and_candidate(probabilities)
            coverage_vector_raw = raw.get("coverage_vector")
            embedding_values = raw.get("canonical_embedding_values")
            if (
                raw.get("graph_identity_sha256") != key
                or type(coverage_vector_raw) is not list
                or len(coverage_vector_raw) != self.parent_count
                or any(type(value) is not int or value not in (0, 1) for value in coverage_vector_raw)
                or type(raw.get("valid_fullgraph")) is not bool
                or type(raw.get("candidate")) is not bool
                or raw["candidate"] is not bool(raw["valid_fullgraph"] and candidate_condition)
                or raw.get("prediction") != prediction
                or raw.get("score") != score
                or raw.get("coverage_ratio")
                != sum(coverage_vector_raw) / float(self.parent_count)
                or type(raw.get("failure_reason")) is not str
                or type(raw.get("canonical_embedding_dtype")) is not str
                or type(embedding_values) is not list
                or not embedding_values
            ):
                raise TasteGCFFullResumeError("T12 bridge record semantics changed")
            try:
                embedding = np.asarray(
                    embedding_values, dtype=np.dtype(raw["canonical_embedding_dtype"])
                )
            except (TypeError, ValueError) as exc:
                raise TasteGCFFullResumeError("T12 checkpoint embedding is invalid") from exc
            if (
                embedding.ndim != 1
                or embedding.dtype.kind != "f"
                or _embedding_sha256(embedding)
                != _require_sha256(
                    raw.get("canonical_embedding_sha256"),
                    field="T12 checkpoint embedding SHA",
                )
            ):
                raise TasteGCFFullResumeError("T12 checkpoint embedding bytes changed")
            records[key] = T12StableGraphRecord(
                graph_identity_sha256=key,
                collision_payload=dict(collision),
                probabilities=tuple(float(value) for value in probabilities),
                prediction=prediction,
                score=score,
                candidate=raw["candidate"],
                valid_fullgraph=raw["valid_fullgraph"],
                failure_reason=raw["failure_reason"],
                coverage_vector=tuple(coverage_vector_raw),
                coverage_ratio=float(raw["coverage_ratio"]),
                canonical_embedding_dtype=embedding.dtype.str,
                canonical_embedding_values=tuple(float(value) for value in embedding.tolist()),
                canonical_embedding_sha256=raw["canonical_embedding_sha256"],
            )
        self.records = records
        lineages: dict[str, Counter[str]] = {}
        for key, raw in payload["lineage_occurrences"].items():
            if key not in records or type(raw) is not dict or not raw:
                raise TasteGCFFullResumeError("T12 checkpoint lineage domain changed")
            counter: Counter[str] = Counter()
            for lineage, count in raw.items():
                lineage = _require_sha256(lineage, field="T12 checkpoint lineage")
                counter[lineage] = _native_int(
                    count, field="T12 checkpoint lineage count", minimum=1
                )
            lineages[key] = counter
        if set(lineages) != set(records):
            raise TasteGCFFullResumeError("T12 checkpoint lost stable-key lineage")
        self.lineage_occurrences = lineages
        for field in (
            "call_count", "evaluated_graph_count", "calculate_hash_count",
            "distance_call_count", "distance_evaluated_graph_count",
            "canonical_row_reuse_count",
        ):
            setattr(self, field, _native_int(payload.get(field), field=f"T12 bridge {field}"))
        if (
            self.evaluated_graph_count != self.calculate_hash_count
            or self.distance_evaluated_graph_count != self.evaluated_graph_count
            or sum(
                sum(values.values()) for values in self.lineage_occurrences.values()
            )
            != self.evaluated_graph_count
        ):
            raise TasteGCFFullResumeError("T12 bridge checkpoint lost call closure")

    def report(self) -> dict[str, Any]:
        state = self.checkpoint_state()
        return {
            **{key: value for key, value in state.items() if key != "records"},
            "unique_graph_count": len(self.records),
            "unique_lineage_count": sum(
                len(values) for values in self.lineage_occurrences.values()
            ),
            "lineage_occurrence_count": sum(
                sum(values.values()) for values in self.lineage_occurrences.values()
            ),
            "registry_identity": GRAPH_IDENTITY_CONTRACT,
            "dedup_identity": GRAPH_IDENTITY_CONTRACT,
            "lineage_identity": GRAPH_IDENTITY_CONTRACT,
            "record_cache_identity": GRAPH_IDENTITY_CONTRACT,
            "strict_counterfactual_count": sum(row.candidate for row in self.records.values()),
            "destination_prediction_counts": {
                str(label): sum(
                    row.candidate and row.prediction == label
                    for row in self.records.values()
                )
                for label in (0, 2)
            },
        }

    @contextmanager
    def installed(self) -> Iterator[None]:
        self._assert_idle()
        originals = {
            "importance_call": self.importance.call,
            "calculate_hash": self.vrrw.calculate_hash,
            "is_graph_counterfactual": self.vrrw.is_graph_counterfactual,
        }
        self.importance.call = self.call
        self.vrrw.calculate_hash = self.calculate_hash
        self.vrrw.is_graph_counterfactual = self.is_graph_counterfactual
        try:
            yield
            self._assert_idle()
        finally:
            self.importance.call = originals["importance_call"]
            self.vrrw.calculate_hash = originals["calculate_hash"]
            self.vrrw.is_graph_counterfactual = originals["is_graph_counterfactual"]


_IDENTITY_FIELDS = frozenset(
    {
        "schema_version", "stage", "purpose", "attempt_id", "generation_token",
        "total_steps", "checkpoint_cursor", "source_cohort_sha256",
        "train_split_sha256", "model_checkpoint_sha256", "model_config_sha256",
        "neurosed_checkpoint_sha256", "neurosed_distance_threshold_hex",
        "neurosed_threshold_authority_sha256", "official_source_inventory_sha256",
        "execution_commit", "execution_tree", "runtime_identity_sha256", "gpu_uuid",
        "device", "graph_identity_contract", "seed", "alpha_hex", "teleport_hex",
        "sample_size", "candidate_capacity", "train_loaded", "calibration_loaded",
        "test_loaded", "rf_oracle_used",
    }
)


def validate_checkpoint_identity(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable science/runtime identity of one checkpoint."""

    if type(raw) is not dict or set(raw) != _IDENTITY_FIELDS:
        raise TasteGCFFullResumeError("T12 checkpoint identity keys changed")
    purpose = raw.get("purpose")
    total_steps = _native_int(raw.get("total_steps"), field="T12 total_steps", minimum=2)
    cursor = _native_int(raw.get("checkpoint_cursor"), field="T12 checkpoint cursor", minimum=1)
    if purpose == "production":
        if total_steps != PRODUCTION_TOTAL_STEPS or cursor not in PRODUCTION_CHECKPOINT_CURSORS:
            raise TasteGCFFullResumeError("T12 production checkpoint is not 10k/20k")
    elif purpose == "gpu_replay_canary":
        if total_steps > 512 or cursor >= total_steps:
            raise TasteGCFFullResumeError("T12 replay canary is not bounded/intermediate")
    else:
        raise TasteGCFFullResumeError("T12 checkpoint purpose is invalid")
    if (
        raw.get("schema_version") != "tastemolnet_t12_checkpoint_identity_v1"
        or raw.get("stage") != STAGE
        or raw.get("graph_identity_contract") != GRAPH_IDENTITY_CONTRACT
        or raw.get("device") != "cuda:0"
        or type(raw.get("gpu_uuid")) is not str
        or _GPU_UUID.fullmatch(raw["gpu_uuid"]) is None
        or type(raw.get("seed")) is not int
        or raw["seed"] != 7
        or any(raw.get(field) is not value for field, value in {
            "train_loaded": True,
            "calibration_loaded": False,
            "test_loaded": False,
            "rf_oracle_used": False,
        }.items())
    ):
        raise TasteGCFFullResumeError("T12 checkpoint identity semantics changed")
    _require_uuid4(raw.get("attempt_id"), field="T12 attempt_id")
    _require_sha256(raw.get("generation_token"), field="T12 generation_token")
    for field in _IDENTITY_FIELDS:
        if field.endswith("_sha256"):
            _require_sha256(raw.get(field), field=f"T12 {field}")
    for field in ("execution_commit", "execution_tree"):
        if type(raw.get(field)) is not str or _GIT_SHA.fullmatch(raw[field]) is None:
            raise TasteGCFFullResumeError(f"T12 {field} must be a git SHA")
    for field in (
        "neurosed_distance_threshold_hex", "alpha_hex", "teleport_hex"
    ):
        try:
            value = float.fromhex(raw.get(field))
        except (TypeError, ValueError) as exc:
            raise TasteGCFFullResumeError(f"T12 {field} is invalid") from exc
        if not math.isfinite(value):
            raise TasteGCFFullResumeError(f"T12 {field} is non-finite")
    if float.fromhex(raw["neurosed_distance_threshold_hex"]) < 0.0:
        raise TasteGCFFullResumeError("T12 NeuroSED threshold is negative")
    if float.fromhex(raw["alpha_hex"]) != 1.0:
        raise TasteGCFFullResumeError("T12 alpha must preserve official endpoint 1.0")
    if not 0.0 <= float.fromhex(raw["teleport_hex"]) <= 1.0:
        raise TasteGCFFullResumeError("T12 teleport probability is invalid")
    _native_int(raw.get("sample_size"), field="T12 sample_size", minimum=1)
    _native_int(raw.get("candidate_capacity"), field="T12 candidate_capacity", minimum=20)
    return dict(raw)


def _adapter_state(adapter: Any) -> dict[str, Any]:
    scorer = adapter.scorer
    if scorer.cache_capacity != 0 or len(scorer._cache) != 0:
        raise TasteGCFFullResumeError("T12 requires the exact no-cache GINE scorer")
    return {
        "decode_failures": dict(adapter.decode_failures),
        "decode_success_count": adapter.decode_success_count,
        "empty_valid_batch_count": adapter.empty_valid_batch_count,
        "call_count": adapter.call_count,
        "scorer_calls": scorer.calls,
        "scorer_cache_hits": scorer.cache_hits,
        "scorer_cache_misses": scorer.cache_misses,
        "scorer_scored_rows": scorer.scored_rows,
        "scorer_last_trace": scorer.last_trace,
    }


def _restore_adapter_state(adapter: Any, state: Mapping[str, Any]) -> None:
    expected = {
        "decode_failures", "decode_success_count", "empty_valid_batch_count",
        "call_count", "scorer_calls", "scorer_cache_hits", "scorer_cache_misses",
        "scorer_scored_rows", "scorer_last_trace",
    }
    if type(state) is not dict or set(state) != expected or type(state["decode_failures"]) is not dict:
        raise TasteGCFFullResumeError("T12 adapter checkpoint schema changed")
    adapter.decode_failures = Counter()
    for key, value in state["decode_failures"].items():
        if type(key) is not str or not key:
            raise TasteGCFFullResumeError("T12 adapter failure key is invalid")
        adapter.decode_failures[key] = _native_int(value, field="T12 adapter failure count")
    for field in ("decode_success_count", "empty_valid_batch_count", "call_count"):
        setattr(adapter, field, _native_int(state[field], field=f"T12 adapter {field}"))
    scorer = adapter.scorer
    if scorer.cache_capacity != 0:
        raise TasteGCFFullResumeError("T12 restored scorer enabled a cache")
    scorer._cache.clear()
    for target, source in (
        ("calls", "scorer_calls"),
        ("cache_hits", "scorer_cache_hits"),
        ("cache_misses", "scorer_cache_misses"),
        ("scored_rows", "scorer_scored_rows"),
    ):
        setattr(scorer, target, _native_int(state[source], field=f"T12 {source}"))
    scorer.last_trace = state["scorer_last_trace"]


def _validate_stable_official_state(
    state: Mapping[str, Any], *, current_graph_identity: str
) -> None:
    if type(state) is not dict or set(state) != set(_VRRW_PROGRESS_STATE_FIELDS):
        raise TasteGCFFullResumeError("T12 official mutable-state keys changed")
    graph_map = state.get("graph_map")
    graph_index_map = state.get("graph_index_map")
    candidates = state.get("counterfactual_candidates")
    transitions = state.get("transitions")
    traversed = state.get("traversed_hashes")
    if (
        not isinstance(graph_map, Mapping)
        or not isinstance(graph_index_map, Mapping)
        or not isinstance(candidates, list)
        or not isinstance(transitions, Mapping)
        or not isinstance(traversed, list)
        or set(graph_map) != set(graph_index_map)
        or current_graph_identity not in graph_map
    ):
        raise TasteGCFFullResumeError("T12 official graph registry is incomplete")
    for key in graph_map:
        _stable_graph_hash(key, field="T12 official graph key")
    for key in traversed:
        _stable_graph_hash(key, field="T12 traversed graph identity")
    for index, row in enumerate(candidates):
        if type(row) is not dict:
            raise TasteGCFFullResumeError("T12 official candidate is malformed")
        key = _stable_graph_hash(row.get("graph_hash"), field="T12 candidate graph identity")
        if key not in graph_map or graph_index_map.get(key) != index:
            raise TasteGCFFullResumeError("T12 candidate/index registry differs")
    for source, transition in transitions.items():
        _stable_graph_hash(source, field="T12 transition source")
        if source not in graph_map or not isinstance(transition, (list, tuple)) or len(transition) != 4:
            raise TasteGCFFullResumeError("T12 transition payload is malformed")
        for target in transition[0]:
            _stable_graph_hash(target, field="T12 transition target")


def capture_checkpoint_payload(
    *,
    identity: Mapping[str, Any],
    vrrw: Any,
    bridge: T12StableGCFBridge,
    adapter: Any,
    action_counts: Mapping[str, int],
    current_graph_identity: str,
    np: Any,
    torch: Any,
) -> dict[str, Any]:
    """Capture every mutable scientific and RNG component after one step."""

    frozen_identity = validate_checkpoint_identity(identity)
    current = _stable_graph_hash(
        current_graph_identity, field="T12 checkpoint current graph"
    )
    if len(vrrw.traversed_hashes) != frozen_identity["checkpoint_cursor"]:
        raise TasteGCFFullResumeError("T12 checkpoint cursor differs from official trace")
    official = {field: getattr(vrrw, field) for field in _VRRW_PROGRESS_STATE_FIELDS}
    _validate_stable_official_state(official, current_graph_identity=current)
    bridge_state = bridge.checkpoint_state()
    adapter_state = _adapter_state(adapter)
    counts: dict[str, int] = {}
    for key, value in action_counts.items():
        if type(key) is not str or not key:
            raise TasteGCFFullResumeError("T12 action key is invalid")
        counts[key] = _native_int(value, field=f"T12 action {key}")
    rng = _capture_rng_state(np=np, torch=torch)
    state = {
        "official": official,
        "bridge": bridge_state,
        "adapter": adapter_state,
        "action_counts": counts,
        "current_graph_identity": current,
    }
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "identity": frozen_identity,
        "state": state,
        "rng": rng,
        "identity_sha256": _sha256_bytes(_canonical_bytes(frozen_identity)),
        "state_sha256": _semantic_sha256(state),
        "rng_sha256": _semantic_sha256(rng),
        "generated_to_original_neurosed_assertion": True,
        "test_loaded": False,
    }


def validate_checkpoint_payload(
    payload: Mapping[str, Any], *, expected_identity: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    expected = {
        "schema_version", "identity", "state", "rng", "identity_sha256",
        "state_sha256", "rng_sha256", "generated_to_original_neurosed_assertion",
        "test_loaded",
    }
    if type(payload) is not dict or set(payload) != expected:
        raise TasteGCFFullResumeError("T12 checkpoint payload keys changed")
    identity = validate_checkpoint_identity(payload.get("identity"))
    if expected_identity is not None and identity != validate_checkpoint_identity(expected_identity):
        raise TasteGCFFullResumeError("T12 checkpoint identity differs from the caller")
    state = payload.get("state")
    rng = payload.get("rng")
    if (
        payload.get("schema_version") != CHECKPOINT_SCHEMA
        or type(state) is not dict
        or set(state) != {"official", "bridge", "adapter", "action_counts", "current_graph_identity"}
        or type(rng) is not dict
        or payload.get("generated_to_original_neurosed_assertion") is not True
        or payload.get("test_loaded") is not False
        or payload.get("identity_sha256") != _sha256_bytes(_canonical_bytes(identity))
        or payload.get("state_sha256") != _semantic_sha256(state)
        or payload.get("rng_sha256") != _semantic_sha256(rng)
    ):
        raise TasteGCFFullResumeError("T12 checkpoint digest/semantics changed")
    _validate_stable_official_state(
        state["official"], current_graph_identity=state["current_graph_identity"]
    )
    if len(state["official"]["traversed_hashes"]) != identity["checkpoint_cursor"]:
        raise TasteGCFFullResumeError("T12 checkpoint trace length changed")
    return dict(payload)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_no_replace(temporary: Path, final: Path) -> None:
    try:
        os.link(temporary, final, follow_symlinks=False)
    except FileExistsError as exc:
        raise TasteGCFFullResumeError(f"T12 checkpoint already exists: {final}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    _fsync_directory(final.parent)


def _prepare_checkpoint_root(path: str | Path) -> Path:
    root = _normalized_absolute(path, field="T12 checkpoint root")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.resolve(strict=True) != root or not stat.S_ISDIR(root.stat().st_mode):
        raise TasteGCFFullResumeError("T12 checkpoint root is an alias or not a directory")
    return root


def write_checkpoint(
    root: str | Path, payload: Mapping[str, Any], *, torch: Any
) -> Path:
    """Durably publish one immutable payload plus its hash manifest."""

    validated = validate_checkpoint_payload(payload)
    checkpoint_root = _prepare_checkpoint_root(root)
    cursor = validated["identity"]["checkpoint_cursor"]
    payload_name = f"checkpoint-{cursor:08d}.pt"
    manifest_name = f"checkpoint-{cursor:08d}.manifest.json"
    payload_path = checkpoint_root / payload_name
    manifest_path = checkpoint_root / manifest_name
    token = str(uuid.uuid4())
    temporary_payload = checkpoint_root / f".{payload_name}.{token}.tmp"
    try:
        with temporary_payload.open("xb") as stream:
            torch.save(validated, stream)
            stream.flush()
            os.fsync(stream.fileno())
        payload_sha = _sha256_file(temporary_payload)
        payload_bytes = temporary_payload.stat().st_size
        if payload_bytes <= 0:
            raise TasteGCFFullResumeError("T12 checkpoint payload is empty")
        _publish_no_replace(temporary_payload, payload_path)
        manifest = {
            "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
            "status": "COMMITTED",
            "stage": STAGE,
            "payload_file": payload_name,
            "payload_sha256": payload_sha,
            "payload_bytes": payload_bytes,
            "checkpoint_cursor": cursor,
            "total_steps": validated["identity"]["total_steps"],
            "purpose": validated["identity"]["purpose"],
            "attempt_id": validated["identity"]["attempt_id"],
            "generation_token": validated["identity"]["generation_token"],
            "identity_sha256": validated["identity_sha256"],
            "state_sha256": validated["state_sha256"],
            "rng_sha256": validated["rng_sha256"],
            "written_at": _utc_now(),
            "immutable_no_replace": True,
        }
        manifest_data = _pretty_bytes(manifest)
        temporary_manifest = checkpoint_root / f".{manifest_name}.{token}.tmp"
        with temporary_manifest.open("xb") as stream:
            stream.write(manifest_data)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_no_replace(temporary_manifest, manifest_path)
        return manifest_path
    finally:
        if temporary_payload.exists():
            temporary_payload.unlink()


def reopen_checkpoint(
    manifest_path: str | Path,
    *,
    expected_identity: Mapping[str, Any],
    torch: Any,
) -> dict[str, Any]:
    """Reopen one committed checkpoint from a new process and rehash it."""

    path = _normalized_absolute(manifest_path, field="T12 checkpoint manifest")
    if path.resolve(strict=True) != path or path.is_symlink():
        raise TasteGCFFullResumeError("T12 checkpoint manifest is an alias")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteGCFFullResumeError("T12 checkpoint manifest is unreadable") from exc
    expected_manifest = {
        "schema_version", "status", "stage", "payload_file", "payload_sha256",
        "payload_bytes", "checkpoint_cursor", "total_steps", "purpose", "attempt_id",
        "generation_token", "identity_sha256", "state_sha256", "rng_sha256",
        "written_at", "immutable_no_replace",
    }
    if (
        type(manifest) is not dict
        or set(manifest) != expected_manifest
        or manifest.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA
        or manifest.get("status") != "COMMITTED"
        or manifest.get("stage") != STAGE
        or manifest.get("immutable_no_replace") is not True
        or type(manifest.get("payload_file")) is not str
        or _CHECKPOINT_FILE.fullmatch(manifest["payload_file"]) is None
    ):
        raise TasteGCFFullResumeError("T12 checkpoint manifest schema changed")
    identity = validate_checkpoint_identity(expected_identity)
    for field in ("checkpoint_cursor", "total_steps", "purpose", "attempt_id", "generation_token"):
        if manifest.get(field) != identity[field]:
            raise TasteGCFFullResumeError(f"T12 checkpoint manifest {field} changed")
    payload_path = path.parent / manifest["payload_file"]
    if payload_path.resolve(strict=True) != payload_path or payload_path.is_symlink():
        raise TasteGCFFullResumeError("T12 checkpoint payload is an alias")
    payload_stat = payload_path.stat()
    if (
        not stat.S_ISREG(payload_stat.st_mode)
        or payload_stat.st_nlink != 1
        or payload_stat.st_size != manifest.get("payload_bytes")
        or _sha256_file(payload_path)
        != _require_sha256(manifest.get("payload_sha256"), field="T12 payload SHA")
    ):
        raise TasteGCFFullResumeError("T12 checkpoint payload bytes changed")
    try:
        loaded = torch.load(payload_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older supported Torch
        loaded = torch.load(payload_path, map_location="cpu")
    validated = validate_checkpoint_payload(loaded, expected_identity=identity)
    for field in ("identity_sha256", "state_sha256", "rng_sha256"):
        if manifest.get(field) != validated[field]:
            raise TasteGCFFullResumeError(f"T12 checkpoint manifest {field} changed")
    return validated


def restore_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    vrrw: Any,
    bridge: T12StableGCFBridge,
    adapter: Any,
    action_counts: MutableMapping[str, int],
    np: Any,
    torch: Any,
) -> str:
    """Restore and then prove the complete live state equals the payload."""

    validated = validate_checkpoint_payload(payload, expected_identity=expected_identity)
    state = validated["state"]
    for field in _VRRW_PROGRESS_STATE_FIELDS:
        setattr(vrrw, field, state["official"][field])
    bridge.restore_checkpoint_state(state["bridge"])
    _restore_adapter_state(adapter, state["adapter"])
    action_counts.clear()
    for key, value in state["action_counts"].items():
        if type(key) is not str or not key:
            raise TasteGCFFullResumeError("T12 restored action key is invalid")
        action_counts[key] = _native_int(value, field=f"T12 restored action {key}")
    _restore_rng_state(validated["rng"], np=np, torch=torch)
    # Capture the scientific state again without consuming RNG.
    observed_state = {
        "official": {field: getattr(vrrw, field) for field in _VRRW_PROGRESS_STATE_FIELDS},
        "bridge": bridge.checkpoint_state(),
        "adapter": _adapter_state(adapter),
        "action_counts": dict(action_counts),
        "current_graph_identity": state["current_graph_identity"],
    }
    if _semantic_sha256(observed_state) != validated["state_sha256"]:
        raise TasteGCFFullResumeError("T12 live state differs after checkpoint restore")
    if _semantic_sha256(_capture_rng_state(np=np, torch=torch)) != validated["rng_sha256"]:
        raise TasteGCFFullResumeError("T12 RNG differs after checkpoint restore")
    return state["current_graph_identity"]


def build_replay_scientific_state(
    *,
    vrrw: Any,
    bridge: T12StableGCFBridge,
    adapter: Any,
    action_counts: Mapping[str, int],
    current_graph_identity: str,
    native_result: Mapping[str, Any],
    np: Any,
    torch: Any,
) -> dict[str, Any]:
    """Snapshot every field that must match after a process restart."""

    current = _stable_graph_hash(
        current_graph_identity, field="T12 replay current graph"
    )
    official = {field: getattr(vrrw, field) for field in _VRRW_PROGRESS_STATE_FIELDS}
    _validate_stable_official_state(official, current_graph_identity=current)
    traversed = list(official["traversed_hashes"])
    candidates: list[list[Any]] = []
    for row in official["counterfactual_candidates"]:
        key = _stable_graph_hash(
            row.get("graph_hash"), field="T12 replay candidate identity"
        )
        candidates.append(
            [
                key,
                _native_int(row.get("frequency"), field="T12 replay frequency", minimum=1),
                _semantic_sha256(row.get("importance_parts")),
                _semantic_sha256(row.get("input_graphs_covering_list")),
            ]
        )
    bridge_state = bridge.checkpoint_state()
    adapter_state = _adapter_state(adapter)
    counts = dict(action_counts)
    for key, value in counts.items():
        if type(key) is not str or not key:
            raise TasteGCFFullResumeError("T12 replay action key is invalid")
        _native_int(value, field=f"T12 replay action {key}")
    if type(native_result) is not dict or not native_result:
        raise TasteGCFFullResumeError("T12 official native result is absent")
    coverage = {
        key: list(record.coverage_vector)
        for key, record in sorted(bridge.records.items())
    }
    return {
        "completed_steps": len(traversed),
        "traversed_graph_identities": traversed,
        "candidate_frequency_order": candidates,
        "graph_map_sha256": _semantic_sha256(official["graph_map"]),
        "graph_index_map_sha256": _semantic_sha256(official["graph_index_map"]),
        "transitions_sha256": _semantic_sha256(official["transitions"]),
        "current_graph_identity": current,
        "bridge_state_sha256": _semantic_sha256(bridge_state),
        "adapter_state_sha256": _semantic_sha256(adapter_state),
        "action_counts_sha256": _semantic_sha256(counts),
        "rng_state_sha256": _semantic_sha256(
            _capture_rng_state(np=np, torch=torch)
        ),
        "generated_to_original_coverage_sha256": _semantic_sha256(coverage),
        "official_native_result_semantic_sha256": _semantic_sha256(native_result),
    }


def _validate_canary_process_identity(value: Any, *, field: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _CANARY_PROCESS_FIELDS:
        raise TasteGCFFullResumeError(f"{field} process identity keys changed")
    _native_int(value.get("pid"), field=f"{field} pid", minimum=1)
    _native_int(value.get("start_ticks"), field=f"{field} start_ticks", minimum=1)
    for name in ("command_sha256", "executable_sha256", "cwd_sha256"):
        _require_sha256(value.get(name), field=f"{field} {name}")
    return dict(value)


def _validate_canary_scientific_state(value: Any, *, field: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _CANARY_SCIENCE_FIELDS:
        raise TasteGCFFullResumeError(f"{field} scientific state keys changed")
    steps = _native_int(value.get("completed_steps"), field=f"{field} steps", minimum=1)
    traversed = value.get("traversed_graph_identities")
    candidates = value.get("candidate_frequency_order")
    if type(traversed) is not list or len(traversed) != steps:
        raise TasteGCFFullResumeError(f"{field} traversed identities are incomplete")
    for key in traversed:
        _stable_graph_hash(key, field=f"{field} traversed identity")
    if type(candidates) is not list or not candidates:
        raise TasteGCFFullResumeError(f"{field} candidate order is empty")
    for row in candidates:
        if (
            type(row) is not list
            or len(row) != 4
            or type(row[1]) is not int
            or row[1] < 1
        ):
            raise TasteGCFFullResumeError(f"{field} candidate order is malformed")
        _stable_graph_hash(row[0], field=f"{field} candidate identity")
        _require_sha256(row[2], field=f"{field} candidate importance SHA")
        _require_sha256(row[3], field=f"{field} candidate coverage SHA")
    _stable_graph_hash(
        value.get("current_graph_identity"), field=f"{field} current graph"
    )
    for name in _CANARY_SCIENCE_FIELDS:
        if name.endswith("_sha256"):
            _require_sha256(value.get(name), field=f"{field} {name}")
    return dict(value)


def capture_linux_process_identity() -> dict[str, Any]:
    """Capture the exact Linux process instance producing an observation."""

    proc = Path("/proc/self")
    try:
        stat_text = (proc / "stat").read_text(encoding="ascii")
        close = stat_text.rfind(")")
        fields = stat_text[close + 2 :].split()
        start_ticks = int(fields[19])
        command = (proc / "cmdline").read_bytes()
        executable = Path(os.readlink(proc / "exe")).resolve(strict=True)
        cwd = Path(os.readlink(proc / "cwd")).resolve(strict=True)
    except (OSError, UnicodeDecodeError, ValueError, IndexError) as exc:
        raise TasteGCFFullResumeError(
            "T12 canary requires readable Linux procfs process identity"
        ) from exc
    if not command or not executable.is_file():
        raise TasteGCFFullResumeError("T12 canary process evidence is incomplete")
    return _validate_canary_process_identity(
        {
            "pid": os.getpid(),
            "start_ticks": start_ticks,
            "command_sha256": _sha256_bytes(command),
            "executable_sha256": _sha256_file(executable),
            "cwd_sha256": _sha256_bytes(str(cwd).encode("utf-8")),
        },
        field="T12 live",
    )


def build_canary_observation(
    *,
    role: str,
    canary_identity_sha256: str,
    gpu_uuid: str,
    process_identity: Mapping[str, Any],
    scientific_state: Mapping[str, Any],
    native_result_sha256: str,
    checkpoint_reloaded: bool,
    generated_to_original_neurosed_assertion: bool,
) -> dict[str, Any]:
    """Build the small JSON compared by the independent canary gate."""

    if role not in {"uninterrupted", "cross_process_resumed"}:
        raise TasteGCFFullResumeError("T12 canary role is invalid")
    process = _validate_canary_process_identity(
        process_identity, field="T12 canary"
    )
    scientific = _validate_canary_scientific_state(
        scientific_state, field="T12 canary"
    )
    return {
        "schema_version": CANARY_OBSERVATION_SCHEMA,
        "stage": STAGE,
        "role": role,
        "canary_identity_sha256": _require_sha256(
            canary_identity_sha256, field="T12 canary identity"
        ),
        "gpu_uuid": gpu_uuid if _GPU_UUID.fullmatch(gpu_uuid or "") else _invalid_gpu_uuid(),
        "cuda_used": True,
        "process_identity": process,
        "checkpoint_reloaded": checkpoint_reloaded,
        "generated_to_original_neurosed_assertion": generated_to_original_neurosed_assertion,
        "graph_identity_contract": GRAPH_IDENTITY_CONTRACT,
        "python_builtin_hash_used": False,
        "embedding_identity_used": False,
        "native_result_sha256": _require_sha256(
            native_result_sha256, field="T12 native result SHA"
        ),
        "scientific_state": scientific,
        "scientific_state_sha256": _semantic_sha256(scientific),
    }


def _invalid_gpu_uuid() -> str:
    raise TasteGCFFullResumeError("T12 canary GPU UUID is invalid")


def compare_canary_observations(
    uninterrupted: Mapping[str, Any], resumed: Mapping[str, Any]
) -> dict[str, Any]:
    """Require byte/semantic exactness; allclose is intentionally forbidden."""

    expected = {
        "schema_version", "stage", "role", "canary_identity_sha256", "gpu_uuid",
        "cuda_used", "process_identity", "checkpoint_reloaded",
        "generated_to_original_neurosed_assertion", "graph_identity_contract",
        "python_builtin_hash_used", "embedding_identity_used", "native_result_sha256",
        "scientific_state", "scientific_state_sha256",
    }
    for label, value, role, reloaded in (
        ("uninterrupted", uninterrupted, "uninterrupted", False),
        ("resumed", resumed, "cross_process_resumed", True),
    ):
        if (
            type(value) is not dict
            or set(value) != expected
            or value.get("schema_version") != CANARY_OBSERVATION_SCHEMA
            or value.get("stage") != STAGE
            or value.get("role") != role
            or value.get("cuda_used") is not True
            or value.get("checkpoint_reloaded") is not reloaded
            or value.get("generated_to_original_neurosed_assertion") is not True
            or value.get("graph_identity_contract") != GRAPH_IDENTITY_CONTRACT
            or value.get("python_builtin_hash_used") is not False
            or value.get("embedding_identity_used") is not False
            or _validate_canary_process_identity(
                value.get("process_identity"), field=f"T12 {label}"
            )
            != value.get("process_identity")
            or _validate_canary_scientific_state(
                value.get("scientific_state"), field=f"T12 {label}"
            )
            != value.get("scientific_state")
            or value.get("scientific_state_sha256")
            != _semantic_sha256(value.get("scientific_state"))
        ):
            raise TasteGCFFullResumeError(f"T12 {label} canary observation is invalid")
        _require_sha256(value.get("canary_identity_sha256"), field=f"T12 {label} identity")
        _require_sha256(value.get("native_result_sha256"), field=f"T12 {label} result")
        if type(value.get("gpu_uuid")) is not str or _GPU_UUID.fullmatch(value["gpu_uuid"]) is None:
            raise TasteGCFFullResumeError(f"T12 {label} GPU UUID is invalid")
    if uninterrupted["process_identity"] == resumed["process_identity"]:
        raise TasteGCFFullResumeError("T12 canary did not cross a process boundary")
    for field in ("canary_identity_sha256", "gpu_uuid", "native_result_sha256", "scientific_state_sha256"):
        if uninterrupted[field] != resumed[field]:
            raise TasteGCFFullResumeError(f"T12 replay diverged at {field}")
    if uninterrupted["scientific_state"] != resumed["scientific_state"]:
        raise TasteGCFFullResumeError("T12 replay scientific state is not exact")
    return {
        "schema_version": CANARY_GATE_SCHEMA,
        "status": "PASS",
        "marker": CANARY_PASS_MARKER,
        "stage": STAGE,
        "canary_identity_sha256": uninterrupted["canary_identity_sha256"],
        "gpu_uuid": uninterrupted["gpu_uuid"],
        "uninterrupted_observation_sha256": _sha256_bytes(_canonical_bytes(uninterrupted)),
        "resumed_observation_sha256": _sha256_bytes(_canonical_bytes(resumed)),
        "scientific_state_sha256": uninterrupted["scientific_state_sha256"],
        "native_result_sha256": uninterrupted["native_result_sha256"],
        "cross_process": True,
        "cuda_used": True,
        "exact_equality": True,
        "approximate_comparison_used": False,
        "production_released": False,
        "verified_at": _utc_now(),
    }


def write_canary_observation(
    output_path: str | Path, observation: Mapping[str, Any]
) -> Path:
    """Publish one immutable observation after validating its exact schema."""

    expected_role = observation.get("role") if type(observation) is dict else None
    expected_reload = expected_role == "cross_process_resumed"
    if expected_role not in {"uninterrupted", "cross_process_resumed"}:
        raise TasteGCFFullResumeError("T12 canary observation role is invalid")
    # Rebuilding through the public constructor closes every field before I/O.
    validated = build_canary_observation(
        role=expected_role,
        canary_identity_sha256=observation.get("canary_identity_sha256"),
        gpu_uuid=observation.get("gpu_uuid"),
        process_identity=observation.get("process_identity"),
        scientific_state=observation.get("scientific_state"),
        native_result_sha256=observation.get("native_result_sha256"),
        checkpoint_reloaded=expected_reload,
        generated_to_original_neurosed_assertion=observation.get(
            "generated_to_original_neurosed_assertion"
        ),
    )
    if validated != dict(observation):
        raise TasteGCFFullResumeError("T12 canary observation contains drift")
    path = _normalized_absolute(output_path, field="T12 canary observation")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.resolve(strict=True) != path.parent:
        raise TasteGCFFullResumeError("T12 canary observation parent is an alias")
    temporary = path.parent / f".{path.name}.{uuid.uuid4()}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(_pretty_bytes(validated))
            stream.flush()
            os.fsync(stream.fileno())
        _publish_no_replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def write_canary_gate(
    output_path: str | Path,
    uninterrupted: Mapping[str, Any],
    resumed: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically create (never replace) the independent canary gate."""

    gate = compare_canary_observations(uninterrupted, resumed)
    path = _normalized_absolute(output_path, field="T12 canary gate")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.resolve(strict=True) != path.parent:
        raise TasteGCFFullResumeError("T12 canary gate parent is an alias")
    temporary = path.parent / f".{path.name}.{uuid.uuid4()}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(_pretty_bytes(gate))
            stream.flush()
            os.fsync(stream.fileno())
        _publish_no_replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return gate


__all__ = [
    "BRIDGE_SCHEMA",
    "CANARY_GATE_SCHEMA",
    "CANARY_OBSERVATION_SCHEMA",
    "CANARY_PASS_MARKER",
    "CHECKPOINT_MANIFEST_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "GRAPH_IDENTITY_CONTRACT",
    "PRODUCTION_CHECKPOINT_CURSORS",
    "PRODUCTION_TOTAL_STEPS",
    "STAGE",
    "T12StableGCFBridge",
    "T12StableGraphRecord",
    "TasteGCFFullResumeError",
    "build_canary_observation",
    "build_replay_scientific_state",
    "capture_checkpoint_payload",
    "capture_linux_process_identity",
    "compare_canary_observations",
    "reopen_checkpoint",
    "restore_checkpoint_payload",
    "validate_checkpoint_identity",
    "validate_checkpoint_payload",
    "write_canary_gate",
    "write_canary_observation",
    "write_checkpoint",
]
