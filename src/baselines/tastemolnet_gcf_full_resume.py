"""TasteMolNet T12 deterministic VRRW identity and restart substrate.

This module is deliberately dataset-specific.  It does not replace the
official GCFExplainer walk, edit operators, transition weights, NeuroSED
coverage, or three-class GINE scorer.  It changes only the process-local
``hash(embedding.tobytes())`` registry key at the project boundary: the T12
registry key is the SHA-256 of the canonical, parent-free chemistry identity
plus the exact normalized GINE model-input digest.  This distinction matters
because the native graph omits bond/charge/aromatic/stereo sidecars that the
production decoder restores before frozen-GINE scoring.

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
import copy
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Iterator, Mapping, MutableMapping, Sequence
import uuid

from src.baselines.tastemolnet_comrecgc_smoke import (
    _canonical_sha256,
    _identity_graph_sha256,
    _invalid_unscored_model_graph_payload,
    _normalize_model_graph_payload,
    canonical_attributed_graph,
)
from src.baselines.tastemolnet_gcf_smoke import (
    _VRRW_PROGRESS_STATE_FIELDS,
    _capture_rng_state,
    _embedding_sha256,
    _restore_rng_state,
    _semantic_sha256,
    score_and_candidate,
)
from src.baselines.tastemolnet_gcf_production_state import (
    PINNED_CANDIDATE_CAPACITY,
    PINNED_SAMPLE_SIZE,
    T12CompactHistoryJournal,
    T12ProductionBounds,
    TasteT12ProductionStateError,
)
from src.baselines.tastemolnet_gcf_candidate_store import (
    TasteT12CandidateStoreError,
    reopen_native_candidate_snapshot,
    write_native_candidate_snapshot,
)
from src.baselines.tastemolnet_gcf_transition_store import (
    TRANSITION_SNAPSHOT_SCHEMA,
    TRANSITION_STORE_POLICY,
    T12ExternalTransitionStore,
    TasteT12TransitionStoreError,
)


STAGE = "T12_GCF_FULL"
DATASET = "tastemolnet"
METHOD = "GCFExplainer"
SOURCE_LABEL = 1
NUM_CLASSES = 3
PRODUCTION_TOTAL_STEPS = 20_000
PRODUCTION_CHECKPOINT_CURSORS = frozenset({10_000, 20_000})
GRAPH_IDENTITY_CONTRACT = (
    "canonical_parent_free_gine_and_neurosed_graph_sha256_v2"
)
CHECKPOINT_SCHEMA = "tastemolnet_t12_vrrw_checkpoint_v1"
CHECKPOINT_MANIFEST_SCHEMA = "tastemolnet_t12_vrrw_checkpoint_manifest_v1"
BRIDGE_SCHEMA = "tastemolnet_t12_gcf_stable_bridge_v2"
PRODUCTION_BRIDGE_SCHEMA = "tastemolnet_t12_gcf_bounded_bridge_v3"
NEUROSED_QUERY_PERMUTATION_CONTRACT = (
    "canonical_graph_deterministic_encoding_coverage_v1"
)
CANARY_OBSERVATION_SCHEMA = "tastemolnet_t12_gpu_replay_observation_v1"
CANARY_GATE_SCHEMA = "tastemolnet_t12_gpu_replay_gate_v3"
CANARY_PREFIX_RECEIPT_SCHEMA = "tastemolnet_t12_gpu_replay_prefix_receipt_v1"
CANARY_PASS_MARKER = "[TASTE_T12_GPU_CROSS_PROCESS_REPLAY_CANARY_PASS]"
NATIVE_RESULT_SEMANTIC_CONTRACT = (
    "tastemolnet_t12_native_result_recursive_exact_v1"
)

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
        "official_state_sha256",
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


T12_MODEL_INPUT_IDENTITY_SCHEMA = (
    "tastemolnet_t12_gine_neurosed_model_input_identity_v2"
)


def _t12_model_input_identity(
    *,
    native_collision: Mapping[str, Any],
    supplied_identity: Mapping[str, Any] | None,
    supplied_model: Mapping[str, Any] | None,
    valid_fullgraph: bool,
) -> tuple[str, dict[str, Any]]:
    """Bind the official T12 key to the exact parent-free GINE input.

    Native GCF tensors omit bond/charge/aromatic/stereo sidecars that the
    production adapter restores from lineage before GINE scoring.  Therefore
    two native attributed graphs may compare equal while their actual model
    inputs differ.  The exact normalized model-input digest closes that alias;
    lineage or parent IDs themselves remain outside the identity.
    """

    if valid_fullgraph:
        if type(supplied_identity) is not dict or type(supplied_model) is not dict:
            raise TasteGCFFullResumeError(
                "T12 valid graph lacks canonical GINE model-input evidence"
            )
        identity_graph = dict(supplied_identity)
        model_graph = _normalize_model_graph_payload(supplied_model)
        frozen_gine_scored = True
    else:
        if supplied_identity is not None or supplied_model is not None:
            raise TasteGCFFullResumeError(
                "T12 invalid graph unexpectedly has GINE model-input evidence"
            )
        identity_graph = dict(native_collision)
        model_graph = _invalid_unscored_model_graph_payload()
        frozen_gine_scored = False
    # Validate the shared canonical-chemistry payload independently before it
    # becomes one component of the stronger T12 model-input identity.
    _identity_graph_sha256(identity_graph)
    native_graph = dict(native_collision)
    _identity_graph_sha256(native_graph)
    collision = {
        "schema_version": T12_MODEL_INPUT_IDENTITY_SCHEMA,
        "canonical_identity_graph": identity_graph,
        "canonical_neurosed_graph": native_graph,
        "model_graph_sha256": _canonical_sha256(model_graph),
        "frozen_gine_scored": frozen_gine_scored,
    }
    return _canonical_sha256(collision), collision


def _validate_t12_model_input_collision(
    value: Any, *, expected_sha256: str
) -> dict[str, Any]:
    if (
        type(value) is not dict
        or set(value)
        != {
            "schema_version",
            "canonical_identity_graph",
            "canonical_neurosed_graph",
            "model_graph_sha256",
            "frozen_gine_scored",
        }
        or value.get("schema_version") != T12_MODEL_INPUT_IDENTITY_SCHEMA
        or type(value.get("canonical_identity_graph")) is not dict
        or type(value.get("canonical_neurosed_graph")) is not dict
        or type(value.get("model_graph_sha256")) is not str
        or _SHA256.fullmatch(value["model_graph_sha256"]) is None
        or type(value.get("frozen_gine_scored")) is not bool
    ):
        raise TasteGCFFullResumeError(
            "T12 checkpoint model-input identity evidence is invalid"
        )
    _identity_graph_sha256(value["canonical_identity_graph"])
    _identity_graph_sha256(value["canonical_neurosed_graph"])
    if _canonical_sha256(value) != expected_sha256:
        raise TasteGCFFullResumeError("T12 bridge graph identity drifted")
    return dict(value)


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


def _deep_size_bytes(value: Any) -> int:
    """Conservatively count one bridge row without following runtime objects."""

    seen: set[int] = set()

    def visit(item: Any) -> int:
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        size = sys.getsizeof(item)
        if is_dataclass(item) and not isinstance(item, type):
            size += sum(visit(getattr(item, field.name)) for field in fields(item))
        elif isinstance(item, Mapping):
            size += sum(visit(key) + visit(child) for key, child in item.items())
        elif isinstance(item, (tuple, list, set, frozenset, deque, Counter)):
            size += sum(visit(child) for child in item)
        return size

    return visit(value)


@dataclass(frozen=True, slots=True)
class T12StableGraphRecord:
    graph_identity_sha256: str
    collision_payload: Mapping[str, Any]
    neurosed_query_sha256: str
    neurosed_query_sha256_variants: tuple[str, ...]
    neurosed_canonical_query_sha256: str
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
            "neurosed_query_sha256": self.neurosed_query_sha256,
            "neurosed_query_sha256_variants": list(
                self.neurosed_query_sha256_variants
            ),
            "neurosed_canonical_query_sha256": (
                self.neurosed_canonical_query_sha256
            ),
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


def _t12_neurosed_query_sha256(graph: Any) -> str:
    """Hash exactly the generated query bytes consumed by NeuroSED.

    The official auxiliary model reads only ``x`` and ``edge_index`` while
    its normalization also reads the node/edge counts.  Lineage and parent
    metadata are deliberately absent: they are not NeuroSED model input.
    Tensor dtype, shape, order, and bytes are bound by ``_semantic_sha256``.
    """

    x = getattr(graph, "x", None)
    edge_index = getattr(graph, "edge_index", None)
    num_nodes = getattr(graph, "num_nodes", None)
    num_edges = getattr(graph, "num_edges", None)
    if (
        x is None
        or edge_index is None
        or type(num_nodes) is not int
        or num_nodes < 0
        or type(num_edges) is not int
        or num_edges < 0
    ):
        raise TasteGCFFullResumeError(
            "T12 NeuroSED query lacks exact x/edge_index/count evidence"
        )
    return _semantic_sha256(
        {
            "schema_version": "tastemolnet_t12_neurosed_query_bytes_v1",
            "x": x,
            "edge_index": edge_index,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
        }
    )


def _t12_canonical_neurosed_query(
    graph: Any,
    *,
    canonical_graph: Mapping[str, Any],
    feature_atomic_numbers: Sequence[int],
) -> Any:
    """Rebuild NeuroSED ``x``/``edge_index`` in one canonical node order.

    Raw GCF tensors can encode the same attributed graph with different node
    and directed-edge orderings.  NeuroSED is mathematically permutation
    invariant, but GPU reduction order can still change a boundary bit.  A
    canonical SMILES parse supplies a deterministic node order, and sorted
    symmetric edges supply a deterministic edge order, including after a
    production live-record eviction and checkpoint restart.
    """

    if (
        type(canonical_graph) is not dict
        or set(canonical_graph) != {"canonical_graph", "num_nodes", "num_edges"}
    ):
        raise TasteGCFFullResumeError(
            "T12 canonical NeuroSED graph evidence is malformed"
        )
    canonical_smiles = canonical_graph.get("canonical_graph")
    expected_nodes = canonical_graph.get("num_nodes")
    expected_edges = canonical_graph.get("num_edges")
    if (
        type(canonical_smiles) is not str
        or not canonical_smiles
        or type(expected_nodes) is not int
        or expected_nodes < 0
        or type(expected_edges) is not int
        or expected_edges < 0
    ):
        raise TasteGCFFullResumeError(
            "T12 canonical NeuroSED graph evidence is malformed"
        )

    if canonical_smiles == "<EMPTY_ATTRIBUTED_GRAPH>":
        atomic_numbers: list[int] = []
        undirected_edges: list[tuple[int, int]] = []
    else:
        try:
            from rdkit import Chem
        except Exception as exc:  # pragma: no cover - AutoDL dependency
            raise TasteGCFFullResumeError(
                "RDKit is required for canonical T12 NeuroSED queries"
            ) from exc
        molecule = Chem.MolFromSmiles(canonical_smiles, sanitize=False)
        if molecule is None:
            raise TasteGCFFullResumeError(
                "T12 canonical NeuroSED graph cannot be parsed"
            )
        atomic_numbers = [int(atom.GetAtomicNum()) for atom in molecule.GetAtoms()]
        undirected_edges = sorted(
            (min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
             max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            for bond in molecule.GetBonds()
        )
    if (
        len(atomic_numbers) != expected_nodes
        or len(undirected_edges) != expected_edges
    ):
        raise TasteGCFFullResumeError(
            "T12 canonical NeuroSED graph counts changed on reconstruction"
        )

    vocabulary = tuple(feature_atomic_numbers)
    positions = {atomic_number: index for index, atomic_number in enumerate(vocabulary)}
    if len(positions) != len(vocabulary):
        raise TasteGCFFullResumeError(
            "T12 canonical NeuroSED vocabulary is not unique"
        )
    try:
        columns = [positions[atomic_number] for atomic_number in atomic_numbers]
    except KeyError as exc:
        raise TasteGCFFullResumeError(
            "T12 canonical NeuroSED atom is absent from the frozen vocabulary"
        ) from exc

    raw_x = getattr(graph, "x", None)
    raw_edge_index = getattr(graph, "edge_index", None)
    if raw_x is None or raw_edge_index is None:
        raise TasteGCFFullResumeError(
            "T12 NeuroSED query lacks x/edge_index tensors"
        )
    if hasattr(raw_x, "new_zeros"):
        canonical_x = raw_x.new_zeros((expected_nodes, len(vocabulary)))
        for row, column in enumerate(columns):
            canonical_x[row, column] = 1
    else:
        import numpy as np

        raw_x_array = np.asarray(raw_x)
        canonical_x = np.zeros(
            (expected_nodes, len(vocabulary)), dtype=raw_x_array.dtype
        )
        for row, column in enumerate(columns):
            canonical_x[row, column] = 1

    directed_edges = sorted(
        [
            endpoint
            for source, target in undirected_edges
            for endpoint in ((source, target), (target, source))
        ]
    )
    edge_rows = [
        [source for source, _target in directed_edges],
        [target for _source, target in directed_edges],
    ]
    if hasattr(raw_edge_index, "new_tensor"):
        canonical_edge_index = raw_edge_index.new_tensor(edge_rows)
    else:
        import numpy as np

        canonical_edge_index = np.asarray(
            edge_rows, dtype=np.asarray(raw_edge_index).dtype
        )

    query = copy.copy(graph)
    query.x = canonical_x
    query.edge_index = canonical_edge_index
    if (
        type(getattr(query, "num_nodes", None)) is not int
        or query.num_nodes != expected_nodes
        or type(getattr(query, "num_edges", None)) is not int
        or query.num_edges != 2 * expected_edges
    ):
        raise TasteGCFFullResumeError(
            "T12 canonical NeuroSED query counts differ from the source graph"
        )
    return query


def _t12_neurosed_context_sha256(
    *,
    neurosed_model: Any,
    original_graph_element_counts: Any,
    distance_threshold: float,
    parent_count: int,
) -> str:
    """Bind cached coverage to fixed target sizes and the exact threshold.

    The enclosing T12 checkpoint identity already binds the ordered source
    cohort and NeuroSED checkpoint.  Do not persist a digest of freshly
    recomputed CUDA target embeddings here: that would reintroduce the same
    low-bit cross-process instability this cache removes.  Their live tensor
    is still guarded against mutation/replacement within one process.
    """

    return _semantic_sha256(
        {
            "schema_version": "tastemolnet_t12_neurosed_context_v1",
            "target_embeddings_initialized": (
                getattr(neurosed_model, "target_emb", None) is not None
            ),
            "original_graph_element_counts": original_graph_element_counts,
            "distance_threshold_hex": distance_threshold.hex(),
            "parent_count": parent_count,
        }
    )


def _t12_runtime_value_guard(value: Any) -> Any:
    """Cheaply detect mutation/replacement of a fixed NeuroSED tensor."""

    if hasattr(value, "detach") and hasattr(value, "dtype"):
        data_ptr = getattr(value, "data_ptr", None)
        return (
            id(value),
            int(data_ptr()) if callable(data_ptr) else None,
            int(getattr(value, "_version", 0)),
            str(value.dtype),
            tuple(value.shape),
            str(getattr(value, "device", "")),
        )
    return _semantic_sha256(value)


def _t12_live_semantic_mismatch(
    previous: T12StableGraphRecord,
    observed: T12StableGraphRecord,
    *,
    observed_embedding: Any,
) -> tuple[list[str], dict[str, float]]:
    import numpy as np

    mismatch: list[str] = []
    previous_collision = dict(previous.collision_payload)
    observed_collision = dict(observed.collision_payload)
    if previous_collision.get("canonical_identity_graph") != (
        observed_collision.get("canonical_identity_graph")
    ):
        mismatch.append("canonical_identity_graph")
    if previous_collision.get("canonical_neurosed_graph") != (
        observed_collision.get("canonical_neurosed_graph")
    ):
        mismatch.append("canonical_neurosed_graph")
    if previous_collision.get("model_graph_sha256") != observed_collision.get(
        "model_graph_sha256"
    ):
        mismatch.append("model_graph_sha256")
    if previous_collision.get("frozen_gine_scored") is not observed_collision.get(
        "frozen_gine_scored"
    ):
        mismatch.append("frozen_gine_scored")
    if previous_collision != observed_collision and not mismatch:
        mismatch.append("collision_payload")
    for field in (
        "neurosed_canonical_query_sha256",
        "prediction",
        "candidate",
        "valid_fullgraph",
        "failure_reason",
        "coverage_vector",
        "canonical_embedding_dtype",
    ):
        previous_value = getattr(previous, field)
        observed_value = getattr(observed, field)
        if field in {"candidate", "valid_fullgraph"}:
            changed = previous_value is not observed_value
        else:
            changed = previous_value != observed_value
        if changed:
            mismatch.append(field)
    if len(previous.canonical_embedding_values) != len(
        observed.canonical_embedding_values
    ):
        mismatch.append("embedding_shape")

    previous_probabilities = np.asarray(previous.probabilities, dtype=np.float64)
    observed_probabilities = np.asarray(observed.probabilities, dtype=np.float64)
    probability_delta = float(
        np.max(np.abs(previous_probabilities - observed_probabilities))
    )
    if not np.allclose(
        previous_probabilities,
        observed_probabilities,
        rtol=GINE_CANONICAL_REUSE_RTOL,
        atol=GINE_CANONICAL_REUSE_ATOL,
    ):
        mismatch.append("probabilities")
    details = {"probability_max_abs_difference": probability_delta}
    if (
        "embedding_shape" not in mismatch
        and "canonical_embedding_dtype" not in mismatch
    ):
        previous_embedding = np.asarray(
            previous.canonical_embedding_values,
            dtype=np.dtype(previous.canonical_embedding_dtype),
        )
        embedding_delta = float(
            np.max(
                np.abs(
                    np.asarray(previous_embedding, dtype=np.float64)
                    - np.asarray(observed_embedding, dtype=np.float64)
                )
            )
        )
        details["embedding_max_abs_difference"] = embedding_delta
        if not np.allclose(
            previous_embedding,
            observed_embedding,
            rtol=GINE_CANONICAL_REUSE_RTOL,
            atol=GINE_CANONICAL_REUSE_ATOL,
        ):
            mismatch.append("embedding_values")
    return mismatch, details


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
        coverage_runtime: Any | None = None,
        production_history: T12CompactHistoryJournal | None = None,
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
        self.coverage_runtime = coverage_runtime
        self.production_history = production_history
        self.production_bounds = (
            production_history.bounds if production_history is not None else None
        )
        if (
            self.production_bounds is not None
            and self.production_bounds.parent_count != self.parent_count
        ):
            raise TasteGCFFullResumeError(
                "T12 production history parent count differs from the bridge"
            )
        if (
            not self.feature_atomic_numbers
            or len(set(self.feature_atomic_numbers)) != len(self.feature_atomic_numbers)
        ):
            raise TasteGCFFullResumeError(
                "T12 feature atomic-number vocabulary must be unique"
            )
        self.coverage_context_sha256 = _t12_neurosed_context_sha256(
            neurosed_model=self.neurosed_model,
            original_graph_element_counts=self.original_graph_element_counts,
            distance_threshold=self.distance_threshold,
            parent_count=self.parent_count,
        )
        self._coverage_threshold_hex = self.distance_threshold.hex()
        self._coverage_counts_guard = _t12_runtime_value_guard(
            self.original_graph_element_counts
        )
        self._coverage_targets_guard = _t12_runtime_value_guard(
            getattr(self.neurosed_model, "target_emb", None)
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
        self.neurosed_query_variant_observation_count = 0
        self.observed_peak_live_record_count = 0
        self.observed_peak_live_record_deep_bytes = 0
        self.observed_peak_live_record_serialized_bytes = 0
        self.evicted_complete_record_count = 0
        self._record_deep_bytes: dict[str, int] = {}
        self._record_serialized_bytes: dict[str, int] = {}
        self._live_record_deep_bytes_total = 0
        self._live_record_serialized_bytes_total = 0
        # Production scores one transient neighbor batch before the official
        # walk chooses at most one row for ``graph_map``.  Keep the identities
        # introduced since the preceding completed official transition so the
        # batch can be released immediately after that choice.  Durable
        # checkpoints are written only after this set has been emptied.
        self._transient_record_keys: set[str] = set()
        self._live_domain_synced = production_history is None

    def _record_sizes(self, record: T12StableGraphRecord) -> tuple[int, int]:
        deep = _deep_size_bytes(record)
        serialized = len(_canonical_bytes(record.to_dict()))
        if self.production_bounds is not None:
            if deep > self.production_bounds.max_live_record_deep_bytes:
                raise TasteGCFFullResumeError(
                    "one T12 live bridge row exceeds the production RAM bound"
                )
            if serialized > self.production_bounds.max_live_record_serialized_bytes:
                raise TasteGCFFullResumeError(
                    "one T12 live bridge row exceeds the checkpoint bound"
                )
        return deep, serialized

    def _remember_record(self, key: str, record: T12StableGraphRecord) -> None:
        if key in self.records:
            raise TasteGCFFullResumeError(
                "T12 bridge attempted to replace a live record as new"
            )
        deep, serialized = self._record_sizes(record)
        self.records[key] = record
        self._record_deep_bytes[key] = deep
        self._record_serialized_bytes[key] = serialized
        self._live_record_deep_bytes_total += deep
        self._live_record_serialized_bytes_total += serialized
        if self.production_bounds is not None:
            self._transient_record_keys.add(key)
            if len(self.records) > self.production_bounds.max_transient_full_records:
                raise TasteGCFFullResumeError(
                    "T12 live bridge rows exceeded the pinned transient bound"
                )
            deep_total = self._live_record_deep_bytes_total
            serialized_total = self._live_record_serialized_bytes_total
            self.observed_peak_live_record_count = max(
                self.observed_peak_live_record_count, len(self.records)
            )
            self.observed_peak_live_record_deep_bytes = max(
                self.observed_peak_live_record_deep_bytes, deep_total
            )
            self.observed_peak_live_record_serialized_bytes = max(
                self.observed_peak_live_record_serialized_bytes,
                serialized_total,
            )
            if deep_total > self.production_bounds.max_bridge_ram_bytes:
                raise TasteGCFFullResumeError(
                    "T12 live bridge rows exceeded the production RAM cap"
                )
            if serialized_total > self.production_bounds.max_bridge_checkpoint_bytes:
                raise TasteGCFFullResumeError(
                    "T12 live bridge rows exceeded the production checkpoint cap"
                )

    def _replace_record(self, key: str, record: T12StableGraphRecord) -> None:
        """Replace one live row while keeping the bounded counters exact."""

        previous = self.records.get(key)
        if previous is None:
            raise TasteGCFFullResumeError(
                "T12 bridge attempted to replace an unknown live record"
            )
        deep, serialized = self._record_sizes(record)
        old_deep = self._record_deep_bytes[key]
        old_serialized = self._record_serialized_bytes[key]
        projected_deep = self._live_record_deep_bytes_total - old_deep + deep
        projected_serialized = (
            self._live_record_serialized_bytes_total
            - old_serialized
            + serialized
        )
        if self.production_bounds is not None:
            if projected_deep > self.production_bounds.max_bridge_ram_bytes:
                raise TasteGCFFullResumeError(
                    "T12 live bridge rows exceeded the production RAM cap"
                )
            if (
                projected_serialized
                > self.production_bounds.max_bridge_checkpoint_bytes
            ):
                raise TasteGCFFullResumeError(
                    "T12 live bridge rows exceeded the production checkpoint cap"
                )
        self.records[key] = record
        self._record_deep_bytes[key] = deep
        self._record_serialized_bytes[key] = serialized
        self._live_record_deep_bytes_total = projected_deep
        self._live_record_serialized_bytes_total = projected_serialized
        self.observed_peak_live_record_deep_bytes = max(
            self.observed_peak_live_record_deep_bytes, projected_deep
        )
        self.observed_peak_live_record_serialized_bytes = max(
            self.observed_peak_live_record_serialized_bytes,
            projected_serialized,
        )

    def _assert_idle(self) -> None:
        if self._pending_hashes:
            raise TasteGCFFullResumeError(
                "official GCF did not consume every queued structural identity"
            )

    def _assert_coverage_context_unchanged(self) -> None:
        if (
            self.distance_threshold.hex() != self._coverage_threshold_hex
            or _t12_runtime_value_guard(self.original_graph_element_counts)
            != self._coverage_counts_guard
            or _t12_runtime_value_guard(
                getattr(self.neurosed_model, "target_emb", None)
            )
            != self._coverage_targets_guard
        ):
            raise TasteGCFFullResumeError(
                "T12 NeuroSED target cohort or threshold changed during replay"
            )

    def _coverage_for_exact_queries(
        self,
        *,
        graphs: Sequence[Any],
        graph_hashes: Sequence[str],
        collisions: Sequence[Mapping[str, Any]],
        canonical_query_hashes: Sequence[str],
        query_hashes: Sequence[str],
    ) -> tuple[
        Any,
        list[tuple[int, ...]],
        dict[str, tuple[str, ...]],
    ]:
        """Evaluate one raw encoding per canonical graph and audit variants.

        NeuroSED is permutation invariant to node/edge ordering.  The
        canonical attributed-graph identity plus its exact collision payload
        is therefore the scientific cache key.  Raw query hashes remain
        ordered audit evidence: the first observed encoding is evaluated and
        every later encoding variant reuses that binary coverage row.
        """

        self._assert_coverage_context_unchanged()
        if not (
            len(graphs)
            == len(graph_hashes)
            == len(collisions)
            == len(canonical_query_hashes)
            == len(query_hashes)
        ):
            raise TasteGCFFullResumeError(
                "T12 NeuroSED query identity evidence is unaligned"
            )
        rows: list[tuple[int, ...] | None] = [None] * len(graphs)
        variants_by_graph: dict[str, tuple[str, ...]] = {}
        pending: dict[str, dict[str, Any]] = {}
        for index, (
            graph_hash,
            collision,
            canonical_query_hash,
            query_hash,
        ) in enumerate(
            zip(
                graph_hashes,
                collisions,
                canonical_query_hashes,
                query_hashes,
                strict=True,
            )
        ):
            graph_hash = _stable_graph_hash(
                graph_hash, field="T12 NeuroSED graph identity"
            )
            query_hash = _require_sha256(
                query_hash, field="T12 NeuroSED raw query encoding"
            )
            canonical_query_hash = _require_sha256(
                canonical_query_hash,
                field="T12 canonical NeuroSED query encoding",
            )
            collision = _validate_t12_model_input_collision(
                collision, expected_sha256=graph_hash
            )
            previous = self.records.get(graph_hash)
            if previous is not None:
                if dict(previous.collision_payload) != collision:
                    diagnostic = {
                        "graph_identity_sha256": graph_hash,
                        "mismatch_fields": ["collision_payload"],
                    }
                    raise TasteGCFFullResumeError(
                        "one T12 canonical identity collided before NeuroSED reuse; "
                        f"diagnostic={json.dumps(diagnostic, sort_keys=True)}"
                    )
                if (
                    previous.neurosed_canonical_query_sha256
                    != canonical_query_hash
                ):
                    diagnostic = {
                        "graph_identity_sha256": graph_hash,
                        "mismatch_fields": [
                            "neurosed_canonical_query_sha256"
                        ],
                    }
                    raise TasteGCFFullResumeError(
                        "one T12 canonical NeuroSED encoding changed; "
                        f"diagnostic={json.dumps(diagnostic, sort_keys=True)}"
                    )
                variants = list(
                    variants_by_graph.get(
                        graph_hash, previous.neurosed_query_sha256_variants
                    )
                )
                if (
                    not variants
                    or variants[0] != previous.neurosed_query_sha256
                ):
                    raise TasteGCFFullResumeError(
                        "T12 NeuroSED representative query evidence changed"
                    )
                if (
                    self.production_history is None
                    and query_hash not in variants
                ):
                    variants.append(query_hash)
                variants_by_graph[graph_hash] = tuple(variants)
                if query_hash != variants[0]:
                    self.neurosed_query_variant_observation_count += 1
                rows[index] = previous.coverage_vector
                continue
            queued = pending.get(graph_hash)
            if queued is None:
                historical = (
                    self.production_history.lookup_first(graph_hash)
                    if self.production_history is not None
                    else None
                )
                initial_variants = (
                    [historical.neurosed_query_sha256]
                    if historical is not None
                    else [query_hash]
                )
                if (
                    self.production_history is None
                    and query_hash not in initial_variants
                ):
                    initial_variants.append(query_hash)
                if query_hash != initial_variants[0]:
                    self.neurosed_query_variant_observation_count += 1
                pending[graph_hash] = {
                    "graph": graphs[index],
                    "rows": [index],
                    "collision": collision,
                    "canonical_query_sha256": canonical_query_hash,
                    "variants": initial_variants,
                }
            else:
                if queued["collision"] != collision:
                    diagnostic = {
                        "graph_identity_sha256": graph_hash,
                        "mismatch_fields": ["collision_payload"],
                    }
                    raise TasteGCFFullResumeError(
                        "one T12 batch canonical identity collided before NeuroSED; "
                        f"diagnostic={json.dumps(diagnostic, sort_keys=True)}"
                    )
                if queued["canonical_query_sha256"] != canonical_query_hash:
                    diagnostic = {
                        "graph_identity_sha256": graph_hash,
                        "mismatch_fields": [
                            "neurosed_canonical_query_sha256"
                        ],
                    }
                    raise TasteGCFFullResumeError(
                        "one T12 batch canonical NeuroSED encoding changed; "
                        f"diagnostic={json.dumps(diagnostic, sort_keys=True)}"
                    )
                queued["rows"].append(index)
                if (
                    self.production_history is None
                    and query_hash not in queued["variants"]
                ):
                    queued["variants"].append(query_hash)
                if query_hash != queued["variants"][0]:
                    self.neurosed_query_variant_observation_count += 1

        if pending:
            ordered = list(pending.items())
            observed = self.importance.neurosed_threshold_coverage_estimation(
                self.neurosed_model,
                [entry["graph"] for _graph_hash, entry in ordered],
                self.original_graph_element_counts,
                self.distance_threshold,
            )
            self.distance_call_count += 1
            self.distance_evaluated_graph_count += len(ordered)
            if (
                tuple(observed.shape) != (len(ordered), self.parent_count)
                or not bool(self.vrrw.torch.isfinite(observed).all().item())
                or not bool(((observed == 0) | (observed == 1)).all().item())
            ):
                raise TasteGCFFullResumeError(
                    "T12 official generated-to-original NeuroSED coverage changed"
                )
            dense = (
                observed.to_dense()
                if getattr(observed, "is_sparse", False)
                else observed
            ).cpu()
            for observed_index, (graph_hash, entry) in enumerate(ordered):
                vector = tuple(int(value) for value in dense[observed_index].tolist())
                if (
                    len(vector) != self.parent_count
                    or any(value not in (0, 1) for value in vector)
                ):
                    raise TasteGCFFullResumeError(
                        "T12 NeuroSED coverage is not binary"
                    )
                for row_index in entry["rows"]:
                    rows[row_index] = vector
                variants_by_graph[graph_hash] = tuple(entry["variants"])

        if any(row is None for row in rows):
            raise TasteGCFFullResumeError("T12 coverage replay lost an input row")
        complete_rows = [tuple(row) for row in rows if row is not None]
        tensor_kwargs: dict[str, Any] = {}
        float32 = getattr(self.vrrw.torch, "float32", None)
        if float32 is not None:
            tensor_kwargs["dtype"] = float32
        coverage = self.vrrw.torch.tensor(complete_rows, **tensor_kwargs)
        return coverage, complete_rows, variants_by_graph

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
        valid = tuple(batch.valid_fullgraphs)
        failures = tuple(batch.failure_reasons)
        if len(valid) != len(values) or len(failures) != len(values):
            raise TasteGCFFullResumeError("T12 adapter evidence is unaligned")
        identity_payloads = tuple(
            getattr(batch, "identity_graph_payloads", ()) or ()
        )
        model_payloads = tuple(
            getattr(batch, "model_graph_payloads", ()) or ()
        )
        if (
            len(identity_payloads) != len(values)
            or len(model_payloads) != len(values)
        ):
            raise TasteGCFFullResumeError(
                "T12 adapter canonical model-input evidence is unaligned"
            )

        graph_hashes: list[str] = []
        collisions: list[dict[str, Any]] = []
        canonical_queries: list[Any] = []
        canonical_query_hashes: list[str] = []
        query_hashes: list[str] = []
        for index, graph in enumerate(values):
            native_identity = canonical_attributed_graph(
                graph, feature_atomic_numbers=self.feature_atomic_numbers
            )
            graph_hash, collision = _t12_model_input_identity(
                native_collision=native_identity.collision_payload(),
                supplied_identity=identity_payloads[index],
                supplied_model=model_payloads[index],
                valid_fullgraph=bool(valid[index]),
            )
            graph_hashes.append(
                _stable_graph_hash(graph_hash, field="T12 graph identity")
            )
            collisions.append(collision)
            canonical_query = _t12_canonical_neurosed_query(
                graph,
                canonical_graph=collision["canonical_neurosed_graph"],
                feature_atomic_numbers=self.feature_atomic_numbers,
            )
            canonical_queries.append(canonical_query)
            canonical_query_hashes.append(
                _t12_neurosed_query_sha256(canonical_query)
            )
            query_hashes.append(_t12_neurosed_query_sha256(graph))
        coverage, coverage_rows, query_variants = self._coverage_for_exact_queries(
            graphs=canonical_queries,
            graph_hashes=graph_hashes,
            collisions=collisions,
            canonical_query_hashes=canonical_query_hashes,
            query_hashes=query_hashes,
        )

        canonical_parts: list[tuple[float, float]] = []
        canonical_embeddings: list[Any] = []
        for index, graph in enumerate(values):
            graph_hash = graph_hashes[index]
            collision = collisions[index]
            canonical_query_hash = canonical_query_hashes[index]
            query_hash_variants = query_variants[graph_hash]
            representative_query_hash = query_hash_variants[0]
            row = tuple(float(value) for value in probabilities[index].tolist())
            score, prediction, candidate_condition = score_and_candidate(row)
            coverage_vector = coverage_rows[index]
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
                neurosed_query_sha256=representative_query_hash,
                neurosed_query_sha256_variants=query_hash_variants,
                neurosed_canonical_query_sha256=canonical_query_hash,
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
            historical = None
            if previous is None and self.production_history is not None:
                historical = self.production_history.lookup_first(graph_hash)
            if previous is None:
                if historical is None:
                    record = observed
                else:
                    first_score, first_prediction, first_condition = score_and_candidate(
                        historical.probabilities
                    )
                    coverage_sha = _sha256_bytes(bytes(coverage_vector))
                    failure_sha = _sha256_bytes(
                        str(failures[index]).encode("utf-8")
                    )
                    mismatch_fields: list[str] = []
                    if historical.prediction != observed.prediction:
                        mismatch_fields.append("prediction")
                    if historical.prediction != first_prediction:
                        mismatch_fields.append("historical_prediction_replay")
                    if historical.candidate is not observed.candidate:
                        mismatch_fields.append("candidate")
                    if (
                        historical.valid_fullgraph
                        is not observed.valid_fullgraph
                    ):
                        mismatch_fields.append("valid_fullgraph")
                    if historical.candidate is not bool(
                        historical.valid_fullgraph and first_condition
                    ):
                        mismatch_fields.append("historical_candidate_replay")
                    if historical.covered_parent_count != sum(coverage_vector):
                        mismatch_fields.append("covered_parent_count")
                    if historical.coverage_sha256 != coverage_sha:
                        mismatch_fields.append("coverage_vector")
                    if historical.failure_sha256 != failure_sha:
                        mismatch_fields.append("failure_reason")
                    # Once a complete row has left RAM, the compact journal
                    # retains only its embedding digest.  Re-entry is thus
                    # stricter than live-row reuse: low-bit drift blocks
                    # rather than silently choosing a new first row.
                    if (
                        historical.embedding_sha256
                        != observed.canonical_embedding_sha256
                    ):
                        mismatch_fields.append("embedding_sha256")
                    historical_probabilities = np.asarray(
                        historical.probabilities, dtype=np.float64
                    )
                    observed_probabilities = np.asarray(
                        observed.probabilities, dtype=np.float64
                    )
                    probability_delta = float(
                        np.max(
                            np.abs(
                                historical_probabilities
                                - observed_probabilities
                            )
                        )
                    )
                    if not np.allclose(
                        historical_probabilities,
                        observed_probabilities,
                        rtol=GINE_CANONICAL_REUSE_RTOL,
                        atol=GINE_CANONICAL_REUSE_ATOL,
                    ):
                        mismatch_fields.append("probabilities")
                    if mismatch_fields:
                        diagnostic = {
                            "graph_identity_sha256": graph_hash,
                            "mismatch_fields": mismatch_fields,
                            "probability_max_abs_difference": probability_delta,
                        }
                        raise TasteGCFFullResumeError(
                            "one evicted T12 identity changed compact GINE/NeuroSED semantics; "
                            f"diagnostic={json.dumps(diagnostic, sort_keys=True)}"
                        )
                    record = T12StableGraphRecord(
                        graph_identity_sha256=graph_hash,
                        collision_payload=collision,
                        neurosed_query_sha256=representative_query_hash,
                        neurosed_query_sha256_variants=query_hash_variants,
                        neurosed_canonical_query_sha256=(
                            canonical_query_hash
                        ),
                        probabilities=historical.probabilities,
                        prediction=historical.prediction,
                        score=first_score,
                        candidate=historical.candidate,
                        valid_fullgraph=historical.valid_fullgraph,
                        failure_reason=str(failures[index]),
                        coverage_vector=coverage_vector,
                        coverage_ratio=(
                            historical.covered_parent_count / float(self.parent_count)
                        ),
                        canonical_embedding_dtype=raw_embedding.dtype.str,
                        canonical_embedding_values=tuple(
                            float(value) for value in raw_embedding.tolist()
                        ),
                        canonical_embedding_sha256=(
                            observed.canonical_embedding_sha256
                        ),
                    )
                    self.canonical_row_reuse_count += 1
                self._remember_record(graph_hash, record)
            else:
                mismatch_fields, mismatch_details = _t12_live_semantic_mismatch(
                    previous,
                    observed,
                    observed_embedding=raw_embedding,
                )
                if mismatch_fields:
                    diagnostic = {
                        "graph_identity_sha256": graph_hash,
                        "mismatch_fields": mismatch_fields,
                        **mismatch_details,
                    }
                    raise TasteGCFFullResumeError(
                        "one T12 structural identity changed GINE/NeuroSED semantics; "
                        f"diagnostic={json.dumps(diagnostic, sort_keys=True)}"
                    )
                if (
                    previous.neurosed_query_sha256_variants
                    != query_hash_variants
                ):
                    record = replace(
                        previous,
                        neurosed_query_sha256_variants=query_hash_variants,
                    )
                    self._replace_record(graph_hash, record)
                else:
                    record = previous
                self.canonical_row_reuse_count += 1
            canonical_embedding = np.asarray(
                record.canonical_embedding_values,
                dtype=np.dtype(record.canonical_embedding_dtype),
            )
            if (
                _embedding_sha256(canonical_embedding)
                != record.canonical_embedding_sha256
            ):
                raise TasteGCFFullResumeError(
                    "cached T12 canonical GINE embedding changed"
                )
            canonical_embeddings.append(canonical_embedding)
            self._pending_hashes.append(
                (graph_hash, record.canonical_embedding_sha256)
            )
            lineage = _lineage_sha256(graph, graph_identity_sha256=graph_hash)
            if self.production_history is None:
                self.lineage_occurrences.setdefault(graph_hash, Counter())[lineage] += 1
            else:
                # Historical lineage identities belong to the compact journal
                # and its disk index.  RAM retains only the latest lineage for
                # each complete live row; no per-row Counter can grow with M.
                self.lineage_occurrences[graph_hash] = Counter({lineage: 1})
            if self.production_history is not None:
                try:
                    self.production_history.append_observation(
                        graph_identity_sha256=graph_hash,
                        probabilities=row,
                        prediction=prediction,
                        candidate=observed.candidate,
                        valid_fullgraph=observed.valid_fullgraph,
                        coverage_vector=coverage_vector,
                        embedding_sha256=observed.canonical_embedding_sha256,
                        failure_reason=observed.failure_reason,
                        lineage_sha256=lineage,
                        neurosed_query_sha256=query_hashes[index],
                    )
                except TasteT12ProductionStateError as exc:
                    raise TasteGCFFullResumeError(str(exc)) from exc
            canonical_parts.append((record.score, record.coverage_ratio))
        self.call_count += 1
        self.evaluated_graph_count += len(values)
        if self.production_history is not None:
            self._live_domain_synced = False
        return (
            np.asarray(canonical_parts, dtype=float),
            np.stack(canonical_embeddings),
            coverage,
        )

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
        if type(graph_hash) is not str:
            raise TasteGCFFullResumeError(
                "official GCF queried an unknown structural identity"
            )
        record = self.records.get(graph_hash)
        if record is not None:
            return record.candidate
        if self.production_history is not None:
            historical = self.production_history.lookup_first(graph_hash)
            if historical is not None:
                return historical.candidate
        raise TasteGCFFullResumeError(
            "official GCF queried an unknown structural identity"
        )

    def _retain_after_official_step(
        self, *, vrrw: Any, current_graph_identity: str
    ) -> None:
        """Release one scored neighbor batch after the official state commits.

        The official move/restart functions first score a transient batch,
        consume every stable hash, and only then update ``graph_map`` and the
        ordered candidate registry.  Production's resource proof permits one
        such batch in addition to the live official domain; waiting until the
        10k checkpoint to evict it would retain many batches and violate that
        proof.  This boundary performs no model call, RNG draw, reordering, or
        scientific-state mutation: it drops only complete bridge cache rows
        that the just-completed official transition did not retain.

        The expensive transition-journal rescan remains in
        :meth:`retain_official_live_domain` at the durable checkpoint.  Here
        the pinned official invariant that at most the returned current graph
        can newly enter ``graph_map`` is checked directly, keeping each 10k
        neighbor step linear in its own transient batch rather than in all
        historical transition files.
        """

        self._assert_idle()
        if self.production_history is None or self.production_bounds is None:
            raise TasteGCFFullResumeError(
                "T12 per-step live-domain pruning requires production history"
            )
        current = _stable_graph_hash(
            current_graph_identity, field="T12 production current graph"
        )
        graph_map = getattr(vrrw, "graph_map", None)
        graph_index_map = getattr(vrrw, "graph_index_map", None)
        candidates = getattr(vrrw, "counterfactual_candidates", None)
        transitions = getattr(vrrw, "transitions", None)
        live = set(graph_map) if isinstance(graph_map, Mapping) else set()
        if (
            not isinstance(graph_map, Mapping)
            or not isinstance(graph_index_map, Mapping)
            or not isinstance(candidates, list)
            or not isinstance(transitions, Mapping)
            or len(graph_map) != len(graph_index_map)
            or len(graph_map) != len(candidates)
            or live != set(graph_index_map)
            or current not in live
            or len(graph_map) > self.production_bounds.max_full_live_records
        ):
            raise TasteGCFFullResumeError(
                "T12 official step did not close one bounded live graph domain"
            )
        current_index = graph_index_map.get(current)
        if (
            type(current_index) is not int
            or not 0 <= current_index < len(candidates)
            or type(candidates[current_index]) is not dict
            or candidates[current_index].get("graph_hash") != current
        ):
            raise TasteGCFFullResumeError(
                "T12 official step current graph/candidate registry changed"
            )
        if current not in self.records:
            # A previously expanded transition can later select a target that
            # was never admitted to graph_map at its first observation.  Its
            # complete bridge cache row was correctly evicted, while the
            # authenticated compact first-observation row remains sufficient
            # for the official candidate predicate and transition payload.
            if self.production_history.lookup_first(current) is None:
                raise TasteGCFFullResumeError(
                    "T12 official step current graph is absent from compact history"
                )

        evicted = self._transient_record_keys - live
        if not set(self.records).difference(evicted).issubset(live):
            raise TasteGCFFullResumeError(
                "T12 official step and complete bridge live domains differ"
            )
        for key in evicted:
            if key not in self.records:
                raise TasteGCFFullResumeError(
                    "T12 transient bridge row disappeared before step pruning"
                )
            del self.records[key]
            self.lineage_occurrences.pop(key, None)
            self._live_record_deep_bytes_total -= self._record_deep_bytes.pop(key)
            self._live_record_serialized_bytes_total -= (
                self._record_serialized_bytes.pop(key)
            )
        self.evicted_complete_record_count += len(evicted)
        self._transient_record_keys.clear()
        self._live_domain_synced = True
        if (
            len(self.records) > self.production_bounds.max_full_live_records
            or self._live_record_deep_bytes_total
            > self.production_bounds.max_bridge_ram_bytes
            or self._live_record_serialized_bytes_total
            > self.production_bounds.max_bridge_checkpoint_bytes
        ):
            raise TasteGCFFullResumeError(
                "T12 per-step pruned bridge state exceeds a production bound"
            )

    def retain_official_live_domain(
        self, *, vrrw: Any, current_graph_identity: str
    ) -> dict[str, Any]:
        """Evict complete rows not needed by the official live graph registry.

        Transition-only identities remain queryable through the compact disk
        index.  Complete payload/embedding/coverage/lineage rows remain only
        for ``graph_map`` (which exactly equals the ordered candidate registry
        in the pinned official implementation) and the current graph.
        """

        self._assert_idle()
        if self.production_history is None or self.production_bounds is None:
            raise TasteGCFFullResumeError(
                "T12 live-domain pruning requires the production history"
            )
        current = _stable_graph_hash(
            current_graph_identity, field="T12 production current graph"
        )
        graph_map = getattr(vrrw, "graph_map", None)
        graph_index_map = getattr(vrrw, "graph_index_map", None)
        candidates = getattr(vrrw, "counterfactual_candidates", None)
        transitions = getattr(vrrw, "transitions", None)
        if (
            not isinstance(graph_map, Mapping)
            or not isinstance(graph_index_map, Mapping)
            or not isinstance(candidates, list)
            or not isinstance(transitions, Mapping)
            or set(graph_map) != set(graph_index_map)
            or current not in graph_map
            or len(graph_map) > self.production_bounds.max_full_live_records
        ):
            raise TasteGCFFullResumeError(
                "T12 official live graph domain exceeds the production bound"
            )
        live = set(graph_map)
        for index, row in enumerate(candidates):
            if type(row) is not dict:
                raise TasteGCFFullResumeError("T12 live candidate is malformed")
            key = _stable_graph_hash(
                row.get("graph_hash"), field="T12 live candidate identity"
            )
            if key not in live or graph_index_map.get(key) != index:
                raise TasteGCFFullResumeError(
                    "T12 candidate/index registry differs during pruning"
                )
        if getattr(transitions, "T12_BOUNDED_TRANSITION_STATE", False) is True:
            try:
                transition_audit = transitions.validate_live_domain(
                    live_sources=live,
                    known_target=lambda target: (
                        self.production_history.lookup_first(target) is not None
                    ),
                )
            except TasteT12TransitionStoreError as exc:
                raise TasteGCFFullResumeError(str(exc)) from exc
            transition_target_count = transition_audit[
                "transition_target_reference_count"
            ]
        else:
            transition_target_count = 0
            for source, transition in transitions.items():
                source = _stable_graph_hash(
                    source, field="T12 live transition source"
                )
                if (
                    source not in live
                    or not isinstance(transition, (list, tuple))
                    or len(transition) != 4
                ):
                    raise TasteGCFFullResumeError(
                        "T12 live transition payload is malformed"
                    )
                targets = transition[0]
                if (
                    not isinstance(targets, (list, tuple))
                    or len(targets) > self.production_bounds.sample_size + 1
                ):
                    raise TasteGCFFullResumeError(
                        "T12 transition target count exceeds the pinned sample size"
                    )
                transition_target_count += len(targets)
                for target in targets:
                    target = _stable_graph_hash(
                        target, field="T12 live transition target"
                    )
                    if self.production_history.lookup_first(target) is None:
                        raise TasteGCFFullResumeError(
                            "T12 transition target is absent from compact history"
                        )
        missing = live - set(self.records)
        if any(self.production_history.lookup_first(key) is None for key in missing):
            raise TasteGCFFullResumeError(
                "T12 official live graph registry is absent from compact history"
            )
        evicted = set(self.records) - live
        for key in evicted:
            del self.records[key]
            self.lineage_occurrences.pop(key, None)
            self._live_record_deep_bytes_total -= self._record_deep_bytes.pop(key)
            self._live_record_serialized_bytes_total -= (
                self._record_serialized_bytes.pop(key)
            )
        self.evicted_complete_record_count += len(evicted)
        self._transient_record_keys.clear()
        self._live_domain_synced = True
        deep_total = self._live_record_deep_bytes_total
        serialized_total = self._live_record_serialized_bytes_total
        if (
            len(self.records) > self.production_bounds.max_full_live_records
            or deep_total > self.production_bounds.max_bridge_ram_bytes
            or serialized_total
            > self.production_bounds.max_bridge_checkpoint_bytes
        ):
            raise TasteGCFFullResumeError(
                "T12 pruned bridge state exceeds a production bound"
            )
        return {
            "live_complete_record_count": len(self.records),
            "evicted_this_boundary": len(evicted),
            "evicted_complete_record_count": self.evicted_complete_record_count,
            "live_record_deep_bytes": deep_total,
            "live_record_serialized_bytes": serialized_total,
            "transition_entry_count": len(transitions),
            "transition_target_reference_count": transition_target_count,
            "compact_history_observation_count": (
                self.production_history.observation_count
            ),
        }

    def checkpoint_state(self) -> dict[str, Any]:
        self._assert_idle()
        coverage_runtime = (
            self.coverage_runtime.checkpoint_state()
            if self.coverage_runtime is not None
            else {
                "schema_version": "tastemolnet_t12_neurosed_retry_state_v1",
                "bounded_cuda_oom_retry": False,
                "calls": [],
            }
        )
        if type(coverage_runtime) is not dict:
            raise TasteGCFFullResumeError(
                "T12 NeuroSED retry state is not one mapping"
            )
        if self.production_history is not None and (
            not self._live_domain_synced or self._transient_record_keys
        ):
            raise TasteGCFFullResumeError(
                "T12 production checkpoint requested before live-domain pruning"
            )
        state = {
            "schema_version": (
                PRODUCTION_BRIDGE_SCHEMA
                if self.production_history is not None
                else BRIDGE_SCHEMA
            ),
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
            "neurosed_query_variant_observation_count": (
                self.neurosed_query_variant_observation_count
            ),
            "neurosed_query_full_variants_retained": (
                self.production_history is None
            ),
            "neurosed_query_variant_evidence_scope": (
                "bridge_records_all_observations"
                if self.production_history is None
                else "compact_history_authenticated_observations"
            ),
            "parent_count": self.parent_count,
            "distance_threshold_hex": self.distance_threshold.hex(),
            "coverage_context_sha256": self.coverage_context_sha256,
            "feature_atomic_numbers": list(self.feature_atomic_numbers),
            "graph_identity_contract": GRAPH_IDENTITY_CONTRACT,
            "neurosed_query_permutation_contract": (
                NEUROSED_QUERY_PERMUTATION_CONTRACT
            ),
            "generated_to_original_neurosed": True,
            "python_builtin_hash_used": False,
            "embedding_identity_used": False,
            "coverage_runtime": coverage_runtime,
        }
        if self.production_history is not None:
            state.update(
                {
                    "complete_records_are_live_domain_only": True,
                    "history": self.production_history.checkpoint_state(),
                    "production_bounds": self.production_bounds.to_dict(),
                    "production_bounds_sha256": self.production_bounds.sha256,
                    "observed_peak_live_record_count": (
                        self.observed_peak_live_record_count
                    ),
                    "observed_peak_live_record_deep_bytes": (
                        self.observed_peak_live_record_deep_bytes
                    ),
                    "observed_peak_live_record_serialized_bytes": (
                        self.observed_peak_live_record_serialized_bytes
                    ),
                    "evicted_complete_record_count": (
                        self.evicted_complete_record_count
                    ),
                }
            )
        return state

    def restore_checkpoint_state(self, payload: Mapping[str, Any]) -> None:
        import numpy as np

        self._assert_idle()
        production_payload = (
            type(payload) is dict
            and payload.get("schema_version") == PRODUCTION_BRIDGE_SCHEMA
        )
        expected = {
            "schema_version", "records", "lineage_occurrences", "call_count", "evaluated_graph_count",
            "calculate_hash_count", "distance_call_count",
            "distance_evaluated_graph_count", "canonical_row_reuse_count",
            "neurosed_query_variant_observation_count",
            "neurosed_query_full_variants_retained",
            "neurosed_query_variant_evidence_scope",
            "parent_count", "distance_threshold_hex", "feature_atomic_numbers",
            "coverage_context_sha256",
            "graph_identity_contract", "neurosed_query_permutation_contract",
            "generated_to_original_neurosed",
            "python_builtin_hash_used", "embedding_identity_used",
            "coverage_runtime",
        }
        if production_payload:
            expected.update(
                {
                    "complete_records_are_live_domain_only",
                    "history",
                    "production_bounds",
                    "production_bounds_sha256",
                    "observed_peak_live_record_count",
                    "observed_peak_live_record_deep_bytes",
                    "observed_peak_live_record_serialized_bytes",
                    "evicted_complete_record_count",
                }
            )
        if type(payload) is not dict or set(payload) != expected:
            raise TasteGCFFullResumeError("T12 bridge checkpoint keys changed")
        if (
            payload.get("schema_version")
            != (PRODUCTION_BRIDGE_SCHEMA if production_payload else BRIDGE_SCHEMA)
            or payload.get("parent_count") != self.parent_count
            or payload.get("distance_threshold_hex") != self.distance_threshold.hex()
            or payload.get("coverage_context_sha256")
            != self.coverage_context_sha256
            or payload.get("feature_atomic_numbers") != list(self.feature_atomic_numbers)
            or payload.get("graph_identity_contract") != GRAPH_IDENTITY_CONTRACT
            or payload.get("neurosed_query_permutation_contract")
            != NEUROSED_QUERY_PERMUTATION_CONTRACT
            or payload.get("neurosed_query_full_variants_retained")
            is not (not production_payload)
            or payload.get("neurosed_query_variant_evidence_scope")
            != (
                "compact_history_authenticated_observations"
                if production_payload
                else "bridge_records_all_observations"
            )
            or payload.get("generated_to_original_neurosed") is not True
            or payload.get("python_builtin_hash_used") is not False
            or payload.get("embedding_identity_used") is not False
            or type(payload.get("records")) is not dict
            or type(payload.get("lineage_occurrences")) is not dict
            or type(payload.get("coverage_runtime")) is not dict
        ):
            raise TasteGCFFullResumeError("T12 bridge checkpoint semantics changed")
        if production_payload:
            if (
                self.production_history is None
                or self.production_bounds is None
                or payload.get("complete_records_are_live_domain_only") is not True
            ):
                raise TasteGCFFullResumeError(
                    "T12 production checkpoint requires the compact history"
                )
            try:
                restored_bounds = T12ProductionBounds.from_dict(
                    payload.get("production_bounds")
                )
            except TasteT12ProductionStateError as exc:
                raise TasteGCFFullResumeError(str(exc)) from exc
            if (
                restored_bounds != self.production_bounds
                or payload.get("production_bounds_sha256")
                != self.production_bounds.sha256
            ):
                raise TasteGCFFullResumeError(
                    "T12 production bounds changed on restore"
                )
            # The journal constructor has already independently reopened and
            # rebuilt this exact committed prefix.  Equality here binds it to
            # the checkpoint without creating a second active history writer.
            if self.production_history.checkpoint_state() != payload.get("history"):
                raise TasteGCFFullResumeError(
                    "T12 compact history differs from the checkpoint"
                )
        elif self.production_history is not None:
            raise TasteGCFFullResumeError(
                "T12 cannot restore an all-history canary bridge as production"
            )
        records: dict[str, T12StableGraphRecord] = {}
        for key, raw in payload["records"].items():
            key = _stable_graph_hash(key, field="T12 checkpoint graph identity")
            if type(raw) is not dict or set(raw) != set(T12StableGraphRecord(
                graph_identity_sha256=key,
                collision_payload={},
                neurosed_query_sha256="0" * 64,
                neurosed_query_sha256_variants=("0" * 64,),
                neurosed_canonical_query_sha256="1" * 64,
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
            collision = _validate_t12_model_input_collision(
                collision, expected_sha256=key
            )
            probabilities = raw.get("probabilities")
            score, prediction, candidate_condition = score_and_candidate(probabilities)
            coverage_vector_raw = raw.get("coverage_vector")
            embedding_values = raw.get("canonical_embedding_values")
            query_variants = raw.get("neurosed_query_sha256_variants")
            if (
                raw.get("graph_identity_sha256") != key
                or _SHA256.fullmatch(str(raw.get("neurosed_query_sha256"))) is None
                or type(query_variants) is not list
                or not query_variants
                or len(query_variants) != len(set(query_variants))
                or any(
                    type(value) is not str or _SHA256.fullmatch(value) is None
                    for value in query_variants
                )
                or query_variants[0] != raw.get("neurosed_query_sha256")
                or (production_payload and len(query_variants) != 1)
                or _SHA256.fullmatch(
                    str(raw.get("neurosed_canonical_query_sha256"))
                )
                is None
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
                neurosed_query_sha256=str(raw["neurosed_query_sha256"]),
                neurosed_query_sha256_variants=tuple(query_variants),
                neurosed_canonical_query_sha256=str(
                    raw["neurosed_canonical_query_sha256"]
                ),
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
        self._record_deep_bytes = {}
        self._record_serialized_bytes = {}
        for key, record in records.items():
            deep, serialized = self._record_sizes(record)
            self._record_deep_bytes[key] = deep
            self._record_serialized_bytes[key] = serialized
        self._live_record_deep_bytes_total = sum(self._record_deep_bytes.values())
        self._live_record_serialized_bytes_total = sum(
            self._record_serialized_bytes.values()
        )
        self._transient_record_keys.clear()
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
        if self.coverage_runtime is None:
            if payload["coverage_runtime"] != {
                "schema_version": "tastemolnet_t12_neurosed_retry_state_v1",
                "bounded_cuda_oom_retry": False,
                "calls": [],
            }:
                raise TasteGCFFullResumeError(
                    "T12 checkpoint requires a NeuroSED retry runtime"
                )
        else:
            self.coverage_runtime.restore_checkpoint_state(
                payload["coverage_runtime"]
            )
        for field in (
            "call_count", "evaluated_graph_count", "calculate_hash_count",
            "distance_call_count", "distance_evaluated_graph_count",
            "canonical_row_reuse_count",
            "neurosed_query_variant_observation_count",
        ):
            setattr(self, field, _native_int(payload.get(field), field=f"T12 bridge {field}"))
        if production_payload:
            for field in (
                "observed_peak_live_record_count",
                "observed_peak_live_record_deep_bytes",
                "observed_peak_live_record_serialized_bytes",
                "evicted_complete_record_count",
            ):
                setattr(
                    self,
                    field,
                    _native_int(payload.get(field), field=f"T12 bridge {field}"),
                )
            if (
                self.observed_peak_live_record_count < len(self.records)
                or self.observed_peak_live_record_deep_bytes
                < self._live_record_deep_bytes_total
                or self.observed_peak_live_record_serialized_bytes
                < self._live_record_serialized_bytes_total
                or len(self.records) > self.production_bounds.max_full_live_records
                or self.production_history.observation_count
                != self.evaluated_graph_count
            ):
                raise TasteGCFFullResumeError(
                    "T12 production bridge resource counters changed"
                )
            self._live_domain_synced = True
        if (
            self.evaluated_graph_count != self.calculate_hash_count
            or self.distance_evaluated_graph_count > self.evaluated_graph_count
            or (
                self.evaluated_graph_count > 0
                and self.distance_evaluated_graph_count == 0
            )
            or self.distance_call_count > self.call_count
            or self.neurosed_query_variant_observation_count
            > self.evaluated_graph_count
            or self.neurosed_query_variant_observation_count
            < sum(
                len(record.neurosed_query_sha256_variants) - 1
                for record in self.records.values()
            )
            or (
                not production_payload
                and sum(
                    sum(values.values())
                    for values in self.lineage_occurrences.values()
                )
                != self.evaluated_graph_count
            )
        ):
            raise TasteGCFFullResumeError("T12 bridge checkpoint lost call closure")

    def report(self) -> dict[str, Any]:
        state = self.checkpoint_state()
        history = self.production_history
        return {
            **{key: value for key, value in state.items() if key != "records"},
            "unique_graph_count": (
                history.first_seen_graph_count if history is not None else len(self.records)
            ),
            "unique_lineage_count": (
                history.first_seen_lineage_count
                if history is not None
                else sum(len(values) for values in self.lineage_occurrences.values())
            ),
            "lineage_occurrence_count": (
                history.observation_count
                if history is not None
                else sum(
                    sum(values.values())
                    for values in self.lineage_occurrences.values()
                )
            ),
            "live_complete_record_count": len(self.records),
            "neurosed_query_variant_report_complete": history is None,
            "neurosed_query_variant_graph_count": sum(
                len(record.neurosed_query_sha256_variants) > 1
                for record in self.records.values()
            ),
            "neurosed_query_distinct_raw_sha256_count": sum(
                len(record.neurosed_query_sha256_variants)
                for record in self.records.values()
            ),
            "neurosed_query_variants_by_graph": {
                key: {
                    "representative_neurosed_query_sha256": (
                        record.neurosed_query_sha256
                    ),
                    "raw_query_sha256_variants": list(
                        record.neurosed_query_sha256_variants
                    ),
                    "canonical_neurosed_query_sha256": (
                        record.neurosed_canonical_query_sha256
                    ),
                }
                for key, record in sorted(self.records.items())
            },
            "registry_identity": GRAPH_IDENTITY_CONTRACT,
            "dedup_identity": GRAPH_IDENTITY_CONTRACT,
            "lineage_identity": GRAPH_IDENTITY_CONTRACT,
            "record_cache_identity": GRAPH_IDENTITY_CONTRACT,
            "strict_counterfactual_count": (
                history.first_seen_strict_counterfactual_count
                if history is not None
                else sum(row.candidate for row in self.records.values())
            ),
            "destination_prediction_counts": {
                str(label): (
                    history.destination_first_seen_counts[label]
                    if history is not None
                    else sum(
                        row.candidate and row.prediction == label
                        for row in self.records.values()
                    )
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
        production = self.production_history is not None
        if production:
            restart = getattr(self.vrrw, "restart_randomwalk", None)
            move = getattr(self.vrrw, "move_to_next_graph", None)
            if not callable(restart) or not callable(move):
                raise TasteGCFFullResumeError(
                    "T12 production VRRW lacks official restart/move boundaries"
                )
            originals["restart_randomwalk"] = restart
            originals["move_to_next_graph"] = move

            def bounded_restart(*args: Any, **kwargs: Any) -> Any:
                graph_hash = restart(*args, **kwargs)
                self._retain_after_official_step(
                    vrrw=self.vrrw, current_graph_identity=graph_hash
                )
                return graph_hash

            def bounded_move(*args: Any, **kwargs: Any) -> Any:
                result = move(*args, **kwargs)
                if (
                    not isinstance(result, tuple)
                    or len(result) != 2
                    or type(result[1]) is not bool
                ):
                    raise TasteGCFFullResumeError(
                        "T12 official move result changed at the pruning boundary"
                    )
                next_graph, teleported = result
                if not teleported:
                    self._retain_after_official_step(
                        vrrw=self.vrrw,
                        current_graph_identity=next_graph,
                    )
                return result

        self.importance.call = self.call
        self.vrrw.calculate_hash = self.calculate_hash
        self.vrrw.is_graph_counterfactual = self.is_graph_counterfactual
        if production:
            self.vrrw.restart_randomwalk = bounded_restart
            self.vrrw.move_to_next_graph = bounded_move
        try:
            yield
            self._assert_idle()
        finally:
            self.importance.call = originals["importance_call"]
            self.vrrw.calculate_hash = originals["calculate_hash"]
            self.vrrw.is_graph_counterfactual = originals["is_graph_counterfactual"]
            if production:
                self.vrrw.restart_randomwalk = originals["restart_randomwalk"]
                self.vrrw.move_to_next_graph = originals["move_to_next_graph"]


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
        if (
            raw.get("sample_size") != PINNED_SAMPLE_SIZE
            or raw.get("candidate_capacity") != PINNED_CANDIDATE_CAPACITY
        ):
            raise TasteGCFFullResumeError(
                "T12 production must retain official sample_size=10000 and k=100000"
            )
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
    transition_snapshot = (
        type(transitions) is dict
        and transitions.get("schema_version") == TRANSITION_SNAPSHOT_SCHEMA
    )
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
    if transition_snapshot:
        active_sources = transitions.get("active_sources")
        if (
            transitions.get("policy") != TRANSITION_STORE_POLICY
            or type(active_sources) is not list
            or transitions.get("active_entry_count") != len(active_sources)
            or len(set(active_sources)) != len(active_sources)
        ):
            raise TasteGCFFullResumeError(
                "T12 external transition checkpoint is malformed"
            )
        for source in active_sources:
            source = _stable_graph_hash(source, field="T12 transition source")
            if source not in graph_map:
                raise TasteGCFFullResumeError(
                    "T12 external transition source left the graph registry"
                )
    else:
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
    if frozen_identity["purpose"] == "production":
        if bridge.production_history is None or bridge.production_bounds is None:
            raise TasteGCFFullResumeError(
                "T12 production cannot use the canary all-history bridge"
            )
        if (
            bridge.production_bounds.total_steps != frozen_identity["total_steps"]
            or bridge.production_bounds.sample_size != frozen_identity["sample_size"]
            or bridge.production_bounds.candidate_capacity
            != frozen_identity["candidate_capacity"]
        ):
            raise TasteGCFFullResumeError(
                "T12 production resource bounds differ from checkpoint identity"
            )
        transitions = getattr(vrrw, "transitions", None)
        if not (
            getattr(transitions, "T12_BOUNDED_TRANSITION_STATE", False) is True
            and callable(getattr(transitions, "export_checkpoint_state", None))
        ):
            report = production_transition_bound_report(
                bounds=bridge.production_bounds
            )
            raise TasteGCFFullResumeError(
                "T12 raw official transition state is not production bounded: "
                f"bitpacked_coverage_bytes={report['minimum_bitpacked_coverage_bytes']}"
            )
        production_transition_bound_report(
            bounds=bridge.production_bounds, transition_store=transitions
        )
        transition_audit = transitions.audit()
        if (
            transition_audit.get("contract_sha256")
            != production_transition_contract_sha256(frozen_identity)
            or transition_audit.get("attempt_id") != frozen_identity["attempt_id"]
            or transition_audit.get("generation_token")
            != frozen_identity["generation_token"]
        ):
            raise TasteGCFFullResumeError(
                "T12 external transition store identity differs from the checkpoint"
            )
    current = _stable_graph_hash(
        current_graph_identity, field="T12 checkpoint current graph"
    )
    if len(vrrw.traversed_hashes) != frozen_identity["checkpoint_cursor"]:
        raise TasteGCFFullResumeError("T12 checkpoint cursor differs from official trace")
    official = {field: getattr(vrrw, field) for field in _VRRW_PROGRESS_STATE_FIELDS}
    if frozen_identity["purpose"] == "production":
        try:
            official["transitions"] = vrrw.transitions.export_checkpoint_state()
        except TasteT12TransitionStoreError as exc:
            raise TasteGCFFullResumeError(str(exc)) from exc
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
        if validated["identity"]["purpose"] == "production":
            bridge_state = validated["state"].get("bridge")
            bounds_raw = (
                bridge_state.get("production_bounds")
                if type(bridge_state) is dict
                else None
            )
            try:
                bounds = T12ProductionBounds.from_dict(bounds_raw)
            except TasteT12ProductionStateError as exc:
                raise TasteGCFFullResumeError(str(exc)) from exc
            if payload_bytes > bounds.max_full_checkpoint_bytes:
                raise TasteGCFFullResumeError(
                    "T12 production checkpoint exceeds the published size cap"
                )
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
    transition_state = validated["state"]["official"].get("transitions")
    if (
        validated["identity"]["purpose"] == "production"
        and type(transition_state) is dict
        and transition_state.get("schema_version") == TRANSITION_SNAPSHOT_SCHEMA
    ):
        try:
            T12ExternalTransitionStore.verify_checkpoint_state(transition_state)
        except TasteT12TransitionStoreError as exc:
            raise TasteGCFFullResumeError(str(exc)) from exc
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
    transition_state = state["official"].get("transitions")
    is_external_transition = (
        validated["identity"]["purpose"] == "production"
        and type(transition_state) is dict
        and transition_state.get("schema_version") == TRANSITION_SNAPSHOT_SCHEMA
    )
    for field in _VRRW_PROGRESS_STATE_FIELDS:
        if field != "transitions" or not is_external_transition:
            setattr(vrrw, field, state["official"][field])
    if is_external_transition:
        current_store = getattr(vrrw, "transitions", None)
        try:
            if getattr(current_store, "T12_BOUNDED_TRANSITION_STATE", False) is True:
                current_store.restore_checkpoint_state(transition_state)
            else:
                current_store = T12ExternalTransitionStore(
                    root=transition_state["root"],
                    parent_count=transition_state["parent_count"],
                    sample_size=transition_state["sample_size"],
                    candidate_capacity=transition_state["candidate_capacity"],
                    contract_sha256=transition_state["contract_sha256"],
                    attempt_id=transition_state["attempt_id"],
                    generation_token=transition_state["generation_token"],
                    expanded_capacity=transition_state["expanded_capacity"],
                    max_store_bytes=transition_state["max_store_bytes"],
                    resume_snapshot=transition_state,
                )
                setattr(vrrw, "transitions", current_store)
        except TasteT12TransitionStoreError as exc:
            raise TasteGCFFullResumeError(str(exc)) from exc
    bridge.restore_checkpoint_state(state["bridge"])
    _restore_adapter_state(adapter, state["adapter"])
    action_counts.clear()
    for key, value in state["action_counts"].items():
        if type(key) is not str or not key:
            raise TasteGCFFullResumeError("T12 restored action key is invalid")
        action_counts[key] = _native_int(value, field=f"T12 restored action {key}")
    _restore_rng_state(validated["rng"], np=np, torch=torch)
    # Capture the scientific state again without consuming RNG.
    observed_official = {
        field: getattr(vrrw, field) for field in _VRRW_PROGRESS_STATE_FIELDS
    }
    if is_external_transition:
        try:
            observed_official["transitions"] = (
                vrrw.transitions.export_checkpoint_state()
            )
        except TasteT12TransitionStoreError as exc:
            raise TasteGCFFullResumeError(str(exc)) from exc
    observed_state = {
        "official": observed_official,
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


def production_segment_bounds(resume_cursor: int = 0) -> tuple[int, int]:
    """Return the only legal production segment after a durable cursor."""

    cursor = _native_int(
        resume_cursor, field="T12 production resume cursor", minimum=0
    )
    if cursor == 0:
        return 1, 10_000
    if cursor == 10_000:
        return 10_001, 20_000
    if cursor == 20_000:
        raise TasteGCFFullResumeError("T12 production is already at its terminal cursor")
    raise TasteGCFFullResumeError("T12 production resume cursor must be 0/10k/20k")


def production_checkpoint_identity(
    identity_template: Mapping[str, Any], *, checkpoint_cursor: int
) -> dict[str, Any]:
    """Bind one immutable identity to exactly the 10k or 20k cursor."""

    cursor = _native_int(
        checkpoint_cursor, field="T12 production checkpoint cursor", minimum=1
    )
    value = dict(identity_template)
    if (
        value.get("sample_size") != PINNED_SAMPLE_SIZE
        or value.get("candidate_capacity") != PINNED_CANDIDATE_CAPACITY
    ):
        raise TasteGCFFullResumeError(
            "T12 production identity template changed official sample_size/k"
        )
    value["purpose"] = "production"
    value["total_steps"] = PRODUCTION_TOTAL_STEPS
    value["checkpoint_cursor"] = cursor
    return validate_checkpoint_identity(value)


def production_transition_contract_sha256(
    identity: Mapping[str, Any],
) -> str:
    """Bind one external transition journal across the 10k/20k cursors."""

    frozen = validate_checkpoint_identity(identity)
    if frozen["purpose"] != "production":
        raise TasteGCFFullResumeError(
            "T12 transition journal requires a production identity"
        )
    invariant = {
        key: value for key, value in frozen.items() if key != "checkpoint_cursor"
    }
    return _sha256_bytes(_canonical_bytes(invariant))


def production_transition_bound_report(
    *, bounds: T12ProductionBounds, transition_store: Any | None = None
) -> dict[str, Any]:
    """Expose why the raw official transition dictionary cannot run at 20k.

    This is an upper bound, not an assertion that every transition reaches the
    maximum.  It makes the remaining substrate requirement explicit rather
    than allowing a bridge-only memory proof to release production.
    """

    transition_entries = min(bounds.total_steps, bounds.max_full_live_records)
    target_rows = transition_entries * (bounds.sample_size + 1)
    coverage_bits = target_rows * bounds.parent_count
    installed = (
        getattr(transition_store, "T12_BOUNDED_TRANSITION_STATE", False) is True
        and callable(getattr(transition_store, "export_checkpoint_state", None))
        and callable(getattr(transition_store, "validate_live_domain", None))
    )
    store_audit = transition_store.audit() if installed else None
    if installed and (
        store_audit.get("policy") != TRANSITION_STORE_POLICY
        or store_audit.get("parent_count") != bounds.parent_count
        or store_audit.get("sample_size") != bounds.sample_size
        or store_audit.get("candidate_capacity") != bounds.candidate_capacity
        or store_audit.get("expanded_capacity", 3) > 2
        or store_audit.get("external_journal_is_authority") is not True
        or store_audit.get("coverage_payload_in_ram_is_lru_bounded") is not True
        or store_audit.get("model_recomputation_count") != 0
        or store_audit.get("rng_calls_added") != 0
        or store_audit.get("neighbor_order_changed") is not False
        or store_audit.get("candidate_order_changed") is not False
        or store_audit.get("scientific_parameters_changed") is not False
    ):
        raise TasteGCFFullResumeError(
            "T12 external transition store failed its scientific resource audit"
        )
    return {
        "schema_version": "tastemolnet_t12_transition_bound_audit_v1",
        "bounds_sha256": bounds.sha256,
        "M": bounds.total_steps,
        "sample_size": bounds.sample_size,
        "candidate_capacity": bounds.candidate_capacity,
        "parent_count": bounds.parent_count,
        "max_live_transition_entries": transition_entries,
        "max_transition_target_rows": target_rows,
        "dense_float32_coverage_bytes": coverage_bits * 4,
        "minimum_bitpacked_coverage_bytes": (coverage_bits + 7) // 8,
        "official_in_memory_transition_dict_allowed": False,
        "required_substrate": (
            "dataset_specific_external_compact_transition_store_with_"
            "bounded_expanded_lru"
        ),
        "external_transition_store_installed": installed,
        "external_transition_store_audit": store_audit,
        "production_launch_ready": installed,
    }


class T12ProductionCheckpointOrchestrator:
    """Narrow 0->10k->20k checkpoint coordinator for a future full runner."""

    def __init__(
        self,
        *,
        checkpoint_root: str | Path,
        identity_template: Mapping[str, Any],
        bounds: T12ProductionBounds,
    ) -> None:
        self.checkpoint_root = _normalized_absolute(
            checkpoint_root, field="T12 production checkpoint root"
        )
        self.identity_template = dict(identity_template)
        self.bounds = bounds
        # Validate both identities up front.  This prevents a worker from
        # changing any science/runtime pin between the two process segments.
        self._identities = {
            cursor: production_checkpoint_identity(
                self.identity_template, checkpoint_cursor=cursor
            )
            for cursor in sorted(PRODUCTION_CHECKPOINT_CURSORS)
        }
        reference = {
            key: value
            for key, value in self._identities[10_000].items()
            if key != "checkpoint_cursor"
        }
        if reference != {
            key: value
            for key, value in self._identities[20_000].items()
            if key != "checkpoint_cursor"
        }:
            raise TasteGCFFullResumeError(
                "T12 10k/20k checkpoint identities differ"
            )
        if (
            bounds.total_steps != PRODUCTION_TOTAL_STEPS
            or bounds.checkpoint_cursors != (10_000, 20_000)
            or bounds.sample_size != PINNED_SAMPLE_SIZE
            or bounds.candidate_capacity != PINNED_CANDIDATE_CAPACITY
        ):
            raise TasteGCFFullResumeError(
                "T12 checkpoint orchestration bounds changed"
            )

    def plan(self, *, resume_cursor: int = 0) -> dict[str, Any]:
        start, end = production_segment_bounds(resume_cursor)
        return {
            "schema_version": "tastemolnet_t12_10k_20k_plan_v1",
            "resume_cursor": resume_cursor,
            "segment_start": start,
            "segment_end": end,
            "checkpoint_cursor": end,
            "terminal_after_checkpoint": end == PRODUCTION_TOTAL_STEPS,
            "checkpoint_identity_sha256": _sha256_bytes(
                _canonical_bytes(self._identities[end])
            ),
            "calibration_loaded": False,
            "test_loaded": False,
        }

    def identity_at(self, checkpoint_cursor: int) -> dict[str, Any]:
        """Return one validated immutable production checkpoint identity."""

        cursor = _native_int(
            checkpoint_cursor, field="T12 production identity cursor", minimum=1
        )
        if cursor not in self._identities:
            raise TasteGCFFullResumeError(
                "T12 production identity cursor must be 10k or 20k"
            )
        return dict(self._identities[cursor])

    def commit(
        self,
        *,
        completed_steps: int,
        vrrw: Any,
        bridge: T12StableGCFBridge,
        adapter: Any,
        action_counts: Mapping[str, int],
        current_graph_identity: str,
        np: Any,
        torch: Any,
    ) -> Path:
        cursor = _native_int(
            completed_steps, field="T12 completed production steps", minimum=1
        )
        if cursor not in PRODUCTION_CHECKPOINT_CURSORS:
            raise TasteGCFFullResumeError(
                "T12 production may publish only the 10k/20k checkpoints"
            )
        transitions = getattr(vrrw, "transitions", None)
        if not (
            getattr(transitions, "T12_BOUNDED_TRANSITION_STATE", False) is True
            and callable(getattr(transitions, "export_checkpoint_state", None))
        ):
            report = production_transition_bound_report(bounds=self.bounds)
            raise TasteGCFFullResumeError(
                "T12 raw official transition state is not production bounded: "
                f"bitpacked_coverage_bytes={report['minimum_bitpacked_coverage_bytes']}"
            )
        report = production_transition_bound_report(
            bounds=self.bounds, transition_store=transitions
        )
        if report["production_launch_ready"] is not True:
            raise TasteGCFFullResumeError(
                "T12 external transition store is not production ready"
            )
        bridge.retain_official_live_domain(
            vrrw=vrrw, current_graph_identity=current_graph_identity
        )
        payload = capture_checkpoint_payload(
            identity=self._identities[cursor],
            vrrw=vrrw,
            bridge=bridge,
            adapter=adapter,
            action_counts=action_counts,
            current_graph_identity=current_graph_identity,
            np=np,
            torch=torch,
        )
        return write_checkpoint(self.checkpoint_root, payload, torch=torch)

    def reopen(
        self, manifest_path: str | Path, *, checkpoint_cursor: int, torch: Any
    ) -> dict[str, Any]:
        cursor = _native_int(
            checkpoint_cursor, field="T12 production reopen cursor", minimum=1
        )
        if cursor not in self._identities:
            raise TasteGCFFullResumeError(
                "T12 production reopen cursor must be 10k or 20k"
            )
        return reopen_checkpoint(
            manifest_path,
            expected_identity=self._identities[cursor],
            torch=torch,
        )

    def materialize_terminal_candidates(
        self,
        *,
        vrrw: Any,
        completed_steps: int,
        torch: Any,
    ) -> Path:
        """Persist the exact ordered train-side native pool after 20k."""

        cursor = _native_int(
            completed_steps, field="T12 candidate cursor", minimum=1
        )
        if cursor != PRODUCTION_TOTAL_STEPS:
            raise TasteGCFFullResumeError(
                "T12 native candidates may be materialized only after 20k"
            )
        identity = self._identities[PRODUCTION_TOTAL_STEPS]
        manifest = self.checkpoint_root / (
            f"checkpoint-{PRODUCTION_TOTAL_STEPS:08d}.manifest.json"
        )
        if not manifest.is_file():
            raise TasteGCFFullResumeError(
                "T12 terminal checkpoint must commit before candidate persistence"
            )
        # Reopen the checkpoint first so a candidate archive can never outrun
        # the exact generation closure it is intended to materialize.
        self.reopen(
            manifest,
            checkpoint_cursor=PRODUCTION_TOTAL_STEPS,
            torch=torch,
        )
        try:
            result = write_native_candidate_snapshot(
                self.checkpoint_root.parent / "native_candidates",
                vrrw=vrrw,
                checkpoint_cursor=cursor,
                contract_sha256=production_transition_contract_sha256(identity),
                attempt_id=identity["attempt_id"],
                generation_token=identity["generation_token"],
                torch=torch,
            )
            # A distinct read immediately proves both raw bytes and exact
            # recursive scientific content before returning to calibration.
            reopen_native_candidate_snapshot(
                result,
                expected_contract_sha256=production_transition_contract_sha256(
                    identity
                ),
                expected_attempt_id=identity["attempt_id"],
                expected_generation_token=identity["generation_token"],
                torch=torch,
            )
            return result
        except TasteT12CandidateStoreError as exc:
            raise TasteGCFFullResumeError(str(exc)) from exc


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
        "official_state_sha256": _semantic_sha256(official),
        "official_native_result_semantic_sha256": (
            canonical_native_result_sha256(native_result)
        ),
    }


def canonical_native_result_sha256(native_result: Mapping[str, Any]) -> str:
    """Hash the complete native result by scientific value, not archive bytes.

    PyTorch archive bytes also encode serialization representation details such
    as storage identities and object layout.  Those details can differ after a
    genuine process restart while the recursively loaded result is exact.  The
    semantic snapshot covers every mapping key and value, preserves list/tuple
    and tensor element order, and binds tensor dtype and shape.  Mapping
    insertion order is canonicalized by key because it is not part of the
    official result semantics.  Numeric values are exact; no ``allclose`` is
    used.
    """

    if type(native_result) is not dict or not native_result:
        raise TasteGCFFullResumeError("T12 official native result is absent")
    return _semantic_sha256(native_result)


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
    checkpoint_process_identity: Mapping[str, Any] | None = None,
    checkpoint_manifest_sha256: str | None = None,
    checkpoint_identity_sha256: str | None = None,
    checkpoint_state_sha256: str | None = None,
    checkpoint_rng_sha256: str | None = None,
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
    checkpoint_process: dict[str, Any] | None = None
    checkpoint_hashes = (
        checkpoint_manifest_sha256,
        checkpoint_identity_sha256,
        checkpoint_state_sha256,
        checkpoint_rng_sha256,
    )
    if role == "uninterrupted":
        if checkpoint_reloaded is not False or any(
            value is not None
            for value in (checkpoint_process_identity, *checkpoint_hashes)
        ):
            raise TasteGCFFullResumeError(
                "T12 uninterrupted observation contains checkpoint evidence"
            )
    else:
        if checkpoint_reloaded is not True or checkpoint_process_identity is None:
            raise TasteGCFFullResumeError(
                "T12 resumed observation lacks checkpoint-process evidence"
            )
        checkpoint_process = _validate_canary_process_identity(
            checkpoint_process_identity, field="T12 checkpoint producer"
        )
        for name, value in zip(
            (
                "manifest",
                "identity",
                "state",
                "RNG",
            ),
            checkpoint_hashes,
            strict=True,
        ):
            _require_sha256(value, field=f"T12 checkpoint {name} SHA")
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
        "checkpoint_process_identity": checkpoint_process,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "checkpoint_identity_sha256": checkpoint_identity_sha256,
        "checkpoint_state_sha256": checkpoint_state_sha256,
        "checkpoint_rng_sha256": checkpoint_rng_sha256,
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


def validate_canary_prefix_receipt(value: Any) -> dict[str, Any]:
    expected = {
        "schema_version",
        "status",
        "stage",
        "checkpoint_manifest",
        "checkpoint_manifest_sha256",
        "checkpoint_identity_sha256",
        "checkpoint_state_sha256",
        "checkpoint_rng_sha256",
        "checkpoint_cursor",
        "total_steps",
        "canary_identity_sha256",
        "gpu_uuid",
        "process_identity",
        "calibration_loaded",
        "test_loaded",
        "production_released",
    }
    if (
        type(value) is not dict
        or set(value) != expected
        or value.get("schema_version") != CANARY_PREFIX_RECEIPT_SCHEMA
        or value.get("status") != "CHECKPOINT_COMMITTED"
        or value.get("stage") != STAGE
        or value.get("calibration_loaded") is not False
        or value.get("test_loaded") is not False
        or value.get("production_released") is not False
        or type(value.get("checkpoint_cursor")) is not int
        or value["checkpoint_cursor"] < 1
        or type(value.get("total_steps")) is not int
        or value["total_steps"] <= value["checkpoint_cursor"]
        or type(value.get("gpu_uuid")) is not str
        or _GPU_UUID.fullmatch(value["gpu_uuid"]) is None
    ):
        raise TasteGCFFullResumeError("T12 checkpoint prefix receipt is invalid")
    _validate_canary_process_identity(
        value.get("process_identity"), field="T12 checkpoint prefix"
    )
    for field in (
        "checkpoint_manifest_sha256",
        "checkpoint_identity_sha256",
        "checkpoint_state_sha256",
        "checkpoint_rng_sha256",
        "canary_identity_sha256",
    ):
        _require_sha256(value.get(field), field=f"T12 prefix {field}")
    manifest_path = _normalized_absolute(
        value.get("checkpoint_manifest"), field="T12 prefix checkpoint manifest"
    )
    if manifest_path.resolve(strict=True) != manifest_path or manifest_path.is_symlink():
        raise TasteGCFFullResumeError("T12 prefix checkpoint manifest is an alias")
    if _sha256_file(manifest_path) != value["checkpoint_manifest_sha256"]:
        raise TasteGCFFullResumeError("T12 prefix checkpoint manifest bytes changed")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteGCFFullResumeError(
            "T12 prefix checkpoint manifest is unreadable"
        ) from exc
    if (
        type(manifest) is not dict
        or manifest.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA
        or manifest.get("status") != "COMMITTED"
        or manifest.get("stage") != STAGE
        or manifest.get("purpose") != "gpu_replay_canary"
        or manifest.get("checkpoint_cursor") != value["checkpoint_cursor"]
        or manifest.get("total_steps") != value["total_steps"]
        or manifest.get("identity_sha256") != value["checkpoint_identity_sha256"]
        or manifest.get("state_sha256") != value["checkpoint_state_sha256"]
        or manifest.get("rng_sha256") != value["checkpoint_rng_sha256"]
        or manifest.get("immutable_no_replace") is not True
    ):
        raise TasteGCFFullResumeError("T12 prefix checkpoint manifest changed")
    return dict(value)


def compare_canary_observations(
    uninterrupted: Mapping[str, Any],
    resumed: Mapping[str, Any],
    prefix_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Require byte/semantic exactness; allclose is intentionally forbidden."""

    expected = {
        "schema_version", "stage", "role", "canary_identity_sha256", "gpu_uuid",
        "cuda_used", "process_identity", "checkpoint_process_identity",
        "checkpoint_manifest_sha256", "checkpoint_identity_sha256",
        "checkpoint_state_sha256", "checkpoint_rng_sha256", "checkpoint_reloaded",
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
            or (
                role == "uninterrupted"
                and any(
                    value.get(field) is not None
                    for field in (
                        "checkpoint_process_identity",
                        "checkpoint_manifest_sha256",
                        "checkpoint_identity_sha256",
                        "checkpoint_state_sha256",
                        "checkpoint_rng_sha256",
                    )
                )
            )
            or (
                role == "cross_process_resumed"
                and (
                    _validate_canary_process_identity(
                        value.get("checkpoint_process_identity"),
                        field="T12 resumed checkpoint producer",
                    )
                    != value.get("checkpoint_process_identity")
                    or any(
                        _require_sha256(
                            value.get(field), field=f"T12 resumed {field}"
                        )
                        != value.get(field)
                        for field in (
                            "checkpoint_manifest_sha256",
                            "checkpoint_identity_sha256",
                            "checkpoint_state_sha256",
                            "checkpoint_rng_sha256",
                        )
                    )
                )
            )
        ):
            raise TasteGCFFullResumeError(f"T12 {label} canary observation is invalid")
        _require_sha256(value.get("canary_identity_sha256"), field=f"T12 {label} identity")
        _require_sha256(value.get("native_result_sha256"), field=f"T12 {label} result")
        if type(value.get("gpu_uuid")) is not str or _GPU_UUID.fullmatch(value["gpu_uuid"]) is None:
            raise TasteGCFFullResumeError(f"T12 {label} GPU UUID is invalid")
    prefix = validate_canary_prefix_receipt(prefix_receipt)
    uninterrupted_instance = (
        uninterrupted["process_identity"]["pid"],
        uninterrupted["process_identity"]["start_ticks"],
    )
    resumed_instance = (
        resumed["process_identity"]["pid"],
        resumed["process_identity"]["start_ticks"],
    )
    checkpoint_instance = (
        prefix["process_identity"]["pid"],
        prefix["process_identity"]["start_ticks"],
    )
    if len({uninterrupted_instance, resumed_instance, checkpoint_instance}) != 3:
        raise TasteGCFFullResumeError("T12 canary did not cross a process boundary")
    if resumed["checkpoint_process_identity"] != prefix["process_identity"]:
        raise TasteGCFFullResumeError("T12 resumed observation changed checkpoint process")
    if (
        prefix["canary_identity_sha256"]
        != uninterrupted["canary_identity_sha256"]
        or prefix["canary_identity_sha256"]
        != resumed["canary_identity_sha256"]
        or prefix["gpu_uuid"] != uninterrupted["gpu_uuid"]
        or prefix["gpu_uuid"] != resumed["gpu_uuid"]
    ):
        raise TasteGCFFullResumeError(
            "T12 checkpoint prefix belongs to another canary or GPU"
        )
    if (
        prefix["checkpoint_cursor"] != 8
        or prefix["total_steps"] != 16
        or uninterrupted["scientific_state"]["completed_steps"]
        != prefix["total_steps"]
        or resumed["scientific_state"]["completed_steps"]
        != prefix["total_steps"]
    ):
        raise TasteGCFFullResumeError("T12 bounded canary cursor/terminal changed")
    for observation_field, receipt_field in (
        ("checkpoint_manifest_sha256", "checkpoint_manifest_sha256"),
        ("checkpoint_identity_sha256", "checkpoint_identity_sha256"),
        ("checkpoint_state_sha256", "checkpoint_state_sha256"),
        ("checkpoint_rng_sha256", "checkpoint_rng_sha256"),
    ):
        if resumed[observation_field] != prefix[receipt_field]:
            raise TasteGCFFullResumeError(
                f"T12 resumed observation changed {observation_field}"
            )
    for field in ("canary_identity_sha256", "gpu_uuid"):
        if uninterrupted[field] != resumed[field]:
            raise TasteGCFFullResumeError(f"T12 replay diverged at {field}")
    uninterrupted_native_semantic = uninterrupted["scientific_state"][
        "official_native_result_semantic_sha256"
    ]
    resumed_native_semantic = resumed["scientific_state"][
        "official_native_result_semantic_sha256"
    ]
    if uninterrupted_native_semantic != resumed_native_semantic:
        raise TasteGCFFullResumeError(
            "T12 replay native-result scientific content diverged"
        )
    if (
        uninterrupted["scientific_state_sha256"]
        != resumed["scientific_state_sha256"]
    ):
        raise TasteGCFFullResumeError(
            "T12 replay diverged at scientific_state_sha256"
        )
    if uninterrupted["scientific_state"] != resumed["scientific_state"]:
        raise TasteGCFFullResumeError("T12 replay scientific state is not exact")
    native_raw_equal = (
        uninterrupted["native_result_sha256"]
        == resumed["native_result_sha256"]
    )
    return {
        "schema_version": CANARY_GATE_SCHEMA,
        "status": "PASS",
        "marker": CANARY_PASS_MARKER,
        "stage": STAGE,
        "graph_identity_contract": GRAPH_IDENTITY_CONTRACT,
        "neurosed_query_permutation_contract": (
            NEUROSED_QUERY_PERMUTATION_CONTRACT
        ),
        "canary_identity_sha256": uninterrupted["canary_identity_sha256"],
        "gpu_uuid": uninterrupted["gpu_uuid"],
        "uninterrupted_observation_sha256": _sha256_bytes(_canonical_bytes(uninterrupted)),
        "resumed_observation_sha256": _sha256_bytes(_canonical_bytes(resumed)),
        "scientific_state_sha256": uninterrupted["scientific_state_sha256"],
        "native_result_semantic_contract": NATIVE_RESULT_SEMANTIC_CONTRACT,
        "native_result_semantic_sha256": uninterrupted_native_semantic,
        "uninterrupted_native_result_raw_sha256": uninterrupted[
            "native_result_sha256"
        ],
        "resumed_native_result_raw_sha256": resumed["native_result_sha256"],
        "native_result_raw_bytes_equal": native_raw_equal,
        "native_result_difference_classification": (
            "RAW_BYTES_IDENTICAL"
            if native_raw_equal
            else "NON_SEMANTIC_SERIALIZATION_REPRESENTATION_ONLY"
        ),
        "checkpoint_manifest_sha256": prefix["checkpoint_manifest_sha256"],
        "checkpoint_identity_sha256": prefix["checkpoint_identity_sha256"],
        "checkpoint_state_sha256": prefix["checkpoint_state_sha256"],
        "checkpoint_rng_sha256": prefix["checkpoint_rng_sha256"],
        "cross_process": True,
        "cuda_used": True,
        "exact_equality": True,
        "exact_equality_scope": "canonical_scientific_state",
        "scientific_exact_equality": True,
        "native_result_approximate_comparison_used": False,
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
        checkpoint_process_identity=observation.get(
            "checkpoint_process_identity"
        ),
        checkpoint_manifest_sha256=observation.get("checkpoint_manifest_sha256"),
        checkpoint_identity_sha256=observation.get("checkpoint_identity_sha256"),
        checkpoint_state_sha256=observation.get("checkpoint_state_sha256"),
        checkpoint_rng_sha256=observation.get("checkpoint_rng_sha256"),
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
    prefix_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically create (never replace) the independent canary gate."""

    gate = compare_canary_observations(uninterrupted, resumed, prefix_receipt)
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
    "CANARY_PREFIX_RECEIPT_SCHEMA",
    "CHECKPOINT_MANIFEST_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "GRAPH_IDENTITY_CONTRACT",
    "NEUROSED_QUERY_PERMUTATION_CONTRACT",
    "NATIVE_RESULT_SEMANTIC_CONTRACT",
    "PINNED_CANDIDATE_CAPACITY",
    "PINNED_SAMPLE_SIZE",
    "PRODUCTION_BRIDGE_SCHEMA",
    "PRODUCTION_CHECKPOINT_CURSORS",
    "PRODUCTION_TOTAL_STEPS",
    "STAGE",
    "T12StableGCFBridge",
    "T12StableGraphRecord",
    "T12ProductionCheckpointOrchestrator",
    "TasteGCFFullResumeError",
    "build_canary_observation",
    "build_replay_scientific_state",
    "canonical_native_result_sha256",
    "capture_checkpoint_payload",
    "capture_linux_process_identity",
    "compare_canary_observations",
    "production_checkpoint_identity",
    "production_segment_bounds",
    "production_transition_contract_sha256",
    "production_transition_bound_report",
    "reopen_checkpoint",
    "restore_checkpoint_payload",
    "validate_canary_prefix_receipt",
    "validate_checkpoint_identity",
    "validate_checkpoint_payload",
    "write_canary_gate",
    "write_canary_observation",
    "write_checkpoint",
]
