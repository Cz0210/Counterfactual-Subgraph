"""TasteMolNet three-class adapter for the native COMRECGC walk.

The pinned upstream implementation identifies graphs with Python's process-
local ``hash(embedding.tobytes())`` and treats a scalar score of at least 0.5
as a binary counterfactual predicate.  Neither operation is valid for the
three-class TasteMolNet route.  This module supplies the narrow adaptation:

* graph identity is a SHA-256 of a canonical, attributed native graph;
* the walk weight is exactly ``1 - p(Sweet)``;
* counterfactual membership is separately ``argmax(probabilities) != 1``;
* GINE hidden vectors remain distance vectors and never become identities;
* lineage is audited separately and is never included in graph identity.

Importing the module performs no scientific I/O and does not import the
AutoDL-only PyTorch, PyG, or RDKit stack.
"""

from __future__ import annotations

from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import tempfile
from typing import Any, Iterator, Mapping, Sequence


STAGE = "T9_COMRECGC_SMOKE"
DATASET = "tastemolnet"
METHOD = "ComRecGC"
TASK_ID = "tastemolnet_t9_comrecgc_smoke"
NUM_CLASSES = 3
SOURCE_LABEL = 1
SMOKE_STEPS = 500
SMOKE_CHECKPOINT_STEP = 250
SMOKE_SOURCE_POOL = 64
SMOKE_SOURCE_COUNT = 8
SMOKE_HEADS = 5
SMOKE_CANDIDATE_CAPACITY = 2048
SMOKE_SAMPLE_SIZE = 10_000
SMOKE_TELEPORT = 0.1
SMOKE_THETA = 0.1
SMOKE_DELTA = 0.02
SMOKE_CLUSTER_SIZE = 3
SMOKE_RECOURSE_SIZE = 5
SMOKE_SEED = 7
GINE_CANONICAL_REUSE_RTOL = 1e-5
GINE_CANONICAL_REUSE_ATOL = 1e-7
OFFICIAL_COMRECGC_COMMIT = "122f9341a360e9f06bb58a2f5823bb596021f6bf"
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")
_GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")
PASS_FILE = "PASS"
PASS_MARKER = "[TASTE_T9_COMRECGC_SMOKE_PASS]"
PASS_BYTES = f"{PASS_MARKER}\n".encode("ascii")
OUTPUT_PAYLOAD_FILES = frozenset(
    {
        "input_hashes.json",
        "state.json",
        "manifest.json",
        "comrecgc_smoke.json",
        "gate.json",
    }
)
HELD_STAGE_EVIDENCE_KEYS = frozenset(
    {
        "stage",
        "gate_sha256",
        "root_inventory_sha256",
        "checkpoint_dir",
        "checkpoint_id",
        "checkpoint_inventory_sha256",
        "checkpoint_stat_inventory_sha256",
        "checkpoint_sha256s_sha256",
        "t2_adoption_gate_sha256",
        "t2_adoption_receipt_sha256",
        "t2_adoption_binding_sha256",
    }
)


class TasteComRecGCSmokeError(RuntimeError):
    """A three-class COMRECGC semantic or state boundary failed."""


@dataclass(frozen=True, slots=True)
class TasteComRecGCSmokeParameters:
    """Frozen bounded-smoke parameters; the full route uses M=50,000."""

    steps: int = SMOKE_STEPS
    checkpoint_step: int = SMOKE_CHECKPOINT_STEP
    source_pool: int = SMOKE_SOURCE_POOL
    source_count: int = SMOKE_SOURCE_COUNT
    heads: int = SMOKE_HEADS
    candidate_capacity: int = SMOKE_CANDIDATE_CAPACITY
    sample_size: int = SMOKE_SAMPLE_SIZE
    teleport_probability: float = SMOKE_TELEPORT
    theta: float = SMOKE_THETA
    delta: float = SMOKE_DELTA
    cluster_size: int = SMOKE_CLUSTER_SIZE
    recourse_size: int = SMOKE_RECOURSE_SIZE
    seed: int = SMOKE_SEED

    def validate(self) -> "TasteComRecGCSmokeParameters":
        integer_fields = {
            "steps": self.steps,
            "checkpoint_step": self.checkpoint_step,
            "source_pool": self.source_pool,
            "source_count": self.source_count,
            "heads": self.heads,
            "candidate_capacity": self.candidate_capacity,
            "sample_size": self.sample_size,
            "cluster_size": self.cluster_size,
            "recourse_size": self.recourse_size,
            "seed": self.seed,
        }
        for name, value in integer_fields.items():
            _native_int(value, field=name, minimum=1 if name != "seed" else 0)
        if not 0 < self.checkpoint_step < self.steps:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC checkpoint step is outside the walk"
            )
        if not self.heads <= self.source_count <= self.source_pool:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC source/head bounds are inconsistent"
            )
        for name, value in {
            "teleport_probability": self.teleport_probability,
            "theta": self.theta,
            "delta": self.delta,
        }.items():
            if isinstance(value, bool) or type(value) is not float or not math.isfinite(value):
                raise TasteComRecGCSmokeError(f"{name} must be one native finite float")
        expected = {
            "steps": SMOKE_STEPS,
            "checkpoint_step": SMOKE_CHECKPOINT_STEP,
            "source_pool": SMOKE_SOURCE_POOL,
            "source_count": SMOKE_SOURCE_COUNT,
            "heads": SMOKE_HEADS,
            "candidate_capacity": SMOKE_CANDIDATE_CAPACITY,
            "sample_size": SMOKE_SAMPLE_SIZE,
            "teleport_probability": SMOKE_TELEPORT,
            "theta": SMOKE_THETA,
            "delta": SMOKE_DELTA,
            "cluster_size": SMOKE_CLUSTER_SIZE,
            "recourse_size": SMOKE_RECOURSE_SIZE,
            "seed": SMOKE_SEED,
        }
        if asdict(self) != expected:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC bounded-smoke parameters changed"
            )
        if not 0.0 <= self.teleport_probability <= 1.0:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC teleport probability is outside [0,1]"
            )
        if self.theta <= 0.0 or self.delta <= 0.0:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC distance thresholds must be positive"
            )
        return self


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _checkpoint_value_payload(value: Any, *, field: str) -> Any:
    """Return a deterministic, typed digest payload for private walk state.

    The midpoint checkpoint contains PyTorch/PyG objects that cannot be put in
    terminal JSON.  We nevertheless bind every nested state component before
    serialization and after reload.  Tensor/array bytes are hashed; graph
    objects are reduced through their public ``to_dict`` payload.  Unknown
    object types fail closed instead of falling back to ``repr`` or a Python
    process-local hash.
    """

    if value is None:
        return {"type": "none"}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise TasteComRecGCSmokeError(
                f"{field} contains a non-finite float"
            )
        return {"type": "float", "value": value.hex()}
    if type(value) is str:
        return {"type": "str", "value": value}
    if type(value) is bytes:
        return {
            "type": "bytes",
            "size": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, Mapping):
        rows: list[dict[str, Any]] = []
        for key, nested in value.items():
            key_payload = _checkpoint_value_payload(
                key, field=f"{field}.mapping_key"
            )
            rows.append(
                {
                    "key": key_payload,
                    "value": _checkpoint_value_payload(
                        nested, field=f"{field}[{key!r}]"
                    ),
                }
            )
        rows.sort(key=lambda row: _canonical_bytes(row["key"]))
        return {"type": "mapping", "items": rows}
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [
                _checkpoint_value_payload(item, field=f"{field}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "items": [
                _checkpoint_value_payload(item, field=f"{field}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, (set, frozenset)):
        items = [
            _checkpoint_value_payload(item, field=f"{field}.set_item")
            for item in value
        ]
        items.sort(key=_canonical_bytes)
        return {
            "type": "frozenset" if isinstance(value, frozenset) else "set",
            "items": items,
        }
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "numpy") and hasattr(value, "shape"):
        try:
            array = value.numpy()
        except Exception:
            array = None
        if array is not None:
            return _checkpoint_array_payload(array, field=field)
    if hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tobytes"):
        return _checkpoint_array_payload(value, field=field)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return {
            "type": "object_to_dict",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "payload": _checkpoint_value_payload(
                to_dict(), field=f"{field}.to_dict"
            ),
        }
    raise TasteComRecGCSmokeError(
        f"{field} contains unsupported checkpoint state type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _checkpoint_array_payload(value: Any, *, field: str) -> dict[str, Any]:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - AutoDL dependency
        raise TasteComRecGCSmokeError(
            "NumPy is required for Taste COMRECGC checkpoint evidence"
        ) from exc
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise TasteComRecGCSmokeError(
            f"{field} contains an object-dtype checkpoint array"
        )
    if array.dtype.kind in {"f", "c"} and not np.isfinite(array).all():
        raise TasteComRecGCSmokeError(
            f"{field} contains a non-finite checkpoint array"
        )
    raw = array.tobytes(order="C")
    return {
        "type": "array",
        "dtype": str(array.dtype),
        "shape": [int(item) for item in array.shape],
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _checkpoint_state_sha256(value: Any, *, field: str) -> str:
    return _canonical_sha256(_checkpoint_value_payload(value, field=field))


def _lower_sha256(value: Any, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TasteComRecGCSmokeError(f"{field} must be one lowercase SHA-256")
    return value


def _tensor_list(value: Any, *, field: str) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise TasteComRecGCSmokeError(f"{field} must be one tensor-like list")
    return value


def _native_int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise TasteComRecGCSmokeError(
            f"{field} must be one bounded native integer"
        )
    return value


def _finite_probability(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TasteComRecGCSmokeError(f"{field} must be one finite probability")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise TasteComRecGCSmokeError(f"{field} must be one finite probability")
    return result


def score_and_candidate(probabilities: Sequence[Any]) -> tuple[float, int, bool]:
    """Return the native walk weight and the distinct multiclass predicate."""

    if not isinstance(probabilities, (list, tuple)) or len(probabilities) != 3:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC probabilities must contain exactly three classes"
        )
    values = [
        _finite_probability(value, field=f"probabilities[{index}]")
        for index, value in enumerate(probabilities)
    ]
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC probabilities do not sum to one"
        )
    prediction = max(range(NUM_CLASSES), key=lambda index: values[index])
    score = 1.0 - values[SOURCE_LABEL]
    return score, prediction, prediction != SOURCE_LABEL


@dataclass(frozen=True, slots=True)
class CanonicalNativeGraph:
    """Stable native graph identity with parent-free collision evidence."""

    graph_identity_sha256: str
    canonical_graph: str
    atomic_numbers: tuple[int, ...]
    undirected_edges: tuple[tuple[int, int], ...]

    @property
    def num_nodes(self) -> int:
        return len(self.atomic_numbers)

    @property
    def num_edges(self) -> int:
        return len(self.undirected_edges)

    def collision_payload(self) -> dict[str, Any]:
        return {
            "canonical_graph": self.canonical_graph,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
        }


def _identity_graph_sha256(payload: Mapping[str, Any]) -> str:
    """Hash canonical chemistry evidence without making it a model input."""

    if (
        type(payload) is not dict
        or set(payload) != {"canonical_graph", "num_nodes", "num_edges"}
        or type(payload.get("canonical_graph")) is not str
        or not payload["canonical_graph"]
        or type(payload.get("num_nodes")) is not int
        or payload["num_nodes"] < 0
        or type(payload.get("num_edges")) is not int
        or payload["num_edges"] < 0
    ):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC identity graph payload is malformed"
        )
    return _canonical_sha256(
        {
            "schema_version": "tastemolnet_comrecgc_attributed_graph_v1",
            **dict(payload),
            "node_attribute": "atomic_number_from_frozen_one_hot",
            "edge_attribute": "official_untyped_edge",
        }
    )


def canonical_attributed_graph(
    graph: Any,
    *,
    feature_atomic_numbers: Sequence[Any],
) -> CanonicalNativeGraph:
    """Canonicalize the exact upstream graph without parent metadata.

    COMRECGC's native graph has one-hot atom attributes and an untyped,
    symmetric directed edge list.  RDKit is used only as a canonical graph
    labeller: every atom has ``NoImplicit`` set and every native edge is a
    single bond, so the resulting string represents exactly that native edit
    space rather than source-molecule sidecars.
    """

    vocabulary = tuple(
        _native_int(value, field="feature_atomic_numbers", minimum=1)
        for value in feature_atomic_numbers
    )
    if not vocabulary or len(vocabulary) != len(set(vocabulary)):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC atom vocabulary must be non-empty and unique"
        )
    x = _tensor_list(getattr(graph, "x", None), field="graph.x")
    atomic_numbers: list[int] = []
    for row_index, raw_row in enumerate(x):
        if not isinstance(raw_row, list) or len(raw_row) != len(vocabulary):
            raise TasteComRecGCSmokeError(
                f"graph.x[{row_index}] differs from the frozen vocabulary"
            )
        positions: list[int] = []
        for column, raw in enumerate(raw_row):
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TasteComRecGCSmokeError(
                    f"graph.x[{row_index}][{column}] is not numeric"
                )
            value = float(raw)
            if not math.isfinite(value) or value not in (0.0, 1.0):
                raise TasteComRecGCSmokeError(
                    f"graph.x[{row_index}] is not exact one-hot"
                )
            if value == 1.0:
                positions.append(column)
        if len(positions) != 1:
            raise TasteComRecGCSmokeError(
                f"graph.x[{row_index}] is not exact one-hot"
            )
        atomic_numbers.append(vocabulary[positions[0]])
    declared_nodes = getattr(graph, "num_nodes", len(x))
    if hasattr(declared_nodes, "item"):
        declared_nodes = declared_nodes.item()
    if (
        isinstance(declared_nodes, bool)
        or not isinstance(declared_nodes, int)
        or declared_nodes != len(x)
    ):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC graph num_nodes differs from graph.x"
        )
    edge_index = _tensor_list(
        getattr(graph, "edge_index", None), field="graph.edge_index"
    )
    if len(edge_index) != 2 or any(not isinstance(row, list) for row in edge_index):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC edge_index must have shape [2, E]"
        )
    if len(edge_index[0]) != len(edge_index[1]):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC edge_index rows have different lengths"
        )
    directed: Counter[tuple[int, int]] = Counter()
    for raw_source, raw_target in zip(edge_index[0], edge_index[1], strict=True):
        source = _native_int(raw_source, field="edge source", minimum=0)
        target = _native_int(raw_target, field="edge target", minimum=0)
        if source >= len(x) or target >= len(x) or source == target:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC edge_index contains an invalid endpoint"
            )
        directed[(source, target)] += 1
    pairs = sorted({(min(a, b), max(a, b)) for a, b in directed})
    for a, b in pairs:
        if directed[(a, b)] != 1 or directed[(b, a)] != 1:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC native edges must be exactly symmetric"
            )
    if sum(directed.values()) != 2 * len(pairs):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC native graph contains duplicate edges"
        )
    edge_attr = getattr(graph, "edge_attr", None)
    if edge_attr is not None:
        values = _tensor_list(edge_attr, field="graph.edge_attr")
        if values:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC official native graph unexpectedly has edge_attr"
            )

    if atomic_numbers:
        try:
            from rdkit import Chem
        except Exception as exc:  # pragma: no cover - AutoDL dependency
            raise TasteComRecGCSmokeError(
                "RDKit is required for canonical native graph identity"
            ) from exc
        editable = Chem.RWMol()
        for atomic_number in atomic_numbers:
            atom = Chem.Atom(int(atomic_number))
            atom.SetNoImplicit(True)
            editable.AddAtom(atom)
        for source, target in pairs:
            editable.AddBond(int(source), int(target), Chem.BondType.SINGLE)
        molecule = editable.GetMol()
        molecule.UpdatePropertyCache(strict=False)
        try:
            canonical = Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=False,
                allHsExplicit=True,
            )
        except Exception as exc:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC native graph cannot be canonically serialized"
            ) from exc
        if not canonical:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC native graph canonical serialization is empty"
            )
    else:
        # The pinned native removal operator can delete the final isolated
        # node.  That state is scientifically invalid for GINE scoring, but it
        # is still a real walk state and therefore needs one stable structural
        # identity.  It must never collapse onto the invalid scorer's shared
        # zero embedding or a parent-derived sidecar.
        canonical = "<EMPTY_ATTRIBUTED_GRAPH>"
    collision_payload = {
        "canonical_graph": canonical,
        "num_nodes": len(atomic_numbers),
        "num_edges": len(pairs),
    }
    return CanonicalNativeGraph(
        graph_identity_sha256=_identity_graph_sha256(collision_payload),
        canonical_graph=canonical,
        atomic_numbers=tuple(atomic_numbers),
        undirected_edges=tuple(pairs),
    )


def _model_tensor_payload(value: Any, *, field: str) -> dict[str, Any]:
    """Losslessly serialize one tensor that can influence frozen-GINE input."""

    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - AutoDL dependency
        raise TasteComRecGCSmokeError(
            "NumPy is required for Taste COMRECGC model-graph evidence"
        ) from exc
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    try:
        array = np.ascontiguousarray(value)
    except Exception as exc:
        raise TasteComRecGCSmokeError(f"{field} is not tensor-like") from exc
    if array.dtype.hasobject or (
        array.dtype.kind in {"f", "c"} and not np.isfinite(array).all()
    ):
        raise TasteComRecGCSmokeError(f"{field} cannot be serialized losslessly")
    return {
        "dtype": array.dtype.str,
        "shape": [int(item) for item in array.shape],
        "values": array.tolist(),
    }


def _native_model_graph_payload(graph: Any) -> dict[str, Any]:
    """Fallback model graph for adapters that score the native tensors itself."""

    edge_attr = getattr(graph, "edge_attr", None)
    return {
        "schema_version": "tastemolnet_native_model_graph_v1",
        "num_nodes": int(getattr(graph, "num_nodes")),
        "node_features": _model_tensor_payload(
            getattr(graph, "x", None), field="model_graph.x"
        ),
        "edge_index": _model_tensor_payload(
            getattr(graph, "edge_index", None), field="model_graph.edge_index"
        ),
        "edge_attr": None
        if edge_attr is None
        else _model_tensor_payload(edge_attr, field="model_graph.edge_attr"),
    }


def _normalize_model_graph_payload(value: Any) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("schema_version")) is not str:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC model graph evidence is malformed"
        )
    try:
        # Canonical JSON is both a deep, type-stable copy and a finite-value
        # check.  It preserves the ordered tensor values; it is never decoded
        # into a replacement model graph.
        return json.loads(_canonical_bytes(value).decode("utf-8"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC model graph is not losslessly serializable"
        ) from exc


def _embedding_sha256(row: Any) -> str:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - AutoDL dependency
        raise TasteComRecGCSmokeError(
            "NumPy is required for Taste COMRECGC distance evidence"
        ) from exc
    value = np.ascontiguousarray(row)
    if value.ndim != 1 or value.size <= 0 or not np.isfinite(value).all():
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC GINE distance embedding is malformed"
        )
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(_canonical_bytes(list(value.shape)))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _scalar_native_int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        values = value.reshape(-1).tolist()
        if len(values) != 1:
            raise TasteComRecGCSmokeError(f"{field} must be one scalar")
        value = values[0]
    return _native_int(value, field=field, minimum=minimum)


def _lineage_evidence(graph: Any, *, graph_identity_sha256: str) -> str:
    source_index = _scalar_native_int(
        getattr(graph, "comrecgc_source_index", None),
        field="graph.comrecgc_source_index",
        minimum=0,
    )
    origins = _tensor_list(
        getattr(graph, "comrecgc_node_origin", None),
        field="graph.comrecgc_node_origin",
    )
    if len(origins) != int(getattr(graph, "num_nodes")):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC lineage length differs from graph nodes"
        )
    normalized: list[int] = []
    for index, value in enumerate(origins):
        if type(value) is not int or value < -1:
            raise TasteComRecGCSmokeError(
                f"graph.comrecgc_node_origin[{index}] is invalid"
            )
        normalized.append(value)
    return _canonical_sha256(
        {
            "schema_version": "tastemolnet_comrecgc_lineage_v1",
            "graph_identity_sha256": graph_identity_sha256,
            "source_cohort_index": source_index,
            "node_origin": normalized,
        }
    )


@dataclass(frozen=True, slots=True)
class TasteComRecGCRecord:
    graph_identity_sha256: str
    canonical_graph: str
    probabilities: tuple[float, float, float]
    prediction: int
    score: float
    candidate: bool
    valid_fullgraph: bool
    model_graph_sha256: str
    model_graph_payload: Mapping[str, Any]
    embedding_sha256: str
    embedding_dtype: str
    embedding_values: tuple[float, ...]

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "graph_identity_sha256": self.graph_identity_sha256,
            "canonical_graph": self.canonical_graph,
            "probabilities": list(self.probabilities),
            "prediction": self.prediction,
            "score": self.score,
            "candidate": self.candidate,
            "valid_fullgraph": self.valid_fullgraph,
            "model_graph_sha256": self.model_graph_sha256,
            "model_graph_payload": dict(self.model_graph_payload),
            "embedding_sha256": self.embedding_sha256,
            "embedding_dtype": self.embedding_dtype,
            "embedding_values": list(self.embedding_values),
        }


class TasteComRecGCMulticlassBridge:
    """Serial bridge for the pinned stateful multi-head COMRECGC module."""

    def __init__(
        self,
        *,
        adapter: Any,
        feature_atomic_numbers: Sequence[Any],
    ) -> None:
        self.adapter = adapter
        self.feature_atomic_numbers = tuple(feature_atomic_numbers)
        self.records: dict[str, TasteComRecGCRecord] = {}
        self.graph_collision_payloads: dict[str, dict[str, Any]] = {}
        self.lineage_occurrences: dict[str, Counter[str]] = {}
        self._pending_hashes: deque[tuple[str, str]] = deque()
        self.call_count = 0
        self.evaluated_graph_count = 0
        self.calculate_hash_count = 0

    @property
    def pending_hash_count(self) -> int:
        return len(self._pending_hashes)

    def _assert_idle(self) -> None:
        if self._pending_hashes:
            raise TasteComRecGCSmokeError(
                "Official COMRECGC did not consume every structural identity"
            )

    def call(
        self,
        graphs: Sequence[Any],
        _importance_args: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        """Return native importance parts and frozen-GINE distance vectors."""

        self._assert_idle()
        values = list(graphs)
        if not values:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC native scorer received an empty graph batch"
            )
        batch = self.adapter.score(values)
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - AutoDL dependency
            raise TasteComRecGCSmokeError(
                "NumPy is required for Taste COMRECGC"
            ) from exc
        probabilities = np.asarray(batch.probabilities)
        embeddings = np.asarray(batch.graph_embeddings)
        if probabilities.shape != (len(values), NUM_CLASSES):
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC adapter returned a non-three-class batch"
            )
        if embeddings.ndim != 2 or embeddings.shape[0] != len(values):
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC adapter returned unaligned distance embeddings"
            )
        valid = tuple(batch.valid_fullgraphs)
        if len(valid) != len(values) or any(type(value) is not bool for value in valid):
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC adapter validity evidence is untyped"
            )
        identity_payloads = tuple(
            getattr(batch, "identity_graph_payloads", ()) or ()
        )
        model_payloads = tuple(
            getattr(batch, "model_graph_payloads", ()) or ()
        )
        if identity_payloads and len(identity_payloads) != len(values):
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC identity-graph evidence is unaligned"
            )
        if model_payloads and len(model_payloads) != len(values):
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC model-graph evidence is unaligned"
            )
        importance: list[tuple[float, float]] = []
        canonical_embeddings: list[Any] = []
        for index, graph in enumerate(values):
            native_identity = canonical_attributed_graph(
                graph, feature_atomic_numbers=self.feature_atomic_numbers
            )
            supplied_identity = identity_payloads[index] if identity_payloads else None
            if supplied_identity is None:
                collision = native_identity.collision_payload()
            elif type(supplied_identity) is dict:
                collision = dict(supplied_identity)
            else:
                raise TasteComRecGCSmokeError(
                    "Taste COMRECGC identity-graph evidence is malformed"
                )
            graph_identity_sha256 = _identity_graph_sha256(collision)
            previous_collision = self.graph_collision_payloads.get(
                graph_identity_sha256
            )
            if previous_collision is not None and previous_collision != collision:
                raise TasteComRecGCSmokeError(
                    "Stable attributed graph SHA-256 collision detected"
                )
            self.graph_collision_payloads[graph_identity_sha256] = collision
            supplied_model = model_payloads[index] if model_payloads else None
            model_graph_payload = _normalize_model_graph_payload(
                _native_model_graph_payload(graph)
                if supplied_model is None
                else supplied_model
            )
            model_graph_sha256 = _canonical_sha256(model_graph_payload)
            row = tuple(float(value) for value in probabilities[index].tolist())
            score, prediction, candidate = score_and_candidate(row)
            raw_embedding = np.ascontiguousarray(embeddings[index])
            if raw_embedding.ndim != 1 or raw_embedding.dtype.kind != "f":
                raise TasteComRecGCSmokeError(
                    "Taste COMRECGC GINE distance embedding must be one floating row"
                )
            embedding_sha = _embedding_sha256(raw_embedding)
            observed = TasteComRecGCRecord(
                graph_identity_sha256=graph_identity_sha256,
                canonical_graph=collision["canonical_graph"],
                probabilities=row,
                prediction=prediction,
                score=score,
                candidate=bool(valid[index] and candidate),
                valid_fullgraph=valid[index],
                model_graph_sha256=model_graph_sha256,
                model_graph_payload=model_graph_payload,
                embedding_sha256=embedding_sha,
                embedding_dtype=raw_embedding.dtype.str,
                embedding_values=tuple(
                    float(value) for value in raw_embedding.tolist()
                ),
            )
            previous = self.records.get(graph_identity_sha256)
            if previous is None:
                record = observed
                self.records[graph_identity_sha256] = record
            else:
                # CUDA scatter pooling may differ in low bits when the same
                # structural graph is rescored in another official batch.
                # Keep the first frozen-GINE row as the canonical distance
                # vector and importance for this structural identity.  A
                # discrete validity/prediction/candidate change remains a
                # semantic failure; shape/dtype drift or anything beyond the
                # frozen low-bit allclose envelope is also rejected.  This
                # prevents one stable graph key from acquiring multiple
                # recourse vectors without masking a genuinely different
                # molecule/model result.
                cached_embedding = np.asarray(
                    previous.embedding_values,
                    dtype=np.dtype(previous.embedding_dtype),
                )
                if (
                    previous.canonical_graph != observed.canonical_graph
                    or previous.model_graph_sha256
                    != observed.model_graph_sha256
                    or previous.model_graph_payload
                    != observed.model_graph_payload
                    or previous.valid_fullgraph is not observed.valid_fullgraph
                    or previous.prediction != observed.prediction
                    or previous.candidate is not observed.candidate
                    or previous.embedding_dtype != observed.embedding_dtype
                    or len(previous.embedding_values)
                    != len(observed.embedding_values)
                    or not np.allclose(
                        np.asarray(observed.probabilities, dtype=np.float64),
                        np.asarray(previous.probabilities, dtype=np.float64),
                        rtol=GINE_CANONICAL_REUSE_RTOL,
                        atol=GINE_CANONICAL_REUSE_ATOL,
                    )
                    or not np.allclose(
                        raw_embedding,
                        cached_embedding,
                        rtol=GINE_CANONICAL_REUSE_RTOL,
                        atol=GINE_CANONICAL_REUSE_ATOL,
                    )
                ):
                    raise TasteComRecGCSmokeError(
                        "One parent-free graph identity changed GINE semantics"
                    )
                record = previous
            canonical_embedding = np.asarray(
                record.embedding_values,
                dtype=np.dtype(record.embedding_dtype),
            )
            if _embedding_sha256(canonical_embedding) != record.embedding_sha256:
                raise TasteComRecGCSmokeError(
                    "Cached Taste COMRECGC distance embedding changed"
                )
            canonical_embeddings.append(canonical_embedding)
            lineage = _lineage_evidence(
                graph,
                graph_identity_sha256=graph_identity_sha256,
            )
            self.lineage_occurrences.setdefault(
                graph_identity_sha256, Counter()
            )[lineage] += 1
            self._pending_hashes.append(
                (graph_identity_sha256, record.embedding_sha256)
            )
            importance.append((record.score, 1.0))
        self.call_count += 1
        self.evaluated_graph_count += len(values)
        return np.asarray(importance, dtype=float), np.stack(canonical_embeddings)

    def calculate_hash(self, graph_embedding: Any) -> str:
        """Consume the next structural identity; never hash embedding bytes."""

        if not self._pending_hashes:
            raise TasteComRecGCSmokeError(
                "Official COMRECGC requested an identity without a graph batch"
            )
        graph_identity, expected_embedding_sha = self._pending_hashes.popleft()
        if _embedding_sha256(graph_embedding) != expected_embedding_sha:
            raise TasteComRecGCSmokeError(
                "Official COMRECGC graph/embedding call order drifted"
            )
        self.calculate_hash_count += 1
        return graph_identity

    def is_graph_counterfactual(self, graph_hash: Any) -> bool:
        if type(graph_hash) is not str or graph_hash not in self.records:
            raise TasteComRecGCSmokeError(
                "Official COMRECGC queried an unknown structural identity"
            )
        return self.records[graph_hash].candidate

    def checkpoint_state(self) -> dict[str, Any]:
        self._assert_idle()
        return {
            "schema_version": "tastemolnet_comrecgc_bridge_checkpoint_v3",
            "records": {
                key: value.semantic_payload()
                for key, value in sorted(self.records.items())
            },
            "graph_collision_payloads": dict(
                sorted(self.graph_collision_payloads.items())
            ),
            "lineage_occurrences": {
                key: dict(sorted(values.items()))
                for key, values in sorted(self.lineage_occurrences.items())
            },
            "call_count": self.call_count,
            "evaluated_graph_count": self.evaluated_graph_count,
            "calculate_hash_count": self.calculate_hash_count,
        }

    def restore_checkpoint_state(self, payload: Mapping[str, Any]) -> None:
        self._assert_idle()
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - AutoDL dependency
            raise TasteComRecGCSmokeError(
                "NumPy is required to restore the COMRECGC bridge"
            ) from exc
        if (
            type(payload) is not dict
            or payload.get("schema_version")
            != "tastemolnet_comrecgc_bridge_checkpoint_v3"
        ):
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC bridge checkpoint schema changed"
            )
        raw_records = payload.get("records")
        raw_collisions = payload.get("graph_collision_payloads")
        raw_lineages = payload.get("lineage_occurrences")
        if not all(type(value) is dict for value in (raw_records, raw_collisions, raw_lineages)):
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC bridge checkpoint is incomplete"
            )
        records: dict[str, TasteComRecGCRecord] = {}
        for key, raw in raw_records.items():
            if type(raw) is not dict:
                raise TasteComRecGCSmokeError(
                    "Taste COMRECGC checkpoint record is malformed"
                )
            key = _lower_sha256(key, field="checkpoint graph identity")
            if set(raw) != {
                "graph_identity_sha256",
                "canonical_graph",
                "probabilities",
                "prediction",
                "score",
                "candidate",
                "valid_fullgraph",
                "model_graph_sha256",
                "model_graph_payload",
                "embedding_sha256",
                "embedding_dtype",
                "embedding_values",
            }:
                raise TasteComRecGCSmokeError(
                    "Taste COMRECGC checkpoint record keys changed"
                )
            probabilities = raw.get("probabilities")
            score, prediction, candidate = score_and_candidate(probabilities)
            if (
                raw.get("graph_identity_sha256") != key
                or type(raw.get("prediction")) is not int
                or raw.get("prediction") != prediction
                or isinstance(raw.get("score"), bool)
                or not isinstance(raw.get("score"), (int, float))
                or raw.get("score") != score
                or type(raw.get("valid_fullgraph")) is not bool
                or type(raw.get("candidate")) is not bool
                or raw.get("candidate")
                is not bool(raw["valid_fullgraph"] and candidate)
                or type(raw.get("canonical_graph")) is not str
                or not raw["canonical_graph"]
            ):
                raise TasteComRecGCSmokeError(
                    "Taste COMRECGC checkpoint semantics drifted"
                )
            embedding_sha = _lower_sha256(
                raw.get("embedding_sha256"), field="checkpoint embedding SHA"
            )
            model_graph_payload = _normalize_model_graph_payload(
                raw.get("model_graph_payload")
            )
            model_graph_sha = _lower_sha256(
                raw.get("model_graph_sha256"),
                field="checkpoint model graph SHA",
            )
            if _canonical_sha256(model_graph_payload) != model_graph_sha:
                raise TasteComRecGCSmokeError(
                    "Taste COMRECGC checkpoint model graph bytes drifted"
                )
            embedding_dtype = raw.get("embedding_dtype")
            embedding_values = raw.get("embedding_values")
            if (
                type(embedding_dtype) is not str
                or type(embedding_values) is not list
                or not embedding_values
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in embedding_values
                )
            ):
                raise TasteComRecGCSmokeError(
                    "Taste COMRECGC checkpoint embedding payload drifted"
                )
            try:
                embedding_array = np.asarray(
                    embedding_values, dtype=np.dtype(embedding_dtype)
                )
            except (TypeError, ValueError) as exc:
                raise TasteComRecGCSmokeError(
                    "Taste COMRECGC checkpoint embedding dtype is invalid"
                ) from exc
            if (
                embedding_array.ndim != 1
                or embedding_array.dtype.kind != "f"
                or _embedding_sha256(embedding_array) != embedding_sha
            ):
                raise TasteComRecGCSmokeError(
                    "Taste COMRECGC checkpoint embedding bytes drifted"
                )
            records[key] = TasteComRecGCRecord(
                graph_identity_sha256=key,
                canonical_graph=raw["canonical_graph"],
                probabilities=tuple(float(value) for value in probabilities),
                prediction=prediction,
                score=score,
                candidate=bool(raw["valid_fullgraph"] and candidate),
                valid_fullgraph=raw["valid_fullgraph"],
                model_graph_sha256=model_graph_sha,
                model_graph_payload=model_graph_payload,
                embedding_sha256=embedding_sha,
                embedding_dtype=embedding_array.dtype.str,
                embedding_values=tuple(
                    float(value) for value in embedding_array.tolist()
                ),
            )
        lineages: dict[str, Counter[str]] = {}
        for key, raw in raw_lineages.items():
            if key not in records or type(raw) is not dict:
                raise TasteComRecGCSmokeError(
                    "Taste COMRECGC checkpoint lineage is malformed"
                )
            counter: Counter[str] = Counter()
            for lineage, count in raw.items():
                lineage = _lower_sha256(
                    lineage, field="checkpoint lineage SHA"
                )
                counter[lineage] = _native_int(
                    count, field="lineage occurrence", minimum=1
                )
            if not counter:
                raise TasteComRecGCSmokeError(
                    "Taste COMRECGC checkpoint lost graph lineage"
                )
            lineages[key] = counter
        if set(raw_collisions) != set(records) or set(lineages) != set(records):
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC checkpoint identity domains differ"
            )
        self.records = records
        collision_payloads: dict[str, dict[str, Any]] = {}
        for key, raw in raw_collisions.items():
            if (
                type(raw) is not dict
                or set(raw) != {"canonical_graph", "num_nodes", "num_edges"}
                or raw.get("canonical_graph") != records[key].canonical_graph
                or type(raw.get("num_nodes")) is not int
                or raw["num_nodes"] < 0
                or type(raw.get("num_edges")) is not int
                or raw["num_edges"] < 0
                or (raw["num_nodes"] == 0 and raw["num_edges"] != 0)
                or (
                    raw["num_nodes"] == 0
                    and raw["canonical_graph"]
                    != "<EMPTY_ATTRIBUTED_GRAPH>"
                )
            ):
                raise TasteComRecGCSmokeError(
                    "Taste COMRECGC checkpoint collision evidence drifted"
                )
            expected_key = _canonical_sha256(
                {
                    "schema_version": "tastemolnet_comrecgc_attributed_graph_v1",
                    "canonical_graph": raw["canonical_graph"],
                    "num_nodes": raw["num_nodes"],
                    "num_edges": raw["num_edges"],
                    "node_attribute": "atomic_number_from_frozen_one_hot",
                    "edge_attribute": "official_untyped_edge",
                }
            )
            if expected_key != key:
                raise TasteComRecGCSmokeError(
                    "Taste COMRECGC checkpoint graph identity is not canonical"
                )
            collision_payloads[key] = dict(raw)
        self.graph_collision_payloads = collision_payloads
        self.lineage_occurrences = lineages
        self.call_count = _native_int(
            payload.get("call_count"), field="call_count", minimum=0
        )
        self.evaluated_graph_count = _native_int(
            payload.get("evaluated_graph_count"),
            field="evaluated_graph_count",
            minimum=0,
        )
        self.calculate_hash_count = _native_int(
            payload.get("calculate_hash_count"),
            field="calculate_hash_count",
            minimum=0,
        )
        if self.evaluated_graph_count != self.calculate_hash_count:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC checkpoint lost structural hash consumption"
            )

    def report(self) -> dict[str, Any]:
        self._assert_idle()
        candidates = sum(record.candidate for record in self.records.values())
        destinations = {
            str(label): sum(
                record.candidate and record.prediction == label
                for record in self.records.values()
            )
            for label in (0, 2)
        }
        return {
            "schema_version": "tastemolnet_comrecgc_multiclass_bridge_v1",
            "num_classes": NUM_CLASSES,
            "source_label": SOURCE_LABEL,
            "candidate_condition": "predicted_label != source_label",
            "importance": "1.0 - probabilities[:, source_label]",
            "graph_identity": "canonical_parent_free_attributed_graph_sha256",
            "embedding_identity_used": False,
            "python_builtin_hash_used": False,
            "parent_metadata_in_graph_identity": False,
            "distance_embedding": "frozen_gine_graph_hidden",
            "canonical_row_policy": "first_allclose_frozen_gine_row_reused",
            "canonical_reuse_rtol": GINE_CANONICAL_REUSE_RTOL,
            "canonical_reuse_atol": GINE_CANONICAL_REUSE_ATOL,
            "canonical_row_reuse_count": self.evaluated_graph_count
            - len(self.records),
            "canonical_row_cache_checkpointed": True,
            "call_count": self.call_count,
            "evaluated_graph_count": self.evaluated_graph_count,
            "calculate_hash_count": self.calculate_hash_count,
            "unique_graph_count": len(self.records),
            "evaluated_strict_graph_count": candidates,
            "destination_prediction_counts": destinations,
            "unique_lineage_count": sum(
                len(values) for values in self.lineage_occurrences.values()
            ),
            "lineage_occurrence_count": sum(
                sum(values.values())
                for values in self.lineage_occurrences.values()
            ),
        }

    @contextmanager
    def installed(self, module: Any, *, neighbor_wrapper: Any) -> Iterator[None]:
        """Install the narrow patch and restore every upstream global."""

        self._assert_idle()
        originals = {
            "call": module.call,
            "calculate_hash": module.calculate_hash,
            "is_graph_counterfactual": module.is_graph_counterfactual,
            "neighbor_graph_access": module.neighbor_graph_access,
        }
        module.call = self.call
        module.calculate_hash = self.calculate_hash
        module.is_graph_counterfactual = self.is_graph_counterfactual
        module.neighbor_graph_access = neighbor_wrapper(
            originals["neighbor_graph_access"]
        )
        try:
            yield
            self._assert_idle()
        finally:
            module.call = originals["call"]
            module.calculate_hash = originals["calculate_hash"]
            module.is_graph_counterfactual = originals[
                "is_graph_counterfactual"
            ]
            module.neighbor_graph_access = originals["neighbor_graph_access"]


def _seed_all(seed: int) -> None:
    try:
        import numpy as np
        import torch
    except Exception as exc:  # pragma: no cover - AutoDL dependency
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC requires NumPy and PyTorch"
        ) from exc
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [value.clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng_state(value: Mapping[str, Any]) -> None:
    import numpy as np
    import torch

    if type(value) is not dict or set(value) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC RNG checkpoint is incomplete"
        )
    cuda = value["torch_cuda"]
    expected_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if type(cuda) is not list or len(cuda) != expected_cuda:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC CUDA RNG topology changed at reload"
        )
    try:
        random.setstate(value["python"])
        np.random.set_state(value["numpy"])
        torch.set_rng_state(value["torch_cpu"])
        if cuda:
            torch.cuda.set_rng_state_all(cuda)
    except Exception as exc:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC RNG checkpoint cannot be restored"
        ) from exc


def _torch_load_handle(handle: Any) -> Any:
    import torch

    handle.seek(0)
    try:
        return torch.load(handle, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - pinned AutoDL torch
        handle.seek(0)
        return torch.load(handle, map_location="cpu")


def _read_exact_fd(descriptor: int, size: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(8 * 1024 * 1024, size - offset), offset)
        if not chunk:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC checkpoint ended before its frozen size"
            )
        chunks.append(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, size):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC checkpoint grew during retained read"
        )
    return b"".join(chunks)


def _checkpoint_file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_uid),
        int(value.st_gid),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _checkpoint_directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_uid),
        int(value.st_gid),
    )


def _write_reload_checkpoint(
    *,
    module: Any,
    bridge: TasteComRecGCMulticlassBridge,
    loop_state: Any,
    parameters: TasteComRecGCSmokeParameters,
    path: Path,
) -> dict[str, Any]:
    """Persist and reload one exact private midpoint checkpoint."""

    import torch

    from src.baselines.comrecgc.generation_loop import snapshot_official_state

    bridge_state = bridge.checkpoint_state()
    if loop_state.completed_step != parameters.checkpoint_step:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC checkpoint is not at the frozen midpoint"
        )
    loop_state_payload = loop_state.to_checkpoint_state()
    official_state = snapshot_official_state(module)
    transition_state = dict(module.transitions)
    rng_state = _capture_rng_state()
    state_hashes = {
        "loop_state_sha256": _checkpoint_state_sha256(
            loop_state_payload, field="loop_state"
        ),
        "official_state_sha256": _checkpoint_state_sha256(
            official_state, field="official_state"
        ),
        "transition_state_sha256": _checkpoint_state_sha256(
            transition_state, field="transition_state"
        ),
        "bridge_state_sha256": _checkpoint_state_sha256(
            bridge_state, field="bridge_state"
        ),
        "rng_state_sha256": _checkpoint_state_sha256(
            rng_state, field="rng_state"
        ),
    }
    state_hashes["complete_state_sha256"] = _canonical_sha256(state_hashes)
    payload = {
        "schema_version": "tastemolnet_comrecgc_smoke_checkpoint_v1",
        "boundary": "after_fully_completed_step",
        "completed_step": parameters.checkpoint_step,
        "total_steps": parameters.steps,
        "parameters": asdict(parameters),
        "loop_state": loop_state_payload,
        "official_state": official_state,
        "official_transitions": transition_state,
        "bridge_state": bridge_state,
        "rng_state": rng_state,
        "state_hashes": state_hashes,
    }
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    file_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    for flag_name in ("O_CLOEXEC", "O_NOFOLLOW"):
        directory_flags |= getattr(os, flag_name, 0)
        file_flags |= getattr(os, flag_name, 0)
    directory_fd = os.open(path.parent, directory_flags)
    descriptor = -1
    try:
        directory_identity = os.fstat(directory_fd)
        descriptor = os.open(path.name, file_flags, 0o600, dir_fd=directory_fd)
        with os.fdopen(os.dup(descriptor), "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(directory_fd)
        identity = os.fstat(descriptor)
        if (
            not stat.S_ISREG(identity.st_mode)
            or identity.st_nlink != 1
            or stat.S_IMODE(identity.st_mode) != 0o600
        ):
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC checkpoint physical identity is unsafe"
            )
        if _checkpoint_file_identity(identity) != _checkpoint_file_identity(
            os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        ):
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC named checkpoint changed after save"
            )
        raw = _read_exact_fd(descriptor, int(identity.st_size))
        if not raw:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC checkpoint serialization is empty"
            )
        checkpoint_sha = hashlib.sha256(raw).hexdigest()
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            reloaded = _torch_load_handle(handle)
        final_identity = os.fstat(descriptor)
        named_identity = os.stat(
            path.name, dir_fd=directory_fd, follow_symlinks=False
        )
        if (
            _checkpoint_file_identity(identity)
            != _checkpoint_file_identity(final_identity)
            or _checkpoint_file_identity(identity)
            != _checkpoint_file_identity(named_identity)
            or _checkpoint_directory_identity(os.fstat(directory_fd))
            != _checkpoint_directory_identity(directory_identity)
            or _read_exact_fd(descriptor, int(identity.st_size)) != raw
            or type(reloaded) is not dict
            or set(reloaded) != set(payload)
            or reloaded.get("schema_version")
            != "tastemolnet_comrecgc_smoke_checkpoint_v1"
            or reloaded.get("boundary") != "after_fully_completed_step"
            or type(reloaded.get("completed_step")) is not int
            or reloaded["completed_step"] != parameters.checkpoint_step
            or type(reloaded.get("total_steps")) is not int
            or reloaded["total_steps"] != parameters.steps
            or reloaded.get("parameters") != asdict(parameters)
            or type(reloaded.get("state_hashes")) is not dict
            or reloaded["state_hashes"] != state_hashes
        ):
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC checkpoint reload closure changed"
            )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_fd)
    return {
        "payload": reloaded,
        "evidence": {
            "schema_version": "tastemolnet_comrecgc_checkpoint_reload_v1",
            "checkpoint_step": parameters.checkpoint_step,
            "next_step": parameters.checkpoint_step + 1,
            "total_steps": parameters.steps,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_bytes": len(raw),
            **state_hashes,
            "checkpoint_reloaded": True,
            "checkpoint_persisted_in_output": False,
        },
    }


def _restore_reload_checkpoint(
    *,
    module: Any,
    bridge: TasteComRecGCMulticlassBridge,
    loaded: Mapping[str, Any],
) -> Any:
    from src.baselines.comrecgc.generation_loop import (
        GenerationLoopState,
        restore_official_state,
        snapshot_official_state,
    )

    payload = loaded["payload"]
    restore_official_state(module, payload["official_state"])
    transitions = payload["official_transitions"]
    if type(transitions) is not dict:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC checkpoint transition state is malformed"
        )
    module.transitions = dict(transitions)
    bridge.restore_checkpoint_state(payload["bridge_state"])
    _restore_rng_state(payload["rng_state"])
    state = GenerationLoopState.from_checkpoint_state(payload["loop_state"])
    if state.completed_step != payload["completed_step"]:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC loop checkpoint step changed"
        )
    observed_hashes = {
        "loop_state_sha256": _checkpoint_state_sha256(
            state.to_checkpoint_state(), field="restored.loop_state"
        ),
        "official_state_sha256": _checkpoint_state_sha256(
            snapshot_official_state(module), field="restored.official_state"
        ),
        "transition_state_sha256": _checkpoint_state_sha256(
            dict(module.transitions), field="restored.transition_state"
        ),
        "bridge_state_sha256": _checkpoint_state_sha256(
            bridge.checkpoint_state(), field="restored.bridge_state"
        ),
        "rng_state_sha256": _checkpoint_state_sha256(
            _capture_rng_state(), field="restored.rng_state"
        ),
    }
    observed_hashes["complete_state_sha256"] = _canonical_sha256(
        observed_hashes
    )
    if observed_hashes != payload.get("state_hashes"):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC restored midpoint state differs from serialized state"
        )
    return state


def _initialize_source_graphs(
    *,
    checkpoint_payloads: Mapping[str, bytes],
    source_rows: Sequence[Any],
    graph_schema: Any,
    device: str,
    parameters: TasteComRecGCSmokeParameters,
) -> tuple[list[Any], list[Mapping[str, Any]], Any, dict[str, Any]]:
    """Select a deterministic train-only Sweet cohort with the frozen GINE."""

    from src.baselines.tastemolnet_gcf_smoke import (
        TasteFrozenGINENativeAdapter,
        encode_taste_source_graph,
        taste_record_to_pyg,
    )

    if len(source_rows) != parameters.source_pool:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC source pool differs from the frozen train-only pool"
        )
    pool_records = [
        encode_taste_source_graph(row, graph_schema) for row in source_rows
    ]
    pool_graphs = [
        taste_record_to_pyg(record, origin_index=index)
        for index, record in enumerate(pool_records)
    ]
    pool_adapter = TasteFrozenGINENativeAdapter(
        checkpoint_payloads,
        source_records=pool_records,
        graph_schema=graph_schema,
        device=device,
    )
    scored = pool_adapter.score(pool_graphs)
    if (
        len(scored.valid_fullgraphs) != len(pool_graphs)
        or len(scored.predictions) != len(pool_graphs)
        or any(type(value) is not bool for value in scored.valid_fullgraphs)
        or any(
            type(value) is not int or value not in range(NUM_CLASSES)
            for value in scored.predictions
        )
    ):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC source-pool GINE evidence is malformed"
        )
    selected_indices: list[int] = []
    selected_identity_order: list[str] = []
    selected_identity_set: set[str] = set()
    pool_identity_payloads = tuple(
        getattr(scored, "identity_graph_payloads", ()) or ()
    )
    if pool_identity_payloads and len(pool_identity_payloads) != len(pool_graphs):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC source identity evidence is unaligned"
        )
    for index, (valid, prediction) in enumerate(
        zip(scored.valid_fullgraphs, scored.predictions, strict=True)
    ):
        if not valid or prediction != SOURCE_LABEL:
            continue
        supplied_identity = (
            pool_identity_payloads[index] if pool_identity_payloads else None
        )
        identity = (
            _identity_graph_sha256(dict(supplied_identity))
            if type(supplied_identity) is dict
            else canonical_attributed_graph(
                pool_graphs[index],
                feature_atomic_numbers=graph_schema.feature_atomic_numbers,
            ).graph_identity_sha256
        )
        if identity in selected_identity_set:
            continue
        selected_indices.append(index)
        selected_identity_order.append(identity)
        selected_identity_set.add(identity)
        if len(selected_indices) == parameters.source_count:
            break
    if len(selected_indices) != parameters.source_count:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC source pool lacks enough GINE-predicted Sweet parents"
        )
    selected_records = [pool_records[index] for index in selected_indices]
    graphs: list[Any] = []
    for source_index, record in enumerate(selected_records):
        graph = taste_record_to_pyg(record, origin_index=source_index)
        # GCF lineage remains the frozen-GINE decoder authority.  COMRECGC
        # lineage is a separate stateful-walk audit and never graph identity.
        graph.comrecgc_node_origin = graph.gcf_node_origin.clone()
        graph.comrecgc_source_index = source_index
        graph.comrecgc_parent_id = f"private-cohort-{source_index:03d}"
        graphs.append(graph)
    adapter = TasteFrozenGINENativeAdapter(
        checkpoint_payloads,
        source_records=selected_records,
        graph_schema=graph_schema,
        device=device,
    )
    replay = adapter.score(graphs)
    if (
        len(replay.valid_fullgraphs) != len(graphs)
        or len(replay.predictions) != len(graphs)
        or any(type(value) is not bool for value in replay.valid_fullgraphs)
        or any(type(value) is not int for value in replay.predictions)
        or any(
            not valid or prediction != SOURCE_LABEL
            for valid, prediction in zip(
                replay.valid_fullgraphs, replay.predictions, strict=True
            )
        )
    ):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC selected source changed during exact replay"
        )
    identities = [
        canonical_attributed_graph(
            graph, feature_atomic_numbers=graph_schema.feature_atomic_numbers
        ).graph_identity_sha256
        for graph in graphs
    ]
    if identities != selected_identity_order or len(set(identities)) != len(
        identities
    ):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC selected source cohort identity changed"
        )
    evidence = {
        "schema_version": "tastemolnet_comrecgc_source_cohort_v1",
        "source_split": "train",
        "source_label": SOURCE_LABEL,
        "source_pool_count": parameters.source_pool,
        "source_count": parameters.source_count,
        "source_cohort_sha256": _canonical_sha256(identities),
        "source_graph_identities_unique": True,
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "molecule_identifiers_persisted": False,
    }
    return graphs, selected_records, adapter, evidence


def _common_recourse_summary(
    *,
    modules: Mapping[str, Any],
    module: Any,
    bridge: TasteComRecGCMulticlassBridge,
    source_graphs: Sequence[Any],
    adapter: Any,
    parameters: TasteComRecGCSmokeParameters,
) -> dict[str, Any]:
    """Run native DBSCAN/coverage/greedy over frozen-GINE recourse vectors."""

    import numpy as np
    import torch
    from sklearn.cluster import DBSCAN

    from src.baselines.comrecgc.recourse import trace_official_cluster_order

    candidates: list[tuple[str, Any, Any, int]] = []
    for raw in module.counterfactual_candidates:
        if type(raw) is not dict:
            raise TasteComRecGCSmokeError(
                "Official COMRECGC candidate record is malformed"
            )
        graph_hash = raw.get("graph_hash")
        if type(graph_hash) is not str or graph_hash not in bridge.records:
            raise TasteComRecGCSmokeError(
                "Official COMRECGC candidate lost structural identity evidence"
            )
        if not bridge.records[graph_hash].candidate:
            continue
        graph_value = module.graph_map.get(graph_hash)
        if not isinstance(graph_value, (list, tuple)) or len(graph_value) != 3:
            raise TasteComRecGCSmokeError(
                "Official COMRECGC candidate graph payload is malformed"
            )
        graph, embedding, elements = graph_value
        if _embedding_sha256(embedding) != bridge.records[graph_hash].embedding_sha256:
            raise TasteComRecGCSmokeError(
                "Official COMRECGC candidate distance embedding drifted"
            )
        frequency = _native_int(raw.get("frequency"), field="frequency", minimum=1)
        candidates.append((graph_hash, graph, embedding, frequency))
    if not candidates:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC walk produced no strict non-Sweet candidate"
        )
    source_batch = adapter.score(list(source_graphs))
    if (
        len(source_batch.valid_fullgraphs) != len(source_graphs)
        or len(source_batch.predictions) != len(source_graphs)
        or any(type(value) is not bool for value in source_batch.valid_fullgraphs)
        or any(type(value) is not int for value in source_batch.predictions)
        or any(
            not valid or prediction != SOURCE_LABEL
            for valid, prediction in zip(
                source_batch.valid_fullgraphs,
                source_batch.predictions,
                strict=True,
            )
        )
    ):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC source GINE semantics changed before recourse"
        )
    source_embeddings = np.asarray(source_batch.graph_embeddings, dtype=float)
    if (
        source_embeddings.ndim != 2
        or source_embeddings.shape[0] != len(source_graphs)
        or not np.isfinite(source_embeddings).all()
    ):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC source distance embeddings are malformed"
        )
    source_elements = (
        modules["util"].graph_element_counts(list(source_graphs))
        .detach()
        .cpu()
        .numpy()
        .astype(float)
    )
    candidate_graphs = [value[1] for value in candidates]
    candidate_embeddings = np.asarray([value[2] for value in candidates], dtype=float)
    candidate_elements = (
        modules["util"].graph_element_counts(candidate_graphs)
        .detach()
        .cpu()
        .numpy()
        .astype(float)
    )
    if candidate_embeddings.ndim != 2 or not np.isfinite(candidate_embeddings).all():
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC candidate distance embeddings are malformed"
        )
    pair_indices: list[tuple[int, int]] = []
    recourse_vectors: list[Any] = []
    # Preserve the official common-recourse row order exactly.  The upstream
    # implementation obtains ``idxs`` from ``torch.where(S <= theta)`` where
    # ``S`` has shape [original, counterfactual], so rows are source-major.
    # DBSCAN border assignment and the later medoid/greedy tie breaks are
    # order-sensitive; a candidate-major transpose is therefore a scientific
    # trajectory change rather than a cosmetic reordering.
    for source_index, source_embedding in enumerate(source_embeddings):
        for candidate_index, candidate_embedding in enumerate(
            candidate_embeddings
        ):
            scale = candidate_elements[candidate_index] + source_elements[source_index]
            if not math.isfinite(float(scale)) or float(scale) <= 0.0:
                raise TasteComRecGCSmokeError(
                    "Taste COMRECGC graph-element normalization is invalid"
                )
            vector = (candidate_embedding - source_embedding) / float(scale)
            distance = float(np.linalg.norm(vector))
            if distance <= parameters.theta:
                pair_indices.append((source_index, candidate_index))
                recourse_vectors.append(vector)
    if not recourse_vectors:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC smoke has no theta-eligible recourse pair"
        )
    recourse_array = np.asarray(recourse_vectors, dtype=float)
    if not np.isfinite(recourse_array).all():
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC recourse vectors contain NaN/Inf"
        )
    dbscan = DBSCAN(eps=parameters.delta, min_samples=parameters.cluster_size).fit(
        recourse_array
    )
    labels = np.asarray(dbscan.labels_, dtype=int)
    official = modules["common_recourse"].coverage_summary(
        db_2=dbscan,
        rec=torch.tensor(recourse_array),
        idxs=pair_indices,
        radius=parameters.delta,
        threshold_theta=parameters.theta,
        recourse_size=parameters.recourse_size,
    )
    selected = trace_official_cluster_order(
        labels=labels,
        recourse_vectors=recourse_array,
        pair_indices=pair_indices,
        radius=parameters.delta,
        theta=parameters.theta,
        recourse_size=parameters.recourse_size,
        official_greedy=modules[
            "common_recourse"
        ].greedy_counterfactual_summary_from_covering_sets,
    )
    if not selected:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC native DBSCAN/greedy summary is empty"
        )
    rows: list[dict[str, Any]] = []
    for row in selected:
        candidate_index = _native_int(
            row.get("representative_counterfactual_index"),
            field="representative_counterfactual_index",
            minimum=0,
        )
        graph_hash, _graph, _embedding, frequency = candidates[candidate_index]
        record = bridge.records[graph_hash]
        if not record.candidate or record.prediction not in (0, 2):
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC greedy representative is not a strict flip"
            )
        rows.append(
            {
                "rank": _native_int(row.get("rank"), field="rank", minimum=1),
                "cluster_id": _native_int(
                    row.get("cluster_id"), field="cluster_id", minimum=0
                ),
                "representative_graph_identity_sha256": graph_hash,
                "destination_label": record.prediction,
                "score": record.score,
                "frequency": frequency,
                "covered_parent_count": len(row["covered_parent_indices_native"]),
                "cluster_size": _native_int(
                    row.get("cluster_size"), field="cluster_size", minimum=1
                ),
                "lineage_count": len(bridge.lineage_occurrences[graph_hash]),
            }
        )
    return {
        "schema_version": "tastemolnet_comrecgc_common_recourse_smoke_v1",
        "distance_embedding": "frozen_gine_graph_hidden",
        "theta": parameters.theta,
        "delta": parameters.delta,
        "cluster_size": parameters.cluster_size,
        "recourse_size": parameters.recourse_size,
        "retained_strict_candidate_count": len(candidates),
        "theta_eligible_pair_count": len(pair_indices),
        "dbscan_invoked": True,
        "dbscan_cluster_count": len({int(value) for value in labels if value >= 0}),
        "dbscan_noise_count": int(np.count_nonzero(labels < 0)),
        "official_coverage_summary_invoked": True,
        "official_coverage_summary_sha256": _canonical_sha256(
            [list(value) for value in official]
        ),
        "official_greedy_summary_invoked": True,
        "selected_common_recourse_count": len(rows),
        "selected_common_recourses": rows,
        "graph_payload_persisted": False,
        "molecule_payload_persisted": False,
    }


def validate_native_comrecgc_smoke_result(value: Any) -> dict[str, Any]:
    """Strictly reopen one bounded native smoke result.

    This boundary deliberately validates the scientific claims rather than
    merely accepting a self-consistent JSON object.  It is also used by the
    terminal-output consumer before a controller may adopt T9.
    """

    if type(value) is not dict:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC smoke result must be one object"
        )
    expected_top = {
        "schema_version",
        "stage",
        "dataset",
        "method",
        "parameters",
        "source_cohort",
        "checkpoint_reload",
        "bridge",
        "common_recourse",
        "official_native_random_walk",
        "official_stateful_heads_preserved",
        "official_rng_and_collector_serial",
        "random_walk_steps",
        "smoke_budget",
        "full_budget",
        "full_required_steps",
        "same_frozen_three_class_gine",
        "second_classifier_used",
        "rf_oracle_used",
        "validation_loaded",
        "calibration_payload_loaded",
        "test_loaded",
        "dataset_redistributed",
        "paper_result_eligible",
    }
    if set(value) != expected_top or any(
        value.get(key) != expected
        for key, expected in {
            "schema_version": "tastemolnet_comrecgc_native_smoke_v1",
            "stage": STAGE,
            "dataset": DATASET,
            "method": METHOD,
        }.items()
    ):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC smoke top-level schema changed"
        )
    for key, expected in {
        "official_native_random_walk": True,
        "official_stateful_heads_preserved": True,
        "official_rng_and_collector_serial": True,
        "smoke_budget": True,
        "full_budget": False,
        "same_frozen_three_class_gine": True,
        "second_classifier_used": False,
        "rf_oracle_used": False,
        "validation_loaded": False,
        "calibration_payload_loaded": False,
        "test_loaded": False,
        "dataset_redistributed": False,
        "paper_result_eligible": False,
    }.items():
        if type(value.get(key)) is not bool or value[key] is not expected:
            raise TasteComRecGCSmokeError(
                f"Taste COMRECGC smoke boolean changed: {key}"
            )
    if (
        type(value.get("random_walk_steps")) is not int
        or value["random_walk_steps"] != SMOKE_STEPS
        or type(value.get("full_required_steps")) is not int
        or value["full_required_steps"] != 50_000
    ):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC smoke/full budget boundary changed"
        )

    parameters = value.get("parameters")
    if type(parameters) is not dict or set(parameters) != set(
        TasteComRecGCSmokeParameters.__dataclass_fields__
    ):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC smoke parameter schema changed"
        )
    try:
        frozen = TasteComRecGCSmokeParameters(**parameters).validate()
    except (TypeError, TasteComRecGCSmokeError) as exc:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC smoke parameters changed"
        ) from exc
    if asdict(frozen) != parameters:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC smoke parameters are not canonical"
        )

    source = value.get("source_cohort")
    if (
        type(source) is not dict
        or set(source)
        != {
            "schema_version",
            "source_split",
            "source_label",
            "source_pool_count",
            "source_count",
            "source_cohort_sha256",
            "source_graph_identities_unique",
            "validation_loaded",
            "calibration_loaded",
            "test_loaded",
            "molecule_identifiers_persisted",
        }
        or source.get("schema_version")
        != "tastemolnet_comrecgc_source_cohort_v1"
        or source.get("source_split") != "train"
        or type(source.get("source_label")) is not int
        or source["source_label"] != SOURCE_LABEL
        or type(source.get("source_pool_count")) is not int
        or source["source_pool_count"] != frozen.source_pool
        or type(source.get("source_count")) is not int
        or source["source_count"] != frozen.source_count
    ):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC source-cohort contract changed"
        )
    _lower_sha256(
        source.get("source_cohort_sha256"), field="source cohort SHA"
    )
    for key, expected in {
        "source_graph_identities_unique": True,
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "molecule_identifiers_persisted": False,
    }.items():
        if type(source.get(key)) is not bool or source[key] is not expected:
            raise TasteComRecGCSmokeError(
                f"Taste COMRECGC source boundary changed: {key}"
            )

    checkpoint = value.get("checkpoint_reload")
    if (
        type(checkpoint) is not dict
        or set(checkpoint)
        != {
            "schema_version",
            "checkpoint_step",
            "next_step",
            "total_steps",
            "checkpoint_sha256",
            "checkpoint_bytes",
            "loop_state_sha256",
            "official_state_sha256",
            "transition_state_sha256",
            "bridge_state_sha256",
            "rng_state_sha256",
            "complete_state_sha256",
            "checkpoint_reloaded",
            "checkpoint_persisted_in_output",
        }
        or checkpoint.get("schema_version")
        != "tastemolnet_comrecgc_checkpoint_reload_v1"
        or type(checkpoint.get("checkpoint_step")) is not int
        or checkpoint["checkpoint_step"] != frozen.checkpoint_step
        or type(checkpoint.get("next_step")) is not int
        or checkpoint["next_step"] != frozen.checkpoint_step + 1
        or type(checkpoint.get("total_steps")) is not int
        or checkpoint["total_steps"] != frozen.steps
        or type(checkpoint.get("checkpoint_bytes")) is not int
        or checkpoint["checkpoint_bytes"] <= 0
        or checkpoint.get("checkpoint_reloaded") is not True
        or checkpoint.get("checkpoint_persisted_in_output") is not False
    ):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC checkpoint/reload contract changed"
        )
    _lower_sha256(
        checkpoint.get("checkpoint_sha256"), field="checkpoint SHA"
    )
    component_hashes: dict[str, str] = {}
    for key in (
        "loop_state_sha256",
        "official_state_sha256",
        "transition_state_sha256",
        "bridge_state_sha256",
        "rng_state_sha256",
    ):
        component_hashes[key] = _lower_sha256(
            checkpoint.get(key), field=key.replace("_", " ")
        )
    complete_state_sha = _lower_sha256(
        checkpoint.get("complete_state_sha256"), field="complete state SHA"
    )
    if complete_state_sha != _canonical_sha256(component_hashes):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC complete checkpoint state hash is not derived"
        )

    bridge = value.get("bridge")
    expected_bridge_keys = {
        "schema_version",
        "num_classes",
        "source_label",
        "candidate_condition",
        "importance",
        "graph_identity",
        "embedding_identity_used",
        "python_builtin_hash_used",
        "parent_metadata_in_graph_identity",
        "distance_embedding",
        "canonical_row_policy",
        "canonical_reuse_rtol",
        "canonical_reuse_atol",
        "canonical_row_reuse_count",
        "canonical_row_cache_checkpointed",
        "call_count",
        "evaluated_graph_count",
        "calculate_hash_count",
        "unique_graph_count",
        "evaluated_strict_graph_count",
        "destination_prediction_counts",
        "unique_lineage_count",
        "lineage_occurrence_count",
    }
    if (
        type(bridge) is not dict
        or set(bridge) != expected_bridge_keys
        or bridge.get("schema_version")
        != "tastemolnet_comrecgc_multiclass_bridge_v1"
        or bridge.get("num_classes") != NUM_CLASSES
        or type(bridge.get("num_classes")) is not int
        or bridge.get("source_label") != SOURCE_LABEL
        or type(bridge.get("source_label")) is not int
        or bridge.get("candidate_condition")
        != "predicted_label != source_label"
        or bridge.get("importance")
        != "1.0 - probabilities[:, source_label]"
        or bridge.get("graph_identity")
        != "canonical_parent_free_attributed_graph_sha256"
        or bridge.get("distance_embedding") != "frozen_gine_graph_hidden"
        or bridge.get("canonical_row_policy")
        != "first_allclose_frozen_gine_row_reused"
        or type(bridge.get("canonical_reuse_rtol")) is not float
        or bridge["canonical_reuse_rtol"] != GINE_CANONICAL_REUSE_RTOL
        or type(bridge.get("canonical_reuse_atol")) is not float
        or bridge["canonical_reuse_atol"] != GINE_CANONICAL_REUSE_ATOL
        or bridge.get("canonical_row_cache_checkpointed") is not True
    ):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC multiclass bridge contract changed"
        )
    for key, expected in {
        "embedding_identity_used": False,
        "python_builtin_hash_used": False,
        "parent_metadata_in_graph_identity": False,
    }.items():
        if type(bridge.get(key)) is not bool or bridge[key] is not expected:
            raise TasteComRecGCSmokeError(
                f"Taste COMRECGC identity boundary changed: {key}"
            )
    for key, minimum in {
        "call_count": 1,
        "evaluated_graph_count": 1,
        "calculate_hash_count": 1,
        "unique_graph_count": 1,
        "evaluated_strict_graph_count": 1,
        "unique_lineage_count": 1,
        "lineage_occurrence_count": 1,
        "canonical_row_reuse_count": 0,
    }.items():
        _native_int(bridge.get(key), field=f"bridge.{key}", minimum=minimum)
    if (
        bridge["calculate_hash_count"] != bridge["evaluated_graph_count"]
        or bridge["call_count"] > bridge["evaluated_graph_count"]
        or bridge["unique_graph_count"] > bridge["evaluated_graph_count"]
        or bridge["canonical_row_reuse_count"]
        != bridge["evaluated_graph_count"] - bridge["unique_graph_count"]
        or bridge["evaluated_strict_graph_count"] > bridge["unique_graph_count"]
        or bridge["unique_lineage_count"] < bridge["unique_graph_count"]
        or bridge["unique_lineage_count"] > bridge["lineage_occurrence_count"]
        or bridge["lineage_occurrence_count"]
        != bridge["evaluated_graph_count"]
    ):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC bridge counters are inconsistent"
        )
    destinations = bridge.get("destination_prediction_counts")
    if type(destinations) is not dict or set(destinations) != {"0", "2"}:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC destination distribution changed"
        )
    for label in ("0", "2"):
        _native_int(
            destinations[label], field=f"destination {label}", minimum=0
        )
    if sum(destinations.values()) != bridge["evaluated_strict_graph_count"]:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC destination counts lost strict candidates"
        )

    common = value.get("common_recourse")
    expected_common_keys = {
        "schema_version",
        "distance_embedding",
        "theta",
        "delta",
        "cluster_size",
        "recourse_size",
        "retained_strict_candidate_count",
        "theta_eligible_pair_count",
        "dbscan_invoked",
        "dbscan_cluster_count",
        "dbscan_noise_count",
        "official_coverage_summary_invoked",
        "official_coverage_summary_sha256",
        "official_greedy_summary_invoked",
        "selected_common_recourse_count",
        "selected_common_recourses",
        "graph_payload_persisted",
        "molecule_payload_persisted",
    }
    if (
        type(common) is not dict
        or set(common) != expected_common_keys
        or common.get("schema_version")
        != "tastemolnet_comrecgc_common_recourse_smoke_v1"
        or common.get("distance_embedding") != "frozen_gine_graph_hidden"
        or type(common.get("theta")) is not float
        or common["theta"] != frozen.theta
        or type(common.get("delta")) is not float
        or common["delta"] != frozen.delta
        or type(common.get("cluster_size")) is not int
        or common["cluster_size"] != frozen.cluster_size
        or type(common.get("recourse_size")) is not int
        or common["recourse_size"] != frozen.recourse_size
        or common.get("dbscan_invoked") is not True
        or common.get("official_coverage_summary_invoked") is not True
        or common.get("official_greedy_summary_invoked") is not True
        or common.get("graph_payload_persisted") is not False
        or common.get("molecule_payload_persisted") is not False
    ):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC common-recourse contract changed"
        )
    _lower_sha256(
        common.get("official_coverage_summary_sha256"),
        field="official coverage summary SHA",
    )
    for key, minimum in {
        "retained_strict_candidate_count": 1,
        "theta_eligible_pair_count": 1,
        "dbscan_cluster_count": 1,
        "dbscan_noise_count": 0,
        "selected_common_recourse_count": 1,
    }.items():
        _native_int(common.get(key), field=f"common.{key}", minimum=minimum)
    if (
        common["retained_strict_candidate_count"]
        > bridge["evaluated_strict_graph_count"]
    ):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC retained candidates exceed evaluated strict graphs"
        )
    rows = common.get("selected_common_recourses")
    if (
        type(rows) is not list
        or len(rows) != common["selected_common_recourse_count"]
        or len(rows) > frozen.recourse_size
        or common["dbscan_cluster_count"] < len(rows)
    ):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC selected common recourse set changed"
        )
    expected_row_keys = {
        "rank",
        "cluster_id",
        "representative_graph_identity_sha256",
        "destination_label",
        "score",
        "frequency",
        "covered_parent_count",
        "cluster_size",
        "lineage_count",
    }
    selected_cluster_ids: list[int] = []
    for expected_rank, row in enumerate(rows, start=1):
        if type(row) is not dict or set(row) != expected_row_keys:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC common-recourse row schema changed"
            )
        if (
            type(row.get("rank")) is not int
            or row["rank"] != expected_rank
            or type(row.get("cluster_id")) is not int
            or row["cluster_id"] < 0
            or type(row.get("destination_label")) is not int
            or row["destination_label"] not in (0, 2)
            or type(row.get("frequency")) is not int
            or row["frequency"] < 1
            or type(row.get("covered_parent_count")) is not int
            or not 1 <= row["covered_parent_count"] <= frozen.source_count
            or type(row.get("cluster_size")) is not int
            or row["cluster_size"] < 1
            or type(row.get("lineage_count")) is not int
            or row["lineage_count"] < 1
        ):
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC common-recourse row values changed"
            )
        if row["cluster_id"] >= common["dbscan_cluster_count"]:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC selected cluster ID is outside DBSCAN"
            )
        selected_cluster_ids.append(row["cluster_id"])
        _lower_sha256(
            row.get("representative_graph_identity_sha256"),
            field="representative graph identity",
        )
        score = row.get("score")
        if type(score) is not float or not math.isfinite(score) or not 0 <= score <= 1:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC common-recourse score changed"
            )
        if destinations[str(row["destination_label"])] <= 0:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC representative destination was not observed"
            )
    if len(set(selected_cluster_ids)) != len(selected_cluster_ids):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC official greedy selected one cluster twice"
        )
    return dict(value)


def _validate_t2_adoption_binding(value: Any) -> dict[str, Any]:
    from src.utils.tastemolnet_gine_pass_adoption_v1 import (
        ADOPTION_MARKER,
        CHECKPOINT_FILES,
        DOWNSTREAM_BINDING_KEYS,
        DOWNSTREAM_BINDING_SCHEMA,
        FILE_IDENTITY_FIELDS,
        SOURCE_CID,
        SOURCE_RUN_ID,
    )

    if type(value) is not dict or set(value) != set(DOWNSTREAM_BINDING_KEYS):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC T2 adoption binding keys changed"
        )
    if (
        value.get("schema_version") != DOWNSTREAM_BINDING_SCHEMA
        or value.get("stage") != "T2_GINE_FULL"
        or value.get("status") != "PASS"
        or value.get("state") != ADOPTION_MARKER
        or value.get("source_cid") != SOURCE_CID
        or value.get("source_run_id") != SOURCE_RUN_ID
    ):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC T2 adoption binding is not PASS"
        )
    for key in (
        "adoption_root_inventory_sha256",
        "gate_sha256",
        "receipt_sha256",
        "source_evidence_sha256",
        "formal_bundle_inventory_sha256",
        "formal_bundle_model_sha256",
        "formal_bundle_sha256s_sha256",
    ):
        _lower_sha256(value.get(key), field=f"t2_adoption.{key}")
    for key in (
        "adoption_root",
        "gate_path",
        "receipt_path",
        "formal_bundle_root",
    ):
        raw = value.get(key)
        if (
            type(raw) is not str
            or not Path(raw).is_absolute()
            or Path(os.path.abspath(raw)) != Path(raw)
            or "/proc/self/fd/" in raw
        ):
            raise TasteComRecGCSmokeError(
                f"Taste COMRECGC T2 adoption {key} changed"
            )
    if Path(value["gate_path"]) != Path(value["adoption_root"]) / "gate.json":
        raise TasteComRecGCSmokeError("Taste COMRECGC T2 gate path changed")
    if Path(value["receipt_path"]) != Path(value["adoption_root"]) / "manifest.json":
        raise TasteComRecGCSmokeError("Taste COMRECGC T2 receipt path changed")
    inventory = value.get("formal_bundle_inventory")
    if type(inventory) is not list or len(inventory) != len(CHECKPOINT_FILES):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC T2 formal inventory is absent"
        )
    if _canonical_sha256(inventory) != value["formal_bundle_inventory_sha256"]:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC T2 formal inventory digest changed"
        )
    files: dict[str, Mapping[str, Any]] = {}
    for row in inventory:
        if (
            type(row) is not dict
            or set(row) != {"path", "kind", "identity", "sha256"}
            or row.get("kind") != "file"
            or type(row.get("path")) is not str
            or row["path"] not in CHECKPOINT_FILES
            or row["path"] in files
            or type(row.get("identity")) is not dict
            or set(row["identity"]) != FILE_IDENTITY_FIELDS
        ):
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC T2 formal inventory row changed"
            )
        for field, observed in row["identity"].items():
            if type(observed) is not int or observed < 0:
                raise TasteComRecGCSmokeError(
                    f"Taste COMRECGC T2 file identity {field} changed"
                )
        if (
            row["identity"]["nlink"] != 1
            or row["identity"]["mode"] & 0o170000 != 0o100000
            or row["identity"]["mode"] & 0o077 != 0
        ):
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC T2 formal file is not private single-link"
            )
        _lower_sha256(row.get("sha256"), field=f"T2 formal {row['path']} SHA")
        files[row["path"]] = row
    if set(files) != set(CHECKPOINT_FILES):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC T2 formal inventory file set changed"
        )
    if (
        value["formal_bundle_model_sha256"] != files["model.pt"]["sha256"]
        or value["formal_bundle_sha256s_sha256"]
        != files["sha256sums.txt"]["sha256"]
    ):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC T2 formal bundle pins changed"
        )
    return json.loads(json.dumps(value))


def _validate_stage_evidence(
    value: Any,
    *,
    expected_stage: str,
    t2_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != HELD_STAGE_EVIDENCE_KEYS:
        raise TasteComRecGCSmokeError(
            f"Taste COMRECGC {expected_stage} evidence keys changed"
        )
    if value.get("stage") != expected_stage:
        raise TasteComRecGCSmokeError(
            f"Taste COMRECGC expected {expected_stage} evidence"
        )
    for key in HELD_STAGE_EVIDENCE_KEYS - {"stage", "checkpoint_dir"}:
        _lower_sha256(value.get(key), field=f"{expected_stage}.{key}")
    checkpoint_dir = value.get("checkpoint_dir")
    if (
        type(checkpoint_dir) is not str
        or not Path(checkpoint_dir).is_absolute()
        or Path(os.path.abspath(checkpoint_dir)) != Path(checkpoint_dir)
        or "/proc/self/fd/" in checkpoint_dir
    ):
        raise TasteComRecGCSmokeError(
            f"Taste COMRECGC {expected_stage} checkpoint path changed"
        )
    if (
        value["t2_adoption_gate_sha256"] != t2_binding["gate_sha256"]
        or value["t2_adoption_receipt_sha256"]
        != t2_binding["receipt_sha256"]
        or value["t2_adoption_binding_sha256"]
        != _canonical_sha256(t2_binding)
    ):
        raise TasteComRecGCSmokeError(
            f"Taste COMRECGC {expected_stage} lost the fresh T2 binding"
        )
    return json.loads(json.dumps(value))


def validate_terminal_input_authority(value: Any) -> dict[str, Any]:
    """Validate the durable predecessor hashes copied into the T9 output."""

    expected_keys = {
        "schema_version",
        "managed_active_receipt_sha256",
        "t2_adoption_binding",
        "t2_adoption_binding_sha256",
        "t3_stage_evidence",
        "t4_stage_evidence",
        "train_csv_sha256",
        "feature_schema_sha256",
        "temperature_scaling_sha256",
        "official_commit",
        "official_file_sha256",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC terminal input authority keys changed"
        )
    if value.get("schema_version") != "tastemolnet_t9_input_authority_v1":
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC terminal input authority schema changed"
        )
    for key in (
        "managed_active_receipt_sha256",
        "t2_adoption_binding_sha256",
        "train_csv_sha256",
        "feature_schema_sha256",
        "temperature_scaling_sha256",
    ):
        _lower_sha256(value.get(key), field=f"input_authority.{key}")
    official_commit = value.get("official_commit")
    if official_commit != OFFICIAL_COMRECGC_COMMIT:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC official commit changed"
        )
    official = value.get("official_file_sha256")
    if type(official) is not dict or set(official) != {
        "comrecgc.py",
        "common_recourse.py",
        "data.py",
        "distance.py",
        "gnn.py",
        "neurosed/models.py",
        "util.py",
    }:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC official file authority changed"
        )
    from src.baselines.comrecgc.held_upstream import OFFICIAL_SOURCE_SHA256

    for name, digest in official.items():
        _lower_sha256(digest, field=f"official_file_sha256.{name}")
    if official != dict(OFFICIAL_SOURCE_SHA256):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC official source bytes are not commit 122f9341"
        )
    t2_binding = _validate_t2_adoption_binding(value.get("t2_adoption_binding"))
    if value["t2_adoption_binding_sha256"] != _canonical_sha256(t2_binding):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC T2 adoption binding digest changed"
        )
    t3 = _validate_stage_evidence(
        value.get("t3_stage_evidence"),
        expected_stage="T3_GINE_CALIBRATED",
        t2_binding=t2_binding,
    )
    t4 = _validate_stage_evidence(
        value.get("t4_stage_evidence"),
        expected_stage="T4_ORACLE_SMOKE",
        t2_binding=t2_binding,
    )
    for key in (
        "checkpoint_dir",
        "checkpoint_id",
        "checkpoint_inventory_sha256",
        "checkpoint_stat_inventory_sha256",
        "checkpoint_sha256s_sha256",
        "t2_adoption_binding_sha256",
    ):
        if t3[key] != t4[key]:
            raise TasteComRecGCSmokeError(
                f"Taste COMRECGC T3/T4 {key} changed"
            )
    if (
        t3["checkpoint_dir"] != t2_binding["formal_bundle_root"]
        or t3["checkpoint_id"] != t2_binding["formal_bundle_model_sha256"]
        or t3["checkpoint_inventory_sha256"]
        != t2_binding["formal_bundle_inventory_sha256"]
        or t3["checkpoint_sha256s_sha256"]
        != t2_binding["formal_bundle_sha256s_sha256"]
    ):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC checkpoint differs from the fresh T2 formal bundle"
        )
    formal_files = {
        row["path"]: row for row in t2_binding["formal_bundle_inventory"]
    }
    if (
        value["feature_schema_sha256"]
        != formal_files["feature_schema.json"]["sha256"]
        or value["temperature_scaling_sha256"]
        != formal_files["temperature_scaling.json"]["sha256"]
    ):
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC calibrated feature/temperature identity changed"
        )
    return json.loads(json.dumps(value))


def build_terminal_documents(
    *,
    science: Mapping[str, Any],
    input_authority: Mapping[str, Any],
    task_id: str,
    run_id: str,
    gpu_uuid: str,
) -> dict[str, bytes]:
    """Build the aggregate-only, strict-consumer T9 terminal documents."""

    smoke = validate_native_comrecgc_smoke_result(dict(science))
    inputs = validate_terminal_input_authority(dict(input_authority))
    if task_id != TASK_ID:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC task_id changed from the managed route"
        )
    for field, value in {"task_id": task_id, "run_id": run_id}.items():
        if type(value) is not str or _SAFE_ID_RE.fullmatch(value) is None:
            raise TasteComRecGCSmokeError(
                f"Taste COMRECGC {field} is not one bounded identifier"
            )
    if type(gpu_uuid) is not str or _GPU_UUID_RE.fullmatch(gpu_uuid) is None:
        raise TasteComRecGCSmokeError("Taste COMRECGC GPU UUID is malformed")
    input_bytes = _json_bytes(inputs)
    smoke_bytes = _json_bytes(smoke)
    state = {
        "schema_version": "tastemolnet_t9_comrecgc_state_v1",
        "status": "PASS",
        "stage": STAGE,
        "marker": PASS_MARKER,
        "task_id": task_id,
        "run_id": run_id,
        "gpu_index": 2,
        "gpu_uuid": gpu_uuid,
        "num_classes": NUM_CLASSES,
        "source_label": SOURCE_LABEL,
    }
    state_bytes = _json_bytes(state)
    manifest = {
        "schema_version": "tastemolnet_t9_comrecgc_manifest_v1",
        "status": "PASS",
        "stage": STAGE,
        "dataset": DATASET,
        "method": METHOD,
        "num_classes": NUM_CLASSES,
        "source_label": SOURCE_LABEL,
        "destination_labels": [0, 2],
        "candidate_condition": "predicted_label != source_label",
        "importance": "1.0 - probabilities[:, source_label]",
        "input_hashes_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "comrecgc_smoke_sha256": hashlib.sha256(smoke_bytes).hexdigest(),
        "random_walk_steps": smoke["random_walk_steps"],
        "full_required_steps": smoke["full_required_steps"],
        "strict_counterfactual_count": smoke["bridge"][
            "evaluated_strict_graph_count"
        ],
        "destination_prediction_counts": smoke["bridge"][
            "destination_prediction_counts"
        ],
        "selected_common_recourse_count": smoke["common_recourse"][
            "selected_common_recourse_count"
        ],
        "checkpoint_complete_state_sha256": smoke["checkpoint_reload"][
            "complete_state_sha256"
        ],
        "data_redistribution_allowed": False,
        "rf_oracle_used": False,
        "test_loaded": False,
        "smoke_only": True,
        "paper_result_eligible": False,
    }
    manifest_bytes = _json_bytes(manifest)
    gate = {
        "schema_version": "tastemolnet_t9_comrecgc_gate_v1",
        "status": "PASS",
        "stage": STAGE,
        "marker": PASS_MARKER,
        "marker_file": PASS_FILE,
        "input_hashes_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "comrecgc_smoke_sha256": hashlib.sha256(smoke_bytes).hexdigest(),
        "strict_consumer_required": True,
    }
    return {
        "input_hashes.json": input_bytes,
        "state.json": state_bytes,
        "manifest.json": manifest_bytes,
        "comrecgc_smoke.json": smoke_bytes,
        "gate.json": _json_bytes(gate),
    }


def _canonical_json_document(data: bytes, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteComRecGCSmokeError(f"{field} is malformed") from exc
    if type(value) is not dict or data != _json_bytes(value):
        raise TasteComRecGCSmokeError(f"{field} is not canonical JSON")
    return value


@dataclass(slots=True)
class HeldTasteComRecGCOutput:
    """Descriptor-held strict consumer for one aggregate-only T9 output."""

    held: Any
    documents: Mapping[str, Mapping[str, Any]]
    evidence: Mapping[str, Any]

    @property
    def path(self) -> Path:
        return self.held.path

    def _validate(self) -> dict[str, Any]:
        inventory = self.held.revalidate()
        current = {
            name: _canonical_json_document(
                self.held.read_bytes(name), field=name
            )
            for name in OUTPUT_PAYLOAD_FILES
        }
        if current != dict(self.documents):
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC terminal documents changed"
            )
        inputs = validate_terminal_input_authority(current["input_hashes.json"])
        smoke = validate_native_comrecgc_smoke_result(
            current["comrecgc_smoke.json"]
        )
        input_sha = hashlib.sha256(
            _json_bytes(current["input_hashes.json"])
        ).hexdigest()
        state_sha = hashlib.sha256(
            _json_bytes(current["state.json"])
        ).hexdigest()
        manifest_sha = hashlib.sha256(
            _json_bytes(current["manifest.json"])
        ).hexdigest()
        smoke_sha = hashlib.sha256(
            _json_bytes(current["comrecgc_smoke.json"])
        ).hexdigest()
        state = current["state.json"]
        manifest = current["manifest.json"]
        gate = current["gate.json"]
        if (
            set(state)
            != {
                "schema_version",
                "status",
                "stage",
                "marker",
                "task_id",
                "run_id",
                "gpu_index",
                "gpu_uuid",
                "num_classes",
                "source_label",
            }
            or state.get("schema_version")
            != "tastemolnet_t9_comrecgc_state_v1"
            or state.get("status") != "PASS"
            or state.get("stage") != STAGE
            or state.get("marker") != PASS_MARKER
            or state.get("task_id") != TASK_ID
            or type(state.get("run_id")) is not str
            or _SAFE_ID_RE.fullmatch(state["run_id"]) is None
            or type(state.get("gpu_index")) is not int
            or state["gpu_index"] != 2
            or type(state.get("gpu_uuid")) is not str
            or _GPU_UUID_RE.fullmatch(state["gpu_uuid"]) is None
            or type(state.get("num_classes")) is not int
            or state["num_classes"] != NUM_CLASSES
            or type(state.get("source_label")) is not int
            or state["source_label"] != SOURCE_LABEL
        ):
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC terminal state changed"
            )
        expected_manifest = {
            "schema_version": "tastemolnet_t9_comrecgc_manifest_v1",
            "status": "PASS",
            "stage": STAGE,
            "dataset": DATASET,
            "method": METHOD,
            "num_classes": NUM_CLASSES,
            "source_label": SOURCE_LABEL,
            "destination_labels": [0, 2],
            "candidate_condition": "predicted_label != source_label",
            "importance": "1.0 - probabilities[:, source_label]",
            "input_hashes_sha256": input_sha,
            "comrecgc_smoke_sha256": smoke_sha,
            "random_walk_steps": smoke["random_walk_steps"],
            "full_required_steps": smoke["full_required_steps"],
            "strict_counterfactual_count": smoke["bridge"][
                "evaluated_strict_graph_count"
            ],
            "destination_prediction_counts": smoke["bridge"][
                "destination_prediction_counts"
            ],
            "selected_common_recourse_count": smoke["common_recourse"][
                "selected_common_recourse_count"
            ],
            "checkpoint_complete_state_sha256": smoke["checkpoint_reload"][
                "complete_state_sha256"
            ],
            "data_redistribution_allowed": False,
            "rf_oracle_used": False,
            "test_loaded": False,
            "smoke_only": True,
            "paper_result_eligible": False,
        }
        if manifest != expected_manifest:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC terminal manifest cannot be derived"
            )
        expected_gate = {
            "schema_version": "tastemolnet_t9_comrecgc_gate_v1",
            "status": "PASS",
            "stage": STAGE,
            "marker": PASS_MARKER,
            "marker_file": PASS_FILE,
            "input_hashes_sha256": input_sha,
            "state_sha256": state_sha,
            "manifest_sha256": manifest_sha,
            "comrecgc_smoke_sha256": smoke_sha,
            "strict_consumer_required": True,
        }
        if gate != expected_gate:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC terminal gate cannot be derived"
            )
        serialized = b"".join(
            self.held.read_bytes(name) for name in sorted(OUTPUT_PAYLOAD_FILES)
        ).lower()
        for forbidden in (
            b'"smiles"',
            b'"molecule_id"',
            b"randomforestclassifier",
            b"rf_model.pkl",
            b"/proc/self/fd/",
        ):
            if forbidden in serialized:
                raise TasteComRecGCSmokeError(
                    "Taste COMRECGC output contains forbidden payload"
                )
        self.held.revalidate()
        evidence = {
            "schema_version": "tastemolnet_t9_comrecgc_output_evidence_v1",
            "status": "PASS",
            "stage": STAGE,
            "output_root": str(self.path),
            "gate_sha256": hashlib.sha256(
                _json_bytes(gate)
            ).hexdigest(),
            "root_inventory_sha256": inventory["inventory_sha256"],
            "output_inventory_sha256": inventory["inventory_sha256"],
            "task_id": state["task_id"],
            "run_id": state["run_id"],
            "gpu_index": state["gpu_index"],
            "gpu_uuid": state["gpu_uuid"],
            "managed_active_receipt_sha256": inputs[
                "managed_active_receipt_sha256"
            ],
            "t2_adoption_binding_sha256": inputs[
                "t2_adoption_binding_sha256"
            ],
            "t2_adoption_gate_sha256": inputs["t2_adoption_binding"][
                "gate_sha256"
            ],
            "t2_adoption_receipt_sha256": inputs[
                "t2_adoption_binding"
            ]["receipt_sha256"],
            "t2_source_evidence_sha256": inputs["t2_adoption_binding"][
                "source_evidence_sha256"
            ],
            "t3_gate_sha256": inputs["t3_stage_evidence"][
                "gate_sha256"
            ],
            "t3_root_inventory_sha256": inputs["t3_stage_evidence"][
                "root_inventory_sha256"
            ],
            "t4_gate_sha256": inputs["t4_stage_evidence"][
                "gate_sha256"
            ],
            "t4_root_inventory_sha256": inputs["t4_stage_evidence"][
                "root_inventory_sha256"
            ],
            "checkpoint_id": inputs["t3_stage_evidence"]["checkpoint_id"],
            "checkpoint_inventory_sha256": inputs["t3_stage_evidence"][
                "checkpoint_inventory_sha256"
            ],
            "checkpoint_stat_inventory_sha256": inputs[
                "t3_stage_evidence"
            ]["checkpoint_stat_inventory_sha256"],
            "checkpoint_sha256s_sha256": inputs["t3_stage_evidence"][
                "checkpoint_sha256s_sha256"
            ],
            "train_csv_sha256": inputs["train_csv_sha256"],
            "feature_schema_sha256": inputs["feature_schema_sha256"],
            "temperature_scaling_sha256": inputs[
                "temperature_scaling_sha256"
            ],
            "official_commit": inputs["official_commit"],
            "official_file_inventory_sha256": _canonical_sha256(
                inputs["official_file_sha256"]
            ),
            "strict_counterfactual_count": smoke["bridge"][
                "evaluated_strict_graph_count"
            ],
            "destination_prediction_counts": smoke["bridge"][
                "destination_prediction_counts"
            ],
        }
        return evidence

    def revalidate(self) -> dict[str, Any]:
        observed = self._validate()
        if observed != dict(self.evidence):
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC terminal evidence changed"
            )
        return observed

    def close(self) -> None:
        self.held.close()

    def __enter__(self) -> "HeldTasteComRecGCOutput":
        self.revalidate()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def hold_tastemolnet_comrecgc_output(
    path: str | Path,
) -> HeldTasteComRecGCOutput:
    from src.utils.retained_output_directory import HeldPublishedTerminalOutput

    held = HeldPublishedTerminalOutput.open(
        path, marker_name=PASS_FILE, marker_payload=PASS_BYTES
    )
    try:
        inventory = held.revalidate()
        if (
            set(inventory["files"]) != OUTPUT_PAYLOAD_FILES
            or inventory["directories"]
        ):
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC terminal output file set changed"
            )
        documents = {
            name: _canonical_json_document(held.read_bytes(name), field=name)
            for name in OUTPUT_PAYLOAD_FILES
        }
        provisional = HeldTasteComRecGCOutput(
            held=held,
            documents=documents,
            evidence={},
        )
        evidence = provisional._validate()
        result = HeldTasteComRecGCOutput(
            held=held,
            documents=documents,
            evidence=evidence,
        )
        result.revalidate()
        return result
    except Exception:
        held.close()
        raise


def validate_tastemolnet_comrecgc_output(
    path: str | Path,
) -> dict[str, Any]:
    with hold_tastemolnet_comrecgc_output(path) as held:
        return held.revalidate()


def execute_native_comrecgc_smoke(
    *,
    modules: Mapping[str, Any],
    checkpoint_payloads: Mapping[str, bytes],
    source_rows: Sequence[Any],
    graph_schema: Any,
    device: str,
    parameters: TasteComRecGCSmokeParameters | None = None,
) -> dict[str, Any]:
    """Execute native serial COMRECGC M=500 with a real midpoint reload."""

    import torch

    from src.baselines.comrecgc.generation_loop import run_generation_loop
    from src.baselines.comrecgc.runtime import (
        lineage_neighbor_wrapper,
        reset_official_state,
    )
    from src.baselines.gcfexplainer_mutagenicity_adapter import (
        GraphRecordDataset,
        graph_lineage_neighbor_wrapper,
    )

    frozen = (parameters or TasteComRecGCSmokeParameters()).validate()
    if set(modules) < {"comrecgc", "common_recourse", "util"}:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC pinned upstream module closure is incomplete"
        )
    module = modules["comrecgc"]
    source_graphs, _records, adapter, source_evidence = _initialize_source_graphs(
        checkpoint_payloads=checkpoint_payloads,
        source_rows=source_rows,
        graph_schema=graph_schema,
        device=device,
        parameters=frozen,
    )
    dataset = GraphRecordDataset(
        source_graphs, num_features=len(graph_schema.feature_atomic_numbers)
    )
    reset_official_state(
        module,
        candidate_capacity=frozen.candidate_capacity,
        sample_size=frozen.sample_size,
    )
    module.input_graphs_covered = torch.zeros(
        frozen.source_count, dtype=torch.float32
    )
    bridge = TasteComRecGCMulticlassBridge(
        adapter=adapter,
        feature_atomic_numbers=graph_schema.feature_atomic_numbers,
    )

    def combined_lineage(original: Any) -> Any:
        return lineage_neighbor_wrapper(graph_lineage_neighbor_wrapper(original))

    _seed_all(frozen.seed)
    importance_args = {
        "schema_version": "tastemolnet_comrecgc_gine_distance_v1",
        "classifier": "frozen_calibrated_three_class_gine",
        "distance_embedding": "frozen_gine_graph_hidden",
        "num_classes": NUM_CLASSES,
        "source_label": SOURCE_LABEL,
    }
    with bridge.installed(module, neighbor_wrapper=combined_lineage):
        first = run_generation_loop(
            module,
            input_graphs=dataset,
            importance_args=importance_args,
            teleport_probability=frozen.teleport_probability,
            max_steps=frozen.checkpoint_step,
            heads=frozen.heads,
        )
        with tempfile.TemporaryDirectory(
            prefix="taste-t9-comrecgc-checkpoint-"
        ) as temporary:
            checkpoint_path = Path(temporary) / "checkpoint-000250.pt"
            loaded = _write_reload_checkpoint(
                module=module,
                bridge=bridge,
                loop_state=first,
                parameters=frozen,
                path=checkpoint_path,
            )
            reset_official_state(
                module,
                candidate_capacity=frozen.candidate_capacity,
                sample_size=frozen.sample_size,
            )
            resumed = _restore_reload_checkpoint(
                module=module, bridge=bridge, loaded=loaded
            )
            final = run_generation_loop(
                module,
                input_graphs=dataset,
                importance_args=importance_args,
                teleport_probability=frozen.teleport_probability,
                max_steps=frozen.steps,
                heads=frozen.heads,
                initial_state=resumed,
            )
            checkpoint_evidence = dict(loaded["evidence"])
        if final.completed_step != frozen.steps:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC walk did not reach M=500"
            )
        if len(module.traversed_hashes) != frozen.steps:
            raise TasteComRecGCSmokeError(
                "Taste COMRECGC traversed-hash count differs from M=500"
            )
        common = _common_recourse_summary(
            modules=modules,
            module=module,
            bridge=bridge,
            source_graphs=source_graphs,
            adapter=adapter,
            parameters=frozen,
        )
        bridge_evidence = bridge.report()
    if bridge_evidence["evaluated_strict_graph_count"] <= 0:
        raise TasteComRecGCSmokeError(
            "Taste COMRECGC smoke has no strict multiclass candidate"
        )
    result = {
        "schema_version": "tastemolnet_comrecgc_native_smoke_v1",
        "stage": STAGE,
        "dataset": DATASET,
        "method": METHOD,
        "parameters": asdict(frozen),
        "source_cohort": source_evidence,
        "checkpoint_reload": checkpoint_evidence,
        "bridge": bridge_evidence,
        "common_recourse": common,
        "official_native_random_walk": True,
        "official_stateful_heads_preserved": True,
        "official_rng_and_collector_serial": True,
        "random_walk_steps": frozen.steps,
        "smoke_budget": True,
        "full_budget": False,
        "full_required_steps": 50_000,
        "same_frozen_three_class_gine": True,
        "second_classifier_used": False,
        "rf_oracle_used": False,
        "validation_loaded": False,
        "calibration_payload_loaded": False,
        "test_loaded": False,
        "dataset_redistributed": False,
        "paper_result_eligible": False,
    }
    validate_native_comrecgc_smoke_result(result)
    return result


def run_tastemolnet_comrecgc_smoke(
    *,
    config_path: str | Path,
    stage_root: str | Path,
    output_dir: str | Path,
    run_id: str,
    gpu_uuid: str,
    t2_adoption_root: str | Path,
    t2_adoption_gate_sha256: str,
    t2_adoption_receipt_sha256: str,
    t2_source_evidence_sha256: str,
    t3_output_root: str | Path,
    t4_output_root: str | Path,
    checkpoint_dir: str | Path,
    train_csv: str | Path,
    official_root: str | Path,
) -> dict[str, Any]:
    """Run the managed-v2 worker; terminal PASS remains verifier-only."""

    from src.utils.tastemolnet_t9_managed_v2 import (
        hold_t9_inputs,
        run_t9_worker,
    )

    with hold_t9_inputs(
        config_path=config_path,
        run_id=run_id,
        gpu_uuid=gpu_uuid,
        t2_adoption_root=t2_adoption_root,
        t2_adoption_gate_sha256=t2_adoption_gate_sha256,
        t2_adoption_receipt_sha256=t2_adoption_receipt_sha256,
        t2_source_evidence_sha256=t2_source_evidence_sha256,
        t3_output_root=t3_output_root,
        t4_output_root=t4_output_root,
        checkpoint_dir=checkpoint_dir,
        train_csv=train_csv,
        official_root=official_root,
    ) as inputs:
        return run_t9_worker(
            stage_root=stage_root,
            final_path=output_dir,
            inputs=inputs,
        )


__all__ = [
    "CanonicalNativeGraph",
    "DATASET",
    "METHOD",
    "NUM_CLASSES",
    "OFFICIAL_COMRECGC_COMMIT",
    "OUTPUT_PAYLOAD_FILES",
    "PASS_BYTES",
    "PASS_FILE",
    "PASS_MARKER",
    "SMOKE_STEPS",
    "SOURCE_LABEL",
    "STAGE",
    "TASK_ID",
    "TasteComRecGCMulticlassBridge",
    "TasteComRecGCRecord",
    "TasteComRecGCSmokeError",
    "TasteComRecGCSmokeParameters",
    "HeldTasteComRecGCOutput",
    "build_terminal_documents",
    "canonical_attributed_graph",
    "execute_native_comrecgc_smoke",
    "run_tastemolnet_comrecgc_smoke",
    "score_and_candidate",
    "hold_tastemolnet_comrecgc_output",
    "validate_tastemolnet_comrecgc_output",
    "validate_terminal_input_authority",
    "validate_native_comrecgc_smoke_result",
]
