"""TasteMolNet ComRecGC full generation with a train-only frozen cohort.

This module is intentionally dataset specific.  It extends the already
verified T9 native bridge to the preregistered T14 resource cap without
opening validation, calibration, or test payloads during generation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import stat
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence

from src.baselines.tastemolnet_comrecgc_smoke import (
    GINE_CANONICAL_REUSE_ATOL,
    GINE_CANONICAL_REUSE_RTOL,
    SOURCE_LABEL,
    TasteComRecGCMulticlassBridge,
    TasteComRecGCSmokeError,
    _common_recourse_summary,
    _embedding_sha256,
    _identity_graph_sha256,
    _scalar_native_int,
    _seed_all,
    canonical_attributed_graph,
)
from src.baselines.tastemolnet_gcf_smoke import (
    TasteFrozenGINENativeAdapter,
    encode_taste_source_graph,
    load_train_rows,
    taste_record_to_pyg,
)


STAGE = "T14_COMRECGC_FULL"
DATASET = "tastemolnet"
METHOD = "comrecgc"
COHORT_POLICY = "FULL_TRAIN_CORRECT_SOURCE"
VALID_UNIQUE_POLICY = "TRAIN_SIDE_STRICT_FLIP_CANONICAL"
M_MAX = 20_000
M_FALLBACK_MAX = 25_000
MIN_VALID_UNIQUE_RULES = 10
CHECK_INTERVAL = 2_500
GENERATION_PASS_MARKER = "[TASTE_T14_COMRECGC_FULL_GENERATION_PASS]"
RUNTIME_STATE_SCHEMA = "tastemolnet_t14_bounded_runtime_v1"
ROUTE_C_RUNTIME_STATE_SCHEMA = "tastemolnet_t14_route_c_runtime_v1"
CHECKPOINT_PROVENANCE_SCHEMA = "tastemolnet_t14_checkpoint_provenance_v1"
TRANSITION_EXPANDED_CAPACITY = 5
PROGRESS_INTERVAL = 100
COHORT_RECONCILIATION_RECEIPT = "cohort_reconciliation_receipt.json"
COHORT_RECONCILIATION_OBSERVATIONS = "cohort_reconciliation_observations"
COHORT_RECONCILIATION_SCHEMA = "tastemolnet_t14_cohort_reconciliation_v2"
COHORT_RECONCILIATION_OBSERVATION_SCHEMA = (
    "tastemolnet_t14_cohort_reconciliation_observation_v1"
)
COHORT_RECONCILIATION_BINDING_SCHEMA = (
    "tastemolnet_t14_cohort_reconciliation_binding_v1"
)


class TasteComRecGCFullError(TasteComRecGCSmokeError):
    """The fixed T14 scientific contract was violated."""


class TasteComRecGCFullBridge(TasteComRecGCMulticlassBridge):
    """Bind every scored candidate lineage to one frozen train parent."""

    def __init__(self, *, cohort_count: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if type(cohort_count) is not int or cohort_count <= 0:
            raise TasteComRecGCFullError("Taste T14 cohort count is invalid")
        self.cohort_count = cohort_count

    def call(
        self,
        graphs: Sequence[Any],
        importance_args: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        for graph in graphs:
            source_index = _scalar_native_int(
                getattr(graph, "comrecgc_source_index", None),
                field="graph.comrecgc_source_index",
                minimum=0,
            )
            if source_index >= self.cohort_count:
                raise TasteComRecGCFullError(
                    "Taste T14 candidate lineage escapes the train cohort"
                )
        return super().call(graphs, importance_args)

    def _stored_embedding_values(self, values: Any) -> Sequence[float]:
        """Keep T14 embeddings compact without changing their exact bytes."""

        import numpy as np

        array = np.asarray(values)
        if array.ndim != 1 or array.dtype.kind != "f" or not np.isfinite(array).all():
            raise TasteComRecGCFullError("Taste T14 compact embedding is malformed")
        # Scoring rows are views into a transient batch result; take one owned,
        # C-contiguous copy rather than expanding every scalar to a Python float.
        owned = np.array(array, copy=True, order="C")
        owned.setflags(write=False)
        return owned

    def _restored_embedding_values(self, values: Any) -> Sequence[float]:
        """Transfer an immutable mmap/checkpoint row without another copy."""

        import numpy as np

        array = np.asarray(values)
        if array.ndim != 1 or array.dtype.kind != "f" or not np.isfinite(array).all():
            raise TasteComRecGCFullError(
                "Taste T14 restored compact embedding is malformed"
            )
        restored = array if array.flags.c_contiguous else np.ascontiguousarray(array)
        restored.setflags(write=False)
        return restored

    def _accepted_checkpoint_schemas(self) -> frozenset[str]:
        return frozenset(
            {
                "tastemolnet_comrecgc_bridge_checkpoint_v3",
                "tastemolnet_comrecgc_bridge_checkpoint_v4",
            }
        )

    def checkpoint_state(self) -> dict[str, Any]:
        """Serialize T14 embedding arrays as tensor storages, not giant pickle data."""

        self._assert_idle()
        try:
            import numpy as np
            import torch
        except Exception as exc:  # pragma: no cover - AutoDL dependencies
            raise TasteComRecGCFullError(
                "Taste T14 compact checkpoints require NumPy and PyTorch"
            ) from exc
        records: dict[str, dict[str, Any]] = {}
        for key, value in sorted(self.records.items()):
            row = value.semantic_payload()
            embedding = np.asarray(
                value.embedding_values, dtype=np.dtype(value.embedding_dtype)
            )
            if (
                embedding.ndim != 1
                or not embedding.flags.c_contiguous
                or _embedding_sha256(embedding) != value.embedding_sha256
            ):
                raise TasteComRecGCFullError(
                    "Taste T14 checkpoint embedding bytes changed"
                )
            row["embedding_values"] = torch.from_numpy(embedding)
            records[key] = row
        return {
            "schema_version": "tastemolnet_comrecgc_bridge_checkpoint_v4",
            "records": records,
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
            "embedding_storage": "torch_cpu_tensor_zero_copy_v1",
        }


@dataclass(frozen=True, slots=True)
class TasteComRecGCFullParameters:
    """Parameters consumed by the shared native ComRecGC implementation."""

    steps: int = M_FALLBACK_MAX
    checkpoint_step: int = CHECK_INTERVAL
    source_pool: int = 0
    source_count: int = 0
    heads: int = 5
    candidate_capacity: int = 50_000
    sample_size: int = 10_000
    teleport_probability: float = 0.1
    theta: float = 0.1
    delta: float = 0.02
    cluster_size: int = 3
    recourse_size: int = 20
    seed: int = 7

    def validate(self) -> "TasteComRecGCFullParameters":
        if (
            self.steps != M_FALLBACK_MAX
            or self.checkpoint_step
            not in {250, 500, *range(CHECK_INTERVAL, M_FALLBACK_MAX + 1, CHECK_INTERVAL)}
            or self.source_pool <= 0
            or self.source_count != self.source_pool
            or self.heads != 5
            or self.candidate_capacity != 50_000
            or self.sample_size != 10_000
            or self.teleport_probability != 0.1
            or self.theta != 0.1
            or self.delta != 0.02
            or self.cluster_size != 3
            or self.recourse_size != 20
            or self.seed != 7
        ):
            raise TasteComRecGCFullError("Taste T14 full parameters drifted")
        return self


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise TasteComRecGCFullError("Taste T14 atomic write was short")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, _canonical_bytes(dict(value)) + b"\n")


def _cohort_lines(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(dict(row)) + b"\n" for row in rows)


def build_full_train_correct_source_cohort(
    *,
    true_sweet_rows: Sequence[Any],
    predictions: Sequence[int],
    source_probabilities: Sequence[float],
    canonical_graph_hashes: Sequence[str],
    train_csv_sha256: str,
    checkpoint_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], bytes]:
    """Build the exact user-authorized train-only T14 source cohort."""

    sizes = {
        len(true_sweet_rows),
        len(predictions),
        len(source_probabilities),
        len(canonical_graph_hashes),
    }
    if len(sizes) != 1 or not true_sweet_rows:
        raise TasteComRecGCFullError("Taste T14 cohort inputs are unaligned")
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source, prediction, probability, graph_hash in zip(
        true_sweet_rows,
        predictions,
        source_probabilities,
        canonical_graph_hashes,
        strict=True,
    ):
        molecule_id = str(getattr(source, "molecule_id", "")).strip()
        label = getattr(source, "label", None)
        if label != SOURCE_LABEL or not molecule_id:
            raise TasteComRecGCFullError("Taste T14 source row is not true Sweet")
        if type(prediction) is not int or prediction not in (0, 1, 2):
            raise TasteComRecGCFullError("Taste T14 prediction is malformed")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not 0.0 <= float(probability) <= 1.0
        ):
            raise TasteComRecGCFullError("Taste T14 source probability is malformed")
        if (
            type(graph_hash) is not str
            or len(graph_hash) != 64
            or any(character not in "0123456789abcdef" for character in graph_hash)
        ):
            raise TasteComRecGCFullError("Taste T14 canonical graph hash is malformed")
        if prediction != SOURCE_LABEL:
            continue
        if molecule_id in seen_ids:
            raise TasteComRecGCFullError("Taste T14 parent identity is not unique")
        seen_ids.add(molecule_id)
        selected.append(
            {
                "parent_id": molecule_id,
                "canonical_graph_hash": graph_hash,
                "true_label": SOURCE_LABEL,
                "predicted_label": SOURCE_LABEL,
                "source_probability": float(probability),
                "split": "train",
            }
        )
    selected.sort(
        key=lambda row: (str(row["parent_id"]), str(row["canonical_graph_hash"]))
    )
    if not selected:
        raise TasteComRecGCFullError("Taste T14 full cohort is empty")
    lines = _cohort_lines(selected)
    manifest = {
        "schema_version": "tastemolnet_t14_full_train_cohort_v1",
        "status": "PASS",
        "dataset": DATASET,
        "stage": STAGE,
        "policy": COHORT_POLICY,
        "selection": "true_label == 1 and frozen_T3_GINE_prediction == 1",
        "stable_sort": ["molecule_id", "canonical_graph_hash"],
        "split": "train",
        "source_label": SOURCE_LABEL,
        "cohort_count": len(selected),
        "train_csv_sha256": train_csv_sha256,
        "checkpoint_id": checkpoint_id,
        "cohort_jsonl_sha256": _sha256_bytes(lines),
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "result_conditioned_selection": False,
    }
    return selected, manifest, lines


def reconcile_t14_resume_cohort(
    *,
    frozen_cohort_bytes: bytes,
    frozen_manifest_bytes: bytes,
    replayed_rows: Sequence[Mapping[str, Any]],
    replayed_manifest: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    """Reopen one frozen cohort when only diagnostic probabilities drift.

    The cohort membership decision is the discrete frozen-GINE prediction.
    ``source_probability`` is retained only as post-selection diagnostics.
    Resume preserves the original bytes after proving every discrete and
    identity-bearing field still agrees and the numeric difference is within
    the same low-bit replay tolerance used by the stable GINE bridge.
    """

    try:
        frozen_rows = [
            json.loads(line)
            for line in frozen_cohort_bytes.decode("utf-8").splitlines()
        ]
        frozen_manifest = json.loads(frozen_manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteComRecGCFullError(
            "Taste T14 frozen cohort reconciliation input is unreadable"
        ) from exc
    if (
        not frozen_rows
        or any(type(row) is not dict for row in frozen_rows)
        or _cohort_lines(frozen_rows) != frozen_cohort_bytes
        or type(frozen_manifest) is not dict
        or _canonical_bytes(frozen_manifest) + b"\n" != frozen_manifest_bytes
        or type(replayed_manifest) is not dict
        or frozen_manifest.get("cohort_jsonl_sha256")
        != _sha256_bytes(frozen_cohort_bytes)
        or len(frozen_rows) != len(replayed_rows)
    ):
        raise TasteComRecGCFullError("Taste T14 frozen cohort closure changed")
    try:
        replayed_bytes = _cohort_lines(replayed_rows)
    except (TypeError, ValueError) as exc:
        raise TasteComRecGCFullError(
            "Taste T14 replayed cohort is not canonical"
        ) from exc

    frozen_without_payload = {
        key: value
        for key, value in frozen_manifest.items()
        if key != "cohort_jsonl_sha256"
    }
    replayed_without_payload = {
        key: value
        for key, value in replayed_manifest.items()
        if key != "cohort_jsonl_sha256"
    }
    if (
        set(frozen_manifest) != set(replayed_manifest)
        or _canonical_bytes(frozen_without_payload)
        != _canonical_bytes(replayed_without_payload)
        or replayed_manifest.get("cohort_jsonl_sha256")
        != _sha256_bytes(replayed_bytes)
    ):
        raise TasteComRecGCFullError("Taste T14 cohort changed on resume")

    expected_fields = {
        "parent_id",
        "canonical_graph_hash",
        "true_label",
        "predicted_label",
        "source_probability",
        "split",
    }
    mismatch_parent_ids: list[str] = []
    maximum_delta = 0.0
    frozen_probability_hex: list[str] = []
    replayed_probability_hex: list[str] = []
    for frozen, replayed in zip(frozen_rows, replayed_rows, strict=True):
        if set(frozen) != expected_fields or set(replayed) != expected_fields:
            raise TasteComRecGCFullError("Taste T14 frozen cohort row changed")
        frozen_identity = {
            field: frozen[field]
            for field in expected_fields - {"source_probability"}
        }
        replayed_identity = {
            field: replayed[field]
            for field in expected_fields - {"source_probability"}
        }
        if (
            type(frozen["parent_id"]) is not str
            or not frozen["parent_id"]
            or type(frozen["canonical_graph_hash"]) is not str
            or len(frozen["canonical_graph_hash"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in frozen["canonical_graph_hash"]
            )
            or type(frozen["true_label"]) is not int
            or frozen["true_label"] != SOURCE_LABEL
            or type(frozen["predicted_label"]) is not int
            or frozen["predicted_label"] != SOURCE_LABEL
            or frozen["split"] != "train"
            or type(frozen["split"]) is not str
            or _canonical_bytes(frozen_identity)
            != _canonical_bytes(replayed_identity)
        ):
            raise TasteComRecGCFullError("Taste T14 cohort changed on resume")
        frozen_probability = frozen["source_probability"]
        replayed_probability = replayed["source_probability"]
        if (
            isinstance(frozen_probability, bool)
            or not isinstance(frozen_probability, (int, float))
            or isinstance(replayed_probability, bool)
            or not isinstance(replayed_probability, (int, float))
        ):
            raise TasteComRecGCFullError(
                "Taste T14 source probability reconciliation is malformed"
            )
        frozen_value = float(frozen_probability)
        replayed_value = float(replayed_probability)
        if (
            not math.isfinite(frozen_value)
            or not math.isfinite(replayed_value)
            or not 0.0 <= frozen_value <= 1.0
            or not 0.0 <= replayed_value <= 1.0
        ):
            raise TasteComRecGCFullError(
                "Taste T14 source probability reconciliation is invalid"
            )
        delta = abs(frozen_value - replayed_value)
        tolerance = GINE_CANONICAL_REUSE_ATOL + (
            GINE_CANONICAL_REUSE_RTOL * abs(frozen_value)
        )
        if delta > tolerance:
            raise TasteComRecGCFullError(
                "Taste T14 source probability changed beyond low-bit replay"
            )
        maximum_delta = max(maximum_delta, delta)
        if frozen_value != replayed_value:
            mismatch_parent_ids.append(str(frozen["parent_id"]))
        frozen_probability_hex.append(frozen_value.hex())
        replayed_probability_hex.append(replayed_value.hex())

    primary_receipt = {
        "schema_version": COHORT_RECONCILIATION_SCHEMA,
        "status": "PASS",
        "policy": "SOURCE_PROBABILITY_LOW_BIT_ONLY",
        "frozen_cohort_sha256": _sha256_bytes(frozen_cohort_bytes),
        "frozen_manifest_sha256": _sha256_bytes(frozen_manifest_bytes),
        "cohort_count": len(frozen_rows),
        "relative_tolerance_hex": GINE_CANONICAL_REUSE_RTOL.hex(),
        "absolute_tolerance_hex": GINE_CANONICAL_REUSE_ATOL.hex(),
        "tolerance_formula": "delta <= atol + rtol * abs(frozen_reference)",
        "observation_schema": COHORT_RECONCILIATION_OBSERVATION_SCHEMA,
        "observation_directory": COHORT_RECONCILIATION_OBSERVATIONS,
        "required_identity_and_discrete_fields_exact": True,
        "frozen_cohort_preserved": True,
        "frozen_cohort_rewritten": False,
    }
    primary_sha256 = _sha256_bytes(
        _canonical_bytes(primary_receipt) + b"\n"
    )
    observation = {
        "schema_version": COHORT_RECONCILIATION_OBSERVATION_SCHEMA,
        "status": "PASS",
        "primary_receipt_sha256": primary_sha256,
        "frozen_cohort_sha256": primary_receipt["frozen_cohort_sha256"],
        "frozen_manifest_sha256": primary_receipt["frozen_manifest_sha256"],
        "current_replayed_cohort_sha256": _sha256_bytes(replayed_bytes),
        "current_replayed_manifest_sha256": _sha256_bytes(
            _canonical_bytes(dict(replayed_manifest)) + b"\n"
        ),
        "cohort_count": len(frozen_rows),
        "source_probability_mismatch_count": len(mismatch_parent_ids),
        "source_probability_max_abs_delta": maximum_delta,
        "source_probability_max_abs_delta_hex": maximum_delta.hex(),
        "frozen_source_probabilities_sha256": _sha256_bytes(
            _canonical_bytes(frozen_probability_hex)
        ),
        "current_source_probabilities_sha256": _sha256_bytes(
            _canonical_bytes(replayed_probability_hex)
        ),
        "mismatch_parent_ids_sha256": _sha256_bytes(
            _canonical_bytes(mismatch_parent_ids)
        ),
        "relative_tolerance_hex": GINE_CANONICAL_REUSE_RTOL.hex(),
        "absolute_tolerance_hex": GINE_CANONICAL_REUSE_ATOL.hex(),
        "tolerance_formula": "delta <= atol + rtol * abs(frozen_reference)",
        "identity_and_discrete_fields_exact": True,
        "cohort_order_exact": True,
        "manifest_except_cohort_sha_exact": True,
        "frozen_cohort_preserved": True,
        "frozen_cohort_rewritten": False,
    }
    return (
        [dict(row) for row in frozen_rows],
        dict(frozen_manifest),
        primary_receipt,
        observation,
    )


def _persist_cohort_reconciliation_evidence(
    root: Path,
    *,
    primary_receipt: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> None:
    primary_path = root / COHORT_RECONCILIATION_RECEIPT
    observation_root = root / COHORT_RECONCILIATION_OBSERVATIONS
    primary_exists = os.path.lexists(primary_path)
    observations_exist = os.path.lexists(observation_root)
    if primary_exists != observations_exist:
        raise TasteComRecGCFullError(
            "Taste T14 cohort reconciliation evidence is incomplete"
        )
    primary_payload = _canonical_bytes(dict(primary_receipt)) + b"\n"
    if primary_exists:
        if primary_path.is_symlink() or primary_path.read_bytes() != primary_payload:
            raise TasteComRecGCFullError(
                "Taste T14 cohort reconciliation receipt changed on resume"
            )
    else:
        _atomic_write(primary_path, primary_payload)

    if observations_exist:
        if (
            not observation_root.is_dir()
            or observation_root.is_symlink()
            or observation_root.resolve(strict=True) != observation_root
        ):
            raise TasteComRecGCFullError(
                "Taste T14 cohort reconciliation observations are aliased"
            )
    else:
        observation_root.mkdir(mode=0o700)
        _fsync_dir(root)
    replay_sha = observation.get("current_replayed_cohort_sha256")
    if (
        type(replay_sha) is not str
        or len(replay_sha) != 64
        or any(character not in "0123456789abcdef" for character in replay_sha)
    ):
        raise TasteComRecGCFullError(
            "Taste T14 reconciliation observation replay SHA is invalid"
        )
    observation_path = observation_root / f"{replay_sha}.json"
    observation_payload = _canonical_bytes(dict(observation)) + b"\n"
    if observation_path.exists():
        if observation_path.is_symlink() or observation_path.read_bytes() != (
            observation_payload
        ):
            raise TasteComRecGCFullError(
                "Taste T14 cohort reconciliation observation conflicts"
            )
        return
    _atomic_write(observation_path, observation_payload)


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_canonical_evidence_json(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise TasteComRecGCFullError(f"Taste T14 {label} is missing") from exc
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise TasteComRecGCFullError(
            f"Taste T14 {label} is not a physical regular file"
        )
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteComRecGCFullError(f"Taste T14 {label} is unreadable") from exc
    if type(value) is not dict or _canonical_bytes(value) + b"\n" != payload:
        raise TasteComRecGCFullError(f"Taste T14 {label} is not canonical")
    return value, payload


def _validate_cohort_reconciliation_evidence(
    root: Path,
    *,
    frozen_cohort_bytes: bytes,
    frozen_manifest_bytes: bytes,
) -> dict[str, Any] | None:
    """Reopen and bind optional immutable resume-reconciliation evidence."""

    primary_path = root / COHORT_RECONCILIATION_RECEIPT
    observation_root = root / COHORT_RECONCILIATION_OBSERVATIONS
    primary_exists = os.path.lexists(primary_path)
    observations_exist = os.path.lexists(observation_root)
    if not primary_exists and not observations_exist:
        return None
    if primary_exists != observations_exist:
        raise TasteComRecGCFullError(
            "Taste T14 cohort reconciliation evidence is incomplete"
        )
    primary, primary_payload = _read_canonical_evidence_json(
        primary_path,
        label="cohort reconciliation receipt",
    )
    try:
        frozen_rows = [
            json.loads(line)
            for line in frozen_cohort_bytes.decode("utf-8").splitlines()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteComRecGCFullError(
            "Taste T14 frozen reconciliation cohort is unreadable"
        ) from exc
    expected_primary_fields = {
        "schema_version",
        "status",
        "policy",
        "frozen_cohort_sha256",
        "frozen_manifest_sha256",
        "cohort_count",
        "relative_tolerance_hex",
        "absolute_tolerance_hex",
        "tolerance_formula",
        "observation_schema",
        "observation_directory",
        "required_identity_and_discrete_fields_exact",
        "frozen_cohort_preserved",
        "frozen_cohort_rewritten",
    }
    if (
        set(primary) != expected_primary_fields
        or primary.get("schema_version") != COHORT_RECONCILIATION_SCHEMA
        or primary.get("status") != "PASS"
        or primary.get("policy") != "SOURCE_PROBABILITY_LOW_BIT_ONLY"
        or primary.get("frozen_cohort_sha256")
        != _sha256_bytes(frozen_cohort_bytes)
        or primary.get("frozen_manifest_sha256")
        != _sha256_bytes(frozen_manifest_bytes)
        or type(primary.get("cohort_count")) is not int
        or primary["cohort_count"] <= 0
        or len(frozen_rows) != primary["cohort_count"]
        or any(type(row) is not dict for row in frozen_rows)
        or _cohort_lines(frozen_rows) != frozen_cohort_bytes
        or primary.get("relative_tolerance_hex")
        != GINE_CANONICAL_REUSE_RTOL.hex()
        or primary.get("absolute_tolerance_hex")
        != GINE_CANONICAL_REUSE_ATOL.hex()
        or primary.get("tolerance_formula")
        != "delta <= atol + rtol * abs(frozen_reference)"
        or primary.get("observation_schema")
        != COHORT_RECONCILIATION_OBSERVATION_SCHEMA
        or primary.get("observation_directory")
        != COHORT_RECONCILIATION_OBSERVATIONS
        or primary.get("required_identity_and_discrete_fields_exact") is not True
        or primary.get("frozen_cohort_preserved") is not True
        or primary.get("frozen_cohort_rewritten") is not False
    ):
        raise TasteComRecGCFullError(
            "Taste T14 cohort reconciliation receipt changed"
        )
    try:
        observation_info = observation_root.lstat()
        resolved_observation_root = observation_root.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise TasteComRecGCFullError(
            "Taste T14 cohort reconciliation observations are unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(observation_info.st_mode)
        or stat.S_ISLNK(observation_info.st_mode)
        or resolved_observation_root != observation_root
    ):
        raise TasteComRecGCFullError(
            "Taste T14 cohort reconciliation observations are aliased"
        )
    observation_paths = sorted(observation_root.iterdir())
    if not observation_paths:
        raise TasteComRecGCFullError(
            "Taste T14 cohort reconciliation has no observations"
        )
    primary_sha256 = _sha256_bytes(primary_payload)
    expected_observation_fields = {
        "schema_version",
        "status",
        "primary_receipt_sha256",
        "frozen_cohort_sha256",
        "frozen_manifest_sha256",
        "current_replayed_cohort_sha256",
        "current_replayed_manifest_sha256",
        "cohort_count",
        "source_probability_mismatch_count",
        "source_probability_max_abs_delta",
        "source_probability_max_abs_delta_hex",
        "frozen_source_probabilities_sha256",
        "current_source_probabilities_sha256",
        "mismatch_parent_ids_sha256",
        "relative_tolerance_hex",
        "absolute_tolerance_hex",
        "tolerance_formula",
        "identity_and_discrete_fields_exact",
        "cohort_order_exact",
        "manifest_except_cohort_sha_exact",
        "frozen_cohort_preserved",
        "frozen_cohort_rewritten",
    }
    inventory: list[dict[str, str]] = []
    for path in observation_paths:
        if path.suffix != ".json" or not _is_sha256(path.stem):
            raise TasteComRecGCFullError(
                "Taste T14 cohort reconciliation observation name changed"
            )
        observation, payload = _read_canonical_evidence_json(
            path,
            label="cohort reconciliation observation",
        )
        maximum_delta = observation.get("source_probability_max_abs_delta")
        maximum_delta_hex = observation.get(
            "source_probability_max_abs_delta_hex"
        )
        mismatch_count = observation.get("source_probability_mismatch_count")
        try:
            decoded_maximum_delta = float.fromhex(maximum_delta_hex)
        except (TypeError, ValueError):
            decoded_maximum_delta = math.nan
        if (
            set(observation) != expected_observation_fields
            or observation.get("schema_version")
            != COHORT_RECONCILIATION_OBSERVATION_SCHEMA
            or observation.get("status") != "PASS"
            or observation.get("primary_receipt_sha256") != primary_sha256
            or observation.get("frozen_cohort_sha256")
            != primary["frozen_cohort_sha256"]
            or observation.get("frozen_manifest_sha256")
            != primary["frozen_manifest_sha256"]
            or observation.get("current_replayed_cohort_sha256") != path.stem
            or not _is_sha256(observation.get("current_replayed_manifest_sha256"))
            or observation.get("cohort_count") != primary["cohort_count"]
            or type(mismatch_count) is not int
            or not 0 <= mismatch_count <= primary["cohort_count"]
            or type(maximum_delta) is not float
            or not math.isfinite(maximum_delta)
            or maximum_delta < 0.0
            or maximum_delta
            > GINE_CANONICAL_REUSE_ATOL + GINE_CANONICAL_REUSE_RTOL
            or decoded_maximum_delta != maximum_delta
            or maximum_delta_hex != maximum_delta.hex()
            or not _is_sha256(
                observation.get("frozen_source_probabilities_sha256")
            )
            or not _is_sha256(
                observation.get("current_source_probabilities_sha256")
            )
            or not _is_sha256(observation.get("mismatch_parent_ids_sha256"))
            or observation.get("relative_tolerance_hex")
            != GINE_CANONICAL_REUSE_RTOL.hex()
            or observation.get("absolute_tolerance_hex")
            != GINE_CANONICAL_REUSE_ATOL.hex()
            or observation.get("tolerance_formula")
            != "delta <= atol + rtol * abs(frozen_reference)"
            or observation.get("identity_and_discrete_fields_exact") is not True
            or observation.get("cohort_order_exact") is not True
            or observation.get("manifest_except_cohort_sha_exact") is not True
            or observation.get("frozen_cohort_preserved") is not True
            or observation.get("frozen_cohort_rewritten") is not False
        ):
            raise TasteComRecGCFullError(
                "Taste T14 cohort reconciliation observation changed"
            )
        inventory.append({"name": path.name, "sha256": _sha256_bytes(payload)})
    return {
        "schema_version": COHORT_RECONCILIATION_BINDING_SCHEMA,
        "status": "BOUND",
        "primary_receipt_sha256": primary_sha256,
        "observation_count": len(inventory),
        "observation_inventory_sha256": _sha256_bytes(
            _canonical_bytes(inventory)
        ),
        "frozen_cohort_sha256": primary["frozen_cohort_sha256"],
        "frozen_manifest_sha256": primary["frozen_manifest_sha256"],
        "identity_and_discrete_fields_exact": True,
        "frozen_cohort_rewritten": False,
    }


def count_train_side_valid_unique(bridge: TasteComRecGCMulticlassBridge) -> dict[str, Any]:
    """Apply the fixed canonical/train-side strict-flip retention policy."""

    bridge._assert_idle()  # type: ignore[attr-defined]
    retained: list[str] = []
    for graph_hash, record in sorted(bridge.records.items()):
        lineages = bridge.lineage_occurrences.get(graph_hash)
        collision = bridge.graph_collision_payloads.get(graph_hash)
        if (
            record.valid_fullgraph
            and record.candidate
            and record.prediction in (0, 2)
            and record.graph_identity_sha256 == graph_hash
            and record.canonical_graph
            and type(collision) is dict
            and _identity_graph_sha256(collision) == graph_hash
            and lineages
            and all(type(key) is str and len(key) == 64 and count > 0 for key, count in lineages.items())
        ):
            retained.append(graph_hash)
    return {
        "schema_version": "tastemolnet_t14_train_valid_unique_v1",
        "policy": VALID_UNIQUE_POLICY,
        "valid_unique_rule_count": len(retained),
        "valid_unique_rule_hashes_sha256": _sha256_bytes(_canonical_bytes(retained)),
        "lineage_error_count": 0,
        "pred_before": SOURCE_LABEL,
        "pred_after_condition": "pred_after != 1",
        "split": "train",
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }


def resource_cap_decision(*, completed_step: int, valid_unique_rule_count: int) -> dict[str, Any]:
    if completed_step not in {M_MAX, M_FALLBACK_MAX}:
        raise TasteComRecGCFullError("Taste T14 stop decision is off cadence")
    if valid_unique_rule_count >= MIN_VALID_UNIQUE_RULES:
        return {
            "state": "STOP_AND_POSTPROCESS",
            "m_configured_max": M_MAX,
            "m_fallback_max": M_FALLBACK_MAX,
            "m_effective": completed_step,
            "resource_cap_used": True,
            "early_stop_used": False,
            "stop_reason": (
                "RESOURCE_CAP_20K_VALID_UNIQUE_PASS"
                if completed_step == M_MAX
                else "FALLBACK_CAP_25K_VALID_UNIQUE_PASS"
            ),
        }
    if completed_step == M_MAX:
        return {
            "state": "EXTEND_ONCE_TO_25K",
            "m_configured_max": M_MAX,
            "m_fallback_max": M_FALLBACK_MAX,
            "m_effective": completed_step,
            "resource_cap_used": True,
            "early_stop_used": False,
            "stop_reason": "20K_INSUFFICIENT_VALID_UNIQUE_RULES",
        }
    return {
        "state": "SCIENTIFIC_FAILED_INSUFFICIENT_VALID_RULES",
        "m_configured_max": M_MAX,
        "m_fallback_max": M_FALLBACK_MAX,
        "m_effective": completed_step,
        "resource_cap_used": True,
        "early_stop_used": False,
        "stop_reason": "25K_INSUFFICIENT_VALID_UNIQUE_RULES",
    }


def fallback_checkpoint_targets(completed_step: int) -> tuple[int, ...]:
    """Return the only permitted durable continuation cadence after 20k."""

    if (
        type(completed_step) is not int
        or completed_step not in {M_MAX, M_MAX + CHECK_INTERVAL}
    ):
        raise TasteComRecGCFullError("Taste T14 fallback cursor is invalid")
    return tuple(
        range(completed_step + CHECK_INTERVAL, M_FALLBACK_MAX + 1, CHECK_INTERVAL)
    )


@dataclass(frozen=True, slots=True)
class _BoundedRuntimeHandles:
    live_graph_state: Any
    transition_map: Any
    route_c_updater: Any | None = None
    step_observation: dict[str, Any] | None = None


def _process_peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _process_rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _checkpoint_identity(
    *,
    authority: Mapping[str, Any],
    cohort_manifest: Mapping[str, Any],
    parameters: TasteComRecGCFullParameters,
    route_c_storage: str | None = None,
) -> tuple[dict[str, str], tuple[str, ...], str]:
    """Build output/run-ID-independent science identity for exact resume."""

    from src.baselines.comrecgc.generation_checkpoint import scientific_command_sha256

    checkpoint = authority.get("checkpoint")
    gpu = authority.get("gpu")
    official = authority.get("official")
    execution = authority.get("execution")
    train = authority.get("train")
    if not all(
        type(value) is dict
        for value in (checkpoint, gpu, official, execution, train)
    ):
        raise TasteComRecGCFullError("Taste T14 checkpoint authority is incomplete")
    scientific_argv = (
        "tastemolnet_t14_comrecgc_full_v1",
        f"train_sha256={train.get('sha256')}",
        f"checkpoint_id={checkpoint.get('checkpoint_id')}",
        f"cohort_sha256={cohort_manifest.get('cohort_jsonl_sha256')}",
        f"parameters_sha256={_sha256_bytes(_canonical_bytes(asdict(parameters)))}",
        f"official_sha256={_sha256_bytes(_canonical_bytes(official))}",
        f"execution_commit={execution.get('commit')}",
        f"physical_gpu_index={gpu.get('physical_index')}",
        f"gpu_uuid={gpu.get('uuid')}",
    )
    command_sha256 = scientific_command_sha256(scientific_argv)
    runtime_schema = (
        ROUTE_C_RUNTIME_STATE_SCHEMA
        if route_c_storage == "lowmemory"
        else RUNTIME_STATE_SCHEMA
    )
    provenance = {
        "schema_version": CHECKPOINT_PROVENANCE_SCHEMA,
        "dataset": DATASET,
        "method": METHOD,
        "stage": STAGE,
        "train_csv_sha256": str(train.get("sha256")),
        "checkpoint_id": str(checkpoint.get("checkpoint_id")),
        "cohort_jsonl_sha256": str(cohort_manifest.get("cohort_jsonl_sha256")),
        "parameters_sha256": _sha256_bytes(_canonical_bytes(asdict(parameters))),
        "official_authority_sha256": _sha256_bytes(_canonical_bytes(official)),
        "execution_commit": str(execution.get("commit")),
        "physical_gpu_index": str(gpu.get("physical_index")),
        "gpu_uuid": str(gpu.get("uuid")),
        "runtime_state_schema": runtime_schema,
        "transition_cache_policy": "compact_transition_action_replay_lru_v1",
        "graph_state_policy": (
            "t14_route_c_append_only_graph_store_bounded_lru_v1"
            if route_c_storage == "lowmemory"
            else "authoritative_backing_live_graph_resolution_v2"
        ),
        "scientific_command_sha256": command_sha256,
        "total_steps": str(M_FALLBACK_MAX),
    }
    if any(not value or value == "None" for value in provenance.values()):
        raise TasteComRecGCFullError("Taste T14 checkpoint identity is incomplete")
    return provenance, scientific_argv, command_sha256


@contextmanager
def _bounded_t14_runtime(
    *,
    module: Any,
    bridge: TasteComRecGCMulticlassBridge,
    graph_store_path: Path,
    seed: int,
    expanded_capacity: int,
    route_c_root: Path | None = None,
    route_c_resume: bool = False,
) -> Iterator[_BoundedRuntimeHandles]:
    """Install the already-reviewed exact full-walk bounded state substrate."""

    from src.baselines.comrecgc.live_graph_state import LiveGraphState
    from src.baselines.comrecgc.runtime import lineage_neighbor_wrapper
    from src.baselines.comrecgc.transition_cache import CompactMoveScopedTransitionMap
    from src.baselines.gcfexplainer_mutagenicity_adapter import (
        graph_lineage_neighbor_wrapper,
    )

    if module.transitions:
        raise TasteComRecGCFullError(
            "Taste T14 bounded transition cache must be installed before generation"
        )
    original_transitions = module.transitions
    original_move = module.move_to_next_graph
    original_neighbor = module.neighbor_graph_access
    route_c_updater = None
    original_candidates = module.counterfactual_candidates
    if route_c_root is None:
        live = LiveGraphState(
            module,
            module.graph_map,
            store_path=graph_store_path,
            seed=seed,
        )
    else:
        from src.baselines.tastemolnet_t14_route_c_fresh import (
            RouteCLiveGraphState,
            RouteCStateUpdater,
        )

        route_c_updater = RouteCStateUpdater(
            route_c_root,
            candidate_capacity=int(module.MAX_COUNTERFACTUAL_SIZE),
            record_capacity=200_000,
            lru_capacity=128,
            resume=route_c_resume,
        )
        live = RouteCLiveGraphState(route_c_updater, module, module.graph_map)
        module.counterfactual_candidates = route_c_updater.candidates
    module.graph_map = live.graph_map
    module.comrecgc_live_graph_state = live
    rebuild_with_gcf_lineage = graph_lineage_neighbor_wrapper(original_neighbor)
    rebuild_with_both_lineages = lineage_neighbor_wrapper(rebuild_with_gcf_lineage)
    transitions = CompactMoveScopedTransitionMap(
        module,
        module.transitions,
        seed=seed,
        expanded_capacity=expanded_capacity,
        rebuild_target=lambda graph, action: rebuild_with_both_lineages(graph, action),
    )
    module.transitions = transitions
    step_observation: dict[str, Any] = {}
    captured_move: dict[str, Any] = {}

    def capture_move(*args: Any, **kwargs: Any) -> Any:
        """Capture chosen actions before the move-scoped transition cache evicts."""

        parents = list(kwargs.get("graphs_hash", args[0] if args else []))
        candidate_before = {
            str(row["graph_hash"]): int(row["frequency"])
            for row in module.counterfactual_candidates
        }
        result = original_move(*args, **kwargs)
        next_hashes = result[0] if isinstance(result, tuple) and result else None
        resolved_next = (
            list(next_hashes)
            if isinstance(next_hashes, (list, tuple))
            else [next_hashes] * len(parents)
        )
        candidate_after = {
            str(row["graph_hash"]): int(row["frequency"])
            for row in module.counterfactual_candidates
        }
        selected_transitions = []
        for source_hash, target_hash in zip(parents, resolved_next, strict=True):
            record = bridge.records.get(target_hash)
            selected_transitions.append(
                {
                    "source_graph_hash": str(source_hash),
                    "target_graph_hash": (
                        None if target_hash is None else str(target_hash)
                    ),
                    "action_records": (
                        transitions.action_records(source_hash, target_hash)
                        if target_hash is not None
                        else []
                    ),
                    "duplicate_before_move": (
                        target_hash is not None
                        and str(target_hash) in candidate_before
                    ),
                    "candidate_frequency_before": (
                        None
                        if target_hash is None
                        else candidate_before.get(str(target_hash))
                    ),
                    "candidate_frequency_after": (
                        None
                        if target_hash is None
                        else candidate_after.get(str(target_hash))
                    ),
                    "accepted_candidate": (
                        target_hash is not None
                        and str(target_hash) in candidate_after
                    ),
                    "prediction": None if record is None else int(record.prediction),
                    "source_probability": (
                        None
                        if record is None
                        else float(record.probabilities[SOURCE_LABEL])
                    ),
                    "strict_flip": (
                        None
                        if record is None
                        else bool(record.prediction != SOURCE_LABEL)
                    ),
                    "valid_fullgraph": (
                        None if record is None else bool(record.valid_fullgraph)
                    ),
                }
            )
        captured_move.clear()
        captured_move.update(
            {
                "parent_graph_hashes": [str(value) for value in parents],
                "next_graph_hashes": (
                    [str(value) for value in next_hashes]
                    if isinstance(next_hashes, (list, tuple))
                    else str(next_hashes)
                ),
                "teleported": result[1] if isinstance(result, tuple) and len(result) > 1 else None,
                "recourse": result[2] if isinstance(result, tuple) and len(result) > 2 else None,
                "next_importance": result[3] if isinstance(result, tuple) and len(result) > 3 else None,
                "move_difference": result[4] if isinstance(result, tuple) and len(result) > 4 else None,
                "selected_transitions": selected_transitions,
            }
        )
        return result

    bounded_move = live.wrap_move(transitions.wrap_move(capture_move))

    def observed_move(*args: Any, **kwargs: Any) -> Any:
        parents = list(kwargs.get("graphs_hash", args[0] if args else []))
        result = bounded_move(*args, **kwargs)
        next_hashes = result[0] if isinstance(result, tuple) and result else None
        resolved_next = (
            list(next_hashes)
            if isinstance(next_hashes, (list, tuple))
            else [next_hashes] * len(parents)
        )
        if route_c_updater is not None:
            for source_hash, target_hash in zip(parents, resolved_next, strict=True):
                if target_hash is not None:
                    route_c_updater.candidates.record_transition(
                        source_hash=source_hash, target_hash=target_hash
                    )
        step_observation.clear()
        step_observation.update(
            {
                **captured_move,
                "transition_move_count": int(transitions.move_count),
            }
        )
        return result

    module.move_to_next_graph = observed_move

    def neighbor_wrapper(original: Any) -> Any:
        return lineage_neighbor_wrapper(
            graph_lineage_neighbor_wrapper(original),
            transition_cache=transitions,
        )

    completed_normally = False
    try:
        with bridge.installed(module, neighbor_wrapper=neighbor_wrapper):
            yield _BoundedRuntimeHandles(
                live_graph_state=live,
                transition_map=transitions,
                route_c_updater=route_c_updater,
                step_observation=step_observation,
            )
            completed_normally = True
    finally:
        module.move_to_next_graph = original_move
        module.graph_map = (
            {} if route_c_updater is not None else dict(live.graph_map)
        )
        if route_c_updater is not None:
            module.counterfactual_candidates = []
        module.transitions = {}
        if hasattr(module, "comrecgc_live_graph_state"):
            delattr(module, "comrecgc_live_graph_state")
        try:
            if route_c_updater is not None:
                route_c_updater.close()
            else:
                live.close()
        finally:
            if not completed_normally:
                # Keep teardown bounded after a science exception.  The latest
                # atomically published checkpoint remains the resume authority.
                module.graph_map = (
                    {} if route_c_updater is not None else dict(live.graph_map)
                )
                module.counterfactual_candidates = (
                    [] if route_c_updater is not None else original_candidates
                )
            module.transitions = original_transitions


def _checkpoint_algorithm_state(
    *,
    module: Any,
    bridge: TasteComRecGCMulticlassBridge,
    loop_state: Any,
    handles: _BoundedRuntimeHandles,
) -> dict[str, Any]:
    from src.baselines.comrecgc.generation_loop import snapshot_official_state

    transition_state = handles.transition_map.export_checkpoint_state(
        tensor_storage=True
    )
    live_state = handles.live_graph_state.export_checkpoint_state()
    required_hashes = {
        *loop_state.start_graph_hashes,
        *loop_state.current_graph_hashes,
        *module.graph_index_map.keys(),
        *module.covering_graphs,
        *(row.get("graph_hash") for row in module.counterfactual_candidates),
        *(row["source_hash"] for row in transition_state["entries"]),
    }
    missing = [
        value
        for value in required_hashes
        if value is not None and not handles.live_graph_state.contains(value)
    ]
    if missing:
        raise TasteComRecGCFullError(
            "Taste T14 checkpoint graph closure is incomplete: "
            f"missing_count={len(missing)} first={missing[0]!r}"
        )
    if handles.route_c_updater is None:
        official_state = snapshot_official_state(module)
        runtime_schema = RUNTIME_STATE_SCHEMA
        route_c_state = None
    else:
        covered = module.input_graphs_covered
        if hasattr(covered, "detach"):
            covered = covered.detach().cpu().clone()
        official_state = {
            "schema_version": "tastemolnet_t14_route_c_official_state_v1",
            "graph_index_map": dict(module.graph_index_map),
            "input_graphs_covered": covered,
            "covering_graphs": set(module.covering_graphs),
            "start": dict(module.start),
            "is_sample": bool(module.is_sample),
            "starting_step": int(module.starting_step),
            "traversed_hashes": list(module.traversed_hashes),
            "sample_size": int(module.sample_size),
            "MAX_COUNTERFACTUAL_SIZE": int(module.MAX_COUNTERFACTUAL_SIZE),
            "graph_objects_saved": 0,
            "full_python_candidate_list_saved": False,
        }
        runtime_schema = ROUTE_C_RUNTIME_STATE_SCHEMA
        route_c_state = handles.route_c_updater.checkpoint_state()
    state = {
        "schema_version": runtime_schema,
        "loop_state": loop_state.to_checkpoint_state(),
        "official_state": official_state,
        "transition_state": transition_state,
        "live_graph_state": live_state,
        "bridge_state": bridge.checkpoint_state(),
    }
    if route_c_state is not None:
        state["route_c_state"] = route_c_state
    return state


def _restore_route_c_official_state(module: Any, value: Mapping[str, Any]) -> None:
    """Restore only non-disk official globals for a Route C segment."""

    if (
        type(value) is not dict
        or value.get("schema_version")
        != "tastemolnet_t14_route_c_official_state_v1"
        or value.get("graph_objects_saved") != 0
        or value.get("full_python_candidate_list_saved") is not False
    ):
        raise TasteComRecGCFullError("Taste T14 Route C official state changed")
    required = {
        "graph_index_map",
        "input_graphs_covered",
        "covering_graphs",
        "start",
        "traversed_hashes",
        "sample_size",
        "MAX_COUNTERFACTUAL_SIZE",
    }
    if not required.issubset(value):
        raise TasteComRecGCFullError("Taste T14 Route C official state is incomplete")
    module.graph_map = {}
    module.graph_index_map = dict(value["graph_index_map"])
    module.counterfactual_candidates = []
    module.input_graphs_covered = value["input_graphs_covered"]
    module.covering_graphs = set(value["covering_graphs"])
    module.transitions = {}
    module.start = dict(value["start"])
    module.is_sample = bool(value.get("is_sample", True))
    module.starting_step = int(value.get("starting_step", 1))
    module.traversed_hashes = list(value["traversed_hashes"])
    module.sample_size = int(value["sample_size"])
    module.MAX_COUNTERFACTUAL_SIZE = int(value["MAX_COUNTERFACTUAL_SIZE"])


def _restore_checkpoint_state(
    *,
    module: Any,
    bridge: TasteComRecGCMulticlassBridge,
    loaded: Any,
    handles: _BoundedRuntimeHandles,
) -> Any:
    from src.baselines.comrecgc.generation_checkpoint import restore_rng_state
    from src.baselines.comrecgc.generation_loop import GenerationLoopState

    state = loaded.algorithm_state
    expected_schema = (
        ROUTE_C_RUNTIME_STATE_SCHEMA
        if handles.route_c_updater is not None
        else RUNTIME_STATE_SCHEMA
    )
    if state.get("schema_version") != expected_schema:
        raise TasteComRecGCFullError("Taste T14 runtime checkpoint schema changed")
    if handles.route_c_updater is not None:
        route_c_state = state.pop("route_c_state")
        handles.route_c_updater.restore_checkpoint_state(route_c_state)
    transition_state = state.pop("transition_state")
    live_graph_state = state.pop("live_graph_state")
    bridge_state = state.pop("bridge_state")
    loop_payload = state.pop("loop_state")
    handles.transition_map.restore_checkpoint_state(transition_state, consume=True)
    handles.live_graph_state.restore_checkpoint_state(live_graph_state)
    bridge.restore_checkpoint_state(bridge_state, consume=True)
    loop_state = GenerationLoopState.from_checkpoint_state(loop_payload)
    if handles.route_c_updater is not None:
        candidates = handles.route_c_updater.candidates
        rebuilt = {
            candidates.graph_hash_at(index): index for index in range(len(candidates))
        }
        if rebuilt != module.graph_index_map:
            raise TasteComRecGCFullError(
                "Taste T14 Route C candidate/index state changed on reload"
            )
    if (
        handles.transition_map.move_count != loop_state.completed_step
        or handles.live_graph_state.move_count != loop_state.completed_step
        or loop_state.completed_step != loaded.completed_step
    ):
        raise TasteComRecGCFullError("Taste T14 restored move counters changed")
    # RNG restoration is deliberately the final mutation before next_step.
    restore_rng_state(loaded.rng_state)
    loaded.rng_state.clear()
    loaded.trace_state.clear()
    state.clear()
    return loop_state


def _write_checkpoint(
    *,
    module: Any,
    bridge: TasteComRecGCMulticlassBridge,
    loop_state: Any,
    parameters: TasteComRecGCFullParameters,
    checkpoint_root: Path,
    handles: _BoundedRuntimeHandles,
    provenance: Mapping[str, str],
    scientific_argv: Sequence[str],
    command_sha256: str,
) -> dict[str, Any]:
    from src.baselines.comrecgc.generation_checkpoint import (
        save_generation_checkpoint,
    )

    completed = int(loop_state.completed_step)
    # ``parameters`` is the frozen scientific configuration.  Route C's
    # checkpoint cadence is a transport/recovery contract and deliberately
    # includes early boundaries at 50 and 100, which are not legal legacy
    # full-run checkpoint values.  Validate the two contracts independently;
    # replacing ``checkpoint_step`` with the live cursor incorrectly reported
    # a scientific-parameter drift at the first Route C checkpoint.
    parameters.validate()
    if handles.route_c_updater is None:
        checkpoint_steps = range(
            CHECK_INTERVAL, M_FALLBACK_MAX + 1, CHECK_INTERVAL
        )
    else:
        from src.baselines.tastemolnet_t14_route_c_fresh import (
            EARLY_CHECKPOINT_STEPS,
            PROMOTABLE_CHECKPOINT_STEP,
        )

        checkpoint_steps = (
            *EARLY_CHECKPOINT_STEPS,
            PROMOTABLE_CHECKPOINT_STEP,
            *range(CHECK_INTERVAL, M_FALLBACK_MAX + 1, CHECK_INTERVAL),
        )
    if completed not in checkpoint_steps:
        raise TasteComRecGCFullError(
            f"Taste T14 checkpoint step is off cadence: {completed}"
        )
    validation = save_generation_checkpoint(
        checkpoint_root,
        completed_step=completed,
        step_complete=True,
        algorithm_state=_checkpoint_algorithm_state(
            module=module,
            bridge=bridge,
            loop_state=loop_state,
            handles=handles,
        ),
        trace_state={"enabled": False, "policy": "generation_trace_disabled"},
        sqlite_source=handles.live_graph_state.store.checkpoint_connection,
        provenance_fingerprints=provenance,
        scientific_argv=scientific_argv,
        command_sha256=command_sha256,
        total_steps=M_FALLBACK_MAX,
        # The live T14 state is hundreds of GiB.  The exact bytes and the
        # already-live state are validated here; the SERIAL_ONLY owner runs
        # the independent reload after science releases memory.
        reload_after_write=False,
    )
    evidence = {
        "schema_version": "tastemolnet_t14_checkpoint_v2",
        "checkpoint_dir": str(validation.checkpoint_dir),
        "checkpoint_digest": validation.checkpoint_digest,
        "checkpoint_step": completed,
        "next_step": completed + 1,
        "checkpoint_persisted_in_output": True,
        "bounded_transition_state": True,
        "authoritative_graph_store_snapshot": True,
        "payload_reload_state": "PENDING_INDEPENDENT_RELOAD",
        "latest_promoted": False,
        "written_at": _utc_now(),
    }
    _atomic_json(checkpoint_root / f"checkpoint-{completed:06d}.json", evidence)
    return evidence


def _write_route_c_boundary(
    *,
    root: Path,
    checkpoint_identity_path: Path,
    route_c_contract: Mapping[str, Any],
    target: int,
    valid_unique_rule_count: int,
    checkpoint_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Seal every Route C boundary, including crash-resumable production ones."""

    replay = target == 510
    if not replay and checkpoint_evidence is None:
        raise TasteComRecGCFullError("Taste T14 Route C checkpoint evidence is missing")
    boundary = {
        "schema_version": "tastemolnet_t14_route_c_first_checkpoint_v1",
        "status": "REPLAY_BOUNDARY_REACHED" if replay else "CHECKPOINT_BOUNDARY_REACHED",
        "attempt_uuid": route_c_contract["attempt_uuid"],
        "spec_sha256": route_c_contract["spec_sha256"],
        "output_root": str(root),
        "completed_step": target,
        "next_step": target + 1,
        "checkpoint_dir": None if replay else checkpoint_evidence["checkpoint_dir"],
        "checkpoint_digest": None if replay else checkpoint_evidence["checkpoint_digest"],
        "checkpoint_identity_sha256": _sha256_file(checkpoint_identity_path),
        "payload_reload_state": (
            "NOT_APPLICABLE_REPLAY_TERMINAL"
            if replay
            else "PENDING_INDEPENDENT_RELOAD"
        ),
        "latest_promoted": False,
        "legacy_checkpoint_loaded": False,
        "valid_unique_rule_count": int(valid_unique_rule_count),
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "written_at": _utc_now(),
    }
    path = root / f"route_c_boundary_{target:06d}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        stable = set(boundary) - {"written_at"}
        if any(existing.get(key) != boundary.get(key) for key in stable):
            raise TasteComRecGCFullError("Taste T14 Route C boundary already differs")
        return existing
    _atomic_json(path, boundary)
    return boundary


def _load_latest_checkpoint(
    checkpoint_root: Path,
    *,
    parameters: TasteComRecGCFullParameters,
    provenance: Mapping[str, str],
    scientific_argv: Sequence[str],
    command_sha256: str,
    route_c_storage: str | None = None,
) -> Any | None:
    from src.baselines.comrecgc.generation_checkpoint import load_generation_checkpoint

    latest_path = checkpoint_root / "LATEST"
    if not latest_path.is_file():
        return None
    try:
        pointer = json.loads(latest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TasteComRecGCFullError("Taste T14 checkpoint pointer is unreadable") from exc
    checkpoint_name = pointer.get("checkpoint_dir") if type(pointer) is dict else None
    completed_pointer = pointer.get("completed_step") if type(pointer) is dict else None
    digest_pointer = pointer.get("checkpoint_digest") if type(pointer) is dict else None
    if (
        pointer.get("schema_version")
        != "comrecgc_generation_checkpoint_latest_v1"
        or type(checkpoint_name) is not str
        or not checkpoint_name.startswith("step-")
        or len(checkpoint_name) != len("step-000000000000")
        or not checkpoint_name[5:].isdigit()
        or type(completed_pointer) is not int
        or completed_pointer != int(checkpoint_name[5:])
        or type(digest_pointer) is not str
        or len(digest_pointer) != 64
        or any(character not in "0123456789abcdef" for character in digest_pointer)
    ):
        raise TasteComRecGCFullError("Taste T14 checkpoint pointer changed")
    checkpoint_dir = checkpoint_root / checkpoint_name
    if (
        not checkpoint_dir.is_dir()
        or checkpoint_dir.is_symlink()
        or checkpoint_dir.resolve(strict=True).parent != checkpoint_root.resolve(strict=True)
    ):
        raise TasteComRecGCFullError("Taste T14 latest checkpoint path is unsafe")
    loaded = load_generation_checkpoint(
        checkpoint_dir,
        expected_provenance=provenance,
        expected_scientific_argv=scientific_argv,
        expected_command_sha256=command_sha256,
        expected_total_steps=M_FALLBACK_MAX,
        expected_completed_step=completed_pointer,
        single_pass=True,
    )
    completed = loaded.completed_step
    valid_steps = (
        {50, 100, 250, 500, *range(CHECK_INTERVAL, M_FALLBACK_MAX + 1, CHECK_INTERVAL)}
        if route_c_storage is not None
        else set(range(CHECK_INTERVAL, M_FALLBACK_MAX + 1, CHECK_INTERVAL))
    )
    expected_schema = (
        ROUTE_C_RUNTIME_STATE_SCHEMA
        if route_c_storage == "lowmemory"
        else RUNTIME_STATE_SCHEMA
    )
    if (
        completed not in valid_steps
        or loaded.algorithm_state.get("schema_version") != expected_schema
        or loaded.validation.checkpoint_digest != digest_pointer
    ):
        raise TasteComRecGCFullError("Taste T14 checkpoint payload changed")
    return loaded


def _prepare_runtime_store(root: Path, loaded: Any | None) -> Path:
    from src.baselines.comrecgc.generation_checkpoint import restore_sqlite_snapshot

    runtime_root = root / "runtime_graph_state"
    runtime_root.mkdir(mode=0o700, exist_ok=True)
    invocation = Path(
        tempfile.mkdtemp(prefix=f"active-{os.getpid()}-", dir=runtime_root)
    )
    path = invocation / "authoritative_graph_store.sqlite3"
    if loaded is not None:
        restore_sqlite_snapshot(loaded.sqlite_snapshot_path, path)
    return path


def _progress(
    output_root: Path,
    *,
    phase: str,
    completed_step: int,
    cohort_count: int,
    valid_unique_rule_count: int | None = None,
    runtime_handles: _BoundedRuntimeHandles | None = None,
) -> None:
    runtime = None
    if runtime_handles is not None:
        transition = runtime_handles.transition_map.audit()
        runtime = {
            "rss_bytes": _process_rss_bytes(),
            "peak_rss_bytes": _process_peak_rss_bytes(),
            "hot_graph_count": len(runtime_handles.live_graph_state.graph_map),
            "backing_graph_count": runtime_handles.live_graph_state.store.count(),
            "transition_entry_count": transition["transition_entry_count"],
            "transition_numeric_bytes": transition["compact_numeric_bytes"],
            "expanded_transition_entry_count": transition[
                "expanded_entry_count"
            ],
            "expanded_transition_capacity": transition["expanded_capacity"],
        }
    _atomic_json(
        output_root / "progress.json",
        {
            "schema_version": "tastemolnet_t14_progress_v1",
            "status": "RUNNING",
            "phase": phase,
            "completed_step": completed_step,
            "m_configured_max": M_MAX,
            "m_fallback_max": M_FALLBACK_MAX,
            "cohort_count": cohort_count,
            "valid_unique_rule_count": valid_unique_rule_count,
            "bounded_runtime": runtime,
            "pid": os.getpid(),
            "updated_at": _utc_now(),
        },
    )


def _initialize_full_source_graphs(
    *,
    checkpoint_payloads: Mapping[str, bytes],
    source_rows: Sequence[Any],
    graph_schema: Any,
    device: str,
) -> tuple[list[Any], list[Mapping[str, Any]], Any, dict[str, Any]]:
    """Reopen every frozen cohort row, including duplicate molecules."""

    records = [encode_taste_source_graph(row, graph_schema) for row in source_rows]
    graphs: list[Any] = []
    parent_ids: list[str] = []
    for index, (row, record) in enumerate(zip(source_rows, records, strict=True)):
        graph = taste_record_to_pyg(record, origin_index=index)
        graph.comrecgc_node_origin = graph.gcf_node_origin.clone()
        graph.comrecgc_source_index = index
        graph.comrecgc_parent_id = f"private-cohort-{index:06d}"
        graphs.append(graph)
        parent_ids.append(str(row.molecule_id))
    adapter = TasteFrozenGINENativeAdapter(
        checkpoint_payloads,
        source_records=records,
        graph_schema=graph_schema,
        device=device,
    )
    predictions: list[int] = []
    identities: list[str] = []
    for offset in range(0, len(graphs), 128):
        chunk = graphs[offset : offset + 128]
        scored = adapter.score(chunk)
        if any(not valid for valid in scored.valid_fullgraphs):
            raise TasteComRecGCFullError("Taste T14 cohort graph failed exact replay")
        predictions.extend(scored.predictions)
        identities.extend(
            canonical_attributed_graph(
                graph,
                feature_atomic_numbers=graph_schema.feature_atomic_numbers,
            ).graph_identity_sha256
            for graph in chunk
        )
    if len(predictions) != len(graphs) or any(
        prediction != SOURCE_LABEL for prediction in predictions
    ):
        raise TasteComRecGCFullError("Taste T14 cohort prediction changed on replay")
    evidence = {
        "schema_version": "tastemolnet_t14_source_cohort_v1",
        "source_split": "train",
        "source_label": SOURCE_LABEL,
        "source_count": len(graphs),
        "source_cohort_sha256": _sha256_bytes(
            _canonical_bytes(
                [
                    {"parent_id": parent_id, "canonical_graph_hash": graph_hash}
                    for parent_id, graph_hash in zip(parent_ids, identities, strict=True)
                ]
            )
        ),
        "parent_ids_unique": len(set(parent_ids)) == len(parent_ids),
        "canonical_graph_dedup_applied": False,
        "duplicate_graph_row_count": len(identities) - len(set(identities)),
        "duplicate_graph_rows_retained": True,
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    return graphs, records, adapter, evidence


def run_t14_full(
    *,
    inputs: Any,
    output_root: str | Path,
    resume: bool,
    resume_spec: str | Path | None = None,
    route_c_spec: str | Path | None = None,
    route_c_storage: str | None = None,
    checkpoint_only_step: int | None = None,
    convergence_receipt: str | Path | None = None,
) -> dict[str, Any]:
    """Run or resume the train-only native full generation and postprocessing."""

    from src.baselines.comrecgc.generation_loop import (
        restore_official_state,
        run_generation_loop,
    )
    from src.baselines.comrecgc.runtime import reset_official_state
    from src.baselines.gcfexplainer_mutagenicity_adapter import (
        GraphRecordDataset,
    )

    root = Path(output_root)
    if not root.is_absolute() or Path(os.path.abspath(root)) != root:
        raise TasteComRecGCFullError("Taste T14 output root must be normalized")
    if root.exists() and not resume:
        raise FileExistsError(f"Taste T14 output already exists: {root}")
    if resume_spec is not None and route_c_spec is not None:
        raise TasteComRecGCFullError(
            "Taste T14 legacy resume and fresh Route C are mutually exclusive"
        )
    if convergence_receipt is not None and (
        route_c_spec is None or not resume or checkpoint_only_step is not None
    ):
        raise TasteComRecGCFullError(
            "Taste T14 convergence receipt requires Route C finalization resume"
        )
    route_c_enabled = route_c_spec is not None
    if bool(route_c_storage) != route_c_enabled:
        raise TasteComRecGCFullError(
            "Taste T14 Route C spec and storage mode must be supplied together"
        )
    if route_c_storage not in {None, "reference", "lowmemory"}:
        raise TasteComRecGCFullError("Taste T14 Route C storage mode is invalid")
    if checkpoint_only_step is not None and (
        route_c_spec is None
        or type(checkpoint_only_step) is not int
        or checkpoint_only_step
        not in {50, 100, 250, 500, 510, *range(2_500, 25_001, 2_500)}
    ):
        raise TasteComRecGCFullError(
            "Taste T14 Route C stop is not a registered checkpoint/replay boundary"
        )
    if not root.exists():
        root.mkdir(parents=True, mode=0o700)
        _fsync_dir(root.parent)
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(mode=0o700, exist_ok=True)
    authority = inputs.revalidate()
    train = authority["train"]
    loaded_train = load_train_rows(
        inputs.train_file.read_bytes(),
        source_path=Path(train["path"]),
        expected_num_records=int(train["num_records"]),
        expected_label_counts=train["label_counts"],
    )
    checkpoint_payloads = {
        name: inputs.checkpoint_payloads[name]
        for name in (
            "model.pt",
            "model_card.json",
            "feature_schema.json",
            "label_map.json",
            "split_manifest.json",
            "test_evaluation_status.json",
            "temperature_scaling.json",
        )
    }
    true_sweet = list(loaded_train.sweet_rows)
    encoded_all = [
        encode_taste_source_graph(row, loaded_train.schema) for row in true_sweet
    ]
    graphs_all = [
        taste_record_to_pyg(record, origin_index=index)
        for index, record in enumerate(encoded_all)
    ]
    cohort_adapter = TasteFrozenGINENativeAdapter(
        checkpoint_payloads,
        source_records=encoded_all,
        graph_schema=loaded_train.schema,
        device="cuda:0",
    )
    predictions: list[int] = []
    source_probabilities: list[float] = []
    graph_hashes: list[str] = []
    for offset in range(0, len(graphs_all), 128):
        graphs = graphs_all[offset : offset + 128]
        scored = cohort_adapter.score(graphs)
        predictions.extend(scored.predictions)
        source_probabilities.extend(
            float(row[SOURCE_LABEL]) for row in scored.probabilities.tolist()
        )
        for graph in graphs:
            graph_hashes.append(
                canonical_attributed_graph(
                    graph,
                    feature_atomic_numbers=loaded_train.schema.feature_atomic_numbers,
                ).graph_identity_sha256
            )
    cohort, cohort_manifest, cohort_bytes = build_full_train_correct_source_cohort(
        true_sweet_rows=true_sweet,
        predictions=predictions,
        source_probabilities=source_probabilities,
        canonical_graph_hashes=graph_hashes,
        train_csv_sha256=str(train["sha256"]),
        checkpoint_id=str(authority["checkpoint"]["checkpoint_id"]),
    )
    existing_cohort = root / "cohort.jsonl"
    existing_manifest = root / "cohort_manifest.json"
    cohort_reconciliation_binding: dict[str, Any] | None = None
    if existing_cohort.exists() or existing_manifest.exists():
        if (
            not existing_cohort.is_file()
            or existing_cohort.is_symlink()
            or not existing_manifest.is_file()
            or existing_manifest.is_symlink()
        ):
            raise TasteComRecGCFullError(
                "Taste T14 frozen cohort files are incomplete or aliased"
            )
        frozen_cohort_bytes = existing_cohort.read_bytes()
        frozen_manifest_bytes = existing_manifest.read_bytes()
        (
            cohort,
            cohort_manifest,
            reconciliation_primary,
            reconciliation_observation,
        ) = reconcile_t14_resume_cohort(
            frozen_cohort_bytes=frozen_cohort_bytes,
            frozen_manifest_bytes=frozen_manifest_bytes,
            replayed_rows=cohort,
            replayed_manifest=cohort_manifest,
        )
        cohort_bytes = frozen_cohort_bytes
        _persist_cohort_reconciliation_evidence(
            root,
            primary_receipt=reconciliation_primary,
            observation=reconciliation_observation,
        )
        cohort_reconciliation_binding = _validate_cohort_reconciliation_evidence(
            root,
            frozen_cohort_bytes=frozen_cohort_bytes,
            frozen_manifest_bytes=frozen_manifest_bytes,
        )
    else:
        _atomic_write(existing_cohort, cohort_bytes)
        _atomic_json(existing_manifest, cohort_manifest)
    selected_ids = {row["parent_id"] for row in cohort}
    selected_rows = [row for row in true_sweet if row.molecule_id in selected_ids]
    selected_rows.sort(key=lambda row: row.molecule_id)
    if [row.molecule_id for row in selected_rows] != [row["parent_id"] for row in cohort]:
        raise TasteComRecGCFullError("Taste T14 cohort/source row order changed")
    parameters = TasteComRecGCFullParameters(
        source_pool=len(selected_rows), source_count=len(selected_rows)
    ).validate()
    provenance, scientific_argv, command_sha256 = _checkpoint_identity(
        authority=authority,
        cohort_manifest=cohort_manifest,
        parameters=parameters,
        route_c_storage=route_c_storage,
    )
    checkpoint_identity = {
        "schema_version": CHECKPOINT_PROVENANCE_SCHEMA,
        "status": "FROZEN",
        "provenance": provenance,
        "scientific_argv": list(scientific_argv),
        "command_sha256": command_sha256,
        "total_steps": M_FALLBACK_MAX,
        "checkpoint_interval": 50 if route_c_enabled else CHECK_INTERVAL,
        "transition_expanded_capacity": TRANSITION_EXPANDED_CAPACITY,
        "raw_neighbor_graphs_retained_unbounded": False,
    }
    checkpoint_identity_path = root / "checkpoint_identity.json"
    resume_transport_receipt: dict[str, Any] | None = None
    if resume_spec is not None:
        if not resume:
            raise TasteComRecGCFullError(
                "Taste T14 resume spec is forbidden for a fresh generation"
            )
        from src.baselines.tastemolnet_t14_resume import (
            T14ResumeError,
            bind_resume_identity,
        )

        try:
            checkpoint_identity, resume_transport_receipt = bind_resume_identity(
                spec_path=resume_spec,
                output_root=root,
                current_execution_commit=str(authority["execution"]["commit"]),
                current_checkpoint_identity=checkpoint_identity,
            )
        except T14ResumeError as exc:
            raise TasteComRecGCFullError(
                "Taste T14 low-memory resume binding failed"
            ) from exc
        provenance = dict(checkpoint_identity["provenance"])
        scientific_argv = tuple(checkpoint_identity["scientific_argv"])
        command_sha256 = str(checkpoint_identity["command_sha256"])
    route_c_contract: dict[str, Any] | None = None
    route_c_promotion_receipt: dict[str, Any] | None = None
    route_c_convergence_receipt: dict[str, Any] | None = None
    if route_c_spec is not None:
        from src.baselines.tastemolnet_t14_route_c_fresh import (
            load_spec as load_route_c_spec,
            promotion_receipt_path,
            validate_promotion_receipt,
        )

        route_c_contract = load_route_c_spec(Path(route_c_spec))
        if (
            route_c_contract["output_root"] != str(root)
            or route_c_contract["execution_commit"]
            != str(authority["execution"]["commit"])
            or route_c_contract["gpu_index"] != 2
            or route_c_contract["storage_mode"] != route_c_storage
        ):
            raise TasteComRecGCFullError(
                "Taste T14 Route C spec differs from the live science identity"
            )
        if resume:
            latest_pointer_path = root / "checkpoints" / "LATEST"
            if not latest_pointer_path.is_file():
                raise TasteComRecGCFullError(
                    "Taste T14 Route C resume lacks a promoted checkpoint"
                )
            latest_pointer = json.loads(
                latest_pointer_path.read_text(encoding="utf-8")
            )
            latest_step = int(latest_pointer.get("completed_step", -1))
            route_c_promotion_receipt = validate_promotion_receipt(
                promotion_receipt_path(route_c_contract, latest_step),
                spec=route_c_contract,
                expected_step=latest_step,
            )
            if convergence_receipt is not None:
                from src.baselines.tastemolnet_t14_route_c_fresh import (
                    validate_route_c_convergence_receipt,
                )

                route_c_convergence_receipt = validate_route_c_convergence_receipt(
                    Path(convergence_receipt),
                    spec=route_c_contract,
                    expected_step=latest_step,
                    expected_checkpoint_digest=str(
                        latest_pointer.get("checkpoint_digest") or ""
                    ),
                )
    if checkpoint_identity_path.exists():
        if json.loads(checkpoint_identity_path.read_text(encoding="utf-8")) != (
            checkpoint_identity
        ):
            raise TasteComRecGCFullError(
                "Taste T14 checkpoint identity changed on resume"
            )
    else:
        _atomic_json(checkpoint_identity_path, checkpoint_identity)
    if resume_transport_receipt is not None:
        receipt_path = root / "resume_transport_receipt.json"
        if receipt_path.exists():
            if json.loads(receipt_path.read_text(encoding="utf-8")) != (
                resume_transport_receipt
            ):
                raise TasteComRecGCFullError(
                    "Taste T14 resume transport receipt changed"
                )
        else:
            _atomic_json(receipt_path, resume_transport_receipt)
    _progress(root, phase="COHORT_FROZEN", completed_step=0, cohort_count=len(cohort))

    # The cohort-selection adapter and all-Sweet graph materialization are no
    # longer needed.  Keeping them alive throughout the 20k walk duplicates a
    # complete GINE/source-graph working set for no scientific purpose.
    del cohort_adapter, graphs_all, encoded_all, predictions
    del source_probabilities, graph_hashes, true_sweet
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    modules = inputs.official.modules
    module = modules["comrecgc"]
    source_graphs, _records, adapter, source_evidence = _initialize_full_source_graphs(
        checkpoint_payloads=checkpoint_payloads,
        source_rows=selected_rows,
        graph_schema=loaded_train.schema,
        device="cuda:0",
    )
    # Preserve the exact historical source-cohort replay used by already
    # committed T14 checkpoints.  Canonical row replay begins only at the
    # generation bridge; restore below primes it from validated bridge rows.
    adapter.enable_canonical_replay_cache()
    dataset = GraphRecordDataset(
        source_graphs, num_features=len(loaded_train.schema.feature_atomic_numbers)
    )
    reset_official_state(
        module,
        candidate_capacity=parameters.candidate_capacity,
        sample_size=parameters.sample_size,
    )
    import torch

    module.input_graphs_covered = torch.zeros(
        parameters.source_count, dtype=torch.float32
    )
    bridge = TasteComRecGCFullBridge(
        adapter=adapter,
        feature_atomic_numbers=loaded_train.schema.feature_atomic_numbers,
        cohort_count=len(source_graphs),
    )

    importance_args = {
        "schema_version": "tastemolnet_comrecgc_gine_distance_v1",
        "classifier": "frozen_calibrated_three_class_gine",
        "distance_embedding": "frozen_gine_graph_hidden",
        "num_classes": 3,
        "source_label": SOURCE_LABEL,
    }
    latest = _load_latest_checkpoint(
        checkpoint_root,
        parameters=parameters,
        provenance=provenance,
        scientific_argv=scientific_argv,
        command_sha256=command_sha256,
        route_c_storage=route_c_storage,
    )
    if resume and latest is None:
        raise TasteComRecGCFullError(
            "Taste T14 resume requested without a complete 2,500-step checkpoint"
        )
    if not resume and latest is not None:
        raise TasteComRecGCFullError(
            "Taste T14 fresh run unexpectedly found a checkpoint"
        )
    if latest is not None:
        runtime_state = latest.algorithm_state
        expected_runtime_schema = (
            ROUTE_C_RUNTIME_STATE_SCHEMA
            if route_c_storage == "lowmemory"
            else RUNTIME_STATE_SCHEMA
        )
        if runtime_state.get("schema_version") != expected_runtime_schema:
            raise TasteComRecGCFullError("Taste T14 checkpoint runtime changed")
        if route_c_storage == "lowmemory":
            if route_c_promotion_receipt is None:
                raise TasteComRecGCFullError(
                    "Taste T14 Route C resume lacks its validated promotion receipt"
                )
            from src.baselines.tastemolnet_t14_route_c_fresh import (
                recover_route_c_external_state,
            )

            recover_route_c_external_state(
                output_root=root,
                loaded=latest,
                promotion_receipt=route_c_promotion_receipt,
            )
            _restore_route_c_official_state(
                module, runtime_state.pop("official_state")
            )
        else:
            restore_official_state(
                module,
                runtime_state.pop("official_state"),
                consume=True,
            )
    graph_store_path = _prepare_runtime_store(root, latest)
    with _bounded_t14_runtime(
        module=module,
        bridge=bridge,
        graph_store_path=graph_store_path,
        seed=parameters.seed,
        expanded_capacity=TRANSITION_EXPANDED_CAPACITY,
        route_c_root=(root / "route_c_state")
        if route_c_storage == "lowmemory"
        else None,
        route_c_resume=bool(latest is not None),
    ) as runtime_handles:
        state = None
        if latest is not None:
            state = _restore_checkpoint_state(
                module=module,
                bridge=bridge,
                loaded=latest,
                handles=runtime_handles,
            )
        else:
            _seed_all(parameters.seed)
        completed = int(state.completed_step) if state is not None else 0

        def progress_callback(loop_state: Any) -> None:
            step = int(loop_state.completed_step)
            if route_c_contract is not None and step <= 510:
                from src.baselines.tastemolnet_t14_route_c_fresh import (
                    append_step_state,
                    scientific_state_digest,
                )

                append_step_state(
                    root / "route_c_step_states.jsonl",
                    scientific_state_digest(
                        module=module,
                        bridge=bridge,
                        loop_state=loop_state,
                        selected=runtime_handles.step_observation or {},
                        transition_map=runtime_handles.transition_map,
                        live_graph_state=runtime_handles.live_graph_state,
                    ),
                )
            if step % PROGRESS_INTERVAL == 0:
                _progress(
                    root,
                    phase="GENERATION",
                    completed_step=step,
                    cohort_count=len(cohort),
                    runtime_handles=runtime_handles,
                )

        primary_stop = (
            int(route_c_convergence_receipt["m_effective"])
            if route_c_convergence_receipt is not None
            else checkpoint_only_step or M_MAX
        )
        if route_c_enabled:
            from src.baselines.tastemolnet_t14_route_c_fresh import (
                checkpoint_targets,
            )

            targets = list(
                checkpoint_targets(
                    completed_step=completed,
                    stop_step=primary_stop,
                    route_c=True,
                )
            )
            if primary_stop == 510 and completed < 510 and 510 not in targets:
                targets.append(510)
        else:
            targets = list(
                range(
                    ((completed // CHECK_INTERVAL) + 1) * CHECK_INTERVAL,
                    primary_stop + 1,
                    CHECK_INTERVAL,
                )
            )
        for target in targets:
            state = run_generation_loop(
                module,
                input_graphs=dataset,
                importance_args=importance_args,
                teleport_probability=parameters.teleport_probability,
                max_steps=target,
                heads=parameters.heads,
                initial_state=state,
                on_step_complete=progress_callback,
            )
            checkpoint_evidence = None
            if target != 510:
                checkpoint_evidence = _write_checkpoint(
                    module=module,
                    bridge=bridge,
                    loop_state=state,
                    parameters=parameters,
                    checkpoint_root=checkpoint_root,
                    handles=runtime_handles,
                    provenance=provenance,
                    scientific_argv=scientific_argv,
                    command_sha256=command_sha256,
                )
            valid = count_train_side_valid_unique(bridge)
            _progress(
                root,
                phase="GENERATION",
                completed_step=target,
                cohort_count=len(cohort),
                valid_unique_rule_count=int(valid["valid_unique_rule_count"]),
                runtime_handles=runtime_handles,
            )
            boundary = None
            if route_c_contract is not None:
                boundary = _write_route_c_boundary(
                    root=root,
                    checkpoint_identity_path=checkpoint_identity_path,
                    route_c_contract=route_c_contract,
                    target=target,
                    valid_unique_rule_count=int(valid["valid_unique_rule_count"]),
                    checkpoint_evidence=checkpoint_evidence,
                )
            if checkpoint_only_step is not None and target == checkpoint_only_step:
                if route_c_contract is None:
                    raise TasteComRecGCFullError(
                        "Taste T14 first-checkpoint stop lacks its Route C spec"
                    )
                if boundary is None:
                    raise TasteComRecGCFullError(
                        "Taste T14 Route C boundary was not durably sealed"
                    )
                _progress(
                    root,
                    phase=(
                        "ROUTE_C_REPLAY_510_COMPLETE"
                        if target == 510
                        else "ROUTE_C_CHECKPOINT_PENDING_INDEPENDENT_RELOAD"
                    ),
                    completed_step=target,
                    cohort_count=len(cohort),
                    valid_unique_rule_count=int(valid["valid_unique_rule_count"]),
                    runtime_handles=runtime_handles,
                )
                return boundary
        if state is None:
            raise TasteComRecGCFullError("Taste T14 generation produced no state")
        completed_now = int(state.completed_step)
        valid = count_train_side_valid_unique(bridge)
        if route_c_convergence_receipt is not None:
            if completed_now != int(route_c_convergence_receipt["m_effective"]):
                raise TasteComRecGCFullError(
                    "Taste T14 convergence checkpoint cursor changed"
                )
            decision = {
                "state": "STOP_AND_POSTPROCESS",
                "m_configured_max": M_MAX,
                "m_fallback_max": M_FALLBACK_MAX,
                "m_effective": completed_now,
                "resource_cap_used": False,
                "early_stop_used": True,
                "stop_reason": route_c_convergence_receipt["stop_reason"],
                "convergence_receipt": str(Path(convergence_receipt)),
                "convergence_receipt_sha256": route_c_convergence_receipt[
                    "receipt_sha256"
                ],
            }
        elif completed_now == M_MAX or completed_now == M_FALLBACK_MAX:
            decision = resource_cap_decision(
                completed_step=completed_now,
                valid_unique_rule_count=int(valid["valid_unique_rule_count"]),
            )
        elif M_MAX < completed_now < M_FALLBACK_MAX:
            receipt_path = root / "resource_cap_receipt.json"
            if not receipt_path.is_file():
                raise TasteComRecGCFullError(
                    "Taste T14 resumed fallback lacks its 20k authorization"
                )
            decision = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                type(decision) is not dict
                or decision.get("state") != "EXTEND_ONCE_TO_25K"
                or decision.get("m_effective") != M_MAX
            ):
                raise TasteComRecGCFullError(
                    "Taste T14 fallback authorization changed"
                )
        else:
            raise TasteComRecGCFullError("Taste T14 resume cursor is invalid")
        if decision["state"] == "EXTEND_ONCE_TO_25K":
            _atomic_json(root / "valid_unique.json", valid)
            _atomic_json(root / "resource_cap_receipt.json", decision)
            for target in fallback_checkpoint_targets(completed_now):
                state = run_generation_loop(
                    module,
                    input_graphs=dataset,
                    importance_args=importance_args,
                    teleport_probability=parameters.teleport_probability,
                    max_steps=target,
                    heads=parameters.heads,
                    initial_state=state,
                    on_step_complete=progress_callback,
                )
                checkpoint_evidence = _write_checkpoint(
                    module=module,
                    bridge=bridge,
                    loop_state=state,
                    parameters=parameters,
                    checkpoint_root=checkpoint_root,
                    handles=runtime_handles,
                    provenance=provenance,
                    scientific_argv=scientific_argv,
                    command_sha256=command_sha256,
                )
                if route_c_contract is not None:
                    fallback_valid = count_train_side_valid_unique(bridge)
                    _write_route_c_boundary(
                        root=root,
                        checkpoint_identity_path=checkpoint_identity_path,
                        route_c_contract=route_c_contract,
                        target=target,
                        valid_unique_rule_count=int(
                            fallback_valid["valid_unique_rule_count"]
                        ),
                        checkpoint_evidence=checkpoint_evidence,
                    )
                _progress(
                    root,
                    phase="FALLBACK_GENERATION",
                    completed_step=target,
                    cohort_count=len(cohort),
                    valid_unique_rule_count=None,
                    runtime_handles=runtime_handles,
                )
            valid = count_train_side_valid_unique(bridge)
            decision = resource_cap_decision(
                completed_step=M_FALLBACK_MAX,
                valid_unique_rule_count=int(valid["valid_unique_rule_count"]),
            )
        _atomic_json(root / "valid_unique.json", valid)
        _atomic_json(root / "resource_cap_receipt.json", decision)
        if decision["state"] != "STOP_AND_POSTPROCESS":
            _progress(
                root,
                phase="SCIENTIFIC_FAILED_INSUFFICIENT_VALID_RULES",
                completed_step=int(decision["m_effective"]),
                cohort_count=len(cohort),
                valid_unique_rule_count=int(valid["valid_unique_rule_count"]),
                runtime_handles=runtime_handles,
            )
            raise TasteComRecGCFullError(
                "SCIENTIFIC_FAILED_INSUFFICIENT_VALID_RULES"
            )
        _progress(
            root,
            phase="POSTPROCESS",
            completed_step=int(decision["m_effective"]),
            cohort_count=len(cohort),
            valid_unique_rule_count=int(valid["valid_unique_rule_count"]),
            runtime_handles=runtime_handles,
        )
        if runtime_handles.route_c_updater is None:
            common = _common_recourse_summary(
                modules=modules,
                module=module,
                bridge=bridge,
                source_graphs=source_graphs,
                adapter=adapter,
                parameters=parameters,
            )
        else:
            # The pinned summary intentionally requires ``type(raw) is dict``.
            # Route C keeps the official ordered sequence behind mmap-backed
            # mapping proxies during generation, so expose an iterator that
            # materializes exactly one immutable candidate row at a time at
            # this post-generation boundary.  This preserves order/frequency
            # and avoids rebuilding the full Python candidate list in memory.
            route_c_candidates = module.counterfactual_candidates
            module.counterfactual_candidates = (
                dict(route_c_candidates[index])
                for index in range(len(route_c_candidates))
            )
            try:
                common = _common_recourse_summary(
                    modules=modules,
                    module=module,
                    bridge=bridge,
                    source_graphs=source_graphs,
                    adapter=adapter,
                    parameters=parameters,
                )
            finally:
                module.counterfactual_candidates = route_c_candidates
        bridge_evidence = bridge.report()
        transition_evidence = runtime_handles.transition_map.audit()
        graph_state_evidence = runtime_handles.live_graph_state.runtime_diagnostics()
    # Release the hundreds-of-GiB live walk before independently reopening the
    # final archive.  Only a payload-valid checkpoint may become LATEST.
    effective_step = int(decision["m_effective"])
    reset_official_state(
        module,
        candidate_capacity=parameters.candidate_capacity,
        sample_size=parameters.sample_size,
    )
    bridge.records.clear()
    bridge.graph_collision_payloads.clear()
    bridge.lineage_occurrences.clear()
    del source_graphs, dataset, adapter, bridge, latest
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    from src.baselines.comrecgc.generation_checkpoint import (
        promote_generation_checkpoint,
    )

    promoted = promote_generation_checkpoint(
        checkpoint_root / f"step-{effective_step:012d}",
        expected_provenance=provenance,
        expected_scientific_argv=scientific_argv,
        expected_command_sha256=command_sha256,
        expected_total_steps=M_FALLBACK_MAX,
        expected_completed_step=effective_step,
    )
    checkpoint_receipt_path = checkpoint_root / f"checkpoint-{effective_step:06d}.json"
    checkpoint_receipt = json.loads(checkpoint_receipt_path.read_text(encoding="utf-8"))
    checkpoint_receipt.update(
        {
            "payload_reload_state": "PASS",
            "latest_promoted": True,
            "checkpoint_digest": promoted.checkpoint_digest,
            "promoted_at": _utc_now(),
        }
    )
    _atomic_json(checkpoint_receipt_path, checkpoint_receipt)
    inputs.revalidate()
    final = {
        "schema_version": "tastemolnet_t14_comrecgc_full_v1",
        "status": "PASS",
        "dataset": DATASET,
        "method": METHOD,
        "stage": STAGE,
        "cohort_manifest_sha256": _sha256_file(existing_manifest),
        "cohort_jsonl_sha256": _sha256_bytes(cohort_bytes),
        "cohort_count": len(cohort),
        "source_cohort": source_evidence,
        "valid_unique": valid,
        "resource_cap": decision,
        "bridge": bridge_evidence,
        "common_recourse": common,
        "bounded_runtime": {
            "transition_cache": transition_evidence,
            "live_graph_state": graph_state_evidence,
            "checkpoint_schema": (
                ROUTE_C_RUNTIME_STATE_SCHEMA
                if route_c_storage == "lowmemory"
                else RUNTIME_STATE_SCHEMA
            ),
            "checkpoint_identity_sha256": _sha256_file(
                checkpoint_identity_path
            ),
            "raw_neighbor_graphs_retained_unbounded": False,
            "process_peak_rss_bytes": _process_peak_rss_bytes(),
            "final_checkpoint_payload_reload_pass": True,
            "final_checkpoint_latest_promoted": True,
        },
        "same_frozen_three_class_gine": True,
        "rf_oracle_used": False,
        "train_loaded": True,
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "calibration_status": "NOT_EVALUATED",
        "held_out_test_status": "NOT_EVALUATED",
        "export_status": "NOT_EVALUATED",
        "paper_result_eligible": False,
        "method_cell_pass": False,
        "completed_at": _utc_now(),
    }
    if cohort_reconciliation_binding is not None:
        final["cohort_reconciliation"] = cohort_reconciliation_binding
    _atomic_json(root / "generation_manifest.json", final)
    _atomic_write(
        root / "GENERATION_PASS",
        f"{GENERATION_PASS_MARKER}\n".encode("utf-8"),
    )
    _atomic_json(
        root / "progress.json",
        {
            "schema_version": "tastemolnet_t14_progress_v1",
            "status": "PASS",
            "phase": "GENERATION_AND_TRAIN_POSTPROCESS_COMPLETE",
            "completed_step": int(decision["m_effective"]),
            "m_configured_max": M_MAX,
            "m_fallback_max": M_FALLBACK_MAX,
            "cohort_count": len(cohort),
            "valid_unique_rule_count": int(valid["valid_unique_rule_count"]),
            "pid": os.getpid(),
            "updated_at": _utc_now(),
        },
    )
    return final


def validate_t14_full_output(output_root: str | Path) -> dict[str, Any]:
    """Independently reopen the bounded full-generation closure."""

    root = Path(output_root)
    if not root.is_absolute() or Path(os.path.abspath(root)) != root:
        raise TasteComRecGCFullError("Taste T14 verification root must be normalized")
    required = {
        "GENERATION_PASS",
        "cohort.jsonl",
        "cohort_manifest.json",
        "generation_manifest.json",
        "progress.json",
        "resource_cap_receipt.json",
        "valid_unique.json",
        "checkpoint_identity.json",
    }
    if not root.is_dir() or not required.issubset(
        {path.name for path in root.iterdir()}
    ):
        raise TasteComRecGCFullError("Taste T14 terminal closure is incomplete")
    for name in required:
        info = (root / name).lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise TasteComRecGCFullError(f"Taste T14 {name} is not a regular file")
    if (root / "GENERATION_PASS").read_bytes() != (
        f"{GENERATION_PASS_MARKER}\n".encode("utf-8")
    ):
        raise TasteComRecGCFullError("Taste T14 generation marker changed")
    manifest = json.loads((root / "generation_manifest.json").read_text("utf-8"))
    cohort_manifest = json.loads((root / "cohort_manifest.json").read_text("utf-8"))
    resource = json.loads((root / "resource_cap_receipt.json").read_text("utf-8"))
    valid = json.loads((root / "valid_unique.json").read_text("utf-8"))
    progress = json.loads((root / "progress.json").read_text("utf-8"))
    checkpoint_identity = json.loads(
        (root / "checkpoint_identity.json").read_text("utf-8")
    )
    cohort_bytes = (root / "cohort.jsonl").read_bytes()
    cohort_manifest_bytes = (root / "cohort_manifest.json").read_bytes()
    reconciliation_binding = _validate_cohort_reconciliation_evidence(
        root,
        frozen_cohort_bytes=cohort_bytes,
        frozen_manifest_bytes=cohort_manifest_bytes,
    )
    effective = resource.get("m_effective") if type(resource) is dict else None
    latest = (
        root / "checkpoints" / f"step-{effective:012d}"
        if type(effective) is int
        else root
    )
    receipt_path = (
        root / "checkpoints" / f"checkpoint-{effective:06d}.json"
        if type(effective) is int
        else root
    )
    checkpoint_validation = None
    checkpoint_receipt = None
    if (
        type(checkpoint_identity) is dict
        and type(checkpoint_identity.get("provenance")) is dict
        and type(checkpoint_identity.get("scientific_argv")) is list
        and type(checkpoint_identity.get("command_sha256")) is str
        and latest.is_dir()
        and receipt_path.is_file()
    ):
        from src.baselines.comrecgc.generation_checkpoint import (
            validate_generation_checkpoint,
        )

        checkpoint_validation = validate_generation_checkpoint(
            latest,
            expected_provenance=checkpoint_identity["provenance"],
            expected_scientific_argv=checkpoint_identity["scientific_argv"],
            expected_command_sha256=checkpoint_identity["command_sha256"],
            expected_total_steps=M_FALLBACK_MAX,
            expected_completed_step=effective,
        )
        checkpoint_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    bounded_runtime = manifest.get("bounded_runtime") if type(manifest) is dict else None
    if (
        type(manifest) is not dict
        or manifest.get("schema_version") != "tastemolnet_t14_comrecgc_full_v1"
        or manifest.get("status") != "PASS"
        or manifest.get("stage") != STAGE
        or manifest.get("train_loaded") is not True
        or manifest.get("validation_loaded") is not False
        or manifest.get("calibration_loaded") is not False
        or manifest.get("test_loaded") is not False
        or manifest.get("rf_oracle_used") is not False
        or manifest.get("calibration_status") != "NOT_EVALUATED"
        or manifest.get("held_out_test_status") != "NOT_EVALUATED"
        or manifest.get("export_status") != "NOT_EVALUATED"
        or manifest.get("paper_result_eligible") is not False
        or manifest.get("method_cell_pass") is not False
        or manifest.get("cohort_manifest_sha256") != _sha256_file(root / "cohort_manifest.json")
        or manifest.get("cohort_jsonl_sha256") != _sha256_bytes(cohort_bytes)
        or manifest.get("cohort_reconciliation") != reconciliation_binding
        or manifest.get("resource_cap") != resource
        or manifest.get("valid_unique") != valid
        or type(bounded_runtime) is not dict
        or bounded_runtime.get("checkpoint_schema")
        != checkpoint_identity.get("provenance", {}).get("runtime_state_schema")
        or bounded_runtime.get("raw_neighbor_graphs_retained_unbounded") is not False
        or bounded_runtime.get("checkpoint_identity_sha256")
        != _sha256_file(root / "checkpoint_identity.json")
        or type(bounded_runtime.get("transition_cache")) is not dict
        or bounded_runtime["transition_cache"].get("patch")
        != "compact_transition_action_replay_lru_v1"
        or bounded_runtime["transition_cache"].get("scientific_parameters_changed")
        is not False
        or type(bounded_runtime.get("live_graph_state")) is not dict
        or bounded_runtime["live_graph_state"].get("unresolved_lookups") != 0
        or type(cohort_manifest) is not dict
        or cohort_manifest.get("status") != "PASS"
        or cohort_manifest.get("policy") != COHORT_POLICY
        or cohort_manifest.get("cohort_jsonl_sha256") != _sha256_bytes(cohort_bytes)
        or resource.get("state") != "STOP_AND_POSTPROCESS"
        or (
            effective not in {M_MAX, M_FALLBACK_MAX}
            and not (
                effective in {12_500, 15_000, 17_500}
                and resource.get("early_stop_used") is True
                and resource.get("resource_cap_used") is False
                and resource.get("stop_reason")
                == "TRAIN_SIDE_CONVERGENCE_TWO_CONSECUTIVE_WINDOWS"
                and type(resource.get("convergence_receipt")) is str
                and type(resource.get("convergence_receipt_sha256")) is str
            )
        )
        or valid.get("valid_unique_rule_count", 0) < MIN_VALID_UNIQUE_RULES
        or progress.get("status") != "PASS"
        or progress.get("completed_step") != effective
        or checkpoint_identity.get("schema_version")
        != CHECKPOINT_PROVENANCE_SCHEMA
        or checkpoint_identity.get("raw_neighbor_graphs_retained_unbounded") is not False
        or checkpoint_identity.get("transition_expanded_capacity")
        != TRANSITION_EXPANDED_CAPACITY
        or checkpoint_validation is None
        or checkpoint_validation.completed_step != effective
        or type(checkpoint_receipt) is not dict
        or checkpoint_receipt.get("schema_version")
        != "tastemolnet_t14_checkpoint_v2"
        or checkpoint_receipt.get("checkpoint_dir") != str(latest)
        or checkpoint_receipt.get("checkpoint_digest")
        != checkpoint_validation.checkpoint_digest
        or checkpoint_receipt.get("bounded_transition_state") is not True
        or checkpoint_receipt.get("authoritative_graph_store_snapshot") is not True
    ):
        raise TasteComRecGCFullError("Taste T14 terminal science closure changed")
    inventory = {
        str(path.relative_to(root)): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    return {
        "schema_version": "tastemolnet_t14_independent_verification_v1",
        "status": "PASS",
        "marker": GENERATION_PASS_MARKER,
        "output_root": str(root),
        "m_effective": effective,
        "valid_unique_rule_count": valid["valid_unique_rule_count"],
        "inventory_sha256": _sha256_bytes(_canonical_bytes(inventory)),
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "paper_result_eligible": False,
        "method_cell_pass": False,
        "verified_at": _utc_now(),
    }


__all__ = [
    "CHECK_INTERVAL",
    "COHORT_POLICY",
    "M_FALLBACK_MAX",
    "M_MAX",
    "MIN_VALID_UNIQUE_RULES",
    "GENERATION_PASS_MARKER",
    "STAGE",
    "TasteComRecGCFullError",
    "TasteComRecGCFullBridge",
    "TasteComRecGCFullParameters",
    "VALID_UNIQUE_POLICY",
    "build_full_train_correct_source_cohort",
    "count_train_side_valid_unique",
    "fallback_checkpoint_targets",
    "resource_cap_decision",
    "run_t14_full",
    "validate_t14_full_output",
]
