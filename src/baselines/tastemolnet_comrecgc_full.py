"""TasteMolNet ComRecGC full generation with a train-only frozen cohort.

This module is intentionally dataset specific.  It extends the already
verified T9 native bridge to the preregistered T14 resource cap without
opening validation, calibration, or test payloads during generation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence

from src.baselines.tastemolnet_comrecgc_smoke import (
    SOURCE_LABEL,
    TasteComRecGCMulticlassBridge,
    TasteComRecGCSmokeError,
    _common_recourse_summary,
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
CHECKPOINT_PROVENANCE_SCHEMA = "tastemolnet_t14_checkpoint_provenance_v1"
TRANSITION_EXPANDED_CAPACITY = 5
PROGRESS_INTERVAL = 100


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
            or self.checkpoint_step % CHECK_INTERVAL != 0
            or not CHECK_INTERVAL <= self.checkpoint_step <= M_FALLBACK_MAX
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


def _reopen_existing_resume_cohort(
    *,
    cohort_bytes: bytes,
    cohort_manifest: Mapping[str, Any],
    replay_rows: Sequence[Mapping[str, Any]],
    replay_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], bytes]:
    """Keep a validated pre-checkpoint cohort while replaying its membership.

    ``source_probability`` is diagnostic once the discrete frozen-GINE source
    membership has been selected.  A new CUDA process may reproduce that
    probability with low-bit drift.  Resume therefore retains the committed
    bytes only when every scientific/discrete field and row order replays
    exactly; any member, graph, label, split, or authority change still fails.
    """

    try:
        rows = [
            json.loads(line)
            for line in cohort_bytes.decode("utf-8").splitlines()
            if line
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteComRecGCFullError("Taste T14 frozen cohort is unreadable") from exc
    if (
        not rows
        or any(type(row) is not dict for row in rows)
        or _cohort_lines(rows) != cohort_bytes
        or type(cohort_manifest) is not dict
        or set(cohort_manifest) != set(replay_manifest)
        or cohort_manifest.get("cohort_jsonl_sha256")
        != _sha256_bytes(cohort_bytes)
    ):
        raise TasteComRecGCFullError("Taste T14 frozen cohort closure changed")
    manifest_without_payload = {
        key: value
        for key, value in cohort_manifest.items()
        if key != "cohort_jsonl_sha256"
    }
    replay_without_payload = {
        key: value
        for key, value in replay_manifest.items()
        if key != "cohort_jsonl_sha256"
    }
    if manifest_without_payload != replay_without_payload or len(rows) != len(
        replay_rows
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
    for frozen, replay in zip(rows, replay_rows, strict=True):
        if set(frozen) != expected_fields or set(replay) != expected_fields:
            raise TasteComRecGCFullError("Taste T14 frozen cohort row changed")
        frozen_probability = frozen.get("source_probability")
        if (
            isinstance(frozen_probability, bool)
            or not isinstance(frozen_probability, (int, float))
            or not 0.0 <= float(frozen_probability) <= 1.0
            or any(
                frozen[field] != replay[field]
                for field in expected_fields - {"source_probability"}
            )
        ):
            raise TasteComRecGCFullError("Taste T14 cohort changed on resume")
    return rows, dict(cohort_manifest), cohort_bytes


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
        "runtime_state_schema": RUNTIME_STATE_SCHEMA,
        "transition_cache_policy": "compact_transition_action_replay_lru_v1",
        "graph_state_policy": "authoritative_backing_live_graph_resolution_v2",
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
    live = LiveGraphState(
        module,
        module.graph_map,
        store_path=graph_store_path,
        seed=seed,
    )
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
    module.move_to_next_graph = live.wrap_move(transitions.wrap_move(original_move))

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
            )
            completed_normally = True
    finally:
        module.move_to_next_graph = original_move
        module.graph_map = dict(live.graph_map)
        module.transitions = {}
        if hasattr(module, "comrecgc_live_graph_state"):
            delattr(module, "comrecgc_live_graph_state")
        try:
            live.close()
        finally:
            if not completed_normally:
                # Keep teardown bounded after a science exception.  The latest
                # atomically published checkpoint remains the resume authority.
                module.graph_map = dict(live.graph_map)
            module.transitions = original_transitions


def _checkpoint_algorithm_state(
    *,
    module: Any,
    bridge: TasteComRecGCMulticlassBridge,
    loop_state: Any,
    handles: _BoundedRuntimeHandles,
) -> dict[str, Any]:
    from src.baselines.comrecgc.generation_loop import snapshot_official_state

    transition_state = handles.transition_map.export_checkpoint_state()
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
    return {
        "schema_version": RUNTIME_STATE_SCHEMA,
        "loop_state": loop_state.to_checkpoint_state(),
        "official_state": snapshot_official_state(module),
        "transition_state": transition_state,
        "live_graph_state": live_state,
        "bridge_state": bridge.checkpoint_state(),
    }


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
    if state.get("schema_version") != RUNTIME_STATE_SCHEMA:
        raise TasteComRecGCFullError("Taste T14 runtime checkpoint schema changed")
    handles.transition_map.restore_checkpoint_state(state["transition_state"])
    handles.live_graph_state.restore_checkpoint_state(state["live_graph_state"])
    bridge.restore_checkpoint_state(state["bridge_state"])
    loop_state = GenerationLoopState.from_checkpoint_state(state["loop_state"])
    if (
        handles.transition_map.move_count != loop_state.completed_step
        or handles.live_graph_state.move_count != loop_state.completed_step
        or loop_state.completed_step != loaded.completed_step
    ):
        raise TasteComRecGCFullError("Taste T14 restored move counters changed")
    # RNG restoration is deliberately the final mutation before next_step.
    restore_rng_state(loaded.rng_state)
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
    step_parameters = replace(parameters, checkpoint_step=completed)
    step_parameters.validate()
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
        "written_at": _utc_now(),
    }
    _atomic_json(checkpoint_root / f"checkpoint-{completed:06d}.json", evidence)
    return evidence


def _load_latest_checkpoint(
    checkpoint_root: Path,
    *,
    parameters: TasteComRecGCFullParameters,
    provenance: Mapping[str, str],
    scientific_argv: Sequence[str],
    command_sha256: str,
) -> Any | None:
    from src.baselines.comrecgc.generation_checkpoint import load_generation_checkpoint

    if not (checkpoint_root / "LATEST").is_file():
        return None
    loaded = load_generation_checkpoint(
        checkpoint_root,
        expected_provenance=provenance,
        expected_scientific_argv=scientific_argv,
        expected_command_sha256=command_sha256,
        expected_total_steps=M_FALLBACK_MAX,
    )
    completed = loaded.completed_step
    if (
        completed % CHECK_INTERVAL != 0
        or not CHECK_INTERVAL <= completed <= M_FALLBACK_MAX
        or loaded.algorithm_state.get("schema_version") != RUNTIME_STATE_SCHEMA
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
    if existing_cohort.exists() or existing_manifest.exists():
        if (
            existing_cohort.read_bytes() != cohort_bytes
            or json.loads(existing_manifest.read_text(encoding="utf-8"))
            != cohort_manifest
        ):
            raise TasteComRecGCFullError("Taste T14 cohort changed on resume")
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
    )
    checkpoint_identity = {
        "schema_version": CHECKPOINT_PROVENANCE_SCHEMA,
        "status": "FROZEN",
        "provenance": provenance,
        "scientific_argv": list(scientific_argv),
        "command_sha256": command_sha256,
        "total_steps": M_FALLBACK_MAX,
        "checkpoint_interval": CHECK_INTERVAL,
        "transition_expanded_capacity": TRANSITION_EXPANDED_CAPACITY,
        "raw_neighbor_graphs_retained_unbounded": False,
    }
    checkpoint_identity_path = root / "checkpoint_identity.json"
    if checkpoint_identity_path.exists():
        if json.loads(checkpoint_identity_path.read_text(encoding="utf-8")) != (
            checkpoint_identity
        ):
            raise TasteComRecGCFullError(
                "Taste T14 checkpoint identity changed on resume"
            )
    else:
        _atomic_json(checkpoint_identity_path, checkpoint_identity)
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
        if runtime_state.get("schema_version") != RUNTIME_STATE_SCHEMA:
            raise TasteComRecGCFullError("Taste T14 checkpoint runtime changed")
        restore_official_state(module, runtime_state["official_state"])
    graph_store_path = _prepare_runtime_store(root, latest)
    with _bounded_t14_runtime(
        module=module,
        bridge=bridge,
        graph_store_path=graph_store_path,
        seed=parameters.seed,
        expanded_capacity=TRANSITION_EXPANDED_CAPACITY,
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
            if step % PROGRESS_INTERVAL == 0:
                _progress(
                    root,
                    phase="GENERATION",
                    completed_step=step,
                    cohort_count=len(cohort),
                    runtime_handles=runtime_handles,
                )

        for target in range(
            ((completed // CHECK_INTERVAL) + 1) * CHECK_INTERVAL,
            M_MAX + 1,
            CHECK_INTERVAL,
        ):
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
            _write_checkpoint(
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
        if state is None:
            raise TasteComRecGCFullError("Taste T14 generation produced no state")
        completed_now = int(state.completed_step)
        valid = count_train_side_valid_unique(bridge)
        if completed_now == M_MAX or completed_now == M_FALLBACK_MAX:
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
                _write_checkpoint(
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
        common = _common_recourse_summary(
            modules=modules,
            module=module,
            bridge=bridge,
            source_graphs=source_graphs,
            adapter=adapter,
            parameters=parameters,
        )
        bridge_evidence = bridge.report()
        transition_evidence = runtime_handles.transition_map.audit()
        graph_state_evidence = runtime_handles.live_graph_state.runtime_diagnostics()
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
            "checkpoint_schema": RUNTIME_STATE_SCHEMA,
            "checkpoint_identity_sha256": _sha256_file(
                checkpoint_identity_path
            ),
            "raw_neighbor_graphs_retained_unbounded": False,
            "process_peak_rss_bytes": _process_peak_rss_bytes(),
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
        or manifest.get("resource_cap") != resource
        or manifest.get("valid_unique") != valid
        or type(bounded_runtime) is not dict
        or bounded_runtime.get("checkpoint_schema") != RUNTIME_STATE_SCHEMA
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
        or effective not in {M_MAX, M_FALLBACK_MAX}
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
