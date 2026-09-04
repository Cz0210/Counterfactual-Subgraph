"""Exact prefix fork and parity gates for the T12 accelerated arm.

This module owns no scheduler and never touches the reference writer.  It
copies one already committed step-250 prefix into a fresh root, rewrites only
the three absolute storage-root fields in the checkpoint, and provides the
fail-closed comparisons required before that branch can be promoted.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any, Mapping


REFERENCE_STEP = 250
PARITY_STEP = 500
RELOAD_STEP = 510
REFERENCE_MANIFEST_SHA256 = (
    "8bfcb25d8e081545e419541ffc1fe3c7d3b2b4e4a0cec6644414700532614dac"
)
REFERENCE_PAYLOAD_SHA256 = (
    "8329c85e513cda1acc32281b963ddea52877a7603e67ef3e1e0a5f93f60cf4f3"
)
REFERENCE_STATE_SHA256 = (
    "e7cc0411528b138712e7d4daec8eba65eddd147802891cc9cbd5f3030c58f6d7"
)
REFERENCE_RNG_SHA256 = (
    "020ceb03b31f4c2c7a069e9ef4c0cf60bfc6a2351c1efffc425256989232a227"
)
REFERENCE_FIRST_SEEN_PREFIX_SHA256 = (
    "0b3bce221ba382dde095e931e3490b48042906122c72c9e6feb253a98c368ac4"
)
REFERENCE_FIRST_SEEN_COMMITTED_BYTES = 779_204_874
REFERENCE_HISTORY_PREFIX_SHA256 = (
    "4ee5500064bcb2ea634999622ec55bd2d9ebd15e03b89f7e2ee6076946d00671"
)
REFERENCE_HISTORY_COMMITTED_BYTES = 242_191_887


class T12AcceleratedError(RuntimeError):
    """The T12 prefix, parity, or continuation binding is unsafe."""


def file_sha256(path: Path, *, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = limit
    with path.open("rb", buffering=0) as stream:
        while remaining is None or remaining:
            size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            chunk = stream.read(size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    if remaining not in (None, 0):
        raise T12AcceleratedError(f"file ended before the bound prefix: {path}")
    return digest.hexdigest()


def _physical_file(path: Path, *, expected_bytes: int | None = None) -> os.stat_result:
    if not path.is_absolute() or path.is_symlink():
        raise T12AcceleratedError(f"T12 evidence is not one physical file: {path}")
    try:
        info = path.stat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise T12AcceleratedError(f"T12 evidence is unreadable: {path}") from exc
    if resolved != path or not stat.S_ISREG(info.st_mode):
        raise T12AcceleratedError(f"T12 evidence is not one physical file: {path}")
    if expected_bytes is not None and info.st_size != expected_bytes:
        raise T12AcceleratedError(f"T12 evidence byte count changed: {path}")
    return info


def _json(path: Path) -> dict[str, Any]:
    _physical_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T12AcceleratedError(f"T12 JSON evidence is unreadable: {path}") from exc
    if type(value) is not dict:
        raise T12AcceleratedError(f"T12 JSON evidence is not an object: {path}")
    return value


def validate_mut_gpu0_release_receipt(path: Path) -> dict[str, Any]:
    """Require the exact physical Mut release receipt bound by the task spec.

    The receipt may be created after the accelerated task spec is sealed, but
    its absolute path is part of that spec.  The science owner calls this
    validator itself before taking the GPU lease or creating an output root;
    the shell launcher is therefore not a security or scheduling boundary.
    """

    value = _json(path)
    if (
        value.get("status") != "PASS"
        or value.get("gpu_index") != 0
        or value.get("gpu_released") is not True
    ):
        raise T12AcceleratedError("Mut has not released GPU0")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "status": "PASS",
        "gpu_index": 0,
        "gpu_released": True,
    }


def validate_reference_step250(
    *,
    task_spec_path: Path,
    expected_manifest_sha256: str = REFERENCE_MANIFEST_SHA256,
    expected_payload_sha256: str = REFERENCE_PAYLOAD_SHA256,
) -> dict[str, Any]:
    """Validate the sealed reference spec, checkpoint, and first-seen receipt."""

    from src.utils.main_ready_task_specs import load_spec

    spec = load_spec(task_spec_path)
    if spec.get("task_kind") != "T12_REFERENCE_ACCELERATED_PARITY_AND_FULL":
        raise T12AcceleratedError("T12 reference task kind changed")
    contract = spec.get("science_contract")
    if not isinstance(contract, Mapping):
        raise T12AcceleratedError("T12 reference science contract is absent")
    root = Path(str(contract.get("reference_root")))
    manifest_path = Path(str(contract.get("reference_checkpoint_250")))
    if manifest_path != root / "checkpoints/checkpoint-00000250.manifest.json":
        raise T12AcceleratedError("T12 reference checkpoint-250 path changed")
    manifest = _json(manifest_path)
    if (
        manifest.get("status") != "COMMITTED"
        or manifest.get("checkpoint_cursor") != REFERENCE_STEP
        or manifest.get("total_steps") != RELOAD_STEP
        or manifest.get("purpose") != "production"
        or manifest.get("payload_file") != "checkpoint-00000250.pt"
        or manifest.get("state_sha256") != REFERENCE_STATE_SHA256
        or manifest.get("rng_sha256") != REFERENCE_RNG_SHA256
        or file_sha256(manifest_path) != expected_manifest_sha256
    ):
        raise T12AcceleratedError("T12 checkpoint-250 manifest changed")
    payload = manifest_path.parent / str(manifest["payload_file"])
    _physical_file(payload, expected_bytes=int(manifest["payload_bytes"]))
    if (
        manifest.get("payload_sha256") != expected_payload_sha256
        or file_sha256(payload) != expected_payload_sha256
    ):
        raise T12AcceleratedError("T12 checkpoint-250 payload bytes changed")
    receipt_path = root / "generation_receipt_00000250.json"
    receipt = _json(receipt_path)
    if (
        receipt.get("status") != "GENERATION_CHECKPOINT_COMMITTED"
        or receipt.get("checkpoint_cursor") != REFERENCE_STEP
        or receipt.get("checkpoint_manifest") != str(manifest_path)
        or receipt.get("checkpoint_manifest_sha256") != expected_manifest_sha256
        or receipt.get("source_cohort_sha256")
        != contract.get("source_cohort_sha256")
        or receipt.get("generation_token") != contract.get("generation_token")
        or receipt.get("calibration_loaded") is not False
        or receipt.get("test_loaded") is not False
    ):
        raise T12AcceleratedError("T12 checkpoint-250 receipt changed")
    bridge = receipt.get("bridge_report")
    history = bridge.get("history") if isinstance(bridge, Mapping) else None
    first_seen = (
        history.get("first_seen_embedding_store")
        if isinstance(history, Mapping)
        else None
    )
    history_segments = history.get("segments") if isinstance(history, Mapping) else None
    first_segments = (
        first_seen.get("segments") if isinstance(first_seen, Mapping) else None
    )
    if (
        not isinstance(history, Mapping)
        or not isinstance(first_seen, Mapping)
        or history.get("history_root") != str(root / "bridge_history")
        or first_seen.get("store_root")
        != str(root / "bridge_history/first-seen-embeddings")
        or type(history_segments) is not list
        or len(history_segments) != 1
        or type(first_segments) is not list
        or len(first_segments) != 1
        or history_segments[0].get("committed_bytes")
        != REFERENCE_HISTORY_COMMITTED_BYTES
        or history_segments[0].get("committed_prefix_sha256")
        != REFERENCE_HISTORY_PREFIX_SHA256
        or first_segments[0].get("committed_bytes")
        != REFERENCE_FIRST_SEEN_COMMITTED_BYTES
        or first_segments[0].get("committed_prefix_sha256")
        != REFERENCE_FIRST_SEEN_PREFIX_SHA256
        or first_seen.get("model_sha256")
        != spec["input_hashes"].get("t3_calibrated_gine")
        or first_seen.get("raw_bytes_authoritative") is not True
        or first_seen.get("append_only_hash_chain") is not True
    ):
        raise T12AcceleratedError("T12 first-seen/history receipt binding changed")
    history_path = root / "bridge_history" / history_segments[0]["segment_file"]
    first_seen_path = (
        root
        / "bridge_history/first-seen-embeddings"
        / first_segments[0]["segment_file"]
    )
    _physical_file(history_path, expected_bytes=REFERENCE_HISTORY_COMMITTED_BYTES)
    _physical_file(
        first_seen_path, expected_bytes=REFERENCE_FIRST_SEEN_COMMITTED_BYTES
    )
    if (
        file_sha256(history_path) != REFERENCE_HISTORY_PREFIX_SHA256
        or file_sha256(first_seen_path) != REFERENCE_FIRST_SEEN_PREFIX_SHA256
    ):
        raise T12AcceleratedError("T12 first-seen/history prefix bytes changed")
    return {
        "schema_version": "tastemolnet_t12_reference_step250_evidence_v1",
        "status": "PASS",
        "task_spec": str(task_spec_path),
        "task_spec_sha256": file_sha256(task_spec_path),
        "reference_root": str(root),
        "checkpoint_manifest": str(manifest_path),
        "checkpoint_manifest_sha256": expected_manifest_sha256,
        "checkpoint_payload": str(payload),
        "checkpoint_payload_sha256": expected_payload_sha256,
        "checkpoint_state_sha256": REFERENCE_STATE_SHA256,
        "checkpoint_rng_sha256": REFERENCE_RNG_SHA256,
        "generation_receipt": str(receipt_path),
        "generation_receipt_sha256": file_sha256(receipt_path),
        "reference_attempt_id": manifest["attempt_id"],
        "generation_token": manifest["generation_token"],
        "first_seen_committed_bytes": REFERENCE_FIRST_SEEN_COMMITTED_BYTES,
        "first_seen_prefix_sha256": REFERENCE_FIRST_SEEN_PREFIX_SHA256,
        "first_seen_segment": str(first_seen_path),
        "first_seen_record_count": first_seen["record_count"],
        "first_seen_chain_sha256": first_seen["chain_head"],
        "history_committed_bytes": REFERENCE_HISTORY_COMMITTED_BYTES,
        "history_prefix_sha256": REFERENCE_HISTORY_PREFIX_SHA256,
        "history_segment": str(history_path),
        "history_observation_count": history["observation_count"],
        "history_chain_sha256": history["chain_head"],
        "calibration_loaded": False,
        "test_loaded": False,
    }


def _copy_exact(source: Path, target: Path, *, expected_bytes: int) -> dict[str, Any]:
    _physical_file(source, expected_bytes=expected_bytes)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise T12AcceleratedError(f"T12 fork target already exists: {target}")
    source_sha = file_sha256(source)
    try:
        with source.open("rb", buffering=0) as reader, target.open("xb", buffering=0) as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    target_sha = file_sha256(target)
    if target.stat().st_size != expected_bytes or target_sha != source_sha:
        target.unlink(missing_ok=True)
        raise T12AcceleratedError("T12 fork copy changed immutable prefix bytes")
    return {
        "source": str(source),
        "target": str(target),
        "bytes": expected_bytes,
        "sha256": target_sha,
    }


def _clone_manifest_segments(
    *, source_root: Path, target_root: Path, segments: Any
) -> list[dict[str, Any]]:
    if type(segments) is not list or not segments:
        raise T12AcceleratedError("T12 checkpoint has no committed prefix segments")
    copied: list[dict[str, Any]] = []
    for row in segments:
        if type(row) is not dict:
            raise T12AcceleratedError("T12 prefix segment manifest is malformed")
        name = row.get("segment_file")
        size = row.get("committed_bytes")
        if (
            type(name) is not str
            or Path(name).name != name
            or type(size) is not int
            or size <= 0
        ):
            raise T12AcceleratedError("T12 prefix segment binding is malformed")
        copied.append(
            _copy_exact(source_root / name, target_root / name, expected_bytes=size)
        )
    return copied


def fork_step250_prefix(
    *,
    source_root: Path,
    target_root: Path,
    source_checkpoint_manifest: Path,
    expected_identity: Mapping[str, Any],
    torch: Any,
) -> dict[str, Any]:
    """Fork step 250 into a fresh root without changing scientific state.

    Segment file names and record bytes are retained.  Only their absolute
    parent paths change, so the checkpoint state hash is recomputed while the
    RNG, graph registry, frequencies, lineage, and append-only chain heads stay
    byte-identical.
    """

    from src.baselines.tastemolnet_gcf_full_resume import (
        reopen_checkpoint,
        write_checkpoint,
    )
    from src.baselines.tastemolnet_gcf_smoke import _semantic_sha256

    if target_root.exists() or target_root.is_symlink():
        raise T12AcceleratedError("T12 accelerated output root must be fresh")
    source_root = source_root.resolve(strict=True)
    target_root = Path(os.path.abspath(target_root))
    payload = reopen_checkpoint(
        source_checkpoint_manifest,
        expected_identity=expected_identity,
        torch=torch,
    )
    if payload["identity"]["checkpoint_cursor"] != REFERENCE_STEP:
        raise T12AcceleratedError("T12 accelerated fork is not rooted at step 250")
    cloned = copy.deepcopy(payload)
    history = cloned["state"]["bridge"]["history"]
    first_seen = history["first_seen_embedding_store"]
    transitions = cloned["state"]["official"]["transitions"]
    source_history = source_root / "bridge_history"
    target_history = target_root / "bridge_history"
    source_first_seen = source_history / "first-seen-embeddings"
    target_first_seen = target_history / "first-seen-embeddings"
    source_transitions = source_root / "transition_store"
    target_transitions = target_root / "transition_store"
    if (
        history.get("history_root") != str(source_history)
        or not isinstance(first_seen, Mapping)
        or first_seen.get("store_root") != str(source_first_seen)
        or transitions.get("root") != str(source_transitions)
    ):
        raise T12AcceleratedError("T12 checkpoint storage roots differ from reference")
    target_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    copied: list[dict[str, Any]] = []
    try:
        for name in ("run_identity.json", "cohort_manifest.json", "cohort.jsonl"):
            source = source_root / name
            copied.append(
                _copy_exact(source, target_root / name, expected_bytes=source.stat().st_size)
            )
        copied.extend(
            _clone_manifest_segments(
                source_root=source_history,
                target_root=target_history,
                segments=history["segments"],
            )
        )
        copied.extend(
            _clone_manifest_segments(
                source_root=source_first_seen,
                target_root=target_first_seen,
                segments=first_seen["segments"],
            )
        )
        copied.extend(
            _clone_manifest_segments(
                source_root=source_transitions,
                target_root=target_transitions,
                segments=transitions["segments"],
            )
        )
        history["history_root"] = str(target_history)
        first_seen["store_root"] = str(target_first_seen)
        transitions["root"] = str(target_transitions)
        old_state_sha = cloned["state_sha256"]
        cloned["state_sha256"] = _semantic_sha256(cloned["state"])
        manifest = write_checkpoint(
            target_root / "checkpoints", cloned, torch=torch
        )
        result = {
            "schema_version": "tastemolnet_t12_step250_prefix_fork_v1",
            "status": "PASS",
            "source_root": str(source_root),
            "target_root": str(target_root),
            "source_checkpoint_manifest": str(source_checkpoint_manifest),
            "source_checkpoint_manifest_sha256": file_sha256(source_checkpoint_manifest),
            "target_checkpoint_manifest": str(manifest),
            "target_checkpoint_manifest_sha256": file_sha256(manifest),
            "source_state_sha256": old_state_sha,
            "target_state_sha256": cloned["state_sha256"],
            "rng_sha256": cloned["rng_sha256"],
            "scientific_state_mutated": False,
            "storage_roots_relocated": True,
            "first_seen_embedding_record_bytes_copied_exactly": True,
            "copied_files": copied,
        }
        receipt = target_root / "step250_fork_receipt.json"
        from src.utils.main_ready_task_specs import atomic_json

        atomic_json(receipt, result)
        return {**result, "receipt": str(receipt), "receipt_sha256": file_sha256(receipt)}
    except BaseException:
        # A failed, unpublished fresh fork is not a resumable science root.
        # Preserve it for diagnosis rather than deleting user data.
        raise


def checkpoint_science_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only file-layout details from one checkpoint payload."""

    value = copy.deepcopy(dict(payload))
    value.pop("state_sha256", None)
    history = value["state"]["bridge"]["history"]
    first_seen = history["first_seen_embedding_store"]
    transitions = value["state"]["official"]["transitions"]
    history["history_root"] = "<BOUND_HISTORY_ROOT>"
    first_seen["store_root"] = "<BOUND_FIRST_SEEN_ROOT>"
    transitions["root"] = "<BOUND_TRANSITION_ROOT>"
    # Segment IDs/header hashes are transport framing.  Chain heads, counts,
    # active order, and every scientific record remain in the projection.
    history["segments"] = [
        {
            key: row[key]
            for key in (
                "anchor_sequence",
                "anchor_chain_head",
                "record_count",
                "terminal_sequence",
                "terminal_chain_head",
            )
        }
        for row in history["segments"]
    ]
    first_seen["segments"] = [
        {
            key: row[key]
            for key in (
                "anchor_sequence",
                "anchor_chain_head",
                "record_count",
                "terminal_sequence",
                "terminal_chain_head",
            )
        }
        for row in first_seen["segments"]
    ]
    transitions["segments"] = [
        {
            key: row[key]
            for key in ("segment_index", "event_count", "final_chain_sha256")
        }
        for row in transitions["segments"]
    ]
    return value


def compare_checkpoint_payloads(
    *, reference: Mapping[str, Any], accelerated: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare complete endpoint checkpoint state at step 500 or 510.

    This comparison is deliberately not a claim of per-step 251--500 parity:
    the current checkpoint schema does not retain every selected action,
    pre-softmax logit, and NeuroSED distance for every step.  A matching
    endpoint is useful diagnostic evidence, but it cannot authorize promotion
    to the 10k production schedule.
    """

    from src.baselines.tastemolnet_gcf_smoke import _semantic_sha256

    left = checkpoint_science_projection(reference)
    right = checkpoint_science_projection(accelerated)
    cursor = left["identity"].get("checkpoint_cursor")
    if cursor not in (PARITY_STEP, RELOAD_STEP) or right["identity"].get(
        "checkpoint_cursor"
    ) != cursor:
        raise T12AcceleratedError("T12 parity checkpoints are at different cursors")
    components = {
        "identity": (left["identity"], right["identity"]),
        "rng": (left["rng"], right["rng"]),
        "official": (left["state"]["official"], right["state"]["official"]),
        "bridge": (left["state"]["bridge"], right["state"]["bridge"]),
        "adapter": (left["state"]["adapter"], right["state"]["adapter"]),
        "action_counts": (
            left["state"]["action_counts"],
            right["state"]["action_counts"],
        ),
        "current_graph_identity": (
            left["state"]["current_graph_identity"],
            right["state"]["current_graph_identity"],
        ),
    }
    comparison = {
        name: {
            "reference_sha256": _semantic_sha256(values[0]),
            "accelerated_sha256": _semantic_sha256(values[1]),
            "exact": _semantic_sha256(values[0]) == _semantic_sha256(values[1]),
        }
        for name, values in components.items()
    }
    mismatches = [name for name, row in comparison.items() if not row["exact"]]
    if mismatches:
        raise T12AcceleratedError(
            "T12 accelerated checkpoint scientific mismatch: " + ",".join(mismatches)
        )
    first_left = left["state"]["bridge"]["history"][
        "first_seen_embedding_store"
    ]
    first_right = right["state"]["bridge"]["history"][
        "first_seen_embedding_store"
    ]
    return {
        "schema_version": "tastemolnet_t12_accelerated_checkpoint_parity_v1",
        "status": "ENDPOINT_STATE_MATCH",
        "comparison_scope": "ENDPOINT_CHECKPOINT_STATE_ONLY",
        "checkpoint_cursor": cursor,
        "components": comparison,
        "first_seen_record_count": first_left["record_count"],
        "first_seen_chain_sha256": first_left["chain_head"],
        "first_seen_authenticated_state_exact": first_left == first_right,
        "rng_exact": comparison["rng"]["exact"],
        "endpoint_discrete_state_exact": True,
        "per_step_251_500_parity_proven": False,
        "per_step_selected_action_parity_proven": False,
        "per_step_logits_parity_proven": False,
        "per_step_neurosed_distance_parity_proven": False,
        "promotion_allowed": False,
        "approximate_comparison_used": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }


def build_prebound_continuation(
    *,
    accelerated_spec_path: Path,
    accelerated_root: Path,
    full_root: Path,
    postprocess_root: Path,
    publisher_root: Path,
    matrix_authority_root: Path,
) -> dict[str, Any]:
    """Prebind downstream owners without making any of them runnable early."""

    return {
        "schema_version": "tastemolnet_t12_accelerated_continuation_v1",
        "status": "BLOCKED_PENDING_PRODUCTION_IDENTITY_REFRAME",
        "dispatchable": False,
        "accelerated_task_spec": str(accelerated_spec_path),
        "accelerated_root": str(accelerated_root),
        "checkpoint_500": str(
            accelerated_root / "checkpoints/checkpoint-00000500.manifest.json"
        ),
        "checkpoint_510": str(
            accelerated_root / "checkpoints/checkpoint-00000510.manifest.json"
        ),
        "endpoint_comparison_receipt": str(
            accelerated_root / "endpoint_250_500_510_comparison.json"
        ),
        "endpoint_comparison_is_not_per_step_parity": True,
        "full": {
            "state": "BLOCKED_PENDING_PRODUCTION_IDENTITY_REFRAME",
            "dispatchable": False,
            "task_spec": None,
            "output_root": str(full_root),
            "diagnostic_source_checkpoint": str(
                accelerated_root / "checkpoints/checkpoint-00000500.manifest.json"
            ),
            "diagnostic_source_cursor": PARITY_STEP,
            "production_identity_reframe_implemented": False,
            "checkpoint_cursors": [10_000, 20_000],
            "M_MIN": 10_000,
            "M_MAX": 20_000,
            "train_side_gate_only": True,
        },
        "postprocess": {
            "state": "BLOCKED_WAITING_FULL_GENERATION",
            "dispatchable": False,
            "task_spec": None,
            "output_root": str(postprocess_root),
            "required_generation_terminal": str(full_root / "GENERATION_PASS"),
            "calibration_only_selection": True,
            "test_after_freeze_only": True,
        },
        "publisher": {
            "dispatchable": False,
            "task_spec": None,
            "output_root": str(publisher_root),
            "matrix_authority_root": str(matrix_authority_root),
            "cell": {"dataset": "TasteMolNet", "method": "GCFExplainer"},
            "required_final_audit": str(postprocess_root / "final_audit.json"),
            "state": "PREBOUND_NOT_DISPATCHED",
        },
        "reference_must_not_be_signaled": True,
        "gpu0_mut_priority_gate": True,
    }


def build_promotion_blocker() -> dict[str, Any]:
    """Describe why the current diagnostic state cannot enter 20k production.

    This is intentionally data, not a partially implemented migrator.  The
    live reference checkpoint does not contain the required per-step ledger,
    and its authenticated journals bind the 510-step bounds in their headers.
    Relabeling those fields would create a checkpoint that never existed.
    """

    return {
        "schema_version": "tastemolnet_t12_accelerated_promotion_blocker_v1",
        "status": "BLOCKED_UNSUPPORTED_BY_CURRENT_STATE_SCHEMA",
        "promotion_allowed": False,
        "reference_mutation_required": False,
        "reference_mutation_allowed": False,
        "per_step_parity": {
            "status": "BLOCKED_MISSING_REFERENCE_LEDGER",
            "source_checkpoint_cursor": REFERENCE_STEP,
            "comparison_end_cursor": PARITY_STEP,
            "endpoint_checkpoint_comparison_available": True,
            "endpoint_comparison_is_sufficient": False,
            "missing_authenticated_per_step_fields": [
                "selected_parent",
                "selected_action",
                "pre_softmax_gine_logits",
                "normalized_neurosed_distance",
            ],
            "existing_history_retains": [
                "graph_identity",
                "probabilities",
                "prediction",
                "coverage_digest",
                "frequency_and_lineage_endpoint_state",
            ],
            "retroactive_reconstruction_claimed": False,
        },
        "production_identity_reframe": {
            "status": "BLOCKED_DIAGNOSTIC_BOUNDS_AUTHENTICATED_IN_JOURNALS",
            "diagnostic_total_steps": RELOAD_STEP,
            "diagnostic_checkpoint_cursors": [REFERENCE_STEP, PARITY_STEP, RELOAD_STEP],
            "diagnostic_source_cursor": PARITY_STEP,
            "target_total_steps": 20_000,
            "target_checkpoint_cursors": list(range(2_500, 20_001, 2_500)),
            "direct_identity_relabel_allowed": False,
            "bound_components_requiring_verified_reemission": [
                "compact_history_segment_headers_and_bounds",
                "first_seen_embedding_segment_headers_and_bounds",
                "external_transition_segment_headers_and_contract",
                "checkpoint_identity_state_and_manifest_digests",
            ],
        },
        "minimum_correct_code_points": [
            "observational_per_step_ledger_with_no_rng_or_control_flow_calls",
            "shadow_reference_replay_from_sealed_checkpoint250_because_live_reference_has_no_ledger",
            "shadow_endpoint_binding_to_live_reference_checkpoint500_and_reload510",
            "exact_shadow_vs_accelerated_ledger_comparator",
            "fresh_root_journal_reader_writer_reemission_under_20k_bounds_and_contract",
            "scientific_projection_and_rng_equality_verifier_across_reframe",
            "production_seed_cursor500_to_checkpoint2500_planner_support",
            "2500_through_20000_schedule_support_in_bounds_checkpoint_and_final_verifiers",
        ],
        "full_owner_spec_created": False,
        "postprocess_spec_created": False,
        "publisher_spec_created": False,
        "gpu_task_started": False,
    }


__all__ = [
    "PARITY_STEP",
    "REFERENCE_STEP",
    "RELOAD_STEP",
    "T12AcceleratedError",
    "build_prebound_continuation",
    "build_promotion_blocker",
    "checkpoint_science_projection",
    "compare_checkpoint_payloads",
    "file_sha256",
    "fork_step250_prefix",
    "validate_mut_gpu0_release_receipt",
    "validate_reference_step250",
]
