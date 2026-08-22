"""Recover a completed COMRECGC walk that failed during payload freezing."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import GenerationParameters, require_empty_output, sha256_file, write_json
from .frozen_payload import (
    atomic_torch_save,
    build_frozen_payload_closure,
    payload_graphs_by_official_hash,
    payload_file_audit,
    torch_load_payload,
)
from .graph_trace import (
    iter_candidate_lineage_from_selected_trace,
    iter_selected_trace,
    stable_untyped_graph_sha256,
)
from .live_graph_state import AuthoritativeGraphStore
from .runtime import validate_counterfactual_payload


class UnsafeCompletedGenerationFreezeError(RuntimeError):
    """Raised after a complete unsafe freeze-validation audit is persisted."""

    def __init__(
        self,
        *,
        audit: Mapping[str, Any],
        audit_output: str | Path | None,
    ) -> None:
        self.audit = dict(audit)
        self.audit_output = (
            Path(audit_output).expanduser().resolve()
            if audit_output is not None
            else None
        )
        failed_checks = sorted(
            str(name)
            for name, passed in (self.audit.get("checks") or {}).items()
            if passed is not True
        )
        closure_error = self.audit.get("closure_error")
        location = str(self.audit_output) if self.audit_output is not None else "none"
        super().__init__(
            "COMRECGC completed generation is not safe for freeze-only recovery; "
            f"validation_audit={location}; closure_error={closure_error!r}; "
            f"failed_checks={failed_checks}."
        )


def recovery_population_counts(
    *,
    candidate_count: int,
    candidate_lineage_resolved_count: int,
    lineage_recovery_audit: Mapping[str, Any],
) -> dict[str, int]:
    """Keep unique candidates separate from selected-trace multiplicity."""

    selected = int(lineage_recovery_audit.get("selected_transition_count", -1))
    recorded = int(lineage_recovery_audit.get("recorded_action_present_count", -1))
    legacy = int(lineage_recovery_audit.get("legacy_missing_action_count", -1))
    if selected < 0 or recorded < 0 or legacy < 0 or selected != recorded + legacy:
        raise ValueError(
            "Recovered COMRECGC selected-transition counters are inconsistent: "
            f"selected={selected}, recorded={recorded}, legacy={legacy}."
        )
    if int(candidate_count) != int(candidate_lineage_resolved_count):
        raise ValueError(
            "Recovered COMRECGC candidate lineage population is incomplete: "
            f"{candidate_lineage_resolved_count}/{candidate_count}."
        )
    return {
        "candidate_count": int(candidate_count),
        "candidate_lineage_resolved_count": int(candidate_lineage_resolved_count),
        "selected_transition_count": selected,
    }


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _materialize(
    source: Path, destination: Path, *, allow_hardlink: bool = True
) -> str:
    if destination.exists():
        raise FileExistsError(f"Recovery destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not allow_hardlink:
            raise OSError("independent physical copy required")
        os.link(source, destination)
        mode = "hardlink"
    except OSError:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with source.open("rb") as source_handle, os.fdopen(
                descriptor, "wb"
            ) as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle, 1024 * 1024)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        mode = "atomic_copy"
    if destination.stat().st_size != source.stat().st_size:
        raise ValueError(f"Recovered artifact size mismatch: {destination}")
    if sha256_file(destination) != sha256_file(source):
        raise ValueError(f"Recovered artifact checksum mismatch: {destination}")
    return mode


def _rng_state_present(root: Path) -> bool:
    for name in (
        "generation_checkpoint_manifest.json",
        "checkpoint_manifest.json",
        "rng_state.pt",
        "rng_state.json",
    ):
        for path in (root / name, root / "graph_state" / name):
            if not path.is_file() or path.stat().st_size <= 0:
                continue
            if "rng_state" not in name:
                manifest = _json(path)
                if manifest.get("rng_state"):
                    return True
            else:
                return True
    return False


def _audit_selected_trace_manifest(trace_manifest: Path) -> dict[str, Any]:
    """Verify and summarize the immutable selected-trace source inventory."""

    manifest = _json(trace_manifest)
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Selected-trace manifest has no chunks.")
    trace_root = trace_manifest.parent.resolve()
    row_count = 0
    paths: set[Path] = set()
    for expected_index, row in enumerate(chunks):
        if not isinstance(row, Mapping) or int(row.get("index", -1)) != expected_index:
            raise ValueError("Selected-trace chunk indices are not contiguous.")
        relative = Path(str(row.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts or relative in paths:
            raise ValueError("Selected-trace chunk path is unsafe or repeated.")
        paths.add(relative)
        chunk = trace_root / relative
        resolved = chunk.resolve(strict=True)
        if (
            chunk.is_symlink()
            or not resolved.is_file()
            or trace_root not in resolved.parents
            or resolved.stat().st_size != int(row.get("bytes", -1))
            or sha256_file(resolved) != str(row.get("sha256") or "")
        ):
            raise ValueError(f"Selected-trace chunk identity mismatch: {chunk}")
        rows = int(row.get("row_count", -1))
        if rows < 0:
            raise ValueError(f"Selected-trace chunk row count is invalid: {chunk}")
        row_count += rows
    if row_count != int(manifest.get("row_count", -1)):
        raise ValueError("Selected-trace manifest row total is inconsistent.")
    return {
        "manifest_sha256": sha256_file(trace_manifest),
        "chunk_count": len(chunks),
        "row_count": row_count,
        "all_chunk_sha256_verified": True,
    }


def _load_sources(
    *, dataset: str, dataset_dir: str | Path, source_csv: str | Path | None, parent_limit: int
) -> tuple[dict[str, Any], str]:
    if dataset == "aids":
        if source_csv is None:
            raise ValueError("AIDS freeze recovery requires --source-csv.")
        from .project_dataset import load_aids_generation_bundle

        bundle = load_aids_generation_bundle(
            dataset_dir=dataset_dir,
            source_csv=source_csv,
            parent_limit=parent_limit,
        )
    elif dataset == "mutagenicity":
        from .project_dataset import load_mutagenicity_generation_bundle

        bundle = load_mutagenicity_generation_bundle(
            dataset_dir=dataset_dir,
            parent_limit=parent_limit,
        )
    else:
        raise ValueError(f"Unsupported COMRECGC recovery dataset: {dataset!r}")
    return (
        dict(zip(bundle.parent_ids, bundle.graphs, strict=True)),
        bundle.dataset_fingerprint,
    )


def validate_completed_generation_freeze(
    *,
    source_generation_dir: str | Path,
    dataset: str,
    dataset_dir: str | Path,
    source_csv: str | Path | None,
    expected_steps: int = 50_000,
    expected_project_commit: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return a strict audit and the closed payload only when recovery is safe."""

    root = Path(source_generation_dir).expanduser().resolve()
    config = _json(root / "resolved_config.json")
    failure = _json(root / "_RUN_FAILED.json")
    graph_audit = _json(root / "graph_state_audit.json")
    payload_path = root / "counterfactuals.pt"
    trace_manifest = root / "trace" / "selected_action_trace_manifest.json"
    backing_store = root / "graph_state" / "authoritative_graph_store.sqlite3"
    expected = GenerationParameters.for_mode("full").__dict__

    selected_trace_audit = _audit_selected_trace_manifest(trace_manifest)

    store = AuthoritativeGraphStore(backing_store, read_only=True)
    try:
        backing_audit = store.integrity_audit()
    finally:
        store.close()

    payload = torch_load_payload(payload_path)
    graph_map, candidates = validate_counterfactual_payload(payload)
    selected_events = iter_selected_trace(trace_manifest)
    closed_payload: dict[str, Any] | None = None
    closure_audit: dict[str, Any] | None = None
    closure_error: str | None = None
    try:
        closed_payload, closure_audit = build_frozen_payload_closure(
            payload,
            selected_events,
            backing_store_path=backing_store,
        )
    except Exception as exc:
        closure_error = f"{type(exc).__name__}: {exc}"

    random_walk_complete = int(graph_audit.get("move_count", -1)) == int(
        expected_steps
    )
    rng_present = _rng_state_present(root)
    serialized_transitions = payload.get("transitions")
    serialized_transition_state_present = (
        isinstance(serialized_transitions, Mapping) and bool(serialized_transitions)
    )
    serialized_transition_hash_count = int(
        (closure_audit or {}).get("transition_hash_count", 0)
    )
    serialized_transition_closure_complete = bool(
        closure_audit
        and closure_audit.get("closure_complete")
        and (
            not serialized_transition_state_present
            or serialized_transition_hash_count > 0
        )
    )
    historical_transition_audit = {
        "source_count": int(graph_audit.get("transition_source_count", 0)),
        "destination_count": int(
            graph_audit.get("transition_destination_count", 0)
        ),
        "unresolved_source_count": int(
            graph_audit.get("unresolved_transition_source_count", -1)
        ),
        "invalid_destination_count": int(
            graph_audit.get("invalid_transition_destination_count", -1)
        ),
    }
    historical_transition_audit["passed"] = (
        historical_transition_audit["unresolved_source_count"] == 0
        and historical_transition_audit["invalid_destination_count"] == 0
    )
    failure_message = " ".join(
        str(failure.get("message", "")).strip().lower().split()
    ).removesuffix(".")
    accepted_post_generation_failures = {
        "frozen_payload_missing_graph": (
            "selected trace references a graph absent from the frozen payload"
        ),
        "recorded_action_lineage_ambiguity": (
            "selected comrecgc transition is not one unique pinned-upstream "
            "single edit"
        ),
    }
    matched_failure_signatures = sorted(
        name
        for name, signature in accepted_post_generation_failures.items()
        if failure_message == " ".join(signature.lower().split()).removesuffix(".")
    )
    actual_project_commit = config.get("project_commit")
    project_commit_matches = (
        expected_project_commit is None
        or actual_project_commit == expected_project_commit
    )
    project_commit_identity = {
        "actual": actual_project_commit,
        "expected": expected_project_commit,
        "actual_type": type(actual_project_commit).__name__,
        "expected_type": type(expected_project_commit).__name__,
        "actual_repr": repr(actual_project_commit),
        "expected_repr": repr(expected_project_commit),
        "actual_length": (
            len(actual_project_commit)
            if isinstance(actual_project_commit, str)
            else None
        ),
        "expected_length": (
            len(expected_project_commit)
            if isinstance(expected_project_commit, str)
            else None
        ),
        "matches": project_commit_matches,
    }
    checks = {
        "failure_is_post_generation_freeze": (
            failure.get("stage") == "project_generation"
            and len(matched_failure_signatures) == 1
        ),
        "dataset_matches": config.get("dataset") == dataset,
        "mode_is_full": config.get("mode") == "full",
        "generation_config_match": config.get("parameters") == expected,
        "project_commit_matches": project_commit_matches,
        "random_walk_complete": random_walk_complete,
        "official_payload_present": payload_path.is_file() and bool(graph_map),
        "candidate_payload_present": bool(candidates),
        "selected_trace_manifest_complete": trace_manifest.is_file(),
        "backing_store_integrity": backing_audit.get("integrity_passed") is True,
        "unresolved_runtime_lookup_zero": int(
            graph_audit.get("unresolved_lookups", -1)
        )
        == 0,
        # Runtime transitions are an ephemeral proposal cache.  A completed
        # walk only needs them in the frozen closure when the source payload
        # actually serialized them.  The selected trace is the authoritative
        # record of transitions consumed by downstream reconstruction.
        "serialized_transition_closure_complete": (
            serialized_transition_closure_complete
        ),
        "frozen_payload_closure_complete": bool(
            closure_audit and closure_audit.get("closure_complete")
        ),
    }
    freeze_only_safe = all(checks.values())
    result = {
        "schema_version": "comrecgc_completed_generation_freeze_audit_v4",
        "source_generation_dir": str(root),
        "dataset": dataset,
        "actual_project_commit": actual_project_commit,
        "expected_project_commit": expected_project_commit,
        "project_commit_identity": project_commit_identity,
        "completed_steps": graph_audit.get("move_count"),
        "expected_steps": int(expected_steps),
        "random_walk_complete": random_walk_complete,
        "checkpoint_atomic": payload_path.is_file() and trace_manifest.is_file(),
        "backing_store_integrity": backing_audit.get("integrity_passed") is True,
        "all_selected_trace_hashes_resolvable": bool(
            closure_audit and closure_audit.get("closure_complete")
        ),
        "all_transition_hashes_resolvable": (
            serialized_transition_closure_complete
        ),
        "serialized_transition_state_present": serialized_transition_state_present,
        "serialized_transition_hash_count": serialized_transition_hash_count,
        "transition_closure_policy": (
            "serialized_transition_closure_if_present_otherwise_"
            "completed_selected_trace_authority_v1"
        ),
        "historical_transition_state_required_for_freeze_only": False,
        "historical_transition_audit": historical_transition_audit,
        "RNG_state_present": rng_present,
        "rng_state_required_for_freeze_only": False,
        "rng_state_reason": (
            "Random walk is complete; freeze-only performs no proposal or RNG call."
            if random_walk_complete
            else "Incomplete walks require an exact RNG checkpoint."
        ),
        "generation_config_match": checks["generation_config_match"],
        "candidate_count": len(candidates),
        "graph_map_count_before_closure": len(graph_map),
        "backing_store_audit": backing_audit,
        "selected_trace_audit": selected_trace_audit,
        "frozen_payload_closure": closure_audit,
        "closure_error": closure_error,
        "checks": checks,
        "matched_post_generation_failure_signatures": matched_failure_signatures,
        "FREEZE_ONLY_RECOVERY_SAFE": freeze_only_safe,
        "RESUME_SAFE": False,
        "resume_reason": "Completed-walk freeze recovery is distinct from random-walk resume.",
        "fresh_rerun_required": not freeze_only_safe,
        "failure": failure,
    }
    return result, closed_payload if freeze_only_safe else None


def recover_completed_generation_freeze(
    *,
    source_generation_dir: str | Path,
    output_dir: str | Path,
    dataset: str,
    dataset_dir: str | Path,
    source_csv: str | Path | None,
    expected_steps: int = 50_000,
    expected_project_commit: str | None = None,
    audit_output: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze an already complete walk into a new versioned generation root."""

    audit, payload = validate_completed_generation_freeze(
        source_generation_dir=source_generation_dir,
        dataset=dataset,
        dataset_dir=dataset_dir,
        source_csv=source_csv,
        expected_steps=expected_steps,
        expected_project_commit=expected_project_commit,
    )
    if payload is None or audit["FREEZE_ONLY_RECOVERY_SAFE"] is not True:
        # Validation is the expensive pass over the complete frozen graph and
        # selected-trace closure.  Persist that exact result before failing so
        # callers never need to repeat the scan merely to recover diagnostics.
        failure_audit_output = (
            Path(audit_output).expanduser().resolve()
            if audit_output is not None
            else Path(output_dir).expanduser().resolve()
            / "fresh_recovery_audit.json"
        )
        write_json(failure_audit_output, audit)
        raise UnsafeCompletedGenerationFreezeError(
            audit=audit,
            audit_output=failure_audit_output,
        )

    source = Path(source_generation_dir).expanduser().resolve()
    output = require_empty_output(output_dir)
    trace_output = output / "trace"
    trace_output.mkdir(parents=True, exist_ok=True)
    source_manifest_path = source / "trace" / "selected_action_trace_manifest.json"
    source_manifest = _json(source_manifest_path)
    materialized: dict[str, str] = {}
    for chunk in source_manifest.get("chunks") or []:
        relative = Path(str(chunk["path"]))
        materialized[relative.as_posix()] = _materialize(
            source / "trace" / relative, trace_output / relative
        )
    materialized[source_manifest_path.name] = _materialize(
        source_manifest_path, trace_output / source_manifest_path.name
    )

    source_graphs, dataset_fingerprint = _load_sources(
        dataset=dataset,
        dataset_dir=dataset_dir,
        source_csv=source_csv,
        parent_limit=int(_json(source / "resolved_config.json")["parent_limit"]),
    )
    lineage_index = trace_output / "candidate_action_lineage_index.jsonl"
    temporary = lineage_index.with_name(f".{lineage_index.name}.{os.getpid()}.tmp")
    lineage_count = 0
    lineage_resolved = 0
    lineage_recovery_audit: dict[str, Any] = {}
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for expected_index, row in enumerate(
                iter_candidate_lineage_from_selected_trace(
                    payload,
                    iter_selected_trace(trace_output / source_manifest_path.name),
                    source_graphs_by_parent_id=source_graphs,
                    include_actions=False,
                    recovery_audit=lineage_recovery_audit,
                )
            ):
                if int(row["candidate_index"]) != expected_index:
                    raise ValueError("Recovered COMRECGC lineage order changed.")
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True))
                handle.write("\n")
                lineage_count += 1
                lineage_resolved += int(row["action_lineage_resolved"] is True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, lineage_index)
    finally:
        temporary.unlink(missing_ok=True)
    if lineage_count != lineage_resolved:
        raise RuntimeError(
            f"Recovered COMRECGC lineage is incomplete: {lineage_resolved}/{lineage_count}."
        )

    population_counts = recovery_population_counts(
        candidate_count=lineage_count,
        candidate_lineage_resolved_count=lineage_resolved,
        lineage_recovery_audit=lineage_recovery_audit,
    )

    lineage_counter_fields = {
        key: int(lineage_recovery_audit.get(key, 0))
        for key in (
            "recorded_action_present_count",
            "recorded_action_replay_ok_count",
            "recorded_action_replay_mismatch_count",
            "recorded_action_index_remap_count",
            "legacy_missing_action_count",
            "legacy_inference_called_count",
            "legacy_inference_ambiguous_count",
            "predecessor_target_count",
            "predecessor_duplicate_event_count",
            "predecessor_duplicate_exact_transition_count",
            "predecessor_duplicate_content_equivalent_count",
            "predecessor_source_official_alias_count",
            "predecessor_conflicting_exact_event_count",
            "predecessor_cross_parent_convergence_count",
            "predecessor_recorded_upgrade_count",
            "predecessor_unverified_conflict_count",
            "predecessor_unresolved_legacy_conflict_count",
            "predecessor_selected_parent_mismatch_count",
            "selected_event_source_parent_mismatch_count",
            "selected_event_target_parent_mismatch_count",
        )
    }

    lineage_summary = {
        "schema_version": 2,
        "format": "selected_trace_predecessor_index",
        "candidate_count": lineage_count,
        "candidate_lineage_resolved_count": lineage_resolved,
        "candidate_index_path": lineage_index.name,
        "candidate_index_sha256": sha256_file(lineage_index),
        "selected_trace_manifest_path": source_manifest_path.name,
        "selected_trace_manifest_sha256": sha256_file(
            trace_output / source_manifest_path.name
        ),
        "candidate_actions_inlined": False,
        "reconstruction_policy": "stream_one_candidate_from_selected_trace_v1",
        "lineage_recovery_audit": dict(lineage_recovery_audit),
        **lineage_counter_fields,
    }
    write_json(trace_output / "candidate_action_lineage.json", lineage_summary)

    payload_path = output / "counterfactuals.pt"
    source_closure_fields = {
        "canonical_graph_records_persisted": isinstance(
            payload.get("canonical_graph_records"), Mapping
        ),
        "alias_to_canonical_persisted": isinstance(
            payload.get("alias_to_canonical"), Mapping
        ),
        "original_trace_hashes_persisted": isinstance(
            payload.get("original_trace_hashes"), list
        ),
    }
    if not all(source_closure_fields.values()):
        raise RuntimeError(
            "Validated recovery payload omitted a typed frozen-closure field: "
            f"{source_closure_fields}."
        )
    expected_canonical_records = {
        str(key): stable_untyped_graph_sha256(graph)
        for key, graph in payload["canonical_graph_records"].items()
    }
    expected_aliases = {
        str(alias): str(canonical)
        for alias, canonical in payload["alias_to_canonical"].items()
    }
    expected_original_trace_hashes = [
        str(value) for value in payload["original_trace_hashes"]
    ]
    atomic_torch_save(payload, payload_path)
    reloaded = torch_load_payload(payload_path)
    persisted_closure_fields = {
        "canonical_graph_records_persisted": isinstance(
            reloaded.get("canonical_graph_records"), Mapping
        ),
        "alias_to_canonical_persisted": isinstance(
            reloaded.get("alias_to_canonical"), Mapping
        ),
        "original_trace_hashes_persisted": isinstance(
            reloaded.get("original_trace_hashes"), list
        ),
    }
    if not all(persisted_closure_fields.values()):
        raise RuntimeError(
            "Recovered payload omitted a typed frozen-closure field: "
            f"{persisted_closure_fields}."
        )
    actual_canonical_records = {
        str(key): stable_untyped_graph_sha256(graph)
        for key, graph in reloaded["canonical_graph_records"].items()
    }
    actual_aliases = {
        str(alias): str(canonical)
        for alias, canonical in reloaded["alias_to_canonical"].items()
    }
    actual_original_trace_hashes = [
        str(value) for value in reloaded["original_trace_hashes"]
    ]
    persisted_roundtrip_fields = {
        "canonical_graph_records_roundtrip_verified": (
            actual_canonical_records == expected_canonical_records
        ),
        "alias_to_canonical_roundtrip_verified": actual_aliases == expected_aliases,
        "original_trace_hashes_roundtrip_verified": (
            actual_original_trace_hashes == expected_original_trace_hashes
        ),
    }
    if not all(persisted_roundtrip_fields.values()):
        raise RuntimeError(
            "Recovered payload frozen-closure fields changed across serialization: "
            f"{persisted_roundtrip_fields}."
        )
    verified_payload, post_write = build_frozen_payload_closure(
        reloaded,
        iter_selected_trace(trace_output / source_manifest_path.name),
        backing_store_path=None,
    )
    if post_write["closure_complete"] is not True:
        raise RuntimeError("Recovered payload failed post-write closure verification.")
    persisted_counts = {
        "canonical_graph_record_count": len(reloaded["canonical_graph_records"]),
        "alias_count": len(reloaded["alias_to_canonical"]),
        "original_trace_hash_count": len(reloaded["original_trace_hashes"]),
    }
    for field, actual in persisted_counts.items():
        if int(post_write.get(field, -1)) != actual:
            raise RuntimeError(
                f"Recovered payload {field} differs after serialization: "
                f"audit={post_write.get(field)}, payload={actual}."
            )
    for field in (
        "closure_digest",
        "canonical_graph_record_count",
        "alias_count",
        "original_trace_hash_count",
        "selected_trace_rows_sha256",
        "selected_trace_row_count",
    ):
        if (audit["frozen_payload_closure"] or {}).get(field) != post_write.get(field):
            raise RuntimeError(
                f"Recovered payload closure changed across serialization: {field}."
            )
    expected_trace_hashes = set(expected_original_trace_hashes)
    actual_trace_hashes = set(verified_payload.get("original_trace_hashes") or [])
    resolvable_hashes = payload_graphs_by_official_hash(verified_payload)
    missing_roundtrip_hashes = sorted(expected_trace_hashes - set(resolvable_hashes))
    if expected_trace_hashes != actual_trace_hashes or missing_roundtrip_hashes:
        raise RuntimeError(
            "Recovered payload changed or lost original selected-trace hashes: "
            f"expected={len(expected_trace_hashes)}, actual={len(actual_trace_hashes)}, "
            f"missing={missing_roundtrip_hashes[:20]}."
        )

    graph_state = output / "graph_state"
    store_mode = _materialize(
        source / "graph_state" / "authoritative_graph_store.sqlite3",
        graph_state / "authoritative_graph_store.sqlite3",
        # Downstream tools may open the recovered store with SQLite.  A hard
        # link would make any accidental journal/write mutate the immutable
        # preserved input inode, so the recovery boundary requires a distinct
        # physical copy even when source and destination share a filesystem.
        allow_hardlink=False,
    )
    graph_audit_mode = _materialize(
        source / "graph_state_audit.json", output / "graph_state_audit.json"
    )
    resolved_config_mode = _materialize(
        source / "resolved_config.json", output / "resolved_config.json"
    )
    source_checksums = {
        relative: sha256_file(source / relative)
        for relative in (
            "counterfactuals.pt",
            "resolved_config.json",
            "graph_state_audit.json",
            "graph_state/authoritative_graph_store.sqlite3",
            "trace/selected_action_trace_manifest.json",
        )
    }
    adoption_manifest = {
        "schema_version": "comrecgc_fresh_root_adoption_v1",
        "generation_mode": "adopted_read_only_cache",
        "adopted_from": str(source),
        "source_checksums": source_checksums,
        "source_dataset_fingerprint": dataset_fingerprint,
        "source_candidate_count": int(audit["candidate_count"]),
        "source_trace_row_count": int(audit["selected_trace_audit"]["row_count"]),
        "serialization_rerun": True,
        "lineage_resolution_rerun": True,
        "freeze_rerun": True,
        "bare_symlink_used": False,
        "fresh_output_root": str(output),
    }
    write_json(output / "adoption_manifest.json", adoption_manifest)
    closure_audit = {
        **audit["frozen_payload_closure"],
        **payload_file_audit(payload_path),
        **persisted_closure_fields,
        **persisted_roundtrip_fields,
        **persisted_counts,
        "post_write_reload_verified": True,
        "original_trace_hash_roundtrip_verified": True,
        "original_trace_hash_roundtrip_count": len(expected_trace_hashes),
    }
    write_json(output / "frozen_payload_closure_audit.json", closure_audit)
    trace_summary = {
        "trace_schema_version": 1,
        "trace_only": True,
        "candidate_count": lineage_count,
        "selected_transition_count": population_counts["selected_transition_count"],
        "candidate_lineage_resolved_count": lineage_resolved,
        "selected_trace_path": str(trace_output / source_manifest_path.name),
        "candidate_lineage_path": str(trace_output / "candidate_action_lineage.json"),
        "candidate_lineage_index_path": str(lineage_index),
        "candidate_lineage_format": "selected_trace_predecessor_index",
        "lineage_recovery_policy": "authoritative_backing_freeze_only_v3",
        "lineage_recovery_audit": dict(lineage_recovery_audit),
        **lineage_counter_fields,
        "algorithm_rerun": False,
        "frozen_payload_closure": closure_audit,
    }
    write_json(trace_output / "trace_summary.json", trace_summary)
    write_json(
        trace_output / "_TRACE_COMPLETE.json",
        {
            "trace_complete": True,
            "selected_trace_manifest_sha256": sha256_file(
                trace_output / source_manifest_path.name
            ),
            "candidate_lineage_sha256": sha256_file(
                trace_output / "candidate_action_lineage.json"
            ),
            "freeze_only_recovery": True,
        },
    )

    config = _json(source / "resolved_config.json")
    graph_map, candidates = validate_counterfactual_payload(reloaded)
    manifest = {
        **config,
        "counterfactuals_path": str(payload_path),
        "counterfactuals_sha256": sha256_file(payload_path),
        "counterfactuals_bytes": payload_path.stat().st_size,
        "counterfactual_candidate_count": len(candidates),
        "selected_transition_count": population_counts["selected_transition_count"],
        "visited_graph_count": len(graph_map),
        "trace_enabled": True,
        "trace_summary": trace_summary,
        "lineage_recovery_audit": dict(lineage_recovery_audit),
        **lineage_counter_fields,
        "graph_state_audit_path": str(output / "graph_state_audit.json"),
        "graph_state_audit_sha256": sha256_file(output / "graph_state_audit.json"),
        "frozen_payload_closure_audit_path": str(
            output / "frozen_payload_closure_audit.json"
        ),
        "frozen_payload_closure_audit_sha256": sha256_file(
            output / "frozen_payload_closure_audit.json"
        ),
        "algorithm_rerun": False,
        "freeze_only_recovery": True,
        "source_generation_dir": str(source),
        "source_dataset_fingerprint": dataset_fingerprint,
        "generation_mode": "adopted_read_only_cache",
        "adoption_manifest_path": str(output / "adoption_manifest.json"),
        "adoption_manifest_sha256": sha256_file(output / "adoption_manifest.json"),
        "serialization_rerun": True,
        "lineage_resolution_rerun": True,
        "freeze_rerun": True,
        "run_complete": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "run_manifest.json", manifest)
    write_json(
        output / "progress.json",
        {
            "stage": "generation",
            "current_step": int(expected_steps),
            "max_steps": int(expected_steps),
            "run_complete": True,
            "freeze_only_recovery": True,
        },
    )
    recovery = {
        **audit,
        "recovery_completed": True,
        "algorithm_rerun": False,
        "source_generation_dir": str(source),
        "output_dir": str(output),
        "materialization": {
            "trace": materialized,
            "backing_store": store_mode,
            "graph_state_audit": graph_audit_mode,
            "resolved_config": resolved_config_mode,
        },
        "counterfactuals_sha256": manifest["counterfactuals_sha256"],
        "candidate_count": len(candidates),
        "selected_transition_count": population_counts["selected_transition_count"],
        "candidate_lineage_resolved_count": lineage_resolved,
        "lineage_recovery_audit": dict(lineage_recovery_audit),
        **lineage_counter_fields,
    }
    write_json(output / "freeze_only_recovery.json", recovery)
    write_json(
        output / "_RUN_COMPLETE.json",
        {
            "run_complete": True,
            "freeze_only_recovery": True,
            "counterfactuals_sha256": manifest["counterfactuals_sha256"],
            "recovery_manifest_sha256": sha256_file(
                output / "freeze_only_recovery.json"
            ),
        },
    )
    return recovery
