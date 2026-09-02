#!/usr/bin/env python3
"""Persistent Mut historical-50k successor using the existing GPU controller.

The sidecar performs one read-only inventory, then remains in
``WAITING_FOR_EMPIRICAL_ADMISSION`` until both an exclusive project GPU and at
least 96 GiB of parent cgroup headroom are observed for the frozen stability
window.  It never signals the superseded 440-GiB waiter.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.contracts import sha256_file, stable_json_sha256  # noqa: E402
from src.eval.am_legacy_standardization import scan_live_writers  # noqa: E402
from src.utils.autodl_mut_traceoff_parity_v1 import (  # noqa: E402
    INSTRUMENTATION_PROJECT_COMMIT,
    INSTRUMENTATION_SOURCE_INVENTORY_SHA256,
    LEGACY_SOURCE_INVENTORY_SHA256,
    SOURCE_CANDIDATE_COUNT,
    SOURCE_DATASET_SHA256,
    SOURCE_PARENT_ORDER_SHA256,
    SOURCE_PAYLOAD_SHA256,
    SOURCE_PROJECT_COMMIT,
    validate_instrumentation_equivalence_gate,
    verify_traced_source,
)
from src.utils.autodl_mut_fast_accurate_v2 import (  # noqa: E402
    derive_empirical_memory_admission,
    read_cgroup_snapshot,
)
from src.utils.autodl_mut_trace_on_adoption_v1 import (  # noqa: E402
    AUDIT_SCHEMA as TRACE_AUDIT_SCHEMA,
    CANARY_HEADROOM_STOP_BYTES,
    CANARY_REQUIRED_HEADROOM_BYTES,
    CANARY_RSS_STOP_BYTES,
    EXPECTED_CANDIDATE_UNIVERSE_SHA256,
    validate_authorization_receipt,
    verify_mut_candidate_pair_dbscan_binding,
)
from src.utils.autodl_runtime import (  # noqa: E402
    FOUR_GPU_RECOVERY_LIMIT,
    gpu_lock_available,
    query_gpu_inventory,
)


SCHEMA = "mut_fast_accurate_v2"
ADOPTION_SCHEMA = "mut_comrecgc_historical50k_adoption_v2"
LOCATOR_SCHEMA = "fast16_matrix_cell_root_locator_v1"
# The original 96-GiB no-child threshold remains the frozen legacy route.  The
# separately authorized trace-mode one-shot uses CANARY_REQUIRED_HEADROOM_BYTES
# and never changes the full-generation admission formula.
MINIMUM_HEADROOM_BYTES = 96 * 1024**3
HEX64 = re.compile(r"[0-9a-f]{64}")
MUT_UPSTREAM_COMMIT = "122f9341a360e9f06bb58a2f5823bb596021f6bf"
MUT_GNN_SHA256 = "22045e5a6a833d6ed980cef9834859859136a1e2f644d19d78bd63345585f239"
MUT_DISTANCE_SHA256 = "bc64c16340c9170388ff1b3951d2ee4cb9a372456b09691ecd6bb2a881f17648"
MUT_RF_ORACLE_SHA256 = "af213aa766626decaf99876b43ede725412a355adf37f1aa0d56233d8653e204"
DEFAULT_SEMANTIC_FINALIZER_PROJECT_ROOT = Path(
    "/root/autodl-tmp/worktrees/final-five-closeout-582bc4b-20260902T040000Z"
)


class MutFastError(RuntimeError):
    """The successor cannot proceed without weakening the frozen contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise MutFastError(f"Refusing symlink JSON target: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_pass(path: Path) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, b"PASS\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json(path: Path, *, label: str = "JSON") -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise MutFastError(f"{label} is not a physical nonempty file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MutFastError(f"Invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise MutFastError(f"{label} must be one object: {path}")
    return dict(value)


def _absolute(value: Any, *, label: str, exists: bool = True) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise MutFastError(f"{label} must be an absolute non-symlink path: {path}")
    try:
        return path.resolve(strict=exists)
    except OSError as exc:
        raise MutFastError(f"{label} is absent: {path}") from exc


def load_spec(path: Path, *, require_inputs: bool = True) -> dict[str, Any]:
    spec_path = _absolute(path, label="spec")
    value = _json(spec_path, label="Mut fast-accurate spec")
    required = {
        "schema_version": SCHEMA,
        "paper_frozen": True,
        "run_gnn_ablation": False,
        "allow_historical_adoption_without_full_50k_parity": True,
        # The current historical 50k source is trace-enabled, while the user
        # froze automatic Route-A adoption to trace-disabled artifacts.  The
        # successor may run the explicitly authorized 500-step diagnostic but
        # must not publish that source without a separate scientific decision.
        "allow_trace_on_historical_adoption": False,
        "historical_source_trace_enabled": True,
        "trace_parity_passed": False,
        "full_50k_rerun_performed": False,
        "cgroup_version": 1,
        "cgroup_mount_read_only": True,
        "systemd_scope_available": False,
        "child_cgroup_created": False,
        "no_child_fallback_reason": (
            "cgroup_v1_mount_read_only_and_systemd_manager_offline"
        ),
        "minimum_parent_headroom_bytes": MINIMUM_HEADROOM_BYTES,
    }
    changed = [key for key, expected in required.items() if value.get(key) != expected]
    if changed:
        raise MutFastError(f"Frozen Mut successor contract changed: {changed}")
    controller_id = str(value.get("controller_id") or "")
    if not controller_id or re.fullmatch(r"[A-Za-z0-9_.-]+", controller_id) is None:
        raise MutFastError("controller_id is unsafe")
    runtime = _absolute(value.get("runtime_root"), label="runtime root", exists=require_inputs)
    control = _absolute(value.get("control_root"), label="control root", exists=require_inputs)
    if control != (runtime / "control").resolve(strict=require_inputs):
        raise MutFastError("control_root must be runtime_root/control")
    output = _absolute(value.get("fresh_output_root"), label="fresh output", exists=False)
    repairs = runtime / "outputs/autodl/paper_matrix/four_methods_four_datasets_v1/repairs"
    try:
        output.relative_to(repairs)
    except ValueError as exc:
        raise MutFastError("fresh output escapes the paper-matrix repairs namespace") from exc
    for key in ("stable_admission_seconds", "poll_seconds"):
        if int(value.get(key, 0)) < 5:
            raise MutFastError(f"{key} is too small")
    if require_inputs:
        for key in (
            "project_root",
            "proc_root",
            "cgroup_memory_root",
            "historical_source_root",
            "completed_common_root",
            "legacy_project_root",
            "instrumentation_project_root",
        ):
            _absolute(value.get(key), label=key)
        python_requested = Path(str(value.get("python") or "")).expanduser()
        if not python_requested.is_absolute():
            raise MutFastError("configured Python must be an absolute path")
        try:
            python = python_requested.resolve(strict=True)
        except OSError as exc:
            raise MutFastError("configured Python is absent") from exc
        if not python.is_file() or not os.access(python, os.X_OK):
            raise MutFastError("configured Python is not executable")
        value["python"] = str(python)
        for section in ("replay", "standardization"):
            if not isinstance(value.get(section), Mapping):
                raise MutFastError(f"{section} must be one object")
    return {**value, "spec_path": str(spec_path)}


def _replace(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace(item, replacements) for key, item in value.items()}
    return value


def materialize(
    *, template: Path, output: Path, project_root: Path, legacy_root: Path,
    instrumentation_root: Path, timestamp: str,
) -> dict[str, Any]:
    source = _json(_absolute(template, label="template"), label="template")
    target = _absolute(output, label="materialized spec", exists=False)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"materialized spec must be fresh: {target}")
    value = _replace(
        source,
        {
            "__PROJECT_ROOT__": str(project_root.resolve(strict=True)),
            "__LEGACY_PROJECT_ROOT__": str(legacy_root.resolve(strict=True)),
            "__INSTRUMENTATION_PROJECT_ROOT__": str(
                instrumentation_root.resolve(strict=True)
            ),
            "__TIMESTAMP__": timestamp,
        },
    )
    _atomic_json(target, value)
    return {"spec": str(target), "controller_id": value["controller_id"]}


def _lineage_contract(root: Path) -> dict[str, Any]:
    path = root / "trace/candidate_action_lineage.json"
    value = _json(path, label="historical lineage")
    audit = value.get("lineage_recovery_audit")
    if not isinstance(audit, Mapping):
        audit = value
    expected_zero = (
        "recorded_action_replay_mismatch_count",
        "predecessor_unverified_conflict_count",
        "predecessor_unresolved_legacy_conflict_count",
        "predecessor_selected_parent_mismatch_count",
        "selected_event_source_parent_mismatch_count",
    )
    failures = [name for name in expected_zero if int(audit.get(name, -1)) != 0]
    target_mismatch = int(audit.get("selected_event_target_parent_mismatch_count", -1))
    cross_parent = int(audit.get("predecessor_cross_parent_convergence_count", -1))
    conflicting_exact = int(audit.get("predecessor_conflicting_exact_event_count", -1))
    if target_mismatch != 14 or cross_parent != 1 or conflicting_exact != 1:
        failures.append("reviewed_cross_parent_and_exact_event_counts")
    if failures:
        raise MutFastError(f"Historical lineage adoption gate failed: {failures}")
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": sha256_file(path),
        "candidate_count": int(value.get("candidate_count", -1)),
        "candidate_lineage_resolved_count": int(
            value.get("candidate_lineage_resolved_count", -1)
        ),
        "recorded_action_replay_mismatch_count": 0,
        "selected_event_target_parent_mismatch_count": 14,
        "predecessor_cross_parent_convergence_count": 1,
        "predecessor_conflicting_exact_event_count": 1,
        "predecessor_selected_parent_mismatch_count": 0,
        "cross_parent_interpretation": (
            "canonical_representative_convergence_not_lineage_error"
        ),
    }


def publish_inventory(*, spec: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    output = _absolute(output_dir, label="inventory output", exists=False)
    if output.exists() or output.is_symlink():
        prior = output / "historical_inventory.json"
        marker = output / "PASS"
        if prior.is_file() and marker.read_bytes() == b"PASS\n":
            return _json(prior, label="existing historical inventory")
        raise FileExistsError(f"inventory output is not a reusable PASS root: {output}")
    source_root = _absolute(spec["historical_source_root"], label="historical source")
    evidence = verify_traced_source(
        source_root=source_root,
        proc_root=_absolute(spec["proc_root"], label="proc root"),
        hash_payload=True,
    )
    lineage = _lineage_contract(source_root)
    if lineage["candidate_count"] != SOURCE_CANDIDATE_COUNT:
        raise MutFastError("Historical lineage candidate count changed")
    payload = {
        "schema_version": "mut_historical_50k_inventory_v2",
        "status": "PASS",
        "dataset": "mutagenicity",
        "method": "COMRECGC",
        "historical_source_trace_enabled": True,
        "trace_parity_passed": False,
        "full_50k_rerun_performed": False,
        "source": evidence,
        "lineage": lineage,
        "verified_at": _utc_now(),
    }
    output.mkdir(parents=True)
    _atomic_json(output / "historical_inventory.json", payload)
    _write_pass(output / "PASS")
    return payload


def _nested_values(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        result: list[Any] = []
        for item in value.values():
            result.extend(_nested_values(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_nested_values(item))
        return result
    return [value]


def _require_bound(document: Mapping[str, Any], value: str, *, label: str) -> None:
    if value not in {str(item) for item in _nested_values(document)}:
        raise MutFastError(f"{label} is not transitively bound in its manifest")


def publish_adoption(
    *, spec: Mapping[str, Any], inventory_gate: Path,
    equivalence_gate: Path, output_dir: Path,
    authorization_receipt: Path | None = None,
    trace_code_audit: Path | None = None,
    instrumentation_equivalence_gate: Path | None = None,
    canary_memory_receipt: Path | None = None,
) -> dict[str, Any]:
    if (
        authorization_receipt is None
        or trace_code_audit is None
        or instrumentation_equivalence_gate is None
        or canary_memory_receipt is None
    ):
        raise MutFastError(
            "Trace-on source adoption requires both authorization and static "
            "audit, plus historical/instrumented equivalence and memory gates"
        )
    inventory_path = _absolute(inventory_gate, label="inventory gate")
    inventory = _json(inventory_path, label="inventory gate")
    if (
        inventory.get("schema_version") != "mut_historical_50k_inventory_v2"
        or inventory.get("status") != "PASS"
        or inventory.get("trace_parity_passed") is not False
    ):
        raise MutFastError("Historical inventory gate is invalid")
    equivalence_path = _absolute(equivalence_gate, label="equivalence gate")
    authorization: dict[str, Any] | None = None
    authorization_file_sha256: str | None = None
    audit: dict[str, Any] | None = None
    audit_path: Path | None = None
    instrumentation_equivalence: dict[str, Any] | None = None
    instrumentation_equivalence_path: Path | None = None
    memory: dict[str, Any] | None = None
    memory_path: Path | None = None
    if authorization_receipt is not None or trace_code_audit is not None:
        if (
            authorization_receipt is None
            or trace_code_audit is None
            or instrumentation_equivalence_gate is None
            or canary_memory_receipt is None
        ):
            raise MutFastError(
                "Trace-on adoption requires authorization, static audit, "
                "historical/instrumented equivalence, and memory gates"
            )
        authorization_path = _absolute(
            authorization_receipt, label="trace-on authorization"
        )
        authorization, authorization_file_sha256 = validate_authorization_receipt(
            authorization_path,
            expected_controller_id=str(spec["controller_id"]),
            expected_source_root=_absolute(
                spec["historical_source_root"], label="historical source"
            ),
        )
        audit_path = _absolute(trace_code_audit, label="trace code audit")
        audit = _json(audit_path, label="trace code audit")
        audit_unhashed = {
            key: value for key, value in audit.items() if key != "audit_sha256"
        }
        audit_trees_valid = True
        for tree_key, expected_commit in (
            ("historical", SOURCE_PROJECT_COMMIT),
            ("instrumentation", INSTRUMENTATION_PROJECT_COMMIT),
        ):
            tree = audit.get(tree_key)
            branches = tree.get("branches") if isinstance(tree, Mapping) else None
            assertions = (
                tree.get("scientific_assertions")
                if isinstance(tree, Mapping)
                else None
            )
            audit_trees_valid = audit_trees_valid and bool(
                isinstance(tree, Mapping)
                and tree.get("status") == "PASS"
                and tree.get("commit") == expected_commit
                and tree.get("unknown_branches") == []
                and tree.get("failed_scientific_assertions") == []
                and isinstance(assertions, Mapping)
                and assertions
                and all(value is True for value in assertions.values())
                and isinstance(branches, list)
                and branches
                and all(
                    isinstance(row, Mapping)
                    and row.get("classification")
                    in {
                        "OBSERVATIONAL_WRITE_ONLY",
                        "CHECKPOINT_SERIALIZATION_ONLY",
                    }
                    for row in branches
                )
            )
        if (
            audit.get("schema_version") != TRACE_AUDIT_SCHEMA
            or audit.get("status") != "PASS"
            or audit.get("trace_is_observational") is not True
            or audit.get("trace_rng_mutation_found") is not False
            or audit.get("trace_algorithm_state_mutation_found") is not False
            or audit.get("trace_control_flow_mutation_found") is not False
            or audit.get("trace_candidate_selection_is_observational") is not True
            or audit.get("trace_operational_side_effects_found") is not True
            or audit.get("trace_post_walk_payload_serialization_mutation_found")
            is not True
            or audit.get("trace_post_walk_graph_closure_only") is not True
            or audit.get("static_audit_sufficient_for_adoption") is not False
            or audit.get("dynamic_500_step_equivalence_required") is not True
            or audit.get("full_trace_on_off_parity_claimed") is not False
            or audit.get("audit_sha256") != stable_json_sha256(audit_unhashed)
            or not audit_trees_valid
        ):
            raise MutFastError("Trace code audit is not one valid observational PASS")
        instrumentation_equivalence_path = _absolute(
            instrumentation_equivalence_gate,
            label="checkpoint-instrumentation equivalence gate",
        )
        instrumentation_equivalence = validate_instrumentation_equivalence_gate(
            gate_path=instrumentation_equivalence_path,
            expected_legacy_inventory_sha256=LEGACY_SOURCE_INVENTORY_SHA256,
            expected_instrumentation_inventory_sha256=(
                INSTRUMENTATION_SOURCE_INVENTORY_SHA256
            ),
        )
        equivalence = _json(equivalence_path, label="trace-mode equivalence gate")
        if (
            equivalence.get("schema_version")
            != "mut_trace_on_off_500_step_equivalence_v1"
            or equivalence.get("status") != "PASS"
            or equivalence.get("trace_on_off_stepwise_exact") is not True
            or equivalence.get("first_semantic_divergence_step") is not None
            or equivalence.get("trace_on_checkpoint_reload_pass") is not True
            or equivalence.get("trace_off_checkpoint_reload_pass") is not True
            or equivalence.get("post_reload_trace_mode_equivalence_pass") is not True
            or equivalence.get("trace_on_trace_enabled") is not True
            or equivalence.get("trace_off_trace_enabled") is not False
            or equivalence.get(
                "trace_only_files_excluded_from_scientific_digest"
            )
            is not True
            or equivalence.get("step_action_trace_exact") is not True
            or equivalence.get("rng_state_exact") is not True
            or equivalence.get("classifier_probability_trace_exact") is not True
            or equivalence.get("step_semantic_fields_present") is not True
            or equivalence.get(
                "step500_checkpoint_serialized_candidate_records_exact"
            )
            is not True
            or equivalence.get("step500_checkpoint_candidate_universe_exact")
            is not True
            or equivalence.get("checkpoint_algorithm_scientific_state_exact")
            is not True
            or equivalence.get("checkpoint_rng_state_exact") is not True
            or equivalence.get("checkpoint_sqlite_logical_state_exact") is not True
            or equivalence.get("checkpoint_graph_registry_exact") is not True
            or equivalence.get("resolved_config_scientific_binding_exact")
            is not True
            or equivalence.get("post_walk_prefix_finalization_performed") is not False
            or equivalence.get(
                "post_walk_candidate_semantics_bound_by_static_audit"
            )
            is not True
            or equivalence.get("full_50k_trace_on_off_parity_claimed") is not False
            or equivalence.get("arms_overlapped") is not False
            or int(equivalence.get("max_concurrent_arms", -1)) != 1
            or int(equivalence.get("steps_compared", -1)) != 500
            or int(equivalence.get("post_reload_steps_compared", -1)) != 10
            or equivalence.get("calibration_loaded") is not False
            or equivalence.get("test_loaded") is not False
            or not isinstance(equivalence.get("checkpoint_gates"), Mapping)
            or not equivalence["checkpoint_gates"]
            or any(
                value is not True
                for value in equivalence["checkpoint_gates"].values()
            )
        ):
            raise MutFastError("Trace-on/off 500-step equivalence gate is invalid")
        expected_equivalence_sha = equivalence.get("summary_sha256")
        equivalence_unhashed = {
            key: value for key, value in equivalence.items() if key != "summary_sha256"
        }
        if expected_equivalence_sha != stable_json_sha256(equivalence_unhashed):
            raise MutFastError("Trace-mode equivalence self hash changed")
        input_manifest_path = _absolute(
            equivalence.get("input_manifest"),
            label="trace-mode equivalence input manifest",
        )
        input_manifest = _json(
            input_manifest_path, label="trace-mode equivalence input manifest"
        )
        input_unhashed = {
            key: value
            for key, value in input_manifest.items()
            if key != "manifest_sha256"
        }
        rf_binding = input_manifest.get("rf_oracle")
        input_files = input_manifest.get("input_files")
        gnn_binding = (
            input_files.get("gnn_checkpoint")
            if isinstance(input_files, Mapping)
            else None
        )
        distance_binding = (
            input_files.get("distance_checkpoint")
            if isinstance(input_files, Mapping)
            else None
        )
        if (
            sha256_file(input_manifest_path)
            != equivalence.get("input_manifest_sha256")
            or input_manifest.get("schema_version")
            != "mut_trace_equivalence_input_manifest_v1"
            or input_manifest.get("manifest_sha256")
            != stable_json_sha256(input_unhashed)
            or input_manifest.get("source_algorithm_commit")
            != SOURCE_PROJECT_COMMIT
            or input_manifest.get("execution_commit")
            != INSTRUMENTATION_PROJECT_COMMIT
            or input_manifest.get("upstream_commit") != MUT_UPSTREAM_COMMIT
            or int(input_manifest.get("formal_M_MAX", -1)) != 50_000
            or int(input_manifest.get("candidate_capacity", -1)) != 100_000
            or int(input_manifest.get("seed", -1)) != 0
            or int(input_manifest.get("parent_limit", -1)) != 1448
            or int(input_manifest.get("batch_size", -1)) != 128
            or input_manifest.get("device") != "cuda:0"
            or input_manifest.get("pythonhashseed") != "0"
            or input_manifest.get("algorithm_registry_identity")
            != "pinned_upstream_embedding_bytes_python_hash_seed0"
            or input_manifest.get("audit_graph_identity")
            != "stable_untyped_graph_sha256"
            or input_manifest.get("dataset_sha256") != SOURCE_DATASET_SHA256
            or input_manifest.get("split_source_cohort_sha256")
            != SOURCE_PARENT_ORDER_SHA256
            or input_manifest.get("calibration_loaded") is not False
            or input_manifest.get("test_loaded") is not False
            or input_manifest.get("historical_artifact_root")
            != str(
                _absolute(
                    spec["historical_source_root"], label="historical source"
                )
            )
            or not isinstance(rf_binding, Mapping)
            or rf_binding.get("loaded_by_generation_canary")
            is not False
            or rf_binding.get("sha256") != MUT_RF_ORACLE_SHA256
            or not isinstance(gnn_binding, Mapping)
            or gnn_binding.get("sha256") != MUT_GNN_SHA256
            or not isinstance(distance_binding, Mapping)
            or distance_binding.get("sha256") != MUT_DISTANCE_SHA256
        ):
            raise MutFastError("Trace-mode frozen input manifest is invalid")
        memory_path = _absolute(
            canary_memory_receipt, label="trace-mode canary memory receipt"
        )
        memory = _json(memory_path, label="trace-mode canary memory receipt")
        memory_unhashed = {
            key: value for key, value in memory.items() if key != "summary_sha256"
        }
        phases = memory.get("phases")
        required_phases = {
            "trace_on_continuous",
            "trace_on_reload",
            "trace_off_continuous",
            "trace_off_reload",
        }
        if (
            memory.get("schema_version") != "mut_trace_mode_canary_memory_v1"
            or memory.get("status") != "PASS"
            or int(memory.get("initial_parent_headroom_bytes", -1))
            < CANARY_REQUIRED_HEADROOM_BYTES
            or int(memory.get("process_rss_peak_bytes", -1))
            > CANARY_RSS_STOP_BYTES
            or int(memory.get("parent_headroom_min_bytes", -1))
            < CANARY_HEADROOM_STOP_BYTES
            or int(memory.get("cgroup_failcnt_delta", -1)) != 0
            or int(memory.get("cgroup_oom_delta", -1)) != 0
            or int(memory.get("cgroup_oom_kill_delta", -1)) != 0
            or not isinstance(phases, Mapping)
            or not required_phases.issubset(phases)
            or any(
                not isinstance(phases[name], Mapping)
                or phases[name].get("status") != "PASS"
                or int(phases[name].get("sample_count", 0)) <= 0
                or int(phases[name].get("peak_rss_bytes", -1))
                > CANARY_RSS_STOP_BYTES
                or int(phases[name].get("minimum_parent_headroom_bytes", -1))
                < CANARY_HEADROOM_STOP_BYTES
                for name in required_phases
            )
            or not isinstance(memory.get("protected_throughput_gate"), Mapping)
            or memory["protected_throughput_gate"].get("status") != "PASS"
            or memory["protected_throughput_gate"].get(
                "missing_complete_five_minute_windows"
            )
            not in ([], None)
            or memory.get("summary_sha256")
            != stable_json_sha256(memory_unhashed)
        ):
            raise MutFastError("Trace-mode canary memory/throughput gate is invalid")
        equivalence = {
            **equivalence,
            "sha256": sha256_file(equivalence_path),
            "checkpoint_resume_exercised": bool(
                equivalence.get("trace_on_checkpoint_reload_pass") is True
                and equivalence.get("trace_off_checkpoint_reload_pass") is True
            ),
            "checkpoint_mirror_verified": bool(
                isinstance(
                    equivalence.get("trace_on_checkpoint_state_audit"), Mapping
                )
                and isinstance(
                    equivalence.get("trace_off_checkpoint_state_audit"), Mapping
                )
            ),
        }
    else:
        equivalence = validate_instrumentation_equivalence_gate(
            gate_path=equivalence_path,
            expected_legacy_inventory_sha256=LEGACY_SOURCE_INVENTORY_SHA256,
            expected_instrumentation_inventory_sha256=(
                INSTRUMENTATION_SOURCE_INVENTORY_SHA256
            ),
        )
    source_root = _absolute(spec["historical_source_root"], label="historical source")
    common_root = _absolute(spec["completed_common_root"], label="completed common")
    source_manifest_path = source_root / "run_manifest.json"
    common_manifest_path = common_root / "run_manifest.json"
    pair_adoption_manifest_path = (
        common_root / "external_memory/pair_store_adoption/run_manifest.json"
    )
    dbscan_manifest_path = common_root / "external_memory/dbscan/run_manifest.json"
    source_manifest = _json(source_manifest_path, label="source manifest")
    common_manifest = _json(common_manifest_path, label="common manifest")
    pair_adoption_manifest = _json(
        pair_adoption_manifest_path, label="pair-store adoption manifest"
    )
    pair_manifest_path = _absolute(
        pair_adoption_manifest.get("source_manifest_path"),
        label="source pair-store manifest",
    )
    pair_manifest = _json(pair_manifest_path, label="source pair-store manifest")
    dbscan_manifest = _json(dbscan_manifest_path, label="DBSCAN manifest")
    source_manifest_sha = sha256_file(source_manifest_path)
    pair_manifest_sha = sha256_file(pair_manifest_path)
    dbscan_manifest_sha = sha256_file(dbscan_manifest_path)
    failures: list[str] = []
    source_payload_path = source_root / "counterfactuals.pt"
    if sha256_file(source_payload_path) != SOURCE_PAYLOAD_SHA256:
        failures.append("source_payload_bytes")
    if source_manifest.get("counterfactuals_sha256") != SOURCE_PAYLOAD_SHA256:
        failures.append("source_payload")
    if common_manifest.get("run_complete") is not True:
        failures.append("common_complete")
    if common_manifest.get("counterfactuals_sha256") != SOURCE_PAYLOAD_SHA256:
        failures.append("common_payload")
    generation_sha = str(common_manifest.get("generation_manifest_sha256") or "")
    if generation_sha and generation_sha != source_manifest_sha:
        failures.append("common_generation_manifest_sha256")
    scientific = pair_manifest.get("scientific_identity")
    if not isinstance(scientific, Mapping):
        failures.append("pair_scientific_identity")
        scientific = {}
    expected_pair = {
        "counterfactuals_sha256": SOURCE_PAYLOAD_SHA256,
        "dataset_fingerprint": SOURCE_DATASET_SHA256,
        "parent_ids_sha256": SOURCE_PARENT_ORDER_SHA256,
        "generation_manifest_sha256": source_manifest_sha,
    }
    for key, expected in expected_pair.items():
        if scientific.get(key) != expected:
            failures.append(f"pair.{key}")
    candidate_universe = str(scientific.get("candidate_graph_hashes_sha256") or "")
    if HEX64.fullmatch(candidate_universe) is None:
        failures.append("candidate_universe_sha256")
    if authorization is not None and candidate_universe != EXPECTED_CANDIDATE_UNIVERSE_SHA256:
        failures.append("authorized_candidate_universe_sha256")
    if int(scientific.get("candidate_count", -1)) != 50_620:
        failures.append("pair_candidate_count")
    source_pair_manifest = pair_manifest_path
    source_pair_manifest_sha = pair_manifest_sha
    expected_source_pair_sha = str(
        pair_adoption_manifest.get("source_manifest_sha256") or ""
    )
    if expected_source_pair_sha and expected_source_pair_sha != source_pair_manifest_sha:
        failures.append("source_pair_store_manifest_sha256")
    if dbscan_manifest.get("run_complete") is not True:
        failures.append("dbscan_complete")
    if dbscan_manifest.get("approximation_used") is not False:
        failures.append("dbscan_approximation")
    external = common_manifest.get("external_memory_artifacts")
    dbscan_identity = dbscan_manifest.get("scientific_identity")
    if (
        not isinstance(external, Mapping)
        or external.get("engine") != "external_memory_exact_v1"
        or Path(str(external.get("pair_store_manifest") or "")).resolve(
            strict=True
        )
        != pair_manifest_path
        or external.get("pair_store_manifest_sha256") != pair_manifest_sha
        or Path(str(external.get("dbscan_manifest") or "")).resolve(strict=True)
        != dbscan_manifest_path.resolve(strict=True)
        or external.get("dbscan_manifest_sha256") != dbscan_manifest_sha
    ):
        failures.append("common_external_memory_binding")
    if (
        not isinstance(dbscan_identity, Mapping)
        or dbscan_identity.get("vectors_path") != pair_manifest.get("vectors_path")
        or dbscan_identity.get("vectors_sha256")
        != pair_manifest.get("vectors_sha256")
    ):
        failures.append("dbscan_pair_store_vector_binding")
    dbscan_native_universe_fields = {
        "source_candidate_universe_sha256",
        "candidate_universe_sha256",
        "candidate_graph_hashes_sha256",
    }
    if any(key in dbscan_manifest for key in dbscan_native_universe_fields) or (
        isinstance(dbscan_identity, Mapping)
        and any(key in dbscan_identity for key in dbscan_native_universe_fields)
    ):
        failures.append("dbscan_native_candidate_universe_field_present")
    if failures:
        raise MutFastError(f"Historical/common binding failed: {failures}")
    try:
        universe_binding = verify_mut_candidate_pair_dbscan_binding(
            source_payload_path=source_payload_path,
            pair_manifest_path=pair_manifest_path,
            dbscan_manifest_path=dbscan_manifest_path,
            expected_candidate_universe_sha256=(
                EXPECTED_CANDIDATE_UNIVERSE_SHA256
            ),
            expected_source_payload_sha256=SOURCE_PAYLOAD_SHA256,
            expected_candidate_count=50_620,
            candidate_capacity=100_000,
        )
    except Exception as exc:
        raise MutFastError(
            f"Historical candidate/pair/DBSCAN transitive binding failed: {exc}"
        ) from exc
    if (
        universe_binding.get("status") != "PASS"
        or universe_binding.get("source_native_candidate_universe_sha")
        != candidate_universe
        or universe_binding.get("pair_store_source_candidate_universe_sha")
        != candidate_universe
        or universe_binding.get("dbscan_native_candidate_universe_sha") is not None
        or universe_binding.get(
            "dbscan_transitively_bound_candidate_universe_sha"
        )
        != candidate_universe
        or universe_binding.get("dbscan_approximation_used") is not False
        or universe_binding.get("binding_kind")
        != "transitive_generation_pair_store_vectors_dbscan_v1"
    ):
        raise MutFastError(
            "Historical candidate/pair/DBSCAN binding receipt is inconsistent"
        )
    # Require the completed common manifest to mention both exact subordinate
    # identities.  Field names may differ across the already-frozen producer.
    _require_bound(common_manifest, pair_manifest_sha, label="pair-store manifest SHA")
    _require_bound(common_manifest, dbscan_manifest_sha, label="DBSCAN manifest SHA")
    source_writer = scan_live_writers(source_root, proc_root=spec["proc_root"])
    common_writer = scan_live_writers(common_root, proc_root=spec["proc_root"])
    pair_writer = scan_live_writers(
        pair_manifest_path.parent, proc_root=spec["proc_root"]
    )
    source_evidence = inventory.get("source")
    lineage_evidence = inventory.get("lineage")
    verified_contract = {
        "semantic_equivalence": (
            equivalence.get("status") == "PASS"
            and equivalence.get("step_action_trace_exact") is True
            and equivalence.get("rng_state_exact") is True
            and (
                authorization is None
                or (
                    instrumentation_equivalence is not None
                    and instrumentation_equivalence.get("status") == "PASS"
                    and instrumentation_equivalence.get(
                        "step_action_trace_exact"
                    )
                    is True
                    and instrumentation_equivalence.get("rng_state_exact")
                    is True
                )
            )
        ),
        "checkpoint_reload": (
            instrumentation_equivalence is not None
            and instrumentation_equivalence.get("checkpoint_resume_exercised")
            is True
            and instrumentation_equivalence.get("checkpoint_mirror_verified")
            is True
            and equivalence.get("trace_on_checkpoint_reload_pass") is True
            and equivalence.get("trace_off_checkpoint_reload_pass") is True
            and equivalence.get("post_reload_trace_mode_equivalence_pass") is True
        ),
        "generation_complete": (
            isinstance(source_evidence, Mapping)
            and source_evidence.get("status") == "PASS"
            and int(source_evidence.get("source_candidate_count", -1))
            == SOURCE_CANDIDATE_COUNT
        ),
        "candidate_freeze": (
            isinstance(source_evidence, Mapping)
            and source_evidence.get("source_payload_actual_sha256")
            == SOURCE_PAYLOAD_SHA256
        ),
        "lineage": (
            isinstance(lineage_evidence, Mapping)
            and int(lineage_evidence.get("candidate_count", -1))
            == SOURCE_CANDIDATE_COUNT
            and int(lineage_evidence.get("candidate_lineage_resolved_count", -2))
            == SOURCE_CANDIDATE_COUNT
            and int(lineage_evidence.get("recorded_action_replay_mismatch_count", -1))
            == 0
        ),
        "no_test_leakage": (
            isinstance(source_evidence, Mapping)
            and source_evidence.get("calibration_loaded") is False
            and source_evidence.get("test_loaded") is False
            and equivalence.get("calibration_loaded") is False
            and equivalence.get("test_loaded") is False
        ),
        "no_active_writer": (
            int(source_writer.get("writable_fd_count", -1)) == 0
            and int(common_writer.get("writable_fd_count", -1)) == 0
            and int(pair_writer.get("writable_fd_count", -1)) == 0
        ),
    }
    failed_contract = sorted(
        key for key, passed in verified_contract.items() if passed is not True
    )
    if failed_contract:
        raise MutFastError(
            f"Historical adoption evidence is incomplete: {failed_contract}"
        )
    source_parameters = source_manifest.get("parameters")
    if not isinstance(source_parameters, Mapping):
        raise MutFastError("Historical generation parameters are absent")
    generation_steps = int(source_parameters.get("steps", -1))
    candidate_capacity = int(source_parameters.get("candidate_capacity", -1))
    common_recourse_count = int(common_manifest.get("common_recourse_count", -1))
    if generation_steps != 50_000 or candidate_capacity != 100_000:
        raise MutFastError("Historical generation budget changed")
    if common_recourse_count <= 0:
        raise MutFastError("Completed common-recourse count is invalid")
    output = _absolute(output_dir, label="adoption output", exists=False)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"adoption output must be fresh: {output}")
    payload = {
        "schema_version": ADOPTION_SCHEMA,
        "status": "PASS",
        "dataset": "mutagenicity",
        "method": "COMRECGC",
        "historical_artifact_adopted": True,
        "historical_generation_adopted": True,
        "historical_source_trace_enabled": True,
        "trace_parity_passed": False,
        "traceoff_reference_rerun": False,
        "full_50k_rerun_performed": False,
        "source_trace_enabled": True,
        "target_default_trace_mode": False,
        "trace_is_observational": authorization is not None,
        "trace_on_off_500_step_equivalence_pass": authorization is not None,
        "trace_off_full_rerun_performed": False,
        "full_trace_on_off_parity_claimed": False,
        "500_step_semantic_equivalence_passed": verified_contract[
            "semantic_equivalence"
        ],
        "adoption_without_full_50k_parity_rerun_authorized": spec[
            "allow_historical_adoption_without_full_50k_parity"
        ],
        "checkpoint_instrumentation_equivalence_passed": (
            instrumentation_equivalence is not None
            or authorization is None
        ),
        "checkpoint_instrumentation_equivalence_steps": 500,
        "generation_complete": verified_contract["generation_complete"],
        "generation_steps": generation_steps,
        "M_MAX": generation_steps,
        "M_EFFECTIVE": generation_steps,
        "candidate_capacity": candidate_capacity,
        "candidate_count": SOURCE_CANDIDATE_COUNT,
        "lineage_pass": verified_contract["lineage"],
        "candidate_freeze_pass": verified_contract["candidate_freeze"],
        "checkpoint_reload_pass": verified_contract["checkpoint_reload"],
        "no_test_leakage": verified_contract["no_test_leakage"],
        "no_active_writer": verified_contract["no_active_writer"],
        "lineage_verified": True,
        "lineage_contract": inventory["lineage"],
        "M_configured_max": 50_000,
        "M_effective": 50_000,
        "resource_cap_used": False,
        "early_stop_used": False,
        "stop_reason": "historical_completed_50k_artifact_adoption",
        "generation_rerun": False,
        "common_recourse_reused": True,
        "common_recourse_rerun": False,
        "pair_store_reused": True,
        "dbscan_reused": True,
        "pair_store_recompute_performed": False,
        "dbscan_recompute_performed": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "source_generation_root": str(source_root),
        "source_generation_manifest": str(source_manifest_path),
        "source_generation_manifest_sha256": source_manifest_sha,
        "source_payload_path": str(source_payload_path),
        "source_payload_sha256": SOURCE_PAYLOAD_SHA256,
        "source_candidate_count": SOURCE_CANDIDATE_COUNT,
        "source_lineage_path": inventory["lineage"]["path"],
        "source_lineage_sha256": inventory["lineage"]["sha256"],
        "completed_common_root": str(common_root),
        "source_common_recourse_root": str(common_root),
        "source_common_recourse_manifest_path": str(common_manifest_path),
        "source_common_recourse_manifest_sha256": sha256_file(
            common_manifest_path
        ),
        "common_recourse_count": common_recourse_count,
        "common_manifest_sha256": sha256_file(common_manifest_path),
        "pair_store_manifest": str(pair_manifest_path),
        "pair_store_manifest_sha256": pair_manifest_sha,
        "pair_store_adoption_manifest_path": str(pair_adoption_manifest_path),
        "pair_store_adoption_manifest_sha256": sha256_file(
            pair_adoption_manifest_path
        ),
        "source_pair_store_manifest": str(source_pair_manifest),
        "source_pair_store_manifest_sha256": source_pair_manifest_sha,
        "source_pair_store_manifest_path": str(source_pair_manifest),
        "dbscan_manifest": str(dbscan_manifest_path),
        "dbscan_manifest_sha256": dbscan_manifest_sha,
        "source_dbscan_manifest_path": str(dbscan_manifest_path),
        "source_dbscan_manifest_sha256": dbscan_manifest_sha,
        "candidate_universe_sha": candidate_universe,
        "source_native_candidate_universe_sha": candidate_universe,
        "pair_store_source_candidate_universe_sha": candidate_universe,
        "dbscan_native_candidate_universe_sha": None,
        "dbscan_transitively_bound_candidate_universe_sha": candidate_universe,
        "candidate_universe_binding_state": "PASS",
        "transitive_binding_kind": universe_binding["binding_kind"],
        "dbscan_native_candidate_universe_field_present": False,
        "dbscan_universe_binding_via_pair_vectors": True,
        "candidate_pair_dbscan_binding_receipt": universe_binding,
        "candidate_pair_dbscan_binding_sha256": universe_binding[
            "binding_sha256"
        ],
        "pair_candidate_graph_hashes_sha256": candidate_universe,
        "candidate_universe_binding": (
            "generation_payload_to_pair_store_scientific_identity_to_exact_dbscan"
        ),
        "inventory_path": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "equivalence_path": str(equivalence_path),
        "equivalence_sha256": equivalence["sha256"],
        "500_step_semantic_equivalence_receipt_path": str(equivalence_path),
        "500_step_semantic_equivalence_receipt_sha256": equivalence["sha256"],
        "trace_on_adoption_authorization_path": (
            str(_absolute(authorization_receipt, label="trace-on authorization"))
            if authorization_receipt is not None
            else None
        ),
        "trace_on_adoption_authorization_file_sha256": authorization_file_sha256,
        "trace_code_observational_audit_path": str(audit_path) if audit_path else None,
        "trace_code_observational_audit_sha256": (
            sha256_file(audit_path) if audit_path else None
        ),
        "trace_mode_equivalence_path": str(equivalence_path),
        "trace_mode_equivalence_sha256": sha256_file(equivalence_path),
        "checkpoint_instrumentation_equivalence_path": (
            str(instrumentation_equivalence_path)
            if instrumentation_equivalence_path
            else str(equivalence_path)
        ),
        "checkpoint_instrumentation_equivalence_sha256": (
            sha256_file(instrumentation_equivalence_path)
            if instrumentation_equivalence_path
            else equivalence["sha256"]
        ),
        "trace_mode_canary_memory_receipt_path": (
            str(memory_path) if memory_path else None
        ),
        "trace_mode_canary_memory_receipt_sha256": (
            sha256_file(memory_path) if memory_path else None
        ),
        "trace_operational_side_effects_disclosed": authorization is not None,
        "historical_random_walk_complete": True,
        "historical_post_walk_trace_freeze_failed": authorization is not None,
        "freeze_only_recovery_performed": authorization is not None,
        "freeze_only_recovery_code_commit_attested": False,
        "source_live_writer_audit": source_writer,
        "common_live_writer_audit": common_writer,
        "pair_store_live_writer_audit": pair_writer,
        "published_at": _utc_now(),
    }
    output.mkdir(parents=True)
    universe_binding_path = output / "candidate_universe_binding.json"
    _atomic_json(universe_binding_path, universe_binding)
    payload["candidate_pair_dbscan_binding_path"] = str(universe_binding_path)
    payload["candidate_pair_dbscan_binding_file_sha256"] = sha256_file(
        universe_binding_path
    )
    payload["binding_sha256"] = stable_json_sha256(payload)
    _atomic_json(output / "historical_adoption.json", payload)
    _write_pass(output / "PASS")
    return payload


def _mount_is_read_only(mountinfo: str, mountpoint: Path) -> bool:
    wanted = str(mountpoint)
    for line in mountinfo.splitlines():
        before, separator, _after = line.partition(" - ")
        if not separator:
            continue
        fields = before.split()
        if len(fields) >= 6 and fields[4] == wanted:
            return "ro" in fields[5].split(",")
    raise MutFastError(f"cgroup mountpoint is absent from mountinfo: {wanted}")


def inspect_admission(spec: Mapping[str, Any], *, require_gpu: bool = True) -> dict[str, Any]:
    root = _absolute(spec["cgroup_memory_root"], label="cgroup memory root")
    limit_path = root / "memory.limit_in_bytes"
    usage_path = root / "memory.usage_in_bytes"
    try:
        limit = int(limit_path.read_text(encoding="utf-8").strip())
        usage = int(usage_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise MutFastError("cgroup-v1 memory counters are unavailable") from exc
    mountinfo_path = _absolute(spec["mountinfo_path"], label="mountinfo")
    mount_ro = _mount_is_read_only(mountinfo_path.read_text(encoding="utf-8"), root)
    if not mount_ro:
        raise MutFastError("configured cgroup-v1 mount is not read-only as frozen")
    headroom = max(0, limit - usage)
    gpus: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    gpu_error: str | None = None
    if require_gpu:
        try:
            for gpu in query_gpu_inventory():
                record = gpu.as_json()
                record["project_lock_available"] = gpu_lock_available(
                    _absolute(spec["runtime_root"], label="runtime root") / "locks",
                    gpu.uuid,
                )
                gpus.append(record)
                if (
                    gpu.process_count == 0
                    and gpu.memory_free_mb >= int(spec["gpu_min_free_memory_mb"])
                    and gpu.utilization_gpu_percent <= int(
                        spec["gpu_max_utilization_percent"]
                    )
                    and record["project_lock_available"]
                ):
                    eligible.append(record)
        except Exception as exc:  # inventory uncertainty is a wait, not a launch
            gpu_error = f"{type(exc).__name__}: {exc}"
    admitted = headroom >= MINIMUM_HEADROOM_BYTES and (bool(eligible) or not require_gpu)
    return {
        "schema_version": "mut_empirical_admission_observation_v2",
        "state": "ADMITTED" if admitted else "WAITING_FOR_EMPIRICAL_ADMISSION",
        "cgroup_version": 1,
        "cgroup_mount_read_only": True,
        "child_cgroup_created": False,
        "no_child_reason": spec["no_child_fallback_reason"],
        "systemd_scope_available": False,
        "limit_bytes": limit,
        "usage_bytes": usage,
        "parent_headroom_bytes": headroom,
        "required_parent_headroom_bytes": MINIMUM_HEADROOM_BYTES,
        "headroom_pass": headroom >= MINIMUM_HEADROOM_BYTES,
        "eligible_idle_exclusive_gpus": eligible,
        "gpu_observations": gpus,
        "gpu_inventory_error": gpu_error,
        "observed_at": _utc_now(),
    }


def _science_children(proc_root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in proc_root.iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid():
            continue
        try:
            command = (item / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except OSError:
            continue
        equivalence = (
            "run_mut_checkpoint_instrumentation_equivalence.py" in command
            and "run-pair" in command
        )
        generation = (
            "scripts/baselines/comrecgc/run_generation.py" in command
            and "mutagenicity" in command
        )
        if equivalence or generation:
            result.append({"pid": int(item.name), "command": command})
    return result


def _cgroup_events(root: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for filename in ("memory.failcnt", "memory.oom_control"):
        path = root / filename
        if not path.is_file():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if filename == "memory.failcnt" and len(fields) == 1:
                    result["failcnt"] = int(fields[0])
                elif len(fields) == 2 and fields[0] in {"oom_kill", "under_oom"}:
                    result[fields[0]] = int(fields[1])
        except (OSError, ValueError):
            continue
    return result


def _process_tree_memory(proc_root: Path, root_pid: int) -> dict[str, Any]:
    parents: dict[int, int] = {}
    for item in proc_root.iterdir():
        if not item.name.isdigit():
            continue
        try:
            raw = (item / "stat").read_text(encoding="utf-8")
            closing = raw.rfind(")")
            parents[int(item.name)] = int(raw[closing + 2 :].split()[1])
        except (OSError, ValueError, IndexError):
            continue
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in selected and pid not in selected:
                selected.add(pid)
                changed = True
    rss_kib = 0
    pss_kib = 0
    live: list[int] = []
    for pid in sorted(selected):
        status = proc_root / str(pid) / "status"
        if not status.is_file():
            continue
        live.append(pid)
        try:
            for line in status.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    rss_kib += int(line.split()[1])
                    break
            rollup = proc_root / str(pid) / "smaps_rollup"
            if rollup.is_file():
                for line in rollup.read_text(encoding="utf-8").splitlines():
                    if line.startswith("Pss:"):
                        pss_kib += int(line.split()[1])
                        break
        except (OSError, ValueError, IndexError):
            continue
    return {
        "root_pid": root_pid,
        "live_pids": live,
        "aggregate_rss_bytes": rss_kib * 1024,
        "aggregate_pss_bytes": pss_kib * 1024,
    }


def _process_start_ticks(proc_root: Path, pid: int) -> int:
    raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    closing = raw.rfind(")")
    fields = raw[closing + 2 :].split()
    return int(fields[19])


def _terminate_exact_process_group(
    process: subprocess.Popen[Any], *, pgid: int, start_ticks: int,
    proc_root: Path, reason: str,
) -> None:
    _stop_exact_process_group(
        process,
        pgid=pgid,
        start_ticks=start_ticks,
        proc_root=proc_root,
    )
    raise MutFastError(f"equivalence watchdog stopped exact PGID: {reason}")


def _stop_exact_process_group(
    process: subprocess.Popen[Any], *, pgid: int, start_ticks: int,
    proc_root: Path,
) -> None:
    """SIGTERM only the fresh equivalence session and wait for its root."""

    if process.poll() is not None:
        return
    if _process_start_ticks(proc_root, process.pid) != start_ticks:
        raise MutFastError("equivalence PID identity changed before watchdog stop")
    if os.getpgid(process.pid) != pgid or pgid != process.pid:
        raise MutFastError("equivalence process-group identity changed")
    os.killpg(pgid, signal.SIGTERM)
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired as exc:
        raise MutFastError(
            "equivalence process group ignored SIGTERM; SIGKILL is forbidden"
        ) from exc


def run_equivalence_monitored(
    spec: Mapping[str, Any], *, run_root: Path, output_dir: Path
) -> dict[str, Any]:
    """Spawn the exact 500-step pair and sample cgroup/process memory every 10s."""

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise MutFastError("monitored equivalence requires an assigned exclusive GPU")
    run_path = _absolute(run_root, label="equivalence run root", exists=False)
    output = _absolute(output_dir, label="equivalence output", exists=False)
    if run_path.exists() or output.exists():
        raise FileExistsError("equivalence roots must both be fresh")
    monitor = output.parent / f".{output.name}.memory-monitor"
    if monitor.exists() or monitor.is_symlink():
        raise FileExistsError(f"memory monitor root must be fresh: {monitor}")
    monitor.mkdir(parents=True)
    monitor_jsonl = monitor / "memory_monitor.jsonl"
    runner_log = monitor / "equivalence.log"
    replay = dict(spec["replay"])
    semantic_finalizer_root = _absolute(
        spec.get("semantic_finalizer_project_root")
        or os.environ.get("MUT_SEMANTIC_FINALIZER_PROJECT_ROOT")
        or DEFAULT_SEMANTIC_FINALIZER_PROJECT_ROOT,
        label="exact 582 semantic finalizer worktree",
    )
    command = [
        str(spec["python"]),
        str(PROJECT_ROOT / "scripts/autodl/run_mut_checkpoint_instrumentation_equivalence.py"),
        "--config", "configs/hpc.yaml", "--set", "inference.fallback_to_heuristic=false",
        "run-pair", "--python", str(spec["python"]),
        "--legacy-project-root", str(spec["legacy_project_root"]),
        "--execution-project-root", str(spec["instrumentation_project_root"]),
        "--execution-commit", INSTRUMENTATION_PROJECT_COMMIT,
        "--expected-legacy-inventory-sha256", LEGACY_SOURCE_INVENTORY_SHA256,
        "--expected-instrumentation-inventory-sha256",
        INSTRUMENTATION_SOURCE_INVENTORY_SHA256,
        "--run-root", str(run_path), "--output-dir", str(output),
        "--upstream-root", str(replay["upstream_root"]),
        "--dataset-dir", str(replay["dataset_dir"]),
        "--gnn-checkpoint", str(replay["gnn_checkpoint"]),
        "--distance-checkpoint", str(replay["distance_checkpoint"]),
        "--parent-limit", "1448", "--device", "cuda:0", "--batch-size", "128",
        "--semantic-finalizer-project-root", str(semantic_finalizer_root),
    ]
    cgroup = _absolute(spec["cgroup_memory_root"], label="cgroup memory root")
    proc_root = _absolute(spec["proc_root"], label="proc root")
    initial_snapshot = read_cgroup_snapshot(cgroup, version="v1")
    initial_events = _cgroup_events(cgroup)
    peak_rss = peak_pss = checkpoint_window_peak = samples = pressure_samples = 0
    peak_parent_current = int(initial_snapshot["memory_current_bytes"])
    previous_process_peak = 0
    observed_checkpoint_markers: set[str] = set()
    checkpoint_peak_events: list[dict[str, Any]] = []
    started = _utc_now()
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    with runner_log.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            command, cwd=spec["project_root"], env=environment,
            stdout=log_handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        process_start_ticks = _process_start_ticks(proc_root, process.pid)
        process_pgid = os.getpgid(process.pid)
        if process_pgid != process.pid:
            _stop_exact_process_group(
                process,
                pgid=process_pgid,
                start_ticks=process_start_ticks,
                proc_root=proc_root,
            )
            raise MutFastError("start_new_session did not create an isolated process group")
        try:
            with monitor_jsonl.open("a", encoding="utf-8") as monitor_handle:
                while process.poll() is None:
                    snapshot = read_cgroup_snapshot(cgroup, version="v1")
                    limit = int(snapshot["memory_max_bytes"])
                    usage = int(snapshot["memory_current_bytes"])
                    tree = _process_tree_memory(proc_root, process.pid)
                    events = _cgroup_events(cgroup)
                    peak_rss = max(peak_rss, int(tree["aggregate_rss_bytes"]))
                    peak_pss = max(peak_pss, int(tree["aggregate_pss_bytes"]))
                    peak_parent_current = max(peak_parent_current, usage)
                    headroom = max(0, limit - usage)
                    pressure_samples = (
                        pressure_samples + 1 if usage > 0.8 * limit else 0
                    )
                    checkpoint_markers = {
                        str(path.resolve())
                        for path in run_path.glob(
                            "instrumented-checkpoint-mirror/step-*/"
                            "_CHECKPOINT_MIRRORED.json"
                        )
                    }
                    new_checkpoint_markers = sorted(
                        checkpoint_markers - observed_checkpoint_markers
                    )
                    current_process_peak = max(
                        int(tree["aggregate_rss_bytes"]),
                        int(tree["aggregate_pss_bytes"]),
                    )
                    if new_checkpoint_markers:
                        event_peak = max(previous_process_peak, current_process_peak)
                        checkpoint_window_peak = max(
                            checkpoint_window_peak,
                            event_peak,
                        )
                        checkpoint_peak_events.append(
                            {
                                "sample": samples,
                                "markers": new_checkpoint_markers,
                                "pre_sample_process_peak_bytes": previous_process_peak,
                                "post_sample_process_peak_bytes": current_process_peak,
                                "checkpoint_window_peak_bytes": event_peak,
                            }
                        )
                        observed_checkpoint_markers.update(new_checkpoint_markers)
                    sample = {
                        "sample": samples,
                        "sampled_at": _utc_now(),
                        "cgroup_version": 1,
                        "child_cgroup_created": False,
                        "no_child_reason": spec["no_child_fallback_reason"],
                        "cgroup_limit_bytes": limit,
                        "cgroup_usage_bytes": usage,
                        "cgroup_headroom_bytes": headroom,
                        "cgroup_events": events,
                        "cgroup_snapshot": snapshot,
                        "process_tree": tree,
                        "process_identity": {
                            "pid": process.pid,
                            "start_ticks": process_start_ticks,
                            "pgid": process_pgid,
                        },
                        "checkpoint_markers": sorted(checkpoint_markers),
                        "new_checkpoint_markers": new_checkpoint_markers,
                    }
                    monitor_handle.write(json.dumps(sample, sort_keys=True) + "\n")
                    monitor_handle.flush()
                    os.fsync(monitor_handle.fileno())
                    samples += 1
                    event_deltas = {
                        key: events.get(key, 0) - initial_events.get(key, 0)
                        for key in ("failcnt", "oom_kill")
                    }
                    stop_reasons = []
                    if event_deltas["failcnt"] > 0:
                        stop_reasons.append("memory.failcnt_delta")
                    if event_deltas["oom_kill"] > 0:
                        stop_reasons.append("oom_kill_delta")
                    if events.get("under_oom", 0) > 0:
                        stop_reasons.append("under_oom")
                    if pressure_samples >= 3:
                        stop_reasons.append(
                            "parent_current_gt_0.8_max_for_3_samples"
                        )
                    if stop_reasons:
                        _terminate_exact_process_group(
                            process,
                            pgid=process_pgid,
                            start_ticks=process_start_ticks,
                            proc_root=proc_root,
                            reason=",".join(stop_reasons),
                        )
                    previous_process_peak = current_process_peak
                    time.sleep(10)
            returncode = process.wait()
        except BaseException:
            # Monitoring is the containment boundary.  If any parsing or I/O
            # failure occurs while the exact child session is alive, never
            # leave its generation descendants orphaned.
            if process.poll() is None:
                _stop_exact_process_group(
                    process,
                    pgid=process_pgid,
                    start_ticks=process_start_ticks,
                    proc_root=proc_root,
                )
            raise
    if returncode != 0:
        raise MutFastError(
            f"monitored equivalence failed returncode={returncode}; log={runner_log}"
        )
    if not (output / "PASS").is_file():
        raise MutFastError("equivalence exited zero without PASS")
    monitor_bytes = monitor_jsonl.read_bytes()
    (output / "memory_monitor.jsonl").write_bytes(monitor_bytes)
    equivalence_gate = validate_instrumentation_equivalence_gate(
        gate_path=output / "equivalence.json",
        expected_legacy_inventory_sha256=LEGACY_SOURCE_INVENTORY_SHA256,
        expected_instrumentation_inventory_sha256=(
            INSTRUMENTATION_SOURCE_INVENTORY_SHA256
        ),
    )
    final_events = _cgroup_events(cgroup)
    final_snapshot = read_cgroup_snapshot(cgroup, version="v1")
    event_deltas = {
        "max": max(0, final_events.get("failcnt", 0) - initial_events.get("failcnt", 0)),
        "oom": int(final_events.get("under_oom", 0) > 0),
        "oom_kill": max(
            0, final_events.get("oom_kill", 0) - initial_events.get("oom_kill", 0)
        ),
        "high": 0,
    }
    checkpoint_peak_for_formula = checkpoint_window_peak or peak_rss
    checkpoint_peak_measurement_state = (
        "SAMPLED_PRE_POST_MIRRORED_CHECKPOINT"
        if checkpoint_peak_events
        else "CONSERVATIVE_OVERALL_PROCESS_PEAK_CHECKPOINT_WINDOW_MISSED"
    )
    # With a read-only cgroup-v1 parent there is no attributable child peak.
    # The exact process-tree RSS is used as the conservative formula input;
    # parent max_usage remains explicit, non-attributable evidence only.
    empirical = derive_empirical_memory_admission(
        cgroup_memory_peak_bytes=peak_rss,
        process_peak_rss_bytes=peak_rss,
        checkpoint_peak_bytes=checkpoint_peak_for_formula,
        memory_event_deltas=event_deltas,
        protected_task_slowdown_fraction=0.0,
        semantic_equivalence_pass=equivalence_gate.get("status") == "PASS",
        checkpoint_reload_pass=(
            equivalence_gate.get("checkpoint_resume_exercised") is True
            and equivalence_gate.get("checkpoint_mirror_verified") is True
        ),
    )
    receipt = {
        "schema_version": "mut_empirical_memory_receipt_v2",
        "status": (
            "PASS_WITH_PROTECTED_SLOWDOWN_UNAVAILABLE"
            if empirical["status"] == "PASS"
            else "BLOCKED"
        ),
        "sample_interval_seconds": 10,
        "sample_count": samples,
        "started_at": started,
        "completed_at": _utc_now(),
        "cgroup_version": 1,
        "cgroup_mount_read_only": True,
        "child_cgroup_created": False,
        "no_child_reason": spec["no_child_fallback_reason"],
        "process_identity": {
            "pid": process.pid,
            "start_ticks": process_start_ticks,
            "pgid": process_pgid,
            "start_new_session": True,
        },
        "peak_process_rss_bytes": peak_rss,
        "peak_process_pss_bytes": peak_pss,
        "checkpoint_window_peak_bytes": checkpoint_window_peak,
        "checkpoint_peak_for_formula_bytes": checkpoint_peak_for_formula,
        "checkpoint_peak_measurement_state": checkpoint_peak_measurement_state,
        "checkpoint_peak_events": checkpoint_peak_events,
        "parent_cgroup_peak_current_bytes": peak_parent_current,
        "parent_cgroup_max_usage_bytes": final_snapshot.get("memory_peak_bytes"),
        "parent_cgroup_max_usage_attributable_to_mut": False,
        "mut_attributable_child_cgroup_peak_bytes": None,
        "formula_cgroup_peak_input_bytes": peak_rss,
        "formula_cgroup_peak_input_kind": (
            "exact_process_tree_rss_proxy_no_child_cgroup"
        ),
        "initial_cgroup_events": initial_events,
        "final_cgroup_events": final_events,
        "initial_cgroup_snapshot": initial_snapshot,
        "final_cgroup_snapshot": final_snapshot,
        "oom_kill_delta": final_events.get("oom_kill", 0) - initial_events.get("oom_kill", 0),
        "failcnt_delta": final_events.get("failcnt", 0) - initial_events.get("failcnt", 0),
        "memory_event_deltas_for_formula": event_deltas,
        "protected_task_slowdown_state": "UNAVAILABLE_NO_BASELINE",
        "protected_task_slowdown_fraction": None,
        "protected_task_slowdown_gate_complete": False,
        "empirical_admission": empirical,
        "derived_full_max_bytes": empirical["full_memory_max_bytes"],
        "derived_full_high_bytes": empirical["full_memory_high_bytes"],
        "derived_required_parent_headroom_bytes": empirical[
            "parent_required_headroom_bytes"
        ],
        "derivation": "ceil(clamp(3*peak+16GiB,48GiB,128GiB)); high=floor(0.75*maxGiB)",
        "monitor_sha256": hashlib.sha256(monitor_bytes).hexdigest(),
    }
    _atomic_json(output / "empirical_memory_receipt.json", receipt)
    if empirical["status"] != "PASS":
        raise MutFastError(
            "empirical memory admission blocked: " + ",".join(empirical["blockers"])
        )
    return receipt


def build_controller_manifest(spec: Mapping[str, Any], output: Path) -> dict[str, Any]:
    target = _absolute(output, label="controller manifest", exists=False)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"controller manifest must be fresh: {target}")
    fresh = Path(spec["fresh_output_root"])
    replay = dict(spec["replay"])
    standard = dict(spec["standardization"])
    wrapper = "{project_root}/scripts/autodl/run_mut_fast_accurate_stage_v2.sh"
    inventory = fresh / "historical-inventory/historical_inventory.json"
    equivalence_output = str(fresh / "equivalence/attempt-{attempt}")
    binding_output = str(fresh / "adoption-binding/attempt-{attempt}")
    threshold_output = str(fresh / "threshold-freeze/attempt-{attempt}")
    standard_output = str(fresh / "standardized/attempt-{attempt}")
    base_env = {
        "AUTODL_PYTHON": "{python}",
        "MUT_FAST_SPEC": str(spec["spec_path"]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "RUN_GNN_ABLATION": "0",
    }
    tasks = [
        {
            "id": "mut_fast_equivalence_500",
            "dataset": "mutagenicity",
            "stage": "MUT_FAST_EQUIVALENCE_500",
            "runner_dataset": "mutagenicity-fast-equivalence",
            "runner_stage": "MUT_FAST_EQUIVALENCE_500",
            "depends_on": [],
            "resource": "gpu",
            "gpu_lock_mode": "exclusive",
            "priority": 1,
            "data_splits": ["train"],
            "command": ["bash", wrapper],
            "input_manifest": str(inventory),
            "expected_output": equivalence_output,
            "required_output_files": [
                "equivalence.json", "memory_monitor.jsonl",
                "empirical_memory_receipt.json", "PASS"
            ],
            "required_log_marker": "[MUT_CHECKPOINT_INSTRUMENTATION_EQUIVALENCE_PASS]",
            "semantic_failure_markers": [
                "legacy scientific worktree changed",
                "instrumentation execution commit changed",
                "step_action_trace", "rng_state", "payload", "checkpoint",
                "test leakage"
            ],
            "environment": {
                **base_env,
                "MUT_FAST_STAGE": "equivalence",
                "MUT_STAGE_OUTPUT": "{task_output}",
                "MUT_LEGACY_PROJECT_ROOT": spec["legacy_project_root"],
                "MUT_INSTRUMENTATION_PROJECT_ROOT": spec[
                    "instrumentation_project_root"
                ],
                "MUT_EQUIVALENCE_RUN_ROOT": str(
                    fresh / "equivalence-runs/attempt-{attempt}"
                ),
                "MUT_UPSTREAM_ROOT": replay["upstream_root"],
                "MUT_DATASET_DIR": replay["dataset_dir"],
                "MUT_GNN_CHECKPOINT": replay["gnn_checkpoint"],
                "MUT_DISTANCE_CHECKPOINT": replay["distance_checkpoint"],
                "GPU_REQUIRED": "1",
                "DEVICE": "cuda:0",
            },
        },
        {
            "id": "mut_fast_historical_binding",
            "dataset": "mutagenicity",
            "stage": "MUT_FAST_HISTORICAL_BINDING",
            "runner_dataset": "mutagenicity-fast-binding",
            "runner_stage": "MUT_FAST_HISTORICAL_BINDING",
            "depends_on": ["mut_fast_equivalence_500"],
            "resource": "cpu",
            "priority": 2,
            "data_splits": [],
            "manifest_only": True,
            "command": ["bash", wrapper],
            "input_manifest": "{dep_mut_fast_equivalence_500_output}/equivalence.json",
            "expected_output": binding_output,
            "required_output_files": ["historical_adoption.json", "PASS"],
            "required_log_marker": "[MUT_HISTORICAL_50K_ADOPTION_PASS]",
            "semantic_failure_markers": [
                "historical/common binding failed", "live writable file",
                "test leakage"
            ],
            "environment": {
                **base_env,
                "MUT_FAST_STAGE": "bind-adoption",
                "MUT_STAGE_OUTPUT": "{task_output}",
                "MUT_INVENTORY_GATE": str(inventory),
                "MUT_EQUIVALENCE_GATE": (
                    "{dep_mut_fast_equivalence_500_output}/equivalence.json"
                ),
            },
        },
        {
            "id": "mut_fast_threshold_freeze",
            "dataset": "mutagenicity",
            "stage": "AM_COMRECGC_THRESHOLD_FREEZE",
            "runner_dataset": "paper-threshold-mut-fast-v2",
            "runner_stage": "AM_COMRECGC_THRESHOLD_FREEZE",
            "depends_on": ["mut_fast_historical_binding"],
            "resource": "cpu",
            "priority": 3,
            "data_splits": [],
            "manifest_only": True,
            "freezes_selector": True,
            "command": [
                "{python}",
                "{project_root}/scripts/autodl/verify_frozen_threshold_contract.py",
                "--config", "configs/hpc.yaml", "--dataset", "mutagenicity",
                "--source", standard["thresholds_path"], "--output", "{task_output}"
            ],
            "input_manifest": standard["thresholds_path"],
            "expected_output": threshold_output,
            "required_output_files": [
                "frozen_threshold_contract.json", "threshold_adoption_audit.json", "PASS"
            ],
            "required_log_marker": (
                "[FROZEN_THRESHOLD_CONTRACT_PASS] dataset=mutagenicity"
            ),
            "semantic_failure_markers": [
                "threshold", "test leakage"
            ],
            "environment": {"PYTHONDONTWRITEBYTECODE": "1"},
        },
        {
            "id": "mut_fast_standardized",
            "dataset": "mutagenicity",
            "stage": "AM_COMRECGC_HELDOUT_EVAL",
            "runner_dataset": "paper-cell-mutagenicity-comrecgc-fast-v2",
            "runner_stage": "AM_COMRECGC_HELDOUT_EVAL",
            "depends_on": [
                "mut_fast_historical_binding", "mut_fast_threshold_freeze"
            ],
            "resource": "cpu",
            "priority": 4,
            "data_splits": ["test"],
            "selector_parameters_frozen": True,
            "read_only_test": True,
            "command": ["bash", wrapper],
            "input_manifest": (
                "{dep_mut_fast_historical_binding_output}/historical_adoption.json"
            ),
            "expected_output": standard_output,
            "required_output_files": [
                "standardized/_FINALIZED.json", "standardized/run_manifest.json",
                "run_manifest.json", "final_gate.json", "_RUN_COMPLETE.json", "PASS"
            ],
            "required_log_marker": (
                "[MUT_COMRECGC_FAST_ACCURATE_STANDARDIZATION_PASS]"
            ),
            "semantic_failure_markers": [
                "historical adoption gate is invalid",
                "chemistry repair cannot be frozen", "source closure changed",
                "live writer detected", "test leakage"
            ],
            "environment": {
                **base_env,
                "MUT_FAST_STAGE": "standardize",
                "MUT_STAGE_OUTPUT": "{task_output}",
                "MUT_SOURCE_GENERATION_ROOT": spec["historical_source_root"],
                "MUT_UPSTREAM_ROOT": replay["upstream_root"],
                "MUT_DATASET_DIR": replay["dataset_dir"],
                "MUT_DISTANCE_CHECKPOINT": replay["distance_checkpoint"],
                "MUT_DATASET_CSV": standard["dataset_csv"],
                "MUT_TEACHER_PATH": standard["teacher_path"],
                "MUT_MOLCLR_ROOT": standard["molclr_root"],
                "MUT_MOLCLR_CHECKPOINT": standard["molclr_checkpoint"],
                "MUT_THRESHOLDS_PATH": (
                    "{dep_mut_fast_threshold_freeze_output}/frozen_threshold_contract.json"
                ),
                "MUT_HISTORICAL_ADOPTION": (
                    "{dep_mut_fast_historical_binding_output}/historical_adoption.json"
                ),
                "DEVICE": "cpu",
            },
        },
    ]
    if spec["allow_trace_on_historical_adoption"] is False:
        tasks = [tasks[0]]
    manifest = {
        "schema_version": 1,
        "controller_id": spec["controller_id"],
        "paper_frozen": True,
        "runtime": {
            "max_gpus": FOUR_GPU_RECOVERY_LIMIT,
            "stable_idle_seconds": 60,
            "sample_interval_seconds": 5,
            "poll_seconds": 60,
            "min_free_memory_mb": int(spec["gpu_min_free_memory_mb"]),
            "idle_util_threshold": int(spec["gpu_max_utilization_percent"]),
            "worker_launcher": "auto",
            "max_cpu_tasks": 1,
            "launch_grace_seconds": 180,
            # A watchdog stop must preserve its committed checkpoint evidence;
            # automatically starting attempt 1 from step zero is forbidden.
            "max_transient_retries": 0,
            "keep_alive_when_blocked": False,
        },
        "resource_gates": {
            "min_available_ram_gb": 1,
            "min_free_disk_gb": 20,
            "max_cpu_load_fraction": 0.95,
        },
        "tasks": tasks,
    }
    _atomic_json(target, manifest)
    return {"manifest": str(target), "sha256": sha256_file(target)}


def _controller_state(spec: Mapping[str, Any]) -> tuple[Path, dict[str, Any] | None]:
    root = Path(spec["control_root"]) / "four_gpu_recovery" / spec["controller_id"]
    state = root / "controller_state.json"
    return root, _json(state, label="controller state") if state.is_file() else None


def _publish_matrix_queue(spec: Mapping[str, Any], terminal_root: Path) -> dict[str, Any]:
    fresh = Path(spec["fresh_output_root"])
    matrix = fresh / "matrix-publication"
    locator = matrix / "mut_comrecgc_terminal_locator.json"
    queue = matrix / "publisher_queue.json"
    state_path = Path(spec["control_root"]) / "fast16_matrix_authority/state.json"
    state = _json(state_path, label="fast16 authority pointer")
    initial = _absolute(state.get("latest_authority_root"), label="latest authority")
    _atomic_json(
        locator,
        {
            "schema_version": LOCATOR_SCHEMA,
            "status": "READY",
            "dataset": "Mutagenicity",
            "method": "ComRecGC",
            "terminal_root": str(terminal_root.resolve(strict=True)),
            "published_at": _utc_now(),
        },
    )
    output_root = matrix / "authority-append"
    _atomic_json(
        queue,
        {
            "schema_version": "fast16_matrix_publisher_queue_v1",
            "initial_authority_root": str(initial),
            "authority_state_path": str(state_path),
            "authority_lock_path": str(state_path.parent / "publish.lock"),
            "poll_seconds": 60,
            "cells": [
                {
                    "dataset": "Mutagenicity",
                    "method": "ComRecGC",
                    "terminal_root_locator": str(locator),
                    "output_root": str(output_root),
                }
            ],
        },
    )
    heartbeat = matrix / "publisher_heartbeat.json"
    log = matrix / "publisher.log"
    pid = matrix / "publisher.pid"
    environment = {
        **os.environ,
        "PROJECT_ROOT": str(spec["project_root"]),
        "AUTODL_PYTHON": str(spec["python"]),
        "QUEUE_MANIFEST": str(queue),
        "HEARTBEAT_PATH": str(heartbeat),
        "LOG_PATH": str(log),
        "PID_PATH": str(pid),
    }
    subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts/autodl/launch_fast16_matrix_publisher_queue.sh")],
        cwd=spec["project_root"], env=environment, check=True,
    )
    return {"locator": str(locator), "queue": str(queue), "heartbeat": str(heartbeat)}


def run_sidecar(spec: Mapping[str, Any]) -> None:
    fresh = Path(spec["fresh_output_root"])
    if fresh.exists() or fresh.is_symlink():
        raise FileExistsError(f"fresh successor output already exists: {fresh}")
    fresh.mkdir(parents=True)
    control = Path(spec["control_root"]) / "mut_fast_accurate_v2" / spec["controller_id"]
    control.mkdir(parents=True, exist_ok=False)
    heartbeat = control / "heartbeat.json"
    inventory_root = fresh / "historical-inventory"
    _atomic_json(
        heartbeat,
        {
            "schema_version": SCHEMA,
            "controller_id": spec["controller_id"],
            "pid": os.getpid(),
            "state": "HISTORICAL_INVENTORY_RUNNING",
            "heartbeat_at": _utc_now(),
            "old_440_waiter_signaled_by_successor": False,
        },
    )
    inventory = publish_inventory(spec=spec, output_dir=inventory_root)
    _atomic_json(control / "inventory_receipt.json", inventory)
    admitted_since: float | None = None
    manifest_path = control / "four_gpu_controller_manifest.json"
    submitted = False
    matrix_submitted = False
    while True:
        if not submitted:
            observation = inspect_admission(spec)
            children = _science_children(Path(spec["proc_root"]))
            now = time.monotonic()
            eligible = observation["state"] == "ADMITTED" and not children
            if eligible:
                admitted_since = admitted_since if admitted_since is not None else now
            else:
                admitted_since = None
            stable = 0 if admitted_since is None else int(now - admitted_since)
            state = "WAITING_FOR_EMPIRICAL_ADMISSION"
            _atomic_json(
                heartbeat,
                {
                    "schema_version": SCHEMA,
                    "controller_id": spec["controller_id"],
                    "pid": os.getpid(),
                    "state": state,
                    "heartbeat_at": _utc_now(),
                    "stable_admission_seconds": stable,
                    "required_stable_admission_seconds": int(
                        spec["stable_admission_seconds"]
                    ),
                    "observation": observation,
                    "existing_mut_science_children": children,
                    "old_440_waiter_action": "main_agent_exact_handover_only",
                    "old_440_waiter_signaled_by_successor": False,
                },
            )
            if stable >= int(spec["stable_admission_seconds"]):
                build_controller_manifest(spec, manifest_path)
                subprocess.run(
                    [
                        "bash",
                        str(PROJECT_ROOT / "scripts/autodl/launch_four_gpu_recovery.sh"),
                        str(manifest_path),
                    ],
                    cwd=spec["project_root"],
                    env={
                        **os.environ,
                        "AUTODL_DATA_ROOT": str(Path(spec["runtime_root"]).parent),
                        "AUTODL_CONTROL_ROOT": str(spec["control_root"]),
                        "AUTODL_PYTHON": str(spec["python"]),
                        "PROJECT_ROOT": str(spec["project_root"]),
                        "RUN_GNN_ABLATION": "0",
                    },
                    check=True,
                )
                submitted = True
            else:
                time.sleep(int(spec["poll_seconds"]))
                continue
        controller_root, controller = _controller_state(spec)
        _atomic_json(
            heartbeat,
            {
                "schema_version": SCHEMA,
                "controller_id": spec["controller_id"],
                "pid": os.getpid(),
                "state": "CONTROLLER_SUBMITTED" if controller is None else controller.get("state"),
                "heartbeat_at": _utc_now(),
                "four_gpu_controller_root": str(controller_root),
                "four_gpu_controller_state": controller,
                "matrix_publisher_submitted": matrix_submitted,
                "old_440_waiter_signaled_by_successor": False,
            },
        )
        if controller is not None and controller.get("state") == "PASS":
            if spec["allow_trace_on_historical_adoption"] is False:
                task_state = _json(
                    controller_root / "tasks/mut_fast_equivalence_500/state.json",
                    label="equivalence task state",
                )
                instance = (task_state.get("instances") or {}).get("main") or {}
                equivalence_root = _absolute(
                    instance.get("expected_output"),
                    label="Mut equivalence terminal",
                )
                _atomic_json(
                    heartbeat,
                    {
                        "schema_version": SCHEMA,
                        "controller_id": spec["controller_id"],
                        "pid": os.getpid(),
                        "state": "WAITING_FOR_TRACE_ON_ADOPTION_AUTHORIZATION",
                        "heartbeat_at": _utc_now(),
                        "four_gpu_controller_root": str(controller_root),
                        "equivalence_root": str(equivalence_root),
                        "historical_source_trace_enabled": True,
                        "traceoff_reference_rerun": False,
                        "trace_on_historical_adoption_authorized": False,
                        "matrix_publisher_submitted": False,
                        "manual_intervention_required": True,
                        "manual_intervention_reason": (
                            "explicit approval is required to adopt a trace-enabled "
                            "historical 50k source where the frozen automatic Route-A "
                            "contract requires trace_enabled=false"
                        ),
                        "old_440_waiter_signaled_by_successor": False,
                    },
                )
                time.sleep(int(spec["poll_seconds"]))
                continue
            task_state = _json(
                controller_root / "tasks/mut_fast_standardized/state.json",
                label="standardization task state",
            )
            instance = (task_state.get("instances") or {}).get("main") or {}
            terminal = _absolute(
                instance.get("expected_output"), label="Mut standardized terminal"
            )
            if not matrix_submitted:
                matrix_receipt = _publish_matrix_queue(spec, terminal)
                _atomic_json(control / "matrix_publisher_receipt.json", matrix_receipt)
                matrix_submitted = True
            _atomic_json(
                heartbeat,
                {
                    "schema_version": SCHEMA,
                    "controller_id": spec["controller_id"],
                    "pid": os.getpid(),
                    "state": "PASS_MATRIX_PUBLISHER_SUBMITTED",
                    "heartbeat_at": _utc_now(),
                    "four_gpu_controller_root": str(controller_root),
                    "terminal_root": str(terminal),
                    "matrix_publisher_submitted": True,
                    "old_440_waiter_signaled_by_successor": False,
                },
            )
            return
        if controller is not None and controller.get("state") in {
            "FAILED", "BLOCKED", "SKIPPED"
        }:
            return
        time.sleep(int(spec["poll_seconds"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="action", required=True)
    material = commands.add_parser("materialize")
    material.add_argument("--template", type=Path, required=True)
    material.add_argument("--output", type=Path, required=True)
    material.add_argument("--project-root", type=Path, required=True)
    material.add_argument("--legacy-project-root", type=Path, required=True)
    material.add_argument("--instrumentation-project-root", type=Path, required=True)
    material.add_argument("--timestamp", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--spec", type=Path, required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--spec", type=Path, required=True)
    inventory.add_argument("--output-dir", type=Path, required=True)
    admission = commands.add_parser("admission")
    admission.add_argument("--spec", type=Path, required=True)
    monitored = commands.add_parser("run-equivalence")
    monitored.add_argument("--spec", type=Path, required=True)
    monitored.add_argument("--run-root", type=Path, required=True)
    monitored.add_argument("--output-dir", type=Path, required=True)
    wait = commands.add_parser("wait-admission")
    wait.add_argument("--spec", type=Path, required=True)
    wait.add_argument("--require-assigned-gpu", action="store_true")
    binding = commands.add_parser("bind-adoption")
    binding.add_argument("--spec", type=Path, required=True)
    binding.add_argument("--inventory-gate", type=Path, required=True)
    binding.add_argument("--equivalence-gate", type=Path, required=True)
    binding.add_argument("--authorization-receipt", type=Path, required=True)
    binding.add_argument("--trace-code-audit", type=Path, required=True)
    binding.add_argument(
        "--instrumentation-equivalence-gate", type=Path, required=True
    )
    binding.add_argument("--canary-memory-receipt", type=Path, required=True)
    binding.add_argument("--output-dir", type=Path, required=True)
    manifest = commands.add_parser("build-manifest")
    manifest.add_argument("--spec", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--spec", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "materialize":
        result = materialize(
            template=args.template, output=args.output,
            project_root=args.project_root, legacy_root=args.legacy_project_root,
            instrumentation_root=args.instrumentation_project_root,
            timestamp=args.timestamp,
        )
    else:
        spec = load_spec(args.spec)
        if args.action == "validate":
            result = {"status": "PASS", "controller_id": spec["controller_id"]}
        elif args.action == "inventory":
            result = publish_inventory(spec=spec, output_dir=args.output_dir)
        elif args.action == "admission":
            result = inspect_admission(spec)
        elif args.action == "run-equivalence":
            result = run_equivalence_monitored(
                spec, run_root=args.run_root, output_dir=args.output_dir
            )
        elif args.action == "wait-admission":
            while True:
                result = inspect_admission(spec, require_gpu=False)
                if result["headroom_pass"]:
                    break
                print(json.dumps(result, sort_keys=True), flush=True)
                time.sleep(int(spec["poll_seconds"]))
        elif args.action == "bind-adoption":
            result = publish_adoption(
                spec=spec, inventory_gate=args.inventory_gate,
                equivalence_gate=args.equivalence_gate, output_dir=args.output_dir,
                authorization_receipt=args.authorization_receipt,
                trace_code_audit=args.trace_code_audit,
                instrumentation_equivalence_gate=(
                    args.instrumentation_equivalence_gate
                ),
                canary_memory_receipt=args.canary_memory_receipt,
            )
        elif args.action == "build-manifest":
            result = build_controller_manifest(spec, args.output)
        else:
            run_sidecar(spec)
            result = {"status": "EXITED", "controller_id": spec["controller_id"]}
    print(json.dumps(result, indent=2, sort_keys=True))
    markers = {
        "validate": "[MUT_FAST_ACCURATE_V2_VALIDATE_PASS]",
        "inventory": "[MUT_HISTORICAL_50K_INVENTORY_PASS]",
        "bind-adoption": "[MUT_HISTORICAL_50K_ADOPTION_PASS]",
        "build-manifest": "[MUT_FAST_ACCURATE_V2_MANIFEST_PASS]",
        "run-equivalence": "[MUT_CHECKPOINT_INSTRUMENTATION_EQUIVALENCE_PASS]",
    }
    if args.action in markers:
        print(markers[args.action], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
