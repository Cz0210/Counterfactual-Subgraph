"""Strict Mut historical-50k adoption from the current same-contract A/B.

This is the dataset-specific bridge between the bounded, fresh trace-on/off
comparison and the already completed Mut generation/pair-store/DBSCAN chain.
It deliberately does not consume the superseded checkpoint-instrumentation or
canary-memory receipts: the current A/B gate already binds step semantics and
both continuous/reload executions.  Route-B selection remains owned by the
post-A/B decision and is not performed here.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from src.baselines.comrecgc.contracts import sha256_file, stable_json_sha256
from src.eval.am_legacy_standardization import scan_live_writers
from src.utils.autodl_mut_post_ab_continuation_v1 import (
    classify_same_contract_gate,
    validate_ab_owner_terminal,
    validate_same_contract_gate,
)
from src.utils.autodl_mut_same_contract_ab_v1 import (
    validate_same_contract_ab_spec,
)
from src.utils.autodl_mut_trace_on_adoption_v1 import (
    EXPECTED_CANDIDATE_UNIVERSE_SHA256,
    atomic_bytes,
    atomic_json,
    validate_authorization_receipt,
    verify_mut_candidate_pair_dbscan_binding,
)
from src.utils.autodl_mut_traceoff_parity_v1 import (
    SOURCE_CANDIDATE_COUNT,
    SOURCE_DATASET_SHA256,
    SOURCE_PARENT_ORDER_SHA256,
    SOURCE_PAYLOAD_SHA256,
    SOURCE_STEPS,
    verify_traced_source,
)


ADOPTION_SCHEMA = "mut_comrecgc_historical50k_adoption_v2"
VERIFICATION_SCHEMA = "mut_same_contract_historical50k_adoption_verification_v1"
TRANSITIVE_BINDING_KIND = "transitive_generation_pair_store_vectors_dbscan_v1"
PAIR_CANDIDATE_COUNT = 50_620


class MutSameContractAdoptionError(RuntimeError):
    """Current A/B or historical science evidence is incomplete."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _physical(path: Path, *, field: str, kind: str = "file") -> Path:
    logical = path.expanduser()
    if not logical.is_absolute() or logical.is_symlink():
        raise MutSameContractAdoptionError(
            f"{field} must be an absolute non-symlink path"
        )
    try:
        value = logical.resolve(strict=True)
    except OSError as exc:
        raise MutSameContractAdoptionError(f"{field} is absent: {logical}") from exc
    valid = value.is_file() if kind == "file" else value.is_dir()
    if not valid or (kind == "file" and value.stat().st_size <= 0):
        raise MutSameContractAdoptionError(f"{field} is not a nonempty {kind}")
    return value


def _json(path: Path, *, field: str) -> dict[str, Any]:
    source = _physical(path, field=field)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MutSameContractAdoptionError(f"{field} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise MutSameContractAdoptionError(f"{field} must contain one object")
    return value


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


def _require_bound(document: Mapping[str, Any], value: str, *, field: str) -> None:
    if value not in {str(item) for item in _nested_values(document)}:
        raise MutSameContractAdoptionError(f"{field} is not bound by common manifest")


def _lineage_contract(source_root: Path) -> dict[str, Any]:
    path = _physical(
        source_root / "trace/candidate_action_lineage.json",
        field="historical lineage",
    )
    value = _json(path, field="historical lineage")
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
    if (
        int(audit.get("selected_event_target_parent_mismatch_count", -1)) != 14
        or int(audit.get("predecessor_cross_parent_convergence_count", -1)) != 1
        or int(audit.get("predecessor_conflicting_exact_event_count", -1)) != 1
    ):
        failures.append("reviewed_cross_parent_and_exact_event_counts")
    candidate_count = int(value.get("candidate_count", -1))
    resolved_count = int(value.get("candidate_lineage_resolved_count", -1))
    if candidate_count != SOURCE_CANDIDATE_COUNT:
        failures.append("candidate_count")
    if resolved_count != SOURCE_CANDIDATE_COUNT:
        failures.append("candidate_lineage_resolved_count")
    if failures:
        raise MutSameContractAdoptionError(
            f"historical lineage contract failed: {sorted(set(failures))}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "candidate_count": candidate_count,
        "candidate_lineage_resolved_count": resolved_count,
        "recorded_action_replay_mismatch_count": 0,
        "selected_event_target_parent_mismatch_count": 14,
        "predecessor_cross_parent_convergence_count": 1,
        "predecessor_conflicting_exact_event_count": 1,
        "predecessor_selected_parent_mismatch_count": 0,
        "cross_parent_interpretation": (
            "canonical_representative_convergence_not_lineage_error"
        ),
    }


def _validate_current_ab(
    *, task_spec_path: Path, owner_terminal_path: Path, gate_path: Path,
    historical_source_root: Path,
) -> dict[str, Any]:
    spec_path = _physical(task_spec_path, field="same-contract A/B task spec")
    spec = validate_same_contract_ab_spec(
        _json(spec_path, field="same-contract A/B task spec"), check_files=False
    )
    expected_gate = Path(str(spec["output_dir"])) / (
        "trace_on_off_500_step_equivalence.json"
    )
    gate = _physical(gate_path, field="same-contract A/B gate")
    if gate != expected_gate.resolve(strict=True):
        raise MutSameContractAdoptionError("A/B gate is not derived from task spec")
    expected_terminal = Path(str(spec["control_root"])) / "terminal.json"
    terminal = _physical(owner_terminal_path, field="same-contract owner terminal")
    if terminal != expected_terminal.resolve(strict=True):
        raise MutSameContractAdoptionError(
            "A/B owner terminal is not derived from task spec"
        )
    if Path(str(spec["historical_artifact_root"])).resolve(strict=True) != (
        historical_source_root
    ):
        raise MutSameContractAdoptionError("A/B used another historical artifact")
    gate_value = validate_same_contract_gate(
        _json(gate, field="same-contract A/B gate"), gate_path=gate
    )
    if classify_same_contract_gate(gate_value) != "PASS_TRACE_MODE_EQUIVALENCE":
        raise MutSameContractAdoptionError("same-contract A/B is not exact PASS")
    terminal_value = validate_ab_owner_terminal(
        _json(terminal, field="same-contract owner terminal"),
        task_id=str(spec["task_id"]),
        gate_path=gate,
    )
    if terminal_value.get("status") != "PASS_TRACE_MODE_EQUIVALENCE":
        raise MutSameContractAdoptionError("same-contract owner terminal is not PASS")
    return {
        "task_spec": str(spec_path),
        "task_spec_sha256": sha256_file(spec_path),
        "owner_terminal": str(terminal),
        "owner_terminal_sha256": sha256_file(terminal),
        "gate": str(gate),
        "gate_sha256": sha256_file(gate),
        "gate_summary_sha256": gate_value["summary_sha256"],
        "task_id": str(spec["task_id"]),
        "gate_value": gate_value,
    }


def publish_same_contract_adoption(
    *,
    task_spec_path: Path,
    owner_terminal_path: Path,
    gate_path: Path,
    authorization_receipt_path: Path,
    historical_source_root: Path,
    completed_common_root: Path,
    output_root: Path,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Verify and seal the current-protocol historical Mut adoption."""

    source_root = _physical(
        historical_source_root, field="historical source root", kind="directory"
    )
    common_root = _physical(
        completed_common_root, field="completed common root", kind="directory"
    )
    proc = _physical(proc_root, field="proc root", kind="directory")
    output = output_root.expanduser()
    if not output.is_absolute() or output.is_symlink() or output.exists():
        raise MutSameContractAdoptionError(
            "adoption output must be one fresh absolute non-symlink path"
        )

    ab = _validate_current_ab(
        task_spec_path=task_spec_path,
        owner_terminal_path=owner_terminal_path,
        gate_path=gate_path,
        historical_source_root=source_root,
    )
    authorization_path = _physical(
        authorization_receipt_path, field="trace-on adoption authorization"
    )
    authorization_raw = _json(
        authorization_path, field="trace-on adoption authorization"
    )
    controller_id = str(authorization_raw.get("controller_id") or "")
    if not controller_id:
        raise MutSameContractAdoptionError("authorization controller_id is absent")
    authorization, authorization_file_sha = validate_authorization_receipt(
        authorization_path,
        expected_controller_id=controller_id,
        expected_source_root=source_root,
    )

    source_evidence = verify_traced_source(
        source_root=source_root, proc_root=proc, hash_payload=True
    )
    lineage = _lineage_contract(source_root)
    source_manifest_path = _physical(
        source_root / "run_manifest.json", field="historical generation manifest"
    )
    source_manifest = _json(source_manifest_path, field="historical generation manifest")
    source_payload_path = _physical(
        source_root / "counterfactuals.pt", field="historical generation payload"
    )
    parameters = source_manifest.get("parameters")
    if (
        not isinstance(parameters, Mapping)
        or int(parameters.get("steps", -1)) != SOURCE_STEPS
        or int(parameters.get("candidate_capacity", -1)) != 100_000
    ):
        raise MutSameContractAdoptionError("historical generation budget changed")

    common_manifest_path = _physical(
        common_root / "run_manifest.json", field="completed common manifest"
    )
    pair_adoption_path = _physical(
        common_root / "external_memory/pair_store_adoption/run_manifest.json",
        field="pair-store adoption manifest",
    )
    dbscan_manifest_path = _physical(
        common_root / "external_memory/dbscan/run_manifest.json",
        field="exact DBSCAN manifest",
    )
    common_manifest = _json(common_manifest_path, field="completed common manifest")
    pair_adoption = _json(pair_adoption_path, field="pair-store adoption manifest")
    pair_manifest_path = _physical(
        Path(str(pair_adoption.get("source_manifest_path") or "")),
        field="source pair-store manifest",
    )
    pair_manifest = _json(pair_manifest_path, field="source pair-store manifest")
    dbscan_manifest = _json(dbscan_manifest_path, field="exact DBSCAN manifest")
    source_manifest_sha = sha256_file(source_manifest_path)
    pair_manifest_sha = sha256_file(pair_manifest_path)
    dbscan_manifest_sha = sha256_file(dbscan_manifest_path)
    scientific = pair_manifest.get("scientific_identity")
    dbscan_identity = dbscan_manifest.get("scientific_identity")
    external = common_manifest.get("external_memory_artifacts")
    failures: list[str] = []
    if (
        common_manifest.get("dataset") != "mutagenicity"
        or common_manifest.get("method") != "COMRECGC"
        or common_manifest.get("run_complete") is not True
        or common_manifest.get("counterfactuals_sha256") != SOURCE_PAYLOAD_SHA256
    ):
        failures.append("common_identity")
    generation_sha = str(common_manifest.get("generation_manifest_sha256") or "")
    if generation_sha and generation_sha != source_manifest_sha:
        failures.append("common_generation_manifest")
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
    if candidate_universe != EXPECTED_CANDIDATE_UNIVERSE_SHA256:
        failures.append("candidate_universe")
    if int(scientific.get("candidate_count", -1)) != PAIR_CANDIDATE_COUNT:
        failures.append("pair_candidate_count")
    if (
        not isinstance(external, Mapping)
        or external.get("engine") != "external_memory_exact_v1"
        or Path(str(external.get("pair_store_manifest") or "")).resolve(strict=True)
        != pair_manifest_path
        or external.get("pair_store_manifest_sha256") != pair_manifest_sha
        or Path(str(external.get("dbscan_manifest") or "")).resolve(strict=True)
        != dbscan_manifest_path
        or external.get("dbscan_manifest_sha256") != dbscan_manifest_sha
    ):
        failures.append("common_external_memory_binding")
    if (
        dbscan_manifest.get("run_complete") is not True
        or dbscan_manifest.get("approximation_used") is not False
        or not isinstance(dbscan_identity, Mapping)
        or dbscan_identity.get("vectors_path") != pair_manifest.get("vectors_path")
        or dbscan_identity.get("vectors_sha256") != pair_manifest.get("vectors_sha256")
    ):
        failures.append("dbscan_pair_store_binding")
    if failures:
        raise MutSameContractAdoptionError(
            f"historical/common contract failed: {sorted(set(failures))}"
        )
    _require_bound(common_manifest, pair_manifest_sha, field="pair-store SHA")
    _require_bound(common_manifest, dbscan_manifest_sha, field="DBSCAN SHA")

    universe_binding = verify_mut_candidate_pair_dbscan_binding(
        source_payload_path=source_payload_path,
        pair_manifest_path=pair_manifest_path,
        dbscan_manifest_path=dbscan_manifest_path,
        expected_candidate_universe_sha256=EXPECTED_CANDIDATE_UNIVERSE_SHA256,
        expected_source_payload_sha256=SOURCE_PAYLOAD_SHA256,
        expected_candidate_count=PAIR_CANDIDATE_COUNT,
        candidate_capacity=100_000,
    )
    if (
        universe_binding.get("status") != "PASS"
        or universe_binding.get("binding_kind") != TRANSITIVE_BINDING_KIND
        or universe_binding.get("source_native_candidate_universe_sha")
        != candidate_universe
        or universe_binding.get("pair_store_source_candidate_universe_sha")
        != candidate_universe
        or universe_binding.get("dbscan_native_candidate_universe_sha") is not None
        or universe_binding.get("dbscan_transitively_bound_candidate_universe_sha")
        != candidate_universe
        or universe_binding.get("dbscan_approximation_used") is not False
    ):
        raise MutSameContractAdoptionError("three-way candidate binding is not PASS")

    source_writer = source_evidence["live_writer_audit"]
    common_writer = scan_live_writers(common_root, proc_root=proc)
    pair_writer = scan_live_writers(pair_manifest_path.parent, proc_root=proc)
    if any(
        int(row.get("writable_fd_count", -1)) != 0
        for row in (source_writer, common_writer, pair_writer)
    ):
        raise MutSameContractAdoptionError("historical artifact has an active writer")
    common_count = int(common_manifest.get("common_recourse_count", -1))
    if common_count <= 0:
        raise MutSameContractAdoptionError("common recourse count is invalid")

    output.mkdir(parents=True, mode=0o755)
    binding_path = output / "candidate_universe_binding.json"
    atomic_json(binding_path, universe_binding, fresh=True)
    now = _utc_now()
    payload: dict[str, Any] = {
        "schema_version": ADOPTION_SCHEMA,
        "adoption_evidence_contract": "same_contract_ab_attachment_5_4_v1",
        "status": "PASS",
        "dataset": "mutagenicity",
        "method": "COMRECGC",
        "historical_artifact_adopted": True,
        "historical_generation_adopted": True,
        "historical_source_trace_enabled": True,
        "source_trace_enabled": True,
        "target_default_trace_mode": False,
        "trace_is_observational": True,
        "traceoff_reference_rerun": False,
        "trace_parity_passed": False,
        "trace_off_full_rerun_performed": False,
        "full_trace_on_off_parity_claimed": False,
        "full_50k_rerun_performed": False,
        "500_step_semantic_equivalence_passed": True,
        "trace_on_off_500_step_equivalence_pass": True,
        "adoption_without_full_50k_parity_rerun_authorized": True,
        "generation_complete": True,
        "generation_steps": SOURCE_STEPS,
        "M_MAX": SOURCE_STEPS,
        "M_EFFECTIVE": SOURCE_STEPS,
        "M_configured_max": SOURCE_STEPS,
        "M_effective": SOURCE_STEPS,
        "candidate_capacity": 100_000,
        "candidate_count": SOURCE_CANDIDATE_COUNT,
        "lineage_pass": True,
        "candidate_freeze_pass": True,
        "checkpoint_reload_pass": True,
        "no_test_leakage": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "no_active_writer": True,
        "generation_rerun": False,
        "common_recourse_reused": True,
        "common_recourse_rerun": False,
        "pair_store_reused": True,
        "dbscan_reused": True,
        "pair_store_recompute_performed": False,
        "dbscan_recompute_performed": False,
        "resource_cap_used": False,
        "early_stop_used": False,
        "stop_reason": "historical_completed_50k_same_contract_ab_adoption",
        "source_generation_root": str(source_root),
        "source_generation_manifest": str(source_manifest_path),
        "source_generation_manifest_sha256": source_manifest_sha,
        "source_payload_path": str(source_payload_path),
        "source_payload_sha256": SOURCE_PAYLOAD_SHA256,
        "source_candidate_count": SOURCE_CANDIDATE_COUNT,
        "source_lineage_path": lineage["path"],
        "source_lineage_sha256": lineage["sha256"],
        "lineage_contract": lineage,
        "completed_common_root": str(common_root),
        "source_common_recourse_root": str(common_root),
        "source_common_recourse_manifest_path": str(common_manifest_path),
        "source_common_recourse_manifest_sha256": sha256_file(common_manifest_path),
        "common_recourse_count": common_count,
        "pair_store_manifest": str(pair_manifest_path),
        "pair_store_manifest_sha256": pair_manifest_sha,
        "pair_store_candidate_count": PAIR_CANDIDATE_COUNT,
        "source_pair_store_manifest": str(pair_manifest_path),
        "source_pair_store_manifest_path": str(pair_manifest_path),
        "source_pair_store_manifest_sha256": pair_manifest_sha,
        "source_dbscan_manifest_path": str(dbscan_manifest_path),
        "source_dbscan_manifest_sha256": dbscan_manifest_sha,
        "candidate_universe_sha": candidate_universe,
        "source_native_candidate_universe_sha": candidate_universe,
        "pair_store_source_candidate_universe_sha": candidate_universe,
        "dbscan_native_candidate_universe_sha": None,
        "dbscan_transitively_bound_candidate_universe_sha": candidate_universe,
        "candidate_universe_binding_state": "PASS",
        "transitive_binding_kind": TRANSITIVE_BINDING_KIND,
        "dbscan_native_candidate_universe_field_present": False,
        "dbscan_universe_binding_via_pair_vectors": True,
        "pair_candidate_graph_hashes_sha256": candidate_universe,
        "candidate_pair_dbscan_binding_receipt": universe_binding,
        "candidate_pair_dbscan_binding_sha256": universe_binding["binding_sha256"],
        "candidate_pair_dbscan_binding_path": str(binding_path),
        "candidate_pair_dbscan_binding_file_sha256": sha256_file(binding_path),
        "same_contract_ab_spec_path": ab["task_spec"],
        "same_contract_ab_spec_sha256": ab["task_spec_sha256"],
        "same_contract_ab_owner_terminal_path": ab["owner_terminal"],
        "same_contract_ab_owner_terminal_sha256": ab["owner_terminal_sha256"],
        "same_contract_gate_path": ab["gate"],
        "same_contract_gate_sha256": ab["gate_sha256"],
        "same_contract_gate_summary_sha256": ab["gate_summary_sha256"],
        "500_step_semantic_equivalence_receipt_path": ab["gate"],
        "500_step_semantic_equivalence_receipt_sha256": ab["gate_sha256"],
        "trace_mode_equivalence_path": ab["gate"],
        "trace_mode_equivalence_sha256": ab["gate_sha256"],
        "trace_on_adoption_authorization_path": str(authorization_path),
        "trace_on_adoption_authorization_file_sha256": authorization_file_sha,
        "trace_on_adoption_authorization_sha256": authorization[
            "authorization_sha256"
        ],
        "legacy_instrumentation_equivalence_used": False,
        "legacy_canary_memory_receipt_used": False,
        "source_live_writer_audit": source_writer,
        "common_live_writer_audit": common_writer,
        "pair_store_live_writer_audit": pair_writer,
        "published_at": now,
    }
    payload["binding_sha256"] = stable_json_sha256(payload)
    adoption_path = output / "historical_adoption.json"
    atomic_json(adoption_path, payload, fresh=True)
    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS",
        "historical_adoption": str(adoption_path),
        "historical_adoption_sha256": sha256_file(adoption_path),
        "same_contract_gate": ab["gate"],
        "same_contract_gate_sha256": ab["gate_sha256"],
        "candidate_universe_sha": candidate_universe,
        "lineage_pass": True,
        "checkpoint_reload_pass": True,
        "no_test_leakage": True,
        "pair_store_reused": True,
        "dbscan_reused": True,
        "fresh_50k_started": False,
        "verified_at": now,
    }
    verification["verification_sha256"] = stable_json_sha256(verification)
    atomic_json(output / "verification.json", verification, fresh=True)
    atomic_bytes(output / "PASS", b"PASS\n", fresh=True)
    return payload


__all__ = [
    "ADOPTION_SCHEMA",
    "MutSameContractAdoptionError",
    "VERIFICATION_SCHEMA",
    "publish_same_contract_adoption",
]
