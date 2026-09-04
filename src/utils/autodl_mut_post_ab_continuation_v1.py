"""Fail-closed routing after the bounded Mut same-contract trace A/B.

This module makes no scientific output.  It validates the sealed 500-step
trace-on/off result and emits exactly one durable next-action receipt.  A
separate, immutable deployment must consume that receipt before historical
adoption or a fresh Route-B generation can start.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .autodl_mut_first_divergence_v1 import file_sha256, stable_sha256


GATE_SCHEMA = "mut_trace_on_off_500_step_equivalence_v1"
INPUT_SCHEMA = "mut_trace_equivalence_input_manifest_v1"
DECISION_SCHEMA = "mut_post_same_contract_ab_decision_v1"
OWNER_TERMINAL_SCHEMA = "mut_same_contract_ab_owner_terminal_v1"
EXECUTION_COMMIT = "66487c062c86d53ef2f762ce04d0fb965af5af08"
SOURCE_COMMIT = "7f7ed51a1176de1c23344cda0fbf0e6c5ba210b4"
UPSTREAM_COMMIT = "122f9341a360e9f06bb58a2f5823bb596021f6bf"


class MutPostABError(RuntimeError):
    """The bounded A/B terminal or result is incomplete or contradictory."""


def _self_hash(value: Mapping[str, Any], key: str) -> str:
    observed = value.get(key)
    expected = stable_sha256(
        {name: item for name, item in value.items() if name != key}
    )
    if observed != expected:
        raise MutPostABError(f"{key} changed")
    return str(observed)


def validate_same_contract_gate(
    raw: Mapping[str, Any], *, gate_path: Path | None = None
) -> dict[str, Any]:
    """Validate both successful and causal-divergence A/B terminal gates."""

    value = dict(raw)
    if value.get("schema_version") != GATE_SCHEMA:
        raise MutPostABError("same-contract A/B gate schema changed")
    _self_hash(value, "summary_sha256")
    fixed = {
        "dataset": "mutagenicity",
        "method": "COMRECGC",
        "source_algorithm_commit": SOURCE_COMMIT,
        "execution_commit": EXECUTION_COMMIT,
        "formal_M_MAX": 50_000,
        "steps_compared": 500,
        "post_reload_steps_compared": 10,
        "trace_on_trace_enabled": True,
        "trace_off_trace_enabled": False,
        "trace_only_files_excluded_from_scientific_digest": True,
        "post_walk_prefix_finalization_performed": False,
        "full_50k_trace_on_off_parity_claimed": False,
        "arms_overlapped": False,
        "max_concurrent_arms": 1,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    changed = [key for key, expected in fixed.items() if value.get(key) != expected]
    if changed:
        raise MutPostABError(f"same-contract A/B fields changed: {changed}")
    if value.get("status") not in {"PASS", "FAIL"}:
        raise MutPostABError("same-contract A/B status is not terminal")
    manifest_path = Path(str(value.get("input_manifest") or ""))
    if (
        not manifest_path.is_absolute()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
    ):
        raise MutPostABError("same-contract input manifest is absent or indirect")
    if file_sha256(manifest_path) != value.get("input_manifest_sha256"):
        raise MutPostABError("same-contract input manifest bytes changed")
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise MutPostABError("same-contract input manifest is not one object")
    _self_hash(manifest, "manifest_sha256")
    expected_manifest = {
        "schema_version": INPUT_SCHEMA,
        "dataset": "mutagenicity",
        "source_algorithm_commit": SOURCE_COMMIT,
        "execution_commit": EXECUTION_COMMIT,
        "upstream_commit": UPSTREAM_COMMIT,
        "formal_M_MAX": 50_000,
        "comparison_steps": 500,
        "post_reload_steps": 10,
        "candidate_capacity": 100_000,
        "seed": 0,
        "parent_limit": 1448,
        "batch_size": 128,
        "device": "cuda:0",
        "arms_sequential": True,
        "max_concurrent_arms": 1,
        "calibration_loaded": False,
        "test_loaded": False,
        "pythonhashseed": "0",
    }
    changed_manifest = [
        key
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    ]
    if changed_manifest:
        raise MutPostABError(
            f"same-contract input manifest changed: {changed_manifest}"
        )
    if gate_path is not None and file_sha256(gate_path) == "":  # pragma: no cover
        raise MutPostABError("same-contract gate cannot be hashed")
    return value


def classify_same_contract_gate(raw: Mapping[str, Any]) -> str:
    """Classify a terminal gate without turning instrumentation failure into science."""

    value = validate_same_contract_gate(raw)
    observer_audits = (
        value.get("trace_on_observer_log_audit"),
        value.get("trace_off_observer_log_audit"),
    )
    observer_logs_valid = all(
        isinstance(audit, Mapping) and audit.get("status") == "PASS"
        for audit in observer_audits
    )
    if value["status"] == "PASS":
        exact_fields = (
            "trace_on_off_stepwise_exact",
            "step_action_trace_exact",
            "rng_state_exact",
            "classifier_probability_trace_exact",
            "step_semantic_fields_present",
            "trace_on_checkpoint_reload_pass",
            "trace_off_checkpoint_reload_pass",
            "post_reload_trace_mode_equivalence_pass",
            "step500_checkpoint_serialized_candidate_records_exact",
            "step500_checkpoint_candidate_universe_exact",
            "checkpoint_algorithm_scientific_state_exact",
            "checkpoint_rng_state_exact",
            "checkpoint_sqlite_logical_state_exact",
            "checkpoint_graph_registry_exact",
            "resolved_config_scientific_binding_exact",
        )
        if (
            all(value.get(field) is True for field in exact_fields)
            and value.get("first_semantic_divergence_step") is None
            and value.get("failures") in ([], None)
            and observer_logs_valid
        ):
            return "PASS_TRACE_MODE_EQUIVALENCE"
        return "ENGINEERING_REPAIR_REQUIRED"
    first = value.get("first_semantic_divergence_step")
    causal_difference = (
        isinstance(first, int)
        and not isinstance(first, bool)
        and 1 <= first <= 500
        and value.get("trace_on_off_stepwise_exact") is False
        and value.get("step_semantic_fields_present") is True
        and observer_logs_valid
    )
    return (
        "SCIENTIFIC_STATE_DIVERGENCE_CONFIRMED"
        if causal_difference
        else "ENGINEERING_REPAIR_REQUIRED"
    )


def validate_ab_owner_terminal(
    raw: Mapping[str, Any], *, task_id: str, gate_path: Path
) -> dict[str, Any]:
    value = dict(raw)
    if value.get("schema_version") != OWNER_TERMINAL_SCHEMA:
        raise MutPostABError("A/B owner terminal schema changed")
    if value.get("task_id") != task_id:
        raise MutPostABError("A/B owner terminal binds a different task")
    if value.get("fresh_50k_started") is not False:
        raise MutPostABError("bounded A/B terminal claims a fresh 50k launch")
    if value.get("equivalence_gate") != str(gate_path):
        raise MutPostABError("A/B owner terminal binds a different gate")
    if value.get("equivalence_gate_sha256") != file_sha256(gate_path):
        raise MutPostABError("A/B owner terminal gate bytes changed")
    return value


def select_post_ab_action(
    *,
    terminal: Mapping[str, Any],
    gate: Mapping[str, Any],
    ab_spec_path: Path,
    gate_path: Path,
) -> dict[str, Any]:
    """Select adoption, Route B, or a fail-closed engineering repair lane."""

    checked = validate_same_contract_gate(gate, gate_path=gate_path)
    terminal_value = dict(terminal)
    if terminal_value.get("fresh_50k_started") is not False:
        raise MutPostABError("bounded A/B terminal claims a fresh 50k launch")
    owner_status = str(terminal_value.get("status") or "")
    first = checked.get("first_semantic_divergence_step")
    classified_owner_status = classify_same_contract_gate(checked)
    if classified_owner_status == "PASS_TRACE_MODE_EQUIVALENCE":
        if owner_status != "PASS_TRACE_MODE_EQUIVALENCE":
            raise MutPostABError("owner terminal disagrees with PASS A/B gate")
        required_true = (
            "trace_on_off_stepwise_exact",
            "step_action_trace_exact",
            "rng_state_exact",
            "classifier_probability_trace_exact",
            "step_semantic_fields_present",
            "trace_on_checkpoint_reload_pass",
            "trace_off_checkpoint_reload_pass",
            "post_reload_trace_mode_equivalence_pass",
            "step500_checkpoint_serialized_candidate_records_exact",
            "step500_checkpoint_candidate_universe_exact",
            "checkpoint_algorithm_scientific_state_exact",
            "checkpoint_rng_state_exact",
            "checkpoint_sqlite_logical_state_exact",
            "checkpoint_graph_registry_exact",
            "resolved_config_scientific_binding_exact",
        )
        failed = [key for key in required_true if checked.get(key) is not True]
        if failed or first is not None or checked.get("failures") not in ([], None):
            raise MutPostABError(f"PASS A/B gate lacks exact semantics: {failed}")
        branch = "HISTORICAL_ADOPTION_GATES_REQUIRED"
        classification = "TRACE_ALIAS_ONLY"
        historical_adoption_gate_eligible = True
        route_b_evidence_eligible = False
        reason = "fresh_same_contract_trace_on_off_and_reload_exact"
    elif classified_owner_status == "SCIENTIFIC_STATE_DIVERGENCE_CONFIRMED":
        if owner_status != "SCIENTIFIC_STATE_DIVERGENCE_CONFIRMED":
            raise MutPostABError("owner terminal disagrees with divergent A/B gate")
        branch = "ROUTE_B_AUTHORIZATION_REQUIRED"
        classification = "SCIENTIFIC_STATE_DIVERGENCE"
        historical_adoption_gate_eligible = False
        route_b_evidence_eligible = True
        reason = f"fresh_same_contract_first_semantic_divergence_step_{first}"
    else:
        branch = "ENGINEERING_REPAIR_REQUIRED"
        classification = "COMPARATOR_BUG"
        historical_adoption_gate_eligible = False
        route_b_evidence_eligible = False
        reason = "A/B_failed_without_a_causal_step_1_500_scientific_divergence"
    decision: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA,
        "status": "READY",
        "branch": branch,
        "classification": classification,
        "reason": reason,
        "same_contract_ab_spec": str(ab_spec_path),
        "same_contract_ab_spec_sha256": file_sha256(ab_spec_path),
        "same_contract_gate": str(gate_path),
        "same_contract_gate_sha256": file_sha256(gate_path),
        "same_contract_gate_summary_sha256": checked["summary_sha256"],
        "first_semantic_divergence_step": first,
        # This watcher selects the next lane.  It is deliberately not a launch
        # or adoption authority: both branches still have independent gates.
        "historical_adoption_gate_eligible": historical_adoption_gate_eligible,
        "historical_adoption_permitted": False,
        "route_b_evidence_eligible": route_b_evidence_eligible,
        "route_b_permitted": False,
        "requires_immutable_deployed_consumer": True,
        "fresh_50k_started": False,
        "pair_store_recomputed": False,
        "dbscan_recomputed": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    decision["decision_sha256"] = stable_sha256(decision)
    return decision


__all__ = [
    "DECISION_SCHEMA",
    "MutPostABError",
    "classify_same_contract_gate",
    "select_post_ab_action",
    "validate_ab_owner_terminal",
    "validate_same_contract_gate",
]
