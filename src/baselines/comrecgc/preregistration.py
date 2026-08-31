"""Immutable preregistration records for COMRECGC recovery experiments."""

from __future__ import annotations

import json
import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .contracts import UPSTREAM_COMMIT, sha256_file, stable_json_sha256, write_json
from .contracts import GenerationParameters
from .aids_dbscan_audit import resolve_upstream_contract


def _load(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {source}")
    return value


def _write_new(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Preregistration already exists and is immutable: {destination}")
    value = dict(payload)
    value["preregistration_payload_sha256"] = stable_json_sha256(value)
    write_json(destination, value)
    return {
        **value,
        "path": str(destination),
        "file_sha256": sha256_file(destination),
    }


def write_aids_density_preregistration(
    *, existing_audit_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    audit_path = Path(existing_audit_path).expanduser().resolve()
    audit = _load(audit_path)
    funnel = audit.get("funnel") or {}
    if audit.get("audit_passed") is not True:
        raise ValueError("AIDS existing-artifact audit did not pass.")
    if int(audit.get("candidate_count", -1)) != 31:
        raise ValueError("AIDS density retry requires the frozen 31-candidate set.")
    if int(funnel.get("distance_pairs", -1)) != 1984 or int(funnel.get("theta_eligible_pairs", -1)) != 28:
        raise ValueError("AIDS existing-artifact audit statistics differ from the frozen blocker.")
    contract = dict(audit["upstream_contract"])
    return _write_new(
        output_path,
        {
            "schema_version": 1,
            "purpose": "diagnostic_only",
            "upstream_commit": UPSTREAM_COMMIT,
            "project_commit": str(audit["project_commit"]),
            "source_artifact_path": str(audit["source_counterfactuals_path"]),
            "source_artifact_sha256": str(audit["source_counterfactuals_sha256"]),
            "source_audit_path": str(audit_path),
            "source_audit_sha256": sha256_file(audit_path),
            "candidate_set_sha256": str(audit["candidate_set_sha256"]),
            "candidate_count": 31,
            "candidate_regeneration": False,
            "parents": "all_native_reject_parents_sorted_by_stable_id",
            "seed": 0,
            "theta": float(contract["theta"]),
            "eps": float(contract["eps"]),
            "min_samples": int(contract["min_samples"]),
            "metric": str(contract["metric"]),
            "algorithm": str(contract["algorithm"]),
            "centroid_filter": {
                "radius_operator": "<",
                "centroid_norm_operator": "<",
                "radius": float(contract["eps"]),
                "theta": float(contract["theta"]),
            },
            "maximum_retries": 1,
            "used_for_final_metrics": False,
            "parameter_selection_from_result": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def validate_chemistry_trace_evidence(
    path: str | Path,
    *,
    dataset: str,
) -> dict[str, Any]:
    """Validate dataset-scoped trace evidence without overstating parity."""

    source = Path(path).expanduser().resolve()
    payload = _load(source)
    if payload.get("trace_parity_passed") is True:
        return {
            "trace_evidence_kind": "trace_on_off_parity",
            "trace_parity_required": True,
            "trace_parity_passed": True,
            "trace_integrity_passed": True,
            "candidate_count": payload.get("candidate_count"),
        }
    if dataset not in {"aids", "bace"}:
        raise ValueError("Chemistry repair cannot be frozen before trace parity passes.")
    marker = source.parent / "_TRACE_COMPLETE.json"
    marker_payload = _load(marker)
    candidate_count = int(payload.get("candidate_count", -1))
    resolved_count = int(payload.get("candidate_lineage_resolved_count", -2))
    rng_evidence: dict[str, Any] | None = None
    if int(payload.get("rng_calls_added", -1)) == 0:
        rng_evidence = {
            "trace_rng_evidence_kind": "trace_summary_explicit_zero",
            "rng_calls_added": 0,
        }
    elif "rng_calls_added" not in payload:
        rng_evidence = _validate_freeze_only_no_rng_compatibility(
            source=source,
            payload=payload,
            marker_payload=marker_payload,
            candidate_count=candidate_count,
            resolved_count=resolved_count,
            dataset=dataset,
        )
    valid = bool(
        payload.get("trace_only") is True
        and rng_evidence is not None
        and candidate_count >= 0
        and resolved_count == candidate_count
        and marker_payload.get("trace_complete") is True
    )
    if not valid:
        raise ValueError(f"{dataset} chemistry trace integrity evidence is incomplete.")
    return {
        "trace_evidence_kind": "streamed_trace_integrity_no_on_off_reference",
        "trace_parity_required": False,
        "trace_parity_passed": False,
        "trace_integrity_passed": True,
        "candidate_count": candidate_count,
        "trace_complete_marker_path": str(marker),
        "trace_complete_marker_sha256": sha256_file(marker),
        **rng_evidence,
    }


def _validate_freeze_only_no_rng_compatibility(
    *,
    source: Path,
    payload: Mapping[str, Any],
    marker_payload: Mapping[str, Any],
    candidate_count: int,
    resolved_count: int,
    dataset: str,
) -> dict[str, Any] | None:
    """Reopen the exact completed-walk proof emitted before RNG was explicit.

    The v3 freeze-only producer proved that the random walk had completed and
    that recovery performed no proposal or RNG call, but omitted the redundant
    ``rng_calls_added`` field from ``trace_summary.json``.  Do not generalize
    that omission: only the hash-closed v4 recovery receipt and its exact v3
    trace schema may supply the missing zero.
    """

    trace_root = source.parent
    generation_root = trace_root.parent
    lineage_path = trace_root / "candidate_action_lineage.json"
    selected_trace_path = trace_root / "selected_action_trace_manifest.json"
    recovery_path = generation_root / "freeze_only_recovery.json"
    terminal_path = generation_root / "_RUN_COMPLETE.json"
    run_manifest_path = generation_root / "run_manifest.json"
    required_paths = (
        lineage_path,
        selected_trace_path,
        recovery_path,
        terminal_path,
        run_manifest_path,
    )
    if dataset != "aids" or trace_root.name != "trace" or any(
        not path.is_file() or path.is_symlink() for path in required_paths
    ):
        return None

    try:
        lineage = _load(lineage_path)
        recovery = _load(recovery_path)
        terminal = _load(terminal_path)
        run_manifest = _load(run_manifest_path)
    except (OSError, TypeError, ValueError):
        return None

    closure = payload.get("frozen_payload_closure")
    recovery_checks = recovery.get("checks")
    lineage_audit = payload.get("lineage_recovery_audit")
    if not all(
        isinstance(value, Mapping)
        for value in (closure, recovery_checks, lineage_audit)
    ):
        return None

    expected_reason = (
        "Random walk is complete; freeze-only performs no proposal or RNG call."
    )
    counterfactuals_sha256 = recovery.get("counterfactuals_sha256")
    compatible = bool(
        payload.get("trace_schema_version") == 1
        and payload.get("trace_only") is True
        and payload.get("algorithm_rerun") is False
        and payload.get("lineage_recovery_policy")
        == "authoritative_backing_freeze_only_v3"
        and payload.get("candidate_lineage_format")
        == "selected_trace_predecessor_index"
        and Path(str(payload.get("candidate_lineage_path") or "")).resolve()
        == lineage_path.resolve()
        and Path(str(payload.get("selected_trace_path") or "")).resolve()
        == selected_trace_path.resolve()
        and marker_payload.get("trace_complete") is True
        and marker_payload.get("freeze_only_recovery") is True
        and marker_payload.get("candidate_lineage_sha256")
        == sha256_file(lineage_path)
        and marker_payload.get("selected_trace_manifest_sha256")
        == sha256_file(selected_trace_path)
        and lineage.get("schema_version") == 2
        and lineage.get("format") == "selected_trace_predecessor_index"
        and int(lineage.get("candidate_count", -1)) == candidate_count
        and int(lineage.get("candidate_lineage_resolved_count", -2))
        == resolved_count
        and closure.get("schema_version") == "comrecgc_frozen_payload_closure_v7"
        and closure.get("closure_complete") is True
        and closure.get("scientific_parameters_changed") is False
        and closure.get("candidate_order_changed") is False
        and closure.get("candidate_payload_changed") is False
        and closure.get("post_write_reload_verified") is True
        and closure.get("original_trace_hash_roundtrip_verified") is True
        and int(closure.get("sha_mismatch_count", -1)) == 0
        and int(closure.get("unresolved_hash_count", -1)) == 0
        and recovery.get("schema_version")
        == "comrecgc_completed_generation_freeze_audit_v4"
        and recovery.get("FREEZE_ONLY_RECOVERY_SAFE") is True
        and recovery.get("recovery_completed") is True
        and recovery.get("algorithm_rerun") is False
        and recovery.get("random_walk_complete") is True
        and recovery.get("rng_state_required_for_freeze_only") is False
        and recovery.get("rng_state_reason") == expected_reason
        and bool(recovery_checks)
        and all(value is True for value in recovery_checks.values())
        and int(recovery.get("candidate_count", -1)) == candidate_count
        and int(recovery.get("candidate_lineage_resolved_count", -2))
        == resolved_count
        and recovery.get("output_dir") == str(generation_root.resolve())
        and isinstance(counterfactuals_sha256, str)
        and len(counterfactuals_sha256) == 64
        and terminal.get("run_complete") is True
        and terminal.get("freeze_only_recovery") is True
        and terminal.get("recovery_manifest_sha256") == sha256_file(recovery_path)
        and terminal.get("counterfactuals_sha256") == counterfactuals_sha256
        and run_manifest.get("run_complete") is True
        and run_manifest.get("freeze_only_recovery") is True
        and run_manifest.get("algorithm_rerun") is False
        and int(run_manifest.get("counterfactual_candidate_count", -1))
        == candidate_count
        and run_manifest.get("counterfactuals_sha256") == counterfactuals_sha256
        and run_manifest.get("trace_summary") == payload
        and int(lineage_audit.get("legacy_inference_invocation_count", -1)) == 0
        and int(lineage_audit.get("legacy_missing_action_count", -1)) == 0
        and int(lineage_audit.get("missing_action_fallback_count", -1)) == 0
        and int(lineage_audit.get("recorded_action_replay_failed_count", -1)) == 0
        and int(lineage_audit.get("recorded_action_replay_mismatch_count", -1))
        == 0
    )
    if not compatible:
        return None
    return {
        "trace_rng_evidence_kind": "completed_walk_freeze_only_v3_v4_receipt",
        "rng_calls_added": 0,
        "freeze_only_recovery_path": str(recovery_path),
        "freeze_only_recovery_sha256": sha256_file(recovery_path),
        "freeze_only_terminal_path": str(terminal_path),
        "freeze_only_terminal_sha256": sha256_file(terminal_path),
    }


def write_mutagenicity_chem_repair_preregistration(
    *,
    project_commit: str,
    source_counterfactuals_path: str | Path,
    trace_parity_path: str | Path,
    atom_mapping_path: str | Path,
    bond_mapping_path: str | Path,
    output_path: str | Path,
    dataset: str = "mutagenicity",
) -> dict[str, Any]:
    trace_evidence = validate_chemistry_trace_evidence(
        trace_parity_path,
        dataset=dataset,
    )
    return _write_new(
        output_path,
        {
            "schema_version": 1,
            "repair_policy_version": 1,
            "method": "COMRECGC-Adapted-DeterministicChemRepair",
            "upstream_commit": UPSTREAM_COMMIT,
            "project_commit": project_commit,
            "source_counterfactuals_path": str(Path(source_counterfactuals_path).resolve()),
            "source_counterfactuals_sha256": sha256_file(source_counterfactuals_path),
            "trace_evidence_path": str(Path(trace_parity_path).resolve()),
            "trace_evidence_sha256": sha256_file(trace_parity_path),
            "trace_evidence_kind": trace_evidence["trace_evidence_kind"],
            "trace_parity_required": trace_evidence["trace_parity_required"],
            "trace_parity_passed": trace_evidence["trace_parity_passed"],
            "trace_integrity_passed": trace_evidence["trace_integrity_passed"],
            "trace_rng_evidence_kind": trace_evidence.get(
                "trace_rng_evidence_kind"
            ),
            "rng_calls_added": trace_evidence.get("rng_calls_added"),
            "freeze_only_recovery_path": trace_evidence.get(
                "freeze_only_recovery_path"
            ),
            "freeze_only_recovery_sha256": trace_evidence.get(
                "freeze_only_recovery_sha256"
            ),
            "freeze_only_terminal_path": trace_evidence.get(
                "freeze_only_terminal_path"
            ),
            "freeze_only_terminal_sha256": trace_evidence.get(
                "freeze_only_terminal_sha256"
            ),
            "source_attributes": "preserve",
            "retained_atom_attributes": "preserve",
            "retained_bond_attributes": "preserve",
            "new_untyped_edge_bond_type": "SINGLE",
            "new_atom_mapping_file": str(Path(atom_mapping_path).resolve()),
            "new_atom_mapping_sha256": sha256_file(atom_mapping_path),
            "bond_mapping_file": str(Path(bond_mapping_path).resolve()),
            "bond_mapping_sha256": sha256_file(bond_mapping_path),
            "invalid_action_policy": "rollback_and_skip_action",
            "dependent_action_policy": "skip",
            "alternative_action_search": False,
            "alternative_bond_order_search": False,
            "charge_neutralization": False,
            "automatic_edge_deletion": False,
            "automatic_atom_replacement": False,
            "rf_used_in_repair": False,
            "wnode_used_in_repair": False,
            "strict_flip_used_in_repair": False,
            "max_outputs_per_raw_candidate": 1,
            "representative_policy": "repaired_original_official_medoid",
            "cluster_backfill": False,
            "rank_backfill": False,
            "invalid_slot_backfill": False,
            "rank_compaction": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _resolve_generation_defaults(upstream_root: str | Path) -> dict[str, Any]:
    source = (Path(upstream_root).expanduser().resolve() / "comrecgc.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    wanted = {"--theta", "--teleport", "--steps", "--heads", "--k", "--sample_size"}
    values: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        try:
            option = ast.literal_eval(node.args[0])
        except Exception:
            continue
        if option not in wanted:
            continue
        default_node = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "default"),
            None,
        )
        if default_node is not None:
            values[str(option)] = ast.literal_eval(default_node)
    if set(values) != wanted:
        raise ValueError(f"Pinned upstream generation defaults are incomplete: {values}")
    return values


def write_aids_native_full_preregistration(
    *,
    project_root: str | Path,
    upstream_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve()
    upstream = Path(upstream_root).expanduser().resolve()
    generation_defaults = _resolve_generation_defaults(upstream)
    generation = GenerationParameters.for_mode("full")
    expected = {
        "--theta": generation.theta,
        "--teleport": generation.teleport,
        "--steps": generation.steps,
        "--heads": generation.heads,
        "--k": generation.candidate_capacity,
        "--sample_size": generation.sample_size,
    }
    if generation_defaults != expected:
        raise ValueError(
            "Project full generation contract differs from pinned upstream defaults: "
            f"upstream={generation_defaults}, project={expected}"
        )
    recourse = resolve_upstream_contract(upstream)
    gnn = upstream / "data/aids/gnn/model_best.pth"
    neurosed = upstream / "data/aids/neurosed/best_model.pt"
    prediction = upstream / "data/aids/gnn/preds.pt"
    for required in (gnn, neurosed, prediction):
        if not required.is_file():
            raise FileNotFoundError(required)
    project_commit = __import__("subprocess").run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    return _write_new(
        output_path,
        {
            "schema_version": 1,
            "purpose": "official_native_full",
            "dataset": "TU/AIDS",
            "parent_universe": "all_official_prediction_eq_0_sorted_by_dataset_index",
            "upstream_commit": UPSTREAM_COMMIT,
            "project_commit": project_commit,
            "generation_parameters": generation.__dict__,
            "upstream_generation_defaults": generation_defaults,
            "common_recourse_parameters": recourse.__dict__,
            "gnn_checkpoint_path": str(gnn),
            "gnn_checkpoint_sha256": sha256_file(gnn),
            "prediction_path": str(prediction),
            "prediction_sha256": sha256_file(prediction),
            "neurosed_checkpoint_path": str(neurosed),
            "neurosed_checkpoint_sha256": sha256_file(neurosed),
            "seed": generation.seed,
            "empty_common_recourse_is_valid": True,
            "empty_cost_semantics": "N/A",
            "parameter_selection_from_result": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
