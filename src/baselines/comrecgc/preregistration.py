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
    valid = bool(
        payload.get("trace_only") is True
        and int(payload.get("rng_calls_added", -1)) == 0
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
