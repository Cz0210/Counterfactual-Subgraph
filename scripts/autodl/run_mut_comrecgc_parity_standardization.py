#!/usr/bin/env python3
"""Resume Mutagenicity ComRecGC at chemistry from one explicit evidence route.

The immutable repair-v2 common-recourse output is consumed read-only.  This
entry point creates a fresh root and runs only chemistry, unified evaluation,
the recovery gate, and freeze.  The legacy route requires true trace-on/off
parity.  The fast-accurate route instead consumes a strict historical trace-on
50k adoption receipt plus the separately authorized 500-step semantic-
equivalence proof; it never describes that evidence as trace-off parity.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.autodl.run_comrecgc_standardized_continuation import (  # noqa: E402
    ContinuationInputs,
    _git_head,
    _load_object,
    _require_directory,
    _require_file,
    _run_stage,
    _utc_now,
    _verify_adopted_generation_integrity,
    validate_adopted_generation,
)
from scripts.verify_comrecgc_checkout import verify_checkout  # noqa: E402
from src.baselines.comrecgc.contracts import (  # noqa: E402
    CF_MODE,
    DISTANCE_LINE,
    METHOD,
    UPSTREAM_COMMIT,
    atomic_write_bytes,
    sha256_file,
    stable_json_sha256,
    write_json,
)
from src.utils.autodl_mut_post_ab_continuation_v1 import (  # noqa: E402
    GATE_SCHEMA as SAME_CONTRACT_GATE_SCHEMA,
    classify_same_contract_gate,
    validate_ab_owner_terminal,
    validate_same_contract_gate,
)
from src.utils.autodl_mut_same_contract_ab_v1 import (  # noqa: E402
    validate_same_contract_ab_spec,
)
from src.utils.autodl_mut_trace_on_adoption_v1 import (  # noqa: E402
    validate_authorization_receipt,
)
from src.utils.autodl_mut_traceoff_parity_v1 import (  # noqa: E402
    INSTRUMENTATION_PROJECT_COMMIT,
    INSTRUMENTATION_SOURCE_INVENTORY_SHA256,
    LEGACY_SOURCE_INVENTORY_SHA256,
    SOURCE_CANDIDATE_COUNT,
    SOURCE_PAYLOAD_SHA256,
    SOURCE_PROJECT_COMMIT,
    SOURCE_STEPS,
    validate_instrumentation_equivalence_gate,
)


HISTORICAL_ADOPTION_SCHEMA = "mut_comrecgc_historical50k_adoption_v2"
FAST_ACCURATE_RUN_SCHEMA = "mut_comrecgc_fast_accurate_standardization_v2"
HISTORICAL_TRACE_EVIDENCE_KIND = (
    "historical_trace_on_50k_with_500_step_semantic_equivalence"
)
TRANSITIVE_BINDING_KIND = "transitive_generation_pair_store_vectors_dbscan_v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def _object(path: Path, *, label: str) -> dict[str, Any]:
    value = _load_object(_require_file(path))
    if not isinstance(value, dict):  # defensive; _load_object already enforces
        raise ValueError(f"{label} must be a JSON object")
    return value


def _validate_parity(path: Path, *, source_root: Path) -> dict[str, Any]:
    value = _object(path, label="trace parity")
    failures: list[str] = []
    for key, expected in {
        "schema_version": "mut_trace_on_off_parity_v1",
        "status": "PASS",
        "trace_parity_passed": True,
        "candidate_count": SOURCE_CANDIDATE_COUNT,
        "reference_trace_enabled": False,
        "traced_source_trace_enabled": True,
        "self_comparison": False,
        "trace_fields_stripped": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "traced_payload_sha256": SOURCE_PAYLOAD_SHA256,
    }.items():
        if value.get(key) != expected:
            failures.append(key)
    if Path(str(value.get("traced_source_root") or "")).resolve(strict=True) != source_root:
        failures.append("traced_source_root")
    reference_root = Path(str(value.get("reference_root") or "")).resolve(strict=True)
    if reference_root == source_root:
        failures.append("self_comparison_root")
    evidence = value.get("reference_evidence")
    if not isinstance(evidence, Mapping):
        failures.append("reference_evidence")
    else:
        checkpoint = evidence.get("checkpoint_evidence")
        if (
            evidence.get("status") != "PASS"
            or evidence.get("reference_root") != str(reference_root)
            or evidence.get("source_algorithm_commit") != SOURCE_PROJECT_COMMIT
            or evidence.get("reference_trace_enabled") is not False
            or evidence.get("reference_generation_rerun") is not True
            or evidence.get("calibration_loaded") is not False
            or evidence.get("test_loaded") is not False
            or not isinstance(checkpoint, Mapping)
            or int(checkpoint.get("completed_step", -1)) != SOURCE_STEPS
        ):
            failures.append("reference_evidence_contract")
        payload = Path(str(evidence.get("reference_payload") or "")).resolve(
            strict=True
        )
        if (
            payload.parent != reference_root
            or sha256_file(payload) != evidence.get("reference_payload_sha256")
            or evidence.get("reference_payload_sha256")
            != value.get("reference_payload_sha256")
        ):
            failures.append("reference_payload")
    if failures:
        raise ValueError(f"Trace parity gate is invalid: {failures}")
    return {**value, "path": str(path), "sha256": sha256_file(path)}


def _validate_common_adoption(path: Path, *, parity: Mapping[str, Any]) -> dict[str, Any]:
    value = _object(path, label="common-recourse adoption")
    evidence = value.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("Common-recourse adoption has no evidence object")
    failures: list[str] = []
    if value.get("schema_version") != "mut_common_recourse_adoption_gate_v1":
        failures.append("schema_version")
    if value.get("status") != "PASS" or value.get("trace_parity_passed") is not True:
        failures.append("status")
    if value.get("trace_parity_sha256") != parity["sha256"]:
        failures.append("trace_parity_sha256")
    if evidence.get("status") != "PASS" or evidence.get("common_recourse_adopted") is not True:
        failures.append("evidence")
    common_root = Path(str(evidence.get("source_common_recourse_root") or "")).resolve(
        strict=True
    )
    if not common_root.is_dir():
        failures.append("common_root")
    source_files = evidence.get("source_files")
    if not isinstance(source_files, Mapping):
        failures.append("source_files")
    else:
        for name, record in source_files.items():
            if not isinstance(record, Mapping):
                failures.append(f"source_files.{name}")
                continue
            source = Path(str(record.get("path") or "")).resolve(strict=True)
            if source.parent != common_root or sha256_file(source) != record.get("sha256"):
                failures.append(f"source_files.{name}")
    if failures:
        raise ValueError(f"Common-recourse adoption gate is invalid: {failures}")
    return {
        **value,
        "path": str(path),
        "sha256": sha256_file(path),
        "common_root": str(common_root),
    }


def _physical_bound_file(
    raw: Any,
    *,
    label: str,
    exact_path: Path | None = None,
) -> Path:
    logical = Path(str(raw or "")).expanduser()
    if not logical.is_absolute() or logical.is_symlink():
        raise ValueError(f"{label} must be an absolute physical file")
    path = logical.resolve(strict=True)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is not a nonempty file")
    if exact_path is not None and path != exact_path.resolve(strict=True):
        raise ValueError(f"{label} is not the frozen expected path")
    return path


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "")
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return digest


def _validate_historical_adoption(
    path: Path,
    *,
    source_root: Path,
) -> dict[str, Any]:
    """Reopen the authorized historical trace-on 50k adoption receipt.

    The generation-native, pair-store, and DBSCAN-transitive aliases must all
    equal the real strict-flip hash in
    ``pair_store.scientific_identity.candidate_graph_hashes_sha256``.  The
    legacy DBSCAN manifest has no native universe field, so its relationship is
    verified only through the exact pair-vector path/SHA chain and never
    represented as a fabricated native three-way identity.
    """

    receipt_path = _physical_bound_file(path, label="historical adoption receipt")
    value = _object(receipt_path, label="historical adoption")
    source = source_root.expanduser().resolve(strict=True)
    failures: list[str] = []
    expected = {
        "schema_version": HISTORICAL_ADOPTION_SCHEMA,
        "status": "PASS",
        "dataset": "mutagenicity",
        "method": METHOD,
        "historical_artifact_adopted": True,
        "historical_source_trace_enabled": True,
        "traceoff_reference_rerun": False,
        "trace_parity_passed": False,
        "500_step_semantic_equivalence_passed": True,
        "adoption_without_full_50k_parity_rerun_authorized": True,
        "generation_complete": True,
        "generation_steps": SOURCE_STEPS,
        "M_MAX": SOURCE_STEPS,
        "M_EFFECTIVE": SOURCE_STEPS,
        "candidate_capacity": 100_000,
        "candidate_count": SOURCE_CANDIDATE_COUNT,
        "lineage_pass": True,
        "candidate_freeze_pass": True,
        "checkpoint_reload_pass": True,
        "no_test_leakage": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "no_active_writer": True,
        "pair_store_reused": True,
        "dbscan_reused": True,
        "pair_store_recompute_performed": False,
        "dbscan_recompute_performed": False,
        "candidate_universe_binding_state": "PASS",
        "transitive_binding_kind": TRANSITIVE_BINDING_KIND,
        "dbscan_native_candidate_universe_sha": None,
        "dbscan_native_candidate_universe_field_present": False,
        "dbscan_universe_binding_via_pair_vectors": True,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            failures.append(key)
    if value.get("binding_sha256") != stable_json_sha256(
        {key: item for key, item in value.items() if key != "binding_sha256"}
    ):
        failures.append("binding_sha256")
    try:
        recorded_source = Path(str(value.get("source_generation_root") or "")).resolve(
            strict=True
        )
    except (OSError, RuntimeError):
        recorded_source = Path("/")
    if recorded_source != source:
        failures.append("source_generation_root")
    if value.get("source_payload_sha256") != SOURCE_PAYLOAD_SHA256:
        failures.append("source_payload_sha256")

    universe_fields = (
        "candidate_universe_sha",
        "source_native_candidate_universe_sha",
        "pair_store_source_candidate_universe_sha",
        "dbscan_transitively_bound_candidate_universe_sha",
    )
    universe_values: list[str] = []
    for field in universe_fields:
        try:
            universe_values.append(_require_sha256(value.get(field), label=field))
        except ValueError:
            failures.append(field)
    if len(universe_values) == len(universe_fields) and len(set(universe_values)) != 1:
        failures.append("candidate_universe_transitive_equality")
    if value.get("dbscan_native_candidate_universe_sha") is not None:
        failures.append("dbscan_native_candidate_universe_sha")

    try:
        candidate_binding_path = _physical_bound_file(
            value.get("candidate_pair_dbscan_binding_path"),
            label="candidate/pair/DBSCAN binding receipt",
        )
        if sha256_file(candidate_binding_path) != value.get(
            "candidate_pair_dbscan_binding_file_sha256"
        ):
            failures.append("candidate_pair_dbscan_binding_file_sha256")
        candidate_binding = _object(
            candidate_binding_path,
            label="candidate/pair/DBSCAN binding receipt",
        )
        candidate_binding_unhashed = {
            key: item
            for key, item in candidate_binding.items()
            if key != "binding_sha256"
        }
        if (
            candidate_binding.get("schema_version")
            != "mut_candidate_pair_dbscan_binding_v1"
            or candidate_binding.get("status") != "PASS"
            or candidate_binding.get("binding_kind") != TRANSITIVE_BINDING_KIND
            or candidate_binding.get("binding_sha256")
            != stable_json_sha256(candidate_binding_unhashed)
            or candidate_binding.get("source_native_candidate_universe_sha")
            != value.get("candidate_universe_sha")
            or candidate_binding.get("pair_store_source_candidate_universe_sha")
            != value.get("candidate_universe_sha")
            or candidate_binding.get("dbscan_native_candidate_universe_sha")
            is not None
            or candidate_binding.get(
                "dbscan_transitively_bound_candidate_universe_sha"
            )
            != value.get("candidate_universe_sha")
            or candidate_binding.get("dbscan_approximation_used") is not False
        ):
            failures.append("candidate_pair_dbscan_binding_receipt")
    except (OSError, RuntimeError, ValueError) as exc:
        failures.append(f"candidate_pair_dbscan_binding:{type(exc).__name__}")
        candidate_binding_path = Path("/")

    try:
        payload_path = _physical_bound_file(
            value.get("source_payload_path"),
            label="historical source payload",
            exact_path=source / "counterfactuals.pt",
        )
        lineage_path = _physical_bound_file(
            value.get("source_lineage_path"),
            label="historical source lineage",
            exact_path=source / "trace/candidate_action_lineage.json",
        )
        if sha256_file(lineage_path) != value.get("source_lineage_sha256"):
            failures.append("source_lineage_sha256")
        equivalence_path = _physical_bound_file(
            value.get("500_step_semantic_equivalence_receipt_path"),
            label="500-step semantic equivalence receipt",
        )
        if sha256_file(equivalence_path) != value.get(
            "500_step_semantic_equivalence_receipt_sha256"
        ):
            failures.append("500_step_semantic_equivalence_receipt_sha256")
        equivalence_raw = _object(
            equivalence_path, label="500-step semantic equivalence receipt"
        )
        if equivalence_raw.get("schema_version") == SAME_CONTRACT_GATE_SCHEMA:
            equivalence = validate_same_contract_gate(
                equivalence_raw, gate_path=equivalence_path
            )
            if classify_same_contract_gate(equivalence) != (
                "PASS_TRACE_MODE_EQUIVALENCE"
            ):
                failures.append("same_contract_equivalence_classification")
            spec_path = _physical_bound_file(
                value.get("same_contract_ab_spec_path"),
                label="same-contract A/B task spec",
            )
            if sha256_file(spec_path) != value.get("same_contract_ab_spec_sha256"):
                failures.append("same_contract_ab_spec_sha256")
            ab_spec = validate_same_contract_ab_spec(
                _object(spec_path, label="same-contract A/B task spec"),
                check_files=False,
            )
            expected_gate = Path(str(ab_spec["output_dir"])) / (
                "trace_on_off_500_step_equivalence.json"
            )
            if expected_gate.resolve(strict=True) != equivalence_path:
                failures.append("same_contract_gate_not_spec_derived")
            if Path(str(ab_spec["historical_artifact_root"])).resolve(
                strict=True
            ) != source:
                failures.append("same_contract_historical_source")
            owner_terminal_path = _physical_bound_file(
                value.get("same_contract_ab_owner_terminal_path"),
                label="same-contract A/B owner terminal",
            )
            expected_owner_terminal = Path(str(ab_spec["control_root"])) / (
                "terminal.json"
            )
            if expected_owner_terminal.resolve(strict=True) != owner_terminal_path:
                failures.append("same_contract_owner_terminal_not_spec_derived")
            if sha256_file(owner_terminal_path) != value.get(
                "same_contract_ab_owner_terminal_sha256"
            ):
                failures.append("same_contract_ab_owner_terminal_sha256")
            owner_terminal = validate_ab_owner_terminal(
                _object(owner_terminal_path, label="same-contract A/B owner terminal"),
                task_id=str(ab_spec["task_id"]),
                gate_path=equivalence_path,
            )
            if owner_terminal.get("status") != "PASS_TRACE_MODE_EQUIVALENCE":
                failures.append("same_contract_owner_terminal_status")
            if (
                value.get("same_contract_gate_path") != str(equivalence_path)
                or value.get("same_contract_gate_sha256")
                != sha256_file(equivalence_path)
                or value.get("same_contract_gate_summary_sha256")
                != equivalence.get("summary_sha256")
            ):
                failures.append("same_contract_gate_receipt_binding")
            authorization_path = _physical_bound_file(
                value.get("trace_on_adoption_authorization_path"),
                label="trace-on adoption authorization",
            )
            if sha256_file(authorization_path) != value.get(
                "trace_on_adoption_authorization_file_sha256"
            ):
                failures.append("trace_on_adoption_authorization_file_sha256")
            authorization_raw = _object(
                authorization_path, label="trace-on adoption authorization"
            )
            authorization, _authorization_file_sha = validate_authorization_receipt(
                authorization_path,
                expected_controller_id=str(authorization_raw.get("controller_id") or ""),
                expected_source_root=source,
            )
            if authorization.get("authorization_sha256") != value.get(
                "trace_on_adoption_authorization_sha256"
            ):
                failures.append("trace_on_adoption_authorization_sha256")
            equivalence = {
                **equivalence,
                "sha256": sha256_file(equivalence_path),
            }
        else:
            equivalence = validate_instrumentation_equivalence_gate(
                gate_path=equivalence_path,
                expected_legacy_inventory_sha256=LEGACY_SOURCE_INVENTORY_SHA256,
                expected_instrumentation_inventory_sha256=(
                    INSTRUMENTATION_SOURCE_INVENTORY_SHA256
                ),
            )
        if equivalence.get("sha256") != value.get(
            "500_step_semantic_equivalence_receipt_sha256"
        ):
            failures.append("500_step_semantic_equivalence_reopen_sha256")
    except (OSError, RuntimeError, ValueError) as exc:
        failures.append(f"historical_generation_evidence:{type(exc).__name__}")
        payload_path = source / "counterfactuals.pt"
        lineage_path = source / "trace/candidate_action_lineage.json"
        equivalence_path = Path("/")

    try:
        common_root = _require_directory(
            Path(str(value.get("source_common_recourse_root") or ""))
        )
        common_manifest_path = _physical_bound_file(
            value.get("source_common_recourse_manifest_path"),
            label="historical common-recourse manifest",
            exact_path=common_root / "run_manifest.json",
        )
        pair_manifest_path = _physical_bound_file(
            value.get("source_pair_store_manifest_path"),
            label="historical pair-store manifest",
        )
        dbscan_manifest_path = _physical_bound_file(
            value.get("source_dbscan_manifest_path"),
            label="historical DBSCAN manifest",
        )
        for file_path, field in (
            (common_manifest_path, "source_common_recourse_manifest_sha256"),
            (pair_manifest_path, "source_pair_store_manifest_sha256"),
            (dbscan_manifest_path, "source_dbscan_manifest_sha256"),
        ):
            if sha256_file(file_path) != value.get(field):
                failures.append(field)

        common_manifest = _object(
            common_manifest_path, label="historical common-recourse manifest"
        )
        pair_manifest = _object(pair_manifest_path, label="historical pair-store manifest")
        dbscan_manifest = _object(dbscan_manifest_path, label="historical DBSCAN manifest")
        external = common_manifest.get("external_memory_artifacts")
        pair_identity = pair_manifest.get("scientific_identity")
        dbscan_identity = dbscan_manifest.get("scientific_identity")
        pair_candidate_universe = (
            pair_identity.get("candidate_graph_hashes_sha256")
            if isinstance(pair_identity, Mapping)
            else None
        )
        common_payload_claims = {
            str(common_manifest.get(field))
            for field in (
                "counterfactuals_sha256",
                "source_counterfactuals_sha256",
            )
            if common_manifest.get(field) not in (None, "")
        }
        if (
            common_manifest.get("dataset") != "mutagenicity"
            or common_manifest.get("method") != METHOD
            or common_manifest.get("run_complete") is not True
            or SOURCE_PAYLOAD_SHA256 not in common_payload_claims
            or not isinstance(external, Mapping)
            or external.get("engine") != "external_memory_exact_v1"
            or Path(str(external.get("pair_store_manifest") or "")).resolve(strict=True)
            != pair_manifest_path
            or external.get("pair_store_manifest_sha256") != sha256_file(pair_manifest_path)
            or Path(str(external.get("dbscan_manifest") or "")).resolve(strict=True)
            != dbscan_manifest_path
            or external.get("dbscan_manifest_sha256") != sha256_file(dbscan_manifest_path)
        ):
            failures.append("common_recourse_source_binding")
        if (
            pair_manifest.get("run_complete") is not True
            or not isinstance(pair_identity, Mapping)
            or pair_identity.get("dataset") != "mutagenicity"
            or pair_identity.get("counterfactuals_sha256") != SOURCE_PAYLOAD_SHA256
            or _SHA256_RE.fullmatch(
                str(pair_identity.get("candidate_graph_hashes_sha256") or "")
            )
            is None
            or _SHA256_RE.fullmatch(
                str(pair_identity.get("generation_indices_sha256") or "")
            )
            is None
        ):
            failures.append("pair_store_generation_binding")
        if (
            _SHA256_RE.fullmatch(str(pair_candidate_universe or "")) is None
            or value.get("candidate_universe_sha") != pair_candidate_universe
            or value.get("source_native_candidate_universe_sha")
            != pair_candidate_universe
            or value.get("pair_store_source_candidate_universe_sha")
            != pair_candidate_universe
            or value.get("dbscan_transitively_bound_candidate_universe_sha")
            != pair_candidate_universe
            or value.get("dbscan_native_candidate_universe_sha") is not None
            or value.get("pair_candidate_graph_hashes_sha256")
            != pair_candidate_universe
        ):
            failures.append("pair_store_candidate_universe_binding")
        dbscan_native_universe_fields = (
            "source_candidate_universe_sha256",
            "candidate_universe_sha256",
        )
        if any(field in dbscan_manifest for field in dbscan_native_universe_fields) or (
            isinstance(dbscan_identity, Mapping)
            and any(field in dbscan_identity for field in dbscan_native_universe_fields)
        ):
            failures.append("dbscan_native_candidate_universe_field_present")
        if (
            dbscan_manifest.get("run_complete") is not True
            or dbscan_manifest.get("approximation_used") is not False
            or not isinstance(dbscan_identity, Mapping)
            or dbscan_identity.get("vectors_path") != pair_manifest.get("vectors_path")
            or dbscan_identity.get("vectors_sha256")
            != pair_manifest.get("vectors_sha256")
        ):
            failures.append("dbscan_pair_store_transitive_binding")
        common_count = common_manifest.get("common_recourse_count")
        if (
            type(value.get("common_recourse_count")) is not int
            or int(value["common_recourse_count"]) <= 0
            or value.get("common_recourse_count") != common_count
        ):
            failures.append("common_recourse_count")
    except (OSError, RuntimeError, ValueError) as exc:
        failures.append(f"historical_common_evidence:{type(exc).__name__}")
        common_root = Path("/")
        common_manifest_path = Path("/")
        pair_manifest_path = Path("/")
        dbscan_manifest_path = Path("/")

    if failures:
        raise ValueError(f"Historical Mut adoption is invalid: {sorted(set(failures))}")
    return {
        **value,
        "path": str(receipt_path),
        "sha256": sha256_file(receipt_path),
        "source_payload_path": str(payload_path),
        "source_lineage_path": str(lineage_path),
        "500_step_semantic_equivalence_receipt_path": str(equivalence_path),
        "common_root": str(common_root),
        "source_common_recourse_manifest_path": str(common_manifest_path),
        "source_pair_store_manifest_path": str(pair_manifest_path),
        "source_dbscan_manifest_path": str(dbscan_manifest_path),
        "candidate_pair_dbscan_binding_path": str(candidate_binding_path),
    }


def _commands(
    inputs: ContinuationInputs,
    *,
    common_root: Path,
    trace_evidence_path: Path,
    lineage_path: Path,
    expected_common_recourse_count: int | None,
    project_commit: str,
    teacher_sha256: str,
) -> list[tuple[str, list[str], Path, str]]:
    chemistry = inputs.output_root / "chemistry"
    evaluation = inputs.output_root / "unified_eval"
    gate = inputs.output_root / "full_gate"
    standardized = inputs.output_root / "standardized"
    chemistry_argv = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/baselines/comrecgc/audit_mutagenicity_chemistry.py"),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--project-root",
        str(PROJECT_ROOT),
        "--dataset",
        "mutagenicity",
        "--dataset-dir",
        str(inputs.dataset_dir),
        "--generation-dir",
        str(inputs.source_generation_root),
        "--trace-lineage-path",
        str(lineage_path),
        "--trace-evidence-path",
        str(trace_evidence_path),
        "--common-recourse-dir",
        str(common_root),
        "--output-dir",
        str(chemistry),
        "--preregistration-path",
        str(inputs.output_root / "preregistration/deterministic_chem_repair.json"),
        "--parent-limit",
        "1448",
        "--expected-candidate-count",
        str(SOURCE_CANDIDATE_COUNT),
        "--expected-counterfactuals-sha256",
        SOURCE_PAYLOAD_SHA256,
    ]
    if expected_common_recourse_count is not None:
        chemistry_argv.extend(
            ["--expected-medoid-count", str(expected_common_recourse_count)]
        )
    evaluation_argv = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/baselines/comrecgc/run_slot_unified_eval.py"),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--dataset",
        "mutagenicity",
        "--mode",
        "full",
        "--chemistry-dir",
        str(chemistry),
        "--dataset-csv",
        str(inputs.dataset_csv),
        "--teacher-path",
        str(inputs.teacher_path),
        "--molclr-root",
        str(inputs.molclr_root),
        "--molclr-checkpoint",
        str(inputs.molclr_checkpoint),
        "--thresholds-json",
        str(inputs.thresholds_path),
        "--output-dir",
        str(evaluation),
        "--expected-parent-count",
        "217",
        "--max-k",
        "20",
        "--device",
        "cpu",
    ]
    gate_argv = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/baselines/comrecgc/gate_recovery.py"),
        "--stage",
        "project-full",
        "--dataset",
        "mutagenicity",
        "--expected-parent-count",
        "217",
        "--expected-teacher-sha256",
        teacher_sha256,
        "--expected-project-commit",
        project_commit,
        "--input-dir",
        str(evaluation),
        "--output-dir",
        str(gate),
    ]
    freeze_argv = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/baselines/comrecgc/freeze_recovery_result.py"),
        "--dataset",
        "mutagenicity",
        "--source-dir",
        str(evaluation),
        "--gate-dir",
        str(gate),
        "--output-dir",
        str(standardized),
    ]
    return [
        ("chemistry", chemistry_argv, chemistry / "_RUN_COMPLETE.json", "run_complete"),
        ("unified_eval", evaluation_argv, evaluation / "_RUN_COMPLETE.json", "run_complete"),
        ("full_gate", gate_argv, gate / "gate_result.json", "audit_passed"),
        ("freeze", freeze_argv, standardized / "_FINALIZED.json", "finalized"),
    ]


def run(
    inputs: ContinuationInputs,
    *,
    common_adoption_path: Path | None,
    parity_path: Path | None = None,
    historical_adoption_path: Path | None = None,
) -> dict[str, Any]:
    if inputs.dataset != "mutagenicity" or inputs.device != "cpu":
        raise ValueError("This continuation is Mutagenicity CPU-only")
    if inputs.output_root.exists() or inputs.output_root.is_symlink():
        raise FileExistsError(f"Fresh OUTPUT_ROOT already exists: {inputs.output_root}")
    inputs.output_root.parent.mkdir(parents=True, exist_ok=True)
    inputs.output_root.mkdir(mode=0o755)
    try:
        if (parity_path is None) == (historical_adoption_path is None):
            raise ValueError(
                "Exactly one trace-parity or historical-adoption evidence path is required"
            )
        historical: dict[str, Any] | None = None
        parity: dict[str, Any] | None = None
        common: dict[str, Any] | None = None
        if historical_adoption_path is not None:
            if common_adoption_path is not None:
                raise ValueError(
                    "Historical adoption already binds common recourse; --common-adoption is forbidden"
                )
            historical = _validate_historical_adoption(
                historical_adoption_path,
                source_root=inputs.source_generation_root,
            )
            common_root = _require_directory(Path(historical["common_root"]))
            trace_evidence_path = historical_adoption_path
            lineage_path = _require_file(Path(historical["source_lineage_path"]))
            expected_common_recourse_count = int(historical["common_recourse_count"])
        else:
            assert parity_path is not None
            if common_adoption_path is None:
                raise ValueError("Trace parity route requires --common-adoption")
            parity = _validate_parity(
                parity_path, source_root=inputs.source_generation_root
            )
            common = _validate_common_adoption(common_adoption_path, parity=parity)
            common_root = _require_directory(Path(common["common_root"]))
            trace_evidence_path = parity_path
            lineage_path = _require_file(
                inputs.source_generation_root / "trace/candidate_action_lineage.json"
            )
            expected_common_recourse_count = None
        adoption = validate_adopted_generation(inputs)
        if int(adoption["counterfactual_candidate_count"]) != SOURCE_CANDIDATE_COUNT:
            raise ValueError("Frozen generation candidate count changed")
        checkout = verify_checkout(
            inputs.upstream_root,
            expected_commit=UPSTREAM_COMMIT,
            validate_imports=True,
        )
        write_json(inputs.output_root / "generation_adoption_manifest.json", adoption)
        if historical is not None:
            write_json(inputs.output_root / "historical_adoption_manifest.json", historical)
        else:
            assert common is not None and parity is not None
            write_json(inputs.output_root / "common_recourse_adoption_manifest.json", common)
            write_json(inputs.output_root / "trace_parity_adoption_manifest.json", parity)
        write_json(inputs.output_root / "upstream_checkout_audit.json", checkout)
        project_commit = _git_head()
        teacher_sha256 = sha256_file(inputs.teacher_path)
        environment = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(PROJECT_ROOT),
            "TOKENIZERS_PARALLELISM": "false",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
        for stage, argv, marker, field in _commands(
            inputs,
            common_root=common_root,
            trace_evidence_path=trace_evidence_path,
            lineage_path=lineage_path,
            expected_common_recourse_count=expected_common_recourse_count,
            project_commit=project_commit,
            teacher_sha256=teacher_sha256,
        ):
            _run_stage(
                stage=stage,
                argv=argv,
                marker=marker,
                required_field=field,
                environment=environment,
                output_root=inputs.output_root,
            )
        standardized = inputs.output_root / "standardized"
        standardized_manifest = _object(
            standardized / "run_manifest.json", label="standardized manifest"
        )
        freeze_manifest = _object(
            standardized / "freeze_manifest.json", label="freeze manifest"
        )
        failures: list[str] = []
        if standardized_manifest.get("dataset_key") != "mutagenicity":
            failures.append("dataset")
        if standardized_manifest.get("cf_mode") != CF_MODE:
            failures.append("cf_mode")
        if standardized_manifest.get("distance_line") != DISTANCE_LINE:
            failures.append("distance_line")
        if standardized_manifest.get("teacher_sha256") != teacher_sha256:
            failures.append("teacher")
        if freeze_manifest.get("dataset_key") != "mutagenicity":
            failures.append("freeze_dataset")
        if failures:
            raise ValueError(f"Standardized output identity mismatch: {failures}")
        source_integrity = _verify_adopted_generation_integrity(adoption)
        write_json(inputs.output_root / "source_integrity_final.json", source_integrity)
        final = {
            "schema_version": (
                FAST_ACCURATE_RUN_SCHEMA
                if historical is not None
                else "mut_comrecgc_parity_standardization_v1"
            ),
            "status": "PASS",
            "dataset": "mutagenicity",
            "method": METHOD,
            "oracle_backend": "rf",
            "classifier_family": "random_forest",
            "rf_oracle_used": True,
            "cf_mode": CF_MODE,
            "distance_line": DISTANCE_LINE,
            "generation_adopted": True,
            "generation_rerun": False,
            "traceoff_reference_rerun": historical is None,
            "trace_parity_passed": historical is None,
            "trace_fields_stripped": False,
            "common_recourse_adopted": True,
            "common_recourse_rerun": False,
            "chemistry_rerun": True,
            "evaluation_rerun": True,
            "source_generation_root": str(inputs.source_generation_root),
            "source_common_recourse_root": str(common_root),
            "trace_parity_path": str(parity_path) if parity_path is not None else None,
            "trace_parity_sha256": parity["sha256"] if parity is not None else None,
            "historical_adoption_path": (
                historical["path"] if historical is not None else None
            ),
            "historical_adoption_sha256": (
                historical["sha256"] if historical is not None else None
            ),
            "standardized_output_root": str(standardized),
            "project_commit": project_commit,
            "source_payload_sha256": SOURCE_PAYLOAD_SHA256,
            "standardized_run_manifest_sha256": sha256_file(
                standardized / "run_manifest.json"
            ),
            "freeze_manifest_sha256": sha256_file(standardized / "freeze_manifest.json"),
            "teacher_sha256": teacher_sha256,
            "calibration_loaded": False,
            "test_loaded_only_in_unified_evaluation": True,
            "completed_at": _utc_now(),
        }
        if historical is not None:
            final.update(
                {
                    "historical_artifact_adopted": True,
                    "historical_source_trace_enabled": True,
                    "full_50k_rerun_performed": False,
                    "traceoff_reference_rerun": False,
                    "trace_parity_passed": False,
                    "500_step_semantic_equivalence_passed": True,
                    "adoption_without_full_50k_parity_rerun_authorized": True,
                    "generation_steps": SOURCE_STEPS,
                    "M_MAX": SOURCE_STEPS,
                    "M_EFFECTIVE": SOURCE_STEPS,
                    "early_stop_used": False,
                    "stop_reason": "HISTORICAL_FULL_50K_ARTIFACT_ADOPTION",
                    "candidate_capacity": 100_000,
                    "candidate_universe_sha": historical["candidate_universe_sha"],
                    "source_native_candidate_universe_sha": historical[
                        "source_native_candidate_universe_sha"
                    ],
                    "pair_store_source_candidate_universe_sha": historical[
                        "pair_store_source_candidate_universe_sha"
                    ],
                    "dbscan_native_candidate_universe_sha": None,
                    "dbscan_transitively_bound_candidate_universe_sha": historical[
                        "dbscan_transitively_bound_candidate_universe_sha"
                    ],
                    "candidate_universe_binding_state": "PASS",
                    "transitive_binding_kind": TRANSITIVE_BINDING_KIND,
                    "pair_candidate_graph_hashes_sha256": historical[
                        "pair_candidate_graph_hashes_sha256"
                    ],
                    "dbscan_native_candidate_universe_field_present": False,
                    "dbscan_universe_binding_via_pair_vectors": True,
                    "pair_store_reused": True,
                    "dbscan_reused": True,
                    "pair_store_rerun": False,
                    "dbscan_rerun": False,
                    "source_pair_store_manifest_path": historical[
                        "source_pair_store_manifest_path"
                    ],
                    "source_pair_store_manifest_sha256": historical[
                        "source_pair_store_manifest_sha256"
                    ],
                    "source_dbscan_manifest_path": historical[
                        "source_dbscan_manifest_path"
                    ],
                    "source_dbscan_manifest_sha256": historical[
                        "source_dbscan_manifest_sha256"
                    ],
                    "500_step_semantic_equivalence_receipt_path": historical[
                        "500_step_semantic_equivalence_receipt_path"
                    ],
                    "500_step_semantic_equivalence_receipt_sha256": historical[
                        "500_step_semantic_equivalence_receipt_sha256"
                    ],
                }
            )
        write_json(inputs.output_root / "run_manifest.json", final)
        write_json(inputs.output_root / "final_gate.json", final)
        write_json(inputs.output_root / "_RUN_COMPLETE.json", {**final, "run_complete": True})
        atomic_write_bytes(inputs.output_root / "PASS", b"PASS\n")
        print(
            "[MUT_COMRECGC_FAST_ACCURATE_STANDARDIZATION_PASS]"
            if historical is not None
            else "[MUT_COMRECGC_PARITY_STANDARDIZATION_PASS]",
            flush=True,
        )
        return final
    except Exception as exc:
        write_json(
            inputs.output_root / "FAILED.json",
            {
                "schema_version": (
                    "mut_comrecgc_fast_accurate_standardization_failure_v2"
                    if historical_adoption_path is not None
                    else "mut_comrecgc_parity_standardization_failure_v1"
                ),
                "status": "FAILED",
                "dataset": "mutagenicity",
                "error_class": type(exc).__name__,
                "message": str(exc),
                "output_root": str(inputs.output_root),
                "failed_at": _utc_now(),
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--source-generation-root", type=_absolute, required=True)
    parser.add_argument("--upstream-root", type=_absolute, required=True)
    parser.add_argument("--dataset-dir", type=_absolute, required=True)
    parser.add_argument("--distance-checkpoint", type=_absolute, required=True)
    parser.add_argument("--dataset-csv", type=_absolute, required=True)
    parser.add_argument("--teacher-path", type=_absolute, required=True)
    parser.add_argument("--molclr-root", type=_absolute, required=True)
    parser.add_argument("--molclr-checkpoint", type=_absolute, required=True)
    parser.add_argument("--thresholds-path", type=_absolute, required=True)
    parser.add_argument("--common-adoption", type=_absolute)
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--trace-parity", type=_absolute)
    evidence.add_argument("--historical-adoption", type=_absolute)
    parser.add_argument("--output-root", type=_absolute, required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = ContinuationInputs(
        dataset="mutagenicity",
        source_generation_root=_require_directory(args.source_generation_root),
        upstream_root=_require_directory(args.upstream_root),
        dataset_dir=_require_directory(args.dataset_dir),
        source_csv=None,
        distance_checkpoint=_require_file(args.distance_checkpoint),
        dataset_csv=_require_file(args.dataset_csv),
        teacher_path=_require_file(args.teacher_path),
        molclr_root=_require_directory(args.molclr_root),
        molclr_checkpoint=_require_file(args.molclr_checkpoint),
        thresholds_path=_require_file(args.thresholds_path),
        output_root=args.output_root,
        device=str(args.device),
        theta_star=None,
        cost_cap=None,
    )
    run(
        inputs,
        common_adoption_path=(
            _require_file(args.common_adoption) if args.common_adoption is not None else None
        ),
        parity_path=(
            _require_file(args.trace_parity) if args.trace_parity is not None else None
        ),
        historical_adoption_path=(
            _require_file(args.historical_adoption)
            if args.historical_adoption is not None
            else None
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
