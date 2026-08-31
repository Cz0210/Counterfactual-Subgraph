"""Real T7 worker/verifier successor consuming ``TasteGCFReleasePinsV1``.

The worker executes the existing native full-graph VRRW smoke and may only
seal aggregate evidence.  A separate process reopens the typed release and
every source it binds, validates the sealed science, and is the only process
that can publish the managed-v2 terminal directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from src.baselines.tastemolnet_gcf_smoke import (
    DATASET,
    LABEL_MAP,
    METHOD,
    NUM_CLASSES,
    PASS_MARKER,
    SMOKE_GPU_INDEX,
    SMOKE_SOURCE_POOL_LIMIT,
    SMOKE_STEPS,
    SOURCE_LABEL,
    STAGE,
    _validate_progress_evidence,
    execute_native_vrrw_smoke,
    load_train_rows,
    parse_candidate_trace,
)
from src.utils.managed_execution_v2 import (
    ATTEMPT_MANIFEST_SCHEMA,
    WORKER_EXIT_SCHEMA,
    WORKER_RAW_EVIDENCE_SCHEMA,
    create_managed_attempt,
    create_worker_staging,
    load_verified_gate,
    write_worker_exit,
    write_worker_raw_evidence,
)
from src.utils.process_identity_v2 import canonical_json_bytes
from src.utils.tastemolnet_t7_typed_release_v1 import (
    HeldVerifiedT7ReleaseV1,
    T7_TYPED_RAW_EVIDENCE_SCHEMA,
    T7_TYPED_VERIFICATION_SCHEMA,
)
from src.utils.terminal_publisher_v2 import (
    HeldSealedArtifactV2,
    SealedWorkerArtifactV2,
    TerminalPublicationV2,
    open_sealed_worker_artifact,
    seal_worker_staging,
    verify_and_publish_sealed_attempt,
)


TASK_ID = "T7_GCF_SMOKE_TYPED_RELEASE_V1"
TRUST_MODEL = "TRUSTED_SINGLE_OPERATOR_ROOT"
SCIENCE_FILE = "artifacts/gcf_smoke.json"
RELEASE_BINDING_FILE = "artifacts/typed_release_binding.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_GPU_UUID = re.compile(r"^GPU-[A-Za-z0-9-]+$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class TasteGCFManagedRuntimeError(RuntimeError):
    """The typed T7 runtime, science, or terminal binding changed."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_mapping(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(canonical_json_bytes(dict(value)))


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise TasteGCFManagedRuntimeError(f"{label} must be lowercase SHA-256")
    return value


def _absolute(
    value: str | Path, *, label: str, must_exist: bool = True
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise TasteGCFManagedRuntimeError(f"{label} must be normalized and absolute")
    if must_exist:
        try:
            physical = path.resolve(strict=True)
        except OSError as exc:
            raise TasteGCFManagedRuntimeError(f"{label} is unavailable") from exc
        if physical != path:
            raise TasteGCFManagedRuntimeError(f"{label} contains a symlink or alias")
    return path


def _json_object(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteGCFManagedRuntimeError(f"{label} is malformed JSON") from exc
    if type(value) is not dict:
        raise TasteGCFManagedRuntimeError(f"{label} must be one JSON object")
    return value


def _jsonl_bytes(rows: list[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def t7_release_binding(release: HeldVerifiedT7ReleaseV1) -> dict[str, Any]:
    """Project a held typed release onto immutable runtime authority."""

    evidence = dict(release.revalidate())
    authority = release.authority
    pins = release.pins
    predecessor = release.sources.neurosed_evidence
    if type(predecessor) is not dict:
        raise TasteGCFManagedRuntimeError(
            "T7 held NeuroSED predecessor evidence is absent"
        )
    binding = {
        "schema_version": "tastemolnet_t7_typed_runtime_binding_v1",
        "release_root": str(release.release_root),
        "release_id": evidence["release_id"],
        "release_pins_sha256": evidence["release_pins_sha256"],
        "source_authority_sha256": evidence["source_authority_sha256"],
        "release_verification_sha256": evidence["verification_sha256"],
        "release_gate_sha256": evidence["gate_sha256"],
        "release_pass_sha256": evidence["pass_sha256"],
        "official_gcf_commit": pins.official_gcf_commit,
        "neurosed_commit": pins.neurosed_commit,
        "managed_neurosed_pass_sha256": authority.managed_neurosed_pass_sha256,
        "managed_neurosed_gate_sha256": authority.managed_neurosed_gate_sha256,
        "managed_neurosed_verification_sha256": (
            authority.managed_neurosed_verification_sha256
        ),
        "neurosed_predecessor_sha256": _sha256_mapping(predecessor),
        "neurosed_model_sha256": pins.neurosed_model_sha,
        "neurosed_config_sha256": pins.neurosed_config_sha,
        "neurosed_pair_manifest_sha256": pins.neurosed_pair_manifest_sha,
        "t3_calibrated_gine_sha256": pins.t3_calibrated_gine_sha,
        "t3_temperature_sha256": pins.t3_temperature_sha,
        "t3_checkpoint_id_sha256": authority.t3_checkpoint_id,
        "dataset_sha256": pins.dataset_sha,
        "train_split_sha256": pins.train_split_sha,
        "validation_split_sha256": pins.validation_split_sha,
        "calibration_split_sha256": pins.calibration_split_sha,
        "test_split_sha256": pins.test_split_sha,
        "official_gcf_inventory_sha256": authority.official_gcf_inventory_sha256,
        "execution_commit": authority.implementation_commit,
        "execution_tree": authority.implementation_tree,
        "inference_direction": pins.inference_direction,
        "neurosed_distance_threshold": authority.neurosed_distance_threshold,
        "neurosed_retrained": False,
        "validation_payload_loaded": False,
        "calibration_payload_loaded": False,
        "test_payload_loaded": False,
        "rf_oracle_used": False,
    }
    validate_t7_release_binding(binding)
    return binding


def validate_t7_release_binding(raw: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version", "release_root", "release_id",
        "release_pins_sha256", "source_authority_sha256",
        "release_verification_sha256", "release_gate_sha256",
        "release_pass_sha256", "official_gcf_commit", "neurosed_commit",
        "managed_neurosed_pass_sha256",
        "managed_neurosed_gate_sha256",
        "managed_neurosed_verification_sha256", "neurosed_model_sha256",
        "neurosed_predecessor_sha256",
        "neurosed_config_sha256", "neurosed_pair_manifest_sha256",
        "t3_calibrated_gine_sha256", "t3_temperature_sha256",
        "t3_checkpoint_id_sha256",
        "dataset_sha256", "train_split_sha256", "validation_split_sha256",
        "calibration_split_sha256", "test_split_sha256",
        "official_gcf_inventory_sha256", "execution_commit", "execution_tree",
        "inference_direction", "neurosed_distance_threshold",
        "neurosed_retrained", "validation_payload_loaded",
        "calibration_payload_loaded", "test_payload_loaded", "rf_oracle_used",
    }
    if type(raw) is not dict or set(raw) != expected:
        raise TasteGCFManagedRuntimeError("T7 runtime release binding keys changed")
    if (
        raw.get("schema_version") != "tastemolnet_t7_typed_runtime_binding_v1"
        or raw.get("inference_direction") != "generated_to_original"
        or any(
            raw.get(name) is not False
            for name in (
                "neurosed_retrained", "validation_payload_loaded",
                "calibration_payload_loaded", "test_payload_loaded",
                "rf_oracle_used",
            )
        )
        or type(raw.get("release_id")) is not str
        or not raw["release_id"]
        or _GIT_SHA.fullmatch(str(raw.get("execution_commit"))) is None
        or _GIT_SHA.fullmatch(str(raw.get("execution_tree"))) is None
        or _GIT_SHA.fullmatch(str(raw.get("official_gcf_commit"))) is None
        or _GIT_SHA.fullmatch(str(raw.get("neurosed_commit"))) is None
        or isinstance(raw.get("neurosed_distance_threshold"), bool)
        or not isinstance(raw.get("neurosed_distance_threshold"), (int, float))
        or not math.isfinite(float(raw["neurosed_distance_threshold"]))
        or float(raw["neurosed_distance_threshold"]) < 0.0
    ):
        raise TasteGCFManagedRuntimeError("T7 runtime release semantics changed")
    _absolute(raw["release_root"], label="T7 typed release root")
    for name in expected:
        if name.endswith("_sha256"):
            _require_sha256(raw[name], label=name)
    return dict(raw)


def t7_managed_input_hashes(binding: Mapping[str, Any]) -> dict[str, str]:
    frozen = validate_t7_release_binding(binding)
    names = (
        "release_pins_sha256", "source_authority_sha256",
        "release_verification_sha256", "release_gate_sha256",
        "release_pass_sha256", "managed_neurosed_pass_sha256",
        "managed_neurosed_gate_sha256",
        "managed_neurosed_verification_sha256", "neurosed_model_sha256",
        "neurosed_predecessor_sha256",
        "neurosed_config_sha256", "neurosed_pair_manifest_sha256",
        "t3_calibrated_gine_sha256", "t3_temperature_sha256",
        "t3_checkpoint_id_sha256",
        "dataset_sha256", "train_split_sha256", "validation_split_sha256",
        "calibration_split_sha256", "test_split_sha256",
        "official_gcf_inventory_sha256",
    )
    return {name.removesuffix("_sha256"): frozen[name] for name in names}


def t7_managed_config_hash(binding: Mapping[str, Any]) -> str:
    """Hash the actual fixed smoke configuration, not an unrelated YAML path."""

    frozen = validate_t7_release_binding(binding)
    return _sha256_mapping(
        {
            "schema_version": "tastemolnet_t7_managed_config_v1",
            "release_pins_sha256": frozen["release_pins_sha256"],
            "source_authority_sha256": frozen["source_authority_sha256"],
            "physical_gpu_index": SMOKE_GPU_INDEX,
            "num_classes": NUM_CLASSES,
            "source_label": SOURCE_LABEL,
            "source_pool_limit": SMOKE_SOURCE_POOL_LIMIT,
            "random_walk_steps": SMOKE_STEPS,
            "inference_direction": "generated_to_original",
            "neurosed_distance_threshold": frozen[
                "neurosed_distance_threshold"
            ],
            "validation_payload_loaded": False,
            "calibration_payload_loaded": False,
            "test_payload_loaded": False,
            "rf_oracle_used": False,
        }
    )


def require_gpu0_runtime(gpu_uuid: str) -> None:
    if _GPU_UUID.fullmatch(gpu_uuid) is None:
        raise TasteGCFManagedRuntimeError("T7 GPU UUID is invalid")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise TasteGCFManagedRuntimeError("T7 requires CUDA_VISIBLE_DEVICES=0")
    if os.environ.get("AUTODL_PHYSICAL_GPU_INDEX") != "0":
        raise TasteGCFManagedRuntimeError("T7 physical GPU0 binding is absent")
    if os.environ.get("AUTODL_PHYSICAL_GPU_UUID") != gpu_uuid:
        raise TasteGCFManagedRuntimeError("T7 physical GPU UUID binding changed")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise TasteGCFManagedRuntimeError("T7 requires exactly one visible CUDA device")


def _valid_batch_scorer(value: Any) -> bool:
    expected = {
        "schema_version", "calls", "cache_capacity", "cache_entries",
        "cache_hits", "cache_misses", "scored_rows", "cache_scope",
        "partial_row_reuse", "deduplication", "chunking",
    }
    return bool(
        type(value) is dict
        and set(value) == expected
        and value.get("schema_version") == 1
        and all(
            type(value.get(name)) is int and value[name] >= 0
            for name in (
                "calls", "cache_capacity", "cache_entries", "cache_hits",
                "cache_misses", "scored_rows",
            )
        )
        and value.get("calls", 0) > 0
        and value.get("scored_rows", 0) > 0
        and value.get("cache_scope") == "exact_complete_ordered_batch_v1"
        and value.get("partial_row_reuse") is False
        and value.get("deduplication") is False
        and value.get("chunking") is False
    )


def validate_native_t7_science(
    raw: Mapping[str, Any],
    *,
    expected_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the aggregate result emitted by the existing native core."""

    if type(raw) is not dict or set(raw) != {"trace", "summary"}:
        raise TasteGCFManagedRuntimeError("T7 native science shape changed")
    trace_value = raw.get("trace")
    summary = raw.get("summary")
    if type(trace_value) is not list or type(summary) is not dict:
        raise TasteGCFManagedRuntimeError("T7 native science values changed")
    trace = parse_candidate_trace(_jsonl_bytes(trace_value))
    expected_summary_keys = {
        "schema_version", "stage", "dataset", "method", "parent_evidence",
        "official_random_walk_steps", "progress_checkpoint",
        "official_candidate_count", "strict_counterfactual_candidate_count",
        "destination_prediction_counts", "native_action_invocation_counts",
        "importance_bridge_calls", "importance_bridge_evaluated_graphs",
        "neurosed_distance_calls", "neurosed_distance_evaluated_graphs",
        "adapter", "alpha", "coverage_mode", "neurosed_distance_threshold",
        "neurosed_predecessor", "candidate_condition", "score_definition",
        "native_full_graph_semantics", "deletion_only_semantics",
        "neurosed_status", "distance_status", "selector_status",
        "full_route_status", "bace_artifacts_used", "rf_oracle_used",
        "train_loaded", "validation_payload_loaded",
        "calibration_payload_loaded", "test_payload_loaded",
        "native_graph_payload_persisted", "molecule_payload_persisted",
        "paper_result_eligible",
    }
    if set(summary) != expected_summary_keys:
        raise TasteGCFManagedRuntimeError("T7 native summary keys changed")
    progress = summary.get("progress_checkpoint")
    neurosed = summary.get("neurosed_predecessor")
    parent = summary.get("parent_evidence")
    adapter = summary.get("adapter")
    actions = summary.get("native_action_invocation_counts")
    strict_count = sum(row["candidate_condition"] for row in trace)
    destinations = {
        str(label): sum(
            row["candidate_condition"] and row["pred_candidate"] == label
            for row in trace
        )
        for label in (0, 2)
    }
    if (
        summary.get("schema_version") != "tastemolnet_gcf_native_vrrw_smoke_v2"
        or summary.get("stage") != STAGE
        or summary.get("dataset") != DATASET
        or summary.get("method") != METHOD
        or summary.get("official_random_walk_steps") != SMOKE_STEPS
        or summary.get("official_candidate_count") != len(trace)
        or summary.get("strict_counterfactual_candidate_count") != strict_count
        or strict_count <= 0
        or summary.get("destination_prediction_counts") != destinations
        or type(parent) is not dict
        or set(parent)
        != {
            "source_pool_count", "source_pool_gine_correct_sweet",
            "selected_parent_count", "selected_parent_graph_hashes_sha256",
            "pred_before",
        }
        or parent.get("source_pool_count") != SMOKE_SOURCE_POOL_LIMIT
        or type(parent.get("source_pool_gine_correct_sweet")) is not int
        or parent["source_pool_gine_correct_sweet"] < 8
        or parent.get("selected_parent_count") != 8
        or parent.get("pred_before") != SOURCE_LABEL
        or _SHA256.fullmatch(
            str(parent.get("selected_parent_graph_hashes_sha256"))
        ) is None
        or type(actions) is not dict
        or not actions
        or any(type(value) is not int or value <= 0 for value in actions.values())
        or set(actions) - {"NOTHING", "NLC", "NA", "INA", "NR", "INR", "ER", "ERR", "EA"}
        or sum(value for name, value in actions.items() if name != "NOTHING") <= 0
        or any(
            type(summary.get(name)) is not int or summary[name] <= 0
            for name in (
                "importance_bridge_calls", "importance_bridge_evaluated_graphs",
                "neurosed_distance_calls", "neurosed_distance_evaluated_graphs",
            )
        )
        or type(adapter) is not dict
        or set(adapter)
        != {
            "schema_version", "checkpoint_id", "num_classes", "source_label",
            "call_count", "decode_success_count", "empty_valid_batch_count",
            "decode_failures", "batch_scorer", "rf_oracle_used",
        }
        or adapter.get("schema_version")
        != "tastemolnet_gcf_native_gine_adapter_v1"
        or _SHA256.fullmatch(str(adapter.get("checkpoint_id"))) is None
        or adapter.get("num_classes") != NUM_CLASSES
        or adapter.get("source_label") != SOURCE_LABEL
        or type(adapter.get("call_count")) is not int
        or adapter["call_count"] <= 0
        or type(adapter.get("decode_success_count")) is not int
        or adapter["decode_success_count"] <= 0
        or type(adapter.get("empty_valid_batch_count")) is not int
        or adapter["empty_valid_batch_count"] < 0
        or type(adapter.get("decode_failures")) is not dict
        or adapter.get("rf_oracle_used") is not False
        or not _valid_batch_scorer(adapter.get("batch_scorer"))
        or summary.get("alpha") != 1.0
        or summary.get("candidate_condition") != "pred_candidate != source_label"
        or summary.get("score_definition") != "1.0 - p_source"
        or summary.get("coverage_mode")
        != "official_taste_neurosed_threshold_coverage"
        or summary.get("native_full_graph_semantics") is not True
        or summary.get("deletion_only_semantics") is not False
        or summary.get("neurosed_status") != "PASS_INPUT_REVALIDATED"
        or summary.get("distance_status") != "EVALUATED"
        or summary.get("selector_status") != "NOT_EVALUATED"
        or summary.get("train_loaded") is not True
        or any(
            summary.get(name) is not False
            for name in (
                "validation_payload_loaded", "calibration_payload_loaded",
                "test_payload_loaded", "bace_artifacts_used", "rf_oracle_used",
                "native_graph_payload_persisted", "molecule_payload_persisted",
                "paper_result_eligible",
            )
        )
        or type(neurosed) is not dict
        or set(neurosed)
        != {
            "schema_version", "status", "marker", "final_root", "attempt_id",
            "generation_token", "pass_path", "pass_sha256", "gate_path",
            "gate_sha256", "verification_path", "verification_sha256",
            "source_inventory_sha256", "published_inventory_sha256",
            "checkpoint_path", "checkpoint_sha256", "feature_schema_path",
            "feature_schema_sha256", "sha256s_path", "sha256s_sha256",
            "t7_consumer",
        }
        or neurosed.get("schema_version")
        != "tastemolnet_gcf_neurosed_managed_final_v1"
        or neurosed.get("status") != "PASS"
        or neurosed.get("marker") != "MANAGED_EXECUTION_V2_PASS"
        or type(neurosed.get("t7_consumer")) is not dict
        or neurosed["t7_consumer"].get("role")
        != "GCF_AUXILIARY_DISTANCE_MODEL"
        or neurosed["t7_consumer"].get("classifier") is not False
        or neurosed["t7_consumer"].get("calibration_loaded") is not False
        or neurosed["t7_consumer"].get("test_loaded") is not False
        or type(progress) is not dict
    ):
        raise TasteGCFManagedRuntimeError("T7 native science semantics changed")
    for name in (
        "pass_sha256", "gate_sha256", "verification_sha256",
        "source_inventory_sha256", "published_inventory_sha256",
        "checkpoint_sha256", "feature_schema_sha256", "sha256s_sha256",
    ):
        _require_sha256(neurosed[name], label=f"NeuroSED {name}")
    if expected_binding is not None:
        binding = validate_t7_release_binding(expected_binding)
        if (
            adapter["checkpoint_id"] != binding["t3_checkpoint_id_sha256"]
            or summary["neurosed_distance_threshold"]
            != binding["neurosed_distance_threshold"]
            or neurosed["pass_sha256"]
            != binding["managed_neurosed_pass_sha256"]
            or neurosed["gate_sha256"]
            != binding["managed_neurosed_gate_sha256"]
            or neurosed["verification_sha256"]
            != binding["managed_neurosed_verification_sha256"]
            or neurosed["checkpoint_sha256"]
            != binding["neurosed_model_sha256"]
            or _sha256_mapping(neurosed)
            != binding["neurosed_predecessor_sha256"]
            or neurosed["t7_consumer"].get("checkpoint_sha256")
            != binding["neurosed_model_sha256"]
        ):
            raise TasteGCFManagedRuntimeError(
                "T7 science differs from typed release inputs"
            )
    _validate_progress_evidence(progress)
    return {"trace": trace, "summary": dict(summary)}


def _write_artifact_json(staging: Any, name: str, value: Mapping[str, Any]) -> None:
    if "/" in name or name in {"", ".", "..", ".generation_token.json"}:
        raise TasteGCFManagedRuntimeError("T7 artifact name is unsafe")
    staging.revalidate()
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=staging.artifact_descriptor,
    )
    try:
        view = memoryview(canonical_json_bytes(dict(value)))
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise TasteGCFManagedRuntimeError("T7 artifact write was short")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(staging.artifact_descriptor)


def _validate_attempt_manifest(
    manifest: Mapping[str, Any],
    *,
    attempt_id: str,
    attempt_path: Path,
    attempt_generation_token: str,
    run_id: str,
    binding: Mapping[str, Any],
) -> None:
    frozen = validate_t7_release_binding(binding)
    expected_keys = {
        "schema_version", "status", "attempt_id", "controller_id", "task_id",
        "git_commit", "config_hash", "input_hashes", "created_at", "hostname",
        "boot_id", "attempt_path", "generation_token",
        "auto_terminate_uncontrolled_children",
    }
    if (
        type(manifest) is not dict
        or set(manifest) != expected_keys
        or manifest.get("schema_version") != ATTEMPT_MANIFEST_SCHEMA
        or manifest.get("status") != "ACTIVE"
        or manifest.get("attempt_id") != attempt_id
        or manifest.get("controller_id") != run_id
        or manifest.get("task_id") != TASK_ID
        or manifest.get("git_commit") != frozen["execution_commit"]
        or manifest.get("config_hash") != t7_managed_config_hash(frozen)
        or manifest.get("input_hashes") != t7_managed_input_hashes(frozen)
        or manifest.get("attempt_path") != str(attempt_path)
        or manifest.get("generation_token") != attempt_generation_token
        or manifest.get("auto_terminate_uncontrolled_children") is not False
    ):
        raise TasteGCFManagedRuntimeError("T7 managed attempt manifest changed")


def seal_t7_worker_evidence(
    staging: Any,
    *,
    science: Mapping[str, Any],
    release_binding: Mapping[str, Any],
    expected_final_path: str | Path,
    run_id: str,
    gpu_uuid: str,
) -> SealedWorkerArtifactV2:
    """Seal only aggregate science and release bindings; never write PASS."""

    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise TasteGCFManagedRuntimeError("T7 run ID is unsafe")
    if _GPU_UUID.fullmatch(gpu_uuid) is None:
        raise TasteGCFManagedRuntimeError("T7 GPU UUID is invalid")
    binding = validate_t7_release_binding(release_binding)
    frozen_science = validate_native_t7_science(
        science, expected_binding=binding
    )
    destination = _absolute(
        expected_final_path, label="T7 final path", must_exist=False
    )
    _absolute(destination.parent, label="T7 final parent")
    if destination.exists() or destination.is_symlink():
        raise TasteGCFManagedRuntimeError("T7 final path must remain fresh")
    staging.revalidate()
    manifest = dict(staging.attempt.revalidate())
    _validate_attempt_manifest(
        manifest,
        attempt_id=staging.attempt.attempt_id,
        attempt_path=staging.attempt.attempt_path,
        attempt_generation_token=staging.attempt.generation_token,
        run_id=run_id,
        binding=binding,
    )
    _write_artifact_json(staging, "gcf_smoke.json", frozen_science)
    _write_artifact_json(staging, "typed_release_binding.json", binding)
    raw = write_worker_raw_evidence(
        staging,
        {
            "schema_version": T7_TYPED_RAW_EVIDENCE_SCHEMA,
            "status": "RAW_EVIDENCE_ONLY",
            "stage": STAGE,
            "dataset": DATASET,
            "method": METHOD,
            "task_id": TASK_ID,
            "trust_model": TRUST_MODEL,
            "expected_final_path": str(destination),
            "run_id": run_id,
            "physical_gpu_index": SMOKE_GPU_INDEX,
            "gpu_uuid": gpu_uuid,
            "attempt_manifest": manifest,
            "release_binding_sha256": _sha256_mapping(binding),
            "science_sha256": _sha256_mapping(frozen_science),
            "science_file": SCIENCE_FILE,
            "release_binding_file": RELEASE_BINDING_FILE,
            "num_classes": NUM_CLASSES,
            "source_label": SOURCE_LABEL,
            "inference_direction": "generated_to_original",
            "worker_wrote_verification": False,
            "worker_wrote_gate": False,
            "worker_wrote_pass": False,
            "neurosed_retrained": False,
            "train_loaded": True,
            "validation_payload_loaded": False,
            "calibration_payload_loaded": False,
            "test_payload_loaded": False,
            "rf_oracle_used": False,
            "independent_verification_required": True,
        },
    )
    raw.close()
    worker_exit = write_worker_exit(
        staging,
        {
            "exit_code": 0,
            "science_complete": True,
            "official_random_walk_steps": SMOKE_STEPS,
            "worker_closed_science_state_writers": True,
            "worker_wrote_verifier_output": False,
            "neurosed_retrained": False,
        },
    )
    worker_exit.close()
    return seal_worker_staging(staging)


def run_t7_worker(
    *,
    stage_root: str | Path,
    final_path: str | Path,
    release: HeldVerifiedT7ReleaseV1,
    run_id: str,
    gpu_uuid: str,
) -> dict[str, Any]:
    """Execute the existing native 16-step VRRW core and return SEALED only."""

    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise TasteGCFManagedRuntimeError("T7 run ID is unsafe")
    binding = t7_release_binding(release)
    require_gpu0_runtime(gpu_uuid)
    root = _absolute(stage_root, label="T7 managed stage root")
    destination = _absolute(final_path, label="T7 final path", must_exist=False)
    if destination.exists() or destination.is_symlink():
        raise TasteGCFManagedRuntimeError("T7 final path must be fresh")
    with create_managed_attempt(
        stage_root=root,
        controller_id=run_id,
        task_id=TASK_ID,
        git_commit=binding["execution_commit"],
        config_hash=t7_managed_config_hash(binding),
        input_hashes=t7_managed_input_hashes(binding),
    ) as attempt, create_worker_staging(attempt) as staging:
        release.revalidate()
        sources = release.sources
        loaded = load_train_rows(
            sources.train_bytes,
            source_path=Path(sources.train_contract["path"]),
            expected_num_records=sources.train_contract["num_records"],
            expected_label_counts=sources.train_contract["label_counts"],
        )
        science = execute_native_vrrw_smoke(
            checkpoint_payloads=sources.checkpoint_payloads,
            source_rows=loaded.sweet_rows[:SMOKE_SOURCE_POOL_LIMIT],
            graph_schema=loaded.schema,
            official_root=sources.official_root,
            neurosed_checkpoint_path=f"/proc/self/fd/{sources.neurosed_model.file_fd}",
            neurosed_distance_threshold=(
                release.authority.neurosed_distance_threshold
            ),
            neurosed_evidence=sources.neurosed_evidence,
            neurosed_revalidate=release.revalidate,
            device="cuda:0",
        )
        release.revalidate()
        sealed = seal_t7_worker_evidence(
            staging,
            science=science,
            release_binding=binding,
            expected_final_path=destination,
            run_id=run_id,
            gpu_uuid=gpu_uuid,
        )
        return {
            "status": "SEALED_PENDING_INDEPENDENT_VERIFICATION",
            "stage": STAGE,
            "attempt_id": sealed.attempt_id,
            "generation_token": sealed.generation_token,
            "staging_path": str(sealed.staging_path),
            "seal_path": str(sealed.seal_path),
            "seal_sha256": sealed.seal_sha256,
            "inventory_sha256": sealed.inventory_sha256,
            "expected_final_path": str(destination),
        }


def _read_held_json(
    held: HeldSealedArtifactV2, relative_path: str
) -> dict[str, Any]:
    matches = [
        item for item in held.files if item.evidence.relative_path == relative_path
    ]
    if len(matches) != 1:
        raise TasteGCFManagedRuntimeError(
            f"T7 SEALED evidence lacks {relative_path}"
        )
    item = matches[0]
    item.revalidate()
    data = bytearray()
    offset = 0
    while offset < item.evidence.size:
        block = os.pread(
            item.descriptor,
            min(1024 * 1024, item.evidence.size - offset),
            offset,
        )
        if not block:
            raise TasteGCFManagedRuntimeError("T7 SEALED JSON ended early")
        data.extend(block)
        offset += len(block)
    if os.pread(item.descriptor, 1, item.evidence.size):
        raise TasteGCFManagedRuntimeError("T7 SEALED JSON grew")
    value = _json_object(bytes(data), label=relative_path)
    if canonical_json_bytes(value) != bytes(data):
        raise TasteGCFManagedRuntimeError(f"{relative_path} is not canonical JSON")
    item.revalidate()
    return value


def _validate_sealed_t7(
    held: HeldSealedArtifactV2,
    *,
    expected_binding: Mapping[str, Any],
    final_path: Path,
    run_id: str,
    gpu_uuid: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = validate_t7_release_binding(expected_binding)
    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise TasteGCFManagedRuntimeError("T7 run ID is unsafe")
    if _GPU_UUID.fullmatch(gpu_uuid) is None:
        raise TasteGCFManagedRuntimeError("T7 GPU UUID is invalid")
    held.revalidate()
    files = {item.evidence.relative_path for item in held.files}
    directories = {item.relative_path for item in held.inventory.directories}
    if files != {
        ".generation_token.json",
        "artifacts/.generation_token.json",
        SCIENCE_FILE,
        RELEASE_BINDING_FILE,
        "raw_evidence.json",
        "worker_exit.json",
    } or directories != {"artifacts"}:
        raise TasteGCFManagedRuntimeError("T7 SEALED aggregate inventory changed")
    persisted_binding = validate_t7_release_binding(
        _read_held_json(held, RELEASE_BINDING_FILE)
    )
    raw = _read_held_json(held, "raw_evidence.json")
    worker_exit = _read_held_json(held, "worker_exit.json")
    if persisted_binding != binding:
        raise TasteGCFManagedRuntimeError(
            "T7 worker release binding differs from verifier"
        )
    science = validate_native_t7_science(
        _read_held_json(held, SCIENCE_FILE), expected_binding=binding
    )
    evidence = raw.get("evidence")
    manifest = evidence.get("attempt_manifest") if type(evidence) is dict else None
    expected_evidence = {
        "schema_version": T7_TYPED_RAW_EVIDENCE_SCHEMA,
        "status": "RAW_EVIDENCE_ONLY",
        "stage": STAGE,
        "dataset": DATASET,
        "method": METHOD,
        "task_id": TASK_ID,
        "trust_model": TRUST_MODEL,
        "expected_final_path": str(final_path),
        "run_id": run_id,
        "physical_gpu_index": SMOKE_GPU_INDEX,
        "gpu_uuid": gpu_uuid,
        "attempt_manifest": manifest,
        "release_binding_sha256": _sha256_mapping(binding),
        "science_sha256": _sha256_mapping(science),
        "science_file": SCIENCE_FILE,
        "release_binding_file": RELEASE_BINDING_FILE,
        "num_classes": NUM_CLASSES,
        "source_label": SOURCE_LABEL,
        "inference_direction": "generated_to_original",
        "worker_wrote_verification": False,
        "worker_wrote_gate": False,
        "worker_wrote_pass": False,
        "neurosed_retrained": False,
        "train_loaded": True,
        "validation_payload_loaded": False,
        "calibration_payload_loaded": False,
        "test_payload_loaded": False,
        "rf_oracle_used": False,
        "independent_verification_required": True,
    }
    if (
        set(raw)
        != {"schema_version", "attempt_id", "generation_token", "recorded_at", "evidence"}
        or raw.get("schema_version") != WORKER_RAW_EVIDENCE_SCHEMA
        or raw.get("attempt_id") != held.sealed.attempt_id
        or raw.get("generation_token") != held.sealed.generation_token
        or type(raw.get("recorded_at")) is not str
        or not raw["recorded_at"]
        or evidence != expected_evidence
        or type(manifest) is not dict
    ):
        raise TasteGCFManagedRuntimeError("T7 worker raw evidence changed")
    _validate_attempt_manifest(
        manifest,
        attempt_id=held.sealed.attempt_id,
        attempt_path=held.staging_path.parent.parent,
        attempt_generation_token=str(manifest.get("generation_token")),
        run_id=run_id,
        binding=binding,
    )
    exit_payload = worker_exit.get("exit")
    if (
        set(worker_exit)
        != {"schema_version", "attempt_id", "generation_token", "recorded_at", "exit"}
        or worker_exit.get("schema_version") != WORKER_EXIT_SCHEMA
        or worker_exit.get("attempt_id") != held.sealed.attempt_id
        or worker_exit.get("generation_token") != held.sealed.generation_token
        or type(worker_exit.get("recorded_at")) is not str
        or not worker_exit["recorded_at"]
        or exit_payload
        != {
            "exit_code": 0,
            "science_complete": True,
            "official_random_walk_steps": SMOKE_STEPS,
            "worker_closed_science_state_writers": True,
            "worker_wrote_verifier_output": False,
            "neurosed_retrained": False,
        }
    ):
        raise TasteGCFManagedRuntimeError("T7 worker exit evidence changed")
    held.revalidate()
    return science, binding


def verify_and_publish_t7_sealed(
    held: HeldSealedArtifactV2,
    *,
    final_path: str | Path,
    release: HeldVerifiedT7ReleaseV1,
    run_id: str,
    gpu_uuid: str,
    force_cross_filesystem: bool = False,
) -> tuple[TerminalPublicationV2, dict[str, Any]]:
    """Independently reopen release inputs, validate, and publish T7 PASS."""

    destination = _absolute(final_path, label="T7 final path", must_exist=False)
    _absolute(destination.parent, label="T7 final parent")
    if destination.exists() or destination.is_symlink():
        raise TasteGCFManagedRuntimeError("T7 final path is not fresh")
    expected_binding = t7_release_binding(release)
    science, sealed_binding = _validate_sealed_t7(
        held,
        expected_binding=expected_binding,
        final_path=destination,
        run_id=run_id,
        gpu_uuid=gpu_uuid,
    )
    if sealed_binding != expected_binding:
        raise TasteGCFManagedRuntimeError("T7 SEALED release binding changed")
    release.revalidate()
    if t7_release_binding(release) != expected_binding:
        raise TasteGCFManagedRuntimeError("T7 release changed before publication")
    strict_count = science["summary"]["strict_counterfactual_candidate_count"]
    verification = {
        "schema_version": T7_TYPED_VERIFICATION_SCHEMA,
        "status": "PASS",
        "stage": STAGE,
        "dataset": DATASET,
        "method": METHOD,
        "task_id": TASK_ID,
        "domain_marker": PASS_MARKER,
        "trust_model": TRUST_MODEL,
        "physical_gpu_index": SMOKE_GPU_INDEX,
        "gpu_uuid": gpu_uuid,
        "execution_commit": expected_binding["execution_commit"],
        "execution_tree": expected_binding["execution_tree"],
        "release_binding_sha256": _sha256_mapping(expected_binding),
        "typed_release_root": expected_binding["release_root"],
        "typed_release_id": expected_binding["release_id"],
        "release_pins_sha256": expected_binding["release_pins_sha256"],
        "source_authority_sha256": expected_binding["source_authority_sha256"],
        "official_gcf_commit": expected_binding["official_gcf_commit"],
        "neurosed_commit": expected_binding["neurosed_commit"],
        "neurosed_model_sha256": expected_binding["neurosed_model_sha256"],
        "neurosed_predecessor_sha256": expected_binding[
            "neurosed_predecessor_sha256"
        ],
        "t3_calibrated_gine_sha256": expected_binding[
            "t3_calibrated_gine_sha256"
        ],
        "t3_temperature_sha256": expected_binding["t3_temperature_sha256"],
        "science_sha256": _sha256_mapping(science),
        "official_random_walk_steps": SMOKE_STEPS,
        "strict_counterfactual_candidate_count": strict_count,
        "same_calibrated_three_class_gine": True,
        "native_full_graph_semantics": True,
        "generated_to_original_neurosed": True,
        "neurosed_retrained": False,
        "train_only": True,
        "validation_payload_loaded": False,
        "calibration_payload_loaded": False,
        "test_payload_loaded": False,
        "rf_oracle_used": False,
        "worker_self_signed": False,
        "independent_verifier": True,
    }
    publication = verify_and_publish_sealed_attempt(
        held,
        final_path=destination,
        verification=verification,
        force_cross_filesystem=force_cross_filesystem,
    )
    return publication, verification


def open_t7_sealed(
    path: str | Path,
    *,
    expected_attempt_id: str,
    expected_generation_token: str,
) -> HeldSealedArtifactV2:
    return open_sealed_worker_artifact(
        path,
        expected_attempt_id=expected_attempt_id,
        expected_generation_token=expected_generation_token,
    )


def load_t7_verified_gate(path: str | Path) -> Mapping[str, Any]:
    return load_verified_gate(path)


__all__ = [
    "RELEASE_BINDING_FILE",
    "SCIENCE_FILE",
    "TASK_ID",
    "TRUST_MODEL",
    "TasteGCFManagedRuntimeError",
    "load_t7_verified_gate",
    "open_t7_sealed",
    "require_gpu0_runtime",
    "run_t7_worker",
    "seal_t7_worker_evidence",
    "t7_managed_config_hash",
    "t7_managed_input_hashes",
    "t7_release_binding",
    "validate_native_t7_science",
    "validate_t7_release_binding",
    "verify_and_publish_t7_sealed",
]
