"""Publish one verified T8 deadline recovery as a managed-v2 typed final.

The deadline runner owns the real two-branch GlobalGCE science.  This module
does not execute, copy, or alter that science.  A first process reopens the
aggregate deadline terminal plus its private state tree and emits only SEALED
adoption evidence.  A second process repeats every source check and is the
only process allowed to publish the managed-v2 PASS consumed by T13.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
from typing import Any, Mapping

from scripts.autodl import run_tastemolnet_t8_deadline as deadline
from src.baselines.globalgce_bace_native_rules import OFFICIAL_GLOBALGCE_COMMIT
from src.baselines.tastemolnet_globalgce_smoke import (
    DATASET,
    MANAGED_TASK_ID,
    METHOD,
    PASS_MARKER as T8_PASS_MARKER,
    STAGE,
    TARGET_BRANCHES,
    TasteGlobalGCESmokeConfig,
    TasteGlobalGCESmokeError,
    ZERO_CANDIDATE_RECOVERY_EPOCHS,
    validate_science_summary,
)
from src.utils.managed_execution_v2 import (
    ATTEMPT_MANIFEST_SCHEMA,
    WORKER_EXIT_SCHEMA,
    WORKER_RAW_EVIDENCE_SCHEMA,
    create_managed_attempt,
    create_worker_staging,
    write_worker_exit,
    write_worker_raw_evidence,
)
from src.utils.process_identity_v2 import (
    ProcessIdentityV2Error,
    canonical_json_bytes,
    require_uuid4,
)
from src.utils.retained_output_directory import (
    HeldPublishedTerminalOutput,
    RetainedOutputTree,
    _hold_existing_output,
)
from src.utils.tastemolnet_t8_managed_v2 import (
    T8_VERIFICATION_SCHEMA,
    collect_t8_official_startup_evidence,
)
from src.utils.terminal_publisher_v2 import (
    HeldSealedArtifactV2,
    TerminalPublicationV2,
    open_sealed_worker_artifact,
    seal_worker_staging,
    verify_and_publish_sealed_attempt,
)


ADOPTION_SCHEMA = "tastemolnet_t8_deadline_managed_adoption_v1"
WORKER_SCHEMA = "tastemolnet_t8_deadline_adoption_worker_v1"
CONFIG_SCHEMA = "tastemolnet_t8_deadline_adoption_config_v1"
ADOPTION_MARKER = "[TASTE_T8_DEADLINE_MANAGED_ADOPTION_PASS]"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(canonical_json_bytes(dict(value)))


def _json_document_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_exact_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteGlobalGCESmokeError(f"T8 deadline adoption {label} is invalid JSON") from exc
    if type(value) is not dict or _json_document_bytes(value) != payload:
        raise TasteGlobalGCESmokeError(
            f"T8 deadline adoption {label} is not one canonical deadline document"
        )
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise TasteGlobalGCESmokeError(f"T8 deadline adoption {label} is not SHA-256")
    return value


def _absolute(path: str | Path, *, label: str, must_exist: bool = True) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or Path(os.path.abspath(candidate)) != candidate:
        raise TasteGlobalGCESmokeError(
            f"T8 deadline adoption {label} must be normalized and absolute"
        )
    if must_exist and candidate.resolve(strict=True) != candidate:
        raise TasteGlobalGCESmokeError(
            f"T8 deadline adoption {label} contains a symlink or alias"
        )
    return candidate


@dataclass(frozen=True, slots=True)
class DeadlineRecoveryInputs:
    config: Path
    deadline_output_root: Path
    deadline_state_root: Path
    deadline_attempt_id: str
    recovery_source_attempt_id: str
    t3_output: Path
    t4_output: Path
    gnn_checkpoint: Path
    train_csv: Path
    official_root: Path

    def validate(self) -> None:
        for name in (
            "config",
            "deadline_output_root",
            "deadline_state_root",
            "t3_output",
            "t4_output",
            "gnn_checkpoint",
            "train_csv",
            "official_root",
        ):
            _absolute(getattr(self, name), label=name)
        require_uuid4(self.deadline_attempt_id, label="deadline_attempt_id")
        require_uuid4(
            self.recovery_source_attempt_id,
            label="recovery_source_attempt_id",
        )
        if self.deadline_attempt_id == self.recovery_source_attempt_id:
            raise TasteGlobalGCESmokeError(
                "T8 deadline adoption requires a fresh recovery attempt"
            )


def _derive_deadline_preflight(inputs: DeadlineRecoveryInputs) -> dict[str, Any]:
    """Repeat the deadline runner's independent source preflight verbatim."""

    inputs.validate()
    evidence, _payloads, _train_payload = deadline.deadline_preflight(
        SimpleNamespace(
            config=inputs.config,
            set=["inference.fallback_to_heuristic=false"],
            attempt_id=inputs.deadline_attempt_id,
            zero_candidate_recovery=True,
            recovery_source_attempt_id=inputs.recovery_source_attempt_id,
            t3_output=inputs.t3_output,
            t4_output=inputs.t4_output,
            gnn_checkpoint=inputs.gnn_checkpoint,
            train_csv=inputs.train_csv,
            official_root=inputs.official_root,
        )
    )
    return evidence


def adoption_config_hash() -> str:
    return _mapping_sha256(
        {
            "schema_version": CONFIG_SCHEMA,
            "deadline_schema": deadline.SCHEMA,
            "recovery_schema": deadline.RECOVERY_SCHEMA,
            "managed_verification_schema": T8_VERIFICATION_SCHEMA,
            "epochs": ZERO_CANDIDATE_RECOVERY_EPOCHS,
            "source_label": 1,
            "target_branches": list(TARGET_BRANCHES),
            "science_copied": False,
            "test_loaded": False,
            "calibration_loaded": False,
        }
    )


def adoption_input_hashes(evidence: Mapping[str, Any]) -> dict[str, str]:
    return {
        "deadline.source_evidence": _mapping_sha256(evidence),
        "deadline.output_inventory": _require_sha256(
            evidence.get("deadline_output_inventory_sha256"),
            label="output inventory",
        ),
        "deadline.state_inventory": _require_sha256(
            evidence.get("deadline_state_inventory_sha256"),
            label="state inventory",
        ),
        "deadline.science": _require_sha256(
            evidence.get("science_semantic_sha256"),
            label="science semantic hash",
        ),
        "deadline.preflight": _require_sha256(
            evidence.get("preflight_sha256"),
            label="preflight hash",
        ),
    }


def _expected_terminal_documents(
    *,
    preflight: Mapping[str, Any],
    science: Mapping[str, Any],
    science_document_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    strict = science["strict_flip_validation"]
    manifest = {
        **dict(preflight),
        "status": "PASS",
        "science_sha256": science_document_sha256,
        "strict_flip_count": strict["strict_flip_count"],
        "destination_distribution": strict["destination_distribution"],
        "canonical_rule_merge_complete": True,
        "canonical_candidate_dedup_complete": True,
        "untargeted_strict_flip_complete": True,
    }
    gate = {
        "schema_version": deadline.SCHEMA,
        "status": "PASS",
        "marker": T8_PASS_MARKER,
        "attempt_id": preflight["attempt_id"],
        "checkpoint_id": preflight["checkpoint_id"],
        "target_branches": [0, 2],
        "strict_flip_count": strict["strict_flip_count"],
        "destination_distribution": strict["destination_distribution"],
        "rf_oracle_used": False,
        "test_loaded": False,
        "zero_candidate_recovery": preflight["zero_candidate_recovery"],
    }
    return manifest, gate


def _build_source_evidence(
    *,
    inputs: DeadlineRecoveryInputs,
    preflight: Mapping[str, Any],
    output: HeldPublishedTerminalOutput,
    state_tree: RetainedOutputTree,
) -> dict[str, Any]:
    if set(output.tree.leaf_paths) != {"science.json", "manifest.json", "gate.json"}:
        raise TasteGlobalGCESmokeError(
            "T8 deadline adoption terminal aggregate inventory changed"
        )
    science_bytes = output.read_bytes("science.json")
    manifest_bytes = output.read_bytes("manifest.json")
    gate_bytes = output.read_bytes("gate.json")
    science = _read_exact_json(science_bytes, label="science")
    manifest = _read_exact_json(manifest_bytes, label="manifest")
    gate = _read_exact_json(gate_bytes, label="gate")
    validate_science_summary(science)

    fixed_config = TasteGlobalGCESmokeConfig(
        epochs=ZERO_CANDIDATE_RECOVERY_EPOCHS
    ).to_dict()
    recovery = {
        "schema_version": deadline.RECOVERY_SCHEMA,
        "enabled": True,
        "source_attempt_id": inputs.recovery_source_attempt_id,
        "stop_reason": "prior_native_generation_had_zero_valid_connected_candidates",
        "epochs": ZERO_CANDIDATE_RECOVERY_EPOCHS,
    }
    if (
        preflight.get("schema_version") != deadline.SCHEMA
        or preflight.get("status") != "READY"
        or preflight.get("attempt_id") != inputs.deadline_attempt_id
        or preflight.get("zero_candidate_recovery") != recovery
        or preflight.get("science_config") != fixed_config
        or science.get("config") != fixed_config
        or science.get("oracle_checkpoint_hash") != preflight.get("checkpoint_id")
        or science.get("target_branches") != [0, 2]
        or science.get("rf_oracle_used") is not False
        or science.get("gnn_ablation_started") is not False
    ):
        raise TasteGlobalGCESmokeError(
            "T8 deadline adoption is not the fixed 25-epoch fresh recovery"
        )
    train_boundary = science.get("train_boundary")
    if (
        type(train_boundary) is not dict
        or train_boundary.get("external_validation_loaded") is not False
        or train_boundary.get("calibration_loaded") is not False
        or train_boundary.get("test_loaded") is not False
        or preflight.get("calibration_loaded") is not False
        or preflight.get("test_loaded") is not False
        or preflight.get("rf_oracle_used") is not False
        or preflight.get("gnn_ablation_started") is not False
    ):
        raise TasteGlobalGCESmokeError(
            "T8 deadline adoption split/oracle boundary changed"
        )

    science_document_sha256 = _sha256_bytes(science_bytes)
    expected_manifest, expected_gate = _expected_terminal_documents(
        preflight=preflight,
        science=science,
        science_document_sha256=science_document_sha256,
    )
    if manifest != expected_manifest or gate != expected_gate:
        raise TasteGlobalGCESmokeError(
            "T8 deadline adoption manifest/gate differs from verified science"
        )

    state_inventory = state_tree.revalidate()
    private = science["private_state"]
    if (
        private.get("inventory_sha256") != state_inventory.get("inventory_sha256")
        or private.get("file_count") != len(state_inventory.get("files", {}))
    ):
        raise TasteGlobalGCESmokeError(
            "T8 deadline adoption private state inventory changed"
        )
    official_startup = collect_t8_official_startup_evidence(
        state_tree=state_tree,
        science=science,
    )
    if official_startup.get("official_globalgce_commit") != OFFICIAL_GLOBALGCE_COMMIT:
        raise TasteGlobalGCESmokeError(
            "T8 deadline adoption official startup commit changed"
        )

    strict = science["strict_flip_validation"]
    evidence = {
        "schema_version": ADOPTION_SCHEMA,
        "status": "SOURCE_VERIFIED",
        "stage": STAGE,
        "dataset": DATASET,
        "method": METHOD,
        "task_id": MANAGED_TASK_ID,
        "deadline_output_root": str(inputs.deadline_output_root),
        "deadline_state_root": str(inputs.deadline_state_root),
        "deadline_attempt_id": inputs.deadline_attempt_id,
        "recovery_source_attempt_id": inputs.recovery_source_attempt_id,
        "zero_candidate_recovery": recovery,
        "science_config": fixed_config,
        "preflight_sha256": _mapping_sha256(preflight),
        "deadline_output_inventory_sha256": output.tree.inventory[
            "inventory_sha256"
        ],
        "deadline_state_inventory_sha256": state_inventory["inventory_sha256"],
        "science_document_sha256": science_document_sha256,
        "science_semantic_sha256": _mapping_sha256(science),
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "gate_sha256": _sha256_bytes(gate_bytes),
        "official_startup_sha256": _mapping_sha256(official_startup),
        "official_globalgce_commit": OFFICIAL_GLOBALGCE_COMMIT,
        "official_source_inventory_sha256": preflight[
            "official_runtime_source_inventory_sha256"
        ],
        "oracle_checkpoint_hash": science["oracle_checkpoint_hash"],
        "train_sha256": preflight["train_sha256"],
        "train_rows": preflight["train_rows"],
        "t3_verification_sha256": preflight["t3_verification_sha256"],
        "t4_verification_sha256": preflight["t4_verification_sha256"],
        "strict_flip_count": strict["strict_flip_count"],
        "destination_distribution": strict["destination_distribution"],
        "target_branches": [0, 2],
        "checkpoint_resume_verified": True,
        "official_lhs_to_rhs_verified": True,
        "isolated_imports_verified": True,
        "same_three_class_gine": True,
        "external_validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "rf_oracle_used": False,
        "gnn_ablation_started": False,
        "science_copied": False,
        "private_state_published": False,
        "data_redistributed": False,
    }
    for field in (
        "deadline_output_inventory_sha256",
        "deadline_state_inventory_sha256",
        "science_document_sha256",
        "science_semantic_sha256",
        "manifest_sha256",
        "gate_sha256",
        "official_startup_sha256",
        "official_source_inventory_sha256",
        "oracle_checkpoint_hash",
        "train_sha256",
        "t3_verification_sha256",
        "t4_verification_sha256",
    ):
        _require_sha256(evidence[field], label=field)
    return evidence


@dataclass(slots=True)
class HeldDeadlineRecovery:
    inputs: DeadlineRecoveryInputs
    output: HeldPublishedTerminalOutput
    state_output: Any
    state_tree: RetainedOutputTree
    preflight: Mapping[str, Any]
    evidence: Mapping[str, Any]
    _closed: bool = False

    @classmethod
    def open(cls, inputs: DeadlineRecoveryInputs) -> "HeldDeadlineRecovery":
        inputs.validate()
        preflight = _derive_deadline_preflight(inputs)
        output: HeldPublishedTerminalOutput | None = None
        state_output = None
        state_tree: RetainedOutputTree | None = None
        try:
            output = HeldPublishedTerminalOutput.open(
                inputs.deadline_output_root,
                marker_name="PASS",
                marker_payload=(T8_PASS_MARKER + "\n").encode("utf-8"),
            )
            state_output = _hold_existing_output(inputs.deadline_state_root)
            state_tree = RetainedOutputTree.capture(state_output.descriptor)
            evidence = _build_source_evidence(
                inputs=inputs,
                preflight=preflight,
                output=output,
                state_tree=state_tree,
            )
            result = cls(
                inputs=inputs,
                output=output,
                state_output=state_output,
                state_tree=state_tree,
                preflight=preflight,
                evidence=evidence,
            )
            result.revalidate()
            return result
        except BaseException:
            if state_tree is not None:
                state_tree.close()
            if state_output is not None:
                state_output.close()
            if output is not None:
                output.close()
            raise

    def revalidate(self) -> Mapping[str, Any]:
        if self._closed:
            raise TasteGlobalGCESmokeError("T8 deadline recovery hold is closed")
        self.output.revalidate()
        self.state_output.revalidate()
        self.state_tree.revalidate()
        repeated_preflight = _derive_deadline_preflight(self.inputs)
        if repeated_preflight != dict(self.preflight):
            raise TasteGlobalGCESmokeError(
                "T8 deadline adoption source preflight changed"
            )
        repeated = _build_source_evidence(
            inputs=self.inputs,
            preflight=repeated_preflight,
            output=self.output,
            state_tree=self.state_tree,
        )
        if repeated != dict(self.evidence):
            raise TasteGlobalGCESmokeError(
                "T8 deadline adoption source evidence changed"
            )
        return self.evidence

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.state_tree.close()
        self.state_output.close()
        self.output.close()

    def __enter__(self) -> "HeldDeadlineRecovery":
        self.revalidate()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def inspect_clean_execution(repo_root: Path) -> str:
    env = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }
    try:
        status = subprocess.run(
            ["/usr/bin/git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        ).stdout.strip()
        commit = subprocess.run(
            ["/usr/bin/git", "-C", str(repo_root), "rev-parse", "HEAD^{commit}"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TasteGlobalGCESmokeError(
            "T8 deadline adoption Git identity is unavailable"
        ) from exc
    if status or _GIT_OID.fullmatch(commit) is None:
        raise TasteGlobalGCESmokeError(
            "T8 deadline adoption requires one clean immutable checkout"
        )
    return commit


def _validate_run_identity(run_id: str, execution_commit: str) -> None:
    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise TasteGlobalGCESmokeError("T8 deadline adoption run ID is unsafe")
    if type(execution_commit) is not str or _GIT_OID.fullmatch(execution_commit) is None:
        raise TasteGlobalGCESmokeError(
            "T8 deadline adoption execution commit is invalid"
        )


def create_deadline_adoption_sealed(
    *,
    inputs: DeadlineRecoveryInputs,
    stage_root: str | Path,
    final_path: str | Path,
    managed_attempt_id: str,
    run_id: str,
    execution_commit: str,
) -> dict[str, Any]:
    """Worker process: write source hashes and SEALED, never PASS."""

    _validate_run_identity(run_id, execution_commit)
    selected_attempt = require_uuid4(
        managed_attempt_id, label="managed_attempt_id"
    )
    stage = _absolute(stage_root, label="stage_root")
    destination = _absolute(final_path, label="final_path", must_exist=False)
    if destination.exists() or destination.is_symlink():
        raise TasteGlobalGCESmokeError(
            "T8 deadline adoption final path must be fresh"
        )
    with HeldDeadlineRecovery.open(inputs) as source, create_managed_attempt(
        stage_root=stage,
        controller_id=run_id,
        task_id=MANAGED_TASK_ID,
        git_commit=execution_commit,
        config_hash=adoption_config_hash(),
        input_hashes=adoption_input_hashes(source.evidence),
        attempt_id=selected_attempt,
    ) as attempt, create_worker_staging(attempt) as staging:
        manifest = dict(attempt.manifest.revalidate())
        raw = write_worker_raw_evidence(
            staging,
            {
                "schema_version": WORKER_SCHEMA,
                "status": "RAW_EVIDENCE_ONLY",
                "stage": STAGE,
                "dataset": DATASET,
                "method": METHOD,
                "task_id": MANAGED_TASK_ID,
                "expected_final_path": str(destination),
                "run_id": run_id,
                "attempt_manifest": manifest,
                "source_evidence": dict(source.evidence),
                "source_evidence_sha256": _mapping_sha256(source.evidence),
                "worker_wrote_verification": False,
                "worker_wrote_gate": False,
                "worker_wrote_pass": False,
                "worker_reran_science": False,
                "science_copied": False,
                "independent_verification_required": True,
            },
        )
        raw.close()
        worker_exit = write_worker_exit(
            staging,
            {
                "exit_code": 0,
                "deadline_output_revalidated": True,
                "deadline_state_revalidated": True,
                "worker_reran_science": False,
                "worker_wrote_verifier_output": False,
            },
        )
        worker_exit.close()
        source.revalidate()
        sealed = seal_worker_staging(staging)
        return {
            "status": "SEALED_PENDING_INDEPENDENT_VERIFICATION",
            "attempt_id": sealed.attempt_id,
            "generation_token": sealed.generation_token,
            "staging_path": str(sealed.staging_path),
            "seal_path": str(sealed.seal_path),
            "seal_sha256": sealed.seal_sha256,
            "inventory_sha256": sealed.inventory_sha256,
            "expected_final_path": str(destination),
            "deadline_attempt_id": inputs.deadline_attempt_id,
        }


def _read_held_json(held: HeldSealedArtifactV2, relative: str) -> dict[str, Any]:
    matches = [
        item for item in held.files if item.evidence.relative_path == relative
    ]
    if len(matches) != 1:
        raise TasteGlobalGCESmokeError(
            f"T8 deadline adoption SEALED lacks {relative}"
        )
    item = matches[0]
    item.revalidate()
    payload = bytearray()
    offset = 0
    while offset < item.evidence.size:
        block = os.pread(
            item.descriptor,
            min(1024 * 1024, item.evidence.size - offset),
            offset,
        )
        if not block:
            raise TasteGlobalGCESmokeError(
                "T8 deadline adoption SEALED JSON ended early"
            )
        payload.extend(block)
        offset += len(block)
    if os.pread(item.descriptor, 1, item.evidence.size):
        raise TasteGlobalGCESmokeError(
            "T8 deadline adoption SEALED JSON grew"
        )
    try:
        value = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteGlobalGCESmokeError(
            "T8 deadline adoption SEALED JSON is malformed"
        ) from exc
    if type(value) is not dict or canonical_json_bytes(value) != bytes(payload):
        raise TasteGlobalGCESmokeError(
            "T8 deadline adoption SEALED JSON is not canonical"
        )
    item.revalidate()
    return value


def _validate_attempt_manifest(
    manifest: Mapping[str, Any],
    *,
    held: HeldSealedArtifactV2,
    run_id: str,
    execution_commit: str,
    source_evidence: Mapping[str, Any],
) -> None:
    try:
        manifest_generation = require_uuid4(
            manifest.get("generation_token"),
            label="managed attempt generation_token",
        )
    except (TypeError, ValueError, ProcessIdentityV2Error) as exc:
        raise TasteGlobalGCESmokeError(
            "T8 deadline adoption managed attempt generation is invalid"
        ) from exc
    expected_keys = {
        "schema_version",
        "status",
        "attempt_id",
        "controller_id",
        "task_id",
        "git_commit",
        "config_hash",
        "input_hashes",
        "created_at",
        "hostname",
        "boot_id",
        "attempt_path",
        "generation_token",
        "auto_terminate_uncontrolled_children",
    }
    if (
        type(manifest) is not dict
        or set(manifest) != expected_keys
        or manifest.get("schema_version") != ATTEMPT_MANIFEST_SCHEMA
        or manifest.get("status") != "ACTIVE"
        or manifest.get("attempt_id") != held.sealed.attempt_id
        or manifest.get("controller_id") != run_id
        or manifest.get("task_id") != MANAGED_TASK_ID
        or manifest.get("git_commit") != execution_commit
        or manifest.get("config_hash") != adoption_config_hash()
        or manifest.get("input_hashes") != adoption_input_hashes(source_evidence)
        or manifest.get("attempt_path") != str(held.staging_path.parent.parent)
        or manifest_generation != manifest.get("generation_token")
        or manifest.get("auto_terminate_uncontrolled_children") is not False
        or type(manifest.get("created_at")) is not str
        or not manifest["created_at"]
        or type(manifest.get("hostname")) is not str
        or not manifest["hostname"]
        or type(manifest.get("boot_id")) is not str
        or not manifest["boot_id"]
    ):
        raise TasteGlobalGCESmokeError(
            "T8 deadline adoption managed attempt manifest changed"
        )


def verify_and_publish_deadline_adoption(
    held: HeldSealedArtifactV2,
    *,
    inputs: DeadlineRecoveryInputs,
    final_path: str | Path,
    run_id: str,
    execution_commit: str,
    force_cross_filesystem: bool = False,
) -> tuple[TerminalPublicationV2, dict[str, Any]]:
    """Verifier process: repeat every source check and publish typed PASS."""

    _validate_run_identity(run_id, execution_commit)
    destination = _absolute(final_path, label="final_path", must_exist=False)
    with HeldDeadlineRecovery.open(inputs) as source:
        held.revalidate()
        raw = _read_held_json(held, "raw_evidence.json")
        worker_exit = _read_held_json(held, "worker_exit.json")
        evidence = raw.get("evidence")
        expected_raw_keys = {
            "schema_version",
            "status",
            "stage",
            "dataset",
            "method",
            "task_id",
            "expected_final_path",
            "run_id",
            "attempt_manifest",
            "source_evidence",
            "source_evidence_sha256",
            "worker_wrote_verification",
            "worker_wrote_gate",
            "worker_wrote_pass",
            "worker_reran_science",
            "science_copied",
            "independent_verification_required",
        }
        if (
            type(raw) is not dict
            or set(raw)
            != {
                "schema_version",
                "attempt_id",
                "generation_token",
                "recorded_at",
                "evidence",
            }
            or raw.get("schema_version") != WORKER_RAW_EVIDENCE_SCHEMA
            or raw.get("attempt_id") != held.sealed.attempt_id
            or raw.get("generation_token") != held.sealed.generation_token
            or type(raw.get("recorded_at")) is not str
            or not raw["recorded_at"]
            or type(evidence) is not dict
            or set(evidence) != expected_raw_keys
            or evidence.get("schema_version") != WORKER_SCHEMA
            or evidence.get("status") != "RAW_EVIDENCE_ONLY"
            or evidence.get("stage") != STAGE
            or evidence.get("dataset") != DATASET
            or evidence.get("method") != METHOD
            or evidence.get("task_id") != MANAGED_TASK_ID
            or evidence.get("expected_final_path") != str(destination)
            or evidence.get("run_id") != run_id
            or evidence.get("source_evidence") != dict(source.evidence)
            or evidence.get("source_evidence_sha256")
            != _mapping_sha256(source.evidence)
            or evidence.get("worker_wrote_verification") is not False
            or evidence.get("worker_wrote_gate") is not False
            or evidence.get("worker_wrote_pass") is not False
            or evidence.get("worker_reran_science") is not False
            or evidence.get("science_copied") is not False
            or evidence.get("independent_verification_required") is not True
        ):
            raise TasteGlobalGCESmokeError(
                "T8 deadline adoption worker evidence changed"
            )
        _validate_attempt_manifest(
            evidence["attempt_manifest"],
            held=held,
            run_id=run_id,
            execution_commit=execution_commit,
            source_evidence=source.evidence,
        )
        expected_exit = {
            "exit_code": 0,
            "deadline_output_revalidated": True,
            "deadline_state_revalidated": True,
            "worker_reran_science": False,
            "worker_wrote_verifier_output": False,
        }
        if (
            type(worker_exit) is not dict
            or set(worker_exit)
            != {
                "schema_version",
                "attempt_id",
                "generation_token",
                "recorded_at",
                "exit",
            }
            or worker_exit.get("schema_version") != WORKER_EXIT_SCHEMA
            or worker_exit.get("attempt_id") != held.sealed.attempt_id
            or worker_exit.get("generation_token") != held.sealed.generation_token
            or type(worker_exit.get("recorded_at")) is not str
            or not worker_exit["recorded_at"]
            or worker_exit.get("exit") != expected_exit
        ):
            raise TasteGlobalGCESmokeError(
                "T8 deadline adoption worker exit changed"
            )
        source.revalidate()
        verified = source.evidence
        verification = {
            "schema_version": T8_VERIFICATION_SCHEMA,
            "status": "PASS",
            "stage": STAGE,
            "dataset": DATASET,
            "method": METHOD,
            "task_id": MANAGED_TASK_ID,
            "attempt_id": held.sealed.attempt_id,
            "generation_token": held.sealed.generation_token,
            "input_authority_sha256": _mapping_sha256(verified),
            "science_sha256": verified["science_semantic_sha256"],
            "official_startup_sha256": verified["official_startup_sha256"],
            "official_globalgce_commit": verified[
                "official_globalgce_commit"
            ],
            "oracle_checkpoint_hash": verified["oracle_checkpoint_hash"],
            "target_branches": [0, 2],
            "strict_flip_count": verified["strict_flip_count"],
            "destination_distribution": verified[
                "destination_distribution"
            ],
            "same_three_class_gine": True,
            "checkpoint_resume_verified": True,
            "official_lhs_to_rhs_verified": True,
            "isolated_imports_verified": True,
            "rf_oracle_used": False,
            "data_redistributed": False,
            "worker_self_signed": False,
            "external_authority_revalidated": True,
            "deadline_adoption_schema": ADOPTION_SCHEMA,
            "deadline_attempt_id": inputs.deadline_attempt_id,
            "recovery_source_attempt_id": inputs.recovery_source_attempt_id,
            "zero_candidate_recovery_epochs": ZERO_CANDIDATE_RECOVERY_EPOCHS,
            "deadline_output_inventory_sha256": verified[
                "deadline_output_inventory_sha256"
            ],
            "deadline_state_inventory_sha256": verified[
                "deadline_state_inventory_sha256"
            ],
            "external_validation_loaded": False,
            "calibration_loaded": False,
            "test_loaded": False,
            "gnn_ablation_started": False,
            "science_copied": False,
        }
        publication = verify_and_publish_sealed_attempt(
            held,
            final_path=destination,
            verification=verification,
            force_cross_filesystem=force_cross_filesystem,
        )
        return publication, verification


__all__ = [
    "ADOPTION_MARKER",
    "ADOPTION_SCHEMA",
    "DeadlineRecoveryInputs",
    "HeldDeadlineRecovery",
    "adoption_config_hash",
    "adoption_input_hashes",
    "create_deadline_adoption_sealed",
    "inspect_clean_execution",
    "verify_and_publish_deadline_adoption",
]
