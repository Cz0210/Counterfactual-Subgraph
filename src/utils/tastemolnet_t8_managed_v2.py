"""Managed-execution-v2 worker/verifier boundary for TasteMolNet T8.

The scientific worker may call :func:`seal_t8_worker_evidence` only after its
private state writers are closed.  It writes managed raw evidence, worker-exit
evidence, and ``SEALED.json``; it cannot write a verifier gate or PASS.  A
separate process must retain an independent T2/T3/T4/GINE/train/official/policy
authority and call :func:`verify_and_publish_t8_sealed`.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from src.baselines.globalgce_bace_native_rules import (
    OFFICIAL_GLOBALGCE_COMMIT,
)
from src.baselines.globalgce_mutagenicity_adapter import (
    OFFICIAL_API_SIGNATURE_FILE,
    PYTHON_MODULE_PROVENANCE_FILE,
)
from src.baselines.tastemolnet_globalgce_smoke import (
    DATASET,
    MANAGED_TASK_ID,
    METHOD,
    STAGE,
    TARGET_BRANCHES,
    TasteGlobalGCESmokeConfig,
    TasteGlobalGCESmokeError,
    TasteGlobalGCETerminalAuthority,
    _canonical_sha256,
    _held_external_terminal_authority,
    _is_sha256,
    _read_json_bytes,
    _validate_official_startup_documents,
    _validate_terminal_input_authority,
    validate_science_summary,
)
from src.utils.managed_execution_v2 import (
    ATTEMPT_MANIFEST_SCHEMA,
    HeldWorkerStagingV2,
    WORKER_EXIT_SCHEMA,
    WORKER_RAW_EVIDENCE_SCHEMA,
    write_worker_exit,
    write_worker_raw_evidence,
)
from src.utils.process_identity_v2 import (
    ProcessIdentityV2Error,
    canonical_json_bytes,
    require_uuid4,
)
from src.utils.retained_output_directory import RetainedOutputTree
from src.utils.terminal_publisher_v2 import (
    HeldSealedArtifactV2,
    SealedWorkerArtifactV2,
    TerminalPublicationV2,
    seal_worker_staging,
    verify_and_publish_sealed_attempt,
)


T8_WORKER_RAW_SCHEMA = "tastemolnet_t8_managed_worker_raw_v2"
T8_OFFICIAL_STARTUP_SCHEMA = "tastemolnet_t8_official_startup_bundle_v1"
T8_VERIFICATION_SCHEMA = "tastemolnet_t8_independent_verification_v2"


def _identity_document_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _json_clone(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def _sha256_mapping(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def t8_managed_config_hash() -> str:
    return _sha256_mapping(TasteGlobalGCESmokeConfig().to_dict())


def t8_managed_input_hashes(
    input_authority: Mapping[str, Any],
) -> dict[str, str]:
    authority = _validate_terminal_input_authority(input_authority)
    return {
        f"authority.{name}": _sha256_mapping(authority[name])
        for name in sorted(authority)
    }


def collect_t8_official_startup_evidence(
    *,
    state_tree: RetainedOutputTree,
    science: Mapping[str, Any],
) -> dict[str, Any]:
    """Collect full API/import evidence from the retained private branch tree."""

    validate_science_summary(science)
    inventory = state_tree.revalidate()
    branches: dict[str, Any] = {}
    for target in TARGET_BRANCHES:
        key = str(target)
        prefix = f"target-{target}/"
        api_relative = prefix + OFFICIAL_API_SIGNATURE_FILE
        provenance_relative = prefix + PYTHON_MODULE_PROVENANCE_FILE
        for relative in (api_relative, provenance_relative):
            if relative not in inventory.get("files", {}):
                raise TasteGlobalGCESmokeError(
                    "T8 retained state lacks official startup evidence"
                )
        api = _read_json_bytes(
            state_tree.read_bytes(api_relative),
            label=f"target-{target} official API signature",
        )
        provenance = _read_json_bytes(
            state_tree.read_bytes(provenance_relative),
            label=f"target-{target} Python module provenance",
        )
        api_sha256 = inventory["files"][api_relative]["sha256"]
        provenance_sha256 = inventory["files"][provenance_relative]["sha256"]
        if (
            hashlib.sha256(_identity_document_bytes(api)).hexdigest()
            != api_sha256
            or hashlib.sha256(_identity_document_bytes(provenance)).hexdigest()
            != provenance_sha256
        ):
            raise TasteGlobalGCESmokeError(
                "T8 official startup document is not canonical"
            )
        _validate_official_startup_documents(
            api=api,
            provenance=provenance,
            api_sha256=api_sha256,
            provenance_sha256=provenance_sha256,
            training_summary=science["branches"][key],
        )
        branches[key] = {
            "official_api_signature": api,
            "official_api_signature_sha256": api_sha256,
            "python_module_provenance": provenance,
            "python_module_provenance_sha256": provenance_sha256,
        }
    if branches["0"] != branches["2"]:
        raise TasteGlobalGCESmokeError(
            "T8 target branches used different official/import startup identity"
        )
    return {
        "schema_version": T8_OFFICIAL_STARTUP_SCHEMA,
        "official_globalgce_commit": OFFICIAL_GLOBALGCE_COMMIT,
        "branches": branches,
    }


def _validate_official_startup_bundle(
    value: Mapping[str, Any],
    *,
    science: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        type(value) is not dict
        or set(value)
        != {"schema_version", "official_globalgce_commit", "branches"}
        or value.get("schema_version") != T8_OFFICIAL_STARTUP_SCHEMA
        or value.get("official_globalgce_commit")
        != OFFICIAL_GLOBALGCE_COMMIT
        or type(value.get("branches")) is not dict
        or set(value["branches"]) != {"0", "2"}
    ):
        raise TasteGlobalGCESmokeError(
            "T8 managed official startup bundle changed"
        )
    for target in TARGET_BRANCHES:
        key = str(target)
        branch = value["branches"][key]
        if (
            type(branch) is not dict
            or set(branch)
            != {
                "official_api_signature",
                "official_api_signature_sha256",
                "python_module_provenance",
                "python_module_provenance_sha256",
            }
            or type(branch.get("official_api_signature")) is not dict
            or type(branch.get("python_module_provenance")) is not dict
            or hashlib.sha256(
                _identity_document_bytes(branch["official_api_signature"])
            ).hexdigest()
            != branch.get("official_api_signature_sha256")
            or hashlib.sha256(
                _identity_document_bytes(branch["python_module_provenance"])
            ).hexdigest()
            != branch.get("python_module_provenance_sha256")
        ):
            raise TasteGlobalGCESmokeError(
                "T8 managed branch startup evidence changed"
            )
        _validate_official_startup_documents(
            api=branch["official_api_signature"],
            provenance=branch["python_module_provenance"],
            api_sha256=branch["official_api_signature_sha256"],
            provenance_sha256=branch["python_module_provenance_sha256"],
            training_summary=science["branches"][key],
        )
    if value["branches"]["0"] != value["branches"]["2"]:
        raise TasteGlobalGCESmokeError(
            "T8 managed target branches differ in official/import identity"
        )
    return _json_clone(value)


def _validate_attempt_manifest(
    manifest: Mapping[str, Any],
    *,
    attempt_id: str,
    attempt_path: Path,
    authority: Mapping[str, Any],
    input_hashes: Mapping[str, str],
) -> None:
    try:
        require_uuid4(attempt_id, label="T8 managed attempt ID")
        require_uuid4(
            manifest.get("generation_token"),
            label="T8 managed generation token",
        )
    except ProcessIdentityV2Error as exc:
        raise TasteGlobalGCESmokeError(
            "T8 managed attempt identity is not canonical UUIDv4"
        ) from exc
    if (
        type(manifest) is not dict
        or set(manifest)
        != {
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
        or manifest.get("schema_version") != ATTEMPT_MANIFEST_SCHEMA
        or manifest.get("status") != "ACTIVE"
        or manifest.get("attempt_id") != attempt_id
        or manifest.get("controller_id")
        != authority["managed_execution"]["run_id"]
        or manifest.get("task_id") != MANAGED_TASK_ID
        or manifest.get("git_commit") != authority["execution"]["commit"]
        or manifest.get("config_hash") != t8_managed_config_hash()
        or manifest.get("input_hashes") != dict(input_hashes)
        or type(manifest.get("created_at")) is not str
        or not manifest["created_at"]
        or type(manifest.get("hostname")) is not str
        or not manifest["hostname"]
        or type(manifest.get("boot_id")) is not str
        or not manifest["boot_id"]
        or manifest.get("attempt_path") != str(attempt_path)
        or type(manifest.get("generation_token")) is not str
        or not manifest["generation_token"]
        or manifest.get("auto_terminate_uncontrolled_children") is not False
    ):
        raise TasteGlobalGCESmokeError(
            "T8 managed attempt manifest does not match external authority"
        )


def seal_t8_worker_evidence(
    staging: HeldWorkerStagingV2,
    *,
    science: Mapping[str, Any],
    state_tree: RetainedOutputTree,
    input_authority: Mapping[str, Any],
    expected_final_path: str | Path,
) -> SealedWorkerArtifactV2:
    """Worker-only raw/exit/SEALED close; never writes verification or PASS."""

    staging.revalidate()
    validate_science_summary(science)
    authority = _validate_terminal_input_authority(input_authority)
    if (
        authority["official_globalgce"]["commit"]
        != OFFICIAL_GLOBALGCE_COMMIT
        or science.get("oracle_checkpoint_hash")
        != authority["frozen_gine"]["checkpoint_id"]
        or science["train_boundary"]["train_row_count"]
        != authority["train_split"]["row_count"]
    ):
        raise TasteGlobalGCESmokeError(
            "T8 managed raw science differs from its input authority"
        )
    final_path = Path(expected_final_path)
    if not final_path.is_absolute():
        final_path = Path(os.path.abspath(final_path))
    if final_path.exists() or final_path.is_symlink():
        raise TasteGlobalGCESmokeError(
            "T8 managed final path must remain fresh before verification"
        )
    startup = collect_t8_official_startup_evidence(
        state_tree=state_tree,
        science=science,
    )
    input_hashes = t8_managed_input_hashes(authority)
    manifest = dict(staging.attempt.revalidate())
    _validate_attempt_manifest(
        manifest,
        attempt_id=staging.attempt.attempt_id,
        attempt_path=staging.attempt.attempt_path,
        authority=authority,
        input_hashes=input_hashes,
    )
    evidence = {
        "schema_version": T8_WORKER_RAW_SCHEMA,
        "status": "RAW_EVIDENCE_ONLY",
        "stage": STAGE,
        "dataset": DATASET,
        "method": METHOD,
        "task_id": MANAGED_TASK_ID,
        "expected_final_path": str(final_path),
        "attempt_manifest": manifest,
        "input_authority": authority,
        "input_authority_sha256": _canonical_sha256(authority),
        "science": _json_clone(science),
        "science_sha256": _canonical_sha256(science),
        "official_startup": startup,
        "official_startup_sha256": _canonical_sha256(startup),
        "worker_wrote_verification": False,
        "worker_wrote_gate": False,
        "worker_wrote_pass": False,
        "independent_verification_required": True,
        "private_state_published": False,
        "data_redistributed": False,
    }
    raw = write_worker_raw_evidence(staging, evidence)
    raw.close()
    state_inventory = state_tree.revalidate()
    worker_exit = write_worker_exit(
        staging,
        {
            "exit_code": 0,
            "worker_closed_science_state_writers": True,
            "private_state_inventory_sha256": _canonical_sha256(
                state_inventory
            ),
            "private_state_not_published": True,
            "worker_wrote_verifier_output": False,
        },
    )
    worker_exit.close()
    return seal_worker_staging(staging)


def _read_held_json(
    held: HeldSealedArtifactV2,
    relative_path: str,
) -> dict[str, Any]:
    matches = [
        item for item in held.files if item.evidence.relative_path == relative_path
    ]
    if len(matches) != 1:
        raise TasteGlobalGCESmokeError(
            f"T8 SEALED evidence lacks exact {relative_path}"
        )
    item = matches[0]
    item.revalidate()
    size = item.evidence.size
    data = bytearray()
    offset = 0
    while offset < size:
        block = os.pread(item.descriptor, min(1024 * 1024, size - offset), offset)
        if not block:
            raise TasteGlobalGCESmokeError("T8 SEALED JSON ended early")
        data.extend(block)
        offset += len(block)
    if os.pread(item.descriptor, 1, size):
        raise TasteGlobalGCESmokeError("T8 SEALED JSON grew")
    try:
        value = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteGlobalGCESmokeError("T8 SEALED JSON is malformed") from exc
    if type(value) is not dict or canonical_json_bytes(value) != bytes(data):
        raise TasteGlobalGCESmokeError("T8 SEALED JSON is not canonical")
    item.revalidate()
    return value


def _validate_sealed_t8_evidence(
    held: HeldSealedArtifactV2,
    *,
    authority: Mapping[str, Any],
    final_path: Path,
) -> dict[str, Any]:
    held.revalidate()
    raw = _read_held_json(held, "raw_evidence.json")
    worker_exit = _read_held_json(held, "worker_exit.json")
    if (
        set(raw)
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
        or type(raw.get("evidence")) is not dict
    ):
        raise TasteGlobalGCESmokeError("T8 managed raw wrapper changed")
    evidence = raw["evidence"]
    evidence_keys = {
        "schema_version",
        "status",
        "stage",
        "dataset",
        "method",
        "task_id",
        "expected_final_path",
        "attempt_manifest",
        "input_authority",
        "input_authority_sha256",
        "science",
        "science_sha256",
        "official_startup",
        "official_startup_sha256",
        "worker_wrote_verification",
        "worker_wrote_gate",
        "worker_wrote_pass",
        "independent_verification_required",
        "private_state_published",
        "data_redistributed",
    }
    if (
        set(evidence) != evidence_keys
        or evidence.get("schema_version") != T8_WORKER_RAW_SCHEMA
        or evidence.get("status") != "RAW_EVIDENCE_ONLY"
        or evidence.get("stage") != STAGE
        or evidence.get("dataset") != DATASET
        or evidence.get("method") != METHOD
        or evidence.get("task_id") != MANAGED_TASK_ID
        or evidence.get("expected_final_path") != str(final_path)
        or evidence.get("input_authority") != dict(authority)
        or evidence.get("input_authority_sha256")
        != _canonical_sha256(authority)
        or type(evidence.get("science")) is not dict
        or evidence.get("science_sha256")
        != _canonical_sha256(evidence["science"])
        or type(evidence.get("official_startup")) is not dict
        or evidence.get("official_startup_sha256")
        != _canonical_sha256(evidence["official_startup"])
        or evidence.get("worker_wrote_verification") is not False
        or evidence.get("worker_wrote_gate") is not False
        or evidence.get("worker_wrote_pass") is not False
        or evidence.get("independent_verification_required") is not True
        or evidence.get("private_state_published") is not False
        or evidence.get("data_redistributed") is not False
    ):
        raise TasteGlobalGCESmokeError("T8 managed raw evidence changed")
    science = evidence["science"]
    validate_science_summary(science)
    if (
        authority["official_globalgce"]["commit"]
        != OFFICIAL_GLOBALGCE_COMMIT
        or science["oracle_checkpoint_hash"]
        != authority["frozen_gine"]["checkpoint_id"]
        or science["train_boundary"]["train_row_count"]
        != authority["train_split"]["row_count"]
    ):
        raise TasteGlobalGCESmokeError(
            "T8 independently verified science/input binding changed"
        )
    _validate_official_startup_bundle(
        evidence["official_startup"], science=science
    )
    attempt_path = held.staging_path.parent.parent
    input_hashes = t8_managed_input_hashes(authority)
    _validate_attempt_manifest(
        evidence["attempt_manifest"],
        attempt_id=held.sealed.attempt_id,
        attempt_path=attempt_path,
        authority=authority,
        input_hashes=input_hashes,
    )
    if (
        set(worker_exit)
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
        or type(worker_exit.get("exit")) is not dict
        or set(worker_exit["exit"])
        != {
            "exit_code",
            "worker_closed_science_state_writers",
            "private_state_inventory_sha256",
            "private_state_not_published",
            "worker_wrote_verifier_output",
        }
        or worker_exit["exit"].get("exit_code") != 0
        or worker_exit["exit"].get("worker_closed_science_state_writers")
        is not True
        or len(str(worker_exit["exit"].get("private_state_inventory_sha256")))
        != 64
        or not _is_sha256(
            worker_exit["exit"].get("private_state_inventory_sha256")
        )
        or worker_exit["exit"].get("private_state_not_published") is not True
        or worker_exit["exit"].get("worker_wrote_verifier_output") is not False
    ):
        raise TasteGlobalGCESmokeError("T8 managed worker-exit evidence changed")
    return evidence


def verify_and_publish_t8_sealed(
    held: HeldSealedArtifactV2,
    *,
    final_path: str | Path,
    authority: TasteGlobalGCETerminalAuthority,
    force_cross_filesystem: bool = False,
) -> TerminalPublicationV2:
    """Independently verify T8 raw science and atomically publish managed PASS."""

    expected_authority = _held_external_terminal_authority(authority)
    destination = Path(final_path)
    if not destination.is_absolute():
        destination = Path(os.path.abspath(destination))
    evidence = _validate_sealed_t8_evidence(
        held,
        authority=expected_authority,
        final_path=destination,
    )
    startup_revalidate = getattr(
        authority,
        "revalidate_t8_official_startup_authority",
        None,
    )
    if not callable(startup_revalidate):
        raise TasteGlobalGCESmokeError(
            "T8 independent official startup expectation authority is absent"
        )
    expected_startup_raw = startup_revalidate()
    if type(expected_startup_raw) is not dict:
        raise TasteGlobalGCESmokeError(
            "T8 independent official startup expectation changed"
        )
    expected_startup = _validate_official_startup_bundle(
        expected_startup_raw,
        science=evidence["science"],
    )
    if expected_startup != evidence["official_startup"]:
        raise TasteGlobalGCESmokeError(
            "T8 worker official startup evidence differs from held expectation"
        )
    repeated = _held_external_terminal_authority(authority)
    if repeated != expected_authority:
        raise TasteGlobalGCESmokeError(
            "T8 independent verifier authority changed before publication"
        )
    repeated_startup_raw = startup_revalidate()
    if (
        type(repeated_startup_raw) is not dict
        or _validate_official_startup_bundle(
            repeated_startup_raw,
            science=evidence["science"],
        )
        != expected_startup
    ):
        raise TasteGlobalGCESmokeError(
            "T8 independent official startup expectation changed before publication"
        )
    science = evidence["science"]
    startup = evidence["official_startup"]
    verification = {
        "schema_version": T8_VERIFICATION_SCHEMA,
        "status": "PASS",
        "stage": STAGE,
        "dataset": DATASET,
        "method": METHOD,
        "task_id": MANAGED_TASK_ID,
        "attempt_id": held.sealed.attempt_id,
        "generation_token": held.sealed.generation_token,
        "input_authority_sha256": evidence["input_authority_sha256"],
        "science_sha256": evidence["science_sha256"],
        "official_startup_sha256": evidence["official_startup_sha256"],
        "official_globalgce_commit": startup["official_globalgce_commit"],
        "oracle_checkpoint_hash": science["oracle_checkpoint_hash"],
        "target_branches": [0, 2],
        "strict_flip_count": science["strict_flip_validation"][
            "strict_flip_count"
        ],
        "destination_distribution": science["strict_flip_validation"][
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
    }
    return verify_and_publish_sealed_attempt(
        held,
        final_path=destination,
        verification=verification,
        force_cross_filesystem=force_cross_filesystem,
    )


__all__ = [
    "T8_OFFICIAL_STARTUP_SCHEMA",
    "T8_VERIFICATION_SCHEMA",
    "T8_WORKER_RAW_SCHEMA",
    "collect_t8_official_startup_evidence",
    "seal_t8_worker_evidence",
    "t8_managed_config_hash",
    "t8_managed_input_hashes",
    "verify_and_publish_t8_sealed",
]
