"""Lossless managed-v2 adoption of one verified fixed-budget NeuroSED root.

The fixed-budget trainer already has an independent scientific verifier.  This
module does not retrain, select a checkpoint, or reinterpret that result.  It
closes the complete PASS root, copies it byte-for-byte into a managed-v2
``artifacts`` directory, and lets a separate managed verifier publish the
result.  The T7 consumer document uses fixed-budget sampler/label hashes under
their real names instead of pretending they are the legacy split graph hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping

from src.eval.tastemolnet_neurosed_official_fixed_budget import (
    DISTANCE_DIRECTION_SCHEMA,
    OFFICIAL_FIXED_MODEL_CARD_SCHEMA,
    READINESS_SCHEMA,
    validate_official_fixed_budget_model_card,
)
from src.train.tastemolnet_neurosed_fixed_budget import NEUROSED_PASS_MARKER
from src.utils.managed_execution_v2 import (
    WORKER_EXIT_SCHEMA,
    WORKER_RAW_EVIDENCE_SCHEMA,
)
from src.utils.process_identity_v2 import canonical_json_bytes


ADOPTION_TASK_ID = "TASTE_GCF_NEUROSED_FIXED_BUDGET_ADOPTION"
ADOPTION_VERIFICATION_SCHEMA = (
    "tastemolnet_neurosed_fixed_budget_managed_adoption_v1"
)
T7_FIXED_BUDGET_CONSUMER_SCHEMA = (
    "tastemolnet_gcf_neurosed_t7_fixed_budget_consumer_v2"
)
FIXED_BUDGET_VERIFICATION_SCHEMA = (
    "tastemolnet_neurosed_fixed_budget_verification_v1"
)
FIXED_BUDGET_FEATURE_SCHEMA = "tastemolnet_gcf_neurosed_feature_schema_v1"

_REQUIRED_FILES = {
    "PASS",
    "best.pt",
    "checkpoint_manifest.json",
    "config.yaml",
    "distance_direction_trace.json",
    "environment.json",
    "feature_schema.json",
    "ged_label_manifest.json",
    "git_state.json",
    "health_gate.json",
    "model.pt",
    "model_card.json",
    "pair_manifest.json",
    "readiness.json",
    "selector_trace.json",
    "sha256sums.txt",
    "split_manifest.json",
    "training_metrics.json",
    "validation_metrics.json",
    "verification.json",
    "verification_sha256s.txt",
}
_VERIFIER_ONLY_FILES = {
    "PASS",
    "verification.json",
    "verification_sha256s.txt",
}


class FixedBudgetNeuroSEDAdoptionError(RuntimeError):
    """The fixed-budget PASS or its managed-v2 adoption is not exact."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FixedBudgetNeuroSEDAdoptionError(f"{label} must be lowercase SHA-256")
    return value


def _json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise FixedBudgetNeuroSEDAdoptionError(f"{label} is not JSON") from exc
    if type(payload) is not dict:
        raise FixedBudgetNeuroSEDAdoptionError(f"{label} must be a JSON object")
    return payload


def _stable_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    )


def _physical_root(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise FixedBudgetNeuroSEDAdoptionError(f"{label} must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FixedBudgetNeuroSEDAdoptionError(f"{label} is unavailable") from exc
    if resolved != path or not path.is_dir():
        raise FixedBudgetNeuroSEDAdoptionError(
            f"{label} must be one physical directory"
        )
    return path


def _parse_sha256s(data: bytes) -> dict[str, str]:
    rows: dict[str, str] = {}
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise FixedBudgetNeuroSEDAdoptionError("sha256sums.txt is not UTF-8") from exc
    for line in lines:
        digest, separator, relative = line.partition("  ")
        candidate = PurePosixPath(relative)
        if (
            not separator
            or relative in rows
            or candidate.is_absolute()
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise FixedBudgetNeuroSEDAdoptionError("sha256sums.txt is malformed")
        rows[relative] = _require_sha256(digest, label=f"checksum {relative}")
    if not rows:
        raise FixedBudgetNeuroSEDAdoptionError("sha256sums.txt is empty")
    return rows


def _inventory(
    root: Path, *, ignore_managed_generation_token: bool = False
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    directories: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if ignore_managed_generation_token and relative == ".generation_token.json":
            continue
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode):
            raise FixedBudgetNeuroSEDAdoptionError(
                f"fixed-budget root contains symlink: {relative}"
            )
        if stat.S_ISDIR(info.st_mode):
            directories.append(relative)
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise FixedBudgetNeuroSEDAdoptionError(
                f"fixed-budget root contains unsafe file: {relative}"
            )
        files[relative] = {
            "sha256": _sha256_file(path),
            "size": int(info.st_size),
        }
    payload = {
        "schema_version": "tastemolnet_neurosed_fixed_budget_inventory_v1",
        "files": files,
        "directories": directories,
    }
    payload["inventory_sha256"] = _stable_sha256(payload)
    return payload


def _expect_fields(payload: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    for name, value in expected.items():
        if type(payload.get(name)) is not type(value) or payload.get(name) != value:
            raise FixedBudgetNeuroSEDAdoptionError(
                f"{label} field changed: {name}"
            )


def inspect_fixed_budget_neurosed_pass(
    root: str | Path,
    *,
    vendored_gcf_root: str | Path,
    allow_managed_generation_token: bool = False,
) -> dict[str, Any]:
    """Close and validate one already independently verified PASS root."""

    source = _physical_root(root, label="fixed-budget NeuroSED root")
    top = {path.name for path in source.iterdir()}
    expected_top = _REQUIRED_FILES | {"checkpoints"}
    if allow_managed_generation_token:
        expected_top = expected_top | {".generation_token.json"}
    if top != expected_top:
        raise FixedBudgetNeuroSEDAdoptionError(
            "fixed-budget NeuroSED top-level inventory changed"
        )
    inventory = _inventory(
        source,
        ignore_managed_generation_token=allow_managed_generation_token,
    )
    files = inventory["files"]
    checksum_rows = _parse_sha256s((source / "sha256sums.txt").read_bytes())
    expected_scientific = {
        relative: evidence["sha256"]
        for relative, evidence in files.items()
        if relative != "sha256sums.txt"
        and PurePosixPath(relative).name not in _VERIFIER_ONLY_FILES
    }
    if checksum_rows != expected_scientific:
        raise FixedBudgetNeuroSEDAdoptionError(
            "fixed-budget sha256sums inventory changed"
        )
    if (source / "PASS").read_bytes() != (NEUROSED_PASS_MARKER + "\n").encode():
        raise FixedBudgetNeuroSEDAdoptionError("fixed-budget PASS marker changed")
    verification = _json(source / "verification.json", label="verification")
    _expect_fields(
        verification,
        {
            "schema_version": FIXED_BUDGET_VERIFICATION_SCHEMA,
            "status": "PASS",
            "marker": NEUROSED_PASS_MARKER,
            "independent_process_reopened_worker_root": True,
            "worker_wrote_scientific_pass": False,
            "checkpoint_reload_passed": True,
            "official_selector_trace_replayed": True,
            "batch_single_agreement_reproduced": True,
            "gcf_runner_load_reproduced": True,
            "generated_query_to_original_target_reproduced": True,
            "validation_metrics_reproduced": True,
            "calibration_loaded": False,
            "test_loaded": False,
        },
        label="fixed-budget verification",
    )
    claimed_verification = _require_sha256(
        verification.get("verification_sha256"), label="verification self-hash"
    )
    verification_body = dict(verification)
    verification_body.pop("verification_sha256")
    if claimed_verification != _stable_sha256(verification_body):
        raise FixedBudgetNeuroSEDAdoptionError("verification self-hash changed")
    verification_file_sha = files["verification.json"]["sha256"]
    expected_verification_row = f"{verification_file_sha}  verification.json\n"
    if (source / "verification_sha256s.txt").read_text() != expected_verification_row:
        raise FixedBudgetNeuroSEDAdoptionError(
            "verification checksum receipt changed"
        )

    model_card = _json(source / "model_card.json", label="model card")
    if model_card.get("schema_version") != OFFICIAL_FIXED_MODEL_CARD_SCHEMA:
        raise FixedBudgetNeuroSEDAdoptionError("fixed-budget model-card schema changed")
    validate_official_fixed_budget_model_card(
        model_card, vendored_gcf_root=vendored_gcf_root
    )
    feature = _json(source / "feature_schema.json", label="feature schema")
    pair = _json(source / "pair_manifest.json", label="pair manifest")
    split = _json(source / "split_manifest.json", label="split manifest")
    checkpoint = _json(
        source / "checkpoint_manifest.json", label="checkpoint manifest"
    )
    health = _json(source / "health_gate.json", label="health gate")
    readiness = _json(source / "readiness.json", label="readiness")
    selector = _json(source / "selector_trace.json", label="selector trace")
    direction = _json(
        source / "distance_direction_trace.json", label="direction trace"
    )
    selected_sha = files["best.pt"]["sha256"]
    if files["model.pt"]["sha256"] != selected_sha:
        raise FixedBudgetNeuroSEDAdoptionError("best.pt/model.pt bytes differ")
    _expect_fields(
        feature,
        {
            "schema_version": FIXED_BUDGET_FEATURE_SCHEMA,
            "dataset": "tastemolnet",
            "train_derived_only": True,
            "validation_unseen_atomic_numbers": [],
        },
        label="feature schema",
    )
    atoms = feature.get("feature_atomic_numbers")
    if (
        type(atoms) is not list
        or atoms != sorted(set(atoms))
        or type(feature.get("input_dim")) is not int
        or feature["input_dim"] != len(atoms)
    ):
        raise FixedBudgetNeuroSEDAdoptionError("feature dimensions changed")
    _expect_fields(
        pair,
        {
            "schema_version": "tastemolnet_neurosed_fixed_budget_pair_bundle_v1",
            "train_pair_count": 5000,
            "validation_pair_count": 1000,
            "independent_pairs": True,
            "query_graph_id_differs_from_target_graph_id": True,
            "class_labels_used_as_supervision": False,
            "calibration_loaded": False,
            "test_loaded": False,
        },
        label="pair manifest",
    )
    _expect_fields(
        split,
        {
            "schema_version": "tastemolnet_neurosed_fixed_budget_split_manifest_v1",
            "opened_payload_splits": ["train", "validation"],
            "train_pair_roles_subset_of_train": True,
            "validation_pair_roles_subset_of_validation": True,
            "train_validation_graph_id_intersection_empty": True,
            "calibration_loaded": False,
            "test_loaded": False,
        },
        label="split manifest",
    )
    _expect_fields(
        health,
        {
            "schema_version": "tastemolnet_neurosed_fixed_budget_worker_health_v1",
            "status": "READY_FOR_INDEPENDENT_VERIFICATION",
            "worker_wrote_scientific_pass": False,
            "finite_loss": True,
            "finite_validation_metric": True,
            "no_split_leakage": True,
            "official_selector_trace": True,
            "generated_query_to_original_target_assertion": True,
            "gcf_runner_load_passed": True,
        },
        label="health gate",
    )
    _expect_fields(
        readiness,
        {
            "schema_version": READINESS_SCHEMA,
            "status": "READY_FOR_MANAGED_INDEPENDENT_VERIFICATION",
            "marker": None,
            "scientific_pass_claimed": False,
            "model_card_contract_valid": True,
            "official_selector_contract_valid": True,
            "generated_query_original_target_direction_valid": True,
        },
        label="readiness",
    )
    if (
        checkpoint.get("selected_checkpoint_sha256") != selected_sha
        or checkpoint.get("best_pt_sha256") != selected_sha
        or checkpoint.get("model_pt_sha256") != selected_sha
        or checkpoint.get("best_and_model_bytes_identical") is not True
        or model_card.get("selected_checkpoint_sha256") != selected_sha
        or verification.get("checkpoint_sha256") != selected_sha
        or selector.get("selected_checkpoint_sha256") != selected_sha
    ):
        raise FixedBudgetNeuroSEDAdoptionError("selected checkpoint binding changed")
    feature_sha = files["feature_schema.json"]["sha256"]
    if model_card.get("feature_schema_sha256") != feature_sha:
        raise FixedBudgetNeuroSEDAdoptionError("feature-schema binding changed")
    if (
        selector.get("trace_sha256") != model_card.get("selector_trace_sha256")
        or direction.get("schema_version") != DISTANCE_DIRECTION_SCHEMA
        or direction.get("direction") != "generated_query_to_original_target"
        or direction.get("reverse_direction_used") is not False
        or direction.get("trace_sha256")
        != model_card.get("distance_direction_trace_sha256")
        or verification.get("selector_trace_sha256")
        != model_card.get("selector_trace_sha256")
        or verification.get("distance_direction_trace_sha256")
        != model_card.get("distance_direction_trace_sha256")
    ):
        raise FixedBudgetNeuroSEDAdoptionError("selector/direction binding changed")
    bindings = readiness.get("evidence_bindings")
    if type(bindings) is not dict:
        raise FixedBudgetNeuroSEDAdoptionError("readiness bindings are absent")
    for name in (
        "train_pair_sampler_manifest_sha256",
        "validation_pair_sampler_manifest_sha256",
        "train_pair_labels_manifest_sha256",
        "validation_pair_labels_manifest_sha256",
    ):
        digest = _require_sha256(model_card.get(name), label=name)
        if pair.get(name) != digest or bindings.get(name) != digest:
            raise FixedBudgetNeuroSEDAdoptionError(f"pair binding changed: {name}")
    split_isolation_sha = _require_sha256(
        split.get("source_split_isolation_sha256"), label="split isolation"
    )
    consumer = {
        "schema_version": T7_FIXED_BUDGET_CONSUMER_SCHEMA,
        "dataset": "tastemolnet",
        "role": "GCF_AUXILIARY_DISTANCE_MODEL",
        "classifier": False,
        "source_label_independent": True,
        "train_only_fit": True,
        "validation_only_selection": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "health_gate_status": "PASS",
        "checkpoint_relative_path": "artifacts/best.pt",
        "checkpoint_sha256": selected_sha,
        "feature_schema_relative_path": "artifacts/feature_schema.json",
        "feature_schema_sha256": feature_sha,
        "feature_atomic_numbers": list(atoms),
        "feature_input_dim": feature["input_dim"],
        "sha256s_relative_path": "artifacts/sha256sums.txt",
        "sha256s_sha256": files["sha256sums.txt"]["sha256"],
        "train_pair_budget": 5000,
        "validation_pair_budget": 1000,
        "train_pair_sampler_manifest_sha256": model_card[
            "train_pair_sampler_manifest_sha256"
        ],
        "validation_pair_sampler_manifest_sha256": model_card[
            "validation_pair_sampler_manifest_sha256"
        ],
        "train_pair_labels_manifest_sha256": model_card[
            "train_pair_labels_manifest_sha256"
        ],
        "validation_pair_labels_manifest_sha256": model_card[
            "validation_pair_labels_manifest_sha256"
        ],
        "split_isolation_sha256": split_isolation_sha,
        "selector_trace_sha256": model_card["selector_trace_sha256"],
        "distance_direction_trace_sha256": model_card[
            "distance_direction_trace_sha256"
        ],
        "fixed_budget_pass_sha256": files["PASS"]["sha256"],
        "fixed_budget_verification_sha256": verification_file_sha,
        "fixed_budget_source_inventory_sha256": inventory["inventory_sha256"],
    }
    return {
        "root": str(source),
        "inventory": inventory,
        "inventory_sha256": inventory["inventory_sha256"],
        "pass_sha256": files["PASS"]["sha256"],
        "verification_sha256": verification_file_sha,
        "checkpoint_sha256": selected_sha,
        "feature_schema_sha256": feature_sha,
        "sha256s_sha256": files["sha256sums.txt"]["sha256"],
        "t7_consumer": consumer,
    }


def copy_fixed_budget_neurosed_pass(
    *,
    source_root: str | Path,
    artifact_root: str | Path,
    expected_source_inventory_sha256: str,
    vendored_gcf_root: str | Path,
) -> dict[str, Any]:
    """Copy the complete source root into one empty managed artifact root."""

    expected = _require_sha256(
        expected_source_inventory_sha256, label="expected source inventory"
    )
    source = inspect_fixed_budget_neurosed_pass(
        source_root, vendored_gcf_root=vendored_gcf_root
    )
    if source["inventory_sha256"] != expected:
        raise FixedBudgetNeuroSEDAdoptionError("fixed-budget source pin changed")
    destination = _physical_root(artifact_root, label="managed artifact root")
    destination_names = {path.name for path in destination.iterdir()}
    if destination_names not in (set(), {".generation_token.json"}):
        raise FixedBudgetNeuroSEDAdoptionError("managed artifact root is not empty")
    has_managed_generation_token = bool(destination_names)
    source_path = Path(source["root"])
    for relative in source["inventory"]["directories"]:
        (destination / relative).mkdir(mode=0o700)
    for relative in source["inventory"]["files"]:
        source_file = source_path / relative
        destination_file = destination / relative
        descriptor = os.open(
            destination_file,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            with source_file.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    view = memoryview(block)
                    while view:
                        written = os.write(descriptor, view)
                        if written <= 0:
                            raise FixedBudgetNeuroSEDAdoptionError(
                                f"managed copy was short: {relative}"
                            )
                        view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for path in sorted(
        (item for item in destination.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    destination_descriptor = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(destination_descriptor)
    finally:
        os.close(destination_descriptor)
    source_after = inspect_fixed_budget_neurosed_pass(
        source_root, vendored_gcf_root=vendored_gcf_root
    )
    copied = inspect_fixed_budget_neurosed_pass(
        destination,
        vendored_gcf_root=vendored_gcf_root,
        allow_managed_generation_token=has_managed_generation_token,
    )
    if (
        source_after["inventory_sha256"] != expected
        or copied["inventory_sha256"] != expected
        or copied["t7_consumer"] != source_after["t7_consumer"]
    ):
        raise FixedBudgetNeuroSEDAdoptionError("managed copy differs from source")
    return copied


def _held_json(held: Any, relative_path: str) -> dict[str, Any]:
    matches = [
        item for item in held.files if item.evidence.relative_path == relative_path
    ]
    if len(matches) != 1:
        raise FixedBudgetNeuroSEDAdoptionError(
            f"managed evidence is absent: {relative_path}"
        )
    item = matches[0]
    remaining = int(item.evidence.size)
    offset = 0
    chunks: list[bytes] = []
    while remaining:
        block = os.pread(item.descriptor, min(64 * 1024, remaining), offset)
        if not block:
            raise FixedBudgetNeuroSEDAdoptionError(
                f"managed evidence is short: {relative_path}"
            )
        chunks.append(block)
        remaining -= len(block)
        offset += len(block)
    payload = json.loads(b"".join(chunks))
    if type(payload) is not dict:
        raise FixedBudgetNeuroSEDAdoptionError(
            f"managed evidence is not an object: {relative_path}"
        )
    return payload


def verify_fixed_budget_managed_adoption(
    held: Any,
    *,
    source_root: str | Path,
    expected_source_inventory_sha256: str,
    vendored_gcf_root: str | Path,
) -> dict[str, Any]:
    """Independently verify one SEALED lossless adoption before publication."""

    expected = _require_sha256(
        expected_source_inventory_sha256, label="expected source inventory"
    )
    held.revalidate()
    source = inspect_fixed_budget_neurosed_pass(
        source_root, vendored_gcf_root=vendored_gcf_root
    )
    copied = inspect_fixed_budget_neurosed_pass(
        held.sealed.artifact_root,
        vendored_gcf_root=vendored_gcf_root,
        allow_managed_generation_token=True,
    )
    raw = _held_json(held, "raw_evidence.json")
    worker_exit = _held_json(held, "worker_exit.json")
    evidence = raw.get("evidence")
    exit_evidence = worker_exit.get("exit")
    if (
        raw.get("schema_version") != WORKER_RAW_EVIDENCE_SCHEMA
        or worker_exit.get("schema_version") != WORKER_EXIT_SCHEMA
        or raw.get("attempt_id") != held.sealed.attempt_id
        or worker_exit.get("attempt_id") != held.sealed.attempt_id
        or raw.get("generation_token") != held.sealed.generation_token
        or worker_exit.get("generation_token") != held.sealed.generation_token
        or type(evidence) is not dict
        or type(exit_evidence) is not dict
    ):
        raise FixedBudgetNeuroSEDAdoptionError("managed generation binding changed")
    attempt = evidence.get("attempt_manifest")
    command = evidence.get("scientific_command")
    process_audit = exit_evidence.get("process_audit")
    expected_inputs = {
        "fixed_budget_neurosed_checkpoint_sha256": source["checkpoint_sha256"],
        "fixed_budget_neurosed_pass_sha256": source["pass_sha256"],
        "fixed_budget_neurosed_source_inventory_sha256": expected,
        "fixed_budget_neurosed_verification_sha256": source[
            "verification_sha256"
        ],
    }
    if (
        type(attempt) is not dict
        or attempt.get("task_id") != ADOPTION_TASK_ID
        or attempt.get("attempt_id") != held.sealed.attempt_id
        or attempt.get("input_hashes") != expected_inputs
        or type(command) is not list
        or not command
        or type(process_audit) is not dict
        or process_audit.get("state") != "EXITED"
        or process_audit.get("attempt_id") != held.sealed.attempt_id
        or exit_evidence.get("exit_code") != 0
        or exit_evidence.get("worker_closed_artifact_writers") is not True
        or source["inventory_sha256"] != expected
        or copied["inventory_sha256"] != expected
        or copied["t7_consumer"] != source["t7_consumer"]
    ):
        raise FixedBudgetNeuroSEDAdoptionError(
            "managed fixed-budget adoption binding changed"
        )
    command_text = "\0".join(str(value) for value in command)
    if (
        "adopt_tastemolnet_fixed_budget_neurosed_v2.py" not in command_text
        or " copy " not in f" {command_text.replace(chr(0), ' ')} "
        or "calibration.csv" in command_text
        or "test.csv" in command_text
    ):
        raise FixedBudgetNeuroSEDAdoptionError("managed adoption command changed")
    held.revalidate()
    return {
        "schema_version": ADOPTION_VERIFICATION_SCHEMA,
        "status": "PASS",
        "marker": "[TASTE_NEUROSED_FIXED_BUDGET_MANAGED_ADOPTION_PASS]",
        "scientific_artifact_modified": False,
        "source_root_copied_byte_for_byte": True,
        "source_fixed_budget_pass_reopened": True,
        "source_independent_verification_reopened": True,
        "managed_copy_independently_rehashed": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "source_root": str(Path(source_root)),
        "source_inventory_sha256": expected,
        "source_pass_sha256": source["pass_sha256"],
        "source_verification_sha256": source["verification_sha256"],
        "managed_attempt_binding": {
            "controller_id": attempt.get("controller_id"),
            "task_id": ADOPTION_TASK_ID,
            "attempt_id": held.sealed.attempt_id,
            "generation_token": held.sealed.generation_token,
            "input_hashes": expected_inputs,
            "worker_exit_code": 0,
            "worker_closed_artifact_writers": True,
        },
        "t7_consumer": dict(copied["t7_consumer"]),
    }


def validate_t7_fixed_budget_consumer(
    consumer: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    feature_schema_sha256: str,
    sha256s_sha256: str,
    feature_atomic_numbers: list[int],
    feature_input_dim: int,
) -> dict[str, Any]:
    """Validate the exact consumer document used by the T7 worker/verifier."""

    payload = dict(consumer)
    _expect_fields(
        payload,
        {
            "schema_version": T7_FIXED_BUDGET_CONSUMER_SCHEMA,
            "dataset": "tastemolnet",
            "role": "GCF_AUXILIARY_DISTANCE_MODEL",
            "classifier": False,
            "source_label_independent": True,
            "train_only_fit": True,
            "validation_only_selection": True,
            "calibration_loaded": False,
            "test_loaded": False,
            "health_gate_status": "PASS",
            "checkpoint_relative_path": "artifacts/best.pt",
            "checkpoint_sha256": checkpoint_sha256,
            "feature_schema_relative_path": "artifacts/feature_schema.json",
            "feature_schema_sha256": feature_schema_sha256,
            "feature_atomic_numbers": feature_atomic_numbers,
            "feature_input_dim": feature_input_dim,
            "sha256s_relative_path": "artifacts/sha256sums.txt",
            "sha256s_sha256": sha256s_sha256,
            "train_pair_budget": 5000,
            "validation_pair_budget": 1000,
        },
        label="T7 fixed-budget NeuroSED consumer",
    )
    for name in (
        "train_pair_sampler_manifest_sha256",
        "validation_pair_sampler_manifest_sha256",
        "train_pair_labels_manifest_sha256",
        "validation_pair_labels_manifest_sha256",
        "split_isolation_sha256",
        "selector_trace_sha256",
        "distance_direction_trace_sha256",
        "fixed_budget_pass_sha256",
        "fixed_budget_verification_sha256",
        "fixed_budget_source_inventory_sha256",
    ):
        _require_sha256(payload.get(name), label=f"consumer {name}")
    return payload


__all__ = [
    "ADOPTION_TASK_ID",
    "ADOPTION_VERIFICATION_SCHEMA",
    "FixedBudgetNeuroSEDAdoptionError",
    "T7_FIXED_BUDGET_CONSUMER_SCHEMA",
    "copy_fixed_budget_neurosed_pass",
    "inspect_fixed_budget_neurosed_pass",
    "validate_t7_fixed_budget_consumer",
    "verify_fixed_budget_managed_adoption",
]
