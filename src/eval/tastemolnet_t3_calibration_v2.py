"""Fresh validation-only TasteMolNet T3 calibration under managed v2.

The worker creates a candidate checkpoint but has no terminal authority. A
separate process retains the complete SEALED worker tree, independently binds
the T2 adoption receipt and source bundle, repeats the scalar-temperature fit,
and only then asks the managed-v2 terminal publisher to publish PASS.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Any, Mapping

from src.oracles.gnn_oracle import (
    REQUIRED_CHECKPOINT_FILES,
    TASTE_REQUIRED_CHECKPOINT_FILES,
    fit_temperature_scaling,
    verify_checkpoint_bundle,
)
from src.utils.managed_execution_v2 import WORKER_EXIT_SCHEMA, WORKER_RAW_EVIDENCE_SCHEMA
from src.utils.tastemolnet_t2_adoption_v2 import (
    HeldBundle,
    HeldFile,
    PASS_MARKER as T2_PASS_MARKER,
    TasteT2AdoptionError,
    _read_validation_predictions,
    _stable_sha256,
)
from src.utils.terminal_publisher_v2 import (
    HeldSealedArtifactV2,
    TerminalPublicationV2,
    open_sealed_worker_artifact,
    verify_and_publish_sealed_attempt,
)


SCHEMA_VERSION = "tastemolnet_t3_calibration_v2"
STAGE = "T3_GINE_CALIBRATED"
TASK_ID = "T3_CALIBRATION"
PASS_MARKER = "[TASTE_T3_CALIBRATION_PASS]"
CANDIDATE_NAME = "t3_calibration_candidate.json"
CHECKPOINT_DIRECTORY = "checkpoint"
T2_RECEIPT_FILES = frozenset(
    {
        "PASS",
        "artifact_hashes.json",
        "gate.json",
        "input_hashes.json",
        "sha256s.txt",
        "source_evidence.json",
        "verification.json",
    }
)
SOURCE_CHECKPOINT_FILES = frozenset(
    set(REQUIRED_CHECKPOINT_FILES) | set(TASTE_REQUIRED_CHECKPOINT_FILES)
)
CANDIDATE_CHECKPOINT_FILES = frozenset(SOURCE_CHECKPOINT_FILES | {CANDIDATE_NAME})
MODIFIED_SOURCE_FILES = frozenset(
    {
        "model_card.json",
        "oracle_manifest.json",
        "temperature_scaling.json",
        "sha256sums.txt",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TasteT3CalibrationError(RuntimeError):
    """T3 candidate construction or independent verification failed."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_object(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteT3CalibrationError(f"{label} is not valid JSON") from exc
    if type(value) is not dict:
        raise TasteT3CalibrationError(f"{label} must be one JSON object")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise TasteT3CalibrationError(f"short write for {path.name}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parse_hash_manifest(
    data: bytes, *, expected: set[str] | frozenset[str], label: str
) -> dict[str, str]:
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise TasteT3CalibrationError(f"{label} is not UTF-8") from exc
    result: dict[str, str] = {}
    for line in lines:
        digest, separator, name = line.partition("  ")
        if (
            not separator
            or not SHA256_RE.fullmatch(digest)
            or Path(name).name != name
            or name in result
        ):
            raise TasteT3CalibrationError(f"{label} is malformed")
        result[name] = digest
    if set(result) != set(expected):
        raise TasteT3CalibrationError(f"{label} does not close its exact inventory")
    return result


class HeldT2Receipt:
    """Descriptor-retained exact T2 adoption receipt."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.is_absolute() or self.root.resolve(strict=True) != self.root:
            raise TasteT3CalibrationError("T2 receipt root must be an exact physical path")
        self.descriptor = os.open(
            self.root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        info = os.fstat(self.descriptor)
        named = os.stat(self.root, follow_symlinks=False)
        if (
            not stat.S_ISDIR(info.st_mode)
            or (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino)
        ):
            os.close(self.descriptor)
            raise TasteT3CalibrationError("T2 receipt root is not one physical directory")
        self.identity = (int(info.st_dev), int(info.st_ino), int(info.st_mtime_ns))
        names = {entry.name for entry in os.scandir(self.descriptor)}
        if names != T2_RECEIPT_FILES:
            os.close(self.descriptor)
            raise TasteT3CalibrationError("T2 receipt inventory changed")
        self.files: dict[str, HeldFile] = {}
        try:
            for name in sorted(names):
                self.files[name] = HeldFile(
                    self.root / name, label=f"T2 receipt {name}"
                )
        except TasteT2AdoptionError as exc:
            self.close()
            raise TasteT3CalibrationError(str(exc)) from exc
        except BaseException:
            self.close()
            raise

    def verify(self) -> None:
        info = os.fstat(self.descriptor)
        named = os.stat(self.root, follow_symlinks=False)
        if (
            (int(info.st_dev), int(info.st_ino), int(info.st_mtime_ns)) != self.identity
            or (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino)
            or {entry.name for entry in os.scandir(self.descriptor)} != T2_RECEIPT_FILES
        ):
            raise TasteT3CalibrationError("T2 receipt changed while held")
        for item in self.files.values():
            item.verify()

    def close(self) -> None:
        for item in getattr(self, "files", {}).values():
            item.close()
        if getattr(self, "descriptor", -1) >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


def _validate_t2_authorities(
    receipt: HeldT2Receipt,
    bundle: HeldBundle,
) -> dict[str, Any]:
    receipt.verify()
    bundle.verify()
    manifest = _parse_hash_manifest(
        receipt.files["sha256s.txt"].bytes(),
        expected=T2_RECEIPT_FILES - {"PASS", "sha256s.txt"},
        label="T2 receipt sha256s.txt",
    )
    for name, digest in manifest.items():
        if receipt.files[name].sha256 != digest:
            raise TasteT3CalibrationError(f"T2 receipt checksum changed: {name}")
    if receipt.files["PASS"].bytes() != (T2_PASS_MARKER + "\n").encode("utf-8"):
        raise TasteT3CalibrationError("T2 receipt PASS marker changed")
    gate = receipt.files["gate.json"].json()
    source = receipt.files["source_evidence.json"].json()
    verification = receipt.files["verification.json"].json()
    artifact_document = receipt.files["artifact_hashes.json"].json()
    receipt_id = receipt.root.name
    source_hash = source.get("source_evidence_sha256")
    source_without_hash = dict(source)
    source_without_hash.pop("source_evidence_sha256", None)
    actual_hashes = {name: item.sha256 for name, item in bundle.files.items()}
    if (
        gate.get("status") != "PASS"
        or gate.get("state") != "ADOPTED_SCIENTIFIC_PASS"
        or gate.get("stage") != "T2_GINE"
        or gate.get("receipt_id") != receipt_id
        or gate.get("marker") != T2_PASS_MARKER
        or gate.get("artifact_root") != str(bundle.root)
        or gate.get("model_sha256") != actual_hashes["model.pt"]
        or gate.get("source_evidence_sha256") != source_hash
        or source.get("receipt_id") != receipt_id
        or source.get("artifact_root") != str(bundle.root)
        or source.get("artifact_hashes") != actual_hashes
        or source.get("old_failure_superseded_for_scientific_artifact") is not True
        or source.get("old_process_evidence_not_rewritten") is not True
        or source.get("calibration_loaded") is not False
        or source.get("test_loaded") is not False
        or source.get("rf_oracle_used") is not False
        or source_hash != _stable_sha256(source_without_hash)
        or verification.get("verification_result") != "PASS"
        or verification.get("source_evidence_sha256") != source_hash
        or artifact_document.get("artifact_root") != str(bundle.root)
        or artifact_document.get("artifact_hashes") != actual_hashes
    ):
        raise TasteT3CalibrationError("T2 adoption receipt does not bind the source bundle")
    receipt.verify()
    bundle.verify()
    return {
        "receipt_id": receipt_id,
        "receipt_root": str(receipt.root),
        "receipt_gate_sha256": receipt.files["gate.json"].sha256,
        "source_evidence_sha256": source_hash,
        "source_bundle_root": str(bundle.root),
        "source_artifact_hashes": actual_hashes,
        "validation_row_ids_sha256": source.get("validation_row_ids_sha256"),
        "model_sha256": actual_hashes["model.pt"],
        "feature_schema_file_sha256": actual_hashes["feature_schema.json"],
    }


def fit_fresh_temperature(
    validation_predictions: bytes,
    *,
    attempt_id: str,
    generation_token: str,
    receipt_id: str,
    receipt_gate_sha256: str,
    source_model_sha256: str,
    source_predictions_sha256: str,
    max_iter: int,
    fitted_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit one new scalar using only held validation prediction rows."""

    if type(max_iter) is not int or max_iter <= 0:
        raise TasteT3CalibrationError("max_iter must be one positive native integer")
    stored = _read_validation_predictions(validation_predictions)
    result = fit_temperature_scaling(stored["logits"], stored["labels"], max_iter=max_iter)
    if (
        result.get("status") != "fit"
        or result.get("selection_split") != "validation"
        or result.get("test_used_for_fit") is not False
        or result.get("argmax_invariant") is not True
        or result.get("num_classes") != 3
        or result.get("num_examples") != len(stored["molecule_ids"])
    ):
        raise TasteT3CalibrationError("fresh temperature optimizer returned an unsafe result")
    for key in (
        "temperature",
        "nll_before",
        "nll_after",
        "ece_before",
        "ece_after",
        "brier_before",
        "brier_after",
    ):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TasteT3CalibrationError(f"fresh temperature {key} is not numeric")
        if not math.isfinite(float(value)):
            raise TasteT3CalibrationError(f"fresh temperature {key} is not finite")
    if float(result["temperature"]) <= 0.0:
        raise TasteT3CalibrationError("fresh temperature is not positive")
    result.update(
        {
            "fit_generation": "FRESH_T3_REFIT",
            "temperature_refit_performed": True,
            "validation_row_ids_sha256": stored["row_ids_sha256"],
            "validation_predictions_sha256": source_predictions_sha256,
            "source_model_sha256": source_model_sha256,
            "source_t2_receipt_id": receipt_id,
            "source_t2_gate_sha256": receipt_gate_sha256,
            "managed_attempt_id": attempt_id,
            "managed_generation_token": generation_token,
            "fitted_at": fitted_at or _utc_now(),
            "optimizer": "torch_lbfgs_log_temperature",
            "optimizer_max_iter": max_iter,
            "calibration_payload_loaded": False,
            "test_payload_loaded": False,
            "rf_oracle_used": False,
        }
    )
    evidence = {
        "num_validation_rows": len(stored["molecule_ids"]),
        "validation_row_ids_sha256": stored["row_ids_sha256"],
        "predicted_classes": stored["predicted_classes"],
    }
    return result, evidence


def _modified_model_card(
    source: Mapping[str, Any],
    temperature: Mapping[str, Any],
    temperature_sha256: str,
) -> dict[str, Any]:
    result = json.loads(json.dumps(source))
    result["temperature_calibration"] = {
        "stage": STAGE,
        "status": "fit",
        "split": "validation",
        "test_used_for_fit": False,
        "temperature_refit_performed": True,
        "temperature": temperature["temperature"],
        "temperature_scaling_sha256": temperature_sha256,
        "source_t2_receipt_id": temperature["source_t2_receipt_id"],
        "managed_attempt_id": temperature["managed_attempt_id"],
    }
    return result


def _modified_oracle_manifest(
    source: Mapping[str, Any],
    temperature: Mapping[str, Any],
    temperature_sha256: str,
) -> dict[str, Any]:
    result = json.loads(json.dumps(source))
    result["temperature_scaling"] = dict(temperature)
    result["t3_calibration"] = {
        "stage": STAGE,
        "candidate_status": "SEALED_CANDIDATE",
        "temperature_refit_performed": True,
        "selection_split": "validation",
        "calibration_payload_loaded": False,
        "test_payload_loaded": False,
        "source_t2_receipt_id": temperature["source_t2_receipt_id"],
        "managed_attempt_id": temperature["managed_attempt_id"],
        "temperature_scaling_sha256": temperature_sha256,
        "independent_verification_required": True,
    }
    return result


def build_t3_candidate(
    *,
    t2_receipt_root: str | Path,
    source_bundle_root: str | Path,
    artifact_root: str | Path,
    attempt_id: str,
    generation_token: str,
    max_iter: int = 100,
) -> dict[str, Any]:
    """Worker-side construction. The returned artifact is not a PASS."""

    output = Path(artifact_root)
    if not output.is_absolute() or output.resolve(strict=True) != output:
        raise TasteT3CalibrationError("managed artifact root must be exact and physical")
    if {entry.name for entry in os.scandir(output)} != {".generation_token.json"}:
        raise TasteT3CalibrationError("managed artifact root is not fresh")
    receipt = HeldT2Receipt(t2_receipt_root)
    try:
        bundle = HeldBundle(source_bundle_root)
    except TasteT2AdoptionError as exc:
        receipt.close()
        raise TasteT3CalibrationError(str(exc)) from exc
    except BaseException:
        receipt.close()
        raise
    try:
        binding = _validate_t2_authorities(receipt, bundle)
        if binding["validation_row_ids_sha256"] is None:
            raise TasteT3CalibrationError("T2 receipt lacks validation row ID binding")
        validation_bytes = bundle.files["validation_predictions.csv"].bytes()
        temperature, validation = fit_fresh_temperature(
            validation_bytes,
            attempt_id=attempt_id,
            generation_token=generation_token,
            receipt_id=binding["receipt_id"],
            receipt_gate_sha256=binding["receipt_gate_sha256"],
            source_model_sha256=binding["model_sha256"],
            source_predictions_sha256=bundle.files["validation_predictions.csv"].sha256,
            max_iter=max_iter,
        )
        if validation["validation_row_ids_sha256"] != binding["validation_row_ids_sha256"]:
            raise TasteT3CalibrationError("validation row order differs from T2 adoption")
        checkpoint = output / CHECKPOINT_DIRECTORY
        os.mkdir(checkpoint, 0o700)
        source_card = bundle.files["model_card.json"].json()
        source_oracle = bundle.files["oracle_manifest.json"].json()
        temperature_bytes = _canonical_json_bytes(temperature)
        temperature_sha256 = _sha256(temperature_bytes)
        replacements = {
            "model_card.json": _canonical_json_bytes(
                _modified_model_card(source_card, temperature, temperature_sha256)
            ),
            "oracle_manifest.json": _canonical_json_bytes(
                _modified_oracle_manifest(
                    source_oracle, temperature, temperature_sha256
                )
            ),
            "temperature_scaling.json": temperature_bytes,
        }
        for name in sorted(SOURCE_CHECKPOINT_FILES - {"sha256sums.txt"}):
            data = replacements.get(name, bundle.files[name].bytes())
            _write_exclusive(checkpoint / name, data)
        candidate = {
            "schema_version": SCHEMA_VERSION,
            "stage": STAGE,
            "candidate_status": "SEALED_CANDIDATE",
            "managed_attempt_id": attempt_id,
            "managed_generation_token": generation_token,
            "created_at": _utc_now(),
            "t2_binding": binding,
            "temperature_scaling": temperature,
            "temperature_scaling_sha256": temperature_sha256,
            "validation_evidence": validation,
            "model_sha256": _sha256((checkpoint / "model.pt").read_bytes()),
            "feature_schema_file_sha256": _sha256(
                (checkpoint / "feature_schema.json").read_bytes()
            ),
            "calibration_payload_loaded": False,
            "test_payload_loaded": False,
            "rf_oracle_used": False,
            "independent_verification_required": True,
            "matrix_method_cell": False,
        }
        _write_exclusive(checkpoint / CANDIDATE_NAME, _canonical_json_bytes(candidate))
        hashes = {
            path.name: _sha256(path.read_bytes())
            for path in sorted(checkpoint.iterdir())
            if path.name != "sha256sums.txt"
        }
        _write_exclusive(
            checkpoint / "sha256sums.txt",
            "".join(
                f"{digest}  {name}\n" for name, digest in sorted(hashes.items())
            ).encode("utf-8"),
        )
        directory = os.open(checkpoint, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        verify_checkpoint_bundle(checkpoint)
        receipt.verify()
        bundle.verify()
        return {
            "state": "SEALED_CANDIDATE",
            "stage": STAGE,
            "checkpoint_dir": str(checkpoint),
            "temperature": float(temperature["temperature"]),
            "model_sha256": candidate["model_sha256"],
            "temperature_scaling_sha256": hashes["temperature_scaling.json"],
            "feature_schema_file_sha256": candidate["feature_schema_file_sha256"],
            "validation_row_ids_sha256": validation["validation_row_ids_sha256"],
            "calibration_payload_loaded": False,
            "test_payload_loaded": False,
            "independent_verification_required": True,
        }
    except TasteT2AdoptionError as exc:
        raise TasteT3CalibrationError(str(exc)) from exc
    finally:
        bundle.close()
        receipt.close()


def _read_held_file(held: HeldSealedArtifactV2, relative: str) -> bytes:
    held.revalidate()
    matches = [item for item in held.files if item.evidence.relative_path == relative]
    if len(matches) != 1:
        raise TasteT3CalibrationError(f"SEALED candidate file is absent: {relative}")
    item = matches[0]
    os.lseek(item.descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        block = os.read(item.descriptor, 1024 * 1024)
        if not block:
            break
        chunks.append(block)
    os.lseek(item.descriptor, 0, os.SEEK_SET)
    data = b"".join(chunks)
    if _sha256(data) != item.evidence.sha256:
        raise TasteT3CalibrationError(f"SEALED candidate hash changed: {relative}")
    held.revalidate()
    return data


def _candidate_payloads(held: HeldSealedArtifactV2) -> dict[str, bytes]:
    expected_files = {
        ".generation_token.json",
        "raw_evidence.json",
        "worker_exit.json",
        "artifacts/.generation_token.json",
        *{
            f"artifacts/{CHECKPOINT_DIRECTORY}/{name}"
            for name in CANDIDATE_CHECKPOINT_FILES
        },
    }
    observed_files = {item.evidence.relative_path for item in held.files}
    observed_directories = {item.relative_path for item in held.inventory.directories}
    if observed_files != expected_files or observed_directories != {
        "artifacts",
        f"artifacts/{CHECKPOINT_DIRECTORY}",
    }:
        raise TasteT3CalibrationError("SEALED T3 candidate inventory is not exact")
    prefix = f"artifacts/{CHECKPOINT_DIRECTORY}/"
    return {
        name: _read_held_file(held, prefix + name)
        for name in sorted(CANDIDATE_CHECKPOINT_FILES)
    }


def _same_number(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)


def _independently_verify_t3(
    held: HeldSealedArtifactV2,
    *,
    receipt: HeldT2Receipt,
    bundle: HeldBundle,
    expected_controller_id: str,
    expected_git_commit: str,
    expected_config_hash: str,
    max_iter: int,
) -> dict[str, Any]:
    binding = _validate_t2_authorities(receipt, bundle)
    payloads = _candidate_payloads(held)
    raw = _json_object(
        _read_held_file(held, "raw_evidence.json"), label="worker raw evidence"
    )
    worker_exit = _json_object(
        _read_held_file(held, "worker_exit.json"), label="worker exit evidence"
    )
    raw_evidence = raw.get("evidence")
    if not isinstance(raw_evidence, Mapping):
        raise TasteT3CalibrationError("managed T3 raw evidence is absent")
    attempt_manifest = raw_evidence.get("attempt_manifest")
    if not isinstance(attempt_manifest, Mapping):
        raise TasteT3CalibrationError("managed T3 attempt manifest is absent")
    worker_exit_body = worker_exit.get("exit")
    if not isinstance(worker_exit_body, Mapping):
        raise TasteT3CalibrationError("managed T3 worker-exit body is absent")
    process_audit = worker_exit_body.get("process_audit")
    if not isinstance(process_audit, Mapping):
        raise TasteT3CalibrationError("managed T3 process audit is absent")
    process_lineage = raw_evidence.get("process_lineage")
    if not isinstance(process_lineage, Mapping):
        raise TasteT3CalibrationError("managed T3 process lineage is absent")
    expected_inputs = {
        "t2_receipt_gate": binding["receipt_gate_sha256"],
        "t2_source_evidence": binding["source_evidence_sha256"],
        "t2_source_sha256s": bundle.files["sha256sums.txt"].sha256,
    }
    if (
        raw.get("schema_version") != WORKER_RAW_EVIDENCE_SCHEMA
        or worker_exit.get("schema_version") != WORKER_EXIT_SCHEMA
        or raw.get("attempt_id") != held.sealed.attempt_id
        or raw.get("generation_token") != held.sealed.generation_token
        or worker_exit.get("attempt_id") != held.sealed.attempt_id
        or worker_exit.get("generation_token") != held.sealed.generation_token
        or attempt_manifest.get("attempt_id") != held.sealed.attempt_id
        or attempt_manifest.get("task_id") != TASK_ID
        or attempt_manifest.get("controller_id") != expected_controller_id
        or attempt_manifest.get("git_commit") != expected_git_commit
        or attempt_manifest.get("config_hash") != expected_config_hash
        or attempt_manifest.get("input_hashes") != expected_inputs
        or attempt_manifest.get("auto_terminate_uncontrolled_children") is not False
        or process_lineage.get("controller_id") != expected_controller_id
        or process_lineage.get("attempt_id") != held.sealed.attempt_id
        or worker_exit_body.get("exit_code") != 0
        or worker_exit_body.get("worker_closed_artifact_writers") is not True
        or process_audit.get("state") != "EXITED"
        or process_audit.get("controller_id") != expected_controller_id
        or process_audit.get("attempt_id") != held.sealed.attempt_id
    ):
        raise TasteT3CalibrationError("managed T3 worker evidence is not releasable")
    manifest = _parse_hash_manifest(
        payloads["sha256sums.txt"],
        expected=CANDIDATE_CHECKPOINT_FILES - {"sha256sums.txt"},
        label="candidate sha256sums.txt",
    )
    for name, digest in manifest.items():
        if _sha256(payloads[name]) != digest:
            raise TasteT3CalibrationError(f"candidate checkpoint hash changed: {name}")
    for name in SOURCE_CHECKPOINT_FILES - MODIFIED_SOURCE_FILES:
        if _sha256(payloads[name]) != bundle.files[name].sha256:
            raise TasteT3CalibrationError(f"candidate changed frozen source file: {name}")
    temperature = _json_object(
        payloads["temperature_scaling.json"], label="candidate temperature"
    )
    temperature_sha256 = _sha256(payloads["temperature_scaling.json"])
    fitted_at = temperature.get("fitted_at")
    if type(fitted_at) is not str or not fitted_at:
        raise TasteT3CalibrationError("candidate fresh-fit timestamp is absent")
    candidate = _json_object(payloads[CANDIDATE_NAME], label="T3 candidate manifest")
    independently_fitted, validation = fit_fresh_temperature(
        bundle.files["validation_predictions.csv"].bytes(),
        attempt_id=held.sealed.attempt_id,
        generation_token=held.sealed.generation_token,
        receipt_id=binding["receipt_id"],
        receipt_gate_sha256=binding["receipt_gate_sha256"],
        source_model_sha256=binding["model_sha256"],
        source_predictions_sha256=bundle.files["validation_predictions.csv"].sha256,
        max_iter=max_iter,
        fitted_at=fitted_at,
    )
    for key, expected in independently_fitted.items():
        observed = temperature.get(key)
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if not _same_number(observed, expected):
                raise TasteT3CalibrationError(f"independent T3 refit differs: {key}")
        elif observed != expected:
            raise TasteT3CalibrationError(f"independent T3 provenance differs: {key}")
    source_card = bundle.files["model_card.json"].json()
    source_oracle = bundle.files["oracle_manifest.json"].json()
    if _json_object(
        payloads["model_card.json"], label="candidate model card"
    ) != _modified_model_card(source_card, temperature, temperature_sha256):
        raise TasteT3CalibrationError("candidate model card changed outside T3 calibration")
    if _json_object(
        payloads["oracle_manifest.json"], label="candidate oracle manifest"
    ) != _modified_oracle_manifest(source_oracle, temperature, temperature_sha256):
        raise TasteT3CalibrationError("candidate oracle manifest changed outside T3 calibration")
    feature_schema = _json_object(
        payloads["feature_schema.json"], label="candidate feature schema"
    )
    if (
        candidate.get("schema_version") != SCHEMA_VERSION
        or candidate.get("stage") != STAGE
        or candidate.get("candidate_status") != "SEALED_CANDIDATE"
        or candidate.get("managed_attempt_id") != held.sealed.attempt_id
        or candidate.get("managed_generation_token") != held.sealed.generation_token
        or candidate.get("t2_binding") != binding
        or candidate.get("temperature_scaling") != temperature
        or candidate.get("temperature_scaling_sha256") != temperature_sha256
        or candidate.get("validation_evidence") != validation
        or candidate.get("model_sha256") != binding["model_sha256"]
        or candidate.get("feature_schema_file_sha256")
        != binding["feature_schema_file_sha256"]
        or candidate.get("calibration_payload_loaded") is not False
        or candidate.get("test_payload_loaded") is not False
        or candidate.get("rf_oracle_used") is not False
        or candidate.get("independent_verification_required") is not True
        or candidate.get("matrix_method_cell") is not False
        or feature_schema.get("schema_sha256") is None
    ):
        raise TasteT3CalibrationError("T3 candidate manifest contract changed")
    held.revalidate()
    receipt.verify()
    bundle.verify()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "stage": STAGE,
        "marker": PASS_MARKER,
        "independent_scientific_verifier": True,
        "verifier_git_commit": expected_git_commit,
        "t2_receipt_id": binding["receipt_id"],
        "t2_receipt_gate_sha256": binding["receipt_gate_sha256"],
        "source_evidence_sha256": binding["source_evidence_sha256"],
        "source_bundle_root": binding["source_bundle_root"],
        "model_sha256": binding["model_sha256"],
        "temperature": float(temperature["temperature"]),
        "temperature_scaling_sha256": temperature_sha256,
        "feature_schema_file_sha256": binding["feature_schema_file_sha256"],
        "feature_schema_sha256": feature_schema["schema_sha256"],
        "validation_row_ids_sha256": validation["validation_row_ids_sha256"],
        "temperature_refit_performed": True,
        "selection_split": "validation",
        "calibration_payload_loaded": False,
        "test_payload_loaded": False,
        "rf_oracle_used": False,
        "argmax_invariant": True,
        "nll_before": float(temperature["nll_before"]),
        "nll_after": float(temperature["nll_after"]),
        "ece_before": float(temperature["ece_before"]),
        "ece_after": float(temperature["ece_after"]),
        "brier_before": float(temperature["brier_before"]),
        "brier_after": float(temperature["brier_after"]),
        "matrix_method_cell": False,
        "downstream_same_model_temperature_schema_required": True,
    }


def verify_and_publish_t3(
    *,
    sealed_path: str | Path,
    final_path: str | Path,
    t2_receipt_root: str | Path,
    source_bundle_root: str | Path,
    expected_attempt_id: str,
    expected_generation_token: str,
    expected_controller_id: str,
    expected_git_commit: str,
    expected_config_hash: str,
    max_iter: int = 100,
) -> tuple[TerminalPublicationV2, dict[str, Any]]:
    """Independent verifier entry: repeat science, then publish atomically."""

    receipt = HeldT2Receipt(t2_receipt_root)
    try:
        bundle = HeldBundle(source_bundle_root)
    except TasteT2AdoptionError as exc:
        receipt.close()
        raise TasteT3CalibrationError(str(exc)) from exc
    except BaseException:
        receipt.close()
        raise
    try:
        destination = Path(final_path)
        if (
            not destination.is_absolute()
            or destination.parent != bundle.root.parent
            or not destination.name.startswith("calibrated-")
        ):
            raise TasteT3CalibrationError(
                "T3 final path must be a calibrated-* sibling of the T2 bundle"
            )
        with open_sealed_worker_artifact(
            sealed_path,
            expected_attempt_id=expected_attempt_id,
            expected_generation_token=expected_generation_token,
        ) as held:
            verification = _independently_verify_t3(
                held,
                receipt=receipt,
                bundle=bundle,
                expected_controller_id=expected_controller_id,
                expected_git_commit=expected_git_commit,
                expected_config_hash=expected_config_hash,
                max_iter=max_iter,
            )
            receipt.verify()
            bundle.verify()
            publication = verify_and_publish_sealed_attempt(
                held,
                final_path=final_path,
                verification=verification,
            )
        return publication, verification
    except TasteT2AdoptionError as exc:
        raise TasteT3CalibrationError(str(exc)) from exc
    finally:
        bundle.close()
        receipt.close()


__all__ = [
    "PASS_MARKER",
    "SCHEMA_VERSION",
    "STAGE",
    "TASK_ID",
    "TasteT3CalibrationError",
    "build_t3_candidate",
    "fit_fresh_temperature",
    "verify_and_publish_t3",
]
