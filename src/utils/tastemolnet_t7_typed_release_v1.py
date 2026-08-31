"""Typed runtime release for the adopted TasteMolNet T7 GCF smoke.

The fixed-budget NeuroSED model and the calibrated Taste GINE already have
independently published managed-v2 authorities.  This module joins those two
authorities without retraining either model.  A candidate writer may only
write release evidence; a separate invocation reopens every source and is the
only path that can publish ``[TASTE_T7_TYPED_RELEASE_PASS]``.

The module is deliberately dataset-specific.  It is not a controller or a new
managed-execution framework.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Mapping
import uuid

from src.eval.tastemolnet_neurosed_fixed_budget_adoption import (
    ADOPTION_VERIFICATION_SCHEMA,
    inspect_fixed_budget_neurosed_pass,
    validate_t7_fixed_budget_consumer,
)
from src.eval.tastemolnet_neurosed_official_fixed_budget import (
    OFFICIAL_GCF_COMMIT,
    verify_vendored_gcf_retained_inventory,
)
from src.eval.tastemolnet_neurosed_gate import STRICT_OFFICIAL_PROVENANCE
from src.utils.managed_final_consumer_v2 import hold_verified_managed_final
from src.utils.process_identity_v2 import canonical_json_bytes
from src.utils.retained_readonly_file import hold_readonly_file
from src.utils.terminal_publisher_v2 import _atomic_rename_noreplace


DATASET = "tastemolnet"
NUM_CLASSES = 3
SOURCE_LABEL = 1
INFERENCE_DIRECTION = "generated_to_original"
SOURCE_DIRECTION = "generated_query_to_original_target"
PINS_SCHEMA = "taste_gcf_release_pins_v1"
CANDIDATE_SCHEMA = "taste_gcf_release_candidate_v1"
SOURCE_AUTHORITY_SCHEMA = "taste_gcf_release_source_authority_v1"
VERIFICATION_SCHEMA = "taste_gcf_release_verification_v1"
GATE_SCHEMA = "taste_gcf_release_gate_v1"
PASS_MARKER = "[TASTE_T7_TYPED_RELEASE_PASS]"
READY_MARKER = "[TASTE_T7_TYPED_RELEASE_READY_FOR_INDEPENDENT_VERIFICATION]"
T7_TYPED_RAW_EVIDENCE_SCHEMA = "tastemolnet_t7_gcf_worker_raw_evidence_v3"
T7_TYPED_VERIFICATION_SCHEMA = "tastemolnet_t7_gcf_independent_verification_v3"

REPO_ROOT = Path(__file__).resolve().parents[2]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_GPU_UUID = re.compile(r"^GPU-[A-Za-z0-9-]+$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SPLITS = ("train", "validation", "calibration", "test")


class TasteGCFReleaseError(RuntimeError):
    """The typed T7 release or its source authority changed."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_mapping(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(canonical_json_bytes(dict(value)))


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise TasteGCFReleaseError(f"{label} must be lowercase SHA-256")
    return value


def _require_git_sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _GIT_SHA.fullmatch(value) is None:
        raise TasteGCFReleaseError(f"{label} must be a full lowercase Git SHA")
    return value


def _absolute(
    value: str | Path,
    *,
    label: str,
    must_exist: bool = True,
) -> Path:
    path = Path(value).expanduser()
    normalized = Path(os.path.abspath(path))
    if not path.is_absolute() or path != normalized:
        raise TasteGCFReleaseError(f"{label} must be normalized and absolute")
    if must_exist:
        try:
            physical = path.resolve(strict=True)
        except OSError as exc:
            raise TasteGCFReleaseError(f"{label} does not exist") from exc
        if physical != path:
            raise TasteGCFReleaseError(f"{label} must be one physical path")
    return path


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteGCFReleaseError(f"{label} is malformed JSON") from exc
    if type(value) is not dict:
        raise TasteGCFReleaseError(f"{label} must contain one JSON object")
    return value


def _write_new(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        mode,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise TasteGCFReleaseError(f"short write for {path.name}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json_file(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TasteGCFReleaseError(f"{label} is not one regular file")
    return _json(path.read_bytes(), label=label)


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise TasteGCFReleaseError(
            f"{label} keys changed: missing={missing}, extra={extra}"
        )


@dataclass(frozen=True, slots=True)
class TasteGCFReleasePinsV1:
    """The exact scientific pins authorized by the deadline T7 protocol."""

    dataset: str
    source_label: int
    num_classes: int
    official_gcf_commit: str
    neurosed_commit: str
    neurosed_model_sha: str
    neurosed_config_sha: str
    neurosed_pair_manifest_sha: str
    t3_calibrated_gine_sha: str
    t3_temperature_sha: str
    dataset_sha: str
    train_split_sha: str
    validation_split_sha: str
    calibration_split_sha: str
    test_split_sha: str
    inference_direction: str
    neurosed_calibration_loaded: bool
    neurosed_test_loaded: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TasteGCFReleasePinsV1":
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        _exact_keys(raw, expected | {"schema_version"}, label="T7 typed pins")
        if raw.get("schema_version") != PINS_SCHEMA:
            raise TasteGCFReleaseError("T7 typed-pins schema changed")
        values = {name: raw[name] for name in expected}
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        if (
            self.dataset != DATASET
            or type(self.source_label) is not int
            or self.source_label != SOURCE_LABEL
            or type(self.num_classes) is not int
            or self.num_classes != NUM_CLASSES
            or self.inference_direction != INFERENCE_DIRECTION
            or type(self.neurosed_calibration_loaded) is not bool
            or self.neurosed_calibration_loaded is not False
            or type(self.neurosed_test_loaded) is not bool
            or self.neurosed_test_loaded is not False
        ):
            raise TasteGCFReleaseError("T7 typed-pins scientific semantics changed")
        _require_git_sha(self.official_gcf_commit, label="official_gcf_commit")
        _require_git_sha(self.neurosed_commit, label="neurosed_commit")
        if self.official_gcf_commit != OFFICIAL_GCF_COMMIT:
            raise TasteGCFReleaseError("official GCF commit changed")
        if self.neurosed_commit != STRICT_OFFICIAL_PROVENANCE["greed_commit"]:
            raise TasteGCFReleaseError("official NeuroSED/GREED commit changed")
        for name in (
            "neurosed_model_sha",
            "neurosed_config_sha",
            "neurosed_pair_manifest_sha",
            "t3_calibrated_gine_sha",
            "t3_temperature_sha",
            "dataset_sha",
            "train_split_sha",
            "validation_split_sha",
            "calibration_split_sha",
            "test_split_sha",
        ):
            _require_sha256(getattr(self, name), label=name)

    def mapping(self) -> dict[str, Any]:
        self.validate()
        return {"schema_version": PINS_SCHEMA, **asdict(self)}

    @property
    def sha256(self) -> str:
        return _sha256_mapping(self.mapping())


@dataclass(frozen=True, slots=True)
class TasteGCFSourceAuthorityV1:
    managed_neurosed_root: str
    managed_neurosed_pass_sha256: str
    managed_neurosed_gate_sha256: str
    managed_neurosed_verification_sha256: str
    managed_neurosed_inventory_sha256: str
    neurosed_model_path: str
    neurosed_config_path: str
    neurosed_pair_manifest_path: str
    neurosed_feature_schema_path: str
    neurosed_sha256s_path: str
    t3_root: str
    t3_gate_sha256: str
    t3_verification_sha256: str
    t3_root_inventory_sha256: str
    t3_checkpoint_dir: str
    t3_checkpoint_id: str
    t3_split_manifest_sha256: str
    t3_feature_schema_sha256: str
    split_paths: Mapping[str, str]
    official_gcf_root: str
    official_gcf_inventory_sha256: str
    neurosed_distance_threshold: float
    implementation_commit: str
    implementation_tree: str
    no_neurosed_retraining: bool
    split_payloads_deserialized: bool
    test_payload_deserialized: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TasteGCFSourceAuthorityV1":
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        _exact_keys(raw, expected | {"schema_version"}, label="T7 source authority")
        if raw.get("schema_version") != SOURCE_AUTHORITY_SCHEMA:
            raise TasteGCFReleaseError("T7 source-authority schema changed")
        result = cls(**{name: raw[name] for name in expected})
        result.validate()
        return result

    def validate(self) -> None:
        for name in (
            "managed_neurosed_pass_sha256",
            "managed_neurosed_gate_sha256",
            "managed_neurosed_verification_sha256",
            "managed_neurosed_inventory_sha256",
            "t3_gate_sha256",
            "t3_verification_sha256",
            "t3_root_inventory_sha256",
            "t3_checkpoint_id",
            "t3_split_manifest_sha256",
            "t3_feature_schema_sha256",
            "official_gcf_inventory_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        _require_git_sha(self.implementation_commit, label="implementation_commit")
        _require_git_sha(self.implementation_tree, label="implementation_tree")
        if (
            isinstance(self.neurosed_distance_threshold, bool)
            or not isinstance(self.neurosed_distance_threshold, (int, float))
            or not math.isfinite(float(self.neurosed_distance_threshold))
            or float(self.neurosed_distance_threshold) < 0.0
        ):
            raise TasteGCFReleaseError("NeuroSED distance threshold is invalid")
        for name in (
            "managed_neurosed_root",
            "neurosed_model_path",
            "neurosed_config_path",
            "neurosed_pair_manifest_path",
            "neurosed_feature_schema_path",
            "neurosed_sha256s_path",
            "t3_root",
            "t3_checkpoint_dir",
            "official_gcf_root",
        ):
            _absolute(getattr(self, name), label=name, must_exist=False)
        if type(self.split_paths) is not dict or set(self.split_paths) != set(_SPLITS):
            raise TasteGCFReleaseError("T7 split path authority changed")
        for role, value in self.split_paths.items():
            _absolute(value, label=f"{role} split path", must_exist=False)
        if (
            type(self.no_neurosed_retraining) is not bool
            or self.no_neurosed_retraining is not True
            or type(self.split_payloads_deserialized) is not bool
            or self.split_payloads_deserialized is not False
            or type(self.test_payload_deserialized) is not bool
            or self.test_payload_deserialized is not False
        ):
            raise TasteGCFReleaseError("T7 source-authority access contract changed")

    def mapping(self) -> dict[str, Any]:
        self.validate()
        return {"schema_version": SOURCE_AUTHORITY_SCHEMA, **asdict(self)}

    @property
    def sha256(self) -> str:
        return _sha256_mapping(self.mapping())


def _git_output(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            [
                "/usr/bin/git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                "-C",
                str(REPO_ROOT),
                *arguments,
            ],
            check=True,
            capture_output=True,
            text=True,
            env={
                "PATH": "/usr/bin:/bin",
                "LC_ALL": "C",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_SYSTEM": os.devnull,
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_NO_REPLACE_OBJECTS": "1",
            },
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TasteGCFReleaseError("T7 immutable Git identity is unavailable") from exc
    return completed.stdout.strip()


def inspect_clean_execution() -> dict[str, str]:
    """Return the exact clean checkout identity used by release and science."""

    if _git_output("status", "--porcelain", "--untracked-files=all"):
        raise TasteGCFReleaseError("T7 execution checkout is not clean")
    commit = _git_output("rev-parse", "HEAD^{commit}")
    tree = _git_output("rev-parse", "HEAD^{tree}")
    _require_git_sha(commit, label="execution commit")
    _require_git_sha(tree, label="execution tree")
    return {"commit": commit, "tree": tree}


def _validate_split_manifest(data: bytes) -> tuple[dict[str, Any], dict[str, str]]:
    split = _json(data, label="T3 split manifest")
    _exact_keys(
        split,
        {
            "schema_version",
            "dataset",
            "roles",
            "files",
            "train_manifest",
            "validation_manifest",
            "calibration_loaded_for_training",
            "test_loaded_for_training",
            "test_evaluated_during_training",
            "test_used_for_checkpoint_selection",
        },
        label="T3 split manifest",
    )
    if (
        split.get("schema_version") != "molecular_gnn_split_manifest_v1"
        or split.get("dataset") != DATASET
        or split.get("roles")
        != {
            "train": "model_fitting",
            "validation": "checkpoint_selection_and_temperature_calibration",
            "calibration": "reserved_for_threshold_and_selector_only",
            "test": "frozen_model_final_quality_evaluation",
        }
        or split.get("calibration_loaded_for_training") is not False
        or split.get("test_loaded_for_training") is not False
        or split.get("test_evaluated_during_training") is not False
        or split.get("test_used_for_checkpoint_selection") is not False
    ):
        raise TasteGCFReleaseError("T3 split isolation changed")
    files = split.get("files")
    if type(files) is not dict or set(files) != set(_SPLITS):
        raise TasteGCFReleaseError("T3 split-file authority changed")
    paths: dict[str, str] = {}
    for role in _SPLITS:
        row = files[role]
        if type(row) is not dict or set(row) != {"path", "sha256"}:
            raise TasteGCFReleaseError(f"T3 {role} split binding changed")
        path = _absolute(row["path"], label=f"T3 {role} split")
        digest = _require_sha256(row["sha256"], label=f"T3 {role} split SHA")
        paths[role] = str(path)
        if not digest:
            raise AssertionError("unreachable")
    return split, paths


def _dataset_sha(split_hashes: Mapping[str, str]) -> str:
    if set(split_hashes) != set(_SPLITS):
        raise TasteGCFReleaseError("dataset hash lacks one frozen split")
    return _sha256_mapping(
        {
            "schema_version": "tastemolnet_t7_dataset_identity_v1",
            "dataset": DATASET,
            "split_sha256": {role: split_hashes[role] for role in _SPLITS},
        }
    )


@dataclass(slots=True)
class HeldTasteGCFReleaseSourcesV1:
    """Descriptor-held NeuroSED, T3, official source, and split authority."""

    stack: ExitStack
    managed_neurosed: Any
    t3: Any
    held_files: Mapping[str, Any]
    split_files: Mapping[str, Any]
    pins: TasteGCFReleasePinsV1
    authority: TasteGCFSourceAuthorityV1
    checkpoint_payloads: Mapping[str, bytes]
    train_bytes: bytes
    train_contract: Mapping[str, Any]
    neurosed_evidence: Mapping[str, Any]
    official_root: Path
    neurosed_model: Any

    def revalidate(self) -> tuple[TasteGCFReleasePinsV1, TasteGCFSourceAuthorityV1]:
        managed = self.managed_neurosed.revalidate()
        self.t3.verify()
        for item in self.held_files.values():
            item.revalidate()
        for item in self.split_files.values():
            item.revalidate()
        official = verify_vendored_gcf_retained_inventory(self.official_root)
        if official.get("inventory_sha256") != self.authority.official_gcf_inventory_sha256:
            raise TasteGCFReleaseError("vendored GCF inventory changed while held")
        execution = inspect_clean_execution()
        if (
            managed.get("pass_sha256")
            != self.authority.managed_neurosed_pass_sha256
            or managed.get("gate_sha256")
            != self.authority.managed_neurosed_gate_sha256
            or managed.get("verification_sha256")
            != self.authority.managed_neurosed_verification_sha256
            or managed.get("published_inventory_sha256")
            != self.authority.managed_neurosed_inventory_sha256
            or execution
            != {
                "commit": self.authority.implementation_commit,
                "tree": self.authority.implementation_tree,
            }
        ):
            raise TasteGCFReleaseError("T7 source authority changed while held")
        for role in _SPLITS:
            if self.split_files[role].sha256 != getattr(
                self.pins, f"{role}_split_sha"
            ):
                raise TasteGCFReleaseError(f"T7 {role} split changed while held")
        return self.pins, self.authority

    def close(self) -> None:
        self.stack.close()

    def __enter__(self) -> "HeldTasteGCFReleaseSourcesV1":
        self.revalidate()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def hold_t7_release_sources(
    *,
    managed_neurosed_root: str | Path,
    t3_root: str | Path,
    official_gcf_root: str | Path,
    neurosed_distance_threshold: float,
) -> HeldTasteGCFReleaseSourcesV1:
    """Open and derive pins from the real adopted NeuroSED and managed T3."""

    if (
        isinstance(neurosed_distance_threshold, bool)
        or not isinstance(neurosed_distance_threshold, (int, float))
        or not math.isfinite(float(neurosed_distance_threshold))
        or float(neurosed_distance_threshold) < 0.0
    ):
        raise TasteGCFReleaseError("NeuroSED distance threshold is invalid")
    neurosed_root = _absolute(
        managed_neurosed_root, label="managed NeuroSED root"
    )
    calibrated_root = _absolute(t3_root, label="managed T3 root")
    official_root = _absolute(official_gcf_root, label="vendored GCF root")
    if official_root != REPO_ROOT / "baselines/gcfexplainer_official":
        raise TasteGCFReleaseError("T7 official GCF root is not integrated source")
    stack = ExitStack()
    try:
        from src.eval.tastemolnet_t4_oracle_smoke_v2 import HeldPublishedT3

        managed = stack.enter_context(
            hold_verified_managed_final(
                neurosed_root,
                required_relative_paths=(
                    "artifacts/PASS",
                    "artifacts/best.pt",
                    "artifacts/config.yaml",
                    "artifacts/model_card.json",
                    "artifacts/pair_manifest.json",
                    "artifacts/split_manifest.json",
                    "artifacts/distance_direction_trace.json",
                    "artifacts/feature_schema.json",
                    "artifacts/sha256sums.txt",
                    "artifacts/verification.json",
                ),
            )
        )
        t3 = HeldPublishedT3(calibrated_root)
        stack.callback(t3.close)
        t3.verify()

        relative_files = {
            "managed_pass": "PASS",
            "managed_gate": "gate.json",
            "managed_verification": "verification.json",
            "neurosed_pass": "artifacts/PASS",
            "neurosed_model": "artifacts/best.pt",
            "neurosed_config": "artifacts/config.yaml",
            "neurosed_model_card": "artifacts/model_card.json",
            "neurosed_pair_manifest": "artifacts/pair_manifest.json",
            "neurosed_split_manifest": "artifacts/split_manifest.json",
            "neurosed_direction": "artifacts/distance_direction_trace.json",
            "neurosed_feature_schema": "artifacts/feature_schema.json",
            "neurosed_sha256s": "artifacts/sha256sums.txt",
            "neurosed_verification": "artifacts/verification.json",
        }
        held_files = {
            label: stack.enter_context(
                hold_readonly_file(neurosed_root / relative)
            )
            for label, relative in relative_files.items()
        }
        fixed = inspect_fixed_budget_neurosed_pass(
            neurosed_root / "artifacts",
            vendored_gcf_root=official_root,
            allow_managed_generation_token=True,
        )
        model_card = _json(
            held_files["neurosed_model_card"].read_bytes(),
            label="NeuroSED model card",
        )
        direction = _json(
            held_files["neurosed_direction"].read_bytes(),
            label="NeuroSED direction trace",
        )
        generic_verification = _json(
            held_files["managed_verification"].read_bytes(),
            label="managed NeuroSED verification",
        )
        domain = generic_verification.get("verification")
        consumer = domain.get("t7_consumer") if type(domain) is dict else None
        feature = _json(
            held_files["neurosed_feature_schema"].read_bytes(),
            label="NeuroSED feature schema",
        )
        if type(consumer) is not dict:
            raise TasteGCFReleaseError("managed NeuroSED T7 consumer is absent")
        validated_consumer = validate_t7_fixed_budget_consumer(
            consumer,
            checkpoint_sha256=held_files["neurosed_model"].sha256,
            feature_schema_sha256=held_files["neurosed_feature_schema"].sha256,
            sha256s_sha256=held_files["neurosed_sha256s"].sha256,
            feature_atomic_numbers=list(feature.get("feature_atomic_numbers", [])),
            feature_input_dim=int(feature.get("input_dim", -1)),
        )
        if validated_consumer != fixed["t7_consumer"]:
            raise TasteGCFReleaseError("managed NeuroSED adoption changed its T7 consumer")
        if (
            held_files["managed_pass"].read_bytes()
            != b"[MANAGED_EXECUTION_V2_PASS]\n"
            or held_files["neurosed_pass"].sha256 != fixed["pass_sha256"]
            or held_files["neurosed_model"].sha256 != fixed["checkpoint_sha256"]
            or model_card.get("official_gcf_commit") != OFFICIAL_GCF_COMMIT
            or model_card.get("official_greed_commit")
            != STRICT_OFFICIAL_PROVENANCE["greed_commit"]
            or model_card.get("calibration_loaded") is not False
            or model_card.get("test_loaded") is not False
            or model_card.get("gcf_runtime_direction") != SOURCE_DIRECTION
            or direction.get("direction") != SOURCE_DIRECTION
            or direction.get("reverse_direction_used") is not False
            or domain.get("schema_version") != ADOPTION_VERIFICATION_SCHEMA
            or domain.get("status") != "PASS"
            or domain.get("marker")
            != "[TASTE_NEUROSED_FIXED_BUDGET_MANAGED_ADOPTION_PASS]"
            or domain.get("scientific_artifact_modified") is not False
            or domain.get("source_root_copied_byte_for_byte") is not True
            or domain.get("source_fixed_budget_pass_reopened") is not True
            or domain.get("source_independent_verification_reopened") is not True
            or domain.get("managed_copy_independently_rehashed") is not True
            or domain.get("calibration_loaded") is not False
            or domain.get("test_loaded") is not False
            or domain.get("source_inventory_sha256")
            != fixed["inventory_sha256"]
            or domain.get("source_pass_sha256") != fixed["pass_sha256"]
            or domain.get("source_verification_sha256")
            != fixed["verification_sha256"]
        ):
            raise TasteGCFReleaseError("fixed-budget NeuroSED release semantics changed")

        split_bytes = t3.files[
            "artifacts/checkpoint/split_manifest.json"
        ].bytes()
        split, split_paths = _validate_split_manifest(split_bytes)
        split_files = {
            role: stack.enter_context(
                hold_readonly_file(
                    split_paths[role],
                    expected_sha256=split["files"][role]["sha256"],
                )
            )
            for role in _SPLITS
        }
        split_hashes = {role: split_files[role].sha256 for role in _SPLITS}
        checkpoint_names = (
            "model.pt",
            "model_card.json",
            "feature_schema.json",
            "label_map.json",
            "split_manifest.json",
            "test_evaluation_status.json",
            "temperature_scaling.json",
        )
        checkpoint_payloads = {
            name: t3.files[f"artifacts/checkpoint/{name}"].bytes()
            for name in checkpoint_names
        }
        train_manifest = split.get("train_manifest")
        if type(train_manifest) is not dict:
            raise TasteGCFReleaseError("T3 train manifest is absent")
        train_count = train_manifest.get("num_records")
        label_counts = train_manifest.get("label_counts")
        if type(train_count) is not int or train_count <= 0 or type(label_counts) is not dict:
            raise TasteGCFReleaseError("T3 train manifest changed")
        t3_binding = dict(t3.binding)
        official = verify_vendored_gcf_retained_inventory(official_root)
        official_inventory = _require_sha256(
            official.get("inventory_sha256"), label="vendored GCF inventory"
        )
        if official_inventory != model_card.get(
            "vendored_gcf_retained_inventory_sha256"
        ):
            raise TasteGCFReleaseError("NeuroSED model card does not bind vendored GCF")
        execution = inspect_clean_execution()
        managed_evidence = managed.revalidate()
        pins = TasteGCFReleasePinsV1(
            dataset=DATASET,
            source_label=SOURCE_LABEL,
            num_classes=NUM_CLASSES,
            official_gcf_commit=model_card["official_gcf_commit"],
            neurosed_commit=model_card["official_greed_commit"],
            neurosed_model_sha=held_files["neurosed_model"].sha256,
            neurosed_config_sha=held_files["neurosed_config"].sha256,
            neurosed_pair_manifest_sha=held_files["neurosed_pair_manifest"].sha256,
            t3_calibrated_gine_sha=t3_binding["model_sha256"],
            t3_temperature_sha=t3_binding["temperature_scaling_sha256"],
            dataset_sha=_dataset_sha(split_hashes),
            train_split_sha=split_hashes["train"],
            validation_split_sha=split_hashes["validation"],
            calibration_split_sha=split_hashes["calibration"],
            test_split_sha=split_hashes["test"],
            inference_direction=INFERENCE_DIRECTION,
            neurosed_calibration_loaded=False,
            neurosed_test_loaded=False,
        )
        pins.validate()
        authority = TasteGCFSourceAuthorityV1(
            managed_neurosed_root=str(neurosed_root),
            managed_neurosed_pass_sha256=managed_evidence["pass_sha256"],
            managed_neurosed_gate_sha256=managed_evidence["gate_sha256"],
            managed_neurosed_verification_sha256=managed_evidence[
                "verification_sha256"
            ],
            managed_neurosed_inventory_sha256=managed_evidence[
                "published_inventory_sha256"
            ],
            neurosed_model_path=str(neurosed_root / "artifacts/best.pt"),
            neurosed_config_path=str(neurosed_root / "artifacts/config.yaml"),
            neurosed_pair_manifest_path=str(
                neurosed_root / "artifacts/pair_manifest.json"
            ),
            neurosed_feature_schema_path=str(
                neurosed_root / "artifacts/feature_schema.json"
            ),
            neurosed_sha256s_path=str(
                neurosed_root / "artifacts/sha256sums.txt"
            ),
            t3_root=str(calibrated_root),
            t3_gate_sha256=t3_binding["t3_gate_sha256"],
            t3_verification_sha256=t3_binding["t3_verification_sha256"],
            t3_root_inventory_sha256=t3_binding["t3_root_inventory_sha256"],
            t3_checkpoint_dir=t3_binding["checkpoint_dir"],
            t3_checkpoint_id=t3_binding["checkpoint_id"],
            t3_split_manifest_sha256=_sha256_bytes(split_bytes),
            t3_feature_schema_sha256=t3_binding["feature_schema_file_sha256"],
            split_paths=split_paths,
            official_gcf_root=str(official_root),
            official_gcf_inventory_sha256=official_inventory,
            neurosed_distance_threshold=float(neurosed_distance_threshold),
            implementation_commit=execution["commit"],
            implementation_tree=execution["tree"],
            no_neurosed_retraining=True,
            split_payloads_deserialized=False,
            test_payload_deserialized=False,
        )
        authority.validate()
        # This is the exact predecessor shape consumed by the already-tested
        # native VRRW bridge.  It is reconstructed from the adopted managed
        # final; no NeuroSED training or model-byte write occurs here.
        neurosed_evidence = {
            "schema_version": "tastemolnet_gcf_neurosed_managed_final_v1",
            "status": "PASS",
            "marker": "MANAGED_EXECUTION_V2_PASS",
            "final_root": str(neurosed_root),
            "attempt_id": managed_evidence["attempt_id"],
            "generation_token": managed_evidence["generation_token"],
            "pass_path": str(neurosed_root / "PASS"),
            "pass_sha256": managed_evidence["pass_sha256"],
            "gate_path": str(neurosed_root / "gate.json"),
            "gate_sha256": managed_evidence["gate_sha256"],
            "verification_path": str(neurosed_root / "verification.json"),
            "verification_sha256": managed_evidence["verification_sha256"],
            "source_inventory_sha256": managed_evidence[
                "source_inventory_sha256"
            ],
            "published_inventory_sha256": managed_evidence[
                "published_inventory_sha256"
            ],
            "checkpoint_path": str(neurosed_root / "artifacts/best.pt"),
            "checkpoint_sha256": pins.neurosed_model_sha,
            "feature_schema_path": str(
                neurosed_root / "artifacts/feature_schema.json"
            ),
            "feature_schema_sha256": held_files[
                "neurosed_feature_schema"
            ].sha256,
            "sha256s_path": str(neurosed_root / "artifacts/sha256sums.txt"),
            "sha256s_sha256": held_files["neurosed_sha256s"].sha256,
            "t7_consumer": dict(validated_consumer),
        }
        result = HeldTasteGCFReleaseSourcesV1(
            stack=stack,
            managed_neurosed=managed,
            t3=t3,
            held_files=held_files,
            split_files=split_files,
            pins=pins,
            authority=authority,
            checkpoint_payloads=checkpoint_payloads,
            train_bytes=split_files["train"].read_bytes(),
            train_contract={
                "path": split_paths["train"],
                "sha256": split_hashes["train"],
                "num_records": train_count,
                "label_counts": label_counts,
            },
            neurosed_evidence=neurosed_evidence,
            official_root=official_root,
            neurosed_model=held_files["neurosed_model"],
        )
        result.revalidate()
        return result
    except BaseException:
        stack.close()
        raise


CANDIDATE_FILES = frozenset(
    {
        "release_pins.json",
        "source_authority.json",
        "candidate.json",
        "sha256sums.txt",
        "READY",
    }
)
FINAL_FILES = frozenset(
    {
        "release_pins.json",
        "source_authority.json",
        "candidate_binding.json",
        "verification.json",
        "gate.json",
        "sha256sums.txt",
        "PASS",
    }
)


def _fresh_directory(path: Path) -> None:
    _absolute(path.parent, label=f"{path.name} parent")
    if path.exists() or path.is_symlink():
        raise TasteGCFReleaseError(f"fresh path already exists: {path}")
    os.mkdir(path, mode=0o700)
    _fsync_directory(path.parent)


def _write_sha256s(root: Path, names: tuple[str, ...]) -> bytes:
    rows = b"".join(
        f"{_sha256_file(root / name)}  {name}\n".encode("ascii")
        for name in names
    )
    _write_new(root / "sha256sums.txt", rows)
    return rows


def _parse_sha256s(data: bytes, *, expected: set[str]) -> dict[str, str]:
    try:
        lines = data.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise TasteGCFReleaseError("T7 SHA256SUMS is not ASCII") from exc
    result: dict[str, str] = {}
    for line in lines:
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or _SHA256.fullmatch(digest) is None
            or Path(relative).name != relative
            or relative in result
        ):
            raise TasteGCFReleaseError("T7 SHA256SUMS is malformed")
        result[relative] = digest
    if set(result) != expected:
        raise TasteGCFReleaseError("T7 SHA256SUMS inventory changed")
    return result


def build_t7_release_candidate(
    *,
    managed_neurosed_root: str | Path,
    t3_root: str | Path,
    official_gcf_root: str | Path,
    neurosed_distance_threshold: float,
    output_root: str | Path,
) -> dict[str, Any]:
    """Write non-terminal typed pins from real predecessor authorities."""

    destination = _absolute(output_root, label="T7 candidate root", must_exist=False)
    with hold_t7_release_sources(
        managed_neurosed_root=managed_neurosed_root,
        t3_root=t3_root,
        official_gcf_root=official_gcf_root,
        neurosed_distance_threshold=neurosed_distance_threshold,
    ) as sources:
        pins = sources.pins.mapping()
        authority = sources.authority.mapping()
        release_id = str(uuid.uuid4())
        candidate = {
            "schema_version": CANDIDATE_SCHEMA,
            "status": "READY_FOR_INDEPENDENT_VERIFICATION",
            "release_id": release_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "release_pins_sha256": _sha256_mapping(pins),
            "source_authority_sha256": _sha256_mapping(authority),
            "neurosed_retrained": False,
            "scientific_model_bytes_written": False,
            "split_payloads_deserialized": False,
            "test_payload_deserialized": False,
            "worker_wrote_pass": False,
            "independent_verification_required": True,
        }
        _fresh_directory(destination)
        try:
            _write_new(destination / "release_pins.json", _json_bytes(pins))
            _write_new(
                destination / "source_authority.json", _json_bytes(authority)
            )
            _write_new(destination / "candidate.json", _json_bytes(candidate))
            _write_new(destination / "READY", (READY_MARKER + "\n").encode("ascii"))
            _write_sha256s(
                destination,
                (
                    "release_pins.json",
                    "source_authority.json",
                    "candidate.json",
                    "READY",
                ),
            )
            _fsync_directory(destination)
            sources.revalidate()
        except BaseException:
            shutil.rmtree(destination, ignore_errors=True)
            raise
    return {
        "status": "READY_FOR_INDEPENDENT_VERIFICATION",
        "release_id": release_id,
        "candidate_root": str(destination),
        "release_pins_sha256": candidate["release_pins_sha256"],
        "source_authority_sha256": candidate["source_authority_sha256"],
        "neurosed_retrained": False,
    }


def _load_candidate(
    root: Path,
) -> tuple[TasteGCFReleasePinsV1, TasteGCFSourceAuthorityV1, dict[str, Any], str]:
    if root.resolve(strict=True) != root or root.is_symlink() or not root.is_dir():
        raise TasteGCFReleaseError("T7 candidate root must be physical")
    names = {entry.name for entry in os.scandir(root)}
    if names != set(CANDIDATE_FILES):
        raise TasteGCFReleaseError("T7 candidate inventory changed")
    hashes = _parse_sha256s(
        (root / "sha256sums.txt").read_bytes(),
        expected=set(CANDIDATE_FILES) - {"sha256sums.txt"},
    )
    for name, digest in hashes.items():
        if _sha256_file(root / name) != digest:
            raise TasteGCFReleaseError(f"T7 candidate file changed: {name}")
    if (root / "READY").read_bytes() != (READY_MARKER + "\n").encode("ascii"):
        raise TasteGCFReleaseError("T7 candidate READY marker changed")
    pins = TasteGCFReleasePinsV1.from_mapping(
        _read_json_file(root / "release_pins.json", label="candidate pins")
    )
    authority = TasteGCFSourceAuthorityV1.from_mapping(
        _read_json_file(root / "source_authority.json", label="candidate authority")
    )
    candidate = _read_json_file(root / "candidate.json", label="candidate")
    _exact_keys(
        candidate,
        {
            "schema_version",
            "status",
            "release_id",
            "created_at",
            "release_pins_sha256",
            "source_authority_sha256",
            "neurosed_retrained",
            "scientific_model_bytes_written",
            "split_payloads_deserialized",
            "test_payload_deserialized",
            "worker_wrote_pass",
            "independent_verification_required",
        },
        label="T7 release candidate",
    )
    try:
        parsed_id = uuid.UUID(str(candidate.get("release_id")))
    except (ValueError, AttributeError) as exc:
        raise TasteGCFReleaseError("T7 release ID is not UUIDv4") from exc
    if (
        candidate.get("schema_version") != CANDIDATE_SCHEMA
        or candidate.get("status") != "READY_FOR_INDEPENDENT_VERIFICATION"
        or parsed_id.version != 4
        or str(parsed_id) != candidate.get("release_id")
        or type(candidate.get("created_at")) is not str
        or not candidate["created_at"]
        or candidate.get("release_pins_sha256") != pins.sha256
        or candidate.get("source_authority_sha256") != authority.sha256
        or candidate.get("neurosed_retrained") is not False
        or candidate.get("scientific_model_bytes_written") is not False
        or candidate.get("split_payloads_deserialized") is not False
        or candidate.get("test_payload_deserialized") is not False
        or candidate.get("worker_wrote_pass") is not False
        or candidate.get("independent_verification_required") is not True
    ):
        raise TasteGCFReleaseError("T7 release candidate semantics changed")
    inventory_sha = _sha256_mapping(
        {"files": {name: hashes[name] for name in sorted(hashes)}}
    )
    return pins, authority, candidate, inventory_sha


def verify_and_publish_t7_release(
    *,
    candidate_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Independently reopen source bytes and publish a fresh typed release."""

    candidate_path = _absolute(candidate_root, label="T7 candidate root")
    destination = _absolute(output_root, label="T7 release root", must_exist=False)
    pins, authority, candidate, candidate_inventory_sha = _load_candidate(
        candidate_path
    )
    with hold_t7_release_sources(
        managed_neurosed_root=authority.managed_neurosed_root,
        t3_root=authority.t3_root,
        official_gcf_root=authority.official_gcf_root,
        neurosed_distance_threshold=authority.neurosed_distance_threshold,
    ) as sources:
        if sources.pins.mapping() != pins.mapping():
            raise TasteGCFReleaseError("independent T7 pin replay differs")
        if sources.authority.mapping() != authority.mapping():
            raise TasteGCFReleaseError("independent T7 authority replay differs")
        verification = {
            "schema_version": VERIFICATION_SCHEMA,
            "status": "PASS",
            "release_id": candidate["release_id"],
            "independent_verifier": True,
            "candidate_root": str(candidate_path),
            "candidate_inventory_sha256": candidate_inventory_sha,
            "release_pins_sha256": pins.sha256,
            "source_authority_sha256": authority.sha256,
            "managed_neurosed_reopened": True,
            "managed_t3_reopened": True,
            "all_split_files_sha256_reopened": True,
            "split_payloads_deserialized": False,
            "test_payload_deserialized": False,
            "neurosed_retrained": False,
            "generated_to_original_verified": True,
            "same_three_class_gine_verified": True,
            "calibration_loaded": False,
            "test_loaded": False,
            "rf_oracle_used": False,
        }
        gate = {
            "schema_version": GATE_SCHEMA,
            "status": "PASS",
            "marker": PASS_MARKER,
            "release_id": candidate["release_id"],
            "independent_verifier": True,
            "downstream_released": True,
            "release_pins_sha256": pins.sha256,
            "source_authority_sha256": authority.sha256,
            "verification_sha256": _sha256_mapping(verification),
            "no_neurosed_retraining": True,
            "gpu_index": 0,
            "terminal_pass_written_last": True,
        }
        parent = _absolute(destination.parent, label="T7 release parent")
        if destination.exists() or destination.is_symlink():
            raise TasteGCFReleaseError("T7 release output must be fresh")
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.tmp.", dir=parent)
        )
        os.chmod(temporary, 0o700)
        try:
            _write_new(
                temporary / "release_pins.json", _json_bytes(pins.mapping())
            )
            _write_new(
                temporary / "source_authority.json",
                _json_bytes(authority.mapping()),
            )
            _write_new(
                temporary / "candidate_binding.json",
                _json_bytes(
                    {
                        "schema_version": "taste_gcf_release_candidate_binding_v1",
                        "release_id": candidate["release_id"],
                        "candidate_root": str(candidate_path),
                        "candidate_inventory_sha256": candidate_inventory_sha,
                    }
                ),
            )
            _write_new(
                temporary / "verification.json", _json_bytes(verification)
            )
            _write_new(temporary / "gate.json", _json_bytes(gate))
            _write_sha256s(
                temporary,
                (
                    "release_pins.json",
                    "source_authority.json",
                    "candidate_binding.json",
                    "verification.json",
                    "gate.json",
                ),
            )
            _fsync_directory(temporary)
            sources.revalidate()
            parent_fd = os.open(
                parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                _atomic_rename_noreplace(
                    source_parent_descriptor=parent_fd,
                    source_name=temporary.name,
                    destination_parent_descriptor=parent_fd,
                    destination_name=destination.name,
                )
                _fsync_directory(parent)
            finally:
                os.close(parent_fd)
            _write_new(destination / "PASS", (PASS_MARKER + "\n").encode("ascii"))
            _fsync_directory(destination)
            _fsync_directory(parent)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    validated = validate_t7_release_root(destination, reopen_sources=False)
    return {
        "status": "PASS",
        "marker": PASS_MARKER,
        "release_id": candidate["release_id"],
        "release_root": str(destination),
        "release_pins_sha256": pins.sha256,
        "source_authority_sha256": authority.sha256,
        "verification_sha256": validated["verification_sha256"],
        "gate_sha256": validated["gate_sha256"],
        "neurosed_retrained": False,
    }


def validate_t7_release_root(
    root: str | Path,
    *,
    reopen_sources: bool = True,
) -> dict[str, Any]:
    """Validate the terminal release; optionally replay every external source."""

    release_root = _absolute(root, label="T7 typed release root")
    names = {entry.name for entry in os.scandir(release_root)}
    if names != set(FINAL_FILES):
        raise TasteGCFReleaseError("T7 typed release inventory changed")
    if (release_root / "PASS").read_bytes() != (PASS_MARKER + "\n").encode("ascii"):
        raise TasteGCFReleaseError("T7 typed release PASS changed")
    hashes = _parse_sha256s(
        (release_root / "sha256sums.txt").read_bytes(),
        expected=set(FINAL_FILES) - {"sha256sums.txt", "PASS"},
    )
    for name, digest in hashes.items():
        if _sha256_file(release_root / name) != digest:
            raise TasteGCFReleaseError(f"T7 typed release file changed: {name}")
    pins = TasteGCFReleasePinsV1.from_mapping(
        _read_json_file(release_root / "release_pins.json", label="release pins")
    )
    authority = TasteGCFSourceAuthorityV1.from_mapping(
        _read_json_file(
            release_root / "source_authority.json", label="release authority"
        )
    )
    candidate = _read_json_file(
        release_root / "candidate_binding.json", label="candidate binding"
    )
    verification = _read_json_file(
        release_root / "verification.json", label="release verification"
    )
    gate = _read_json_file(release_root / "gate.json", label="release gate")
    _exact_keys(
        candidate,
        {
            "schema_version", "release_id", "candidate_root",
            "candidate_inventory_sha256",
        },
        label="T7 candidate binding",
    )
    _exact_keys(
        verification,
        {
            "schema_version", "status", "release_id", "independent_verifier",
            "candidate_root", "candidate_inventory_sha256",
            "release_pins_sha256", "source_authority_sha256",
            "managed_neurosed_reopened", "managed_t3_reopened",
            "all_split_files_sha256_reopened", "split_payloads_deserialized",
            "test_payload_deserialized", "neurosed_retrained",
            "generated_to_original_verified", "same_three_class_gine_verified",
            "calibration_loaded", "test_loaded", "rf_oracle_used",
        },
        label="T7 release verification",
    )
    _exact_keys(
        gate,
        {
            "schema_version", "status", "marker", "release_id",
            "independent_verifier", "downstream_released",
            "release_pins_sha256", "source_authority_sha256",
            "verification_sha256", "no_neurosed_retraining", "gpu_index",
            "terminal_pass_written_last",
        },
        label="T7 release gate",
    )
    if (
        candidate.get("schema_version")
        != "taste_gcf_release_candidate_binding_v1"
        or type(candidate.get("release_id")) is not str
        or not candidate["release_id"]
        or type(candidate.get("candidate_root")) is not str
        or not candidate["candidate_root"]
        or _SHA256.fullmatch(
            str(candidate.get("candidate_inventory_sha256"))
        ) is None
        or verification.get("schema_version") != VERIFICATION_SCHEMA
        or verification.get("status") != "PASS"
        or verification.get("independent_verifier") is not True
        or verification.get("release_id") != candidate.get("release_id")
        or verification.get("candidate_root") != candidate.get("candidate_root")
        or verification.get("candidate_inventory_sha256")
        != candidate.get("candidate_inventory_sha256")
        or verification.get("release_pins_sha256") != pins.sha256
        or verification.get("source_authority_sha256") != authority.sha256
        or verification.get("managed_neurosed_reopened") is not True
        or verification.get("managed_t3_reopened") is not True
        or verification.get("all_split_files_sha256_reopened") is not True
        or verification.get("split_payloads_deserialized") is not False
        or verification.get("test_payload_deserialized") is not False
        or verification.get("neurosed_retrained") is not False
        or verification.get("generated_to_original_verified") is not True
        or verification.get("same_three_class_gine_verified") is not True
        or verification.get("calibration_loaded") is not False
        or verification.get("test_loaded") is not False
        or verification.get("rf_oracle_used") is not False
        or gate.get("schema_version") != GATE_SCHEMA
        or gate.get("status") != "PASS"
        or gate.get("marker") != PASS_MARKER
        or gate.get("release_id") != verification.get("release_id")
        or gate.get("independent_verifier") is not True
        or gate.get("downstream_released") is not True
        or gate.get("release_pins_sha256") != pins.sha256
        or gate.get("source_authority_sha256") != authority.sha256
        or gate.get("verification_sha256") != _sha256_mapping(verification)
        or gate.get("no_neurosed_retraining") is not True
        or gate.get("gpu_index") != 0
        or gate.get("terminal_pass_written_last") is not True
        or candidate.get("release_id") != gate.get("release_id")
    ):
        raise TasteGCFReleaseError("T7 typed release cross-binding changed")
    if reopen_sources:
        with hold_t7_release_sources(
            managed_neurosed_root=authority.managed_neurosed_root,
            t3_root=authority.t3_root,
            official_gcf_root=authority.official_gcf_root,
            neurosed_distance_threshold=authority.neurosed_distance_threshold,
        ) as sources:
            if sources.pins.mapping() != pins.mapping():
                raise TasteGCFReleaseError("T7 released pins differ from sources")
            if sources.authority.mapping() != authority.mapping():
                raise TasteGCFReleaseError("T7 released authority differs from sources")
    return {
        "status": "PASS",
        "marker": PASS_MARKER,
        "release_root": str(release_root),
        "release_id": gate["release_id"],
        "release_pins": pins.mapping(),
        "release_pins_sha256": pins.sha256,
        "source_authority": authority.mapping(),
        "source_authority_sha256": authority.sha256,
        "verification_sha256": hashes["verification.json"],
        "gate_sha256": hashes["gate.json"],
        "pass_sha256": _sha256_file(release_root / "PASS"),
    }


@dataclass(slots=True)
class HeldVerifiedT7ReleaseV1:
    stack: ExitStack
    release_root: Path
    release_files: Mapping[str, Any]
    sources: HeldTasteGCFReleaseSourcesV1
    evidence: Mapping[str, Any]

    @property
    def pins(self) -> TasteGCFReleasePinsV1:
        return self.sources.pins

    @property
    def authority(self) -> TasteGCFSourceAuthorityV1:
        return self.sources.authority

    def revalidate(self) -> Mapping[str, Any]:
        for item in self.release_files.values():
            item.revalidate()
        current = validate_t7_release_root(self.release_root, reopen_sources=False)
        if current != dict(self.evidence):
            raise TasteGCFReleaseError("T7 typed release changed while held")
        self.sources.revalidate()
        return current

    def close(self) -> None:
        self.stack.close()

    def __enter__(self) -> "HeldVerifiedT7ReleaseV1":
        self.revalidate()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def hold_verified_t7_release(root: str | Path) -> HeldVerifiedT7ReleaseV1:
    """Hold terminal typed evidence and every external science input."""

    release_root = _absolute(root, label="T7 typed release root")
    evidence = validate_t7_release_root(release_root, reopen_sources=False)
    authority = TasteGCFSourceAuthorityV1.from_mapping(
        evidence["source_authority"]
    )
    pins = TasteGCFReleasePinsV1.from_mapping(evidence["release_pins"])
    stack = ExitStack()
    try:
        release_files = {
            name: stack.enter_context(hold_readonly_file(release_root / name))
            for name in FINAL_FILES
        }
        sources = hold_t7_release_sources(
            managed_neurosed_root=authority.managed_neurosed_root,
            t3_root=authority.t3_root,
            official_gcf_root=authority.official_gcf_root,
            neurosed_distance_threshold=authority.neurosed_distance_threshold,
        )
        stack.callback(sources.close)
        if sources.pins.mapping() != pins.mapping():
            raise TasteGCFReleaseError("held T7 pins differ from live sources")
        if sources.authority.mapping() != authority.mapping():
            raise TasteGCFReleaseError("held T7 authority differs from live sources")
        result = HeldVerifiedT7ReleaseV1(
            stack=stack,
            release_root=release_root,
            release_files=release_files,
            sources=sources,
            evidence=evidence,
        )
        result.revalidate()
        return result
    except BaseException:
        stack.close()
        raise
