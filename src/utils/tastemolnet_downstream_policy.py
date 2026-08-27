"""Exact supplemental authority for TasteMolNet T3/T4 downstream science.

The base TasteMolNet policy authorizes private research and aggregate paper
reporting but freezes the T2 trainer to train/validation payloads.  This module
adds a narrower, independently hashed authority for adopting the temperature
already fitted by T2, a bounded calibration-cache oracle smoke, and a future
train-only T6 Ours smoke using the frozen GINE reward. It never authorizes a
refit, RF fallback, validation/calibration/test access in T6, or redistribution.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from src.utils.tastemolnet_research_policy import (
    POLICY_ID as BASE_POLICY_ID,
    POLICY_SCHEMA as BASE_POLICY_SCHEMA,
    TasteResearchPolicy,
    TasteResearchPolicyError,
    parse_tastemolnet_research_policy,
    stable_json_sha256,
)


DOWNSTREAM_POLICY_SCHEMA = "tastemolnet_downstream_research_policy_v1"
DOWNSTREAM_POLICY_ID = "tastemolnet-downstream-research-no-redistribution-v1-20260828"
DOWNSTREAM_POLICY_VERSION = 1
BASE_POLICY_FILE_SHA256 = "b370ed9655f0a566b3615fc321c547945dd73fcee27d637110b801a766e1ca1b"
BASE_POLICY_CANONICAL_SHA256 = "422e50a0d613c47dd451d9386c7dc4707a079087ebe62b3d72e6a6784dd1da3d"
# Filled after the tracked JSON is finalized. Loading fails closed on drift.
DOWNSTREAM_POLICY_FILE_SHA256 = "0939ce2c016ff840e0f9fe7db65a397185b8acd8089b4fb62f1c7bbd5519b77c"
EXECUTION_SOURCE_ROOT = Path(__file__).resolve(strict=True).parents[2]
TRACKED_DOWNSTREAM_POLICY_PATH = (
    EXECUTION_SOURCE_ROOT
    / "configs/data_usage/tastemolnet_downstream_research_no_redistribution_v1.json"
)
TRACKED_BASE_POLICY_PATH = (
    EXECUTION_SOURCE_ROOT
    / "configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml"
)


class TasteDownstreamPolicyError(TasteResearchPolicyError):
    """The supplemental downstream authority changed or was used out of scope."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _exact(actual: Any, expected: Any, *, field: str) -> None:
    """Compare JSON values without Python bool/int/float coercions."""

    if type(actual) is not type(expected):
        raise TasteDownstreamPolicyError(f"{field} changed native JSON type")
    if type(expected) is dict:
        if any(type(key) is not str for key in actual):
            raise TasteDownstreamPolicyError(f"{field} contains a non-string key")
        if set(actual) != set(expected):
            raise TasteDownstreamPolicyError(f"{field} keys changed")
        for key in expected:
            _exact(actual[key], expected[key], field=f"{field}.{key}")
        return
    if type(expected) is list:
        if len(actual) != len(expected):
            raise TasteDownstreamPolicyError(f"{field} length changed")
        for index, (observed, wanted) in enumerate(zip(actual, expected, strict=True)):
            _exact(observed, wanted, field=f"{field}[{index}]")
        return
    if type(expected) is float and not math.isfinite(actual):
        raise TasteDownstreamPolicyError(f"{field} must be finite")
    if actual != expected:
        raise TasteDownstreamPolicyError(f"{field} changed value")


def _expected_payload() -> dict[str, Any]:
    return {
        "authorization_basis": "explicit_project_owner_instruction",
        "base_policy": {
            "policy_canonical_sha256": BASE_POLICY_CANONICAL_SHA256,
            "policy_file_sha256": BASE_POLICY_FILE_SHA256,
            "policy_id": BASE_POLICY_ID,
            "policy_version": 2,
            "schema_version": BASE_POLICY_SCHEMA,
        },
        "classifier_contract": {
            "backbone": "gine",
            "label_map": {"0": "Bitter", "1": "Sweet", "2": "Tasteless"},
            "num_classes": 3,
            "oracle_backend": "gnn",
            "rf_oracle_used": False,
            "source_label": 1,
            "strict_flip": "pred_before == 1 and pred_after != 1",
        },
        "data_handling": {
            "checkpoint_copy_allowed": False,
            "checkpoint_mutation_allowed": False,
            "data_preparation_allowed": False,
            "graph_cache_rebuild_allowed": False,
            "network_download_allowed": False,
            "public_full_split_output_allowed": False,
            "public_molecule_level_output_allowed": False,
            "source_copy_allowed": False,
            "temperature_refit_allowed": False,
        },
        "dataset": "tastemolnet",
        "effective_date": "2026-08-28",
        "execution": {
            "hpc_execution_allowed": False,
            "platform": "autodl",
            "stages": {
                "T3_GINE_CALIBRATED": {
                    "allowed_input_files": ["immutable_t2_bundle"],
                    "device": "cpu",
                    "fresh_output_required": True,
                    "gpu_uuid_binding_required": False,
                    "max_deletions_per_parent": 0,
                    "mode": "adopt_existing_validation_fit",
                    "num_classes": 3,
                    "physical_gpu_index": None,
                    "run": 1,
                    "source_count": 0,
                    "source_label": 1,
                    "split_payload_access": {
                        "calibration": False,
                        "test": False,
                        "train": False,
                        "validation": False,
                    },
                },
                "T4_ORACLE_SMOKE": {
                    "allowed_input_files": [
                        "immutable_t2_bundle",
                        "graph_cache_manifest",
                        "calibration.pt",
                    ],
                    "device": "cuda:0",
                    "fresh_output_required": True,
                    "gpu_uuid_binding_required": True,
                    "max_deletions_per_parent": 4,
                    "minimum_deletions_per_parent": 4,
                    "mode": "calibration_cache_only_bounded_oracle_smoke",
                    "num_classes": 3,
                    "physical_gpu_index": 1,
                    "run": 1,
                    "source_count": 16,
                    "source_label": 1,
                    "split_payload_access": {
                        "calibration": True,
                        "test": False,
                        "train": False,
                        "validation": False,
                    },
                },
                "T6_OURS_SMOKE": {
                    "allowed_input_files": [
                        "immutable_t2_bundle",
                        "immutable_t3_stage_output",
                        "immutable_t4_stage_output",
                        "immutable_t5_clean_policy",
                        "frozen_train_csv",
                    ],
                    "device": "cuda:0",
                    "fresh_output_required": True,
                    "frozen_gine_reward_required": True,
                    "gpu_uuid_binding_required": True,
                    "minimum_optimizer_steps": 5,
                    "mode": "train_only_frozen_gine_reward_ppo_smoke",
                    "num_classes": 3,
                    "physical_gpu_index": 1,
                    "rf_oracle_used": False,
                    "run": 1,
                    "source_label": 1,
                    "split_payload_access": {
                        "calibration": False,
                        "test": False,
                        "train": True,
                        "validation": False,
                    },
                },
            },
        },
        "permissions": {
            "aggregate_metrics_release_allowed": True,
            "data_redistribution_allowed": False,
            "paper_result_reporting_allowed": True,
            "per_example_public_reporting_allowed": False,
            "research_compute_allowed": True,
            "trained_model_release_allowed": "review_required",
        },
        "policy_id": DOWNSTREAM_POLICY_ID,
        "policy_version": DOWNSTREAM_POLICY_VERSION,
        "schema_version": DOWNSTREAM_POLICY_SCHEMA,
    }


def _stat_identity(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "nlink": int(value.st_nlink),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


@dataclass(slots=True)
class _HeldDirectory:
    path: Path
    name: str | None
    descriptor: int
    evidence: dict[str, int]


@dataclass(slots=True)
class PhysicalPolicyAuthority:
    """One exact tracked file plus its retained root-to-leaf directory chain."""

    path: Path
    descriptor: int
    evidence: dict[str, int]
    file_sha256: str
    directories: list[_HeldDirectory]

    def _read_bytes(self) -> bytes:
        chunks: list[bytes] = []
        offset = 0
        while True:
            if hasattr(os, "pread"):
                chunk = os.pread(self.descriptor, 1024 * 1024, offset)
            else:  # pragma: no cover - AutoDL/Linux and macOS both expose pread.
                os.lseek(self.descriptor, offset, os.SEEK_SET)
                chunk = os.read(self.descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
        return b"".join(chunks)

    def verify(self, *, label: str) -> bytes:
        for index, directory in enumerate(self.directories):
            held = os.fstat(directory.descriptor)
            current_identity = _stat_identity(held)
            keys = (
                tuple(directory.evidence)
                if index >= max(0, len(self.directories) - 2)
                else ("device", "inode", "mode")
            )
            if not stat.S_ISDIR(held.st_mode) or any(
                current_identity[key] != directory.evidence[key] for key in keys
            ):
                raise TasteDownstreamPolicyError(
                    f"{label} ancestor authority changed"
                )
            if index:
                parent = self.directories[index - 1]
                named = os.stat(
                    str(directory.name),
                    dir_fd=parent.descriptor,
                    follow_symlinks=False,
                )
                if _stat_identity(named) != _stat_identity(held):
                    raise TasteDownstreamPolicyError(
                        f"{label} ancestor name changed"
                    )
        held_file = os.fstat(self.descriptor)
        named_file = os.stat(
            self.path.name,
            dir_fd=self.directories[-1].descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(held_file.st_mode)
            or held_file.st_nlink != 1
            or _stat_identity(held_file) != self.evidence
            or _stat_identity(named_file) != self.evidence
        ):
            raise TasteDownstreamPolicyError(f"{label} file authority changed")
        data = self._read_bytes()
        if _sha256_bytes(data) != self.file_sha256:
            raise TasteDownstreamPolicyError(f"{label} bytes changed")
        return data

    def directory_identity_paths(self) -> dict[tuple[int, int], set[Path]]:
        result: dict[tuple[int, int], set[Path]] = {}
        for directory in self.directories:
            identity = (directory.evidence["device"], directory.evidence["inode"])
            result.setdefault(identity, set()).add(directory.path)
        return result

    def close(self) -> None:
        if self.descriptor >= 0:
            try:
                os.close(self.descriptor)
            except OSError:
                pass
            finally:
                self.descriptor = -1
        for directory in reversed(self.directories):
            if directory.descriptor >= 0:
                try:
                    os.close(directory.descriptor)
                except OSError:
                    pass
                finally:
                    directory.descriptor = -1

    def __del__(self) -> None:  # pragma: no cover - process-exit cleanup.
        self.close()


def _open_exact_tracked_file(
    path: str | Path,
    *,
    expected_path: Path,
    label: str,
) -> tuple[PhysicalPolicyAuthority, bytes]:
    source = Path(os.path.abspath(Path(path).expanduser()))
    if source != expected_path:
        raise TasteDownstreamPolicyError(
            f"{label} must be the exact tracked policy path"
        )
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    directories: list[_HeldDirectory] = []
    file_descriptor = -1
    pending_directory = -1
    try:
        pending_directory = os.open(source.anchor, directory_flags)
        root_info = os.fstat(pending_directory)
        directories.append(
            _HeldDirectory(
                path=Path(source.anchor),
                name=None,
                descriptor=pending_directory,
                evidence=_stat_identity(root_info),
            )
        )
        pending_directory = -1
        current = Path(source.anchor)
        for part in source.parts[1:-1]:
            pending_directory = os.open(
                part,
                directory_flags,
                dir_fd=directories[-1].descriptor,
            )
            held = os.fstat(pending_directory)
            named = os.stat(
                part,
                dir_fd=directories[-1].descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(held.st_mode)
                or _stat_identity(held) != _stat_identity(named)
            ):
                raise TasteDownstreamPolicyError(
                    f"{label} ancestor is not one physical directory"
                )
            current = current / part
            directories.append(
                _HeldDirectory(
                    path=current,
                    name=part,
                    descriptor=pending_directory,
                    evidence=_stat_identity(held),
                )
            )
            pending_directory = -1
        file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        file_descriptor = os.open(
            source.name,
            file_flags,
            dir_fd=directories[-1].descriptor,
        )
        held_file = os.fstat(file_descriptor)
        named_file = os.stat(
            source.name,
            dir_fd=directories[-1].descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(held_file.st_mode)
            or held_file.st_nlink != 1
            or _stat_identity(held_file) != _stat_identity(named_file)
        ):
            raise TasteDownstreamPolicyError(
                f"{label} must be one physical single-link file"
            )
        authority = PhysicalPolicyAuthority(
            path=source,
            descriptor=file_descriptor,
            evidence=_stat_identity(held_file),
            file_sha256="",
            directories=directories,
        )
        data = authority._read_bytes()
        authority.file_sha256 = _sha256_bytes(data)
        authority.verify(label=label)
        return authority, data
    except Exception:
        if pending_directory >= 0:
            try:
                os.close(pending_directory)
            except OSError:
                pass
        if file_descriptor >= 0:
            try:
                os.close(file_descriptor)
            except OSError:
                pass
        for directory in reversed(directories):
            try:
                os.close(directory.descriptor)
            except OSError:
                pass
        raise


@dataclass(frozen=True, slots=True)
class TasteDownstreamPolicy:
    path: Path
    file_sha256: str
    canonical_sha256: str
    payload: Mapping[str, Any]
    base_policy: TasteResearchPolicy
    downstream_authority: PhysicalPolicyAuthority
    base_authority: PhysicalPolicyAuthority

    def stage(self, stage: str) -> Mapping[str, Any]:
        stages = self.payload["execution"]["stages"]
        try:
            result = stages[stage]
        except KeyError as exc:
            raise TasteDownstreamPolicyError(
                f"downstream stage is not authorized: {stage}"
            ) from exc
        if not isinstance(result, Mapping):  # pragma: no cover - exact check owns this.
            raise TasteDownstreamPolicyError("downstream stage authority is not a mapping")
        return result

    def evidence(self, *, stage: str) -> dict[str, Any]:
        stage_contract = self.stage(stage)
        return {
            "schema_version": "tastemolnet_downstream_policy_binding_v1",
            "dataset": "tastemolnet",
            "stage": stage,
            "downstream_policy": {
                "path": str(self.path),
                "file_sha256": self.file_sha256,
                "canonical_sha256": self.canonical_sha256,
                "policy_id": DOWNSTREAM_POLICY_ID,
            },
            "base_policy": {
                "path": str(self.base_policy.path),
                "file_sha256": self.base_policy.file_sha256,
                "canonical_sha256": self.base_policy.canonical_sha256,
                "policy_id": BASE_POLICY_ID,
            },
            "stage_contract": dict(stage_contract),
            "research_compute_allowed": True,
            "paper_result_reporting_allowed": True,
            "data_redistribution_allowed": False,
            "dataset_redistributed": False,
            "public_molecule_level_output_allowed": False,
            "hpc_execution_allowed": False,
        }

    def verify_authorities(self) -> None:
        downstream_data = self.downstream_authority.verify(
            label="downstream policy"
        )
        base_data = self.base_authority.verify(label="base policy")
        if _sha256_bytes(downstream_data) != self.file_sha256:
            raise TasteDownstreamPolicyError("downstream policy authority changed")
        if _sha256_bytes(base_data) != self.base_policy.file_sha256:
            raise TasteDownstreamPolicyError("base policy authority changed")
        _exact(self.payload, _expected_payload(), field="downstream_policy")
        if stable_json_sha256(self.payload) != self.canonical_sha256:
            raise TasteDownstreamPolicyError("downstream policy memory image changed")
        if stable_json_sha256(self.base_policy.payload) != self.base_policy.canonical_sha256:
            raise TasteDownstreamPolicyError("base policy memory image changed")

    def revalidate(self, *, stage: str) -> dict[str, Any]:
        self.verify_authorities()
        return self.evidence(stage=stage)

    def protected_paths(self) -> tuple[Path, Path]:
        return (self.downstream_authority.path, self.base_authority.path)

    def directory_identity_paths(self) -> dict[tuple[int, int], set[Path]]:
        result: dict[tuple[int, int], set[Path]] = {}
        for authority in (self.downstream_authority, self.base_authority):
            for identity, paths in authority.directory_identity_paths().items():
                result.setdefault(identity, set()).update(paths)
        return result

    def close(self) -> None:
        self.downstream_authority.close()
        self.base_authority.close()

    def __enter__(self) -> TasteDownstreamPolicy:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - process-exit cleanup.
        self.close()


def load_tastemolnet_downstream_policy(
    path: str | Path,
    *,
    base_policy_path: str | Path,
    expected_file_sha256: str | None = None,
) -> TasteDownstreamPolicy:
    """Load exact supplemental authority and bind it to active policy v2."""

    downstream_authority: PhysicalPolicyAuthority | None = None
    base_authority: PhysicalPolicyAuthority | None = None
    try:
        downstream_authority, data = _open_exact_tracked_file(
            path,
            expected_path=TRACKED_DOWNSTREAM_POLICY_PATH,
            label="downstream policy",
        )
        observed_sha256 = _sha256_bytes(data)
        wanted_sha256 = (
            DOWNSTREAM_POLICY_FILE_SHA256
            if expected_file_sha256 is None
            else str(expected_file_sha256)
        )
        if observed_sha256 != wanted_sha256:
            raise TasteDownstreamPolicyError(
                "downstream policy raw SHA-256 differs from reviewed authority"
            )
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TasteDownstreamPolicyError(
                "downstream policy is not valid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise TasteDownstreamPolicyError(
                "downstream policy must be one JSON object"
            )
        _exact(payload, _expected_payload(), field="downstream_policy")

        base_authority, base_data = _open_exact_tracked_file(
            base_policy_path,
            expected_path=TRACKED_BASE_POLICY_PATH,
            label="base policy",
        )
        base = parse_tastemolnet_research_policy(
            base_data,
            source=base_authority.path,
            expected_file_sha256=BASE_POLICY_FILE_SHA256,
        )
        base.require_main_route()
        if base.canonical_sha256 != BASE_POLICY_CANONICAL_SHA256:
            raise TasteDownstreamPolicyError("base policy canonical SHA-256 changed")
        result = TasteDownstreamPolicy(
            path=downstream_authority.path,
            file_sha256=observed_sha256,
            canonical_sha256=stable_json_sha256(payload),
            payload=payload,
            base_policy=base,
            downstream_authority=downstream_authority,
            base_authority=base_authority,
        )
        result.verify_authorities()
        return result
    except Exception:
        if downstream_authority is not None:
            downstream_authority.close()
        if base_authority is not None:
            base_authority.close()
        raise


def hold_tastemolnet_downstream_policy(
    path: str | Path,
    *,
    base_policy_path: str | Path,
) -> TasteDownstreamPolicy:
    """Public retained policy authority for later Taste stages such as T6."""

    return load_tastemolnet_downstream_policy(
        path,
        base_policy_path=base_policy_path,
    )


__all__ = [
    "BASE_POLICY_CANONICAL_SHA256",
    "BASE_POLICY_FILE_SHA256",
    "DOWNSTREAM_POLICY_FILE_SHA256",
    "DOWNSTREAM_POLICY_ID",
    "DOWNSTREAM_POLICY_SCHEMA",
    "EXECUTION_SOURCE_ROOT",
    "TRACKED_BASE_POLICY_PATH",
    "TRACKED_DOWNSTREAM_POLICY_PATH",
    "PhysicalPolicyAuthority",
    "TasteDownstreamPolicy",
    "TasteDownstreamPolicyError",
    "load_tastemolnet_downstream_policy",
    "hold_tastemolnet_downstream_policy",
]
