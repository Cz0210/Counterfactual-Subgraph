"""Pinned ChemLLM model-scale registry and metadata-only snapshot contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping, Sequence

from .contracts import LLMAblationContractError, canonical_json_sha256, require_sha256


_REVISION = re.compile(r"^[0-9a-f]{40,64}$")

PRIMARY_SMALL_SCALE = "PRIMARY_SMALL_SCALE"
MAIN_REFERENCE = "MAIN_REFERENCE"
OPTIONAL_UPPER_BOUND = "OPTIONAL_UPPER_BOUND"

READY = "READY"
BLOCKED_MISSING_EXACT_REVISION = "BLOCKED_MISSING_EXACT_REVISION"
BLOCKED_MISSING_MAIN_REFERENCE = "BLOCKED_MISSING_MAIN_REFERENCE"


def require_exact_revision(value: object, *, field: str = "revision") -> str:
    revision = str(value or "").strip().lower()
    if revision == "main" or not _REVISION.fullmatch(revision):
        raise LLMAblationContractError(
            f"{field} must be one exact 40-64 character hexadecimal revision, not main"
        )
    return revision


@dataclass(frozen=True, slots=True)
class FileIdentity:
    path: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        if not self.path.strip() or self.path.startswith("/"):
            raise LLMAblationContractError("snapshot inventory paths must be relative")
        object.__setattr__(self, "sha256", require_sha256(self.sha256, field=self.path))
        if isinstance(self.size, bool) or self.size < 0:
            raise LLMAblationContractError("snapshot inventory size must be non-negative")


@dataclass(frozen=True, slots=True)
class ModelScaleEntry:
    key: str
    repository_id: str
    role: str
    run_enabled: bool
    download_weights: bool
    exact_revision: str | None
    status: str
    metadata_only: bool
    expected_dtype: str | None = None
    source_from_reference_contract: bool = False

    def __post_init__(self) -> None:
        if not self.key.strip() or not self.repository_id.strip():
            raise LLMAblationContractError("model registry key/source is required")
        if self.role not in {PRIMARY_SMALL_SCALE, MAIN_REFERENCE, OPTIONAL_UPPER_BOUND}:
            raise LLMAblationContractError(f"unsupported model role: {self.role}")
        if not isinstance(self.run_enabled, bool) or not isinstance(self.download_weights, bool):
            raise LLMAblationContractError("run/download flags must be booleans")
        if self.status == READY:
            object.__setattr__(
                self,
                "exact_revision",
                require_exact_revision(self.exact_revision, field=f"{self.key}.exact_revision"),
            )
        elif self.status not in {
            BLOCKED_MISSING_EXACT_REVISION,
            BLOCKED_MISSING_MAIN_REFERENCE,
        }:
            raise LLMAblationContractError(f"unsupported registry status: {self.status}")
        elif self.exact_revision not in (None, ""):
            raise LLMAblationContractError("blocked entry may not claim an exact revision")
        if self.role == OPTIONAL_UPPER_BOUND:
            if self.run_enabled or self.download_weights or not self.metadata_only:
                raise LLMAblationContractError(
                    "optional 20B upper bound must be metadata-only and disabled"
                )
        if self.role == MAIN_REFERENCE and not self.source_from_reference_contract:
            raise LLMAblationContractError("7B main entry must come from the BACE reference")

    @classmethod
    def from_mapping(cls, key: str, payload: Mapping[str, Any]) -> "ModelScaleEntry":
        if not isinstance(payload, Mapping):
            raise LLMAblationContractError(f"registry entry {key} must be an object")
        return cls(
            key=key,
            repository_id=str(payload.get("source") or ""),
            role=str(payload.get("role") or ""),
            run_enabled=payload.get("run_enabled"),
            download_weights=payload.get("download_weights"),
            exact_revision=payload.get("exact_revision"),
            status=str(payload.get("status") or ""),
            metadata_only=bool(payload.get("metadata_only", False)),
            expected_dtype=(
                str(payload["expected_dtype"])
                if payload.get("expected_dtype") is not None
                else None
            ),
            source_from_reference_contract=bool(
                payload.get("source_from_reference_contract", False)
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelSnapshotManifest:
    """Hash-closed metadata for one exact Hugging Face revision.

    The same schema is used for downloaded 2B snapshots and the 20B
    metadata-only record.  A metadata-only record must have no weight files.
    """

    repository_id: str
    exact_revision: str
    commit_date: str
    model_card: FileIdentity
    config: FileIdentity
    tokenizer_files: tuple[FileIdentity, ...]
    remote_code_files: tuple[FileIdentity, ...]
    weight_files: tuple[FileIdentity, ...]
    license_metadata: Mapping[str, Any]
    weights_downloaded: bool
    metadata_only: bool
    trust_remote_code: bool
    isolated_import_pass: bool
    total_parameters: int | None = None
    schema_version: str = "chemllm_model_snapshot_manifest_v1"

    def __post_init__(self) -> None:
        if not self.repository_id.strip() or not self.commit_date.strip():
            raise LLMAblationContractError("snapshot repository/commit date is required")
        object.__setattr__(self, "exact_revision", require_exact_revision(self.exact_revision))
        for field in ("tokenizer_files", "remote_code_files", "weight_files"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        object.__setattr__(self, "license_metadata", dict(self.license_metadata))
        if not self.tokenizer_files:
            raise LLMAblationContractError("tokenizer inventory may not be empty")
        if self.metadata_only:
            if self.weights_downloaded or self.weight_files:
                raise LLMAblationContractError("metadata-only record may not contain weights")
        elif not self.weights_downloaded or not self.weight_files:
            raise LLMAblationContractError("downloaded snapshot must inventory weight files")
        if self.trust_remote_code and (
            not self.remote_code_files or not self.isolated_import_pass
        ):
            raise LLMAblationContractError(
                "trust_remote_code requires pinned source inventory and isolated import PASS"
            )
        if self.total_parameters is not None and self.total_parameters <= 0:
            raise LLMAblationContractError("actual total parameter count must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["snapshot_manifest_sha256"] = canonical_json_sha256(payload)
        return payload


def load_model_scale_registry(payload: Mapping[str, Any]) -> dict[str, ModelScaleEntry]:
    if payload.get("schema_version") != "chemllm_model_scale_registry_v2":
        raise LLMAblationContractError("unsupported ChemLLM model registry schema")
    if payload.get("parameter_count_source") != "ACTUAL_LOADED_WEIGHTS":
        raise LLMAblationContractError("parameter counts must come from actual loaded weights")
    entries = payload.get("models")
    if not isinstance(entries, Mapping):
        raise LLMAblationContractError("model registry has no models mapping")
    expected = {"chemllm_2b_1_5", "chemllm_7b_main", "chemllm_20b_sft"}
    if set(entries) != expected:
        raise LLMAblationContractError("model registry must contain exact 2B/7B/20B keys")
    parsed = {key: ModelScaleEntry.from_mapping(key, value) for key, value in entries.items()}
    if parsed["chemllm_20b_sft"].repository_id != "AI4Chem/ChemLLM-20B-Chat-SFT":
        raise LLMAblationContractError("20B metadata-only model ID changed")
    if parsed["chemllm_2b_1_5"].repository_id != "AI4Chem/CHEMLLM-2b-1_5":
        raise LLMAblationContractError("2B model ID changed")
    return parsed


def inventory_sha256(files: Sequence[FileIdentity]) -> str:
    return canonical_json_sha256({"files": [asdict(item) for item in files]})


__all__ = [
    "BLOCKED_MISSING_EXACT_REVISION",
    "BLOCKED_MISSING_MAIN_REFERENCE",
    "FileIdentity",
    "MAIN_REFERENCE",
    "ModelScaleEntry",
    "ModelSnapshotManifest",
    "OPTIONAL_UPPER_BOUND",
    "PRIMARY_SMALL_SCALE",
    "READY",
    "inventory_sha256",
    "load_model_scale_registry",
    "require_exact_revision",
]
