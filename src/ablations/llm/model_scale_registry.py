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
VERIFIED_ACTUAL_LOADED_WEIGHTS = "VERIFIED_ACTUAL_LOADED_WEIGHTS"
VERIFIED_SAFETENSORS_HEADER_EXACT = "VERIFIED_SAFETENSORS_HEADER_EXACT"
NOT_MEASURED_METADATA_ONLY = "NOT_MEASURED_METADATA_ONLY"

AUDITED_MODEL_PINS: Mapping[str, Mapping[str, Any]] = {
    "chemllm_2b_1_5": {
        "run_enabled": False,
        "exact_revision": "215c0dbc89417a06bbc3bae43a3ad61e58f0a56e",
        "header_total_parameters": 1_889_110_016,
        "header_base_parameters": 1_889_110_016,
        "header_lora_parameters": 0,
        "header_dtype": "bfloat16",
        "runtime_load_state": "BLOCKED_REMOTE_CODE_ISOLATED_IMPORT",
    },
    "chemllm_7b_main": {
        "run_enabled": True,
        "exact_revision": "b8b2ea19e48f53d190fe8dced94572717f8e89a2",
        "header_total_parameters": 7_756_582_912,
        "header_base_parameters": 7_737_708_544,
        "header_lora_parameters": 18_874_368,
        "header_dtype": "bfloat16",
        "header_adapter_dtype": "float32",
        "executed_stage": "FRESH_LORA_PPO",
        "runtime_load_state": "READY_FROM_MAIN_REFERENCE",
    },
    "chemllm_20b_sft": {
        "run_enabled": False,
        "exact_revision": "e8d0f503e00f143f6787263765ff6ee5f3fe3998",
        "metadata_index_parameter_estimate": 19_861_149_696,
        "parameter_count_state": NOT_MEASURED_METADATA_ONLY,
        "runtime_load_state": "METADATA_ONLY_DISABLED",
    },
}


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
    total_parameters: int | None = None
    base_parameters: int | None = None
    trainable_parameters: int | None = None
    lora_trainable_parameters: int | None = None
    parameter_count_state: str = NOT_MEASURED_METADATA_ONLY
    executed_stage: str = "NOT_EXECUTED"
    actual_loaded_dtype: str | None = None
    adapter_dtype: str | None = None
    metadata_index_parameter_estimate: int | None = None
    header_total_parameters: int | None = None
    header_base_parameters: int | None = None
    header_lora_parameters: int | None = None
    header_dtype: str | None = None
    header_adapter_dtype: str | None = None
    runtime_load_state: str = "BLOCKED_MISSING_RUNTIME_EVIDENCE"

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
        if self.parameter_count_state not in {
            VERIFIED_ACTUAL_LOADED_WEIGHTS,
            VERIFIED_SAFETENSORS_HEADER_EXACT,
            NOT_MEASURED_METADATA_ONLY,
        }:
            raise LLMAblationContractError("unsupported parameter-count evidence state")
        counts = {
            "total_parameters": self.total_parameters,
            "base_parameters": self.base_parameters,
            "trainable_parameters": self.trainable_parameters,
            "lora_trainable_parameters": self.lora_trainable_parameters,
        }
        for name, value in counts.items():
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise LLMAblationContractError(f"{self.key}.{name} must be non-negative")
        if self.parameter_count_state == VERIFIED_ACTUAL_LOADED_WEIGHTS:
            if not self.total_parameters or not self.base_parameters:
                raise LLMAblationContractError(
                    "verified loaded weights require positive total/base parameter counts"
                )
            adapter = self.lora_trainable_parameters or 0
            if self.total_parameters != self.base_parameters + adapter:
                raise LLMAblationContractError(
                    "total_parameters must equal base plus the loaded LoRA parameters"
                )
            if (
                self.trainable_parameters is not None
                and self.trainable_parameters > self.total_parameters
            ):
                raise LLMAblationContractError(
                    "trainable_parameters cannot exceed total_parameters"
                )
        elif any(value is not None for value in counts.values()):
            raise LLMAblationContractError(
                "non-loaded parameter state may not claim actual loaded-weight counts"
            )
        if self.parameter_count_state == VERIFIED_ACTUAL_LOADED_WEIGHTS:
            if not self.actual_loaded_dtype:
                raise LLMAblationContractError(
                    "verified loaded weights require an observed tensor dtype"
                )
        elif self.actual_loaded_dtype is not None:
            raise LLMAblationContractError(
                "non-loaded entry may not claim an actual loaded dtype"
            )
        header_counts = {
            "header_total_parameters": self.header_total_parameters,
            "header_base_parameters": self.header_base_parameters,
            "header_lora_parameters": self.header_lora_parameters,
        }
        for name, value in header_counts.items():
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise LLMAblationContractError(f"{self.key}.{name} must be non-negative")
        if self.parameter_count_state == VERIFIED_SAFETENSORS_HEADER_EXACT:
            if not self.header_total_parameters or not self.header_base_parameters:
                raise LLMAblationContractError(
                    "header-exact evidence requires positive total/base counts"
                )
            header_adapter = self.header_lora_parameters or 0
            if self.header_total_parameters != self.header_base_parameters + header_adapter:
                raise LLMAblationContractError(
                    "header total must equal base plus LoRA header counts"
                )
            if not self.header_dtype:
                raise LLMAblationContractError("header-exact evidence requires dtype")
        elif any(value is not None for value in header_counts.values()) or any(
            value is not None for value in (self.header_dtype, self.header_adapter_dtype)
        ):
            raise LLMAblationContractError(
                "non-header parameter state may not claim safetensors header evidence"
            )
        if self.metadata_index_parameter_estimate is not None and (
            isinstance(self.metadata_index_parameter_estimate, bool)
            or not isinstance(self.metadata_index_parameter_estimate, int)
            or self.metadata_index_parameter_estimate <= 0
        ):
            raise LLMAblationContractError(
                "metadata index parameter estimate must be a positive integer"
            )
        if not self.executed_stage.strip():
            raise LLMAblationContractError("executed_stage must be explicit")
        if not self.runtime_load_state.strip():
            raise LLMAblationContractError("runtime_load_state must be explicit")

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
            total_parameters=payload.get("total_parameters"),
            base_parameters=payload.get("base_parameters"),
            trainable_parameters=payload.get("trainable_parameters"),
            lora_trainable_parameters=payload.get("lora_trainable_parameters"),
            parameter_count_state=str(
                payload.get("parameter_count_state") or NOT_MEASURED_METADATA_ONLY
            ),
            executed_stage=str(payload.get("executed_stage") or "NOT_EXECUTED"),
            actual_loaded_dtype=(
                str(payload["actual_loaded_dtype"])
                if payload.get("actual_loaded_dtype") is not None
                else None
            ),
            adapter_dtype=(
                str(payload["adapter_dtype"])
                if payload.get("adapter_dtype") is not None
                else None
            ),
            metadata_index_parameter_estimate=payload.get(
                "metadata_index_parameter_estimate"
            ),
            header_total_parameters=payload.get("header_total_parameters"),
            header_base_parameters=payload.get("header_base_parameters"),
            header_lora_parameters=payload.get("header_lora_parameters"),
            header_dtype=(
                str(payload["header_dtype"])
                if payload.get("header_dtype") is not None
                else None
            ),
            header_adapter_dtype=(
                str(payload["header_adapter_dtype"])
                if payload.get("header_adapter_dtype") is not None
                else None
            ),
            runtime_load_state=str(
                payload.get("runtime_load_state") or "BLOCKED_MISSING_RUNTIME_EVIDENCE"
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
    for key, expected_values in AUDITED_MODEL_PINS.items():
        entry = parsed[key]
        changed = [
            field
            for field, expected_value in expected_values.items()
            if getattr(entry, field) != expected_value
        ]
        if changed:
            raise LLMAblationContractError(
                f"{key} drifted from audited pin/count evidence: {', '.join(changed)}"
            )
    return parsed


def inventory_sha256(files: Sequence[FileIdentity]) -> str:
    return canonical_json_sha256({"files": [asdict(item) for item in files]})


__all__ = [
    "BLOCKED_MISSING_EXACT_REVISION",
    "BLOCKED_MISSING_MAIN_REFERENCE",
    "AUDITED_MODEL_PINS",
    "FileIdentity",
    "MAIN_REFERENCE",
    "ModelScaleEntry",
    "ModelSnapshotManifest",
    "OPTIONAL_UPPER_BOUND",
    "NOT_MEASURED_METADATA_ONLY",
    "PRIMARY_SMALL_SCALE",
    "READY",
    "VERIFIED_ACTUAL_LOADED_WEIGHTS",
    "VERIFIED_SAFETENSORS_HEADER_EXACT",
    "inventory_sha256",
    "load_model_scale_registry",
    "require_exact_revision",
]
