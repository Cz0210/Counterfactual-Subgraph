"""Physical runtime evidence for the BACE LLM stage/scale framework.

Tracked YAML is a plan, not runtime evidence.  These validators reopen exact
JSON files, verify their bytes and self hashes, and keep GPU science blocked
until model-load parameter reports exist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import LLMAblationContractError, canonical_json_sha256, require_sha256
from .model_scale_registry import ModelScaleEntry


REFERENCE_SCHEMA = "bace_ours_llm_reference_v2"
REFERENCE_REVISION = "b8b2ea19e48f53d190fe8dced94572717f8e89a2"
TWO_B_REVISION = "215c0dbc89417a06bbc3bae43a3ad61e58f0a56e"
TWENTY_B_REVISION = "e8d0f503e00f143f6787263765ff6ee5f3fe3998"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _physical_json(path_like: str | Path, expected_sha256: str, *, role: str):
    lexical = Path(path_like).expanduser()
    if not lexical.is_absolute() or lexical.is_symlink():
        raise LLMAblationContractError(f"{role} must be an absolute physical JSON file")
    path = lexical.resolve(strict=True)
    if not path.is_file():
        raise LLMAblationContractError(f"{role} is not a regular file")
    expected = require_sha256(expected_sha256, field=f"{role}_sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise LLMAblationContractError(f"{role} file SHA256 changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LLMAblationContractError(f"{role} must contain one object")
    return path, payload


def _verify_self_hash(payload: Mapping[str, Any], field: str, *, role: str) -> str:
    claimed = require_sha256(payload.get(field), field=f"{role}.{field}")
    body = dict(payload)
    body.pop(field)
    if canonical_json_sha256(body) != claimed:
        raise LLMAblationContractError(f"{role} self hash changed")
    return claimed


@dataclass(frozen=True, slots=True)
class BACEReferenceEvidence:
    path: str
    file_sha256: str
    self_sha256: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "file_sha256",
            require_sha256(self.file_sha256, field="reference_file_sha256"),
        )
        object.__setattr__(
            self,
            "self_sha256",
            require_sha256(self.self_sha256, field="reference_self_sha256"),
        )
        object.__setattr__(self, "payload", dict(self.payload))


def load_bace_reference_v2(
    path_like: str | Path, expected_sha256: str
) -> BACEReferenceEvidence:
    path, payload = _physical_json(
        path_like, expected_sha256, role="bace_llm_reference_v2"
    )
    required = {
        "schema_version": REFERENCE_SCHEMA,
        "status": "PASS",
        "dataset": "bace",
        "method": "ours",
        "source_label": 1,
        "scientific_values_inferred": False,
        "main_policy_scientific_name": "CHEMLLM_7B_OFF_THE_SHELF_PLUS_FRESH_LORA_PPO",
        "main_policy_must_not_be_named": "CHEMLLM_7B_PROJECT_SFT_PPO",
    }
    changed = [key for key, value in required.items() if payload.get(key) != value]
    if changed:
        raise LLMAblationContractError(
            "BACE reference identity/provenance changed: " + ", ".join(changed)
        )
    self_sha = _verify_self_hash(
        payload, "reference_contract_sha256", role="bace_llm_reference_v2"
    )
    base = payload.get("base_model")
    if not isinstance(base, Mapping):
        raise LLMAblationContractError("BACE reference lacks base_model evidence")
    parameters = base.get("parameters")
    if not isinstance(parameters, Mapping):
        raise LLMAblationContractError("BACE base model lacks parameter evidence")
    expected_base = {
        "repository_id": "AI4Chem/ChemLLM-7B-Chat",
        "revision": REFERENCE_REVISION,
    }
    if any(base.get(key) != value for key, value in expected_base.items()):
        raise LLMAblationContractError("BACE base model repository/revision changed")
    if not Path(str(base.get("path") or "")).is_absolute():
        raise LLMAblationContractError("BACE base model path must remain absolute")
    require_sha256(base.get("directory_contract_sha256"), field="base_model.contract")
    if parameters.get("total_parameters") != 7_737_708_544:
        raise LLMAblationContractError("BACE 7B base parameter header count changed")
    dtype_counts = parameters.get("dtype_parameter_counts")
    if not isinstance(dtype_counts, Mapping) or dtype_counts.get("BF16") != 7_737_708_544:
        raise LLMAblationContractError("BACE 7B BF16 header evidence changed")

    ppo = payload.get("ppo")
    ppo_parameters = ppo.get("parameters") if isinstance(ppo, Mapping) else None
    if not isinstance(ppo, Mapping) or not isinstance(ppo_parameters, Mapping):
        raise LLMAblationContractError("BACE reference lacks PPO parameter evidence")
    ppo_total = ppo_parameters.get("total_parameters", ppo_parameters.get("total"))
    ppo_dtypes = ppo_parameters.get("dtype_parameter_counts")
    if (
        ppo_total != 18_874_368
        or not isinstance(ppo_dtypes, Mapping)
        or ppo_dtypes.get("F32") != 18_874_368
        or ppo.get("trainable_parameters") != 18_874_368
        or ppo.get("loaded_total_parameters_base_plus_adapter") != 7_756_582_912
        or ppo.get("optimizer_updates") != 300
    ):
        raise LLMAblationContractError("BACE fresh-LoRA PPO provenance changed")

    variants = payload.get("stage_variants")
    expected_status = {
        "A0_BRICS_FIXED": "CPU_FRAMEWORK_AVAILABLE",
        "A1_CHEMLLM_7B_OFF_THE_SHELF": "AVAILABLE",
        "A2_CHEMLLM_7B_PROJECT_SFT": "BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT",
        "A3_CHEMLLM_7B_PROJECT_SFT_PPO": "BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT",
    }
    if not isinstance(variants, Mapping) or any(
        not isinstance(variants.get(key), Mapping)
        or variants[key].get("status") != status
        for key, status in expected_status.items()
    ):
        raise LLMAblationContractError("BACE stage availability changed")
    a1 = variants["A1_CHEMLLM_7B_OFF_THE_SHELF"]
    if (
        a1.get("repository_id") != "AI4Chem/ChemLLM-7B-Chat"
        or a1.get("revision") != REFERENCE_REVISION
        or a1.get("project_adapter_loaded") is not False
    ):
        raise LLMAblationContractError("off-the-shelf 7B topology changed")
    if variants["A3_CHEMLLM_7B_PROJECT_SFT_PPO"].get(
        "actual_main_policy_available_as"
    ) != "CHEMLLM_7B_FRESH_LORA_PPO":
        raise LLMAblationContractError("A3 requested-label mismatch evidence changed")
    return BACEReferenceEvidence(
        path=str(path),
        file_sha256=sha256_file(path),
        self_sha256=self_sha,
        payload=payload,
    )


def validate_stage_config_against_reference(
    stage: Mapping[str, Any], reference: BACEReferenceEvidence
) -> None:
    descriptor = stage.get("reference_contract")
    required_descriptor = {
        "binding": "REQUIRED_RUNTIME_PHYSICAL_FILE",
        "expected_schema": REFERENCE_SCHEMA,
        "required_status": "PASS",
        "file_sha256_required": True,
        "self_hash_required": True,
    }
    if not isinstance(descriptor, Mapping) or any(
        descriptor.get(key) != value for key, value in required_descriptor.items()
    ):
        raise LLMAblationContractError("stage config lacks runtime reference binding")
    local_rows = stage.get("design", {}).get("variants")
    if not isinstance(local_rows, list):
        raise LLMAblationContractError("stage variants must be a list")
    remote = reference.payload["stage_variants"]
    for row in local_rows:
        if not isinstance(row, Mapping):
            raise LLMAblationContractError("stage variant row must be an object")
        key = str(row.get("reference_variant_key") or "")
        expected_status = str(row.get("reference_status") or "")
        if key not in remote or remote[key].get("status") != expected_status:
            raise LLMAblationContractError(
                f"stage variant {row.get('id')} disagrees with reference evidence"
            )
    a1 = next(row for row in local_rows if row.get("id") == "A1")
    if a1.get("exact_base_revision") != REFERENCE_REVISION:
        raise LLMAblationContractError("A1 revision disagrees with reference")


def _validate_parameter_report(
    path_like: str | Path,
    expected_sha256: str,
    *,
    expected_total: int,
    expected_lora: int,
    role: str,
) -> dict[str, Any]:
    _, payload = _physical_json(path_like, expected_sha256, role=role)
    if (
        payload.get("schema_version") != "actual_parameter_count_report_v1"
        or payload.get("source") != "ACTUAL_LOADED_WEIGHTS"
    ):
        raise LLMAblationContractError(f"{role} is not an actual loaded-model report")
    _verify_self_hash(payload, "parameter_report_sha256", role=role)
    if (
        payload.get("total_parameters") != expected_total
        or payload.get("lora_trainable_parameters") != expected_lora
    ):
        raise LLMAblationContractError(f"{role} parameter counts changed")
    dtypes = payload.get("dtype")
    if not isinstance(dtypes, list) or not dtypes:
        raise LLMAblationContractError(f"{role} lacks loaded tensor dtypes")
    return payload


def _validate_2b_snapshot(path_like: str | Path, expected_sha256: str) -> dict[str, Any]:
    _, payload = _physical_json(path_like, expected_sha256, role="chemllm_2b_snapshot")
    parameters = payload.get("parameters")
    if (
        payload.get("schema_version") != "chemllm_snapshot_manifest_v1"
        or payload.get("status") != "PASS"
        or payload.get("repository_id") != "AI4Chem/CHEMLLM-2b-1_5"
        or payload.get("revision") != TWO_B_REVISION
        or payload.get("weights_downloaded") is not True
        or not isinstance(parameters, Mapping)
        or parameters.get("count_source") != "downloaded_safetensors_tensor_headers"
        or parameters.get("total_parameters") != 1_889_110_016
    ):
        raise LLMAblationContractError("2B snapshot/header evidence changed")
    return payload


def _validate_20b_metadata(path_like: str | Path, expected_sha256: str) -> dict[str, Any]:
    _, payload = _physical_json(path_like, expected_sha256, role="chemllm_20b_metadata")
    if (
        payload.get("schema_version") != "chemllm_metadata_only_manifest_v1"
        or payload.get("status") != "PASS"
        or payload.get("repository_id") != "AI4Chem/ChemLLM-20B-Chat-SFT"
        or payload.get("revision") != TWENTY_B_REVISION
        or payload.get("weights_downloaded") is not False
        or payload.get("actual_loaded_weight_parameter_count") is not None
        or payload.get("parameter_count_status") != "NOT_MEASURED_METADATA_ONLY"
    ):
        raise LLMAblationContractError("20B metadata-only evidence changed")
    return payload


def evaluate_runtime_model_evidence(
    registry: Mapping[str, ModelScaleEntry],
    *,
    two_b_snapshot: tuple[str, str] | None,
    two_b_parameter_report: tuple[str, str] | None,
    seven_b_parameter_report: tuple[str, str] | None,
    twenty_b_metadata: tuple[str, str] | None,
) -> dict[str, Any]:
    states: dict[str, str] = {}
    if two_b_snapshot is None:
        states["chemllm_2b_1_5"] = "BLOCKED_MISSING_SNAPSHOT_EVIDENCE"
    else:
        snapshot = _validate_2b_snapshot(*two_b_snapshot)
        isolated = bool(snapshot.get("isolated_import_pass", False))
        trust_enabled = bool(snapshot.get("trust_remote_code_enabled", False))
        if not isolated or not trust_enabled:
            states["chemllm_2b_1_5"] = "SNAPSHOT_READY_SCIENCE_LOAD_BLOCKED"
        elif two_b_parameter_report is None:
            states["chemllm_2b_1_5"] = "BLOCKED_MISSING_ACTUAL_PARAMETER_REPORT"
        else:
            _validate_parameter_report(
                *two_b_parameter_report,
                expected_total=registry["chemllm_2b_1_5"].header_total_parameters or 0,
                expected_lora=0,
                role="chemllm_2b_parameter_report",
            )
            states["chemllm_2b_1_5"] = "RUNTIME_EVIDENCE_PASS"
    if seven_b_parameter_report is None:
        states["chemllm_7b_main"] = "BLOCKED_MISSING_ACTUAL_PARAMETER_REPORT"
    else:
        _validate_parameter_report(
            *seven_b_parameter_report,
            expected_total=registry["chemllm_7b_main"].header_total_parameters or 0,
            expected_lora=registry["chemllm_7b_main"].header_lora_parameters or 0,
            role="chemllm_7b_parameter_report",
        )
        states["chemllm_7b_main"] = "RUNTIME_EVIDENCE_PASS"
    if twenty_b_metadata is None:
        states["chemllm_20b_sft"] = "BLOCKED_MISSING_METADATA_EVIDENCE"
    else:
        _validate_20b_metadata(*twenty_b_metadata)
        states["chemllm_20b_sft"] = "METADATA_ONLY_PASS_SCIENCE_DISABLED"
    science_ready = all(
        states.get(key) == "RUNTIME_EVIDENCE_PASS"
        for key in ("chemllm_2b_1_5", "chemllm_7b_main")
    )
    return {
        "schema_version": "llm_runtime_model_evidence_decision_v1",
        "states": states,
        "runtime_science_ready": science_ready,
        "twenty_b_science_enabled": False,
    }


def runtime_run_contract_sha256(
    *,
    file_identities: Mapping[str, Mapping[str, Any]],
    execution_commit: str,
) -> str:
    if len(execution_commit) != 40:
        raise LLMAblationContractError("execution commit must be a Git SHA")
    return canonical_json_sha256(
        {
            "schema_version": "llm_stage_scale_runtime_contract_v2",
            "execution_commit": execution_commit,
            "files": {key: dict(value) for key, value in sorted(file_identities.items())},
            "science_entrypoint": "CONFIG_ONLY_BLOCKED",
        }
    )


__all__ = [
    "BACEReferenceEvidence",
    "evaluate_runtime_model_evidence",
    "load_bace_reference_v2",
    "runtime_run_contract_sha256",
    "sha256_file",
    "validate_stage_config_against_reference",
]
