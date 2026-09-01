"""Typed provenance contracts shared by LLM and GNN ablations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
MAIN_ARTIFACT_RECEIPT_SCHEMA = "ablation_main_artifact_receipt_v1"
RUN_AUTHORIZATION_RECEIPT_SCHEMA = "ablation_run_authorization_receipt_v1"
MAIN_ARTIFACT_KINDS = (
    "FINAL_AUDIT",
    "FIGURE3",
    "FIGURE4",
    "TABLE2",
)


class ContractError(ValueError):
    """Raised when an ablation would proceed with ambiguous provenance."""


class AblationStatus(str, Enum):
    CONFIG_ONLY = "CONFIG_ONLY"
    READY_AFTER_MAIN_16_OF_16 = "READY_AFTER_MAIN_16_OF_16"
    RUNNING = "RUNNING"
    PASS = "PASS"
    FAILED = "FAILED"
    BLOCKED_MISSING_PROVENANCE = "BLOCKED_MISSING_PROVENANCE"


COMMON_HASH_FIELDS = (
    "train_split_sha",
    "validation_split_sha",
    "calibration_split_sha",
    "test_split_sha",
    "oracle_sha",
    "temperature_sha",
    "feature_schema_sha",
    "molclr_sha",
    "wnode_config_sha",
    "selector_config_sha",
    "threshold_config_sha",
    "evaluation_config_sha",
)


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: str, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _SHA256.fullmatch(normalized):
        raise ContractError(f"{field} must be one lowercase SHA256 digest")
    return normalized


def receipt_sha256(payload: Mapping[str, Any], *, hash_field: str) -> str:
    normalized = dict(payload)
    normalized.pop(hash_field, None)
    return canonical_json_sha256(normalized)


def _require_artifact_identity(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{field} must be an artifact identity")
    identity = dict(value)
    path = str(identity.get("path") or "")
    if not Path(path).is_absolute():
        raise ContractError(f"{field}.path must be absolute")
    require_sha256(str(identity.get("sha256") or ""), field=f"{field}.sha256")
    size = identity.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ContractError(f"{field}.size must be a non-negative integer")
    return identity


def validate_main_artifact_receipt(
    payload: Mapping[str, Any],
    *,
    artifact_kind: str,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen one final-output receipt bound to the exact 16/16 authority."""

    if artifact_kind not in MAIN_ARTIFACT_KINDS:
        raise ContractError(f"unsupported main artifact kind: {artifact_kind}")
    required = {
        "schema_version",
        "status",
        "artifact_kind",
        "matrix_authority_root",
        "matrix_status_sha256",
        "combined_audit_sha256",
        "artifact_path",
        "artifact_sha256",
        "artifact_bytes",
        "receipt_sha256",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ContractError(f"missing {artifact_kind} receipt fields: {missing}")
    if payload.get("schema_version") != MAIN_ARTIFACT_RECEIPT_SCHEMA:
        raise ContractError(f"{artifact_kind} receipt schema changed")
    if payload.get("status") != "PASS" or payload.get("artifact_kind") != artifact_kind:
        raise ContractError(f"{artifact_kind} receipt is not exact PASS")
    expected_authority = {
        "matrix_authority_root": authority.get("root"),
        "matrix_status_sha256": authority.get("matrix_status_sha256"),
        "combined_audit_sha256": authority.get("combined_audit_sha256"),
    }
    changed = [
        field
        for field, expected in expected_authority.items()
        if payload.get(field) != expected
    ]
    if changed:
        raise ContractError(
            f"{artifact_kind} receipt is bound to another matrix authority: {changed}"
        )
    raw_path = str(payload.get("artifact_path") or "")
    lexical = Path(raw_path).expanduser()
    if not lexical.is_absolute() or lexical.is_symlink():
        raise ContractError(f"{artifact_kind} artifact must be an absolute physical file")
    try:
        artifact = lexical.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"{artifact_kind} artifact is absent: {lexical}") from exc
    if not artifact.is_file():
        raise ContractError(f"{artifact_kind} artifact is not a file: {artifact}")
    expected_sha = require_sha256(
        str(payload.get("artifact_sha256") or ""),
        field=f"{artifact_kind}.artifact_sha256",
    )
    expected_bytes = payload.get("artifact_bytes")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
        or artifact.stat().st_size != expected_bytes
        or sha256_file(artifact) != expected_sha
    ):
        raise ContractError(f"{artifact_kind} artifact identity changed")
    claimed_receipt_sha = require_sha256(
        str(payload.get("receipt_sha256") or ""),
        field=f"{artifact_kind}.receipt_sha256",
    )
    if receipt_sha256(payload, hash_field="receipt_sha256") != claimed_receipt_sha:
        raise ContractError(f"{artifact_kind} receipt self-hash changed")
    return dict(payload)


def validate_run_authorization_receipt(
    payload: Mapping[str, Any],
    *,
    family: str,
    authority: Mapping[str, Any],
    artifact_receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate a project-owner authorization bound to one frozen launch input."""

    required = {
        "schema_version",
        "status",
        "authorized_by",
        "authorization_id",
        "authorized_at",
        "family",
        "allow_ablation_science",
        "run_contract_sha256",
        "execution_commit",
        "matrix_authority_root",
        "matrix_status_sha256",
        "combined_audit_sha256",
        "main_artifact_receipt_sha256s",
        "authorization_sha256",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ContractError(f"missing authorization receipt fields: {missing}")
    if payload.get("schema_version") != RUN_AUTHORIZATION_RECEIPT_SCHEMA:
        raise ContractError("authorization receipt schema changed")
    if (
        payload.get("status") != "AUTHORIZED"
        or payload.get("authorized_by") != "user_project_owner"
        or payload.get("allow_ablation_science") is not True
        or payload.get("family") != family
        or not str(payload.get("authorization_id") or "").strip()
        or not str(payload.get("authorized_at") or "").strip()
    ):
        raise ContractError("authorization receipt scope is not exact")
    require_sha256(
        str(payload.get("run_contract_sha256") or ""),
        field="run_contract_sha256",
    )
    if _GIT_COMMIT.fullmatch(str(payload.get("execution_commit") or "")) is None:
        raise ContractError("execution_commit must be one 40-character Git commit")
    for field, expected in (
        ("matrix_authority_root", authority.get("root")),
        ("matrix_status_sha256", authority.get("matrix_status_sha256")),
        ("combined_audit_sha256", authority.get("combined_audit_sha256")),
    ):
        if payload.get(field) != expected:
            raise ContractError(f"authorization receipt changed authority field: {field}")
    expected_receipts = {
        kind: receipt["receipt_sha256"]
        for kind, receipt in sorted(artifact_receipts.items())
    }
    if payload.get("main_artifact_receipt_sha256s") != expected_receipts:
        raise ContractError("authorization receipt changed final-artifact receipts")
    claimed_sha = require_sha256(
        str(payload.get("authorization_sha256") or ""),
        field="authorization_sha256",
    )
    if receipt_sha256(payload, hash_field="authorization_sha256") != claimed_sha:
        raise ContractError("authorization receipt self-hash changed")
    return dict(payload)


@dataclass(frozen=True, slots=True)
class AblationRunContract:
    dataset: str
    method: str
    variant: str
    seed: int
    train_split_sha: str
    validation_split_sha: str
    calibration_split_sha: str
    test_split_sha: str
    oracle_sha: str
    temperature_sha: str
    feature_schema_sha: str
    molclr_sha: str
    wnode_config_sha: str
    candidate_budget_contract: Mapping[str, Any]
    selector_config_sha: str
    threshold_config_sha: str
    evaluation_config_sha: str
    status: AblationStatus = AblationStatus.CONFIG_ONLY
    schema_version: str = "ablation_run_contract_v1"

    def __post_init__(self) -> None:
        if self.schema_version != "ablation_run_contract_v1":
            raise ContractError("ablation run contract schema changed")
        for field in ("dataset", "method", "variant"):
            if not str(getattr(self, field)).strip():
                raise ContractError(f"{field} must be non-empty")
        if int(self.seed) < 0:
            raise ContractError("seed must be non-negative")
        for field in COMMON_HASH_FIELDS:
            object.__setattr__(self, field, require_sha256(getattr(self, field), field=field))
        budget = dict(self.candidate_budget_contract)
        if budget.get("primary") != "proposal_attempt_matched":
            raise ContractError(
                "candidate_budget_contract.primary must be proposal_attempt_matched"
            )
        if int(budget.get("attempts_per_parent", 0)) <= 0:
            raise ContractError("candidate attempts_per_parent must be positive")
        object.__setattr__(self, "candidate_budget_contract", budget)
        if not isinstance(self.status, AblationStatus):
            object.__setattr__(self, "status", AblationStatus(str(self.status)))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["contract_sha256"] = canonical_json_sha256(payload)
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AblationRunContract":
        fields = {
            "dataset",
            "method",
            "variant",
            "seed",
            *COMMON_HASH_FIELDS,
            "candidate_budget_contract",
            "status",
            "schema_version",
        }
        missing = sorted(
            field
            for field in fields
            if field not in {"status", "schema_version"} and field not in payload
        )
        if missing:
            raise ContractError(f"missing ablation contract fields: {missing}")
        values = {field: payload[field] for field in fields if field in payload}
        contract = cls(**values)
        claimed = payload.get("contract_sha256")
        if claimed is not None and claimed != contract.to_dict()["contract_sha256"]:
            raise ContractError("ablation run contract self-hash changed")
        return contract


def validate_main_reference_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen BACE/Ours reference without guessing missing pins."""

    required = {
        "schema_version",
        "status",
        "dataset",
        "method",
        "source_label",
        "main_final_root",
        "main_final_audit_sha",
        "main_execution_commit",
        "gine_checkpoint",
        "gine_checkpoint_sha",
        "temperature",
        "temperature_sha",
        "feature_schema_sha",
        "molclr_root",
        "molclr_sha",
        "wnode_config",
        "wnode_config_sha",
        "dataset_split_hashes",
        "proposal_contract",
        "selector_contract",
        "selector_config_sha",
        "threshold_config_sha",
        "evaluation_config_sha",
        "llm_variant_availability",
        "matched_sft_checkpoint_available",
        "scientific_values_inferred",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ContractError(f"missing BACE Ours reference fields: {missing}")
    if (
        payload.get("schema_version") != "bace_ours_main_reference_v1"
        or payload.get("status") != "PASS"
    ):
        raise ContractError("main reference schema/status changed")
    if str(payload["dataset"]).lower() != "bace":
        raise ContractError("main reference dataset must be bace")
    if str(payload["method"]).lower() != "ours":
        raise ContractError("main reference method must be ours")
    if payload.get("source_label") != 1:
        raise ContractError("BACE Ours main reference requires source_label=1")
    if payload.get("scientific_values_inferred") is not False:
        raise ContractError("main reference may not infer scientific values")
    if payload.get("matched_sft_checkpoint_available") is not False:
        raise ContractError("BACE main reference has no independently matched SFT")
    if not Path(str(payload.get("main_final_root") or "")).is_absolute():
        raise ContractError("main_final_root must be absolute")
    if _GIT_COMMIT.fullmatch(str(payload.get("main_execution_commit") or "")) is None:
        raise ContractError("main_execution_commit must be one 40-character Git commit")
    temperature = payload.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ContractError("temperature must be one finite positive value")
    for field in (
        "main_final_audit_sha",
        "gine_checkpoint_sha",
        "temperature_sha",
        "feature_schema_sha",
        "molclr_sha",
        "wnode_config_sha",
        "selector_config_sha",
        "threshold_config_sha",
        "evaluation_config_sha",
    ):
        require_sha256(str(payload[field]), field=field)
    wnode = payload["wnode_config"]
    if not isinstance(wnode, Mapping) or canonical_json_sha256(wnode) != payload["wnode_config_sha"]:
        raise ContractError("WNode configuration hash changed")
    split_hashes = payload["dataset_split_hashes"]
    if not isinstance(split_hashes, Mapping):
        raise ContractError("dataset_split_hashes must be a mapping")
    for split in ("train", "validation", "calibration", "test"):
        require_sha256(str(split_hashes.get(split, "")), field=f"{split}_split_sha")
    availability = payload["llm_variant_availability"]
    if not isinstance(availability, Mapping):
        raise ContractError("llm_variant_availability must be a mapping")
    for variant in (
        "BRICS_FIXED",
        "CHEMLLM_PRETRAINED",
        "CHEMLLM_SFT",
        "CHEMLLM_SFT_PPO",
    ):
        if variant not in availability:
            raise ContractError(f"missing LLM variant availability: {variant}")
    if availability["BRICS_FIXED"] != {"status": "CONFIG_ONLY", "checkpoint": None}:
        raise ContractError("BRICS availability contract changed")
    pretrained = availability["CHEMLLM_PRETRAINED"]
    if (
        not isinstance(pretrained, Mapping)
        or pretrained.get("status") != "AVAILABLE"
        or not Path(str(pretrained.get("checkpoint") or "")).is_absolute()
    ):
        raise ContractError("pretrained ChemLLM availability is incomplete")
    require_sha256(
        str(pretrained.get("checkpoint_sha") or ""),
        field="CHEMLLM_PRETRAINED.checkpoint_sha",
    )
    for variant in ("CHEMLLM_SFT", "CHEMLLM_SFT_PPO"):
        entry = availability[variant]
        if (
            not isinstance(entry, Mapping)
            or entry.get("status") != "BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT"
            or "no independently matched SFT artifact exists"
            not in str(entry.get("reason") or "")
        ):
            raise ContractError(f"{variant} must retain the matched-SFT blocker")

    proposal = payload["proposal_contract"]
    if not isinstance(proposal, Mapping):
        raise ContractError("proposal_contract must be a mapping")
    expected_proposal = {
        "main_proposer_lineage": "CHEMLLM_BASE_FRESH_LORA_PPO",
        "matched_sft_checkpoint": None,
        "proposal_parent_count": 386,
        "candidate_attempts_per_parent": 8,
        "base_sampling_count": 4,
        "high_temperature_sampling_count": 4,
    }
    changed = [
        field
        for field, expected in expected_proposal.items()
        if proposal.get(field) != expected
    ]
    if changed:
        raise ContractError(f"BACE proposal contract changed: {changed}")
    for field in (
        "base_model_sha",
        "clean_policy_initializer_sha",
        "ppo_checkpoint_sha",
    ):
        require_sha256(str(proposal.get(field) or ""), field=f"proposal.{field}")
    for field in ("tokenizer", "ppo_adapter_weights", "ppo_adapter_config"):
        _require_artifact_identity(proposal.get(field), field=f"proposal.{field}")
    parent_cohort = proposal.get("proposal_parent_cohort")
    if not isinstance(parent_cohort, Mapping):
        raise ContractError("proposal parent cohort identity is absent")
    require_sha256(
        str(parent_cohort.get("sha256") or ""),
        field="proposal_parent_cohort.sha256",
    )
    expected_sampling = {
        "base_sampling": {
            "num_return_sequences": 4,
            "seed": 7,
            "temperature": 0.3,
            "top_p": 0.9,
            "max_new_tokens": 96,
        },
        "high_temperature_sampling": {
            "num_return_sequences": 4,
            "seed": 13,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 96,
        },
    }
    for name, expected in expected_sampling.items():
        observed = proposal.get(name)
        if not isinstance(observed, Mapping) or any(
            observed.get(field) != value for field, value in expected.items()
        ):
            raise ContractError(f"{name} does not match the frozen BACE sampling contract")
    merge = proposal.get("candidate_merge_dedup_policy")
    if not isinstance(merge, Mapping) or merge.get("deterministic_merge") is not True:
        raise ContractError("candidate merge/dedup contract is incomplete")
    for field in ("candidate_universe_sha", "pool_sha"):
        require_sha256(str(merge.get(field) or ""), field=f"proposal.merge.{field}")

    selector = payload["selector_contract"]
    if not isinstance(selector, Mapping):
        raise ContractError("selector_contract must be a mapping")
    if (
        selector.get("K") != 20
        or selector.get("Table2_K") != 10
        or selector.get("test_used_for_selection") is not False
        or selector.get("threshold_config_sha") != payload["threshold_config_sha"]
    ):
        raise ContractError("selector K/freeze/threshold contract changed")
    for field in ("ordered_rule_ids_sha", "calibration_input_sha", "threshold_config_sha"):
        require_sha256(str(selector.get(field) or ""), field=f"selector.{field}")
    for field in ("selector_manifest", "thresholds", "variant_configs"):
        _require_artifact_identity(selector.get(field), field=f"selector.{field}")
    normalized = dict(payload)
    claimed_reference_sha = normalized.pop("reference_contract_sha256", None)
    computed_reference_sha = canonical_json_sha256(normalized)
    if (
        claimed_reference_sha is not None
        and claimed_reference_sha != computed_reference_sha
    ):
        raise ContractError("BACE Ours reference contract self-hash changed")
    normalized["reference_contract_sha256"] = computed_reference_sha
    return normalized


__all__ = [
    "AblationRunContract",
    "AblationStatus",
    "COMMON_HASH_FIELDS",
    "ContractError",
    "MAIN_ARTIFACT_KINDS",
    "MAIN_ARTIFACT_RECEIPT_SCHEMA",
    "RUN_AUTHORIZATION_RECEIPT_SCHEMA",
    "canonical_json_sha256",
    "receipt_sha256",
    "require_sha256",
    "sha256_file",
    "validate_main_artifact_receipt",
    "validate_main_reference_contract",
    "validate_run_authorization_receipt",
]
