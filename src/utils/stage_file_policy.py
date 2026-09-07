"""Small next-checkpoint inode overlay for existing Mut/LLM owners.

No scheduler, lock, quota mutation or directory scan lives here. Old resource
configs remain valid; only an explicitly bound fresh overlay changes admission.
Actual statvfs observations are retained alongside the engineering requirement.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "next_executable_stage_file_policy_v1"
BASE_RESERVE = 20_000
PEAK_FACTOR = 2


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _small(identity: Mapping[str, str]) -> dict[str, Any]:
    path = Path(identity["path"])
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or path.stat().st_size > 2 * 1024**2):
        raise ValueError("STAGE_POLICY_SMALL_PHYSICAL_RECEIPT_REQUIRED:" + str(path))
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != identity["sha256"]:
        raise ValueError("STAGE_POLICY_RECEIPT_CHANGED:" + str(path))
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("STAGE_POLICY_RECEIPT_NOT_OBJECT")
    return value


def _natural(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("STAGE_PEAK_UNKNOWN_OR_INVALID:" + field)
    return value


def validate_stage_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in policy.items() if k != "self_sha256"}
    if (policy.get("schema_version") != SCHEMA or policy.get("self_sha256") != canonical_sha(body)
            or policy.get("resource_admission_scope") != "NEXT_EXECUTABLE_STAGE"
            or policy.get("base_reserve") != BASE_RESERVE or policy.get("peak_factor") != PEAK_FACTOR
            or policy.get("modify_platform_quota") is not False
            or policy.get("apply_to_running_main_science") is not False):
        raise ValueError("STAGE_POLICY_CONTRACT_MISMATCH")
    if not Path(str(policy.get("persistent_root", ""))).is_absolute():
        raise ValueError("STAGE_POLICY_ABSOLUTE_RESOURCE_DOMAIN_REQUIRED")
    stages = policy.get("stages")
    if not isinstance(stages, dict) or not stages:
        raise ValueError("STAGE_POLICY_NO_EXECUTABLE_STAGE")
    common = policy.get("concurrent_components")
    if not isinstance(common, list):
        raise ValueError("STAGE_POLICY_CONCURRENT_COVERAGE_REQUIRED")
    # An unknown future stage is allowed, but cannot block an unrelated known
    # next stage or be interpreted as zero by the arithmetic below.
    for stage, definition in stages.items():
        if definition.get("state") == "PEAK_EVIDENCE_PENDING":
            continue
        if definition.get("state") != "BOUNDED" or not definition.get("safe_boundary"):
            raise ValueError("STAGE_BOUNDARY_REQUIRED:" + stage)
        rows = common + definition.get("components", [])
        ids = [row.get("component_id") for row in rows]
        if not ids or any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(ids):
            raise ValueError("STAGE_COMPONENT_DUPLICATED_OR_MISSING:" + stage)
        for row in rows:
            _natural(row.get("peak_new_files"), row["component_id"])
            if row.get("already_existing_files_counted") is not False or not row.get("safe_boundary"):
                raise ValueError("STAGE_NEW_FILES_BOUNDARY_UNBOUND:" + row["component_id"])
            if not row.get("evidence") or not row.get("bound_kind"):
                raise ValueError("STAGE_PEAK_EVIDENCE_REQUIRED:" + row["component_id"])
            if row.get("exclusive_group") is not None and not row.get("mutual_exclusion_evidence"):
                raise ValueError("STAGE_SERIAL_MAX_REQUIRES_EXCLUSION_PROOF")
    return dict(policy)


def load_stage_policy(descriptor: Mapping[str, str], persistent_root: str | Path,
                      stage_id: str | None = None) -> dict[str, Any]:
    """Verify small immutable receipts once when the existing owner starts."""
    policy = validate_stage_policy(_small(descriptor))
    expected, actual = Path(policy["persistent_root"]), Path(persistent_root)
    if expected.resolve(strict=True) != actual.resolve(strict=True):
        raise ValueError("STAGE_POLICY_RESOURCE_PATH_CHANGED")
    observed_device = os.stat(actual).st_dev
    if policy.get("filesystem_device") != observed_device:
        raise ValueError("STAGE_POLICY_FILESYSTEM_CHANGED_RECHECK_REQUIRED")
    authorization = _small(policy["authorization"])
    if (authorization.get("allow_stage_based_inode_policy") is not True
            or authorization.get("inode_base_reserve") != BASE_RESERVE
            or authorization.get("inode_next_stage_peak_factor") != PEAK_FACTOR
            or authorization.get("contact_support_first") is not False):
        raise ValueError("STAGE_POLICY_EXPLICIT_AUTHORIZATION_REQUIRED")
    identities: dict[str, str] = {}
    bounded_rows = policy["concurrent_components"] + [r for d in policy["stages"].values()
        if d.get("state") == "BOUNDED" for r in d.get("components", [])]
    for row in bounded_rows:
        identity = row["evidence"]
        previous = identities.setdefault(identity["path"], identity["sha256"])
        if previous != identity["sha256"]:
            raise ValueError("STAGE_POLICY_EVIDENCE_IDENTITY_CONFLICT")
    verified_evidence = {}
    for path, digest in identities.items():
        evidence = _small({"path": path, "sha256": digest})
        if evidence.get("state") != "CODE_BOUND_FILE_PEAK" or evidence.get("scientific_changes") is not False:
            raise ValueError("STAGE_POLICY_UNSUPPORTED_PEAK_EVIDENCE")
        verified_evidence[path] = evidence
    for row in bounded_rows:
        proof = verified_evidence[row["evidence"]["path"]].get("components", {}).get(row["component_id"], {})
        if (proof.get("peak_new_files") != row["peak_new_files"]
                or proof.get("safe_boundary") != row["safe_boundary"]
                or not proof.get("source_references") or not proof.get("derivation")):
            raise ValueError("STAGE_COMPONENT_NOT_BOUND_TO_PEAK_PROOF:" + row["component_id"])
    if stage_id is not None:
        stage_file_admission(policy, 0, stage_id=stage_id)
    return policy


def stage_file_admission(policy: Mapping[str, Any], available: int, *,
                         stage_id: str, baseline_available: int | None = None) -> dict[str, Any]:
    policy = validate_stage_policy(policy)
    _natural(available, "actual_available_file_slots")
    definition = policy["stages"].get(stage_id)
    if not definition or definition.get("state") != "BOUNDED":
        return {"admitted": False, "runtime_state": "NEXT_STAGE_PEAK_EVIDENCE_PENDING",
                "stage_id": stage_id, "peak_new_files": None,
                "required_free_inodes": None, "actual_available_file_slots": available,
                "pause_requested": True, "policy_sha256": policy["self_sha256"]}
    rows = policy["concurrent_components"] + definition["components"]
    scope_blockers = []
    for row in rows:
        for guard in row.get("scope_guards", []):
            path = Path(guard["path"])
            try:
                if guard.get("must_be_absent"):
                    if path.exists(): scope_blockers.append(row["component_id"] + ":TERMINAL_OR_STAGE_CHANGED")
                    continue
                if not path.is_file() or path.stat().st_size > 512 * 1024:
                    raise ValueError("missing or oversized phase receipt")
                state = json.loads(path.read_text())
                for key, values in guard.get("allowed_values", {}).items():
                    if state.get(key) not in values:
                        scope_blockers.append(row["component_id"] + ":PHASE_CHANGED:" + key)
                for key, maximum in guard.get("maximum_values", {}).items():
                    if type(state.get(key)) is not int or state[key] > maximum:
                        scope_blockers.append(row["component_id"] + ":BOUNDARY_CHANGED:" + key)
            except (OSError, ValueError, TypeError):
                scope_blockers.append(row["component_id"] + ":LIVE_SCOPE_UNAVAILABLE")
    if scope_blockers:
        return {"admitted": False, "runtime_state": "NEXT_STAGE_SCOPE_CHANGED_REBIND_REQUIRED",
                "stage_id": stage_id, "scope_blockers": scope_blockers,
                "peak_new_files": None, "required_free_inodes": None,
                "actual_available_file_slots": available, "pause_requested": True,
                "policy_sha256": policy["self_sha256"]}
    concurrent, exclusive = 0, {}
    for row in rows:
        count = row["peak_new_files"]
        group = row.get("exclusive_group")
        if group is None:
            concurrent += count
        else:
            exclusive[group] = max(exclusive.get(group, 0), count)
    peak = concurrent + sum(exclusive.values())
    required = BASE_RESERVE + PEAK_FACTOR * peak
    # Reserve enough for the *whole remaining* known boundary even after some
    # writes occurred. Never subtract global inode growth as if it were ours.
    remaining_floor = available - peak
    state = ("RESOURCE_EMERGENCY" if available < 5000 else
             "NEW_SCIENCE_STAGE_FORBIDDEN" if available < 10000 else
             "CHECKPOINT_AT_SAFE_BOUNDARY" if remaining_floor < BASE_RESERVE else
             "ADMISSION_PASS" if available >= required else "WAITING_STAGE_RESOURCE")
    return {"stage_id": stage_id, "admitted": available >= required,
            "runtime_state": state, "actual_available_file_slots": available,
            "base_reserve": BASE_RESERVE, "peak_factor": PEAK_FACTOR,
            "peak_new_files": peak, "required_free_inodes": required,
            "estimated_available_at_next_boundary": remaining_floor,
            "shortfall": max(0, required-available),
            "observed_net_slot_consumption": None if baseline_available is None else baseline_available-available,
            "pause_requested": state in {"RESOURCE_EMERGENCY", "NEW_SCIENCE_STAGE_FORBIDDEN", "CHECKPOINT_AT_SAFE_BOUNDARY"},
            "policy_sha256": policy["self_sha256"], "platform_quota_changed": False}


def config_file_admission(config: Mapping[str, Any], available: int, *, stage_id: str,
                          policy: Mapping[str, Any] | None = None,
                          baseline_available: int | None = None) -> dict[str, Any]:
    if "stage_file_policy" in config:
        policy = policy or load_stage_policy(config["stage_file_policy"], config["persistent_root"])
        return stage_file_admission(policy, available, stage_id=stage_id,
                                    baseline_available=baseline_available)
    required = config.get("minimum_free_inodes", 1) + config.get("reserved_new_inodes", 0)
    return {"admitted": available >= required, "required_free_inodes": required,
            "actual_available_file_slots": available, "policy_sha256": None,
            "runtime_state": "LEGACY_POLICY", "pause_requested": available < required}
