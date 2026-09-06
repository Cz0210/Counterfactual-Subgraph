"""Narrow owner/resource overlay for the sealed 3b2605d6 BACE LLM dispatch.

No model loading, preparation, GPU acquisition, process signaling or registry
mutation. Only small JSON bindings are read. Generation stays on its original
science driver; future common evaluation opts into storage-only resource flags.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from src.ablations.llm.contracts import canonical_json_sha256

DISPATCH_SCHEMA = "bace_llm_existing_owner_dispatch_v1"
OVERLAY_SCHEMA = "bace_llm_stage_resource_dispatch_binding_v1"
ORIGINAL_SCIENCE_COMMIT = "3b2605d681e2ef6116f3cacc662eead2f65dd28a"
GENERATION_ENTRY = Path("scripts/ablations/llm/run_bace_llm_successor.py")
DOWNSTREAM_ENTRY = Path("scripts/ablations/llm/run_bace_common_downstream.py")
ORDER = ["CHEMLLM_7B_OFF_THE_SHELF", "CHEMLLM_7B_PPO_LORA_MAIN", "CHEMLLM_2B_OFF_THE_SHELF"]
MAX_JSON_BYTES = 2 * 1024**2


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("LLM_RESOURCE_BINDING:" + reason)


def _same(left: Any, right: Any) -> bool:
    # Canonical JSON equality also distinguishes true from 1.
    return canonical_json_sha256(left) == canonical_json_sha256(right)


def _read_json(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    _require(set(descriptor) == {"path", "sha256"}, "FILE_DESCRIPTOR_FIELDS")
    path = Path(descriptor["path"])
    _require(path.is_absolute() and not path.is_symlink() and path.is_file(), "PHYSICAL_SMALL_JSON_REQUIRED")
    before = path.stat()
    _require(before.st_size <= MAX_JSON_BYTES, "SMALL_JSON_SIZE_LIMIT")
    raw = path.read_bytes()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    _require(all(getattr(before, f) == getattr(after, f) for f in fields), "JSON_CHANGED_DURING_READ")
    _require(hashlib.sha256(raw).hexdigest() == descriptor["sha256"], "FILE_SHA_MISMATCH")
    value = json.loads(raw)
    _require(isinstance(value, dict), "JSON_OBJECT_REQUIRED")
    return value


def small_json_descriptor(path: str | Path) -> dict[str, str]:
    """Descriptor for one small config, never for a model or result archive."""
    path = Path(path)
    _require(path.is_absolute() and not path.is_symlink() and path.is_file(), "PHYSICAL_SMALL_JSON_REQUIRED")
    _require(path.stat().st_size <= MAX_JSON_BYTES, "SMALL_JSON_SIZE_LIMIT")
    descriptor = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    _read_json(descriptor)
    return descriptor


def _basic(spec: Mapping[str, Any]) -> None:
    _require(spec.get("schema_version") == DISPATCH_SCHEMA, "DISPATCH_SCHEMA")
    body = {k: v for k, v in spec.items() if k != "self_sha256"}
    _require(spec.get("self_sha256") == canonical_json_sha256(body), "DISPATCH_SELF_SHA")
    _require(type(spec.get("max_llm_gpus")) is int and spec["max_llm_gpus"] == 1, "MAX_ONE_GPU")
    _require(spec.get("borrow_enabled") is False, "BORROW_FORBIDDEN")
    command = spec.get("command")
    _require(isinstance(command, list) and len(command) >= 4
             and all(isinstance(x, str) for x in command)
             and command[1:3] == ["-I", "-B"], "ISOLATED_GENERATION_COMMAND")


def _original_contract(original: Mapping[str, Any]) -> None:
    _basic(original)
    _require("resource_only_overlay" not in original, "OVERLAY_CHAIN_FORBIDDEN")
    _require(original.get("execution_commit") == ORIGINAL_SCIENCE_COMMIT, "ORIGINAL_SCIENCE_COMMIT")
    _require(original.get("variant_order") == ORDER, "ORIGINAL_VARIANT_ORDER")
    _require(original.get("science_started") is False
             and original.get("state") == "DISPATCHABLE_WAITING_RESOURCE", "ORIGINAL_DISPATCH_NOT_WAITING")
    _require(original.get("main_matrix_count_required") is False
             and original.get("secondary_seeds_required") is False, "NO_NEW_SCIENCE_PREDECESSOR")
    old_entry = Path(original["command"][3])
    _require(old_entry.is_absolute() and old_entry.parts[-4:] == GENERATION_ENTRY.parts, "ORIGINAL_ENTRYPOINT")
    readiness = _read_json(original["readiness"])
    _require(readiness.get("schema_version") == "bace_llm_native_readiness_v1", "READINESS_SCHEMA")
    rows = original.get("downstream_commands", {})
    _require(set(rows) == set(ORDER), "DOWNSTREAM_VARIANT_COVERAGE")
    for variant in ORDER:
        task_ref = readiness["variants"][variant]
        _require(_same(rows[variant]["task_spec"], task_ref), "UNCHANGED_TASK_DESCRIPTOR:" + variant)
        # bace_readiness.prepare emits these two metadata fields alongside the
        # file identity. They remain part of the unchanged dispatch contract;
        # only the checked file-identity pair is passed to the strict reader.
        _require(isinstance(task_ref, dict) and set(task_ref) == {
            "path", "sha256", "generator_state", "downstream_state"}, "TASK_REFERENCE_FIELDS:" + variant)
        task = _read_json({k: task_ref[k] for k in ("path", "sha256")})
        _require(task.get("execution_commit") == ORIGINAL_SCIENCE_COMMIT
                 and task.get("variant") == variant, "ORIGINAL_TASK_COMMIT:" + variant)
        _require(task.get("task_spec_sha256") == canonical_json_sha256({
            k: v for k, v in task.items() if k != "task_spec_sha256"}), "TASK_SELF_SHA:" + variant)
        for field, expected in (
            ("generator_state", "LOADER_IMPLEMENTED_GPU_SMOKE_REQUIRED_AT_DISPATCH"),
            ("downstream_state", "EXECUTABLE_ENTRYPOINT_CORRECTED_CORE_CHECK_AT_DISPATCH"),
        ):
            _require(task_ref[field] == task.get(field) == expected, "TASK_REFERENCE_METADATA:" + field)


def _expected_commands(original: Mapping[str, Any], root: Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    rows = deepcopy(original["downstream_commands"])
    old_root = Path(original["command"][3]).parents[3]
    for variant in ORDER:
        command = rows[variant]["command"]
        _require(isinstance(command, list) and command[1:4] == ["-I", "-B", str(old_root / DOWNSTREAM_ENTRY)]
                 and all(isinstance(x, str) for x in command), "ORIGINAL_DOWNSTREAM_ENTRY:" + variant)
        _require(not any(x in command for x in ("--stage-file-policy", "--stage-file-policy-sha256", "--compact-node-cache")),
                 "PREEXISTING_RESOURCE_FLAGS")
        command[3] = str(root / DOWNSTREAM_ENTRY)
        command.extend(["--stage-file-policy", policy["path"], "--stage-file-policy-sha256", policy["sha256"], "--compact-node-cache"])
    return rows


def validate_dispatch_runtime(spec: Mapping[str, Any], current_commit: str, project_root: str | Path) -> dict[str, Any]:
    """Called by the existing gpu_lock owner before dispatch, not an admission.

    Legacy same-driver dispatch remains valid. An overlay must reopen the exact
    original dispatch and differ only by the fixed resource/storage transform.
    The caller still owns real GPU leases, live reservations and resource gates.
    """
    _basic(spec)
    root = Path(project_root)
    _require(root.is_absolute() and root.is_dir(), "OWNER_PROJECT_ROOT")
    _require(re.fullmatch(r"[0-9a-f]{40}", current_commit) is not None
             and spec.get("execution_commit") == current_commit, "OWNER_DRIVER_COMMIT")
    overlay = spec.get("resource_only_overlay")
    if overlay is None:
        _require(spec["command"][3] == str(root / GENERATION_ENTRY), "LEGACY_ENTRYPOINT")
        return {"owner_driver_commit": current_commit, "science_execution_commit": current_commit,
                "resource_only_overlay": False, "resource_admission_evaluated": False}
    required = {"schema_version", "original_dispatch", "original_execution_commit", "owner_driver_commit",
                "owner_driver_root", "stage_file_policy", "cache_storage_only",
                "original_scientific_sources_unchanged_except_optin_storage"}
    _require(isinstance(overlay, dict) and set(overlay) == required, "OVERLAY_FIELDS")
    _require(overlay["schema_version"] == OVERLAY_SCHEMA
             and overlay["original_execution_commit"] == ORIGINAL_SCIENCE_COMMIT
             and overlay["owner_driver_commit"] == current_commit
             and overlay["owner_driver_root"] == str(root), "OWNER_SCIENCE_ROLE_BINDING")
    _require(overlay["cache_storage_only"] is True
             and overlay["original_scientific_sources_unchanged_except_optin_storage"] is True, "STORAGE_ONLY_DISCLOSURE")
    original = _read_json(overlay["original_dispatch"])
    _original_contract(original)
    old_config = _read_json(original["resource_config"])
    new_config = _read_json(spec["resource_config"])
    _require("stage_file_policy" not in old_config, "ORIGINAL_POLICY_ALREADY_OVERLAID")
    _read_json(overlay["stage_file_policy"])
    _require(_same(new_config, {**old_config, "stage_file_policy": overlay["stage_file_policy"]}), "RESOURCE_CONFIG_NON_POLICY_DRIFT")
    _require((root / DOWNSTREAM_ENTRY).is_file(), "NEW_DOWNSTREAM_ENTRY_MISSING")
    expected = deepcopy(original)
    expected.update(execution_commit=current_commit, resource_config=spec["resource_config"],
                    resource_only_overlay=overlay,
                    downstream_commands=_expected_commands(original, root, overlay["stage_file_policy"]))
    expected["self_sha256"] = canonical_json_sha256({k: v for k, v in expected.items() if k != "self_sha256"})
    _require(_same(spec, expected), "NON_RESOURCE_DISPATCH_DRIFT")
    return {"owner_driver_commit": current_commit, "science_execution_commit": ORIGINAL_SCIENCE_COMMIT,
            "resource_only_overlay": True, "resource_admission_evaluated": False,
            "generation_command_unchanged": True, "model_preparation_performed": False}


def seal_resource_dispatch(*, original_dispatch: Mapping[str, Any], resource_config: Mapping[str, Any],
                           owner_driver_commit: str, project_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Seal one fresh CPU-only dispatch; never activate an owner or claim PASS."""
    original = _read_json(original_dispatch)
    _original_contract(original)
    config = _read_json(resource_config)
    _require("stage_file_policy" in config, "NEW_STAGE_FILE_POLICY_REQUIRED")
    root = Path(project_root)
    policy = config["stage_file_policy"]
    payload = deepcopy(original)
    payload.update(execution_commit=owner_driver_commit, resource_config=dict(resource_config),
                   downstream_commands=_expected_commands(original, root, policy),
                   resource_only_overlay={"schema_version": OVERLAY_SCHEMA,
                       "original_dispatch": dict(original_dispatch), "original_execution_commit": ORIGINAL_SCIENCE_COMMIT,
                       "owner_driver_commit": owner_driver_commit, "owner_driver_root": str(root),
                       "stage_file_policy": policy, "cache_storage_only": True,
                       "original_scientific_sources_unchanged_except_optin_storage": True})
    payload["self_sha256"] = canonical_json_sha256({k: v for k, v in payload.items() if k != "self_sha256"})
    validate_dispatch_runtime(payload, owner_driver_commit, root)
    target = Path(output_path)
    _require(target.is_absolute() and target.parent.is_dir() and not target.exists()
             and not target.is_symlink(), "FRESH_DISPATCH_DESTINATION_REQUIRED")
    raw = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix="." + target.name + ".", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, target)  # Atomic no-replace: an existing dispatch wins.
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.unlink(temporary)
    return {"state": "SEALED_WAITING_RESOURCE", "dispatch_spec": str(target),
            "sha256": hashlib.sha256(raw).hexdigest(), "owner_driver_commit": owner_driver_commit,
            "science_execution_commit": ORIGINAL_SCIENCE_COMMIT, "science_started": False,
            "owner_started": False, "resource_admission_evaluated": False}
