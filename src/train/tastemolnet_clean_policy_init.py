"""Fail-closed TasteMolNet clean-policy initializer contracts.

This module intentionally implements only the dataset-independent path: a
generic ChemLLM base is converted to a fresh, zero-optimizer-step LoRA.  It
does not open a Taste split and it does not contain a train-only SFT fallback.
That fallback requires a separate, explicit authority and implementation.

The release authority is deliberately external and SHA-pinned.  Until an
authority binds policy-v2, the immutable source tree, the generic base, and
the shared T3/T4 frozen-oracle interface, the runnable CLI remains disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import ctypes
import errno
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping

from src.utils.tastemolnet_gine_pass_adoption_v1 import (
    ADOPTION_MARKER,
    DOWNSTREAM_BINDING_KEYS,
    DOWNSTREAM_BINDING_SCHEMA,
    hold_t2_gine_pass_adoption,
)
from src.utils.tastemolnet_research_policy import (
    POLICY_SCHEMA,
    TasteLocalDataAuthority,
    TasteResearchPolicy,
    TasteResearchPolicyError,
    load_tastemolnet_research_policy,
    validate_tastemolnet_policy_receipt,
)


CONFIG_SCHEMA = "tastemolnet_clean_policy_initializer_config_v1"
RELEASE_AUTHORITY_SCHEMA = "tastemolnet_clean_policy_release_authority_v1"
PROVENANCE_SCHEMA = "tastemolnet_clean_policy_provenance_v1"
MANIFEST_SCHEMA = "tastemolnet_clean_policy_initializer_manifest_v1"
STATE_SCHEMA = "tastemolnet_clean_policy_initializer_state_v1"
GATE_SCHEMA = "tastemolnet_clean_policy_initializer_gate_v1"
INPUT_HASHES_SCHEMA = "tastemolnet_clean_policy_initializer_input_hashes_v1"
OUTPUT_HASHES_SCHEMA = "tastemolnet_clean_policy_initializer_output_hashes_v1"
OUTPUT_AUTHORITY_SCHEMA = "tastemolnet_clean_policy_output_authority_v1"
SOURCE_INVENTORY_SCHEMA = "taste_content_tree_sha256_v1"
ADAPTER_INVENTORY_SCHEMA = "taste_adapter_inventory_sha256_v1"
ADAPTER_TENSOR_SCHEMA = "tastemolnet_zero_step_lora_tensor_identity_v1"
REFERENCE_POLICY_SCHEMA = "tastemolnet_reference_policy_identity_v1"
HELD_LOAD_TOKEN_SCHEMA = "tastemolnet_held_policy_load_token_v1"
PASS_MARKER = "[TASTE_CLEAN_POLICY_INITIALIZER_PASS]"
STAGE = "T5_CLEAN_POLICY_READY"
DATASET = "tastemolnet"
INITIALIZER_MODE = "fresh_zero_step_lora"
SOURCE_CLASSIFICATION = "CLEAN_CHEMLLM_BASE"
LABEL_MAP = {"0": "Bitter", "1": "Sweet", "2": "Tasteless"}
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RUN_NAME_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z$")
_GPU_UUID_RE = re.compile(
    r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_FORBIDDEN_SOURCE_TOKENS = (
    "bace",
    "random_forest",
    "randomforest",
    "morgan-rf",
    "morgan_rf",
    "rf-ranked",
    "rf_ranked",
)
_TOP_LEVEL_FILES = {
    "policy_provenance.json",
    "manifest.json",
    "state.json",
    "gate.json",
    "input_hashes.json",
    "output_hashes.json",
    "PASS",
}
_LORA_TENSOR_RE = re.compile(
    r"^(?P<prefix>.+)\.lora_(?P<side>[AB])(?:\.default)?\.weight$"
)
_LORA_TARGETS = frozenset({"wqkv", "wo", "w1", "w2", "w3"})


class TasteCleanPolicyError(RuntimeError):
    """The T5 initializer failed a release, provenance, or output gate."""


class TasteCleanPolicyReleaseDisabled(TasteCleanPolicyError):
    """No exact reviewed release authority is available yet."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalized_lora_key(value: str) -> str:
    return value.replace(".lora_A.default.weight", ".lora_A.weight").replace(
        ".lora_B.default.weight", ".lora_B.weight"
    )


def _lora_tensor_identity(state: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Validate and hash the complete serialized/reloaded zero-step LoRA state."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - AutoDL dependency
        raise TasteCleanPolicyError(f"{label} requires torch") from exc
    normalized: dict[str, Any] = {}
    for raw_name, tensor in state.items():
        if type(raw_name) is not str:
            raise TasteCleanPolicyError(f"{label} contains a non-string tensor key")
        name = _normalized_lora_key(raw_name)
        if name in normalized:
            raise TasteCleanPolicyError(f"{label} repeats a normalized tensor key: {name}")
        if not isinstance(tensor, torch.Tensor):
            raise TasteCleanPolicyError(f"{label} contains a non-tensor value: {name}")
        normalized[name] = tensor.detach().cpu().contiguous()
    if not normalized:
        raise TasteCleanPolicyError(f"{label} contains no LoRA tensors")

    pairs: dict[str, dict[str, Any]] = {}
    tensor_evidence: dict[str, dict[str, Any]] = {}
    observed_targets: set[str] = set()
    observed_dtypes: set[str] = set()
    for name in sorted(normalized):
        tensor = normalized[name]
        matched = _LORA_TENSOR_RE.fullmatch(name)
        if matched is None:
            raise TasteCleanPolicyError(f"{label} has an unexpected tensor key: {name}")
        if tensor.ndim != 2 or not bool(torch.is_floating_point(tensor)):
            raise TasteCleanPolicyError(f"{label} tensor must be a floating matrix: {name}")
        if not bool(torch.isfinite(tensor).all().item()):
            raise TasteCleanPolicyError(f"{label} tensor is non-finite: {name}")
        prefix = matched.group("prefix")
        side = matched.group("side")
        target = prefix.rsplit(".", 1)[-1]
        if target not in _LORA_TARGETS:
            raise TasteCleanPolicyError(f"{label} targets an unreviewed module: {name}")
        pair = pairs.setdefault(prefix, {})
        if side in pair:
            raise TasteCleanPolicyError(f"{label} repeats LoRA side {side}: {prefix}")
        pair[side] = tensor
        observed_targets.add(target)
        dtype = str(tensor.dtype).removeprefix("torch.")
        observed_dtypes.add(dtype)
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        tensor_evidence[name] = {
            "dtype": dtype,
            "shape": [int(value) for value in tensor.shape],
            "sha256": _sha256_bytes(raw),
            "target_module": target,
            "lora_side": side,
        }
    if observed_targets != _LORA_TARGETS:
        raise TasteCleanPolicyError(
            f"{label} target coverage changed: {sorted(observed_targets)}"
        )
    if len(observed_dtypes) != 1:
        raise TasteCleanPolicyError(f"{label} mixes LoRA tensor dtypes")
    for prefix, pair in sorted(pairs.items()):
        if set(pair) != {"A", "B"}:
            raise TasteCleanPolicyError(f"{label} has an incomplete LoRA A/B pair: {prefix}")
        tensor_a = pair["A"]
        tensor_b = pair["B"]
        if int(tensor_a.shape[0]) != 8 or int(tensor_b.shape[1]) != 8:
            raise TasteCleanPolicyError(f"{label} LoRA rank differs from 8: {prefix}")
        if not bool(torch.any(tensor_a != 0).item()):
            raise TasteCleanPolicyError(f"{label} LoRA A is unexpectedly all-zero: {prefix}")
        if bool(torch.any(tensor_b != 0).item()):
            raise TasteCleanPolicyError(
                f"{label} LoRA B is non-zero and is not a zero-step initializer: {prefix}"
            )
    identity = {
        "schema_version": ADAPTER_TENSOR_SCHEMA,
        "rank": 8,
        "target_modules": sorted(_LORA_TARGETS),
        "tensor_count": len(tensor_evidence),
        "dtype": next(iter(observed_dtypes)),
        "tensors": tensor_evidence,
        "all_finite": True,
        "all_lora_b_zero": True,
        "all_lora_a_nonzero": True,
    }
    return {
        **identity,
        "parameter_sha256": _canonical_sha256(identity),
    }


def _load_safetensors_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        from safetensors.torch import load as load_safetensors
    except ImportError as exc:  # pragma: no cover - AutoDL dependency
        raise TasteCleanPolicyError(f"{label} requires safetensors") from exc
    try:
        state = load_safetensors(data)
    except Exception as exc:
        raise TasteCleanPolicyError(f"{label} is not a valid safetensors payload") from exc
    if not isinstance(state, Mapping):
        raise TasteCleanPolicyError(f"{label} did not decode to a tensor mapping")
    return dict(state)


def _reference_policy_hash(
    *,
    source_model_inventory_sha256: str,
    adapter_inventory_sha256: str,
    adapter_parameter_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": REFERENCE_POLICY_SCHEMA,
            "source_model_inventory_sha256": source_model_inventory_sha256,
            "adapter_inventory_sha256": adapter_inventory_sha256,
            "adapter_parameter_sha256": adapter_parameter_sha256,
        }
    )


def _exact_keys(payload: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise TasteCleanPolicyError(f"{label} keys changed: missing={missing}, extra={extra}")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TasteCleanPolicyError(f"{label} must be one mapping")
    return value


def _hex(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HEX_RE.fullmatch(value) is None:
        raise TasteCleanPolicyError(f"{label} must be one lowercase SHA-256")
    return value


def _commit(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise TasteCleanPolicyError(f"{label} must be one lowercase Git object ID")
    return value


def _native_int(value: Any, *, label: str) -> int:
    if type(value) is not int:
        raise TasteCleanPolicyError(f"{label} must be one native integer")
    return value


def _absolute(value: Any, *, label: str, must_exist: bool) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TasteCleanPolicyError(f"{label} must be an absolute path")
    lexical = Path(value).expanduser()
    if not lexical.is_absolute():
        raise TasteCleanPolicyError(f"{label} must be an absolute path")
    _assert_no_symlink_components(lexical, label=label)
    return lexical.resolve(strict=must_exist)


def _assert_no_symlink_components(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(path.expanduser()))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            return
        if stat.S_ISLNK(info.st_mode):
            raise TasteCleanPolicyError(f"{label} contains a symlink component: {current}")


def _paths_overlap(left: Path, right: Path) -> bool:
    first = Path(os.path.abspath(left))
    second = Path(os.path.abspath(right))
    return first == second or first in second.parents or second in first.parents


def _read_regular(path: Path, *, label: str, maximum_bytes: int = 16 * 1024**2) -> bytes:
    _assert_no_symlink_components(path, label=label)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise TasteCleanPolicyError(f"{label} must be one single-link regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise TasteCleanPolicyError(f"{label} exceeds {maximum_bytes} bytes")
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise TasteCleanPolicyError(f"{label} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _hash_regular(path: Path, *, label: str) -> tuple[int, str]:
    """Stream one potentially multi-GB source leaf without materializing it."""

    _assert_no_symlink_components(path, label=label)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise TasteCleanPolicyError(f"{label} must be one single-link regular file")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        def identity(item: os.stat_result) -> tuple[int, ...]:
            return (
                int(item.st_dev),
                int(item.st_ino),
                int(item.st_mode),
                int(item.st_nlink),
                int(item.st_size),
                int(item.st_mtime_ns),
                int(item.st_ctime_ns),
            )
        if identity(before) != identity(after) or identity(after) != identity(named):
            raise TasteCleanPolicyError(f"{label} changed while it was hashed")
        if total != int(after.st_size):
            raise TasteCleanPolicyError(f"{label} size changed while it was hashed")
        return total, digest.hexdigest()
    finally:
        os.close(descriptor)


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    data = _read_regular(path, label=label)
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteCleanPolicyError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise TasteCleanPolicyError(f"{label} must contain one JSON object")
    return payload, data


def _inventory_directory(root: Path, *, label: str) -> tuple[dict[str, dict[str, Any]], str]:
    """Hash one physical tree without following links or accepting specials."""

    _assert_no_symlink_components(root, label=label)
    root_info = os.lstat(root)
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise TasteCleanPolicyError(f"{label} must be one physical directory")
    inventory: dict[str, dict[str, Any]] = {}
    for directory, directories, files in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        for name in sorted(directories):
            info = os.lstat(base / name)
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise TasteCleanPolicyError(f"{label} contains a non-physical directory")
        for name in sorted(files):
            path = base / name
            info = os.lstat(path)
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise TasteCleanPolicyError(f"{label} contains a symlink or special file")
            relative = path.relative_to(root).as_posix()
            size, digest = _hash_regular(path, label=f"{label}:{relative}")
            inventory[relative] = {"bytes": size, "sha256": digest}
    named = os.lstat(root)
    if (root_info.st_dev, root_info.st_ino) != (named.st_dev, named.st_ino):
        raise TasteCleanPolicyError(f"{label} root changed during inventory")
    return inventory, _canonical_sha256(
        {"schema_version": SOURCE_INVENTORY_SCHEMA, "files": inventory}
    )


def inspect_generic_chemllm_base(
    model_path: str | Path,
    *,
    include_files: bool = False,
) -> dict[str, Any]:
    root = _absolute(model_path, label="source_model_path", must_exist=True)
    if not root.is_dir():
        raise TasteCleanPolicyError("source_model_path must be one directory")
    inventory, digest = _inventory_directory(root, label="generic ChemLLM base")
    lowered = [root.name.lower(), *(name.lower() for name in inventory)]
    evidence = sorted(
        {token for token in _FORBIDDEN_SOURCE_TOKENS if any(token in item for item in lowered)}
    )
    if evidence:
        raise TasteCleanPolicyError(f"generic ChemLLM base has dataset/RF evidence: {evidence}")
    if "config.json" not in inventory:
        raise TasteCleanPolicyError("generic ChemLLM base lacks config.json")
    if any(
        name.endswith("adapter_config.json")
        or name.endswith("adapter_model.safetensors")
        or name.endswith("adapter_model.bin")
        for name in inventory
    ):
        raise TasteCleanPolicyError("generic ChemLLM base unexpectedly contains an adapter")
    config, _ = _read_json(root / "config.json", label="ChemLLM config")
    model_type = str(config.get("model_type") or "").lower()
    config_identity = json.dumps(config, sort_keys=True).lower()
    if (
        "chemllm" not in root.name.lower()
        and "chemllm" not in config_identity
    ) or model_type not in {"internlm2", "internlm"}:
        raise TasteCleanPolicyError("source is not an explicitly identifiable generic ChemLLM base")
    result = {
        "schema_version": SOURCE_INVENTORY_SCHEMA,
        "classification": SOURCE_CLASSIFICATION,
        "eligible": True,
        "dataset_specific": False,
        "source_model_path": str(root),
        "source_model_inventory_sha256": digest,
        "source_model_file_count": len(inventory),
        "source_model_config_sha256": inventory["config.json"]["sha256"],
        "source_adapter_present": False,
        "source_adapter_path": None,
        "source_adapter_sha256": None,
        "rf_reference_count": 0,
        "gnn_reward_used": False,
        "initializer_data_split_used": "none",
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    if include_files:
        # The managed-v2 clean-base adoption needs a self-contained source
        # receipt.  Keep the historical default payload unchanged for callers
        # that only consume the aggregate inventory identity.
        result["source_model_files"] = inventory
        result["source_model_total_bytes"] = sum(
            int(item["bytes"]) for item in inventory.values()
        )
    return result


def _load_yaml(path: Path) -> Mapping[str, Any]:
    data = _read_regular(path, label="T5 config")
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(data.decode("utf-8"))
    except Exception as exc:
        raise TasteCleanPolicyError("invalid T5 YAML config") from exc
    return _mapping(payload, label="T5 config")


@dataclass(frozen=True, slots=True)
class TasteCleanPolicyConfig:
    path: Path
    sha256: str
    output_parent: Path
    output_parent_identity: tuple[int, int]
    tracked_release_enabled: bool
    tracked_release_state: str


def load_clean_policy_config(path: str | Path) -> TasteCleanPolicyConfig:
    source = _absolute(path, label="config", must_exist=True)
    payload = _load_yaml(source)
    _exact_keys(
        payload,
        {
            "schema_version",
            "dataset",
            "stage",
            "initializer_mode",
            "tracked_release_enabled",
            "tracked_release_state",
            "external_sha_pinned_release_authority_required",
            "output_parent",
            "fresh_output_required",
            "private_output",
            "initializer_data_split_used",
            "taste_split_access_max",
            "train_only_fallback_implemented",
            "rf_reference_count",
            "gnn_reward_used",
            "validation_loaded",
            "calibration_loaded",
            "test_loaded",
            "data_redistribution_allowed",
            "hpc_execution_allowed",
            "required_pass_marker",
        },
        label="T5 config",
    )
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "dataset": DATASET,
        "stage": STAGE,
        "initializer_mode": INITIALIZER_MODE,
        "tracked_release_enabled": False,
        "external_sha_pinned_release_authority_required": True,
        "fresh_output_required": True,
        "private_output": True,
        "initializer_data_split_used": "none",
        "taste_split_access_max": "train_only",
        "train_only_fallback_implemented": False,
        "rf_reference_count": 0,
        "gnn_reward_used": False,
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "data_redistribution_allowed": False,
        "hpc_execution_allowed": False,
        "required_pass_marker": PASS_MARKER,
    }
    for key, value in expected.items():
        if type(payload.get(key)) is not type(value) or payload.get(key) != value:
            raise TasteCleanPolicyError(f"T5 config field changed: {key}")
    state = payload.get("tracked_release_state")
    if not isinstance(state, str) or not state.startswith("RELEASE_DISABLED_"):
        raise TasteCleanPolicyError("tracked T5 release state must remain explicitly disabled")
    parent = _absolute(payload.get("output_parent"), label="output_parent", must_exist=True)
    parent_info = os.lstat(parent)
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or stat.S_ISLNK(parent_info.st_mode)
        or stat.S_IMODE(parent_info.st_mode) != 0o700
        or parent_info.st_uid != os.geteuid()
    ):
        raise TasteCleanPolicyError(
            "T5 output parent must be an owner-held physical mode-0700 directory"
        )
    return TasteCleanPolicyConfig(
        path=source,
        sha256=_sha256_bytes(_read_regular(source, label="T5 config")),
        output_parent=parent,
        output_parent_identity=(int(parent_info.st_dev), int(parent_info.st_ino)),
        tracked_release_enabled=False,
        tracked_release_state=state,
    )


@dataclass(frozen=True, slots=True)
class TasteFrozenOracleIdentity:
    checkpoint_dir: Path
    checkpoint_id: str
    checkpoint_sha256: str
    checkpoint_inventory_sha256: str
    checkpoint_stat_inventory_sha256: str
    checkpoint_sha256s_sha256: str
    feature_schema_sha256: str
    temperature_calibration_sha256: str
    downstream_policy_sha256: str
    t2_adoption_binding: Mapping[str, Any]
    t3_output_root: Path
    t3_gate_sha256: str
    t3_root_inventory_sha256: str
    t4_output_root: Path
    t4_gate_sha256: str
    t4_root_inventory_sha256: str

    def evidence(self) -> dict[str, Any]:
        return {
            "dataset": DATASET,
            "backbone": "gine",
            "num_classes": 3,
            "label_map": LABEL_MAP,
            "source_label": 1,
            "strict_flip": "pred_before == 1 and pred_after != 1",
            "rf_oracle_used": False,
            "checkpoint_dir": str(self.checkpoint_dir),
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_inventory_sha256": self.checkpoint_inventory_sha256,
            "checkpoint_stat_inventory_sha256": self.checkpoint_stat_inventory_sha256,
            "checkpoint_sha256s_sha256": self.checkpoint_sha256s_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "temperature_calibration_sha256": self.temperature_calibration_sha256,
            "downstream_policy_sha256": self.downstream_policy_sha256,
            "t2_adoption_binding": json.loads(
                json.dumps(self.t2_adoption_binding)
            ),
            "t3_output_root": str(self.t3_output_root),
            "t3_gate_sha256": self.t3_gate_sha256,
            "t3_root_inventory_sha256": self.t3_root_inventory_sha256,
            "t4_output_root": str(self.t4_output_root),
            "t4_gate_sha256": self.t4_gate_sha256,
            "t4_root_inventory_sha256": self.t4_root_inventory_sha256,
        }


TasteManagedEvidenceBindingV2 = TasteFrozenOracleIdentity


@dataclass(slots=True)
class TasteCleanPolicyReleaseAuthority:
    path: Path
    sha256: str
    payload: Mapping[str, Any]
    policy: TasteResearchPolicy
    policy_receipt_path: Path
    policy_receipt_sha256: str
    source_model_path: Path
    source_model_inventory_sha256: str
    project_root: Path
    implementation_commit: str
    implementation_tree: str
    controller_id: str
    controller_task_id: str
    physical_gpu_index: int
    gpu_uuid: str
    oracle: TasteFrozenOracleIdentity
    source_authority: Any
    oracle_authority: Any

    def revalidate(self) -> None:
        _revalidate_release_authority(self)

    def close(self) -> None:
        for held in (self.oracle_authority, self.source_authority):
            close = getattr(held, "close", None)
            if callable(close):
                close()


def _local_authority_from_receipt(payload: Mapping[str, Any]) -> TasteLocalDataAuthority:
    raw = _mapping(payload.get("private_data_authority"), label="private_data_authority")
    _exact_keys(
        raw,
        {
            "schema_version",
            "prepared_root",
            "graph_cache_root",
            "provenance_manifest_sha256",
            "prepared_output_manifest_sha256",
            "split_manifest_sha256",
            "graph_cache_manifest_sha256",
            "source_csv_sha256",
            "prepared_rows",
            "split_rows",
            "graph_cache_rows",
            "data_reprepared",
            "graph_cache_rebuilt",
            "cache_payloads_deserialized_by_audit",
            "test_rows_deserialized_by_audit",
        },
        label="private_data_authority",
    )
    if raw.get("schema_version") != "tastemolnet_existing_private_data_authority_v1":
        raise TasteCleanPolicyError("Taste private-data receipt schema changed")
    for key in (
        "data_reprepared",
        "graph_cache_rebuilt",
        "cache_payloads_deserialized_by_audit",
        "test_rows_deserialized_by_audit",
    ):
        if raw.get(key) is not False:
            raise TasteCleanPolicyError(f"Taste private-data receipt field changed: {key}")
    split_rows = _mapping(raw.get("split_rows"), label="private_data_authority.split_rows")
    _exact_keys(split_rows, {"train", "validation", "calibration", "test"}, label="split_rows")
    typed_rows = {key: _native_int(split_rows[key], label=f"split_rows.{key}") for key in split_rows}
    return TasteLocalDataAuthority(
        prepared_root=_absolute(raw.get("prepared_root"), label="receipt prepared_root", must_exist=False),
        graph_cache_root=_absolute(raw.get("graph_cache_root"), label="receipt graph_cache_root", must_exist=False),
        provenance_manifest_sha256=_hex(raw.get("provenance_manifest_sha256"), label="provenance_manifest_sha256"),
        prepared_output_manifest_sha256=_hex(raw.get("prepared_output_manifest_sha256"), label="prepared_output_manifest_sha256"),
        split_manifest_sha256=_hex(raw.get("split_manifest_sha256"), label="split_manifest_sha256"),
        graph_cache_manifest_sha256=_hex(raw.get("graph_cache_manifest_sha256"), label="graph_cache_manifest_sha256"),
        source_csv_sha256=_hex(raw.get("source_csv_sha256"), label="source_csv_sha256"),
        prepared_rows=_native_int(raw.get("prepared_rows"), label="prepared_rows"),
        split_rows=typed_rows,
        graph_cache_rows=_native_int(raw.get("graph_cache_rows"), label="graph_cache_rows"),
    )


_T3_GATE_KEYS = {
    "schema_version", "stage", "status", "marker", "depends_on",
    "t2_science_bundle_verified", "checkpoint_dir", "checkpoint_id",
    "checkpoint_inventory_sha256", "checkpoint_stat_inventory_sha256",
    "checkpoint_sha256s_sha256", "downstream_policy_sha256",
    "existing_fit_adopted", "temperature_refit_performed", "test_loaded",
    "t2_adoption_binding",
}
_T4_GATE_KEYS = {
    "schema_version", "stage", "status", "marker", "depends_on",
    "t3_gate_sha256", "checkpoint_dir", "checkpoint_id",
    "checkpoint_inventory_sha256", "checkpoint_stat_inventory_sha256",
    "checkpoint_sha256s_sha256", "physical_gpu_index", "gpu_uuid",
    "visible_device", "cuda_visible_devices", "downstream_policy_sha256",
    "selected_count", "calibration_payload_loaded", "test_loaded",
    "per_example_predictions_written",
    "t2_adoption_binding",
}
_T3_REFERENCE_KEYS = {
    "schema_version", "dataset", "checkpoint_id", "selected_inference_asset",
    "model_sha256", "last_checkpoint_terminal_only", "last_sha256",
    "temperature_scaling_sha256", "config_sha256", "feature_schema_sha256",
    "label_map_sha256", "num_classes", "source_label", "rf_oracle_used",
    "t2_adoption_binding",
}
_T4_PROVENANCE_KEYS = {
    "schema_version", "dataset", "checkpoint_dir", "checkpoint_id",
    "checkpoint_inventory_sha256", "checkpoint_stat_inventory_sha256",
    "checkpoint_sha256s_sha256", "checkpoint_payload_files_opened",
    "checkpoint_csv_payload_opened", "selected_inference_asset", "model_sha256",
    "temperature_scaling_sha256", "config_sha256", "feature_schema_sha256",
    "physical_gpu_index", "gpu_uuid", "visible_device", "cuda_visible_devices",
    "checkpoint_load_count", "rf_oracle_used", "test_loaded",
    "t2_adoption_binding",
}
_T4_CHECKPOINT_PAYLOAD_FILES = (
    "model.pt", "config.yaml", "model_card.json", "feature_schema.json",
    "label_map.json", "split_manifest.json", "test_evaluation_status.json",
    "temperature_scaling.json", "data_use_policy_binding.json",
    "graph_cache_usage.json", "oracle_manifest.json", "last.pt",
    "last_checkpoint.json", "checkpoint_reload.json",
)


def _load_taste_gnn_stage_api() -> tuple[Any, Any, Any]:
    try:
        from src.eval.tastemolnet_gnn_stages import hold_taste_checkpoint_bundle
        from src.eval.tastemolnet_t4_oracle_smoke_v2 import HeldPublishedT3
        from src.utils.tastemolnet_t9_managed_v2 import hold_t4_managed_final
    except ImportError as exc:
        raise TasteCleanPolicyReleaseDisabled(
            "reviewed managed-v2 Taste T3/T4 held API is not installed"
        ) from exc
    return HeldPublishedT3, hold_t4_managed_final, hold_taste_checkpoint_bundle


def _validate_t2_adoption_binding(value: Any, *, label: str) -> dict[str, Any]:
    binding = _mapping(value, label=label)
    _exact_keys(binding, set(DOWNSTREAM_BINDING_KEYS), label=label)
    if (
        binding.get("schema_version") != DOWNSTREAM_BINDING_SCHEMA
        or binding.get("stage") != "T2_GINE_FULL"
        or binding.get("status") != "PASS"
        or binding.get("state") != ADOPTION_MARKER
    ):
        raise TasteCleanPolicyReleaseDisabled(
            f"{label} is not the fresh T2 adopted PASS"
        )
    for key in (
        "adoption_root_inventory_sha256",
        "gate_sha256",
        "receipt_sha256",
        "source_evidence_sha256",
        "formal_bundle_inventory_sha256",
        "formal_bundle_model_sha256",
        "formal_bundle_sha256s_sha256",
    ):
        _hex(binding.get(key), label=f"{label}.{key}")
    for key in (
        "adoption_root",
        "gate_path",
        "receipt_path",
        "formal_bundle_root",
    ):
        path = _absolute(binding.get(key), label=f"{label}.{key}", must_exist=True)
        if str(path) != binding.get(key):
            raise TasteCleanPolicyReleaseDisabled(
                f"{label}.{key} is not exact absolute"
            )
    if Path(binding["gate_path"]) != Path(binding["adoption_root"]) / "gate.json":
        raise TasteCleanPolicyReleaseDisabled(f"{label} gate path changed")
    if Path(binding["receipt_path"]) != Path(binding["adoption_root"]) / "manifest.json":
        raise TasteCleanPolicyReleaseDisabled(f"{label} receipt path changed")
    inventory = binding.get("formal_bundle_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise TasteCleanPolicyReleaseDisabled(f"{label} formal inventory is absent")
    if _canonical_sha256(inventory) != binding["formal_bundle_inventory_sha256"]:
        raise TasteCleanPolicyReleaseDisabled(
            f"{label} formal inventory digest changed"
        )
    return json.loads(json.dumps(binding))


def _validate_t3_t4_documents(
    *,
    expected: TasteFrozenOracleIdentity,
    t3: Any,
    t4: Any,
    checkpoint: Any,
) -> TasteFrozenOracleIdentity:
    frozen = expected.evidence()
    if frozen.get("dataset") != DATASET or frozen.get("num_classes") != 3 or frozen.get("source_label") != 1:
        raise TasteCleanPolicyReleaseDisabled("frozen Taste oracle identity changed")
    t2_binding = _validate_t2_adoption_binding(
        frozen.get("t2_adoption_binding"), label="Taste T6 T2 adoption binding"
    )
    t3.verify()
    t3_binding = dict(t3.binding)
    t4_evidence = dict(t4.revalidate())
    t4_science = _mapping(t4_evidence.get("science"), label="held T4 scientific evidence")
    checkpoint_evidence = dict(checkpoint.revalidate())
    feature_schema_payload = checkpoint.read_frozen_gine_payload("feature_schema.json")
    feature_schema_file_sha256 = _sha256_bytes(feature_schema_payload)
    try:
        feature_schema = json.loads(feature_schema_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteCleanPolicyReleaseDisabled(
            "held checkpoint feature schema is malformed"
        ) from exc
    if type(feature_schema) is not dict:
        raise TasteCleanPolicyReleaseDisabled(
            "held checkpoint feature schema is malformed"
        )
    temperature_payload = checkpoint.read_frozen_gine_payload("temperature_scaling.json")
    temperature_file_sha256 = _sha256_bytes(temperature_payload)
    expected_checkpoint_evidence = {
        "stage": "T3_GINE_CALIBRATED",
        "gate_sha256": expected.t3_gate_sha256,
        "root_inventory_sha256": expected.t3_root_inventory_sha256,
        "checkpoint_dir": str(expected.checkpoint_dir),
        "checkpoint_id": expected.checkpoint_id,
        "checkpoint_inventory_sha256": expected.checkpoint_inventory_sha256,
        "checkpoint_stat_inventory_sha256": expected.checkpoint_stat_inventory_sha256,
        "checkpoint_sha256s_sha256": expected.checkpoint_sha256s_sha256,
        "t2_adoption_gate_sha256": t2_binding["gate_sha256"],
        "t2_adoption_receipt_sha256": t2_binding["receipt_sha256"],
        "t2_adoption_binding_sha256": _canonical_sha256(t2_binding),
    }
    if checkpoint_evidence != expected_checkpoint_evidence:
        raise TasteCleanPolicyReleaseDisabled("checkpoint authority differs from frozen T3 identity")
    if (
        t3_binding.get("t3_root") != str(expected.t3_output_root)
        or t3_binding.get("t3_gate_sha256") != expected.t3_gate_sha256
        or t3_binding.get("t3_root_inventory_sha256") != expected.t3_root_inventory_sha256
        or t3_binding.get("checkpoint_dir") != str(expected.checkpoint_dir)
        or t3_binding.get("checkpoint_id") != expected.checkpoint_id
        or t3_binding.get("model_sha256") != expected.checkpoint_sha256
        or t3_binding.get("temperature_scaling_sha256") != expected.temperature_calibration_sha256
        or t3_binding.get("feature_schema_sha256") != expected.feature_schema_sha256
        or t3_binding.get("feature_schema_file_sha256") != feature_schema_file_sha256
        or t3_binding.get("source_t2_gate_sha256") != t2_binding["gate_sha256"]
        or t3_binding.get("source_t2_evidence_sha256") != t2_binding["source_evidence_sha256"]
        or t3_binding.get("selection_split") != "validation"
        or t3_binding.get("temperature_refit_performed") is not True
        or t3_binding.get("calibration_payload_loaded") is not False
        or t3_binding.get("test_payload_loaded") is not False
        or t3_binding.get("rf_oracle_used") is not False
    ):
        raise TasteCleanPolicyReleaseDisabled("managed-v2 T3 binding drifted")
    if (
        t4_science.get("checkpoint_id") != expected.checkpoint_id
        or t4_science.get("temperature_scaling_sha256")
        != expected.temperature_calibration_sha256
        or t4_science.get("feature_schema_sha256")
        != expected.feature_schema_sha256
    ):
        raise TasteCleanPolicyReleaseDisabled(
            "T3/T4 model/feature/temperature differs"
        )
    if (
        t4_evidence.get("root") != str(expected.t4_output_root)
        or t4_evidence.get("root_inventory_sha256") != expected.t4_root_inventory_sha256
        or t4_evidence.get("gate_sha256") != expected.t4_gate_sha256
        or t4_science.get("t3_gate_sha256") != expected.t3_gate_sha256
        or t4_science.get("t3_verification_sha256") != t3_binding.get("t3_verification_sha256")
        or t4_science.get("adaptive_calibration_search") is not True
        or t4_science.get("strict_flip_gate_pass") is not True
        or type(t4_science.get("strict_flip_count")) is not int
        or type(t4_science.get("distinct_flipped_parent_count")) is not int
        or t4_science.get("train_payload_loaded") is not False
        or t4_science.get("validation_payload_loaded") is not False
        or t4_science.get("test_payload_loaded") is not False
        or t4_science.get("rf_oracle_used") is not False
        or t4_science.get("per_example_output_written") is not False
    ):
        raise TasteCleanPolicyReleaseDisabled("managed-v2 T4 binding drifted")
    if feature_schema.get("schema_sha256") != expected.feature_schema_sha256:
        raise TasteCleanPolicyReleaseDisabled(
            "held checkpoint feature schema semantic digest differs from frozen oracle"
        )
    if temperature_file_sha256 != expected.temperature_calibration_sha256:
        raise TasteCleanPolicyReleaseDisabled(
            "held checkpoint payload differs from frozen oracle: temperature_scaling.json"
        )
    label_map_payload = checkpoint.read_frozen_gine_payload("label_map.json")
    if label_map_payload != _json_bytes(LABEL_MAP):
        try:
            label_map = json.loads(label_map_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TasteCleanPolicyReleaseDisabled(
                "held checkpoint label map is malformed"
            ) from exc
        if label_map != LABEL_MAP:
            raise TasteCleanPolicyReleaseDisabled(
                "held checkpoint label map differs from three-class Taste semantics"
            )
    return expected


@dataclass(slots=True)
class HeldTasteFrozenOracleAuthority:
    t3: Any
    t4: Any
    checkpoint: Any
    t2_adoption: Any
    evidence: TasteFrozenOracleIdentity

    def revalidate(self) -> TasteFrozenOracleIdentity:
        if self.t2_adoption.revalidate() != dict(
            self.evidence.t2_adoption_binding
        ):
            raise TasteCleanPolicyReleaseDisabled(
                "held fresh T2 adoption authority changed"
            )
        current = _validate_t3_t4_documents(
            expected=self.evidence,
            t3=self.t3,
            t4=self.t4,
            checkpoint=self.checkpoint,
        )
        if current.evidence() != self.evidence.evidence():
            raise TasteCleanPolicyReleaseDisabled("held T3/T4 frozen oracle changed")
        return current

    def close(self) -> None:
        for held in (self.checkpoint, self.t4, self.t3, self.t2_adoption):
            close = getattr(held, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "HeldTasteFrozenOracleAuthority":
        self.revalidate()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _hold_frozen_oracle_authority(payload: Mapping[str, Any]) -> HeldTasteFrozenOracleAuthority:
    hold_t3, hold_t4, hold_checkpoint = _load_taste_gnn_stage_api()
    expected = TasteFrozenOracleIdentity(
        checkpoint_dir=_absolute(payload.get("checkpoint_dir"), label="checkpoint_dir", must_exist=True),
        checkpoint_id=_hex(payload.get("checkpoint_id"), label="checkpoint_id"),
        checkpoint_sha256=_hex(payload.get("checkpoint_sha256"), label="checkpoint_sha256"),
        checkpoint_inventory_sha256=_hex(
            payload.get("checkpoint_inventory_sha256"),
            label="checkpoint_inventory_sha256",
        ),
        checkpoint_stat_inventory_sha256=_hex(
            payload.get("checkpoint_stat_inventory_sha256"),
            label="checkpoint_stat_inventory_sha256",
        ),
        checkpoint_sha256s_sha256=_hex(
            payload.get("checkpoint_sha256s_sha256"),
            label="checkpoint_sha256s_sha256",
        ),
        feature_schema_sha256=_hex(
            payload.get("feature_schema_sha256"), label="feature_schema_sha256"
        ),
        temperature_calibration_sha256=_hex(
            payload.get("temperature_calibration_sha256"),
            label="temperature_calibration_sha256",
        ),
        downstream_policy_sha256=_hex(
            payload.get("downstream_policy_sha256"), label="downstream_policy_sha256"
        ),
        t2_adoption_binding=_validate_t2_adoption_binding(
            payload.get("t2_adoption_binding"), label="frozen T2 adoption binding"
        ),
        t3_output_root=_absolute(payload.get("t3_output_root"), label="t3_output_root", must_exist=True),
        t3_gate_sha256=_hex(payload.get("t3_gate_sha256"), label="t3_gate_sha256"),
        t3_root_inventory_sha256=_hex(
            payload.get("t3_root_inventory_sha256"), label="t3_root_inventory_sha256"
        ),
        t4_output_root=_absolute(payload.get("t4_output_root"), label="t4_output_root", must_exist=True),
        t4_gate_sha256=_hex(payload.get("t4_gate_sha256"), label="t4_gate_sha256"),
        t4_root_inventory_sha256=_hex(
            payload.get("t4_root_inventory_sha256"), label="t4_root_inventory_sha256"
        ),
    )
    t3 = hold_t3(expected.t3_output_root)
    t4: Any | None = None
    checkpoint: Any | None = None
    t2_adoption: Any | None = None
    try:
        t4 = hold_t4(expected.t4_output_root)
        checkpoint = hold_checkpoint(
            expected.checkpoint_dir,
            expected_stage_evidence={
                "stage": "T3_GINE_CALIBRATED",
                "gate_sha256": expected.t3_gate_sha256,
                "root_inventory_sha256": expected.t3_root_inventory_sha256,
                "checkpoint_dir": str(expected.checkpoint_dir),
                "checkpoint_id": expected.checkpoint_id,
                "checkpoint_inventory_sha256": expected.checkpoint_inventory_sha256,
                "checkpoint_stat_inventory_sha256": expected.checkpoint_stat_inventory_sha256,
                "checkpoint_sha256s_sha256": expected.checkpoint_sha256s_sha256,
                "t2_adoption_gate_sha256": expected.t2_adoption_binding["gate_sha256"],
                "t2_adoption_receipt_sha256": expected.t2_adoption_binding["receipt_sha256"],
                "t2_adoption_binding_sha256": _canonical_sha256(expected.t2_adoption_binding),
            },
        )
        evidence = _validate_t3_t4_documents(
            expected=expected, t3=t3, t4=t4, checkpoint=checkpoint
        )
        t2_binding = dict(expected.t2_adoption_binding)
        t2_adoption = hold_t2_gine_pass_adoption(
            t2_binding["adoption_root"],
            expected_gate_sha256=t2_binding["gate_sha256"],
            expected_receipt_sha256=t2_binding["receipt_sha256"],
            expected_source_evidence_sha256=t2_binding[
                "source_evidence_sha256"
            ],
        )
        if t2_adoption.revalidate() != t2_binding:
            raise TasteCleanPolicyReleaseDisabled(
                "fresh T2 adoption differs from T3/T4 binding"
            )
        expected_payload = {
            "dataset": DATASET,
            "backbone": "gine",
            "num_classes": 3,
            "label_map": LABEL_MAP,
            "source_label": 1,
            "strict_flip": "pred_before == 1 and pred_after != 1",
            "rf_oracle_used": False,
            **{
                key: value
                for key, value in evidence.evidence().items()
                if key not in {
                    "dataset", "backbone", "num_classes", "label_map",
                    "source_label", "strict_flip", "rf_oracle_used",
                }
            },
        }
        if dict(payload) != expected_payload:
            raise TasteCleanPolicyReleaseDisabled(
                "release authority does not exactly and bidirectionally bind T3/T4"
            )
        return HeldTasteFrozenOracleAuthority(
            t3=t3,
            t4=t4,
            checkpoint=checkpoint,
            t2_adoption=t2_adoption,
            evidence=evidence,
        )
    except Exception:
        for held in (t2_adoption, checkpoint, t4, t3):
            close = getattr(held, "close", None)
            if callable(close):
                close()
        raise


def hold_taste_managed_evidence_binding_v2(
    payload: Mapping[str, Any],
) -> HeldTasteFrozenOracleAuthority:
    """Hold the exact managed-v2 T2/T3/T4 evidence consumed by Taste T6."""

    return _hold_frozen_oracle_authority(payload)


def _git_identity(project_root: Path) -> tuple[str, str]:
    environment = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    }
    def run(*arguments: str) -> str:
        try:
            return subprocess.run(
                [
                    "git", "-c", f"safe.directory={project_root}", "-C",
                    str(project_root), *arguments,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise TasteCleanPolicyError("T5 immutable Git identity is unavailable") from exc
    commit = run("rev-parse", "HEAD^{commit}")
    tree = run("rev-parse", "HEAD^{tree}")
    if run("status", "--porcelain", "--untracked-files=all"):
        raise TasteCleanPolicyError("T5 must run from a clean immutable execution tree")
    return commit, tree


def load_release_authority(
    path: str | Path,
    *,
    expected_sha256: str,
    policy_path: str | Path,
    policy_receipt_path: str | Path,
    source_model_path: str | Path,
    project_root: str | Path,
) -> TasteCleanPolicyReleaseAuthority:
    authority_path = _absolute(path, label="release_authority", must_exist=True)
    payload, data = _read_json(authority_path, label="T5 release authority")
    digest = _sha256_bytes(data)
    if digest != _hex(expected_sha256, label="expected_release_authority_sha256"):
        raise TasteCleanPolicyReleaseDisabled("T5 release authority SHA-256 is absent or changed")
    _exact_keys(
        payload,
        {
            "schema_version", "authority_id", "created_at", "dataset", "stage",
            "status", "release_enabled", "initializer_mode",
            "initializer_data_split_used", "taste_split_access_max",
            "train_only_fallback_authorized", "policy_path", "policy_file_sha256",
            "policy_canonical_sha256", "policy_receipt_path", "policy_receipt_sha256",
            "source_model_path", "source_model_inventory_sha256",
            "source_model_classification", "source_model_dataset_specific",
            "source_adapter_required", "source_adapter_path", "source_adapter_sha256",
            "project_root", "implementation_commit", "implementation_tree",
            "controller_id", "controller_task_id", "physical_gpu_index",
            "gpu_uuid", "cuda_visible_devices", "controller_binding_state",
            "gpu_lock_authority_present", "execution_receipt_present",
            "frozen_oracle", "rf_reference_count", "gnn_reward_used",
            "validation_loaded", "calibration_loaded", "test_loaded",
            "data_redistribution_allowed", "hpc_execution_allowed",
        },
        label="T5 release authority",
    )
    exact = {
        "schema_version": RELEASE_AUTHORITY_SCHEMA,
        "dataset": DATASET,
        "stage": STAGE,
        "status": "PASS",
        "release_enabled": False,
        "initializer_mode": INITIALIZER_MODE,
        "initializer_data_split_used": "none",
        "taste_split_access_max": "train_only",
        "train_only_fallback_authorized": False,
        "source_model_classification": SOURCE_CLASSIFICATION,
        "source_model_dataset_specific": False,
        "source_adapter_required": False,
        "source_adapter_path": None,
        "source_adapter_sha256": None,
        "physical_gpu_index": 2,
        "cuda_visible_devices": "2",
        "controller_binding_state": "controller_declared_only",
        "gpu_lock_authority_present": False,
        "execution_receipt_present": False,
        "rf_reference_count": 0,
        "gnn_reward_used": False,
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "data_redistribution_allowed": False,
        "hpc_execution_allowed": False,
    }
    for key, value in exact.items():
        if type(payload.get(key)) is not type(value) or payload.get(key) != value:
            raise TasteCleanPolicyReleaseDisabled(f"T5 release authority field changed: {key}")
    if not isinstance(payload.get("authority_id"), str) or not payload.get("authority_id"):
        raise TasteCleanPolicyReleaseDisabled("T5 release authority lacks authority_id")
    if not isinstance(payload.get("created_at"), str) or not payload.get("created_at"):
        raise TasteCleanPolicyReleaseDisabled("T5 release authority lacks created_at")
    for field in ("controller_id", "controller_task_id"):
        if not isinstance(payload.get(field), str) or not payload.get(field):
            raise TasteCleanPolicyReleaseDisabled(f"T5 release authority lacks {field}")
    gpu_uuid = payload.get("gpu_uuid")
    if not isinstance(gpu_uuid, str) or _GPU_UUID_RE.fullmatch(gpu_uuid) is None:
        raise TasteCleanPolicyReleaseDisabled("T5 release authority GPU UUID is malformed")

    policy_source = _absolute(policy_path, label="policy_path", must_exist=True)
    if policy_source != _absolute(payload.get("policy_path"), label="authority policy_path", must_exist=True):
        raise TasteCleanPolicyReleaseDisabled("T5 release authority binds another policy path")
    try:
        policy = load_tastemolnet_research_policy(
            policy_source,
            expected_file_sha256=_hex(
                payload.get("policy_file_sha256"), label="policy_file_sha256"
            ),
        )
        policy.require_main_route()
    except TasteResearchPolicyError as exc:
        raise TasteCleanPolicyReleaseDisabled(
            "T5 release authority policy-v2 validation failed"
        ) from exc
    if policy.payload.get("schema_version") != POLICY_SCHEMA or policy.canonical_sha256 != _hex(
        payload.get("policy_canonical_sha256"), label="policy_canonical_sha256"
    ):
        raise TasteCleanPolicyReleaseDisabled("T5 release authority policy-v2 identity changed")

    receipt_source = _absolute(policy_receipt_path, label="policy_receipt", must_exist=True)
    if receipt_source != _absolute(payload.get("policy_receipt_path"), label="authority policy_receipt_path", must_exist=True):
        raise TasteCleanPolicyReleaseDisabled("T5 release authority binds another policy receipt")
    receipt_payload, receipt_data = _read_json(receipt_source, label="policy-v2 receipt")
    receipt_sha = _sha256_bytes(receipt_data)
    if receipt_sha != _hex(payload.get("policy_receipt_sha256"), label="policy_receipt_sha256"):
        raise TasteCleanPolicyReleaseDisabled("policy-v2 receipt SHA-256 changed")
    local_authority = _local_authority_from_receipt(receipt_payload)
    try:
        validate_tastemolnet_policy_receipt(
            receipt_source,
            policy=policy,
            authority=local_authority,
            require_active=True,
            require_policy_version=2,
        )
    except TasteResearchPolicyError as exc:
        raise TasteCleanPolicyReleaseDisabled(
            "T5 release authority policy-v2 receipt validation failed"
        ) from exc

    model_source = _absolute(source_model_path, label="source_model_path", must_exist=True)
    if model_source != _absolute(payload.get("source_model_path"), label="authority source_model_path", must_exist=True):
        raise TasteCleanPolicyReleaseDisabled("T5 release authority binds another source model")
    model_evidence = inspect_generic_chemllm_base(model_source)
    model_sha = _hex(payload.get("source_model_inventory_sha256"), label="source_model_inventory_sha256")
    if model_evidence["source_model_inventory_sha256"] != model_sha:
        raise TasteCleanPolicyReleaseDisabled("generic ChemLLM source inventory changed")

    source_root = _absolute(project_root, label="project_root", must_exist=True)
    if source_root != _absolute(payload.get("project_root"), label="authority project_root", must_exist=True):
        raise TasteCleanPolicyReleaseDisabled("T5 release authority binds another source tree")
    implementation_module = Path(__file__).resolve(strict=True)
    if source_root not in implementation_module.parents:
        raise TasteCleanPolicyReleaseDisabled(
            "T5 release authority does not bind the executing implementation tree"
        )
    implementation_commit = _commit(payload.get("implementation_commit"), label="implementation_commit")
    implementation_tree = _commit(payload.get("implementation_tree"), label="implementation_tree")
    if _git_identity(source_root) != (implementation_commit, implementation_tree):
        raise TasteCleanPolicyReleaseDisabled("immutable T5 source identity changed")

    source_authority = hold_source_model_for_clean_policy(model_source, model_sha)
    oracle_authority: HeldTasteFrozenOracleAuthority | None = None
    try:
        oracle = _mapping(payload.get("frozen_oracle"), label="frozen_oracle")
        oracle_authority = _hold_frozen_oracle_authority(oracle)
        frozen = oracle_authority.evidence
    except Exception:
        source_authority.close()
        if oracle_authority is not None:
            oracle_authority.close()
        raise
    return TasteCleanPolicyReleaseAuthority(
        path=authority_path,
        sha256=digest,
        payload=dict(payload),
        policy=policy,
        policy_receipt_path=receipt_source,
        policy_receipt_sha256=receipt_sha,
        source_model_path=model_source,
        source_model_inventory_sha256=model_sha,
        project_root=source_root,
        implementation_commit=implementation_commit,
        implementation_tree=implementation_tree,
        controller_id=str(payload["controller_id"]),
        controller_task_id=str(payload["controller_task_id"]),
        physical_gpu_index=2,
        gpu_uuid=gpu_uuid,
        oracle=frozen,
        source_authority=source_authority,
        oracle_authority=oracle_authority,
    )


def _revalidate_release_authority(authority: TasteCleanPolicyReleaseAuthority) -> None:
    payload, data = _read_json(authority.path, label="held T5 release authority")
    if _sha256_bytes(data) != authority.sha256 or payload != dict(authority.payload):
        raise TasteCleanPolicyReleaseDisabled("held T5 release authority changed")
    try:
        policy = load_tastemolnet_research_policy(
            authority.policy.path,
            expected_file_sha256=authority.policy.file_sha256,
        )
        policy.require_main_route()
    except TasteResearchPolicyError as exc:
        raise TasteCleanPolicyReleaseDisabled("held T5 policy-v2 changed") from exc
    if policy.canonical_sha256 != authority.policy.canonical_sha256:
        raise TasteCleanPolicyReleaseDisabled("held T5 canonical policy changed")
    receipt_payload, receipt_data = _read_json(
        authority.policy_receipt_path, label="held policy-v2 receipt"
    )
    if _sha256_bytes(receipt_data) != authority.policy_receipt_sha256:
        raise TasteCleanPolicyReleaseDisabled("held policy-v2 receipt changed")
    try:
        validate_tastemolnet_policy_receipt(
            authority.policy_receipt_path,
            policy=policy,
            authority=_local_authority_from_receipt(receipt_payload),
            require_active=True,
            require_policy_version=2,
        )
    except TasteResearchPolicyError as exc:
        raise TasteCleanPolicyReleaseDisabled("held policy-v2 receipt is no longer valid") from exc
    source = authority.source_authority.revalidate()
    if source.get("source_model_inventory_sha256") != authority.source_model_inventory_sha256:
        raise TasteCleanPolicyReleaseDisabled("held generic ChemLLM source changed")
    oracle = authority.oracle_authority.revalidate()
    if oracle.evidence() != authority.oracle.evidence():
        raise TasteCleanPolicyReleaseDisabled("held common GINE/T3/T4 authority changed")
    if _git_identity(authority.project_root) != (
        authority.implementation_commit,
        authority.implementation_tree,
    ):
        raise TasteCleanPolicyReleaseDisabled("held T5 implementation identity changed")


def _validate_t5_gpu(authority: TasteCleanPolicyReleaseAuthority) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(authority.physical_gpu_index):
        raise TasteCleanPolicyReleaseDisabled(
            "T5 requires controller-bound CUDA_VISIBLE_DEVICES=2"
        )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TasteCleanPolicyReleaseDisabled("T5 GPU inventory is unavailable") from exc
    inventory: dict[int, str] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise TasteCleanPolicyReleaseDisabled("T5 GPU inventory is malformed")
        try:
            index = int(fields[0])
        except ValueError as exc:
            raise TasteCleanPolicyReleaseDisabled("T5 GPU index is malformed") from exc
        if index in inventory:
            raise TasteCleanPolicyReleaseDisabled("T5 GPU inventory repeats an index")
        inventory[index] = fields[1]
    if inventory.get(authority.physical_gpu_index) != authority.gpu_uuid:
        raise TasteCleanPolicyReleaseDisabled("T5 physical GPU UUID changed")
    return {
        "physical_gpu_index": authority.physical_gpu_index,
        "gpu_uuid": authority.gpu_uuid,
        "cuda_visible_devices": visible,
        "controller_binding_state": "controller_declared_only",
        "gpu_lock_authority_present": False,
        "execution_receipt_present": False,
        "controller_id": authority.controller_id,
        "controller_task_id": authority.controller_task_id,
    }


def _fd_directory_path(descriptor: int, *, label: str) -> Path:
    if not sys.platform.startswith("linux"):
        raise TasteCleanPolicyError(f"{label} requires Linux descriptor-backed loading")
    path = Path(f"/proc/self/fd/{descriptor}")
    try:
        held = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=True)
    except OSError as exc:
        raise TasteCleanPolicyError(f"{label} descriptor path is unavailable") from exc
    if (
        not stat.S_ISDIR(held.st_mode)
        or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise TasteCleanPolicyError(f"{label} descriptor path changed identity")
    return path


def _materialize_zero_step_lora(
    *,
    source_authority: Any,
    adapter_fd: int,
    seed: int,
    rank: int,
    alpha: int,
    dropout: float,
) -> dict[str, Any]:
    try:
        from peft import (
            LoraConfig,
            PeftModel,
            TaskType,
            get_peft_model,
            get_peft_model_state_dict,
        )
    except ImportError as exc:  # pragma: no cover - AutoDL dependency
        raise TasteCleanPolicyError("fresh zero-step LoRA requires peft") from exc
    from scripts.train_ppo import build_quantized_base_model, build_tokenizer, import_training_dependencies

    source_authority.revalidate()
    model_path = source_authority.stable_load_path()
    adapter_dir = _fd_directory_path(adapter_fd, label="fresh adapter")
    dependencies = import_training_dependencies()
    dependencies["set_seed"](int(seed))
    tokenizer = build_tokenizer(
        dependencies, model_path=model_path, trust_remote_code=True, local_files_only=True
    )
    base = build_quantized_base_model(
        dependencies,
        model_path=model_path,
        trust_remote_code=True,
        local_files_only=True,
        prepare_for_training=False,
    )
    if getattr(base, "peft_config", None):
        raise TasteCleanPolicyError("generic ChemLLM base unexpectedly contains PEFT")
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["wqkv", "wo", "w1", "w2", "w3"],
    )
    model = get_peft_model(base, lora)
    peft_configs = getattr(model, "peft_config", None)
    if not isinstance(peft_configs, Mapping) or len(peft_configs) != 1:
        raise TasteCleanPolicyError("fresh model does not contain exactly one PEFT adapter")
    for peft_config in peft_configs.values():
        peft_config.base_model_name_or_path = str(source_authority.source_model_dir)
    in_memory_before = _lora_tensor_identity(
        get_peft_model_state_dict(model), label="fresh in-memory LoRA"
    )
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter_dir))
    _normalize_private_tree_fd(adapter_fd)
    adapter_inventory, adapter_sha, serialized = _validate_adapter_directory_fd(
        adapter_fd,
        expected_source_model_path=source_authority.source_model_dir,
    )
    if serialized["parameter_sha256"] != in_memory_before["parameter_sha256"]:
        raise TasteCleanPolicyError("serialized LoRA differs from its zero-step model")
    adapter_reload_stat_inventory = _physical_stat_inventory_fd(
        adapter_fd, label="serialized adapter before PEFT reload"
    )

    # A save is not proof of a usable policy.  Drop the first wrapper and use
    # the production PEFT entrypoint to reload the descriptor-backed payload.
    del model
    del base
    gc.collect()
    torch_module = dependencies.get("torch")
    if torch_module is not None and getattr(torch_module, "cuda", None) is not None:
        torch_module.cuda.empty_cache()
    source_authority.revalidate()
    reload_base = build_quantized_base_model(
        dependencies,
        model_path=model_path,
        trust_remote_code=True,
        local_files_only=True,
        prepare_for_training=False,
    )
    reloaded = PeftModel.from_pretrained(
        reload_base,
        str(adapter_dir),
        is_trainable=False,
        autocast_adapter_dtype=False,
    )
    in_memory_after = _lora_tensor_identity(
        get_peft_model_state_dict(reloaded), label="reloaded in-memory LoRA"
    )
    if in_memory_after != serialized:
        raise TasteCleanPolicyError("PEFT reload changed the serialized zero-step LoRA")
    if _physical_stat_inventory_fd(
        adapter_fd, label="serialized adapter after PEFT reload"
    ) != adapter_reload_stat_inventory:
        raise TasteCleanPolicyError(
            "serialized adapter physical inventory changed during PEFT reload"
        )
    source_authority.revalidate()
    return {
        "adapter_inventory": adapter_inventory,
        "adapter_inventory_sha256": adapter_sha,
        "adapter_tensor_identity": serialized,
        "peft_reload_verified": True,
    }


def _normalize_private_tree_fd(root_fd: int) -> None:
    def visit(directory_fd: int) -> None:
        os.fchmod(directory_fd, 0o700)
        for name in os.listdir(directory_fd):
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
                try:
                    visit(child)
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                child = os.open(
                    name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd
                )
                try:
                    os.fchmod(child, 0o600)
                finally:
                    os.close(child)
            else:
                raise TasteCleanPolicyError(
                    "adapter contains a symlink/special/multiply-linked entry"
                )

    visit(root_fd)


def _validate_adapter_directory_fd(
    adapter_fd: int,
    *,
    expected_source_model_path: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], str, dict[str, Any]]:
    inventory, digest = _inventory_directory_fd(
        adapter_fd,
        label="fresh zero-step adapter",
        schema_version=ADAPTER_INVENTORY_SCHEMA,
        require_private=True,
    )
    if (
        "adapter_config.json" not in inventory
        or "adapter_model.safetensors" not in inventory
        or "adapter_model.bin" in inventory
        or inventory["adapter_model.safetensors"]["bytes"] <= 0
    ):
        raise TasteCleanPolicyError(
            "fresh zero-step LoRA requires exactly one safetensors payload"
        )
    config_data = _read_regular_at(adapter_fd, "adapter_config.json", label="adapter config")
    try:
        config = json.loads(config_data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteCleanPolicyError("adapter config is not valid JSON") from exc
    if not isinstance(config, Mapping):
        raise TasteCleanPolicyError("adapter config must be one mapping")
    _validate_adapter_config(config, expected_source_model_path=expected_source_model_path)
    weight_size = int(inventory["adapter_model.safetensors"]["bytes"])
    weights_data = _read_regular_at(
        adapter_fd,
        "adapter_model.safetensors",
        label="adapter safetensors",
        maximum_bytes=max(16 * 1024**2, weight_size),
    )
    tensor_identity = _lora_tensor_identity(
        _load_safetensors_bytes(weights_data, label="adapter safetensors"),
        label="serialized zero-step LoRA",
    )
    return inventory, digest, tensor_identity


def _validate_adapter_config(
    config: Mapping[str, Any], *, expected_source_model_path: Path | None = None
) -> None:
    """Reject a loadable-but-different PEFT payload.

    These are part of the reviewed T5 identity, not user-tunable training
    controls.  Checking them again at the consumer boundary prevents a
    self-consistent rewrite of the JSON evidence from silently changing the
    policy architecture.
    """

    target_modules = config.get("target_modules")
    if not isinstance(target_modules, list) or any(
        not isinstance(item, str) for item in target_modules
    ):
        raise TasteCleanPolicyError("fresh adapter target_modules are malformed")
    if (
        str(config.get("peft_type") or "").upper() != "LORA"
        or str(config.get("task_type") or "").upper() != "CAUSAL_LM"
        or type(config.get("r")) is not int
        or config.get("r") != 8
        or type(config.get("lora_alpha")) is not int
        or config.get("lora_alpha") != 16
        or type(config.get("lora_dropout")) is not float
        or config.get("lora_dropout") != 0.05
        or config.get("bias") != "none"
        or set(target_modules) != {"wqkv", "wo", "w1", "w2", "w3"}
        or len(target_modules) != 5
        or config.get("inference_mode") is not True
        or config.get("init_lora_weights") is not True
        or config.get("modules_to_save") is not None
        or config.get("rank_pattern") != {}
        or config.get("alpha_pattern") != {}
        or config.get("use_dora") is not False
        or config.get("use_rslora") is not False
        or config.get("fan_in_fan_out") is not False
        or config.get("layers_to_transform") is not None
        or config.get("layers_pattern") is not None
        or config.get("layer_replication") is not None
        or config.get("lora_bias") not in {None, False}
        or config.get("target_parameters") is not None
    ):
        raise TasteCleanPolicyError("fresh adapter LoRA identity changed")
    if expected_source_model_path is not None and config.get(
        "base_model_name_or_path"
    ) != str(expected_source_model_path):
        raise TasteCleanPolicyError("fresh adapter binds another base model")


def _write_new_at(directory_fd: int, name: str, data: bytes, *, mode: int = 0o600) -> None:
    if "/" in name or name in {"", ".", ".."}:
        raise TasteCleanPolicyError("output leaf name is invalid")
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _renameat2_noreplace(
    parent_fd: int,
    source: str,
    target: str,
    *,
    fsync_parent: bool = True,
) -> None:
    if not sys.platform.startswith("linux"):
        raise TasteCleanPolicyError("T5 publication requires Linux renameat2(RENAME_NOREPLACE)")
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise TasteCleanPolicyError("Linux renameat2 is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = int(renameat2(parent_fd, os.fsencode(source), parent_fd, os.fsencode(target), 1))
    if result != 0:
        observed = ctypes.get_errno()
        if observed in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"fresh T5 output already exists: {target}")
        raise OSError(observed, os.strerror(observed), target)
    if fsync_parent:
        os.fsync(parent_fd)


def build_clean_policy_initializer(
    *,
    config_path: str | Path,
    release_authority_path: str | Path,
    expected_release_authority_sha256: str,
    policy_path: str | Path,
    policy_receipt_path: str | Path,
    source_model_path: str | Path,
    project_root: str | Path,
    output_root: str | Path,
    seed: int = 7,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> dict[str, Any]:
    """Build the successor T5 initializer through retained physical authorities."""

    config = load_clean_policy_config(config_path)
    if config.tracked_release_enabled is not True:
        raise TasteCleanPolicyReleaseDisabled(
            f"{config.tracked_release_state}: controller declaration is not GPU-lock "
            "authority; a separately reviewed physical execution receipt is required"
        )
    authority = load_release_authority(
        release_authority_path,
        expected_sha256=expected_release_authority_sha256,
        policy_path=policy_path,
        policy_receipt_path=policy_receipt_path,
        source_model_path=source_model_path,
        project_root=project_root,
    )
    terminal_committed = False
    try:
        authority.revalidate()
        gpu_identity = _validate_t5_gpu(authority)
        if type(seed) is not int or seed < 0:
            raise TasteCleanPolicyError("seed must be one non-negative integer")
        if (type(lora_rank), type(lora_alpha), type(lora_dropout)) != (int, int, float):
            raise TasteCleanPolicyError("LoRA controls must retain exact numeric types")
        if (lora_rank, lora_alpha, lora_dropout) != (8, 16, 0.05):
            raise TasteCleanPolicyError(
                "T5 zero-step LoRA is frozen to r=8 alpha=16 dropout=0.05"
            )
        output = _absolute(output_root, label="output_root", must_exist=False)
        if output.parent != config.output_parent or _RUN_NAME_RE.fullmatch(output.name) is None:
            raise TasteCleanPolicyError(
                "T5 output must be one timestamp-named direct child of the frozen parent"
            )
        if output.exists():
            raise FileExistsError(f"T5 output must be fresh: {output}")
        for protected in (
            authority.policy.path,
            authority.policy_receipt_path,
            authority.source_model_path,
            authority.project_root,
            authority.oracle.checkpoint_dir,
            authority.oracle.t3_output_root,
            authority.oracle.t4_output_root,
        ):
            if _paths_overlap(output, protected):
                raise TasteCleanPolicyError("T5 output overlaps a read-only input authority")

        parent_fd = os.open(
            config.output_parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        stage_fd = -1
        adapter_fd = -1
        try:
            parent_info = os.fstat(parent_fd)
            if (int(parent_info.st_dev), int(parent_info.st_ino)) != config.output_parent_identity:
                raise TasteCleanPolicyError("T5 output parent changed after config validation")
            staging_name = f".{output.name}.t5-staging-{os.getpid()}"
            if output.name in os.listdir(parent_fd) or staging_name in os.listdir(parent_fd):
                raise FileExistsError("T5 output or owned staging name already exists")
            os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
            stage_fd = os.open(
                staging_name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
            stage_info = os.fstat(stage_fd)
            stage_identity = (int(stage_info.st_dev), int(stage_info.st_ino))
            named_stage = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
            if stage_identity != (int(named_stage.st_dev), int(named_stage.st_ino)):
                raise TasteCleanPolicyError("T5 staging inode changed during open")
            os.mkdir("adapter", mode=0o700, dir_fd=stage_fd)
            adapter_fd = os.open(
                "adapter",
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=stage_fd,
            )
            adapter_info = os.fstat(adapter_fd)
            adapter_named = os.stat("adapter", dir_fd=stage_fd, follow_symlinks=False)
            if (adapter_info.st_dev, adapter_info.st_ino) != (
                adapter_named.st_dev,
                adapter_named.st_ino,
            ):
                raise TasteCleanPolicyError("T5 adapter inode changed during open")

            materialized = _materialize_zero_step_lora(
                source_authority=authority.source_authority,
                adapter_fd=adapter_fd,
                seed=seed,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            _exact_keys(
                materialized,
                {
                    "adapter_inventory", "adapter_inventory_sha256",
                    "adapter_tensor_identity", "peft_reload_verified",
                },
                label="materialized PEFT reload evidence",
            )
            adapter_inventory, adapter_sha, adapter_tensor_identity = (
                _validate_adapter_directory_fd(
                    adapter_fd,
                    expected_source_model_path=authority.source_model_path,
                )
            )
            if (
                materialized.get("adapter_inventory") != adapter_inventory
                or materialized.get("adapter_inventory_sha256") != adapter_sha
                or materialized.get("adapter_tensor_identity") != adapter_tensor_identity
                or materialized.get("peft_reload_verified") is not True
            ):
                raise TasteCleanPolicyError("materialized/reopened PEFT evidence differs")
            authority.revalidate()
            if _validate_t5_gpu(authority) != gpu_identity:
                raise TasteCleanPolicyError("T5 GPU identity changed during PEFT reload")

            parameter_sha = str(adapter_tensor_identity["parameter_sha256"])
            policy_initializer_hash = _canonical_sha256(
                {
                    "schema_version": "tastemolnet_clean_policy_identity_v1",
                    "adapter_inventory_sha256": adapter_sha,
                    "adapter_parameter_sha256": parameter_sha,
                    "source_model_inventory_sha256": authority.source_model_inventory_sha256,
                    "initializer_data_split_used": "none",
                    "optimizer_step_count": 0,
                }
            )
            reference_policy_hash = _reference_policy_hash(
                source_model_inventory_sha256=authority.source_model_inventory_sha256,
                adapter_inventory_sha256=adapter_sha,
                adapter_parameter_sha256=parameter_sha,
            )
            if reference_policy_hash == authority.source_model_inventory_sha256:
                raise TasteCleanPolicyError("reference policy collapsed to the bare base")
            created_at = _utc_now()
            provenance = {
                "schema_version": PROVENANCE_SCHEMA, "dataset": DATASET,
                "stage": STAGE, "created_at": created_at,
                "policy_initialization_type": INITIALIZER_MODE,
                "source_model_classification": SOURCE_CLASSIFICATION,
                "source_model_path": str(authority.source_model_path),
                "source_model_inventory_schema": SOURCE_INVENTORY_SCHEMA,
                "source_model_inventory_sha256": authority.source_model_inventory_sha256,
                "reference_model_path": str(authority.source_model_path),
                "reference_model_hash": authority.source_model_inventory_sha256,
                "reference_policy_hash": reference_policy_hash,
                "source_adapter_present": False, "source_adapter_path": None,
                "source_adapter_sha256": None,
                "produced_adapter_relative_path": "adapter",
                "adapter_dir": str(output / "adapter"),
                "produced_adapter_inventory_schema": ADAPTER_INVENTORY_SCHEMA,
                "produced_adapter_inventory": adapter_inventory,
                "produced_adapter_sha256": adapter_sha,
                "adapter_tensor_schema": ADAPTER_TENSOR_SCHEMA,
                "adapter_parameter_sha256": parameter_sha,
                "peft_reload_verified": True,
                "policy_initializer_hash": policy_initializer_hash,
                "adapter_initialized_from_scratch": True, "optimizer_step_count": 0,
                "initializer_data_split_used": "none", "taste_split_access_max": "train_only",
                "taste_splits_loaded": [], "train_only_fallback_implemented": False,
                "rf_reference_count": 0, "gnn_reward_used": False,
                "validation_loaded": False, "calibration_loaded": False,
                "test_loaded": False, "oracle_neutral": True,
                "data_redistributed": False, "public_release_allowed": False,
                "seed": seed, "lora_rank": lora_rank, "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
            }
            provenance_bytes = _json_bytes(provenance)
            _write_new_at(stage_fd, "policy_provenance.json", provenance_bytes)
            input_hashes = {
                "schema_version": INPUT_HASHES_SCHEMA, "dataset": DATASET, "stage": STAGE,
                "config_path": str(config.path), "config_sha256": config.sha256,
                "release_authority_path": str(authority.path),
                "release_authority_sha256": authority.sha256,
                "policy_path": str(authority.policy.path),
                "policy_file_sha256": authority.policy.file_sha256,
                "policy_canonical_sha256": authority.policy.canonical_sha256,
                "policy_receipt_path": str(authority.policy_receipt_path),
                "policy_receipt_sha256": authority.policy_receipt_sha256,
                "source_model_path": str(authority.source_model_path),
                "source_model_inventory_sha256": authority.source_model_inventory_sha256,
                "implementation_commit": authority.implementation_commit,
                "implementation_tree": authority.implementation_tree,
                "gpu_identity": gpu_identity, "frozen_oracle": authority.oracle.evidence(),
            }
            input_bytes = _json_bytes(input_hashes)
            _write_new_at(stage_fd, "input_hashes.json", input_bytes)
            manifest = {
                "schema_version": MANIFEST_SCHEMA, "dataset": DATASET, "stage": STAGE,
                "status": "PASS", "created_at": created_at,
                "initializer_mode": INITIALIZER_MODE, "initializer_data_split_used": "none",
                "taste_split_access_max": "train_only", "oracle_neutral": True,
                "rf_reference_count": 0, "gnn_reward_used": False,
                "validation_loaded": False, "calibration_loaded": False,
                "test_loaded": False, "data_redistributed": False,
                "private_output": True, "hpc_execution_authorized": False,
                "gpu_identity": gpu_identity,
                "policy_provenance_sha256": _sha256_bytes(provenance_bytes),
                "input_hashes_sha256": _sha256_bytes(input_bytes),
                "adapter_dir": str(output / "adapter"),
                "produced_adapter_sha256": adapter_sha,
                "adapter_parameter_sha256": parameter_sha, "peft_reload_verified": True,
                "policy_initializer_hash": policy_initializer_hash,
                "reference_model_hash": authority.source_model_inventory_sha256,
                "reference_policy_hash": reference_policy_hash,
                "source_model_inventory_sha256": authority.source_model_inventory_sha256,
                "frozen_oracle_identity_sha256": _canonical_sha256(authority.oracle.evidence()),
                "required_pass_marker": PASS_MARKER,
            }
            manifest_bytes = _json_bytes(manifest)
            _write_new_at(stage_fd, "manifest.json", manifest_bytes)
            state = {
                "schema_version": STATE_SCHEMA, "dataset": DATASET, "stage": STAGE,
                "status": "PASS", "created_at": created_at, "updated_at": created_at,
                "release_authority_validated": True, "optimizer_step_count": 0,
                "science_training_performed": False, "peft_reload_verified": True,
            }
            state_bytes = _json_bytes(state)
            _write_new_at(stage_fd, "state.json", state_bytes)
            gate = {
                "schema_version": GATE_SCHEMA, "dataset": DATASET, "stage": STAGE,
                "status": "PASS", "passed": True, "created_at": created_at,
                "same_frozen_gine_t3_t4_identity_bound": True,
                "policy_initializer_hash": policy_initializer_hash,
                "reference_model_hash": authority.source_model_inventory_sha256,
                "reference_policy_hash": reference_policy_hash,
                "adapter_parameter_sha256": parameter_sha, "peft_reload_verified": True,
                "initializer_data_split_used": "none", "taste_split_access_max": "train_only",
                "rf_reference_count": 0, "gnn_reward_used": False,
                "validation_loaded": False, "calibration_loaded": False,
                "test_loaded": False, "data_redistributed": False, "marker": PASS_MARKER,
            }
            gate_bytes = _json_bytes(gate)
            _write_new_at(stage_fd, "gate.json", gate_bytes)
            output_hashes = {
                "schema_version": OUTPUT_HASHES_SCHEMA, "dataset": DATASET, "stage": STAGE,
                "adapter_inventory_schema": ADAPTER_INVENTORY_SCHEMA,
                "adapter_inventory": adapter_inventory, "adapter_sha256": adapter_sha,
                "adapter_tensor_schema": ADAPTER_TENSOR_SCHEMA,
                "adapter_tensor_identity": adapter_tensor_identity,
                "adapter_parameter_sha256": parameter_sha, "peft_reload_verified": True,
                "policy_initializer_hash": policy_initializer_hash,
                "reference_model_hash": authority.source_model_inventory_sha256,
                "reference_policy_hash": reference_policy_hash,
                "source_model_inventory_sha256": authority.source_model_inventory_sha256,
                "policy_provenance_sha256": _sha256_bytes(provenance_bytes),
                "manifest_sha256": _sha256_bytes(manifest_bytes),
                "state_sha256": _sha256_bytes(state_bytes),
                "gate_sha256": _sha256_bytes(gate_bytes),
                "input_hashes_sha256": _sha256_bytes(input_bytes),
                "pass_marker_sha256": _sha256_bytes((PASS_MARKER + "\n").encode("utf-8")),
            }
            _write_new_at(stage_fd, "output_hashes.json", _json_bytes(output_hashes))
            os.fsync(stage_fd)

            authority.revalidate()
            if _validate_t5_gpu(authority) != gpu_identity:
                raise TasteCleanPolicyError("T5 GPU identity changed before publication")
            named_parent = os.stat(config.output_parent, follow_symlinks=False)
            named_stage = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
            held_stage = os.fstat(stage_fd)
            if (
                (parent_info.st_dev, parent_info.st_ino)
                != (named_parent.st_dev, named_parent.st_ino)
                or stage_identity != (int(named_stage.st_dev), int(named_stage.st_ino))
                or stage_identity != (int(held_stage.st_dev), int(held_stage.st_ino))
            ):
                raise TasteCleanPolicyError("T5 parent/staging inode changed before publication")
            _renameat2_noreplace(parent_fd, staging_name, output.name)
            published = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
            if stage_identity != (int(published.st_dev), int(published.st_ino)):
                raise TasteCleanPolicyError("published T5 output is not the held staging inode")

            authority.revalidate()
            if _validate_t5_gpu(authority) != gpu_identity:
                raise TasteCleanPolicyError("T5 GPU identity changed after publication")
            adapter_named = os.stat("adapter", dir_fd=stage_fd, follow_symlinks=False)
            adapter_held = os.fstat(adapter_fd)
            if (adapter_named.st_dev, adapter_named.st_ino) != (
                adapter_held.st_dev, adapter_held.st_ino
            ):
                raise TasteCleanPolicyError("published T5 adapter differs from held adapter")
            # Validate the complete terminal image while it is still
            # non-authorizing.  The prepared marker is durable but is not a
            # consumer-visible terminal name, so any validation exception
            # leaves a root that every public validator rejects.
            prepared_marker = ".PASS.prepared"
            _write_new_at(
                stage_fd,
                prepared_marker,
                (PASS_MARKER + "\n").encode("utf-8"),
            )
            os.fsync(stage_fd)
            prepared_stat_inventory = _physical_stat_inventory_fd(
                stage_fd,
                label="prepared T5 terminal",
            )
            terminal_evidence = _validate_held_clean_policy_output(
                output,
                stage_fd,
                stage_identity,
                adapter_fd,
                marker_name=prepared_marker,
            )
            if _physical_stat_inventory_fd(
                stage_fd,
                label="prepared T5 terminal before commit",
            ) != prepared_stat_inventory:
                raise TasteCleanPolicyError(
                    "prepared T5 terminal physical inventory changed"
                )
            # This no-replace rename is the final commit operation.  There is
            # deliberately no fallible validation, fsync, or external reopen
            # after the terminal name becomes visible.
            _renameat2_noreplace(
                stage_fd,
                prepared_marker,
                "PASS",
                fsync_parent=False,
            )
            terminal_committed = True
            return terminal_evidence
        finally:
            for descriptor in (adapter_fd, stage_fd, parent_fd):
                if descriptor < 0:
                    continue
                try:
                    os.close(descriptor)
                except Exception:
                    if not terminal_committed:
                        raise
    finally:
        try:
            authority.close()
        except Exception:
            if not terminal_committed:
                raise


def _read_regular_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
    maximum_bytes: int = 16 * 1024**2,
    require_private: bool = True,
) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise TasteCleanPolicyError(f"{label} must be one single-link regular file")
        if require_private and stat.S_IMODE(before.st_mode) != 0o600:
            raise TasteCleanPolicyError(f"{label} must remain private mode 0600")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise TasteCleanPolicyError(f"{label} exceeds {maximum_bytes} bytes")
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise TasteCleanPolicyError(f"{label} changed while held")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _hash_regular_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
    require_private: bool,
) -> tuple[int, str]:
    """Stream-hash one descriptor-relative leaf and retain physical identity."""

    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise TasteCleanPolicyError(f"{label} must be one single-link regular file")
        if require_private and stat.S_IMODE(before.st_mode) != 0o600:
            raise TasteCleanPolicyError(f"{label} must remain private mode 0600")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)

        def identity(item: os.stat_result) -> tuple[int, ...]:
            return (
                int(item.st_dev),
                int(item.st_ino),
                int(item.st_mode),
                int(item.st_nlink),
                int(item.st_size),
                int(item.st_mtime_ns),
                int(item.st_ctime_ns),
            )

        if identity(before) != identity(after) or identity(after) != identity(named):
            raise TasteCleanPolicyError(f"{label} changed while it was hashed")
        if total != int(after.st_size):
            raise TasteCleanPolicyError(f"{label} size changed while it was hashed")
        return total, digest.hexdigest()
    finally:
        os.close(descriptor)


def _read_json_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
    require_private: bool = True,
) -> tuple[dict[str, Any], bytes]:
    data = _read_regular_at(
        directory_fd, name, label=label, require_private=require_private
    )
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteCleanPolicyError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise TasteCleanPolicyError(f"{label} must contain one JSON object")
    return payload, data


def _inventory_directory_fd(
    root_fd: int,
    *,
    label: str,
    schema_version: str,
    require_private: bool = True,
) -> tuple[dict[str, dict[str, Any]], str]:
    inventory: dict[str, dict[str, Any]] = {}

    def visit(directory_fd: int, prefix: str) -> None:
        directory_info = os.fstat(directory_fd)
        if not stat.S_ISDIR(directory_info.st_mode) or (
            require_private and stat.S_IMODE(directory_info.st_mode) != 0o700
        ):
            raise TasteCleanPolicyError(f"{label} directory must remain private mode 0700")
        for name in sorted(os.listdir(directory_fd)):
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            relative = f"{prefix}/{name}" if prefix else name
            if stat.S_ISDIR(info.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
                try:
                    visit(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                size, digest = _hash_regular_at(
                    directory_fd,
                    name,
                    label=f"{label}:{relative}",
                    require_private=require_private,
                )
                inventory[relative] = {"bytes": size, "sha256": digest}
            else:
                raise TasteCleanPolicyError(f"{label} contains a symlink or special entry")

    visit(root_fd, "")
    return inventory, _canonical_sha256(
        {"schema_version": schema_version, "files": inventory}
    )


def _physical_stat_inventory_fd(
    root_fd: int, *, label: str
) -> dict[str, dict[str, int]]:
    """Snapshot inode/ctime authority so swap-load-restore cannot disappear."""

    def fields(info: os.stat_result) -> dict[str, int]:
        return {
            "device": int(info.st_dev),
            "inode": int(info.st_ino),
            "mode": int(info.st_mode),
            "links": int(info.st_nlink),
            "uid": int(info.st_uid),
            "gid": int(info.st_gid),
            "size": int(info.st_size),
            "blocks": int(getattr(info, "st_blocks", 0)),
            "mtime_ns": int(info.st_mtime_ns),
            "ctime_ns": int(info.st_ctime_ns),
        }

    inventory: dict[str, dict[str, int]] = {}

    def visit(directory_fd: int, prefix: str) -> None:
        before_directory = os.fstat(directory_fd)
        if not stat.S_ISDIR(before_directory.st_mode):
            raise TasteCleanPolicyError(f"{label} contains a non-directory authority")
        relative_directory = prefix or "."
        for name in sorted(os.listdir(directory_fd)):
            before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            relative = f"{prefix}/{name}" if prefix else name
            if stat.S_ISDIR(before.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
                try:
                    held = os.fstat(child)
                    if (held.st_dev, held.st_ino) != (before.st_dev, before.st_ino):
                        raise TasteCleanPolicyError(
                            f"{label} directory changed during stat inventory: {relative}"
                        )
                    visit(child, relative)
                    after_held = os.fstat(child)
                    after_named = os.stat(
                        name, dir_fd=directory_fd, follow_symlinks=False
                    )
                    if fields(before) != fields(after_held) or fields(after_held) != fields(
                        after_named
                    ):
                        raise TasteCleanPolicyError(
                            f"{label} directory changed during stat inventory: {relative}"
                        )
                    inventory[relative] = fields(after_held)
                finally:
                    os.close(child)
            elif stat.S_ISREG(before.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
                try:
                    held = os.fstat(child)
                    after_named = os.stat(
                        name, dir_fd=directory_fd, follow_symlinks=False
                    )
                    if fields(before) != fields(held) or fields(held) != fields(after_named):
                        raise TasteCleanPolicyError(
                            f"{label} file changed during stat inventory: {relative}"
                        )
                    inventory[relative] = fields(held)
                finally:
                    os.close(child)
            else:
                raise TasteCleanPolicyError(
                    f"{label} contains a symlink or special entry: {relative}"
                )
        after_directory = os.fstat(directory_fd)
        if fields(before_directory) != fields(after_directory):
            raise TasteCleanPolicyError(
                f"{label} directory changed during stat inventory: {relative_directory}"
            )
        inventory[relative_directory] = fields(after_directory)

    visit(root_fd, "")
    return dict(sorted(inventory.items()))


def _validate_adapter_fd(
    output_fd: int,
    *,
    adapter_fd: int | None = None,
    expected_source_model_path: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], str, dict[str, Any]]:
    owned = adapter_fd is None
    if adapter_fd is None:
        adapter_fd = os.open(
            "adapter",
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=output_fd,
        )
    try:
        held = os.fstat(adapter_fd)
        named = os.stat("adapter", dir_fd=output_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(held.st_mode)
            or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise TasteCleanPolicyError("held adapter directory changed identity")
        return _validate_adapter_directory_fd(
            adapter_fd,
            expected_source_model_path=expected_source_model_path,
        )
    finally:
        if owned:
            os.close(adapter_fd)


def _validate_held_clean_policy_output(
    root: Path,
    root_fd: int,
    root_identity: tuple[int, int],
    adapter_fd: int | None = None,
    *,
    marker_name: str = "PASS",
) -> dict[str, Any]:
    if marker_name not in {"PASS", ".PASS.prepared"}:
        raise TasteCleanPolicyError("T5 marker leaf is not reviewed")
    held = os.fstat(root_fd)
    named = os.stat(root, follow_symlinks=False)
    if (
        (held.st_dev, held.st_ino) != root_identity
        or (named.st_dev, named.st_ino) != root_identity
        or not stat.S_ISDIR(held.st_mode)
        or stat.S_IMODE(held.st_mode) != 0o700
    ):
        raise TasteCleanPolicyError("T5 output physical root changed")
    names = set(os.listdir(root_fd))
    expected_names = (_TOP_LEVEL_FILES - {"PASS"}) | {marker_name, "adapter"}
    if names != expected_names:
        raise TasteCleanPolicyError("T5 output top-level inventory changed")
    payloads: dict[str, dict[str, Any]] = {}
    bytes_by_name: dict[str, bytes] = {}
    for name in sorted(_TOP_LEVEL_FILES - {"PASS"}):
        payload, data = _read_json_at(root_fd, name, label=f"T5 {name}")
        payloads[name] = payload
        bytes_by_name[name] = data
    pass_bytes = _read_regular_at(root_fd, marker_name, label="T5 PASS")
    if pass_bytes != (PASS_MARKER + "\n").encode("utf-8"):
        raise TasteCleanPolicyError("T5 PASS marker changed")

    provenance = payloads["policy_provenance.json"]
    manifest = payloads["manifest.json"]
    state = payloads["state.json"]
    gate = payloads["gate.json"]
    inputs = payloads["input_hashes.json"]
    outputs = payloads["output_hashes.json"]
    exact_keys = {
        "policy_provenance.json": {
            "schema_version", "dataset", "stage", "created_at",
            "policy_initialization_type", "source_model_classification",
            "source_model_path", "source_model_inventory_schema",
            "source_model_inventory_sha256", "reference_model_path",
            "reference_model_hash", "source_adapter_present", "source_adapter_path",
            "source_adapter_sha256", "produced_adapter_relative_path", "adapter_dir",
            "produced_adapter_inventory_schema", "produced_adapter_inventory",
            "produced_adapter_sha256", "adapter_tensor_schema",
            "adapter_parameter_sha256", "peft_reload_verified",
            "policy_initializer_hash", "reference_policy_hash",
            "adapter_initialized_from_scratch", "optimizer_step_count",
            "initializer_data_split_used", "taste_split_access_max",
            "taste_splits_loaded", "train_only_fallback_implemented",
            "rf_reference_count", "gnn_reward_used", "validation_loaded",
            "calibration_loaded", "test_loaded", "oracle_neutral",
            "data_redistributed", "public_release_allowed", "seed", "lora_rank",
            "lora_alpha", "lora_dropout",
        },
        "manifest.json": {
            "schema_version", "dataset", "stage", "status", "created_at",
            "initializer_mode", "initializer_data_split_used",
            "taste_split_access_max", "oracle_neutral", "rf_reference_count",
            "gnn_reward_used", "validation_loaded", "calibration_loaded",
            "test_loaded", "data_redistributed", "private_output",
            "hpc_execution_authorized", "gpu_identity", "policy_provenance_sha256",
            "input_hashes_sha256", "adapter_dir", "produced_adapter_sha256",
            "adapter_parameter_sha256", "peft_reload_verified",
            "policy_initializer_hash", "reference_model_hash", "reference_policy_hash",
            "source_model_inventory_sha256", "frozen_oracle_identity_sha256",
            "required_pass_marker",
        },
        "state.json": {
            "schema_version", "dataset", "stage", "status", "created_at",
            "updated_at", "release_authority_validated", "optimizer_step_count",
            "science_training_performed", "peft_reload_verified",
        },
        "gate.json": {
            "schema_version", "dataset", "stage", "status", "passed", "created_at",
            "same_frozen_gine_t3_t4_identity_bound", "policy_initializer_hash",
            "reference_model_hash", "reference_policy_hash",
            "adapter_parameter_sha256", "peft_reload_verified",
            "initializer_data_split_used",
            "taste_split_access_max", "rf_reference_count", "gnn_reward_used",
            "validation_loaded", "calibration_loaded", "test_loaded",
            "data_redistributed", "marker",
        },
        "input_hashes.json": {
            "schema_version", "dataset", "stage", "config_path", "config_sha256",
            "release_authority_path", "release_authority_sha256", "policy_path",
            "policy_file_sha256", "policy_canonical_sha256", "policy_receipt_path",
            "policy_receipt_sha256", "source_model_path",
            "source_model_inventory_sha256", "implementation_commit",
            "implementation_tree", "gpu_identity", "frozen_oracle",
        },
        "output_hashes.json": {
            "schema_version", "dataset", "stage", "adapter_inventory_schema",
            "adapter_inventory", "adapter_sha256", "policy_initializer_hash",
            "adapter_tensor_schema", "adapter_tensor_identity",
            "adapter_parameter_sha256", "peft_reload_verified",
            "reference_model_hash", "reference_policy_hash",
            "source_model_inventory_sha256",
            "policy_provenance_sha256", "manifest_sha256", "state_sha256",
            "gate_sha256", "input_hashes_sha256", "pass_marker_sha256",
        },
    }
    expected_schemas = {
        "policy_provenance.json": PROVENANCE_SCHEMA,
        "manifest.json": MANIFEST_SCHEMA,
        "state.json": STATE_SCHEMA,
        "gate.json": GATE_SCHEMA,
        "input_hashes.json": INPUT_HASHES_SCHEMA,
        "output_hashes.json": OUTPUT_HASHES_SCHEMA,
    }
    for name, schema in expected_schemas.items():
        _exact_keys(payloads[name], exact_keys[name], label=f"T5 {name}")
        if payloads[name].get("schema_version") != schema:
            raise TasteCleanPolicyError(f"T5 {name} schema changed")
        if payloads[name].get("dataset") != DATASET or payloads[name].get("stage") != STAGE:
            raise TasteCleanPolicyError(f"T5 {name} dataset/stage changed")
    if (
        gate.get("status") != "PASS"
        or gate.get("passed") is not True
        or state.get("status") != "PASS"
        or manifest.get("status") != "PASS"
        or gate.get("same_frozen_gine_t3_t4_identity_bound") is not True
        or state.get("release_authority_validated") is not True
        or state.get("science_training_performed") is not False
        or state.get("peft_reload_verified") is not True
    ):
        raise TasteCleanPolicyError("T5 terminal evidence is not PASS")
    created_at = provenance.get("created_at")
    if (
        not isinstance(created_at, str)
        or not created_at
        or manifest.get("created_at") != created_at
        or state.get("created_at") != created_at
        or state.get("updated_at") != created_at
        or gate.get("created_at") != created_at
    ):
        raise TasteCleanPolicyError("T5 terminal timestamps changed")
    for payload in (gate, manifest, provenance):
        if (
            type(payload.get("rf_reference_count")) is not int
            or payload.get("rf_reference_count") != 0
            or payload.get("gnn_reward_used") is not False
            or payload.get("validation_loaded") is not False
            or payload.get("calibration_loaded") is not False
            or payload.get("test_loaded") is not False
            or payload.get("initializer_data_split_used") != "none"
            or payload.get("taste_split_access_max") != "train_only"
        ):
            raise TasteCleanPolicyError("T5 clean/split boundary changed")
    if (
        provenance.get("policy_initialization_type") != INITIALIZER_MODE
        or provenance.get("source_model_classification") != SOURCE_CLASSIFICATION
        or provenance.get("source_model_inventory_schema") != SOURCE_INVENTORY_SCHEMA
        or provenance.get("produced_adapter_relative_path") != "adapter"
        or provenance.get("produced_adapter_inventory_schema")
        != ADAPTER_INVENTORY_SCHEMA
        or provenance.get("taste_splits_loaded") != []
        or provenance.get("optimizer_step_count") != 0
        or type(provenance.get("optimizer_step_count")) is not int
        or provenance.get("adapter_initialized_from_scratch") is not True
        or provenance.get("peft_reload_verified") is not True
        or provenance.get("source_adapter_present") is not False
        or provenance.get("source_adapter_path") is not None
        or provenance.get("source_adapter_sha256") is not None
        or provenance.get("train_only_fallback_implemented") is not False
        or provenance.get("oracle_neutral") is not True
        or provenance.get("data_redistributed") is not False
        or provenance.get("public_release_allowed") is not False
        or type(provenance.get("seed")) is not int
        or provenance.get("seed") < 0
        or type(provenance.get("lora_rank")) is not int
        or provenance.get("lora_rank") != 8
        or type(provenance.get("lora_alpha")) is not int
        or provenance.get("lora_alpha") != 16
        or type(provenance.get("lora_dropout")) is not float
        or provenance.get("lora_dropout") != 0.05
    ):
        raise TasteCleanPolicyError("T5 zero-step/source-adapter boundary changed")
    if (
        manifest.get("initializer_mode") != INITIALIZER_MODE
        or manifest.get("oracle_neutral") is not True
        or manifest.get("data_redistributed") is not False
        or manifest.get("private_output") is not True
        or manifest.get("hpc_execution_authorized") is not False
        or state.get("optimizer_step_count") != 0
        or type(state.get("optimizer_step_count")) is not int
        or gate.get("data_redistributed") is not False
        or outputs.get("adapter_inventory_schema") != ADAPTER_INVENTORY_SCHEMA
        or outputs.get("adapter_tensor_schema") != ADAPTER_TENSOR_SCHEMA
        or outputs.get("peft_reload_verified") is not True
        or manifest.get("peft_reload_verified") is not True
        or gate.get("peft_reload_verified") is not True
    ):
        raise TasteCleanPolicyError("T5 terminal execution boundary changed")
    for field in (
        "config_sha256",
        "release_authority_sha256",
        "policy_file_sha256",
        "policy_canonical_sha256",
        "policy_receipt_sha256",
    ):
        _hex(inputs.get(field), label=f"input_hashes.{field}")
    _commit(inputs.get("implementation_commit"), label="input_hashes.implementation_commit")
    _commit(inputs.get("implementation_tree"), label="input_hashes.implementation_tree")
    for field in (
        "config_path",
        "release_authority_path",
        "policy_path",
        "policy_receipt_path",
        "source_model_path",
    ):
        value = inputs.get(field)
        if (
            not isinstance(value, str)
            or not Path(value).is_absolute()
            or ".." in Path(value).parts
        ):
            raise TasteCleanPolicyError(f"input_hashes.{field} is not an absolute path")
    source_hash = _hex(inputs.get("source_model_inventory_sha256"), label="source_model_inventory_sha256")
    source_model_dir = _absolute(
        inputs.get("source_model_path"), label="bound source_model_path", must_exist=True
    )
    adapter_inventory, adapter_sha, adapter_tensor_identity = _validate_adapter_fd(
        root_fd,
        adapter_fd=adapter_fd,
        expected_source_model_path=source_model_dir,
    )
    policy_hash = _canonical_sha256(
        {
            "schema_version": "tastemolnet_clean_policy_identity_v1",
            "adapter_inventory_sha256": adapter_sha,
            "adapter_parameter_sha256": adapter_tensor_identity["parameter_sha256"],
            "source_model_inventory_sha256": source_hash,
            "initializer_data_split_used": "none",
            "optimizer_step_count": 0,
        }
    )
    reference_policy_hash = _reference_policy_hash(
        source_model_inventory_sha256=source_hash,
        adapter_inventory_sha256=adapter_sha,
        adapter_parameter_sha256=adapter_tensor_identity["parameter_sha256"],
    )
    if reference_policy_hash == source_hash:
        raise TasteCleanPolicyError("reference policy identity collapsed to the bare base")
    expected_adapter_dir = str(root / "adapter")
    if (
        provenance.get("adapter_dir") != expected_adapter_dir
        or manifest.get("adapter_dir") != expected_adapter_dir
        or provenance.get("produced_adapter_inventory") != adapter_inventory
        or outputs.get("adapter_inventory") != adapter_inventory
        or provenance.get("produced_adapter_sha256") != adapter_sha
        or manifest.get("produced_adapter_sha256") != adapter_sha
        or outputs.get("adapter_sha256") != adapter_sha
        or provenance.get("adapter_tensor_schema") != ADAPTER_TENSOR_SCHEMA
        or provenance.get("adapter_parameter_sha256")
        != adapter_tensor_identity["parameter_sha256"]
        or manifest.get("adapter_parameter_sha256")
        != adapter_tensor_identity["parameter_sha256"]
        or gate.get("adapter_parameter_sha256")
        != adapter_tensor_identity["parameter_sha256"]
        or outputs.get("adapter_parameter_sha256")
        != adapter_tensor_identity["parameter_sha256"]
        or outputs.get("adapter_tensor_identity") != adapter_tensor_identity
    ):
        raise TasteCleanPolicyError("T5 adapter inventory changed")
    for payload in (provenance, manifest, gate, outputs):
        if payload.get("policy_initializer_hash") != policy_hash:
            raise TasteCleanPolicyError("T5 policy initializer identity changed")
        if payload.get("reference_model_hash") != source_hash:
            raise TasteCleanPolicyError("T5 reference model identity changed")
        if payload.get("reference_policy_hash") != reference_policy_hash:
            raise TasteCleanPolicyError("T5 reference policy identity changed")
    if (
        provenance.get("source_model_path") != str(source_model_dir)
        or provenance.get("reference_model_path") != str(source_model_dir)
        or provenance.get("source_model_inventory_sha256") != source_hash
        or manifest.get("source_model_inventory_sha256") != source_hash
        or outputs.get("source_model_inventory_sha256") != source_hash
    ):
        raise TasteCleanPolicyError("T5 source model identity changed")

    expected_file_hashes = {
        "policy_provenance_sha256": "policy_provenance.json",
        "manifest_sha256": "manifest.json",
        "state_sha256": "state.json",
        "gate_sha256": "gate.json",
        "input_hashes_sha256": "input_hashes.json",
    }
    for field, name in expected_file_hashes.items():
        if outputs.get(field) != _sha256_bytes(bytes_by_name[name]):
            raise TasteCleanPolicyError(f"T5 output hash changed: {field}")
    if (
        manifest.get("policy_provenance_sha256")
        != _sha256_bytes(bytes_by_name["policy_provenance.json"])
        or manifest.get("input_hashes_sha256")
        != _sha256_bytes(bytes_by_name["input_hashes.json"])
        or outputs.get("pass_marker_sha256") != _sha256_bytes(pass_bytes)
        or gate.get("marker") != PASS_MARKER
        or manifest.get("required_pass_marker") != PASS_MARKER
    ):
        raise TasteCleanPolicyError("T5 manifest/marker hash closure changed")
    frozen_oracle = _mapping(inputs.get("frozen_oracle"), label="frozen_oracle")
    _exact_keys(
        frozen_oracle,
        {
            "dataset", "backbone", "num_classes", "label_map", "source_label",
            "strict_flip", "rf_oracle_used", "checkpoint_dir", "checkpoint_id",
            "checkpoint_sha256", "checkpoint_inventory_sha256",
            "checkpoint_stat_inventory_sha256", "checkpoint_sha256s_sha256",
            "feature_schema_sha256", "temperature_calibration_sha256",
            "downstream_policy_sha256", "t2_adoption_binding",
            "t3_output_root", "t3_gate_sha256",
            "t3_root_inventory_sha256", "t4_output_root", "t4_gate_sha256",
            "t4_root_inventory_sha256",
        },
        label="frozen_oracle",
    )
    gpu_identity = _mapping(inputs.get("gpu_identity"), label="gpu_identity")
    _exact_keys(
        gpu_identity,
        {
            "physical_gpu_index", "gpu_uuid", "cuda_visible_devices",
            "controller_binding_state", "gpu_lock_authority_present",
            "execution_receipt_present", "controller_id", "controller_task_id",
        },
        label="gpu_identity",
    )
    if (
        type(gpu_identity.get("physical_gpu_index")) is not int
        or gpu_identity.get("physical_gpu_index") != 2
        or gpu_identity.get("cuda_visible_devices") != "2"
        or gpu_identity.get("controller_binding_state") != "controller_declared_only"
        or gpu_identity.get("gpu_lock_authority_present") is not False
        or gpu_identity.get("execution_receipt_present") is not False
        or not isinstance(gpu_identity.get("controller_id"), str)
        or not gpu_identity.get("controller_id")
        or not isinstance(gpu_identity.get("controller_task_id"), str)
        or not gpu_identity.get("controller_task_id")
        or not isinstance(gpu_identity.get("gpu_uuid"), str)
        or _GPU_UUID_RE.fullmatch(str(gpu_identity.get("gpu_uuid"))) is None
        or manifest.get("gpu_identity") != gpu_identity
    ):
        raise TasteCleanPolicyError("T5 GPU/controller identity changed")
    if manifest.get("frozen_oracle_identity_sha256") != _canonical_sha256(frozen_oracle):
        raise TasteCleanPolicyError("T5 frozen GINE/T3/T4 identity changed")
    t2_binding = _validate_t2_adoption_binding(
        frozen_oracle.get("t2_adoption_binding"),
        label="frozen_oracle.t2_adoption_binding",
    )
    for key in (
        "checkpoint_sha256",
        "checkpoint_id",
        "checkpoint_inventory_sha256",
        "checkpoint_stat_inventory_sha256",
        "checkpoint_sha256s_sha256",
        "feature_schema_sha256",
        "temperature_calibration_sha256",
        "downstream_policy_sha256",
        "t3_gate_sha256",
        "t3_root_inventory_sha256",
        "t4_gate_sha256",
        "t4_root_inventory_sha256",
    ):
        _hex(frozen_oracle.get(key), label=f"frozen_oracle.{key}")
    if (
        frozen_oracle.get("dataset") != DATASET
        or frozen_oracle.get("backbone") != "gine"
        or type(frozen_oracle.get("num_classes")) is not int
        or frozen_oracle.get("num_classes") != 3
        or frozen_oracle.get("label_map") != LABEL_MAP
        or type(frozen_oracle.get("source_label")) is not int
        or frozen_oracle.get("source_label") != 1
        or frozen_oracle.get("strict_flip")
        != "pred_before == 1 and pred_after != 1"
        or frozen_oracle.get("rf_oracle_used") is not False
        or frozen_oracle.get("checkpoint_id") != frozen_oracle.get("checkpoint_sha256")
        or t2_binding.get("formal_bundle_root")
        != frozen_oracle.get("checkpoint_dir")
        or t2_binding.get("formal_bundle_model_sha256")
        != frozen_oracle.get("checkpoint_sha256")
        or t2_binding.get("formal_bundle_sha256s_sha256")
        != frozen_oracle.get("checkpoint_sha256s_sha256")
    ):
        raise TasteCleanPolicyError("T5 frozen-oracle semantics changed")
    for field in ("checkpoint_dir", "t3_output_root", "t4_output_root"):
        value = frozen_oracle.get(field)
        if (
            not isinstance(value, str)
            or not Path(value).is_absolute()
            or ".." in Path(value).parts
        ):
            raise TasteCleanPolicyError(f"T5 frozen-oracle {field} changed")

    top_level_hashes = {
        name: _sha256_bytes(data) for name, data in sorted(bytes_by_name.items())
    }
    top_level_hashes["PASS"] = _sha256_bytes(pass_bytes)
    output_inventory_sha = _canonical_sha256(
        {
            "schema_version": "tastemolnet_clean_policy_output_inventory_v1",
            "top_level": top_level_hashes,
            "adapter_inventory": adapter_inventory,
        }
    )
    final_held = os.fstat(root_fd)
    final_named = os.stat(root, follow_symlinks=False)
    if (
        (final_held.st_dev, final_held.st_ino) != root_identity
        or (final_named.st_dev, final_named.st_ino) != root_identity
    ):
        raise TasteCleanPolicyError("T5 output root changed during terminal reopen")
    return {
        "schema_version": OUTPUT_AUTHORITY_SCHEMA,
        "status": "PASS",
        "stage": STAGE,
        "output_root": str(root),
        "adapter_dir": expected_adapter_dir,
        "source_model_dir": str(source_model_dir),
        "source_model_path": str(source_model_dir),
        "policy_initializer_hash": policy_hash,
        "reference_model_hash": source_hash,
        "reference_policy_hash": reference_policy_hash,
        "source_model_inventory_sha256": source_hash,
        "adapter_sha256": adapter_sha,
        "manifest_sha256": _sha256_bytes(bytes_by_name["manifest.json"]),
        "gate_sha256": _sha256_bytes(bytes_by_name["gate.json"]),
        "t5_gate_sha256": _sha256_bytes(bytes_by_name["gate.json"]),
        "pass_sha256": _sha256_bytes(pass_bytes),
        "t5_pass_sha256": _sha256_bytes(pass_bytes),
        "input_hashes_sha256": _sha256_bytes(bytes_by_name["input_hashes.json"]),
        "output_hashes_sha256": _sha256_bytes(bytes_by_name["output_hashes.json"]),
        "output_inventory_sha256": output_inventory_sha,
        "root_inventory_sha256": output_inventory_sha,
        "t5_output_inventory_sha256": output_inventory_sha,
        "frozen_oracle_identity": dict(frozen_oracle),
        "frozen_oracle_identity_sha256": _canonical_sha256(frozen_oracle),
        "gpu_identity": dict(gpu_identity),
        "marker": PASS_MARKER,
    }


@dataclass(slots=True)
class HeldTasteCleanPolicyOutput:
    """A repeatable T5 terminal validator that retains the physical root."""

    root: Path
    descriptor: int
    root_identity: tuple[int, int]
    adapter_descriptor: int
    adapter_identity: tuple[int, int]
    physical_stat_inventory: Mapping[str, Mapping[str, int]]
    evidence: Mapping[str, Any]

    def stable_adapter_load_path(self) -> Path:
        self.revalidate()
        return _fd_directory_path(
            self.adapter_descriptor, label="held Taste zero-step adapter"
        )

    def verify_loaded_adapter(self, model: Any) -> str:
        """Bind an actual PEFT in-memory adapter back to this held T5 output."""

        self.revalidate()
        try:
            from peft import get_peft_model_state_dict
        except ImportError as exc:  # pragma: no cover - AutoDL dependency
            raise TasteCleanPolicyError("loaded-adapter verification requires peft") from exc
        observed = _lora_tensor_identity(
            get_peft_model_state_dict(model), label="T6 loaded in-memory LoRA"
        )
        outputs, _data = _read_json_at(
            self.descriptor, "output_hashes.json", label="held T5 output hashes"
        )
        if observed != outputs.get("adapter_tensor_identity"):
            raise TasteCleanPolicyError("loaded T6 adapter differs from held T5 bytes")
        if self.evidence["reference_policy_hash"] != outputs.get("reference_policy_hash"):
            raise TasteCleanPolicyError("loaded T6 reference-policy identity changed")
        self.revalidate()
        return str(self.evidence["reference_policy_hash"])

    def revalidate(self) -> dict[str, Any]:
        if self.descriptor < 0:
            raise TasteCleanPolicyError("held T5 output authority is closed")
        before = _physical_stat_inventory_fd(
            self.descriptor, label="held T5 output"
        )
        if before != dict(self.physical_stat_inventory):
            raise TasteCleanPolicyError("held T5 output physical stat inventory changed")
        result = _validate_held_clean_policy_output(
            self.root,
            self.descriptor,
            self.root_identity,
            self.adapter_descriptor,
        )
        after = _physical_stat_inventory_fd(
            self.descriptor, label="held T5 output"
        )
        if after != before:
            raise TasteCleanPolicyError(
                "held T5 output physical stat inventory changed during validation"
            )
        if dict(result) != dict(self.evidence):
            raise TasteCleanPolicyError("held T5 output evidence changed")
        return result

    def close(self) -> None:
        if self.adapter_descriptor >= 0:
            os.close(self.adapter_descriptor)
            self.adapter_descriptor = -1
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> "HeldTasteCleanPolicyOutput":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def hold_clean_policy_output(output_root: str | Path) -> HeldTasteCleanPolicyOutput:
    root = _absolute(output_root, label="T5 output", must_exist=True)
    descriptor = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    adapter_descriptor = -1
    try:
        info = os.fstat(descriptor)
        identity = (int(info.st_dev), int(info.st_ino))
        adapter_descriptor = os.open(
            "adapter",
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=descriptor,
        )
        adapter_info = os.fstat(adapter_descriptor)
        adapter_named = os.stat("adapter", dir_fd=descriptor, follow_symlinks=False)
        if (
            not stat.S_ISDIR(adapter_info.st_mode)
            or (adapter_info.st_dev, adapter_info.st_ino)
            != (adapter_named.st_dev, adapter_named.st_ino)
        ):
            raise TasteCleanPolicyError("T5 adapter physical identity changed")
        adapter_identity = (int(adapter_info.st_dev), int(adapter_info.st_ino))
        physical_stat_inventory = _physical_stat_inventory_fd(
            descriptor, label="held T5 output"
        )
        evidence = _validate_held_clean_policy_output(
            root, descriptor, identity, adapter_descriptor
        )
        if _physical_stat_inventory_fd(
            descriptor, label="held T5 output"
        ) != physical_stat_inventory:
            raise TasteCleanPolicyError(
                "T5 output physical stat inventory changed while it was held"
            )
        return HeldTasteCleanPolicyOutput(
            root=root,
            descriptor=descriptor,
            root_identity=identity,
            adapter_descriptor=adapter_descriptor,
            adapter_identity=adapter_identity,
            physical_stat_inventory=physical_stat_inventory,
            evidence=evidence,
        )
    except Exception:
        if adapter_descriptor >= 0:
            os.close(adapter_descriptor)
        os.close(descriptor)
        raise


def validate_clean_policy_output(output_root: str | Path) -> dict[str, Any]:
    """Read-only one-shot validation; T6 should use the held form instead."""

    with hold_clean_policy_output(output_root) as authority:
        return authority.revalidate()


@dataclass(slots=True)
class HeldTasteCleanPolicySourceModel:
    """Hold and repeatedly validate the exact generic base consumed by T5/T6."""

    source_model_dir: Path
    descriptor: int
    root_identity: tuple[int, int]
    expected_inventory_sha256: str
    physical_stat_inventory: Mapping[str, Mapping[str, int]]
    evidence: Mapping[str, Any]

    def stable_load_path(self) -> Path:
        self.revalidate()
        return _fd_directory_path(self.descriptor, label="held Taste source model")

    def revalidate(self) -> dict[str, Any]:
        if self.descriptor < 0:
            raise TasteCleanPolicyError("held Taste source-model authority is closed")
        held = os.fstat(self.descriptor)
        named = os.stat(self.source_model_dir, follow_symlinks=False)
        if (
            (held.st_dev, held.st_ino) != self.root_identity
            or (named.st_dev, named.st_ino) != self.root_identity
        ):
            raise TasteCleanPolicyError("Taste source-model physical root changed")
        before_stat = _physical_stat_inventory_fd(
            self.descriptor, label="held generic ChemLLM base"
        )
        if before_stat != dict(self.physical_stat_inventory):
            raise TasteCleanPolicyError(
                "Taste source-model physical stat inventory changed"
            )
        inventory, digest = _inventory_directory_fd(
            self.descriptor,
            label="held generic ChemLLM base",
            schema_version=SOURCE_INVENTORY_SCHEMA,
            require_private=False,
        )
        if "config.json" not in inventory:
            raise TasteCleanPolicyError("held generic ChemLLM base lacks config.json")
        if any(
            name.endswith("adapter_config.json")
            or name.endswith("adapter_model.safetensors")
            or name.endswith("adapter_model.bin")
            for name in inventory
        ):
            raise TasteCleanPolicyError("held generic ChemLLM base contains an adapter")
        lowered = [self.source_model_dir.name.lower(), *(name.lower() for name in inventory)]
        if any(
            token in item
            for token in _FORBIDDEN_SOURCE_TOKENS
            for item in lowered
        ):
            raise TasteCleanPolicyError("held generic ChemLLM base has dataset/RF evidence")
        config, _config_data = _read_json_at(
            self.descriptor,
            "config.json",
            label="held ChemLLM config",
            require_private=False,
        )
        model_type = str(config.get("model_type") or "").lower()
        config_identity = json.dumps(config, sort_keys=True).lower()
        if (
            "chemllm" not in self.source_model_dir.name.lower()
            and "chemllm" not in config_identity
        ) or model_type not in {"internlm2", "internlm"}:
            raise TasteCleanPolicyError("held source is not a generic ChemLLM base")
        result = {
            "source_model_inventory_sha256": digest,
        }
        if result.get("source_model_inventory_sha256") != self.expected_inventory_sha256:
            raise TasteCleanPolicyError("Taste source-model inventory changed")
        final_held = os.fstat(self.descriptor)
        final_named = os.stat(self.source_model_dir, follow_symlinks=False)
        if (
            (final_held.st_dev, final_held.st_ino) != self.root_identity
            or (final_named.st_dev, final_named.st_ino) != self.root_identity
        ):
            raise TasteCleanPolicyError(
                "Taste source-model physical root changed during inventory"
            )
        if _physical_stat_inventory_fd(
            self.descriptor, label="held generic ChemLLM base"
        ) != before_stat:
            raise TasteCleanPolicyError(
                "Taste source-model physical stat inventory changed during inventory"
            )
        stable = {
            "schema_version": "tastemolnet_clean_policy_source_model_authority_v1",
            "source_model_dir": str(self.source_model_dir),
            "source_model_path": str(self.source_model_dir),
            "source_model_inventory_sha256": self.expected_inventory_sha256,
            "classification": SOURCE_CLASSIFICATION,
            "dataset_specific": False,
            "source_adapter_present": False,
        }
        if dict(stable) != dict(self.evidence):
            raise TasteCleanPolicyError("Taste source-model authority evidence changed")
        return stable

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> "HeldTasteCleanPolicySourceModel":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def hold_source_model_for_clean_policy(
    model_path: str | Path,
    expected_inventory_sha256: str,
) -> HeldTasteCleanPolicySourceModel:
    source = _absolute(model_path, label="Taste source_model", must_exist=True)
    expected = _hex(
        expected_inventory_sha256, label="expected_source_model_inventory_sha256"
    )
    descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        info = os.fstat(descriptor)
        identity = (int(info.st_dev), int(info.st_ino))
        physical_stat_inventory = _physical_stat_inventory_fd(
            descriptor, label="held generic ChemLLM base"
        )
        evidence = {
            "schema_version": "tastemolnet_clean_policy_source_model_authority_v1",
            "source_model_dir": str(source),
            "source_model_path": str(source),
            "source_model_inventory_sha256": expected,
            "classification": SOURCE_CLASSIFICATION,
            "dataset_specific": False,
            "source_adapter_present": False,
        }
        authority = HeldTasteCleanPolicySourceModel(
            source_model_dir=source,
            descriptor=descriptor,
            root_identity=identity,
            expected_inventory_sha256=expected,
            physical_stat_inventory=physical_stat_inventory,
            evidence=evidence,
        )
        authority.revalidate()
        return authority
    except Exception:
        os.close(descriptor)
        raise


def validate_source_model_for_clean_policy(
    model_path: str | Path,
    expected_inventory_sha256: str,
) -> dict[str, Any]:
    """One-shot source check; T6 should hold and revalidate around model load."""

    with hold_source_model_for_clean_policy(
        model_path, expected_inventory_sha256
    ) as authority:
        return authority.revalidate()


@dataclass(frozen=True, slots=True)
class TasteCleanPolicyLoadToken:
    """Exact descriptor-backed paths and identities for one held T6 load."""

    output_root: Path
    source_model_load_path: Path
    adapter_load_path: Path
    source_model_inventory_sha256: str
    adapter_inventory_sha256: str
    adapter_parameter_sha256: str
    reference_policy_hash: str
    t5_output_inventory_sha256: str
    frozen_oracle_identity_sha256: str

    def evidence(self) -> dict[str, str]:
        return {
            "schema_version": HELD_LOAD_TOKEN_SCHEMA,
            "output_root": str(self.output_root),
            "source_model_load_path": str(self.source_model_load_path),
            "adapter_load_path": str(self.adapter_load_path),
            "source_model_inventory_sha256": self.source_model_inventory_sha256,
            "adapter_inventory_sha256": self.adapter_inventory_sha256,
            "adapter_parameter_sha256": self.adapter_parameter_sha256,
            "reference_policy_hash": self.reference_policy_hash,
            "t5_output_inventory_sha256": self.t5_output_inventory_sha256,
            "frozen_oracle_identity_sha256": self.frozen_oracle_identity_sha256,
        }


@dataclass(slots=True)
class HeldTasteCleanPolicyLoadAuthority:
    """Hold T5, adapter, and source across every T6 HF/PEFT loader call."""

    output: HeldTasteCleanPolicyOutput
    source: HeldTasteCleanPolicySourceModel

    def _current_token(self) -> TasteCleanPolicyLoadToken:
        output_evidence = self.output.revalidate()
        source_evidence = self.source.revalidate()
        if (
            source_evidence.get("source_model_dir")
            != output_evidence.get("source_model_dir")
            or source_evidence.get("source_model_inventory_sha256")
            != output_evidence.get("source_model_inventory_sha256")
        ):
            raise TasteCleanPolicyError("held T5 output/source authorities differ")
        output_hashes, _data = _read_json_at(
            self.output.descriptor,
            "output_hashes.json",
            label="held T5 load-token output hashes",
        )
        token = TasteCleanPolicyLoadToken(
            output_root=self.output.root,
            source_model_load_path=self.source.stable_load_path(),
            adapter_load_path=self.output.stable_adapter_load_path(),
            source_model_inventory_sha256=_hex(
                output_evidence.get("source_model_inventory_sha256"),
                label="held load token source inventory",
            ),
            adapter_inventory_sha256=_hex(
                output_evidence.get("adapter_sha256"),
                label="held load token adapter inventory",
            ),
            adapter_parameter_sha256=_hex(
                output_hashes.get("adapter_parameter_sha256"),
                label="held load token adapter parameters",
            ),
            reference_policy_hash=_hex(
                output_evidence.get("reference_policy_hash"),
                label="held load token reference policy",
            ),
            t5_output_inventory_sha256=_hex(
                output_evidence.get("t5_output_inventory_sha256"),
                label="held load token T5 output inventory",
            ),
            frozen_oracle_identity_sha256=_hex(
                output_evidence.get("frozen_oracle_identity_sha256"),
                label="held load token frozen oracle",
            ),
        )
        # Both stat/ctime authorities are checked after creating the fd paths;
        # any lexical swap and restoration remains observable and fails here.
        self.source.revalidate()
        self.output.revalidate()
        return token

    def load_token(self) -> TasteCleanPolicyLoadToken:
        return self._current_token()

    def revalidate_load_token(
        self, token: TasteCleanPolicyLoadToken
    ) -> TasteCleanPolicyLoadToken:
        if type(token) is not TasteCleanPolicyLoadToken:
            raise TasteCleanPolicyError("T5 load token has the wrong native type")
        current = self._current_token()
        if current != token:
            raise TasteCleanPolicyError("held T5 load token changed")
        return current

    def verify_loaded_policy(
        self,
        model: Any,
        *,
        token: TasteCleanPolicyLoadToken,
        role: str,
    ) -> str:
        """Verify a policy/reference adapter and both retained authorities."""

        if type(role) is not str or role not in {"policy", "reference"}:
            raise TasteCleanPolicyError("loaded T5 role must be policy or reference")
        self.revalidate_load_token(token)
        observed = self.output.verify_loaded_adapter(model)
        if observed != token.reference_policy_hash:
            raise TasteCleanPolicyError(f"loaded T5 {role} reference identity changed")
        self.revalidate_load_token(token)
        return observed

    def revalidate(self) -> TasteCleanPolicyLoadToken:
        """Call after each tokenizer/base/value loader that has no adapter state."""

        return self._current_token()

    def close(self) -> None:
        self.source.close()
        self.output.close()

    def __enter__(self) -> "HeldTasteCleanPolicyLoadAuthority":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def hold_clean_policy_load_authority(
    output_root: str | Path,
) -> HeldTasteCleanPolicyLoadAuthority:
    """Open the only supported non-path-only T6 policy loading authority."""

    output = hold_clean_policy_output(output_root)
    source: HeldTasteCleanPolicySourceModel | None = None
    try:
        source = hold_source_model_for_clean_policy(
            output.evidence["source_model_dir"],
            output.evidence["source_model_inventory_sha256"],
        )
        authority = HeldTasteCleanPolicyLoadAuthority(output=output, source=source)
        authority.revalidate()
        return authority
    except Exception:
        if source is not None:
            source.close()
        output.close()
        raise


__all__ = [
    "ADAPTER_INVENTORY_SCHEMA",
    "CONFIG_SCHEMA",
    "DATASET",
    "GATE_SCHEMA",
    "HELD_LOAD_TOKEN_SCHEMA",
    "INITIALIZER_MODE",
    "MANIFEST_SCHEMA",
    "PASS_MARKER",
    "PROVENANCE_SCHEMA",
    "RELEASE_AUTHORITY_SCHEMA",
    "STAGE",
    "TasteCleanPolicyConfig",
    "TasteCleanPolicyError",
    "HeldTasteCleanPolicyOutput",
    "HeldTasteCleanPolicyLoadAuthority",
    "HeldTasteCleanPolicySourceModel",
    "HeldTasteFrozenOracleAuthority",
    "TasteCleanPolicyLoadToken",
    "TasteCleanPolicyReleaseAuthority",
    "TasteCleanPolicyReleaseDisabled",
    "TasteManagedEvidenceBindingV2",
    "TasteFrozenOracleIdentity",
    "build_clean_policy_initializer",
    "hold_taste_managed_evidence_binding_v2",
    "inspect_generic_chemllm_base",
    "hold_clean_policy_output",
    "hold_clean_policy_load_authority",
    "hold_source_model_for_clean_policy",
    "load_clean_policy_config",
    "load_release_authority",
    "validate_clean_policy_output",
    "validate_source_model_for_clean_policy",
]
