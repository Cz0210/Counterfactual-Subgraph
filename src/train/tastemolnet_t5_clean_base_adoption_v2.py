"""Managed-v2 adoption of one clean generic ChemLLM base for Taste T5.

This is deliberately an adoption route, not SFT.  The worker inventories the
external read-only Hugging Face model tree and writes only a small candidate
receipt.  A separate verifier rehashes the complete source tree, checks the
safetensors index/shard closure, proves the managed worker lineage, and is the
only process allowed to publish PASS.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
from typing import Any, Mapping

from src.train.tastemolnet_clean_policy_init import (
    TasteCleanPolicyError,
    _physical_stat_inventory_fd,
    inspect_generic_chemllm_base,
)
from src.utils.managed_execution_v2 import (
    WORKER_EXIT_SCHEMA,
    WORKER_RAW_EVIDENCE_SCHEMA,
)
from src.utils.terminal_publisher_v2 import (
    HeldSealedArtifactV2,
    TerminalPublicationV2,
    open_sealed_worker_artifact,
    verify_and_publish_sealed_attempt,
)


SCHEMA_VERSION = "tastemolnet_t5_clean_base_adoption_v2"
SOURCE_EVIDENCE_SCHEMA = "tastemolnet_t5_clean_chemllm_source_v2"
HF_CLOSURE_SCHEMA = "tastemolnet_t5_hf_safetensors_closure_v2"
STAGE = "T5_CLEAN_POLICY_READY"
TASK_ID = "T5_CLEAN_BASE_ADOPTION"
SEMANTIC_STATE = "ADOPTED_CLEAN_GENERIC_BASE"
PASS_MARKER = "[TASTE_T5_CLEAN_SFT_PASS]"
SOURCE_EVIDENCE_NAME = "source_inventory.json"
CANDIDATE_NAME = "clean_base_adoption_candidate.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MODEL_SHARD_RE = re.compile(r"^model-[0-9]{5}-of-[0-9]{5}\.safetensors$")
_FORBIDDEN_SOURCE_MARKERS = (
    "tastemolnet",
    "mutagenicity",
    "random_forest",
    "randomforest",
    "rf_oracle",
    "morgan_rf",
    "rf_ranked",
    "bace",
    "hiv",
    "aids",
)
_FORBIDDEN_ADAPTER_MARKERS = ("adapter", "peft", "lora", "qlora")
_DATA_SUFFIXES = {
    ".arrow",
    ".csv",
    ".feather",
    ".jsonl",
    ".npy",
    ".npz",
    ".parquet",
    ".pickle",
    ".pkl",
    ".smi",
    ".smiles",
    ".tsv",
}


class TasteT5CleanBaseAdoptionError(RuntimeError):
    """The clean-base source or its managed-v2 evidence is not releasable."""


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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_object(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteT5CleanBaseAdoptionError(f"{label} is not valid JSON") from exc
    if type(value) is not dict:
        raise TasteT5CleanBaseAdoptionError(f"{label} must be one JSON object")
    return value


def _read_regular(path: Path, *, label: str, maximum_bytes: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise TasteT5CleanBaseAdoptionError(
                f"{label} must be one physical single-link file"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, maximum_bytes - total + 1))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > maximum_bytes:
                raise TasteT5CleanBaseAdoptionError(
                    f"{label} exceeds the bounded metadata read"
                )
        after = os.fstat(descriptor)
        named_after = os.stat(path, follow_symlinks=False)
        identity = lambda item: (  # noqa: E731
            int(item.st_dev),
            int(item.st_ino),
            int(item.st_size),
            int(item.st_mtime_ns),
            int(item.st_ctime_ns),
            int(item.st_nlink),
        )
        if identity(before) != identity(after) or identity(after) != identity(named_after):
            raise TasteT5CleanBaseAdoptionError(f"{label} changed while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    return _json_object(
        _read_regular(path, label=label, maximum_bytes=64 * 1024 * 1024),
        label=label,
    )


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    data = _canonical_json_bytes(payload)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(data)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise TasteT5CleanBaseAdoptionError(f"short write for {path.name}")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return _sha256(data)


def _require_sha256(value: str, *, label: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise TasteT5CleanBaseAdoptionError(f"{label} is not lowercase SHA-256")
    return value


def _safe_relative(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise TasteT5CleanBaseAdoptionError("source inventory path is unsafe")
    return relative


def _assert_no_task_payload(files: Mapping[str, Mapping[str, Any]]) -> None:
    for raw_name in files:
        relative = _safe_relative(raw_name)
        lowered = raw_name.lower()
        if any(marker in lowered for marker in _FORBIDDEN_SOURCE_MARKERS):
            raise TasteT5CleanBaseAdoptionError(
                f"source tree contains Taste/RF/dataset-specific path: {raw_name}"
            )
        if any(marker in lowered for marker in _FORBIDDEN_ADAPTER_MARKERS):
            raise TasteT5CleanBaseAdoptionError(
                f"source tree contains adapter/PEFT path: {raw_name}"
            )
        if relative.suffix.lower() in _DATA_SUFFIXES or any(
            part.lower() in {"data", "dataset", "datasets"}
            for part in relative.parts[:-1]
        ):
            raise TasteT5CleanBaseAdoptionError(
                f"source tree contains a dataset payload: {raw_name}"
            )


def _metadata_contains_forbidden(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            for marker in (*_FORBIDDEN_SOURCE_MARKERS, *_FORBIDDEN_ADAPTER_MARKERS):
                if marker in lowered:
                    return marker
            found = _metadata_contains_forbidden(item)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _metadata_contains_forbidden(item)
            if found is not None:
                return found
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in _FORBIDDEN_SOURCE_MARKERS:
            if marker in lowered:
                return marker
    return None


def _safetensors_header(path: Path, *, expected_size: int) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != expected_size
            or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise TasteT5CleanBaseAdoptionError(
                f"safetensors shard authority changed: {path.name}"
            )
        prefix = os.read(descriptor, 8)
        if len(prefix) != 8:
            raise TasteT5CleanBaseAdoptionError(
                f"safetensors shard header is truncated: {path.name}"
            )
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size <= 1 or header_size > 64 * 1024 * 1024:
            raise TasteT5CleanBaseAdoptionError(
                f"safetensors header length is unsafe: {path.name}"
            )
        if 8 + header_size >= before.st_size:
            raise TasteT5CleanBaseAdoptionError(
                f"safetensors shard has no tensor payload: {path.name}"
            )
        header_data = bytearray()
        while len(header_data) < header_size:
            block = os.read(descriptor, header_size - len(header_data))
            if not block:
                raise TasteT5CleanBaseAdoptionError(
                    f"safetensors header is truncated: {path.name}"
                )
            header_data.extend(block)
        try:
            header = json.loads(bytes(header_data).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TasteT5CleanBaseAdoptionError(
                f"safetensors header is invalid JSON: {path.name}"
            ) from exc
        after = os.fstat(descriptor)
        named_after = os.stat(path, follow_symlinks=False)
        identity = lambda item: (  # noqa: E731
            int(item.st_dev),
            int(item.st_ino),
            int(item.st_size),
            int(item.st_mtime_ns),
            int(item.st_ctime_ns),
        )
        if identity(before) != identity(after) or identity(after) != identity(named_after):
            raise TasteT5CleanBaseAdoptionError(
                f"safetensors shard changed while its header was read: {path.name}"
            )
    finally:
        os.close(descriptor)
    if type(header) is not dict:
        raise TasteT5CleanBaseAdoptionError(
            f"safetensors header must be one mapping: {path.name}"
        )
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    if not tensors:
        raise TasteT5CleanBaseAdoptionError(
            f"safetensors shard contains no tensors: {path.name}"
        )
    payload_size = expected_size - 8 - header_size
    intervals: list[tuple[int, int]] = []
    for name, raw in tensors.items():
        if type(name) is not str or type(raw) is not dict:
            raise TasteT5CleanBaseAdoptionError(
                f"safetensors tensor entry is malformed: {path.name}"
            )
        dtype = raw.get("dtype")
        shape = raw.get("shape")
        offsets = raw.get("data_offsets")
        if (
            type(dtype) is not str
            or not isinstance(shape, list)
            or any(type(item) is not int or item < 0 for item in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(item) is not int for item in offsets)
        ):
            raise TasteT5CleanBaseAdoptionError(
                f"safetensors tensor metadata is malformed: {path.name}:{name}"
            )
        start, end = offsets
        if start < 0 or end <= start or end > payload_size:
            raise TasteT5CleanBaseAdoptionError(
                f"safetensors tensor offsets are invalid: {path.name}:{name}"
            )
        intervals.append((start, end))
    intervals.sort()
    cursor = 0
    for start, end in intervals:
        if start != cursor:
            raise TasteT5CleanBaseAdoptionError(
                f"safetensors shard payload is not a closed interval: {path.name}"
            )
        cursor = end
    if cursor != payload_size:
        raise TasteT5CleanBaseAdoptionError(
            f"safetensors shard has unindexed payload bytes: {path.name}"
        )
    return {
        "tensor_names": sorted(tensors),
        "tensor_count": len(tensors),
        "header_bytes": int(header_size),
        "tensor_payload_bytes": int(payload_size),
    }


def inspect_clean_chemllm_base(source_model: str | Path) -> dict[str, Any]:
    """Hash and validate one complete generic HF ChemLLM source tree."""

    try:
        base = inspect_generic_chemllm_base(source_model, include_files=True)
    except (TasteCleanPolicyError, OSError) as exc:
        raise TasteT5CleanBaseAdoptionError(str(exc)) from exc
    root = Path(str(base["source_model_path"]))
    raw_files = base.get("source_model_files")
    if not isinstance(raw_files, Mapping) or not raw_files:
        raise TasteT5CleanBaseAdoptionError("complete source inventory is absent")
    files: dict[str, dict[str, Any]] = {}
    for name, raw in sorted(raw_files.items()):
        if type(name) is not str or not isinstance(raw, Mapping):
            raise TasteT5CleanBaseAdoptionError("source file inventory is malformed")
        size = raw.get("bytes")
        digest = raw.get("sha256")
        if type(size) is not int or size <= 0 or not isinstance(digest, str):
            raise TasteT5CleanBaseAdoptionError(
                f"source file evidence is malformed: {name}"
            )
        files[name] = {"bytes": size, "sha256": _require_sha256(digest, label=name)}
    _assert_no_task_payload(files)

    required_metadata = {"config.json", "tokenizer_config.json", "model.safetensors.index.json"}
    missing = required_metadata - set(files)
    if missing:
        raise TasteT5CleanBaseAdoptionError(
            f"generic ChemLLM source lacks required HF metadata: {sorted(missing)}"
        )
    config = _read_json(root / "config.json", label="ChemLLM config")
    tokenizer_config = _read_json(
        root / "tokenizer_config.json", label="ChemLLM tokenizer config"
    )
    index = _read_json(
        root / "model.safetensors.index.json", label="ChemLLM safetensors index"
    )
    for label, document in (
        ("config", config),
        ("tokenizer config", tokenizer_config),
        ("safetensors index", index),
    ):
        forbidden = _metadata_contains_forbidden(document)
        if forbidden is not None:
            raise TasteT5CleanBaseAdoptionError(
                f"{label} contains forbidden task/adapter metadata: {forbidden}"
            )
    model_type = str(config.get("model_type") or "").lower()
    architectures = config.get("architectures")
    if (
        model_type not in {"internlm", "internlm2"}
        or not isinstance(architectures, list)
        or len(architectures) != 1
        or type(architectures[0]) is not str
        or "causallm" not in architectures[0].lower()
        or type(config.get("vocab_size")) is not int
        or int(config["vocab_size"]) <= 0
    ):
        raise TasteT5CleanBaseAdoptionError("ChemLLM causal-LM config is incomplete")
    tokenizer_class = tokenizer_config.get("tokenizer_class")
    if type(tokenizer_class) is not str or "internlm" not in tokenizer_class.lower():
        raise TasteT5CleanBaseAdoptionError("ChemLLM tokenizer class is not InternLM")
    tokenizer_assets = [name for name in ("tokenizer.model", "tokenizer.json") if name in files]
    if len(tokenizer_assets) != 1:
        raise TasteT5CleanBaseAdoptionError(
            "ChemLLM source must contain exactly one tokenizer.model/tokenizer.json asset"
        )

    if set(index) != {"metadata", "weight_map"}:
        raise TasteT5CleanBaseAdoptionError("safetensors index keys changed")
    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    if not isinstance(metadata, Mapping) or not isinstance(weight_map, Mapping) or not weight_map:
        raise TasteT5CleanBaseAdoptionError("safetensors index is incomplete")
    total_size = metadata.get("total_size")
    if type(total_size) is not int or total_size <= 0:
        raise TasteT5CleanBaseAdoptionError("safetensors total_size is invalid")
    shard_names: set[str] = set()
    for tensor_name, shard_name in weight_map.items():
        if (
            type(tensor_name) is not str
            or not tensor_name
            or type(shard_name) is not str
            or _MODEL_SHARD_RE.fullmatch(shard_name) is None
            or PurePosixPath(shard_name).name != shard_name
        ):
            raise TasteT5CleanBaseAdoptionError("safetensors weight_map is malformed")
        shard_names.add(shard_name)
    observed_safetensors = {name for name in files if name.endswith(".safetensors")}
    if shard_names != observed_safetensors:
        raise TasteT5CleanBaseAdoptionError(
            "safetensors index does not close the exact shard inventory"
        )
    if not all(name in files for name in shard_names):
        raise TasteT5CleanBaseAdoptionError("safetensors index references a missing shard")

    shard_evidence: dict[str, dict[str, Any]] = {}
    tensor_to_shard: dict[str, str] = {}
    tensor_payload_bytes = 0
    for shard_name in sorted(shard_names):
        header = _safetensors_header(
            root / shard_name,
            expected_size=int(files[shard_name]["bytes"]),
        )
        for tensor_name in header["tensor_names"]:
            if tensor_name in tensor_to_shard:
                raise TasteT5CleanBaseAdoptionError(
                    f"safetensors tensor is repeated across shards: {tensor_name}"
                )
            tensor_to_shard[tensor_name] = shard_name
        tensor_payload_bytes += int(header["tensor_payload_bytes"])
        shard_evidence[shard_name] = {
            "sha256": files[shard_name]["sha256"],
            "bytes": files[shard_name]["bytes"],
            **header,
        }
    if dict(sorted(weight_map.items())) != dict(sorted(tensor_to_shard.items())):
        raise TasteT5CleanBaseAdoptionError(
            "safetensors index tensor map differs from shard headers"
        )
    if tensor_payload_bytes != total_size:
        raise TasteT5CleanBaseAdoptionError(
            "safetensors total_size differs from exact tensor payload bytes"
        )

    return {
        "schema_version": SOURCE_EVIDENCE_SCHEMA,
        "stage": STAGE,
        "semantic_state": SEMANTIC_STATE,
        "source_model_path": str(root),
        "source_model_inventory_sha256": base["source_model_inventory_sha256"],
        "source_model_file_count": len(files),
        "source_model_total_bytes": sum(int(item["bytes"]) for item in files.values()),
        "source_model_files": files,
        "classification": "CLEAN_CHEMLLM_BASE",
        "hf_closure": {
            "schema_version": HF_CLOSURE_SCHEMA,
            "model_type": model_type,
            "architecture": architectures[0],
            "vocab_size": config["vocab_size"],
            "config_sha256": files["config.json"]["sha256"],
            "tokenizer_class": tokenizer_class,
            "tokenizer_config_sha256": files["tokenizer_config.json"]["sha256"],
            "tokenizer_asset": tokenizer_assets[0],
            "tokenizer_asset_sha256": files[tokenizer_assets[0]]["sha256"],
            "safetensors_index_sha256": files["model.safetensors.index.json"]["sha256"],
            "safetensors_shard_count": len(shard_names),
            "indexed_tensor_count": len(weight_map),
            "indexed_tensor_payload_bytes": total_size,
            "shards": shard_evidence,
            "index_closes_exact_shards": True,
            "index_matches_all_shard_headers": True,
        },
        "source_adapter_present": False,
        "peft_present": False,
        "taste_payload_present": False,
        "rf_payload_present": False,
        "dataset_payload_present": False,
        "dataset_specific": False,
        "optimizer_steps": 0,
        "training_performed": False,
        "taste_splits_loaded": [],
        "rf_reference_count": 0,
        "gnn_reward_used": False,
        "matrix_method_cell": False,
        "source_weights_copied": False,
    }


@dataclass(slots=True)
class HeldCleanChemLLMSource:
    """Retain the source root while one verifier hashes and publishes."""

    root: Path
    descriptor: int
    root_identity: tuple[int, int]
    stat_inventory: Mapping[str, Mapping[str, int]]
    evidence: Mapping[str, Any]
    _closed: bool = False

    @classmethod
    def open(cls, source_model: str | Path) -> "HeldCleanChemLLMSource":
        root = Path(source_model)
        if not root.is_absolute() or root.resolve(strict=True) != root:
            raise TasteT5CleanBaseAdoptionError(
                "source model must be one exact absolute physical path"
            )
        descriptor = os.open(
            root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            info = os.fstat(descriptor)
            named = os.stat(root, follow_symlinks=False)
            if (
                not stat.S_ISDIR(info.st_mode)
                or (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino)
            ):
                raise TasteT5CleanBaseAdoptionError("source root authority changed")
            before = _physical_stat_inventory_fd(
                descriptor, label="held T5 clean ChemLLM source"
            )
            evidence = inspect_clean_chemllm_base(root)
            after = _physical_stat_inventory_fd(
                descriptor, label="held T5 clean ChemLLM source"
            )
            if before != after:
                raise TasteT5CleanBaseAdoptionError(
                    "source tree changed during independent inventory"
                )
            held = cls(
                root=root,
                descriptor=descriptor,
                root_identity=(int(info.st_dev), int(info.st_ino)),
                stat_inventory=before,
                evidence=evidence,
            )
            held.revalidate()
            return held
        except BaseException:
            os.close(descriptor)
            raise

    def revalidate(self) -> None:
        if self._closed:
            raise TasteT5CleanBaseAdoptionError("held source authority is closed")
        held = os.fstat(self.descriptor)
        named = os.stat(self.root, follow_symlinks=False)
        if (
            (held.st_dev, held.st_ino) != self.root_identity
            or (named.st_dev, named.st_ino) != self.root_identity
            or _physical_stat_inventory_fd(
                self.descriptor, label="held T5 clean ChemLLM source"
            )
            != dict(self.stat_inventory)
        ):
            raise TasteT5CleanBaseAdoptionError("held source authority changed")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self.descriptor)
        self.descriptor = -1

    def __enter__(self) -> "HeldCleanChemLLMSource":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def build_clean_base_candidate(
    *,
    source_model: str | Path,
    artifact_root: str | Path,
    attempt_id: str,
    generation_token: str,
    config_sha256: str,
    expected_source_inventory_sha256: str,
) -> dict[str, Any]:
    """Worker action: write only candidate evidence below managed artifacts."""

    expected = _require_sha256(
        expected_source_inventory_sha256,
        label="expected_source_inventory_sha256",
    )
    config_digest = _require_sha256(config_sha256, label="config_sha256")
    output = Path(artifact_root)
    if not output.is_absolute() or output.resolve(strict=True) != output:
        raise TasteT5CleanBaseAdoptionError("artifact_root must already exist physically")
    if {item.name for item in os.scandir(output)} != {".generation_token.json"}:
        raise TasteT5CleanBaseAdoptionError("managed artifact root is not fresh")
    evidence = inspect_clean_chemllm_base(source_model)
    if evidence["source_model_inventory_sha256"] != expected:
        raise TasteT5CleanBaseAdoptionError("source inventory differs from launcher pin")
    source_sha = _write_exclusive(output / SOURCE_EVIDENCE_NAME, evidence)
    candidate = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "task_id": TASK_ID,
        "candidate_status": "SEALED_CANDIDATE",
        "semantic_state": SEMANTIC_STATE,
        "managed_attempt_id": attempt_id,
        "managed_generation_token": generation_token,
        "config_sha256": config_digest,
        "source_model_path": evidence["source_model_path"],
        "source_model_inventory_sha256": expected,
        "source_evidence_sha256": source_sha,
        "source_model_file_count": evidence["source_model_file_count"],
        "source_model_total_bytes": evidence["source_model_total_bytes"],
        "optimizer_steps": 0,
        "training_performed": False,
        "taste_splits_loaded": [],
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "rf_reference_count": 0,
        "gnn_reward_used": False,
        "source_adapter_present": False,
        "peft_present": False,
        "dataset_payload_present": False,
        "source_weights_copied": False,
        "matrix_method_cell": False,
        "independent_verification_required": True,
    }
    candidate_sha = _write_exclusive(output / CANDIDATE_NAME, candidate)
    return {
        "status": "SEALED_CANDIDATE",
        "stage": STAGE,
        "semantic_state": SEMANTIC_STATE,
        "source_model_inventory_sha256": expected,
        "source_evidence_sha256": source_sha,
        "candidate_sha256": candidate_sha,
        "optimizer_steps": 0,
        "taste_splits_loaded": [],
        "matrix_method_cell": False,
        "independent_verification_required": True,
    }


def _read_held_file(held: HeldSealedArtifactV2, relative: str) -> bytes:
    item = next(
        (entry for entry in held.files if entry.evidence.relative_path == relative),
        None,
    )
    if item is None:
        raise TasteT5CleanBaseAdoptionError(f"SEALED file is absent: {relative}")
    item.revalidate()
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
        raise TasteT5CleanBaseAdoptionError(f"SEALED file hash changed: {relative}")
    held.revalidate()
    return data


def _independently_verify_candidate(
    held: HeldSealedArtifactV2,
    *,
    source: HeldCleanChemLLMSource,
    expected_controller_id: str,
    expected_git_commit: str,
    expected_config_sha256: str,
    expected_source_inventory_sha256: str,
) -> dict[str, Any]:
    expected_files = {
        ".generation_token.json",
        "raw_evidence.json",
        "worker_exit.json",
        "artifacts/.generation_token.json",
        f"artifacts/{SOURCE_EVIDENCE_NAME}",
        f"artifacts/{CANDIDATE_NAME}",
    }
    observed_files = {item.evidence.relative_path for item in held.files}
    observed_directories = {item.relative_path for item in held.inventory.directories}
    if observed_files != expected_files or observed_directories != {"artifacts"}:
        raise TasteT5CleanBaseAdoptionError("SEALED T5 candidate inventory is not exact")
    raw = _json_object(
        _read_held_file(held, "raw_evidence.json"), label="worker raw evidence"
    )
    worker_exit = _json_object(
        _read_held_file(held, "worker_exit.json"), label="worker exit evidence"
    )
    candidate_data = _read_held_file(held, f"artifacts/{CANDIDATE_NAME}")
    source_data = _read_held_file(held, f"artifacts/{SOURCE_EVIDENCE_NAME}")
    candidate = _json_object(candidate_data, label="T5 candidate")
    recorded_source = _json_object(source_data, label="T5 source evidence")
    raw_evidence = raw.get("evidence")
    exit_body = worker_exit.get("exit")
    if not isinstance(raw_evidence, Mapping) or not isinstance(exit_body, Mapping):
        raise TasteT5CleanBaseAdoptionError("managed T5 raw/exit evidence is absent")
    attempt = raw_evidence.get("attempt_manifest")
    lineage = raw_evidence.get("process_lineage")
    audit = exit_body.get("process_audit")
    command = raw_evidence.get("scientific_command")
    if not all(isinstance(item, Mapping) for item in (attempt, lineage, audit)):
        raise TasteT5CleanBaseAdoptionError("managed T5 lineage evidence is absent")
    if not isinstance(command, list) or not all(type(item) is str for item in command):
        raise TasteT5CleanBaseAdoptionError("managed T5 scientific command is absent")
    expected_config = _require_sha256(expected_config_sha256, label="config_sha256")
    expected_source = _require_sha256(
        expected_source_inventory_sha256,
        label="source_model_inventory_sha256",
    )
    expected_inputs = {"source_model_inventory": expected_source}
    if (
        raw.get("schema_version") != WORKER_RAW_EVIDENCE_SCHEMA
        or worker_exit.get("schema_version") != WORKER_EXIT_SCHEMA
        or raw.get("attempt_id") != held.sealed.attempt_id
        or raw.get("generation_token") != held.sealed.generation_token
        or worker_exit.get("attempt_id") != held.sealed.attempt_id
        or worker_exit.get("generation_token") != held.sealed.generation_token
        or attempt.get("attempt_id") != held.sealed.attempt_id
        or attempt.get("controller_id") != expected_controller_id
        or attempt.get("task_id") != TASK_ID
        or attempt.get("git_commit") != expected_git_commit
        or attempt.get("config_hash") != expected_config
        or attempt.get("input_hashes") != expected_inputs
        or attempt.get("auto_terminate_uncontrolled_children") is not False
        or lineage.get("controller_id") != expected_controller_id
        or lineage.get("attempt_id") != held.sealed.attempt_id
        or exit_body.get("exit_code") != 0
        or exit_body.get("worker_closed_artifact_writers") is not True
        or audit.get("state") != "EXITED"
        or audit.get("controller_id") != expected_controller_id
        or audit.get("attempt_id") != held.sealed.attempt_id
    ):
        raise TasteT5CleanBaseAdoptionError("managed T5 worker evidence is not releasable")
    joined_command = "\n".join(command)
    if (
        "tastemolnet_t5_clean_base_worker_v2.py" not in joined_command
        or str(source.root) not in command
        or expected_source not in command
    ):
        raise TasteT5CleanBaseAdoptionError("managed T5 command is not the reviewed worker")
    if recorded_source != dict(source.evidence):
        raise TasteT5CleanBaseAdoptionError(
            "worker source inventory differs from independent complete rehash"
        )
    required_candidate = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "task_id": TASK_ID,
        "candidate_status": "SEALED_CANDIDATE",
        "semantic_state": SEMANTIC_STATE,
        "managed_attempt_id": held.sealed.attempt_id,
        "managed_generation_token": held.sealed.generation_token,
        "config_sha256": expected_config,
        "source_model_path": str(source.root),
        "source_model_inventory_sha256": expected_source,
        "source_evidence_sha256": _sha256(source_data),
        "source_model_file_count": source.evidence["source_model_file_count"],
        "source_model_total_bytes": source.evidence["source_model_total_bytes"],
        "optimizer_steps": 0,
        "training_performed": False,
        "taste_splits_loaded": [],
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "rf_reference_count": 0,
        "gnn_reward_used": False,
        "source_adapter_present": False,
        "peft_present": False,
        "dataset_payload_present": False,
        "source_weights_copied": False,
        "matrix_method_cell": False,
        "independent_verification_required": True,
    }
    if candidate != required_candidate:
        raise TasteT5CleanBaseAdoptionError("T5 clean-base candidate contract changed")
    source.revalidate()
    held.revalidate()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "stage": STAGE,
        "task_id": TASK_ID,
        "marker": PASS_MARKER,
        "semantic_state": SEMANTIC_STATE,
        "independent_scientific_verifier": True,
        "verifier_git_commit": expected_git_commit,
        "source_model_path": str(source.root),
        "source_model_inventory_sha256": expected_source,
        "source_evidence_sha256": _sha256(source_data),
        "candidate_sha256": _sha256(candidate_data),
        "source_model_file_count": source.evidence["source_model_file_count"],
        "source_model_total_bytes": source.evidence["source_model_total_bytes"],
        "hf_closure": source.evidence["hf_closure"],
        "optimizer_steps": 0,
        "training_performed": False,
        "taste_splits_loaded": [],
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "rf_reference_count": 0,
        "gnn_reward_used": False,
        "source_adapter_present": False,
        "peft_present": False,
        "taste_payload_present": False,
        "rf_payload_present": False,
        "dataset_payload_present": False,
        "source_weights_copied": False,
        "matrix_method_cell": False,
        "downstream_clean_base_authority": True,
    }


def verify_and_publish_clean_base_adoption(
    *,
    sealed_path: str | Path,
    final_path: str | Path,
    source_model: str | Path,
    expected_attempt_id: str,
    expected_generation_token: str,
    expected_controller_id: str,
    expected_git_commit: str,
    expected_config_sha256: str,
    expected_source_inventory_sha256: str,
) -> tuple[TerminalPublicationV2, dict[str, Any]]:
    """Verifier action: independently rehash source, then publish no-replace."""

    destination = Path(final_path)
    if (
        not destination.is_absolute()
        or not destination.name.startswith("adopted-clean-base-")
        or not destination.parent.exists()
        or destination.parent.resolve(strict=True) != destination.parent
    ):
        raise TasteT5CleanBaseAdoptionError(
            "final path must be a fresh adopted-clean-base-* child of a physical parent"
        )
    with HeldCleanChemLLMSource.open(source_model) as source:
        if source.evidence["source_model_inventory_sha256"] != _require_sha256(
            expected_source_inventory_sha256,
            label="source_model_inventory_sha256",
        ):
            raise TasteT5CleanBaseAdoptionError(
                "independent source inventory differs from launcher pin"
            )
        with open_sealed_worker_artifact(
            sealed_path,
            expected_attempt_id=expected_attempt_id,
            expected_generation_token=expected_generation_token,
        ) as held:
            verification = _independently_verify_candidate(
                held,
                source=source,
                expected_controller_id=expected_controller_id,
                expected_git_commit=expected_git_commit,
                expected_config_sha256=expected_config_sha256,
                expected_source_inventory_sha256=expected_source_inventory_sha256,
            )
            source.revalidate()
            held.revalidate()
            publication = verify_and_publish_sealed_attempt(
                held,
                final_path=destination,
                verification=verification,
            )
        return publication, verification


__all__ = [
    "CANDIDATE_NAME",
    "PASS_MARKER",
    "SCHEMA_VERSION",
    "SEMANTIC_STATE",
    "SOURCE_EVIDENCE_NAME",
    "STAGE",
    "TASK_ID",
    "HeldCleanChemLLMSource",
    "TasteT5CleanBaseAdoptionError",
    "build_clean_base_candidate",
    "inspect_clean_chemllm_base",
    "verify_and_publish_clean_base_adoption",
]
