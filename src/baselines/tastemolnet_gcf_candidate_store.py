"""Lossless native-candidate persistence for TasteMolNet T12.

The official VRRW candidate list is train-side state.  T12 needs that complete
ordered list and every native graph after the 20k generation process exits so
calibration can freeze an ordering before held-out test access.  This module
persists the native objects without decoding, repairing, or canonicalizing
their tensor payload.  Raw Torch archive bytes and an exact recursive
scientific digest are both retained; only the scientific digest is invariant
to non-semantic Torch archive representation details.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping
import uuid

from src.baselines.tastemolnet_gcf_smoke import _semantic_sha256


CANDIDATE_PAYLOAD_SCHEMA = "tastemolnet_t12_native_candidates_v1"
CANDIDATE_MANIFEST_SCHEMA = "tastemolnet_t12_native_candidate_manifest_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TasteT12CandidateStoreError(RuntimeError):
    """A native candidate snapshot is incomplete or changed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: Any, *, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise TasteT12CandidateStoreError(f"{field} must be lowercase SHA-256")
    return value


def _normalized_absolute(path: str | Path, *, field: str) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute() or Path(os.path.abspath(value)) != value:
        raise TasteT12CandidateStoreError(f"{field} must be normalized and absolute")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_no_replace(temporary: Path, final: Path) -> None:
    try:
        os.link(temporary, final, follow_symlinks=False)
    except FileExistsError as exc:
        raise TasteT12CandidateStoreError(
            f"T12 native candidate artifact already exists: {final}"
        ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    _fsync_directory(final.parent)


def validate_native_candidate_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "checkpoint_cursor",
        "graph_map",
        "graph_index_map",
        "counterfactual_candidates",
        "source_split",
        "calibration_loaded",
        "test_loaded",
        "rf_oracle_used",
        "full_graph_semantics",
    }
    if type(value) is not dict or set(value) != expected:
        raise TasteT12CandidateStoreError("T12 candidate payload keys changed")
    graph_map = value.get("graph_map")
    index_map = value.get("graph_index_map")
    candidates = value.get("counterfactual_candidates")
    if (
        value.get("schema_version") != CANDIDATE_PAYLOAD_SCHEMA
        or type(value.get("checkpoint_cursor")) is not int
        or value["checkpoint_cursor"] != 20_000
        or not isinstance(graph_map, Mapping)
        or not isinstance(index_map, Mapping)
        or type(candidates) is not list
        or set(graph_map) != set(index_map)
        or len(graph_map) != len(candidates)
        or value.get("source_split") != "train"
        or value.get("calibration_loaded") is not False
        or value.get("test_loaded") is not False
        or value.get("rf_oracle_used") is not False
        or value.get("full_graph_semantics") is not True
    ):
        raise TasteT12CandidateStoreError(
            "T12 candidate payload semantics changed"
        )
    for index, row in enumerate(candidates):
        if type(row) is not dict:
            raise TasteT12CandidateStoreError("T12 native candidate is malformed")
        graph_hash = _require_sha(
            row.get("graph_hash"), field="T12 native candidate graph"
        )
        if graph_hash not in graph_map or index_map.get(graph_hash) != index:
            raise TasteT12CandidateStoreError(
                "T12 native candidate order/index mapping changed"
            )
        if type(row.get("frequency")) is not int or row["frequency"] < 1:
            raise TasteT12CandidateStoreError(
                "T12 native candidate frequency is invalid"
            )
        if "importance_parts" not in row or "input_graphs_covering_list" not in row:
            raise TasteT12CandidateStoreError(
                "T12 native candidate lost importance/coverage"
            )
    return dict(value)


def write_native_candidate_snapshot(
    root: str | Path,
    *,
    vrrw: Any,
    checkpoint_cursor: int,
    contract_sha256: str,
    attempt_id: str,
    generation_token: str,
    torch: Any,
) -> Path:
    """Write one immutable lossless terminal candidate archive and manifest."""

    if type(checkpoint_cursor) is not int or checkpoint_cursor != 20_000:
        raise TasteT12CandidateStoreError(
            "T12 native candidates may be materialized only at 20k"
        )
    contract = _require_sha(contract_sha256, field="T12 candidate contract")
    token = _require_sha(generation_token, field="T12 candidate generation token")
    if type(attempt_id) is not str or not attempt_id:
        raise TasteT12CandidateStoreError("T12 candidate attempt identity is empty")
    output = _normalized_absolute(root, field="T12 native candidate root")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    if output.resolve(strict=True) != output or output.is_symlink():
        raise TasteT12CandidateStoreError("T12 native candidate root is an alias")
    payload = validate_native_candidate_payload(
        {
            "schema_version": CANDIDATE_PAYLOAD_SCHEMA,
            "checkpoint_cursor": checkpoint_cursor,
            "graph_map": getattr(vrrw, "graph_map"),
            "graph_index_map": getattr(vrrw, "graph_index_map"),
            "counterfactual_candidates": getattr(
                vrrw, "counterfactual_candidates"
            ),
            "source_split": "train",
            "calibration_loaded": False,
            "test_loaded": False,
            "rf_oracle_used": False,
            "full_graph_semantics": True,
        }
    )
    semantic_sha = _semantic_sha256(payload)
    payload_name = f"native-candidates-{checkpoint_cursor:08d}.pt"
    manifest_name = f"native-candidates-{checkpoint_cursor:08d}.manifest.json"
    payload_path = output / payload_name
    manifest_path = output / manifest_name
    temporary_payload = output / f".{payload_name}.{uuid.uuid4()}.tmp"
    try:
        with temporary_payload.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        payload_bytes = temporary_payload.stat().st_size
        if payload_bytes <= 0:
            raise TasteT12CandidateStoreError(
                "T12 native candidate payload is empty"
            )
        raw_sha = _sha256_file(temporary_payload)
        _publish_no_replace(temporary_payload, payload_path)
        manifest = {
            "schema_version": CANDIDATE_MANIFEST_SCHEMA,
            "status": "COMMITTED",
            "payload_file": payload_name,
            "payload_sha256": raw_sha,
            "payload_semantic_sha256": semantic_sha,
            "payload_bytes": payload_bytes,
            "checkpoint_cursor": checkpoint_cursor,
            "candidate_count": len(payload["counterfactual_candidates"]),
            "graph_count": len(payload["graph_map"]),
            "contract_sha256": contract,
            "attempt_id": attempt_id,
            "generation_token": token,
            "source_split": "train",
            "calibration_loaded": False,
            "test_loaded": False,
            "rf_oracle_used": False,
            "full_graph_semantics": True,
            "lossless_native_payload": True,
            "immutable_no_replace": True,
        }
        data = (
            json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        temporary_manifest = output / f".{manifest_name}.{uuid.uuid4()}.tmp"
        with temporary_manifest.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_no_replace(temporary_manifest, manifest_path)
        return manifest_path
    finally:
        if temporary_payload.exists():
            temporary_payload.unlink()


def reopen_native_candidate_snapshot(
    manifest_path: str | Path,
    *,
    expected_contract_sha256: str,
    expected_attempt_id: str,
    expected_generation_token: str,
    torch: Any,
) -> dict[str, Any]:
    """Independently reopen raw bytes and exact scientific candidate content."""

    path = _normalized_absolute(
        manifest_path, field="T12 native candidate manifest"
    )
    if path.resolve(strict=True) != path or path.is_symlink():
        raise TasteT12CandidateStoreError("T12 candidate manifest is an alias")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteT12CandidateStoreError(
            "T12 candidate manifest is unreadable"
        ) from exc
    expected_keys = {
        "schema_version",
        "status",
        "payload_file",
        "payload_sha256",
        "payload_semantic_sha256",
        "payload_bytes",
        "checkpoint_cursor",
        "candidate_count",
        "graph_count",
        "contract_sha256",
        "attempt_id",
        "generation_token",
        "source_split",
        "calibration_loaded",
        "test_loaded",
        "rf_oracle_used",
        "full_graph_semantics",
        "lossless_native_payload",
        "immutable_no_replace",
    }
    if (
        type(manifest) is not dict
        or set(manifest) != expected_keys
        or manifest.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA
        or manifest.get("status") != "COMMITTED"
        or manifest.get("checkpoint_cursor") != 20_000
        or manifest.get("contract_sha256")
        != _require_sha(expected_contract_sha256, field="T12 candidate contract")
        or manifest.get("attempt_id") != expected_attempt_id
        or manifest.get("generation_token")
        != _require_sha(
            expected_generation_token, field="T12 candidate generation token"
        )
        or any(
            manifest.get(field) is not expected
            for field, expected in {
                "calibration_loaded": False,
                "test_loaded": False,
                "rf_oracle_used": False,
                "full_graph_semantics": True,
                "lossless_native_payload": True,
                "immutable_no_replace": True,
            }.items()
        )
        or manifest.get("source_split") != "train"
    ):
        raise TasteT12CandidateStoreError(
            "T12 candidate manifest semantics changed"
        )
    payload_file = manifest.get("payload_file")
    if type(payload_file) is not str or Path(payload_file).name != payload_file:
        raise TasteT12CandidateStoreError("T12 candidate payload path is invalid")
    payload_path = path.parent / payload_file
    info = payload_path.stat()
    if (
        payload_path.resolve(strict=True) != payload_path
        or payload_path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_size != manifest.get("payload_bytes")
        or _sha256_file(payload_path)
        != _require_sha(manifest.get("payload_sha256"), field="T12 payload SHA")
    ):
        raise TasteT12CandidateStoreError(
            "T12 candidate payload bytes changed"
        )
    try:
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older supported Torch
        payload = torch.load(payload_path, map_location="cpu")
    validated = validate_native_candidate_payload(payload)
    if (
        _semantic_sha256(validated)
        != _require_sha(
            manifest.get("payload_semantic_sha256"),
            field="T12 candidate semantic SHA",
        )
        or len(validated["counterfactual_candidates"])
        != manifest.get("candidate_count")
        or len(validated["graph_map"]) != manifest.get("graph_count")
    ):
        raise TasteT12CandidateStoreError(
            "T12 candidate scientific content changed"
        )
    return validated


__all__ = [
    "CANDIDATE_MANIFEST_SCHEMA",
    "CANDIDATE_PAYLOAD_SCHEMA",
    "TasteT12CandidateStoreError",
    "reopen_native_candidate_snapshot",
    "validate_native_candidate_payload",
    "write_native_candidate_snapshot",
]
