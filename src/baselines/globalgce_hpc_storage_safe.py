"""Storage-safe publication for the exact TasteMolNet GlobalGCE HPC merge.

The exhaustive shard results remain the scientific inputs.  This module runs
the existing exact merge and its complete verifier on node-local storage, then
creates one deterministic, lossless gzip-compressed tar stream.  Only that
archive and a small self-hashed receipt are atomically published to the
persistent filesystem.  No classifier, calibration/test payload, GPU, or
matrix authority is available through this surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any, BinaryIO, Iterator, Mapping, Sequence
import uuid

from src.baselines import globalgce_hpc_exact as exact


STORAGE_SAFE_BUNDLE_SCHEMA = "globalgce_hpc_storage_safe_bundle_v1"
STORAGE_SAFE_RECEIPT_SCHEMA = "globalgce_hpc_storage_safe_receipt_v1"
STORAGE_SAFE_VERIFICATION_SCHEMA = "globalgce_hpc_storage_safe_verification_v1"
DEFAULT_MIN_RESERVE_BYTES = 2 * 1024**3
DEFAULT_RESERVE_FRACTION = 0.20
MANIFEST_PUBLICATION_ALLOWANCE_BYTES = 1024 * 1024
STREAM_CHUNK_BYTES = 1024 * 1024
MAX_RESULT_MANIFEST_BYTES = 16 * 1024 * 1024
REQUIRED_ARCHIVE_MEMBERS = frozenset(
    {
        "canary_partition_manifest.json",
        "evidence/environment_manifest.json",
        "evidence/resource_metrics.json",
        "evidence/slurm_inventory.json",
        "merge/events.jsonl",
        "merge/merge_manifest.json",
        "merge/patterns.jsonl",
        "merge/rejection_events.jsonl",
        "merge/stable_top_k.json",
        "parity_receipt.json",
        "partition_manifest.json",
    }
)


class StorageSafeT8Error(exact.GlobalGCEHPCExactError):
    """A storage, archive, exactness, or publication gate failed."""


@dataclass(frozen=True)
class StorageAdmission:
    state: str
    free_bytes: int
    reserve_bytes: int
    required_bytes: int
    usable_bytes: int
    shortfall_bytes: int

    def to_json(self) -> dict[str, int | str]:
        return {
            "state": self.state,
            "free_bytes": self.free_bytes,
            "reserve_bytes": self.reserve_bytes,
            "required_bytes": self.required_bytes,
            "usable_bytes": self.usable_bytes,
            "shortfall_bytes": self.shortfall_bytes,
        }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _self_hashed(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = exact.canonical_sha256(result)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise StorageSafeT8Error(f"malformed JSON: {path}") from exc
    if type(payload) is not dict:
        raise StorageSafeT8Error(f"JSON must contain one object: {path}")
    return payload


def _unsigned(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != field}


def _require_self_hash(payload: Mapping[str, Any], field: str, label: str) -> None:
    claimed = payload.get(field)
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise StorageSafeT8Error(f"{label} self-hash is missing")
    if exact.canonical_sha256(_unsigned(payload, field)) != claimed:
        raise StorageSafeT8Error(f"{label} self-hash mismatch")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def filesystem_free_bytes(path: Path) -> int:
    stats = os.statvfs(path)
    return int(stats.f_bavail) * int(stats.f_frsize)


def validate_storage_path_policy(path: str | Path, *, label: str) -> Path:
    """Reject the explicitly forbidden `/ssdfs` tree before any write."""

    value = Path(path).expanduser().resolve(strict=False)
    forbidden = Path("/ssdfs")
    if value == forbidden or forbidden in value.parents:
        raise StorageSafeT8Error(f"{label} may not use /ssdfs")
    return value


def storage_admission(
    *,
    required_bytes: int,
    free_bytes: int,
    minimum_reserve_bytes: int = DEFAULT_MIN_RESERVE_BYTES,
    reserve_fraction: float = DEFAULT_RESERVE_FRACTION,
) -> StorageAdmission:
    """Return an exact persistent-space decision without writing any bytes."""

    if type(required_bytes) is not int or required_bytes < 0:
        raise StorageSafeT8Error("required_bytes must be a non-negative integer")
    if type(free_bytes) is not int or free_bytes < 0:
        raise StorageSafeT8Error("free_bytes must be a non-negative integer")
    if type(minimum_reserve_bytes) is not int or minimum_reserve_bytes < 0:
        raise StorageSafeT8Error("minimum_reserve_bytes is invalid")
    if not 0.0 <= reserve_fraction < 1.0:
        raise StorageSafeT8Error("reserve_fraction must be in [0, 1)")
    reserve = max(minimum_reserve_bytes, math.ceil(free_bytes * reserve_fraction))
    usable = max(0, free_bytes - reserve)
    shortfall = max(0, required_bytes - usable)
    return StorageAdmission(
        state="PASS" if shortfall == 0 else "BLOCKED_INSUFFICIENT_PERSISTENT_SPACE",
        free_bytes=free_bytes,
        reserve_bytes=reserve,
        required_bytes=required_bytes,
        usable_bytes=usable,
        shortfall_bytes=shortfall,
    )


def _partition_payload_inventory(
    manifest: Mapping[str, Any], shards_root: Path
) -> dict[str, int]:
    event_bytes = pattern_bytes = event_count = pattern_count = rejection_count = 0
    for unit in manifest["partitions"]:
        root = (
            shards_root
            / f"shard-{int(unit['shard_index']):03d}"
            / "partitions"
            / str(unit["partition_id"])
        )
        result = _read_json(root / "partition_manifest.json")
        if (
            result.get("schema_version") != exact.UNIT_RESULT_SCHEMA
            or result.get("status") != "PASS"
            or result.get("partition") != dict(unit)
            or result.get("manifest_sha256") != manifest.get("manifest_sha256")
        ):
            raise StorageSafeT8Error("partition inventory header is invalid")
        event_bytes += (root / "events.jsonl").stat().st_size
        pattern_bytes += (root / "patterns.jsonl").stat().st_size
        event_count += int(result["event_count"])
        pattern_count += int(result["pattern_count"])
        rejection_count += int(result["rejection_count"])
    return {
        "event_bytes": event_bytes,
        "pattern_bytes": pattern_bytes,
        "event_count": event_count,
        "pattern_count": pattern_count,
        "rejection_count": rejection_count,
    }


def node_local_scratch_admission(
    inventory: Mapping[str, int], *, free_bytes: int
) -> StorageAdmission:
    """Conservatively admit merge output, uniqueness DB, and compressed tar.

    The bound assumes every event is repeated in the rejection stream, both
    merged streams coexist with an equally large incompressible archive, and
    each event/pattern identity consumes 160 bytes in the transient SQLite
    uniqueness index.  A 25 percent margin and 2 GiB floor are then added.
    """

    events = int(inventory["event_bytes"])
    patterns = int(inventory["pattern_bytes"])
    identities = int(inventory["event_count"]) + int(inventory["pattern_count"])
    simultaneous_payload = 4 * events + 2 * patterns + 160 * identities
    required = math.ceil(simultaneous_payload * 1.25) + DEFAULT_MIN_RESERVE_BYTES
    return storage_admission(
        required_bytes=required,
        free_bytes=free_bytes,
        minimum_reserve_bytes=0,
        reserve_fraction=0.0,
    )


def _validate_parity_binding(
    parity_path: Path, full_manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    parity = _read_json(parity_path)
    _require_self_hash(parity, "result_sha256", "parity receipt")
    canary_identity = parity.get("canary_partition_manifest")
    if type(canary_identity) is not dict:
        raise StorageSafeT8Error("parity receipt lacks canary manifest identity")
    canary_path_value = canary_identity.get("path")
    if not isinstance(canary_path_value, str) or not canary_path_value:
        raise StorageSafeT8Error("parity receipt canary path is missing")
    canary_path = Path(canary_path_value).expanduser().resolve(strict=True)
    if exact.sha256_file(canary_path) != canary_identity.get("file_sha256"):
        raise StorageSafeT8Error("canary partition manifest file hash changed")
    canary = exact.validate_partition_manifest(canary_path)
    if (
        canary.get("manifest_sha256") != canary_identity.get("manifest_sha256")
        or canary.get("scope")
        not in {"SELECTED_ROOTS_CANARY", "SELECTED_PARTITION_CANARY"}
        or parity.get("schema_version") != exact.PARITY_RESULT_SCHEMA
        or parity.get("status") != "PASS"
        or parity.get("scientific_input_sha256")
        != full_manifest.get("scientific_input_sha256")
        or parity.get("provenance_sha256")
        != full_manifest.get("provenance", {}).get("provenance_sha256")
        or parity.get("target_branches")
        != full_manifest.get("provenance", {}).get("target_branches")
    ):
        raise StorageSafeT8Error("parity receipt is not bound to the full run")
    return parity, canary_path, canary


def _validated_evidence(
    values: Mapping[str, str | Path]
) -> tuple[list[tuple[Path, str]], dict[str, dict[str, Any]]]:
    files: list[tuple[Path, str]] = []
    identities: dict[str, dict[str, Any]] = {}
    for role in ("environment_manifest", "slurm_inventory", "resource_metrics"):
        source = Path(values[role]).expanduser().resolve(strict=True)
        if source.is_symlink() or not source.is_file():
            raise StorageSafeT8Error(f"{role} must be one regular file")
        content = _read_json(source)
        if not content:
            raise StorageSafeT8Error(f"{role} may not be empty")
        archive_name = f"evidence/{role}.json"
        files.append((source, archive_name))
        identities[role] = {
            "bytes": source.stat().st_size,
            "sha256": exact.sha256_file(source),
            "content_sha256": exact.canonical_sha256(content),
        }
    return files, identities


def _archive_inventory(
    files: Sequence[tuple[Path, str]],
    *,
    verified_hashes: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    names = [name for _source, name in files]
    if len(names) != len(set(names)):
        raise StorageSafeT8Error("archive member names are not unique")
    if names != sorted(names):
        raise StorageSafeT8Error("archive inputs must use canonical name order")
    trusted = dict(verified_hashes or {})
    return [
        {
            "name": name,
            "bytes": source.stat().st_size,
            "sha256": trusted[name] if name in trusted else exact.sha256_file(source),
        }
        for source, name in files
    ]


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o600
    return info


def _write_deterministic_archive(
    archive: Path,
    *,
    inner_manifest: Mapping[str, Any],
    files: Sequence[tuple[Path, str]],
) -> None:
    if archive.exists():
        raise StorageSafeT8Error("scratch archive target must be fresh")
    archive.parent.mkdir(parents=True, exist_ok=True)
    inner_bytes = _canonical_bytes(dict(inner_manifest)) + b"\n"
    with archive.open("xb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=6,
            mtime=0,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w|",
                format=tarfile.PAX_FORMAT,
            ) as tar:
                tar.addfile(
                    _tar_info("RESULT_MANIFEST.json", len(inner_bytes)),
                    io.BytesIO(inner_bytes),
                )
                for source, name in files:
                    with source.open("rb") as stream:
                        tar.addfile(_tar_info(name, source.stat().st_size), stream)
        raw.flush()
        os.fsync(raw.fileno())
    _fsync_directory(archive.parent)


def build_storage_safe_archive(
    *,
    partition_manifest: str | Path,
    merge_root: str | Path,
    parity_receipt: str | Path,
    environment_manifest: str | Path,
    slurm_inventory: str | Path,
    resource_metrics: str | Path,
    packaging_commit: str,
    output_archive: str | Path,
    _prevalidated_merge_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the deterministic compressed archive in node-local storage."""

    if (
        not isinstance(packaging_commit, str)
        or len(packaging_commit) != 40
        or any(character not in "0123456789abcdef" for character in packaging_commit)
    ):
        raise StorageSafeT8Error("packaging_commit must be one lowercase Git SHA")
    manifest_path = Path(partition_manifest).expanduser().resolve(strict=True)
    manifest = exact.validate_partition_manifest(manifest_path)
    merge = Path(merge_root).expanduser().resolve(strict=True)
    if _prevalidated_merge_manifest is None:
        merge_manifest = exact.validate_merge_result(
            merge,
            manifest=manifest,
            allowed_scopes=("FULL_MANIFEST",),
        )
    else:
        merge_manifest = dict(_prevalidated_merge_manifest)
        if _read_json(merge / "merge_manifest.json") != merge_manifest:
            raise StorageSafeT8Error("prevalidated merge manifest bytes changed")
    parity_path = Path(parity_receipt).expanduser().resolve(strict=True)
    parity, canary_path, _canary = _validate_parity_binding(parity_path, manifest)
    evidence_files, evidence_identities = _validated_evidence(
        {
            "environment_manifest": environment_manifest,
            "slurm_inventory": slurm_inventory,
            "resource_metrics": resource_metrics,
        }
    )
    files = sorted(
        [
            (manifest_path, "partition_manifest.json"),
            (canary_path, "canary_partition_manifest.json"),
            (merge / "merge_manifest.json", "merge/merge_manifest.json"),
            (merge / "events.jsonl", "merge/events.jsonl"),
            (merge / "patterns.jsonl", "merge/patterns.jsonl"),
            (merge / "rejection_events.jsonl", "merge/rejection_events.jsonl"),
            (merge / "stable_top_k.json", "merge/stable_top_k.json"),
            (parity_path, "parity_receipt.json"),
            *evidence_files,
        ],
        key=lambda item: item[1],
    )
    inventory = _archive_inventory(
        files,
        verified_hashes={
            "merge/events.jsonl": merge_manifest["events_sha256"],
            "merge/patterns.jsonl": merge_manifest["patterns_sha256"],
            "merge/rejection_events.jsonl": merge_manifest[
                "rejection_events_sha256"
            ],
            "merge/stable_top_k.json": merge_manifest["stable_top_k_sha256"],
        },
    )
    inner = _self_hashed(
        {
            "schema_version": STORAGE_SAFE_BUNDLE_SCHEMA,
            "status": "PASS",
            "compression": {
                "algorithm": "gzip",
                "compresslevel": 6,
                "gzip_mtime": 0,
                "tar_format": "PAX",
                "member_metadata_normalized": True,
                "lossless": True,
            },
            "packaging_commit": packaging_commit,
            "manifest_sha256": manifest["manifest_sha256"],
            "merge_result_sha256": merge_manifest["result_sha256"],
            "parity_result_sha256": parity["result_sha256"],
            "parity_manifest_sha256": parity["manifest_sha256"],
            "scientific_input_sha256": manifest["scientific_input_sha256"],
            "provenance_sha256": manifest["provenance"]["provenance_sha256"],
            "target_branches": manifest["provenance"]["target_branches"],
            "raw_stream_hash_semantics": "UNCHANGED_MERGE_JSONL_BYTES",
            "global_order": "OFFICIAL_ROOT_AND_DFS_PREORDER",
            "event_count": merge_manifest["event_count"],
            "pattern_count": merge_manifest["pattern_count"],
            "rejection_count": merge_manifest["rejection_count"],
            "external_evidence": evidence_identities,
            "files": inventory,
            "node_local_merge_verified": True,
            "matrix_write_enabled": False,
            "hpc_gine_inference_used": False,
            "hpc_calibration_or_test_used": False,
        },
        "bundle_content_sha256",
    )
    archive = validate_storage_path_policy(output_archive, label="scratch archive")
    _write_deterministic_archive(archive, inner_manifest=inner, files=files)
    return {
        "inner_manifest": inner,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": exact.sha256_file(archive),
    }


class _HashingLineReader:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def __iter__(self) -> Iterator[bytes]:
        for line in self.stream:
            self.digest.update(line)
            self.bytes_read += len(line)
            yield line


def _decode_json_line(line: bytes, *, member: str, line_number: int) -> dict[str, Any]:
    try:
        payload = json.loads(
            line,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise StorageSafeT8Error(
            f"invalid JSONL in {member} at line {line_number}"
        ) from exc
    if type(payload) is not dict:
        raise StorageSafeT8Error(f"non-object JSONL row in {member}")
    return payload


def _validate_dfs_identity(row: Mapping[str, Any], *, label: str) -> None:
    code = exact.dfs_code_from_json(row.get("dfs_code"))
    if row.get("dfs_code_sha256") != exact.dfs_code_sha256(code):
        raise StorageSafeT8Error(f"{label} DFS identity changed")


def _stream_jsonl_member(
    stream: BinaryIO,
    *,
    name: str,
    target_branches: Sequence[int],
) -> dict[str, Any]:
    reader = _HashingLineReader(stream)
    count = 0
    previous_global_preorder = -1
    for line_number, line in enumerate(reader, start=1):
        row = _decode_json_line(line, member=name, line_number=line_number)
        _validate_dfs_identity(row, label=name)
        global_preorder = row.get("global_preorder")
        if type(global_preorder) is not int:
            raise StorageSafeT8Error(f"{name} lacks integer global_preorder")
        if name == "merge/events.jsonl" and global_preorder != count:
            raise StorageSafeT8Error("event global preorder is not consecutive")
        if name == "merge/patterns.jsonl" and global_preorder != count:
            raise StorageSafeT8Error("pattern global preorder is not consecutive")
        if global_preorder <= previous_global_preorder:
            raise StorageSafeT8Error(f"{name} global preorder is not increasing")
        previous_global_preorder = global_preorder
        if row.get("target_branches") != list(target_branches):
            raise StorageSafeT8Error(f"{name} target branches changed")
        if name.endswith("events.jsonl"):
            status = row.get("status")
            if status not in exact.EVENT_STATUSES:
                raise StorageSafeT8Error(f"{name} has invalid event status")
            if name == "merge/rejection_events.jsonl" and status == "ACCEPTED":
                raise StorageSafeT8Error("rejection stream contains accepted event")
        if name == "merge/patterns.jsonl":
            expected_pattern = exact.canonical_sha256(
                {
                    "dfs_code": exact.dfs_code_to_json(
                        exact.dfs_code_from_json(row["dfs_code"])
                    ),
                    "undirected": True,
                }
            )
            if row.get("pattern_sha256") != expected_pattern:
                raise StorageSafeT8Error("pattern identity changed")
        count += 1
    return {
        "bytes": reader.bytes_read,
        "sha256": reader.digest.hexdigest(),
        "rows": count,
    }


def _read_member_bytes(stream: BinaryIO, expected_size: int) -> tuple[bytes, str]:
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    total = 0
    while True:
        block = stream.read(STREAM_CHUNK_BYTES)
        if not block:
            break
        digest.update(block)
        total += len(block)
        chunks.append(block)
    if total != expected_size:
        raise StorageSafeT8Error("archive member size changed while streaming")
    return b"".join(chunks), digest.hexdigest()


def _load_member_json(data: bytes, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            data,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise StorageSafeT8Error(f"malformed archive JSON member: {name}") from exc
    if type(payload) is not dict:
        raise StorageSafeT8Error(f"archive JSON member is not an object: {name}")
    return payload


def stream_verify_storage_safe_bundle(
    archive_path: str | Path,
    *,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the compressed result in one tar pass without extracting it."""

    archive = Path(archive_path).expanduser().resolve(strict=True)
    if archive.is_symlink() or not archive.is_file():
        raise StorageSafeT8Error("bundle archive must be one regular file")
    archive_bytes = archive.stat().st_size
    archive_sha = exact.sha256_file(archive)
    inner: dict[str, Any] | None = None
    expected: dict[str, dict[str, Any]] = {}
    observed: dict[str, dict[str, Any]] = {}
    json_payloads: dict[str, dict[str, Any]] = {}
    expected_order: list[str] = []
    with archive.open("rb") as raw:
        with tarfile.open(fileobj=raw, mode="r|gz") as tar:
            for member_index, member in enumerate(tar):
                if not member.isfile() or member.issym() or member.islnk():
                    raise StorageSafeT8Error("bundle contains a non-regular member")
                stream = tar.extractfile(member)
                if stream is None:
                    raise StorageSafeT8Error(f"cannot stream archive member: {member.name}")
                if member_index == 0:
                    if member.name != "RESULT_MANIFEST.json":
                        raise StorageSafeT8Error("result manifest must be the first member")
                    if member.size > MAX_RESULT_MANIFEST_BYTES:
                        raise StorageSafeT8Error("result manifest is unexpectedly large")
                    data, _digest = _read_member_bytes(stream, member.size)
                    inner = _load_member_json(data, member.name)
                    _require_self_hash(inner, "bundle_content_sha256", "bundle manifest")
                    if (
                        inner.get("schema_version") != STORAGE_SAFE_BUNDLE_SCHEMA
                        or inner.get("status") != "PASS"
                        or inner.get("matrix_write_enabled") is not False
                        or inner.get("hpc_gine_inference_used") is not False
                        or inner.get("hpc_calibration_or_test_used") is not False
                        or inner.get("raw_stream_hash_semantics")
                        != "UNCHANGED_MERGE_JSONL_BYTES"
                    ):
                        raise StorageSafeT8Error("bundle manifest contract is invalid")
                    packaging_commit = inner.get("packaging_commit")
                    if (
                        not isinstance(packaging_commit, str)
                        or len(packaging_commit) != 40
                        or any(
                            character not in "0123456789abcdef"
                            for character in packaging_commit
                        )
                    ):
                        raise StorageSafeT8Error("bundle packaging commit is malformed")
                    inventory = inner.get("files")
                    if type(inventory) is not list:
                        raise StorageSafeT8Error("bundle inventory is malformed")
                    if any(
                        type(row) is not dict
                        or set(row) != {"name", "bytes", "sha256"}
                        or not isinstance(row.get("name"), str)
                        or type(row.get("bytes")) is not int
                        or row["bytes"] < 0
                        or not isinstance(row.get("sha256"), str)
                        or len(row["sha256"]) != 64
                        for row in inventory
                    ):
                        raise StorageSafeT8Error("bundle inventory identity is malformed")
                    expected_order = [row.get("name") for row in inventory]
                    if (
                        any(not isinstance(name, str) for name in expected_order)
                        or expected_order != sorted(expected_order)
                        or len(expected_order) != len(set(expected_order))
                    ):
                        raise StorageSafeT8Error("bundle inventory order is invalid")
                    if set(expected_order) != REQUIRED_ARCHIVE_MEMBERS:
                        raise StorageSafeT8Error("bundle member allowlist is incomplete")
                    expected = {row["name"]: row for row in inventory}
                    continue
                if inner is None:
                    raise StorageSafeT8Error("bundle manifest was not read")
                expected_index = member_index - 1
                if (
                    expected_index >= len(expected_order)
                    or member.name != expected_order[expected_index]
                ):
                    raise StorageSafeT8Error("archive member order/inventory mismatch")
                identity = expected[member.name]
                if member.size != identity.get("bytes"):
                    raise StorageSafeT8Error(f"archive member size mismatch: {member.name}")
                if member.name in {
                    "merge/events.jsonl",
                    "merge/patterns.jsonl",
                    "merge/rejection_events.jsonl",
                }:
                    details = _stream_jsonl_member(
                        stream,
                        name=member.name,
                        target_branches=inner["target_branches"],
                    )
                else:
                    if member.size > MAX_RESULT_MANIFEST_BYTES:
                        raise StorageSafeT8Error(
                            f"unexpectedly large non-JSONL member: {member.name}"
                        )
                    data, digest = _read_member_bytes(stream, member.size)
                    details = {"bytes": len(data), "sha256": digest}
                    if member.name.endswith(".json"):
                        json_payloads[member.name] = _load_member_json(data, member.name)
                if details["sha256"] != identity.get("sha256"):
                    raise StorageSafeT8Error(f"archive member hash mismatch: {member.name}")
                observed[member.name] = details
    if inner is None or list(observed) != expected_order:
        raise StorageSafeT8Error("archive inventory is incomplete")
    merge_manifest = json_payloads.get("merge/merge_manifest.json")
    if merge_manifest is None:
        raise StorageSafeT8Error("merge manifest is missing from bundle")
    _require_self_hash(merge_manifest, "result_sha256", "merge manifest")
    partition_manifest = json_payloads.get("partition_manifest.json")
    canary_manifest = json_payloads.get("canary_partition_manifest.json")
    parity_receipt = json_payloads.get("parity_receipt.json")
    stable_top_k = json_payloads.get("merge/stable_top_k.json")
    if any(
        payload is None
        for payload in (
            partition_manifest,
            canary_manifest,
            parity_receipt,
            stable_top_k,
        )
    ):
        raise StorageSafeT8Error("bundle scientific manifests are incomplete")
    assert partition_manifest is not None
    assert canary_manifest is not None
    assert parity_receipt is not None
    assert stable_top_k is not None
    _require_self_hash(partition_manifest, "manifest_sha256", "partition manifest")
    _require_self_hash(canary_manifest, "manifest_sha256", "canary manifest")
    _require_self_hash(parity_receipt, "result_sha256", "parity receipt")
    if (
        partition_manifest.get("manifest_sha256") != inner.get("manifest_sha256")
        or partition_manifest.get("scope") != "FULL_ROOT_UNIVERSE"
        or partition_manifest.get("scientific_input_sha256")
        != inner.get("scientific_input_sha256")
        or partition_manifest.get("provenance", {}).get("provenance_sha256")
        != inner.get("provenance_sha256")
        or partition_manifest.get("provenance", {}).get("target_branches")
        != inner.get("target_branches")
        or partition_manifest.get("matrix_write_enabled") is not False
        or canary_manifest.get("scope")
        not in {"SELECTED_ROOTS_CANARY", "SELECTED_PARTITION_CANARY"}
        or parity_receipt.get("schema_version") != exact.PARITY_RESULT_SCHEMA
        or parity_receipt.get("status") != "PASS"
        or parity_receipt.get("result_sha256")
        != inner.get("parity_result_sha256")
        or parity_receipt.get("manifest_sha256")
        != inner.get("parity_manifest_sha256")
        or parity_receipt.get("scientific_input_sha256")
        != inner.get("scientific_input_sha256")
        or parity_receipt.get("provenance_sha256")
        != inner.get("provenance_sha256")
        or parity_receipt.get("target_branches") != inner.get("target_branches")
    ):
        raise StorageSafeT8Error("bundle scientific provenance binding failed")
    canary_identity = parity_receipt.get("canary_partition_manifest")
    if (
        type(canary_identity) is not dict
        or canary_identity.get("manifest_sha256")
        != canary_manifest.get("manifest_sha256")
        or canary_identity.get("file_sha256")
        != observed["canary_partition_manifest.json"]["sha256"]
    ):
        raise StorageSafeT8Error("bundled canary identity differs from parity")
    if (
        merge_manifest.get("schema_version") != exact.MERGE_RESULT_SCHEMA
        or merge_manifest.get("status") != "PASS"
        or merge_manifest.get("scope") != "FULL_MANIFEST"
        or merge_manifest.get("manifest_sha256") != inner.get("manifest_sha256")
        or merge_manifest.get("scientific_input_sha256")
        != inner.get("scientific_input_sha256")
        or merge_manifest.get("provenance_sha256") != inner.get("provenance_sha256")
        or merge_manifest.get("target_branches") != inner.get("target_branches")
        or merge_manifest.get("partition_disjoint") is not True
        or merge_manifest.get("partition_complete") is not True
        or merge_manifest.get("full_root_universe_complete") is not True
        or merge_manifest.get("duplicate_pattern_count") != 0
        or merge_manifest.get("duplicate_event_count") != 0
        or merge_manifest.get("scientific_search_pruned") is not False
        or merge_manifest.get("approximation_used") is not False
        or merge_manifest.get("matrix_write_enabled") is not False
    ):
        raise StorageSafeT8Error("bundled merge manifest contract is invalid")
    comparisons = {
        "merge/events.jsonl": ("events_sha256", "event_count"),
        "merge/patterns.jsonl": ("patterns_sha256", "pattern_count"),
        "merge/rejection_events.jsonl": (
            "rejection_events_sha256",
            "rejection_count",
        ),
    }
    for name, (hash_field, count_field) in comparisons.items():
        details = observed[name]
        if (
            details["sha256"] != merge_manifest.get(hash_field)
            or details["rows"] != merge_manifest.get(count_field)
            or details["rows"] != inner.get(count_field)
        ):
            raise StorageSafeT8Error(f"merged stream contract mismatch: {name}")
    if merge_manifest.get("result_sha256") != inner.get("merge_result_sha256"):
        raise StorageSafeT8Error("bundle/merge identity mismatch")
    if (
        stable_top_k.get("schema_version")
        != "globalgce_hpc_exact_stable_top_k_v1"
        or stable_top_k.get("selected_count")
        != merge_manifest.get("stable_top_k_selected_count")
        or stable_top_k.get("selected_sha256")
        != merge_manifest.get("stable_top_k_selected_sha256")
        or exact.canonical_sha256(stable_top_k.get("selected"))
        != stable_top_k.get("selected_sha256")
        or observed["merge/stable_top_k.json"]["sha256"]
        != merge_manifest.get("stable_top_k_sha256")
    ):
        raise StorageSafeT8Error("stable top-K binding failed")
    for role, identity in inner["external_evidence"].items():
        name = f"evidence/{role}.json"
        payload = json_payloads.get(name)
        if (
            name not in observed
            or payload is None
            or observed[name]["bytes"] != identity.get("bytes")
            or observed[name]["sha256"] != identity.get("sha256")
            or exact.canonical_sha256(payload) != identity.get("content_sha256")
        ):
            raise StorageSafeT8Error(f"external evidence binding failed: {role}")
    receipt: dict[str, Any] | None = None
    if receipt_path is not None:
        receipt = _read_json(Path(receipt_path).expanduser().resolve(strict=True))
        _require_self_hash(receipt, "receipt_sha256", "storage-safe receipt")
        if (
            receipt.get("schema_version") != STORAGE_SAFE_RECEIPT_SCHEMA
            or receipt.get("status") != "PASS"
            or receipt.get("bundle_content_sha256")
            != inner.get("bundle_content_sha256")
            or receipt.get("archive_bytes") != archive_bytes
            or receipt.get("archive_sha256") != archive_sha
            or receipt.get("packaging_commit") != inner.get("packaging_commit")
            or receipt.get("partition_manifest_sha256")
            != inner.get("manifest_sha256")
            or receipt.get("scientific_input_sha256")
            != inner.get("scientific_input_sha256")
            or receipt.get("matrix_write_enabled") is not False
        ):
            raise StorageSafeT8Error("storage-safe receipt does not bind the archive")
    return _self_hashed(
        {
            "schema_version": STORAGE_SAFE_VERIFICATION_SCHEMA,
            "status": "PASS",
            "archive_bytes": archive_bytes,
            "archive_sha256": archive_sha,
            "bundle_content_sha256": inner["bundle_content_sha256"],
            "merge_result_sha256": inner["merge_result_sha256"],
            "partition_manifest_sha256": inner["manifest_sha256"],
            "scientific_input_sha256": inner["scientific_input_sha256"],
            "packaging_commit": inner["packaging_commit"],
            "event_count": inner["event_count"],
            "pattern_count": inner["pattern_count"],
            "rejection_count": inner["rejection_count"],
            "streaming_verification": True,
            "extracted_to_disk": False,
            "matrix_write_enabled": False,
            "receipt_verified": receipt is not None,
        },
        "verification_sha256",
    )


def _copy_fsync(source: Path, destination: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as reader, destination.open("xb") as writer:
        while block := reader.read(STREAM_CHUNK_BYTES):
            writer.write(block)
            digest.update(block)
        writer.flush()
        os.fsync(writer.fileno())
    return digest.hexdigest()


def publish_storage_safe_archive(
    *,
    scratch_archive: str | Path,
    inner_manifest: Mapping[str, Any],
    prepublication_verification: Mapping[str, Any],
    output_root: str | Path,
    minimum_reserve_bytes: int = DEFAULT_MIN_RESERVE_BYTES,
    reserve_fraction: float = DEFAULT_RESERVE_FRACTION,
) -> dict[str, Any]:
    """Atomically publish exactly one compressed bundle and one manifest."""

    archive = validate_storage_path_policy(
        Path(scratch_archive).expanduser().resolve(strict=True),
        label="scratch archive",
    )
    destination = validate_storage_path_policy(output_root, label="persistent result")
    if destination.exists():
        raise StorageSafeT8Error("persistent result root must be fresh")
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_bytes = archive.stat().st_size
    archive_sha = exact.sha256_file(archive)
    verification = dict(prepublication_verification)
    _require_self_hash(
        verification,
        "verification_sha256",
        "prepublication verification",
    )
    if (
        verification.get("status") != "PASS"
        or verification.get("streaming_verification") is not True
        or verification.get("extracted_to_disk") is not False
        or verification.get("archive_bytes") != archive_bytes
        or verification.get("archive_sha256") != archive_sha
        or verification.get("bundle_content_sha256")
        != inner_manifest.get("bundle_content_sha256")
    ):
        raise StorageSafeT8Error("prepublication archive verification is invalid")
    receipt = _self_hashed(
        {
            "schema_version": STORAGE_SAFE_RECEIPT_SCHEMA,
            "status": "PASS",
            "archive_name": "t8_exact_result_bundle.tar.gz",
            "archive_bytes": archive_bytes,
            "archive_sha256": archive_sha,
            "bundle_content_sha256": inner_manifest["bundle_content_sha256"],
            "merge_result_sha256": inner_manifest["merge_result_sha256"],
            "packaging_commit": inner_manifest["packaging_commit"],
            "partition_manifest_sha256": inner_manifest["manifest_sha256"],
            "scientific_input_sha256": inner_manifest["scientific_input_sha256"],
            "event_count": inner_manifest["event_count"],
            "pattern_count": inner_manifest["pattern_count"],
            "rejection_count": inner_manifest["rejection_count"],
            "compression": inner_manifest["compression"],
            "prepublication_verification_sha256": verification[
                "verification_sha256"
            ],
            "persistent_payload_policy": "COMPRESSED_BUNDLE_AND_MANIFEST_ONLY",
            "matrix_write_enabled": False,
        },
        "receipt_sha256",
    )
    receipt_bytes = _canonical_bytes(receipt) + b"\n"
    required = archive_bytes + len(receipt_bytes) + MANIFEST_PUBLICATION_ALLOWANCE_BYTES
    admission = storage_admission(
        required_bytes=required,
        free_bytes=filesystem_free_bytes(destination.parent),
        minimum_reserve_bytes=minimum_reserve_bytes,
        reserve_fraction=reserve_fraction,
    )
    if admission.state != "PASS":
        raise StorageSafeT8Error(
            "BLOCKED_HPC_USER_QUOTA_SHORTFALL "
            f"required_bytes={required} free_bytes={admission.free_bytes} "
            f"reserve_bytes={admission.reserve_bytes} "
            f"usable_bytes={admission.usable_bytes} "
            f"shortfall_bytes={admission.shortfall_bytes}"
        )
    staging = destination.parent / (
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.incomplete"
    )
    staging.mkdir(mode=0o700)
    try:
        persistent_archive = staging / "t8_exact_result_bundle.tar.gz"
        copied_sha = _copy_fsync(archive, persistent_archive)
        if copied_sha != archive_sha:
            raise StorageSafeT8Error("persistent archive copy hash mismatch")
        exact.atomic_write_json(staging / "result_manifest.json", receipt)
        _fsync_directory(staging)
        if destination.exists():
            raise StorageSafeT8Error("persistent result appeared during publication")
        os.rename(staging, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        **receipt,
        "output_root": str(destination),
        "archive": str(destination / "t8_exact_result_bundle.tar.gz"),
        "manifest": str(destination / "result_manifest.json"),
        "storage_admission": admission.to_json(),
    }


def merge_package_storage_safe(
    *,
    partition_manifest: str | Path,
    shards_root: str | Path,
    parity_receipt: str | Path,
    environment_manifest: str | Path,
    slurm_inventory: str | Path,
    resource_metrics: str | Path,
    packaging_commit: str,
    scratch_root: str | Path,
    output_root: str | Path,
    require_distinct_filesystems: bool = True,
    minimum_reserve_bytes: int = DEFAULT_MIN_RESERVE_BYTES,
    reserve_fraction: float = DEFAULT_RESERVE_FRACTION,
) -> dict[str, Any]:
    """Run exact merge/verification locally, then publish the compact result."""

    manifest_path = Path(partition_manifest).expanduser().resolve(strict=True)
    manifest = exact.validate_partition_manifest(manifest_path)
    if manifest.get("scope") != "FULL_ROOT_UNIVERSE":
        raise StorageSafeT8Error("storage-safe production merge requires full manifest")
    shards = Path(shards_root).expanduser().resolve(strict=True)
    scratch_parent = validate_storage_path_policy(
        Path(scratch_root).expanduser().resolve(strict=True),
        label="node-local scratch",
    )
    destination = validate_storage_path_policy(output_root, label="persistent result")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise StorageSafeT8Error("persistent result root must be fresh")
    if scratch_parent == destination.parent or scratch_parent in destination.parents:
        raise StorageSafeT8Error("node-local scratch overlaps persistent output")
    if require_distinct_filesystems and (
        os.stat(scratch_parent).st_dev == os.stat(destination.parent).st_dev
    ):
        raise StorageSafeT8Error("scratch and persistent output are on one filesystem")
    inventory = _partition_payload_inventory(manifest, shards)
    scratch_admission = node_local_scratch_admission(
        inventory,
        free_bytes=filesystem_free_bytes(scratch_parent),
    )
    if scratch_admission.state != "PASS":
        raise StorageSafeT8Error(
            "BLOCKED_NODE_LOCAL_SCRATCH_SHORTFALL "
            f"required_bytes={scratch_admission.required_bytes} "
            f"free_bytes={scratch_admission.free_bytes} "
            f"shortfall_bytes={scratch_admission.shortfall_bytes}"
        )
    work = Path(
        tempfile.mkdtemp(prefix="t8-storage-safe-", dir=scratch_parent)
    )
    merge_root = work / "verified-merge"
    scratch_archive = work / "t8_exact_result_bundle.tar.gz"
    exact.merge_exact_shards(
        partition_manifest=manifest_path,
        shards_root=shards,
        output_root=merge_root,
        scratch_root=None,
    )
    validated_merge = exact.validate_merge_result(
        merge_root,
        manifest=manifest,
        allowed_scopes=("FULL_MANIFEST",),
    )
    built = build_storage_safe_archive(
        partition_manifest=manifest_path,
        merge_root=merge_root,
        parity_receipt=parity_receipt,
        environment_manifest=environment_manifest,
        slurm_inventory=slurm_inventory,
        resource_metrics=resource_metrics,
        packaging_commit=packaging_commit,
        output_archive=scratch_archive,
        _prevalidated_merge_manifest=validated_merge,
    )
    verification = stream_verify_storage_safe_bundle(scratch_archive)
    published = publish_storage_safe_archive(
        scratch_archive=scratch_archive,
        inner_manifest=built["inner_manifest"],
        prepublication_verification=verification,
        output_root=destination,
        minimum_reserve_bytes=minimum_reserve_bytes,
        reserve_fraction=reserve_fraction,
    )
    return {
        "state": "PASS",
        "partition_manifest_sha256": manifest["manifest_sha256"],
        "source_inventory": inventory,
        "scratch_admission": scratch_admission.to_json(),
        "node_local_merge_root": str(merge_root),
        "scratch_archive": str(scratch_archive),
        "prepublication_verification": verification,
        "publication": published,
        "matrix_write_enabled": False,
    }


__all__ = [
    "DEFAULT_MIN_RESERVE_BYTES",
    "DEFAULT_RESERVE_FRACTION",
    "STORAGE_SAFE_BUNDLE_SCHEMA",
    "STORAGE_SAFE_RECEIPT_SCHEMA",
    "StorageAdmission",
    "StorageSafeT8Error",
    "build_storage_safe_archive",
    "merge_package_storage_safe",
    "node_local_scratch_admission",
    "publish_storage_safe_archive",
    "storage_admission",
    "stream_verify_storage_safe_bundle",
    "validate_storage_path_policy",
]
