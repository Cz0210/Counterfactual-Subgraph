"""Durable hierarchical exact merge for the TasteMolNet T8 HPC shards.

The mining shards are immutable scientific inputs.  A group worker validates
and normalises a disjoint subset of their partition streams in parallel, then
stores deterministic gzip chunks.  The final worker only checks cross-group
identity uniqueness and concatenates those chunks in the manifest's official
DFS preorder.  Consequently its three JSONL byte streams are identical to a
single-process :func:`globalgce_hpc_exact.merge_exact_shards` merge.

Checkpoints are deliberately coarse and durable: group workers commit only at
an input-shard boundary and the final worker commits only at a group boundary.
No classifier, calibration/test data, GPU, or matrix publication surface is
available here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import heapq
import io
from io import TextIOWrapper
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import tarfile
import time
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence, TextIO
import uuid

from src.baselines import globalgce_hpc_exact as exact


ARRAY_ADOPTION_SCHEMA = "globalgce_hpc_array_adoption_v1"
GROUP_PLAN_SCHEMA = "globalgce_hpc_hierarchical_group_plan_v1"
GROUP_CHECKPOINT_SCHEMA = "globalgce_hpc_hierarchical_group_checkpoint_v1"
GROUP_RESULT_SCHEMA = "globalgce_hpc_hierarchical_group_result_v1"
CHUNK_RESULT_SCHEMA = "globalgce_hpc_hierarchical_chunk_v1"
FINAL_CHECKPOINT_SCHEMA = "globalgce_hpc_hierarchical_final_checkpoint_v1"
FINAL_VERIFICATION_SCHEMA = "globalgce_hpc_hierarchical_final_verification_v1"
EVIDENCE_MANIFEST_SCHEMA = "globalgce_hpc_hierarchical_evidence_v1"
PACKAGE_READY_SCHEMA = "globalgce_hpc_hierarchical_package_ready_v1"
STREAM_NAMES = ("events.jsonl", "patterns.jsonl", "rejection_events.jsonl")
IDENTITY_NAMES = ("event_ids.txt.gz", "pattern_ids.txt.gz")
DEFAULT_GROUP_COUNT = 4
DEFAULT_PROGRESS_SECONDS = 300
COPY_BLOCK_BYTES = 4 * 1024 * 1024
IDENTITY_MERGE_FANIN = 128


class HierarchicalT8Error(exact.GlobalGCEHPCExactError):
    """A hierarchical exactness, provenance, or resume gate failed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    if field in result:
        raise HierarchicalT8Error(f"reserved hash field already exists: {field}")
    result[field] = exact.canonical_sha256(result)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HierarchicalT8Error(f"malformed JSON: {path}") from exc
    if type(value) is not dict:
        raise HierarchicalT8Error(f"JSON must contain one object: {path}")
    return value


def _require_self_hash(payload: Mapping[str, Any], field: str, label: str) -> None:
    claimed = payload.get(field)
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise HierarchicalT8Error(f"{label} self-hash is missing")
    unsigned = {key: value for key, value in payload.items() if key != field}
    if exact.canonical_sha256(unsigned) != claimed:
        raise HierarchicalT8Error(f"{label} self-hash mismatch")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, _canonical_bytes(dict(payload)) + b"\n")


def _copy_atomic_file(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.partial"
    )
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            for block in iter(lambda: reader.read(COPY_BLOCK_BYTES), b""):
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        if exact.sha256_file(temporary) != expected_sha256:
            raise HierarchicalT8Error("atomic copy hash mismatch")
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _partition_root(shards_root: Path, unit: Mapping[str, Any]) -> Path:
    return (
        shards_root
        / f"shard-{int(unit['shard_index']):03d}"
        / "partitions"
        / str(unit["partition_id"])
    )


def _validated_small_partition_manifest(
    path: Path, *, manifest: Mapping[str, Any], unit: Mapping[str, Any]
) -> dict[str, Any]:
    result = _read_json(path)
    _require_self_hash(result, "result_sha256", "partition manifest")
    if (
        result.get("schema_version") != exact.UNIT_RESULT_SCHEMA
        or result.get("status") != "PASS"
        or result.get("partition") != dict(unit)
        or result.get("manifest_sha256") != manifest.get("manifest_sha256")
        or result.get("scientific_input_sha256")
        != manifest.get("scientific_input_sha256")
        or result.get("provenance_sha256")
        != manifest.get("provenance", {}).get("provenance_sha256")
        or result.get("target_branches")
        != manifest.get("provenance", {}).get("target_branches")
        or result.get("matrix_write_enabled") is not False
    ):
        raise HierarchicalT8Error("partition manifest provenance changed")
    for field in ("event_count", "pattern_count", "rejection_count"):
        if type(result.get(field)) is not int or int(result[field]) < 0:
            raise HierarchicalT8Error(f"partition {field} is invalid")
    return result


def _validated_small_shard_manifest(
    path: Path, *, manifest: Mapping[str, Any], shard_index: int
) -> dict[str, Any]:
    result = _read_json(path)
    _require_self_hash(result, "result_sha256", "shard manifest")
    if (
        result.get("schema_version") != exact.SHARD_RESULT_SCHEMA
        or result.get("status") != "PASS"
        or result.get("manifest_sha256") != manifest.get("manifest_sha256")
        or result.get("scientific_input_sha256")
        != manifest.get("scientific_input_sha256")
        or result.get("provenance_sha256")
        != manifest.get("provenance", {}).get("provenance_sha256")
        or result.get("target_branches")
        != manifest.get("provenance", {}).get("target_branches")
        or result.get("shard_index") != shard_index
        or result.get("matrix_write_enabled") is not False
        or result.get("scientific_search_pruned") is not False
        or result.get("approximation_used") is not False
    ):
        raise HierarchicalT8Error("shard manifest provenance changed")
    return result


def adopt_completed_array(
    *, partition_manifest: str | Path, shards_root: str | Path, output: str | Path
) -> dict[str, Any]:
    """Adopt sealed successful shards without reopening their large streams."""

    manifest_path = Path(partition_manifest).expanduser().resolve(strict=True)
    manifest = exact.validate_partition_manifest(manifest_path)
    root = Path(shards_root).expanduser().resolve(strict=True)
    destination = Path(output).expanduser().absolute()
    if destination.exists():
        existing = _read_json(destination)
        _require_self_hash(existing, "array_adoption_sha256", "array adoption")
        return existing
    shards: list[dict[str, Any]] = []
    for shard_index in range(int(manifest["shard_count"])):
        shard_path = root / f"shard-{shard_index:03d}" / "shard_manifest.json"
        shard = _validated_small_shard_manifest(
            shard_path, manifest=manifest, shard_index=shard_index
        )
        expected_ids = [
            str(row["partition_id"])
            for row in sorted(
                (
                    row
                    for row in manifest["partitions"]
                    if int(row["shard_index"]) == shard_index
                ),
                key=lambda row: int(row["global_partition_order"]),
            )
        ]
        if shard.get("partition_ids") != expected_ids:
            raise HierarchicalT8Error("adopted shard partition coverage changed")
        if any(
            path.name.endswith(".incomplete")
            for path in (root / f"shard-{shard_index:03d}").rglob("*")
        ):
            raise HierarchicalT8Error("adopted shard still has an active partial writer")
        partition_hashes = []
        partition_totals = {
            "event_count": 0,
            "pattern_count": 0,
            "rejection_count": 0,
        }
        for unit in sorted(
            (
                row
                for row in manifest["partitions"]
                if int(row["shard_index"]) == shard_index
            ),
            key=lambda row: int(row["global_partition_order"]),
        ):
            small = _validated_small_partition_manifest(
                _partition_root(root, unit) / "partition_manifest.json",
                manifest=manifest,
                unit=unit,
            )
            partition_hashes.append(small["result_sha256"])
            for field in partition_totals:
                partition_totals[field] += int(small[field])
        if (
            shard.get("partition_result_sha256s") != partition_hashes
            or any(shard.get(field) != total for field, total in partition_totals.items())
        ):
            raise HierarchicalT8Error("adopted shard/partition hash binding changed")
        shards.append(
            {
                "shard_index": shard_index,
                "partition_ids": expected_ids,
                "result_sha256": shard["result_sha256"],
                "manifest_file_sha256": exact.sha256_file(shard_path),
                "event_count": int(shard["event_count"]),
                "pattern_count": int(shard["pattern_count"]),
                "rejection_count": int(shard["rejection_count"]),
            }
        )
    payload = _self_hashed(
        {
            "schema_version": ARRAY_ADOPTION_SCHEMA,
            "status": "PASS",
            "adopted_at": _utc_now(),
            "partition_manifest": str(manifest_path),
            "partition_manifest_sha256": manifest["manifest_sha256"],
            "partition_manifest_file_sha256": exact.sha256_file(manifest_path),
            "scientific_input_sha256": manifest["scientific_input_sha256"],
            "shards_root": str(root),
            "shard_count": len(shards),
            "passed_shard_count": len(shards),
            "successful_shards_rerun": False,
            "shards": shards,
            "matrix_write_enabled": False,
        },
        "array_adoption_sha256",
    )
    _atomic_json(destination, payload)
    return payload


def _balanced_contiguous_ranges(weights: Sequence[int], count: int) -> list[range]:
    if count < 1 or count > len(weights):
        raise HierarchicalT8Error("group count is outside the shard/partition universe")
    total = sum(max(1, int(weight)) for weight in weights)
    ranges: list[range] = []
    start = 0
    cumulative = 0
    for group_index in range(count - 1):
        target = total * (group_index + 1) / count
        end = start
        while end < len(weights) - (count - group_index - 1):
            next_value = cumulative + max(1, int(weights[end]))
            if end > start and abs(cumulative - target) <= abs(next_value - target):
                break
            cumulative = next_value
            end += 1
        ranges.append(range(start, end))
        start = end
    ranges.append(range(start, len(weights)))
    if any(not value for value in ranges):
        raise HierarchicalT8Error("deterministic group plan created an empty range")
    return ranges


def build_group_plan(
    *,
    partition_manifest: str | Path,
    shards_root: str | Path,
    array_adoption: str | Path,
    output: str | Path,
    group_count: int = DEFAULT_GROUP_COUNT,
) -> dict[str, Any]:
    """Build a deterministic, disjoint group plan from small sealed manifests."""

    manifest_path = Path(partition_manifest).expanduser().resolve(strict=True)
    manifest = exact.validate_partition_manifest(manifest_path)
    root = Path(shards_root).expanduser().resolve(strict=True)
    adoption_path = Path(array_adoption).expanduser().resolve(strict=True)
    adoption = _read_json(adoption_path)
    _require_self_hash(adoption, "array_adoption_sha256", "array adoption")
    if (
        adoption.get("status") != "PASS"
        or adoption.get("partition_manifest_sha256") != manifest["manifest_sha256"]
        or adoption.get("shards_root") != str(root)
    ):
        raise HierarchicalT8Error("array adoption is not bound to this plan")

    ordered_units = sorted(
        manifest["partitions"], key=lambda row: int(row["global_partition_order"])
    )
    event_offset = pattern_offset = rejection_offset = 0
    rows: list[dict[str, Any]] = []
    for unit in ordered_units:
        source_root = _partition_root(root, unit)
        result_path = source_root / "partition_manifest.json"
        result = _validated_small_partition_manifest(
            result_path, manifest=manifest, unit=unit
        )
        events_path = source_root / "events.jsonl"
        patterns_path = source_root / "patterns.jsonl"
        weight = events_path.stat().st_size + patterns_path.stat().st_size
        rows.append(
            {
                "partition_id": str(unit["partition_id"]),
                "global_partition_order": int(unit["global_partition_order"]),
                "source_shard_index": int(unit["shard_index"]),
                "source_result_sha256": result["result_sha256"],
                "source_manifest_file_sha256": exact.sha256_file(result_path),
                "source_events_sha256": result["events_sha256"],
                "source_patterns_sha256": result["patterns_sha256"],
                "event_count": int(result["event_count"]),
                "pattern_count": int(result["pattern_count"]),
                "rejection_count": int(result["rejection_count"]),
                "event_global_offset": event_offset,
                "pattern_global_offset": pattern_offset,
                "rejection_global_offset": rejection_offset,
                "source_payload_bytes": weight,
            }
        )
        event_offset += int(result["event_count"])
        pattern_offset += int(result["pattern_count"])
        rejection_offset += int(result["rejection_count"])

    # The production default is exactly four contiguous shard groups.  If the
    # source shard sizes are strongly skewed, deterministic contiguous
    # partition intervals prevent one pathological shard from owning a group.
    shard_weights = [0] * int(manifest["shard_count"])
    for row in rows:
        shard_weights[int(row["source_shard_index"])] += int(
            row["source_payload_bytes"]
        )
    nonzero = [value for value in shard_weights if value > 0]
    uneven = bool(nonzero) and max(nonzero) > 2 * min(nonzero)
    groups: list[dict[str, Any]] = []
    if not uneven and int(manifest["shard_count"]) >= group_count:
        shard_count = int(manifest["shard_count"])
        ranges = [
            range(index * shard_count // group_count, (index + 1) * shard_count // group_count)
            for index in range(group_count)
        ]
        assignments = [
            [row for row in rows if int(row["source_shard_index"]) in interval]
            for interval in ranges
        ]
        strategy = "CONTIGUOUS_SOURCE_SHARD_INTERVALS"
    else:
        ranges = _balanced_contiguous_ranges(
            [max(1, int(row["source_payload_bytes"])) for row in rows], group_count
        )
        assignments = [[rows[index] for index in interval] for interval in ranges]
        strategy = "CONTIGUOUS_GLOBAL_PARTITION_INTERVALS"
    for group_index, assigned in enumerate(assignments):
        assigned.sort(key=lambda row: int(row["global_partition_order"]))
        groups.append(
            {
                "group_index": group_index,
                "partition_ids": [row["partition_id"] for row in assigned],
                "source_shard_indices": sorted(
                    {int(row["source_shard_index"]) for row in assigned}
                ),
                "first_global_partition_order": int(
                    assigned[0]["global_partition_order"]
                ),
                "last_global_partition_order": int(
                    assigned[-1]["global_partition_order"]
                ),
                "source_payload_bytes": sum(
                    int(row["source_payload_bytes"]) for row in assigned
                ),
            }
        )
    flat = [partition for group in groups for partition in group["partition_ids"]]
    if len(flat) != len(set(flat)) or set(flat) != {
        row["partition_id"] for row in rows
    }:
        raise HierarchicalT8Error("group plan is not an exact partition cover")
    payload = _self_hashed(
        {
            "schema_version": GROUP_PLAN_SCHEMA,
            "status": "PASS",
            "created_at": _utc_now(),
            "partition_manifest": str(manifest_path),
            "partition_manifest_sha256": manifest["manifest_sha256"],
            "partition_manifest_file_sha256": exact.sha256_file(manifest_path),
            "scientific_input_sha256": manifest["scientific_input_sha256"],
            "shards_root": str(root),
            "array_adoption": str(adoption_path),
            "array_adoption_sha256": adoption["array_adoption_sha256"],
            "grouping_strategy": strategy,
            "group_count": len(groups),
            "groups": groups,
            "partitions": rows,
            "event_count": event_offset,
            "pattern_count": pattern_offset,
            "rejection_count": rejection_offset,
            "partition_disjoint": True,
            "partition_complete": True,
            "official_global_order_preserved": True,
            "successful_shards_rerun": False,
            "matrix_write_enabled": False,
        },
        "group_plan_sha256",
    )
    destination = Path(output).expanduser().absolute()
    if destination.exists():
        existing = _read_json(destination)
        if existing != payload:
            raise HierarchicalT8Error("fresh group-plan path already exists")
        return existing
    _atomic_json(destination, payload)
    return payload


def validate_group_plan(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path).expanduser().resolve(strict=True)
    plan = _read_json(source)
    _require_self_hash(plan, "group_plan_sha256", "group plan")
    manifest = exact.validate_partition_manifest(plan.get("partition_manifest", ""))
    if (
        plan.get("schema_version") != GROUP_PLAN_SCHEMA
        or plan.get("status") != "PASS"
        or plan.get("partition_manifest_sha256") != manifest["manifest_sha256"]
        or plan.get("scientific_input_sha256") != manifest["scientific_input_sha256"]
        or plan.get("matrix_write_enabled") is not False
        or plan.get("partition_disjoint") is not True
        or plan.get("partition_complete") is not True
        or plan.get("official_global_order_preserved") is not True
    ):
        raise HierarchicalT8Error("group plan contract is invalid")
    groups = plan.get("groups")
    partitions = plan.get("partitions")
    if (
        type(groups) is not list
        or len(groups) != plan.get("group_count")
        or type(partitions) is not list
        or any(type(value) is not dict for value in (*groups, *partitions))
    ):
        raise HierarchicalT8Error("group plan collections are malformed")
    expected = [
        str(row["partition_id"])
        for row in sorted(
            manifest["partitions"], key=lambda row: int(row["global_partition_order"])
        )
    ]
    observed = [value for group in groups for value in group.get("partition_ids", ())]
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise HierarchicalT8Error("group plan partition cover changed")
    return plan, manifest


@dataclass
class _CompressedOutput:
    path: Path
    raw_digest: hashlib._Hash
    raw_bytes: int = 0
    rows: int = 0

    @classmethod
    def open(cls, path: Path) -> tuple["_CompressedOutput", BinaryIO, gzip.GzipFile]:
        raw = path.open("xb")
        compressed = gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0
        )
        return cls(path=path, raw_digest=hashlib.sha256()), raw, compressed

    def write(self, stream: gzip.GzipFile, data: bytes) -> None:
        stream.write(data)
        self.raw_digest.update(data)
        self.raw_bytes += len(data)
        self.rows += 1


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HierarchicalT8Error(f"invalid JSONL {path}:{line_number}") from exc
            if type(value) is not dict:
                raise HierarchicalT8Error(f"non-object JSONL row: {path}")
            yield value


def _write_sorted_ids(path: Path, identities: Iterable[str]) -> dict[str, Any]:
    values = sorted(identities)
    if len(values) != len(set(values)):
        raise HierarchicalT8Error("duplicate identity within one exact partition")
    raw_digest = hashlib.sha256()
    raw_bytes = 0
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            for value in values:
                line = value.encode("ascii") + b"\n"
                stream.write(line)
                raw_digest.update(line)
                raw_bytes += len(line)
        raw.flush()
        os.fsync(raw.fileno())
    return {
        "rows": len(values),
        "raw_bytes": raw_bytes,
        "raw_sha256": raw_digest.hexdigest(),
        "gzip_bytes": path.stat().st_size,
        "gzip_sha256": exact.sha256_file(path),
    }


def _write_partition_chunk(
    *,
    manifest: Mapping[str, Any],
    unit: Mapping[str, Any],
    plan_row: Mapping[str, Any],
    source_root: Path,
    destination: Path,
    scratch_root: Path | None,
) -> dict[str, Any]:
    result = exact.validate_unit_result(
        source_root, manifest=manifest, expected_unit=unit
    )
    if result["result_sha256"] != plan_row["source_result_sha256"]:
        raise HierarchicalT8Error("source partition changed after group planning")
    temporary_parent = destination.parent if scratch_root is None else scratch_root
    temporary_parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_parent / (
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.incomplete"
    )
    temporary.mkdir(parents=True, mode=0o700)
    streams: dict[str, tuple[_CompressedOutput, BinaryIO, gzip.GzipFile]] = {}
    event_ids: list[str] = []
    event_id_set: set[str] = set()
    pattern_ids: list[str] = []
    pattern_id_set: set[str] = set()
    local_top_k: list[dict[str, Any]] = []
    top_k = int(manifest["configuration"]["top_k"])
    try:
        for name in STREAM_NAMES:
            streams[name] = _CompressedOutput.open(temporary / f"{name}.gz")
        for local_index, row in enumerate(_iter_jsonl(source_root / "events.jsonl")):
            identity = str(row.get("dfs_code_sha256"))
            if identity in event_id_set:
                raise HierarchicalT8Error("duplicate event in source partition")
            event_id_set.add(identity)
            event_ids.append(identity)
            merged = {
                **row,
                "global_preorder": int(plan_row["event_global_offset"]) + local_index,
                "global_partition_order": int(plan_row["global_partition_order"]),
            }
            line = _canonical_bytes(merged) + b"\n"
            streams["events.jsonl"][0].write(streams["events.jsonl"][2], line)
            if row.get("status") != "ACCEPTED":
                streams["rejection_events.jsonl"][0].write(
                    streams["rejection_events.jsonl"][2], line
                )
        for local_index, row in enumerate(_iter_jsonl(source_root / "patterns.jsonl")):
            identity = str(row.get("pattern_sha256"))
            if identity in pattern_id_set:
                raise HierarchicalT8Error("duplicate pattern in source partition")
            pattern_id_set.add(identity)
            pattern_ids.append(identity)
            merged = {
                **row,
                "global_preorder": int(plan_row["pattern_global_offset"])
                + local_index,
                "global_partition_order": int(plan_row["global_partition_order"]),
            }
            line = _canonical_bytes(merged) + b"\n"
            streams["patterns.jsonl"][0].write(
                streams["patterns.jsonl"][2], line
            )
            candidate = exact._normalized_pattern(merged) | {  # type: ignore[attr-defined]
                "global_preorder": merged["global_preorder"]
            }
            local_top_k.append(candidate)
            local_top_k.sort(
                key=lambda value: (-int(value["support"]), int(value["global_preorder"]))
            )
            if len(local_top_k) > top_k:
                local_top_k.pop()
        stream_rows: dict[str, Any] = {}
        for name, (summary, raw, compressed) in streams.items():
            compressed.close()
            raw.flush()
            os.fsync(raw.fileno())
            raw.close()
            stream_rows[name] = {
                "rows": summary.rows,
                "raw_bytes": summary.raw_bytes,
                "raw_sha256": summary.raw_digest.hexdigest(),
                "gzip_bytes": summary.path.stat().st_size,
                "gzip_sha256": exact.sha256_file(summary.path),
            }
        if (
            stream_rows["events.jsonl"]["rows"] != int(result["event_count"])
            or stream_rows["patterns.jsonl"]["rows"] != int(result["pattern_count"])
            or stream_rows["rejection_events.jsonl"]["rows"]
            != int(result["rejection_count"])
        ):
            raise HierarchicalT8Error("normalised chunk counts changed")
        identities = {
            "event_ids.txt.gz": _write_sorted_ids(
                temporary / "event_ids.txt.gz", event_ids
            ),
            "pattern_ids.txt.gz": _write_sorted_ids(
                temporary / "pattern_ids.txt.gz", pattern_ids
            ),
        }
        payload = _self_hashed(
            {
                "schema_version": CHUNK_RESULT_SCHEMA,
                "status": "PASS",
                "partition_id": str(unit["partition_id"]),
                "global_partition_order": int(unit["global_partition_order"]),
                "source_shard_index": int(unit["shard_index"]),
                "source_result_sha256": result["result_sha256"],
                "source_events_sha256": result["events_sha256"],
                "source_patterns_sha256": result["patterns_sha256"],
                "event_global_offset": int(plan_row["event_global_offset"]),
                "pattern_global_offset": int(plan_row["pattern_global_offset"]),
                "streams": stream_rows,
                "identities": identities,
                "stable_top_k_candidates": local_top_k,
                "scientific_search_pruned": False,
                "approximation_used": False,
                "matrix_write_enabled": False,
            },
            "chunk_result_sha256",
        )
        exact.atomic_write_json(temporary / "chunk_manifest.json", payload)
        _fsync_directory(temporary)
        if destination.exists():
            raise HierarchicalT8Error("partition chunk appeared during publication")
        if scratch_root is None:
            os.rename(temporary, destination)
            _fsync_directory(destination.parent)
        else:
            persistent_temporary = destination.parent / (
                f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.copying"
            )
            persistent_temporary.mkdir(parents=True, mode=0o700)
            try:
                for source in temporary.iterdir():
                    if not source.is_file():
                        raise HierarchicalT8Error("chunk scratch contains non-file payload")
                    _copy_atomic_file(
                        source, persistent_temporary / source.name, exact.sha256_file(source)
                    )
                if destination.exists():
                    raise HierarchicalT8Error("partition chunk appeared during copy")
                os.rename(persistent_temporary, destination)
                _fsync_directory(destination.parent)
            except BaseException:
                shutil.rmtree(persistent_temporary, ignore_errors=True)
                raise
            shutil.rmtree(temporary)
        return payload
    except BaseException:
        for _summary, raw, compressed in streams.values():
            try:
                compressed.close()
            except Exception:
                pass
            try:
                raw.close()
            except Exception:
                pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_chunk(root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    payload = _read_json(root / "chunk_manifest.json")
    _require_self_hash(payload, "chunk_result_sha256", "partition chunk")
    if (
        payload.get("schema_version") != CHUNK_RESULT_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("partition_id") != expected.get("partition_id")
        or payload.get("source_result_sha256")
        != expected.get("source_result_sha256")
        or payload.get("global_partition_order")
        != expected.get("global_partition_order")
        or payload.get("matrix_write_enabled") is not False
    ):
        raise HierarchicalT8Error("partition chunk contract changed")
    if set(payload.get("streams", {})) != set(STREAM_NAMES):
        raise HierarchicalT8Error("partition chunk stream inventory is incomplete")
    if set(payload.get("identities", {})) != set(IDENTITY_NAMES):
        raise HierarchicalT8Error("partition chunk identity inventory is incomplete")
    for name, identity in payload.get("streams", {}).items():
        path = root / f"{name}.gz"
        if not path.is_file() or exact.sha256_file(path) != identity.get("gzip_sha256"):
            raise HierarchicalT8Error("partition compressed stream changed")
    for name, identity in payload.get("identities", {}).items():
        path = root / name
        if not path.is_file() or exact.sha256_file(path) != identity.get("gzip_sha256"):
            raise HierarchicalT8Error("partition identity stream changed")
    return payload


def _load_checkpoint(path: Path, *, schema: str, hash_field: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _read_json(path)
    _require_self_hash(payload, hash_field, "checkpoint")
    if payload.get("schema_version") != schema:
        raise HierarchicalT8Error("checkpoint schema changed")
    return payload


def _merge_sorted_identity_files(
    paths: Sequence[Path], output: Path | None = None
) -> tuple[int, str | None]:
    if len(paths) > IDENTITY_MERGE_FANIN:
        if output is None:
            raise HierarchicalT8Error("large identity merge requires a scratch output")
        intermediates: list[Path] = []
        try:
            for index in range(0, len(paths), IDENTITY_MERGE_FANIN):
                intermediate = output.parent / (
                    f".{output.name}.level.{index // IDENTITY_MERGE_FANIN:04d}.gz"
                )
                _merge_sorted_identity_files(
                    paths[index : index + IDENTITY_MERGE_FANIN], intermediate
                )
                intermediates.append(intermediate)
            return _merge_sorted_identity_files(intermediates, output)
        finally:
            for intermediate in intermediates:
                intermediate.unlink(missing_ok=True)
    streams: list[TextIO] = []
    compressed: list[gzip.GzipFile] = []
    writer_raw: BinaryIO | None = None
    writer: gzip.GzipFile | None = None
    digest = hashlib.sha256()
    count = 0
    previous: str | None = None
    try:
        for path in paths:
            gz = gzip.open(path, "rb")
            compressed.append(gz)
            streams.append(TextIOWrapper(gz, encoding="ascii"))
        if output is not None:
            writer_raw = output.open("xb")
            writer = gzip.GzipFile(filename="", mode="wb", fileobj=writer_raw, mtime=0)
        iterators = ((line.rstrip("\n") for line in stream) for stream in streams)
        for value in heapq.merge(*iterators):
            if value == previous:
                raise HierarchicalT8Error("duplicate scientific identity across groups")
            if previous is not None and value < previous:
                raise HierarchicalT8Error("identity merge order regressed")
            encoded = value.encode("ascii") + b"\n"
            if writer is not None:
                writer.write(encoded)
                digest.update(encoded)
            previous = value
            count += 1
        if writer is not None:
            writer.close()
            writer = None
            assert writer_raw is not None
            writer_raw.flush()
            os.fsync(writer_raw.fileno())
            writer_raw.close()
            writer_raw = None
        return count, digest.hexdigest() if output is not None else None
    finally:
        if writer is not None:
            writer.close()
        if writer_raw is not None:
            writer_raw.close()
        for stream in streams:
            stream.close()


def run_group_merge(
    *,
    group_plan: str | Path,
    group_index: int,
    output_root: str | Path,
    scratch_root: str | Path | None = None,
    progress_seconds: int = DEFAULT_PROGRESS_SECONDS,
) -> dict[str, Any]:
    """Validate and seal one group; safely resumes after source-shard commits."""

    plan_path = Path(group_plan).expanduser().resolve(strict=True)
    plan, manifest = validate_group_plan(plan_path)
    if type(group_index) is not int or not 0 <= group_index < int(plan["group_count"]):
        raise HierarchicalT8Error("group index is outside the plan")
    group = plan["groups"][group_index]
    root = Path(output_root).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    terminal = root / "group_manifest.json"
    if terminal.exists():
        return validate_group_result(root, plan=plan, group_index=group_index)
    chunks_root = root / "chunks"
    chunks_root.mkdir(exist_ok=True)
    scratch_parent = (
        Path(scratch_root).expanduser().absolute()
        if scratch_root is not None
        else Path(tempfile.gettempdir())
    )
    scratch_parent.mkdir(parents=True, exist_ok=True)
    rows_by_id = {row["partition_id"]: row for row in plan["partitions"]}
    units_by_id = {str(row["partition_id"]): row for row in manifest["partitions"]}
    assigned = [rows_by_id[value] for value in group["partition_ids"]]
    shards_root = Path(plan["shards_root"]).resolve(strict=True)
    checkpoint_path = root / "checkpoint.json"
    checkpoint = _load_checkpoint(
        checkpoint_path,
        schema=GROUP_CHECKPOINT_SCHEMA,
        hash_field="checkpoint_sha256",
    )
    completed_shards = [] if checkpoint is None else list(checkpoint["completed_shard_indices"])
    if checkpoint is not None and (
        checkpoint.get("group_plan_sha256") != plan["group_plan_sha256"]
        or checkpoint.get("group_index") != group_index
    ):
        raise HierarchicalT8Error("group checkpoint belongs to another plan")
    started = time.monotonic()
    last_progress = started
    for shard_index in group["source_shard_indices"]:
        shard_rows = [
            row for row in assigned if int(row["source_shard_index"]) == shard_index
        ]
        if shard_index in completed_shards:
            for row in shard_rows:
                _validate_chunk(chunks_root / row["partition_id"], row)
            continue
        for row in shard_rows:
            unit = units_by_id[row["partition_id"]]
            destination = chunks_root / row["partition_id"]
            if destination.exists():
                _validate_chunk(destination, row)
            else:
                _write_partition_chunk(
                    manifest=manifest,
                    unit=unit,
                    plan_row=row,
                    source_root=_partition_root(shards_root, unit),
                    destination=destination,
                    scratch_root=scratch_parent if scratch_root is not None else None,
                )
            now = time.monotonic()
            if now - last_progress >= max(1, progress_seconds):
                _atomic_json(
                    root / "progress.json",
                    {
                        "state": "RUNNING",
                        "group_index": group_index,
                        "current_shard_index": shard_index,
                        "current_partition_id": row["partition_id"],
                        "elapsed_seconds": now - started,
                        "updated_at": _utc_now(),
                    },
                )
                last_progress = now
        completed_shards.append(shard_index)
        completed_shards.sort()
        checkpoint = _self_hashed(
            {
                "schema_version": GROUP_CHECKPOINT_SCHEMA,
                "state": "COMMITTED_SOURCE_SHARD_BOUNDARY",
                "group_plan_sha256": plan["group_plan_sha256"],
                "group_index": group_index,
                "completed_shard_indices": completed_shards,
                "completed_partition_ids": [
                    row["partition_id"]
                    for row in assigned
                    if int(row["source_shard_index"]) in completed_shards
                ],
                "resume_boundary": "SEALED_INPUT_SHARD_ONLY",
                "written_at": _utc_now(),
            },
            "checkpoint_sha256",
        )
        _atomic_json(checkpoint_path, checkpoint)

    chunk_rows = [
        _validate_chunk(chunks_root / row["partition_id"], row) for row in assigned
    ]
    scratch = Path(tempfile.mkdtemp(prefix=f"t8-group-{group_index:02d}-", dir=scratch_parent))
    try:
        identity_results: dict[str, Any] = {}
        for name in IDENTITY_NAMES:
            source_paths = [chunks_root / row["partition_id"] / name for row in assigned]
            target = scratch / name
            count, raw_sha = _merge_sorted_identity_files(source_paths, target)
            persistent = root / name
            target_sha = exact.sha256_file(target)
            _copy_atomic_file(target, persistent, target_sha)
            identity_results[name] = {
                "rows": count,
                "raw_sha256": raw_sha,
                "gzip_bytes": persistent.stat().st_size,
                "gzip_sha256": target_sha,
            }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    top_k = int(manifest["configuration"]["top_k"])
    candidates = [
        candidate
        for chunk in chunk_rows
        for candidate in chunk["stable_top_k_candidates"]
    ]
    candidates.sort(
        key=lambda value: (-int(value["support"]), int(value["global_preorder"]))
    )
    candidates = candidates[:top_k]
    payload = _self_hashed(
        {
            "schema_version": GROUP_RESULT_SCHEMA,
            "status": "PASS",
            "completed_at": _utc_now(),
            "group_plan_sha256": plan["group_plan_sha256"],
            "partition_manifest_sha256": manifest["manifest_sha256"],
            "scientific_input_sha256": manifest["scientific_input_sha256"],
            "group_index": group_index,
            "source_shard_indices": group["source_shard_indices"],
            "ordered_partition_ids": group["partition_ids"],
            "chunk_result_sha256s": [row["chunk_result_sha256"] for row in chunk_rows],
            "event_count": sum(row["streams"]["events.jsonl"]["rows"] for row in chunk_rows),
            "pattern_count": sum(row["streams"]["patterns.jsonl"]["rows"] for row in chunk_rows),
            "rejection_count": sum(
                row["streams"]["rejection_events.jsonl"]["rows"] for row in chunk_rows
            ),
            "identity_streams": identity_results,
            "stable_top_k_candidates": candidates,
            "checkpoint_sha256": checkpoint["checkpoint_sha256"] if checkpoint else None,
            "checkpoint_boundary": "SEALED_INPUT_SHARD",
            "partition_disjoint_within_group": True,
            "scientific_search_pruned": False,
            "approximation_used": False,
            "matrix_write_enabled": False,
        },
        "group_result_sha256",
    )
    _atomic_json(terminal, payload)
    _atomic_json(
        root / "progress.json",
        {
            "state": "PASS",
            "group_index": group_index,
            "elapsed_seconds": time.monotonic() - started,
            "updated_at": _utc_now(),
        },
    )
    return payload


def validate_group_result(
    root: str | Path, *, plan: Mapping[str, Any], group_index: int
) -> dict[str, Any]:
    directory = Path(root).expanduser().resolve(strict=True)
    payload = _read_json(directory / "group_manifest.json")
    _require_self_hash(payload, "group_result_sha256", "group result")
    group = plan["groups"][group_index]
    if (
        payload.get("schema_version") != GROUP_RESULT_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("group_plan_sha256") != plan.get("group_plan_sha256")
        or payload.get("group_index") != group_index
        or payload.get("ordered_partition_ids") != group.get("partition_ids")
        or payload.get("partition_disjoint_within_group") is not True
        or payload.get("matrix_write_enabled") is not False
    ):
        raise HierarchicalT8Error("group result contract changed")
    rows = {row["partition_id"]: row for row in plan["partitions"]}
    chunks = [
        _validate_chunk(directory / "chunks" / partition_id, rows[partition_id])
        for partition_id in group["partition_ids"]
    ]
    if payload.get("chunk_result_sha256s") != [
        row["chunk_result_sha256"] for row in chunks
    ]:
        raise HierarchicalT8Error("group/chunk binding changed")
    for name in IDENTITY_NAMES:
        path = directory / name
        identity = payload.get("identity_streams", {}).get(name, {})
        if not path.is_file() or exact.sha256_file(path) != identity.get("gzip_sha256"):
            raise HierarchicalT8Error("group identity stream changed")
    return payload


def _copy_gzip_member(
    source: Path, target: BinaryIO, aggregate: hashlib._Hash, expected: Mapping[str, Any]
) -> tuple[int, int]:
    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    with gzip.open(source, "rb") as stream:
        for block in iter(lambda: stream.read(COPY_BLOCK_BYTES), b""):
            target.write(block)
            aggregate.update(block)
            digest.update(block)
            byte_count += len(block)
            row_count += block.count(b"\n")
    if (
        byte_count != expected.get("raw_bytes")
        or row_count != expected.get("rows")
        or digest.hexdigest() != expected.get("raw_sha256")
    ):
        raise HierarchicalT8Error("compressed group chunk did not round-trip")
    return byte_count, row_count


def finalize_hierarchical_merge(
    *,
    group_plan: str | Path,
    groups_root: str | Path,
    state_root: str | Path,
    scratch_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Seal the exact monolithic byte streams from completed group chunks."""

    plan_path = Path(group_plan).expanduser().resolve(strict=True)
    plan, manifest = validate_group_plan(plan_path)
    groups = Path(groups_root).expanduser().resolve(strict=True)
    state = Path(state_root).expanduser().absolute()
    state.mkdir(parents=True, exist_ok=True)
    destination = Path(output_root).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return exact.validate_merge_result(
            destination, manifest=manifest, allowed_scopes=("FULL_MANIFEST",)
        )
    checkpoint_path = state / "checkpoint.json"
    checkpoint = _load_checkpoint(
        checkpoint_path,
        schema=FINAL_CHECKPOINT_SCHEMA,
        hash_field="checkpoint_sha256",
    )
    completed = [] if checkpoint is None else list(checkpoint["completed_group_indices"])
    if checkpoint is not None and checkpoint.get("group_plan_sha256") != plan["group_plan_sha256"]:
        raise HierarchicalT8Error("final checkpoint belongs to another plan")
    group_results: list[dict[str, Any]] = []
    for group_index in range(int(plan["group_count"])):
        result = validate_group_result(
            groups / f"group-{group_index:02d}", plan=plan, group_index=group_index
        )
        group_results.append(result)
        if group_index not in completed:
            completed.append(group_index)
            completed.sort()
            checkpoint = _self_hashed(
                {
                    "schema_version": FINAL_CHECKPOINT_SCHEMA,
                    "state": "COMMITTED_GROUP_BOUNDARY",
                    "group_plan_sha256": plan["group_plan_sha256"],
                    "completed_group_indices": completed,
                    "group_result_sha256s": [
                        group_results[index]["group_result_sha256"]
                        for index in range(len(group_results))
                    ],
                    "resume_boundary": "SEALED_GROUP_ONLY",
                    "written_at": _utc_now(),
                },
                "checkpoint_sha256",
            )
            _atomic_json(checkpoint_path, checkpoint)

    for name, expected_count in (
        ("event_ids.txt.gz", int(plan["event_count"])),
        ("pattern_ids.txt.gz", int(plan["pattern_count"])),
    ):
        count, _digest = _merge_sorted_identity_files(
            [groups / f"group-{index:02d}" / name for index in range(len(group_results))]
        )
        if count != expected_count:
            raise HierarchicalT8Error("global identity count differs from source manifests")

    scratch_parent = Path(scratch_root).expanduser().resolve(strict=True)
    work = Path(tempfile.mkdtemp(prefix="t8-hierarchical-final-", dir=scratch_parent))
    rows_by_id = {row["partition_id"]: row for row in plan["partitions"]}
    chunk_locations: dict[str, Path] = {}
    for group_index, group in enumerate(plan["groups"]):
        for partition_id in group["partition_ids"]:
            chunk_locations[partition_id] = (
                groups / f"group-{group_index:02d}" / "chunks" / partition_id
            )
    ordered_ids = [
        row["partition_id"]
        for row in sorted(plan["partitions"], key=lambda row: row["global_partition_order"])
    ]
    stream_hashes = {name: hashlib.sha256() for name in STREAM_NAMES}
    stream_counts = {name: 0 for name in STREAM_NAMES}
    handles: dict[str, BinaryIO] = {}
    try:
        for name in STREAM_NAMES:
            handles[name] = (work / name).open("xb")
        for partition_id in ordered_ids:
            chunk_root = chunk_locations[partition_id]
            chunk = _validate_chunk(chunk_root, rows_by_id[partition_id])
            for name in STREAM_NAMES:
                _bytes, count = _copy_gzip_member(
                    chunk_root / f"{name}.gz",
                    handles[name],
                    stream_hashes[name],
                    chunk["streams"][name],
                )
                stream_counts[name] += count
        for stream in handles.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        handles.clear()
        if (
            stream_counts["events.jsonl"] != int(plan["event_count"])
            or stream_counts["patterns.jsonl"] != int(plan["pattern_count"])
            or stream_counts["rejection_events.jsonl"] != int(plan["rejection_count"])
        ):
            raise HierarchicalT8Error("final stream counts changed")
        top_k = int(manifest["configuration"]["top_k"])
        selected = [
            row for result in group_results for row in result["stable_top_k_candidates"]
        ]
        selected.sort(
            key=lambda value: (-int(value["support"]), int(value["global_preorder"]))
        )
        selected = selected[:top_k]
        top_payload = {
            "schema_version": "globalgce_hpc_exact_stable_top_k_v1",
            "top_k": top_k,
            "selected_count": len(selected),
            "ordering": "SUPPORT_DESC_OFFICIAL_PREORDER_ASC",
            "selected": selected,
            "selected_sha256": exact.canonical_sha256(selected),
        }
        exact.atomic_write_json(work / "stable_top_k.json", top_payload)
        merge_payload = _self_hashed(
            {
                "schema_version": exact.MERGE_RESULT_SCHEMA,
                "status": "PASS",
                "scope": "FULL_MANIFEST",
                "completed_at": _utc_now(),
                "manifest_sha256": manifest["manifest_sha256"],
                "scientific_input_sha256": manifest["scientific_input_sha256"],
                "provenance_sha256": manifest["provenance"]["provenance_sha256"],
                "target_branches": manifest["provenance"]["target_branches"],
                "ordered_partition_ids": ordered_ids,
                "event_count": stream_counts["events.jsonl"],
                "pattern_count": stream_counts["patterns.jsonl"],
                "rejection_count": stream_counts["rejection_events.jsonl"],
                "events_sha256": stream_hashes["events.jsonl"].hexdigest(),
                "patterns_sha256": stream_hashes["patterns.jsonl"].hexdigest(),
                "rejection_events_sha256": stream_hashes[
                    "rejection_events.jsonl"
                ].hexdigest(),
                "stable_top_k_sha256": exact.sha256_file(work / "stable_top_k.json"),
                "stable_top_k_selected_sha256": top_payload["selected_sha256"],
                "stable_top_k_selected_count": len(selected),
                "global_order": "OFFICIAL_ROOT_AND_DFS_PREORDER",
                "partition_disjoint": True,
                "partition_complete": True,
                "full_root_universe_complete": True,
                "duplicate_pattern_count": 0,
                "duplicate_event_count": 0,
                "scientific_search_pruned": False,
                "approximation_used": False,
                "matrix_write_enabled": False,
                "scratch_staging_used": True,
            },
            "result_sha256",
        )
        exact.atomic_write_json(work / "merge_manifest.json", merge_payload)
        verification = _self_hashed(
            {
                "schema_version": FINAL_VERIFICATION_SCHEMA,
                "status": "PASS",
                "verified_at": _utc_now(),
                "group_plan_sha256": plan["group_plan_sha256"],
                "group_result_sha256s": [
                    row["group_result_sha256"] for row in group_results
                ],
                "merge_result_sha256": merge_payload["result_sha256"],
                "events_sha256": merge_payload["events_sha256"],
                "patterns_sha256": merge_payload["patterns_sha256"],
                "rejection_events_sha256": merge_payload[
                    "rejection_events_sha256"
                ],
                "event_identity_unique": True,
                "pattern_identity_unique": True,
                "group_boundary_checkpoint_sha256": checkpoint[
                    "checkpoint_sha256"
                ],
                "monolithic_byte_contract": "EXACT_CANONICAL_JSONL",
                "matrix_write_enabled": False,
            },
            "verification_sha256",
        )
        exact.atomic_write_json(work / "hierarchical_verification.json", verification)
        _fsync_directory(work)
        staging = destination.parent / (
            f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.copying"
        )
        staging.mkdir(parents=True, mode=0o700)
        for name in (*STREAM_NAMES, "stable_top_k.json", "merge_manifest.json", "hierarchical_verification.json"):
            target = staging / name
            shutil.copyfile(work / name, target)
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
        _fsync_directory(staging)
        if destination.exists():
            raise HierarchicalT8Error("final merge root appeared during publication")
        os.rename(staging, destination)
        _fsync_directory(destination.parent)
        return merge_payload
    finally:
        for stream in handles.values():
            try:
                stream.close()
            except Exception:
                pass
        shutil.rmtree(work, ignore_errors=True)


def monolithic_stream_parity(
    hierarchical_root: str | Path, monolithic_root: str | Path
) -> dict[str, Any]:
    """Compare scientific bytes and stable-top-k identity, ignoring timestamps."""

    hierarchical = Path(hierarchical_root).expanduser().resolve(strict=True)
    monolithic = Path(monolithic_root).expanduser().resolve(strict=True)
    comparisons = {
        name: {
            "hierarchical_sha256": exact.sha256_file(hierarchical / name),
            "monolithic_sha256": exact.sha256_file(monolithic / name),
        }
        for name in (*STREAM_NAMES, "stable_top_k.json")
    }
    for row in comparisons.values():
        row["equal"] = row["hierarchical_sha256"] == row["monolithic_sha256"]
    return {
        "status": "PASS" if all(row["equal"] for row in comparisons.values()) else "FAILED",
        "streams": comparisons,
    }


def publish_hierarchical_evidence(
    *,
    group_plan: str | Path,
    groups_root: str | Path,
    merge_root: str | Path,
    package_root: str | Path,
    storage_safe_receipt: Mapping[str, Any],
    scratch_root: str | Path,
) -> dict[str, Any]:
    """Append compact hierarchy evidence and publish the final READY marker.

    The large exact streams stay inside the existing storage-safe bundle.  The
    supplemental archive contains only small provenance/checkpoint manifests,
    so AutoDL can prove that the bundle came from the adopted 16-shard array
    and the fresh group/final dependency chain.
    """

    plan_path = Path(group_plan).expanduser().resolve(strict=True)
    plan, _manifest = validate_group_plan(plan_path)
    groups = Path(groups_root).expanduser().resolve(strict=True)
    merge = Path(merge_root).expanduser().resolve(strict=True)
    package = Path(package_root).expanduser().resolve(strict=True)
    scratch = Path(scratch_root).expanduser().resolve(strict=True)
    adoption_path = Path(plan["array_adoption"]).resolve(strict=True)
    adoption = _read_json(adoption_path)
    _require_self_hash(adoption, "array_adoption_sha256", "array adoption")
    final_verification_path = merge / "hierarchical_verification.json"
    final_verification = _read_json(final_verification_path)
    _require_self_hash(
        final_verification, "verification_sha256", "hierarchical final verification"
    )
    files: list[tuple[Path, str]] = [
        (adoption_path, "array_adoption_manifest.json"),
        (plan_path, "group_plan.json"),
        (final_verification_path, "final/hierarchical_verification.json"),
    ]
    group_hashes: list[str] = []
    for group_index in range(int(plan["group_count"])):
        root = groups / f"group-{group_index:02d}"
        group = validate_group_result(root, plan=plan, group_index=group_index)
        group_hashes.append(group["group_result_sha256"])
        files.extend(
            [
                (root / "checkpoint.json", f"groups/group-{group_index:02d}/checkpoint.json"),
                (root / "group_manifest.json", f"groups/group-{group_index:02d}/group_manifest.json"),
            ]
        )
    files.sort(key=lambda value: value[1])
    identities = [
        {
            "name": name,
            "bytes": source.stat().st_size,
            "sha256": exact.sha256_file(source),
        }
        for source, name in files
    ]
    evidence_manifest = _self_hashed(
        {
            "schema_version": EVIDENCE_MANIFEST_SCHEMA,
            "status": "PASS",
            "group_plan_sha256": plan["group_plan_sha256"],
            "array_adoption_sha256": adoption["array_adoption_sha256"],
            "group_result_sha256s": group_hashes,
            "final_verification_sha256": final_verification["verification_sha256"],
            "merge_result_sha256": final_verification["merge_result_sha256"],
            "storage_safe_receipt_sha256": storage_safe_receipt["receipt_sha256"],
            "files": identities,
            "successful_shards_rerun": False,
            "matrix_write_enabled": False,
        },
        "evidence_manifest_sha256",
    )
    archive = scratch / "t8_hierarchical_evidence.tar.gz"
    if archive.exists():
        raise HierarchicalT8Error("hierarchical evidence scratch archive must be fresh")
    manifest_bytes = _canonical_bytes(evidence_manifest) + b"\n"
    with archive.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                info = tarfile.TarInfo("EVIDENCE_MANIFEST.json")
                info.size = len(manifest_bytes)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                info.mode = 0o600
                tar.addfile(info, io.BytesIO(manifest_bytes))
                for source, name in files:
                    info = tarfile.TarInfo(name)
                    info.size = source.stat().st_size
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o600
                    with source.open("rb") as stream:
                        tar.addfile(info, stream)
        raw.flush()
        os.fsync(raw.fileno())
    expected_members = {row["name"]: row for row in identities}
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if [member.name for member in members] != [
            "EVIDENCE_MANIFEST.json",
            *[row["name"] for row in identities],
        ]:
            raise HierarchicalT8Error("hierarchical evidence archive order changed")
        extracted_manifest = tar.extractfile(members[0])
        if extracted_manifest is None or extracted_manifest.read() != manifest_bytes:
            raise HierarchicalT8Error("hierarchical evidence manifest round-trip failed")
        for member in members[1:]:
            stream = tar.extractfile(member)
            if stream is None:
                raise HierarchicalT8Error("hierarchical evidence member is unreadable")
            digest = hashlib.sha256()
            size = 0
            for block in iter(lambda: stream.read(COPY_BLOCK_BYTES), b""):
                digest.update(block)
                size += len(block)
            expected = expected_members[member.name]
            if size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
                raise HierarchicalT8Error("hierarchical evidence round-trip failed")
    evidence_archive_sha = exact.sha256_file(archive)
    evidence_receipt = _self_hashed(
        {
            "schema_version": EVIDENCE_MANIFEST_SCHEMA,
            "status": "PASS",
            "archive_name": archive.name,
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": evidence_archive_sha,
            "evidence_manifest_sha256": evidence_manifest[
                "evidence_manifest_sha256"
            ],
            "merge_result_sha256": final_verification["merge_result_sha256"],
            "matrix_write_enabled": False,
        },
        "receipt_sha256",
    )
    for source, name in (
        (archive, archive.name),
    ):
        _copy_atomic_file(source, package / name, evidence_archive_sha)
    exact.atomic_write_json(package / "hierarchical_evidence_manifest.json", evidence_receipt)
    ready = _self_hashed(
        {
            "schema_version": PACKAGE_READY_SCHEMA,
            "status": "PASS",
            "published_at": _utc_now(),
            "result_archive_sha256": storage_safe_receipt["archive_sha256"],
            "result_receipt_sha256": storage_safe_receipt["receipt_sha256"],
            "evidence_archive_sha256": evidence_archive_sha,
            "evidence_receipt_sha256": evidence_receipt["receipt_sha256"],
            "merge_result_sha256": final_verification["merge_result_sha256"],
            "matrix_write_enabled": False,
        },
        "package_ready_sha256",
    )
    exact.atomic_write_json(package / "HIERARCHICAL_PACKAGE_READY.json", ready)
    return ready
