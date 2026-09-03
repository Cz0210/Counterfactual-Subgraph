"""Buffered-writer and ordered-collector canary for TasteMolNet T12.

The canary consumes captured JSON scientific records and proves only byte-level
I/O ordering, hash-chain, checkpoint, and reload equivalence.  It does not run
VRRW, GINE, or NeuroSED and therefore cannot claim scientific parity or replace
the protected production worker by itself.
"""

from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import struct
import time
from typing import Any, Mapping, Sequence


CANARY_SCHEMA = "tastemolnet_t12_buffered_ordered_io_canary_v1"
JOURNAL_SCHEMA = "tastemolnet_t12_buffered_ordered_journal_v1"
MAGIC = b"T12OIO1\n"
EMBEDDING_MAGIC = b"T12EMB1\n"
_LENGTH = struct.Struct(">I")


class T12OrderedIOCanaryError(RuntimeError):
    """The isolated ordered-I/O canary failed closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path, *, byte_limit: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = byte_limit
    with path.open("rb") as stream:
        while remaining is None or remaining > 0:
            size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            block = stream.read(size)
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    if remaining not in (None, 0):
        raise T12OrderedIOCanaryError("journal is shorter than its committed prefix")
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(_canonical_bytes(dict(payload)) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def load_captured_records(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    source = Path(path).expanduser().resolve(strict=True)
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, text in enumerate(stream, start=1):
            try:
                row = json.loads(
                    text,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON constant {value}")
                    ),
                )
            except (ValueError, json.JSONDecodeError) as exc:
                raise T12OrderedIOCanaryError(
                    f"invalid captured record at line {line_number}"
                ) from exc
            if type(row) is not dict or set(row) != {"sequence_id", "scientific_record"}:
                raise T12OrderedIOCanaryError(
                    "captured rows require exactly sequence_id and scientific_record"
                )
            if type(row["sequence_id"]) is not int or row["sequence_id"] != line_number:
                raise T12OrderedIOCanaryError(
                    "captured sequence IDs must be consecutive and start at one"
                )
            if type(row["scientific_record"]) is not dict:
                raise T12OrderedIOCanaryError("scientific_record must be a JSON object")
            # Canonicalization is also a strict serializability/finite-number gate.
            _canonical_bytes(row["scientific_record"])
            _validate_embedding_record(row["scientific_record"])
            records.append(row)
    if not records:
        raise T12OrderedIOCanaryError("captured input is empty")
    return records, _sha256_file(source)


def prepare_record(row: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    """Pure canary executor; it never performs or approximates science."""

    if type(row) is not dict or set(row) != {"sequence_id", "scientific_record"}:
        raise T12OrderedIOCanaryError("executor input schema mismatch")
    sequence = row["sequence_id"]
    if type(sequence) is not int or sequence <= 0 or type(row["scientific_record"]) is not dict:
        raise T12OrderedIOCanaryError("executor input type mismatch")
    scientific_bytes = _canonical_bytes(row["scientific_record"])
    return sequence, {
        "sequence_id": sequence,
        "scientific_record": row["scientific_record"],
        "scientific_record_sha256": hashlib.sha256(scientific_bytes).hexdigest(),
    }


def _validate_embedding_record(record: Mapping[str, Any]) -> tuple[str, str, tuple[int, ...], bytes]:
    required = {
        "graph_identity_sha256",
        "embedding_dtype",
        "embedding_shape",
        "embedding_hex",
    }
    if not required.issubset(record):
        raise T12OrderedIOCanaryError(
            "captured scientific record lacks the bit-exact embedding contract"
        )
    graph_hash = record["graph_identity_sha256"]
    dtype = record["embedding_dtype"]
    shape_raw = record["embedding_shape"]
    encoded = record["embedding_hex"]
    if (
        type(graph_hash) is not str
        or len(graph_hash) != 64
        or any(character not in "0123456789abcdef" for character in graph_hash)
        or dtype not in ("<f4", "<f8")
        or type(shape_raw) is not list
        or not shape_raw
        or any(type(value) is not int or value <= 0 for value in shape_raw)
        or type(encoded) is not str
    ):
        raise T12OrderedIOCanaryError("captured embedding metadata is invalid")
    try:
        raw = bytes.fromhex(encoded)
    except ValueError as exc:
        raise T12OrderedIOCanaryError("captured embedding_hex is invalid") from exc
    element_count = 1
    for dimension in shape_raw:
        element_count *= dimension
    expected_bytes = element_count * (4 if dtype == "<f4" else 8)
    if len(raw) != expected_bytes:
        raise T12OrderedIOCanaryError("captured embedding byte count is inconsistent")
    return graph_hash, dtype, tuple(shape_raw), raw


class AuthoritativeEmbeddingStore:
    """Persist first-seen embedding bytes and reload them without tolerance.

    Graph identity, not recomputed floating-point bytes, selects the authority.
    A later low-bit-different observation is recorded as drift and the exact
    first bytes are returned.  This is intentionally stronger than allclose.
    """

    def __init__(self, *, root: str | Path, resume: bool = False) -> None:
        self.root = Path(root).expanduser().absolute()
        self.path = self.root / "embedding-authority.bin"
        self.manifest_path = self.root / "embedding-checkpoint.json"
        self.entries: dict[str, dict[str, Any]] = {}
        self.observed_drift_count = 0
        self.reload_hit_count = 0
        self._resumed = resume
        if resume:
            self._open_resume()
        else:
            if self.root.exists():
                raise T12OrderedIOCanaryError("embedding authority root must be fresh")
            self.root.mkdir(parents=True, exist_ok=False)
            self._stream = self.path.open("xb", buffering=0)
            self._stream.write(EMBEDDING_MAGIC)
            self._stream.flush()
            os.fsync(self._stream.fileno())
            _fsync_directory(self.root)

    def _open_resume(self) -> None:
        if not self.path.is_file() or not self.manifest_path.is_file():
            raise T12OrderedIOCanaryError("embedding authority resume closure is incomplete")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if (
            type(manifest) is not dict
            or manifest.get("schema_version") != "tastemolnet_t12_embedding_authority_v1"
            or manifest.get("status") != "CHECKPOINT"
            or manifest.get("file_sha256") != _sha256_file(self.path)
            or manifest.get("committed_bytes") != self.path.stat().st_size
            or type(manifest.get("entries")) is not dict
        ):
            raise T12OrderedIOCanaryError("embedding authority manifest/hash mismatch")
        with self.path.open("rb") as stream:
            if stream.read(len(EMBEDDING_MAGIC)) != EMBEDDING_MAGIC:
                raise T12OrderedIOCanaryError("embedding authority magic mismatch")
        self.entries = {str(key): dict(value) for key, value in manifest["entries"].items()}
        for graph_hash in self.entries:
            if (
                len(graph_hash) != 64
                or any(character not in "0123456789abcdef" for character in graph_hash)
            ):
                raise T12OrderedIOCanaryError("embedding authority graph hash is invalid")
            self._load_exact(graph_hash)
        self._stream = self.path.open("ab", buffering=0)

    def _load_exact(self, graph_hash: str) -> tuple[str, tuple[int, ...], bytes]:
        entry = self.entries.get(graph_hash)
        if type(entry) is not dict:
            raise T12OrderedIOCanaryError("embedding authority index lost an identity")
        if set(entry) != {"dtype", "shape", "offset", "byte_count", "sha256"}:
            raise T12OrderedIOCanaryError("embedding authority index schema mismatch")
        offset, byte_count = entry["offset"], entry["byte_count"]
        if type(offset) is not int or type(byte_count) is not int or offset < len(EMBEDDING_MAGIC):
            raise T12OrderedIOCanaryError("embedding authority offset is invalid")
        with self.path.open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(byte_count)
        if len(raw) != byte_count or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise T12OrderedIOCanaryError("authoritative embedding bytes/hash mismatch")
        return str(entry["dtype"]), tuple(entry["shape"]), raw

    def resolve(self, record: Mapping[str, Any]) -> dict[str, Any]:
        graph_hash, dtype, shape, observed = _validate_embedding_record(record)
        if graph_hash not in self.entries:
            offset = self._stream.tell()
            self._stream.write(observed)
            self.entries[graph_hash] = {
                "dtype": dtype,
                "shape": list(shape),
                "offset": offset,
                "byte_count": len(observed),
                "sha256": hashlib.sha256(observed).hexdigest(),
            }
            authoritative = observed
        else:
            stored_dtype, stored_shape, authoritative = self._load_exact(graph_hash)
            if stored_dtype != dtype or stored_shape != shape:
                raise T12OrderedIOCanaryError("re-entered embedding dtype/shape changed")
            if self._resumed:
                self.reload_hit_count += 1
            if authoritative != observed:
                self.observed_drift_count += 1
        resolved = dict(record)
        resolved["embedding_hex"] = authoritative.hex()
        resolved["embedding_sha256"] = hashlib.sha256(authoritative).hexdigest()
        return resolved

    def checkpoint(self) -> dict[str, Any]:
        self._stream.flush()
        os.fsync(self._stream.fileno())
        payload = {
            "schema_version": "tastemolnet_t12_embedding_authority_v1",
            "status": "CHECKPOINT",
            "entries": self.entries,
            "entry_count": len(self.entries),
            "committed_bytes": self.path.stat().st_size,
            "file_sha256": _sha256_file(self.path),
            "bit_exact": True,
            "floating_tolerance_used": False,
        }
        _atomic_json(self.manifest_path, payload)
        return payload

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()

    def __enter__(self) -> "AuthoritativeEmbeddingStore":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def resolve_prepared_embedding(
    record: Mapping[str, Any], *, store: AuthoritativeEmbeddingStore
) -> dict[str, Any]:
    if type(record) is not dict or type(record.get("scientific_record")) is not dict:
        raise T12OrderedIOCanaryError("prepared record has no scientific record")
    resolved_science = store.resolve(record["scientific_record"])
    resolved = dict(record)
    resolved["scientific_record"] = resolved_science
    resolved["scientific_record_sha256"] = hashlib.sha256(
        _canonical_bytes(resolved_science)
    ).hexdigest()
    return resolved


class BufferedHashChainJournal:
    """Append-only ordered journal with batched writes and durable prefixes."""

    def __init__(
        self,
        *,
        root: str | Path,
        contract_sha256: str,
        batch_records: int,
        resume: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().absolute()
        if (
            type(contract_sha256) is not str
            or len(contract_sha256) != 64
            or any(character not in "0123456789abcdef" for character in contract_sha256)
        ):
            raise T12OrderedIOCanaryError("contract_sha256 is invalid")
        if type(batch_records) is not int or batch_records <= 0:
            raise T12OrderedIOCanaryError("batch_records must be positive")
        self.contract_sha256 = contract_sha256
        self.batch_records = batch_records
        self.path = self.root / "ordered-journal.bin"
        self.manifest_path = self.root / "checkpoint.json"
        self.sequence = 0
        self.chain_head = "0" * 64
        self._buffer = bytearray()
        self._buffer_records = 0
        if resume:
            self._open_resume()
        else:
            if self.root.exists():
                raise T12OrderedIOCanaryError("journal root must be fresh")
            self.root.mkdir(parents=True, exist_ok=False)
            self._stream = self.path.open("xb", buffering=0)
            self._stream.write(MAGIC)
            self._stream.flush()
            os.fsync(self._stream.fileno())
            _fsync_directory(self.root)

    def _open_resume(self) -> None:
        if not self.root.is_dir() or not self.path.is_file() or not self.manifest_path.is_file():
            raise T12OrderedIOCanaryError("resume journal closure is incomplete")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if (
            type(manifest) is not dict
            or manifest.get("schema_version") != JOURNAL_SCHEMA
            or manifest.get("status") != "CHECKPOINT"
            or manifest.get("contract_sha256") != self.contract_sha256
        ):
            raise T12OrderedIOCanaryError("resume manifest contract mismatch")
        committed_bytes = manifest.get("committed_bytes")
        if type(committed_bytes) is not int or committed_bytes < len(MAGIC):
            raise T12OrderedIOCanaryError("resume committed byte count is invalid")
        if self.path.stat().st_size != committed_bytes:
            raise T12OrderedIOCanaryError("journal has an unbound tail or truncated prefix")
        if _sha256_file(self.path) != manifest.get("committed_prefix_sha256"):
            raise T12OrderedIOCanaryError("journal committed prefix hash mismatch")
        rows, sequence, chain_head = read_journal(self.path)
        if (
            len(rows) != manifest.get("committed_record_count")
            or sequence != manifest.get("terminal_sequence")
            or chain_head != manifest.get("terminal_chain_head")
        ):
            raise T12OrderedIOCanaryError("journal replay differs from checkpoint")
        self.sequence = sequence
        self.chain_head = chain_head
        self._stream = self.path.open("ab", buffering=0)

    def append(self, sequence: int, record: Mapping[str, Any]) -> None:
        if type(sequence) is not int or sequence != self.sequence + 1:
            raise T12OrderedIOCanaryError("journal sequence is duplicate, missing, or out of order")
        if type(record) is not dict or record.get("sequence_id") != sequence:
            raise T12OrderedIOCanaryError("journal record/sequence mismatch")
        body = _canonical_bytes(dict(record))
        if len(body) > 16 * 1024 * 1024:
            raise T12OrderedIOCanaryError("canary record exceeds the 16 MiB safety cap")
        chain = hashlib.sha256(bytes.fromhex(self.chain_head) + body).digest()
        self._buffer.extend(_LENGTH.pack(len(body)))
        self._buffer.extend(body)
        self._buffer.extend(chain)
        self._buffer_records += 1
        self.sequence = sequence
        self.chain_head = chain.hex()
        if self._buffer_records >= self.batch_records:
            self.flush(durable=False)

    def flush(self, *, durable: bool) -> None:
        if self._buffer:
            self._stream.write(self._buffer)
            self._buffer.clear()
            self._buffer_records = 0
        self._stream.flush()
        if durable:
            os.fsync(self._stream.fileno())

    def checkpoint(self) -> dict[str, Any]:
        self.flush(durable=True)
        _fsync_directory(self.root)
        payload = {
            "schema_version": JOURNAL_SCHEMA,
            "status": "CHECKPOINT",
            "scope": "ORDERED_IO_ONLY",
            "contract_sha256": self.contract_sha256,
            "batch_records": self.batch_records,
            "committed_record_count": self.sequence,
            "terminal_sequence": self.sequence,
            "terminal_chain_head": self.chain_head,
            "committed_bytes": self.path.stat().st_size,
            "committed_prefix_sha256": _sha256_file(self.path),
            "scientific_parity_claimed": False,
            "replacement_authorized": False,
        }
        _atomic_json(self.manifest_path, payload)
        return payload

    def close(self) -> None:
        if not self._stream.closed:
            self.flush(durable=True)
            self._stream.close()

    def __enter__(self) -> "BufferedHashChainJournal":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def read_journal(path: str | Path) -> tuple[list[dict[str, Any]], int, str]:
    source = Path(path).expanduser().resolve(strict=True)
    rows: list[dict[str, Any]] = []
    sequence = 0
    chain_head = "0" * 64
    with source.open("rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise T12OrderedIOCanaryError("journal magic mismatch")
        while True:
            length_bytes = stream.read(_LENGTH.size)
            if not length_bytes:
                break
            if len(length_bytes) != _LENGTH.size:
                raise T12OrderedIOCanaryError("journal length prefix is truncated")
            body_length = _LENGTH.unpack(length_bytes)[0]
            if body_length <= 0 or body_length > 16 * 1024 * 1024:
                raise T12OrderedIOCanaryError("journal body length is invalid")
            body = stream.read(body_length)
            chain = stream.read(32)
            if len(body) != body_length or len(chain) != 32:
                raise T12OrderedIOCanaryError("journal record is truncated")
            expected_chain = hashlib.sha256(bytes.fromhex(chain_head) + body).digest()
            if chain != expected_chain:
                raise T12OrderedIOCanaryError("journal hash chain mismatch")
            row = json.loads(body.decode("utf-8"))
            if type(row) is not dict or row.get("sequence_id") != sequence + 1:
                raise T12OrderedIOCanaryError("journal semantic sequence mismatch")
            rows.append(row)
            sequence += 1
            chain_head = chain.hex()
    return rows, sequence, chain_head


class OrderedCollector:
    """Commit arbitrary completion order in one gap-free sequence."""

    def __init__(
        self,
        journal: BufferedHashChainJournal,
        *,
        embedding_store: AuthoritativeEmbeddingStore | None = None,
    ) -> None:
        self.journal = journal
        self.embedding_store = embedding_store
        self.next_sequence = journal.sequence + 1
        self.pending: dict[int, dict[str, Any]] = {}

    def accept(self, sequence: int, record: Mapping[str, Any]) -> None:
        if type(sequence) is not int or sequence < self.next_sequence or sequence in self.pending:
            raise T12OrderedIOCanaryError("executor returned a duplicate/stale sequence")
        self.pending[sequence] = dict(record)
        while self.next_sequence in self.pending:
            current = self.pending.pop(self.next_sequence)
            if self.embedding_store is not None:
                current = resolve_prepared_embedding(
                    current, store=self.embedding_store
                )
            self.journal.append(self.next_sequence, current)
            self.next_sequence += 1

    def finish(self) -> None:
        if self.pending:
            raise T12OrderedIOCanaryError("executor completion set contains a sequence gap")


def execute_ordered(
    rows: Sequence[Mapping[str, Any]],
    *,
    journal: BufferedHashChainJournal,
    workers: int,
    executor_kind: str,
    embedding_store: AuthoritativeEmbeddingStore | None = None,
) -> None:
    if type(workers) is not int or workers <= 0:
        raise T12OrderedIOCanaryError("workers must be positive")
    executor_class: type[Executor]
    kwargs: dict[str, Any] = {"max_workers": workers}
    if executor_kind == "thread":
        executor_class = ThreadPoolExecutor
    elif executor_kind == "process":
        executor_class = ProcessPoolExecutor
        kwargs["mp_context"] = multiprocessing.get_context("spawn")
    else:
        raise T12OrderedIOCanaryError("executor_kind must be thread or process")
    collector = OrderedCollector(journal, embedding_store=embedding_store)
    with executor_class(**kwargs) as executor:
        futures = [executor.submit(prepare_record, dict(row)) for row in rows]
        for future in as_completed(futures):
            sequence, record = future.result()
            collector.accept(sequence, record)
    collector.finish()


def run_buffered_ordered_io_canary(
    *,
    input_jsonl: str | Path,
    output_root: str | Path,
    checkpoint_at: int = 500,
    post_reload_records: int = 10,
    buffered_batch_records: int = 256,
    workers: int = 4,
    executor_kind: str = "process",
) -> dict[str, Any]:
    records, input_sha256 = load_captured_records(input_jsonl)
    if (
        type(checkpoint_at) is not int
        or checkpoint_at <= 0
        or type(post_reload_records) is not int
        or post_reload_records <= 0
    ):
        raise T12OrderedIOCanaryError("checkpoint and reload record counts must be positive")
    required = checkpoint_at + post_reload_records
    if len(records) < required:
        raise T12OrderedIOCanaryError(
            f"captured input requires at least {required} consecutive rows"
        )
    selected = records[:required]
    contract_sha256 = hashlib.sha256(
        _canonical_bytes(
            {
                "input_sha256": input_sha256,
                "selected_count": required,
                "checkpoint_at": checkpoint_at,
                "transform": "identity_plus_canonical_scientific_record_sha256_v1",
            }
        )
    ).hexdigest()
    output = Path(output_root).expanduser().absolute()
    if output.exists():
        raise T12OrderedIOCanaryError("canary output root must be fresh")
    output.mkdir(parents=True, exist_ok=False)

    reference_started = time.monotonic()
    with BufferedHashChainJournal(
        root=output / "reference",
        contract_sha256=contract_sha256,
        batch_records=1,
    ) as reference:
        with AuthoritativeEmbeddingStore(
            root=output / "reference" / "embedding-authority"
        ) as reference_embeddings:
            for row in selected:
                sequence, prepared = prepare_record(row)
                reference.append(
                    sequence,
                    resolve_prepared_embedding(
                        prepared, store=reference_embeddings
                    ),
                )
                if sequence == checkpoint_at:
                    reference.checkpoint()
                    reference_embeddings.checkpoint()
            reference_manifest = reference.checkpoint()
            reference_embedding_manifest = reference_embeddings.checkpoint()
    reference_seconds = time.monotonic() - reference_started

    accelerated_root = output / "accelerated"
    accelerated_started = time.monotonic()
    with BufferedHashChainJournal(
        root=accelerated_root,
        contract_sha256=contract_sha256,
        batch_records=buffered_batch_records,
    ) as accelerated:
        with AuthoritativeEmbeddingStore(
            root=accelerated_root / "embedding-authority"
        ) as accelerated_embeddings:
            execute_ordered(
                selected[:checkpoint_at],
                journal=accelerated,
                workers=workers,
                executor_kind=executor_kind,
                embedding_store=accelerated_embeddings,
            )
            checkpoint_manifest = accelerated.checkpoint()
            accelerated_embeddings.checkpoint()
    with BufferedHashChainJournal(
        root=accelerated_root,
        contract_sha256=contract_sha256,
        batch_records=buffered_batch_records,
        resume=True,
    ) as resumed:
        with AuthoritativeEmbeddingStore(
            root=accelerated_root / "embedding-authority", resume=True
        ) as resumed_embeddings:
            execute_ordered(
                selected[checkpoint_at:],
                journal=resumed,
                workers=workers,
                executor_kind=executor_kind,
                embedding_store=resumed_embeddings,
            )
            accelerated_manifest = resumed.checkpoint()
            accelerated_embedding_manifest = resumed_embeddings.checkpoint()
            embedding_reload_hits = resumed_embeddings.reload_hit_count
            embedding_observed_drift_count = resumed_embeddings.observed_drift_count
    accelerated_seconds = time.monotonic() - accelerated_started

    reference_rows, _, reference_head = read_journal(output / "reference" / "ordered-journal.bin")
    accelerated_rows, _, accelerated_head = read_journal(
        accelerated_root / "ordered-journal.bin"
    )
    equivalent = (
        reference_rows == accelerated_rows
        and reference_head == accelerated_head
        and reference_manifest["committed_prefix_sha256"]
        == accelerated_manifest["committed_prefix_sha256"]
        and reference_embedding_manifest["file_sha256"]
        == accelerated_embedding_manifest["file_sha256"]
    )
    report = {
        "schema_version": CANARY_SCHEMA,
        "scope": "BUFFERING_AND_ORDERED_COLLECTION_ONLY",
        "status": "PASS" if equivalent else "FAILED",
        "input_jsonl": str(Path(input_jsonl).expanduser().resolve(strict=True)),
        "input_sha256": input_sha256,
        "input_record_count": len(records),
        "selected_record_count": required,
        "checkpoint_at": checkpoint_at,
        "post_reload_records": post_reload_records,
        "checkpoint_terminal_sequence": checkpoint_manifest["terminal_sequence"],
        "checkpoint_reload_pass": accelerated_manifest["terminal_sequence"] == required,
        "ordered_rows_equal": reference_rows == accelerated_rows,
        "chain_heads_equal": reference_head == accelerated_head,
        "journal_bytes_equal": reference_manifest["committed_prefix_sha256"]
        == accelerated_manifest["committed_prefix_sha256"],
        "embedding_authority_enabled": True,
        "embedding_authority_bit_exact": True,
        "embedding_floating_tolerance_used": False,
        "embedding_authority_equal": (
            reference_embedding_manifest["file_sha256"]
            == accelerated_embedding_manifest["file_sha256"]
        ),
        "embedding_reload_hits": embedding_reload_hits,
        "embedding_observed_drift_count_after_reload": (
            embedding_observed_drift_count
        ),
        "reference_seconds": reference_seconds,
        "accelerated_seconds": accelerated_seconds,
        "observed_io_canary_speedup": (
            reference_seconds / accelerated_seconds if accelerated_seconds else None
        ),
        "buffered_batch_records": buffered_batch_records,
        "workers": workers,
        "executor_kind": executor_kind,
        "scientific_parity_claimed": False,
        "vrrw_parity_claimed": False,
        "model_output_parity_claimed": False,
        "replacement_authorized": False,
    }
    _atomic_json(output / "canary_report.json", report)
    if not equivalent:
        raise T12OrderedIOCanaryError("buffered ordered journal diverged from reference")
    return report


__all__ = [
    "BufferedHashChainJournal",
    "AuthoritativeEmbeddingStore",
    "CANARY_SCHEMA",
    "OrderedCollector",
    "T12OrderedIOCanaryError",
    "execute_ordered",
    "load_captured_records",
    "prepare_record",
    "read_journal",
    "resolve_prepared_embedding",
    "run_buffered_ordered_io_canary",
]
