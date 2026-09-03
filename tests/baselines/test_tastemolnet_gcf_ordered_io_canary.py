from __future__ import annotations

import hashlib
import json
import struct

import pytest

from src.baselines.tastemolnet_gcf_ordered_io_canary import (
    AuthoritativeEmbeddingStore,
    BufferedHashChainJournal,
    OrderedCollector,
    T12OrderedIOCanaryError,
    prepare_record,
    run_buffered_ordered_io_canary,
)


def _write_input(path, count: int) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for sequence in range(1, count + 1):
            identity_sequence = 1 if sequence == 11 else sequence
            embedding = (
                struct.pack("<f", 1.0000001192092896)
                if sequence == 11
                else struct.pack("<f", float(identity_sequence))
            )
            stream.write(
                json.dumps(
                    {
                        "sequence_id": sequence,
                        "scientific_record": {
                            "graph_identity_sha256": f"{identity_sequence:064x}",
                            "embedding_dtype": "<f4",
                            "embedding_shape": [1],
                            "embedding_hex": embedding.hex(),
                            "prediction": sequence % 3,
                            "accepted": sequence % 2 == 0,
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def test_buffered_ordered_checkpoint_reload_matches_unbuffered_reference(tmp_path) -> None:
    source = tmp_path / "captured.jsonl"
    _write_input(source, 12)
    report = run_buffered_ordered_io_canary(
        input_jsonl=source,
        output_root=tmp_path / "result",
        checkpoint_at=10,
        post_reload_records=2,
        buffered_batch_records=4,
        workers=3,
        executor_kind="thread",
    )
    assert report["status"] == "PASS"
    assert report["checkpoint_reload_pass"] is True
    assert report["ordered_rows_equal"] is True
    assert report["journal_bytes_equal"] is True
    assert report["embedding_authority_bit_exact"] is True
    assert report["embedding_floating_tolerance_used"] is False
    assert report["embedding_reload_hits"] == 1
    assert report["embedding_observed_drift_count_after_reload"] == 1
    assert report["scientific_parity_claimed"] is False
    assert report["replacement_authorized"] is False


def test_process_executor_commits_one_ordered_prefix(tmp_path) -> None:
    source = tmp_path / "captured-process.jsonl"
    _write_input(source, 4)
    try:
        report = run_buffered_ordered_io_canary(
            input_jsonl=source,
            output_root=tmp_path / "process-result",
            checkpoint_at=3,
            post_reload_records=1,
            buffered_batch_records=3,
            workers=2,
            executor_kind="process",
        )
    except (NotImplementedError, PermissionError) as exc:
        pytest.skip(f"host sandbox does not expose process semaphores: {exc}")
    assert report["status"] == "PASS"
    assert report["executor_kind"] == "process"


def test_ordered_collector_buffers_completion_order_and_rejects_gaps(tmp_path) -> None:
    contract = "a" * 64
    with BufferedHashChainJournal(
        root=tmp_path / "ordered", contract_sha256=contract, batch_records=8
    ) as journal:
        collector = OrderedCollector(journal)
        sequence_two, row_two = prepare_record(
            {"sequence_id": 2, "scientific_record": {"value": "two"}}
        )
        sequence_one, row_one = prepare_record(
            {"sequence_id": 1, "scientific_record": {"value": "one"}}
        )
        collector.accept(sequence_two, row_two)
        assert journal.sequence == 0
        collector.accept(sequence_one, row_one)
        assert journal.sequence == 2
        collector.finish()

    with BufferedHashChainJournal(
        root=tmp_path / "gap", contract_sha256=contract, batch_records=8
    ) as journal:
        collector = OrderedCollector(journal)
        collector.accept(sequence_two, row_two)
        with pytest.raises(T12OrderedIOCanaryError, match="gap"):
            collector.finish()


def test_resume_rejects_an_unbound_tail(tmp_path) -> None:
    root = tmp_path / "journal"
    contract = "b" * 64
    with BufferedHashChainJournal(
        root=root, contract_sha256=contract, batch_records=2
    ) as journal:
        sequence, row = prepare_record(
            {"sequence_id": 1, "scientific_record": {"value": 1}}
        )
        journal.append(sequence, row)
        journal.checkpoint()
    with (root / "ordered-journal.bin").open("ab") as stream:
        stream.write(b"unbound")
    with pytest.raises(T12OrderedIOCanaryError, match="unbound tail"):
        BufferedHashChainJournal(
            root=root,
            contract_sha256=contract,
            batch_records=2,
            resume=True,
        )


def test_evicted_embedding_bit_exact(tmp_path) -> None:
    root = tmp_path / "embedding-authority"
    graph_hash = "c" * 64
    first = struct.pack("<f", 1.0)
    drifted = struct.pack("<f", 1.0000001192092896)
    base_record = {
        "graph_identity_sha256": graph_hash,
        "embedding_dtype": "<f4",
        "embedding_shape": [1],
        "embedding_hex": first.hex(),
    }
    with AuthoritativeEmbeddingStore(root=root) as store:
        assert store.resolve(base_record)["embedding_hex"] == first.hex()
        store.checkpoint()

    with AuthoritativeEmbeddingStore(root=root, resume=True) as store:
        reentered = store.resolve({**base_record, "embedding_hex": drifted.hex()})
        manifest = store.checkpoint()
        assert reentered["embedding_hex"] == first.hex()
        assert reentered["embedding_sha256"] != hashlib.sha256(drifted).hexdigest()
        assert store.reload_hit_count == 1
        assert store.observed_drift_count == 1
        assert manifest["bit_exact"] is True
        assert manifest["floating_tolerance_used"] is False
