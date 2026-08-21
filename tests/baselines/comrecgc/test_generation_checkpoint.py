from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.baselines.comrecgc.generation_checkpoint import (
    CHECKPOINT_BOUNDARY,
    LATEST_FILENAME,
    GenerationCheckpointError,
    list_generation_checkpoints,
    load_generation_checkpoint,
    mirror_generation_checkpoint,
    prune_mirrored_generation_checkpoints,
    restore_generation_checkpoint,
    save_generation_checkpoint,
    scientific_command_sha256,
    validate_generation_checkpoint,
)


torch = pytest.importorskip("torch")

PROVENANCE = {
    "config_sha256": "c" * 64,
    "dataset_fingerprint": "d" * 64,
    "external_commit": "122f9341a360e9f06bb58a2f5823bb596021f6bf",
    "project_commit": "a7c480a3c8499f6803e762c0fc683a03b5b8fb4a",
}
SCIENTIFIC_ARGV = (
    "scripts/baselines/comrecgc/run_generation.py",
    '--dataset="bace"',
    "--mode=\"full\"",
    "--parent-limit=500",
)
COMMAND_SHA256 = scientific_command_sha256(SCIENTIFIC_ARGV)
TOTAL_STEPS = 50_000
PROVENANCE.update(
    {
        "scientific_command_sha256": COMMAND_SHA256,
        "total_steps": str(TOTAL_STEPS),
    }
)


def _database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("CREATE TABLE graph_state (key TEXT PRIMARY KEY, value TEXT)")
    connection.execute("INSERT INTO graph_state VALUES ('head', 'source')")
    connection.commit()
    return connection


def _save(
    root: Path,
    connection: sqlite3.Connection,
    *,
    step: int = 17,
):
    return save_generation_checkpoint(
        root,
        completed_step=step,
        step_complete=True,
        algorithm_state={
            "current_graph_hashes": ["head"],
            "transition_cache": {"head": ["target"]},
            "tensor": torch.tensor([1, 2, 3]),
        },
        trace_state={"move_index": step, "selected_transition_count": step},
        sqlite_source=connection,
        provenance_fingerprints=PROVENANCE,
        scientific_argv=SCIENTIFIC_ARGV,
        command_sha256=COMMAND_SHA256,
        total_steps=TOTAL_STEPS,
    )


def _rows(path: Path) -> list[tuple[str, str]]:
    connection = sqlite3.connect(path)
    try:
        return list(connection.execute("SELECT key, value FROM graph_state ORDER BY key"))
    finally:
        connection.close()


def test_checkpoint_round_trip_uses_consistent_live_sqlite_backup(tmp_path) -> None:
    connection = _database(tmp_path / "live.sqlite3")
    root = tmp_path / "checkpoints"
    validation = _save(root, connection)

    connection.execute("INSERT INTO graph_state VALUES ('later', 'not-in-checkpoint')")
    connection.commit()
    connection.close()
    loaded = load_generation_checkpoint(
        root,
        expected_provenance=PROVENANCE,
        expected_completed_step=17,
    )

    assert validation.completed_step == 17
    assert loaded.completed_step == 17
    assert validation.manifest["next_step"] == 18
    assert loaded.algorithm_state["current_graph_hashes"] == ["head"]
    assert torch.equal(loaded.algorithm_state["tensor"], torch.tensor([1, 2, 3]))
    assert loaded.trace_state == {"move_index": 17, "selected_transition_count": 17}
    assert _rows(loaded.sqlite_snapshot_path) == [("head", "source")]
    assert validation.manifest["boundary"] == CHECKPOINT_BOUNDARY
    assert validation.manifest["scientific_argv"] == list(SCIENTIFIC_ARGV)
    assert validation.manifest["command_sha256"] == COMMAND_SHA256
    assert validation.manifest["total_steps"] == TOTAL_STEPS
    assert validation.provenance_fingerprints["scientific_command_sha256"] == (
        COMMAND_SHA256
    )
    assert validation.provenance_fingerprints["total_steps"] == str(TOTAL_STEPS)
    assert validation.manifest["sqlite_snapshot_method"] == (
        "sqlite_connection_backup_api_v1"
    )
    assert validation.manifest["sqlite_snapshot"]["integrity_check"] == "ok"
    assert not Path(f"{loaded.sqlite_snapshot_path}-wal").exists()
    assert not Path(f"{loaded.sqlite_snapshot_path}-shm").exists()
    assert not list(root.glob(".*.tmp"))


def test_restore_recovers_sqlite_and_python_numpy_torch_rng(tmp_path) -> None:
    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    connection = _database(tmp_path / "live.sqlite3")
    root = tmp_path / "checkpoints"
    _save(root, connection)
    connection.close()

    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )
    for _ in range(5):
        random.random()
        np.random.random()
        torch.rand(3)

    destination = tmp_path / "restored" / "graph_state.sqlite3"
    loaded = restore_generation_checkpoint(
        root,
        destination_sqlite_path=destination,
        expected_provenance=PROVENANCE,
        expected_completed_step=17,
        restore_rng=True,
    )
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )

    assert loaded.completed_step == 17
    assert _rows(destination) == [("head", "source")]
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


@pytest.mark.parametrize(
    ("completed_step", "step_complete"),
    [(0, True), (1, False)],
)
def test_save_rejects_non_completed_step_boundary(
    tmp_path, completed_step: int, step_complete: bool
) -> None:
    connection = _database(tmp_path / "live.sqlite3")
    root = tmp_path / "checkpoints"

    with pytest.raises(GenerationCheckpointError, match="fully completed step"):
        save_generation_checkpoint(
            root,
            completed_step=completed_step,
            step_complete=step_complete,
            algorithm_state={"head": "a"},
            trace_state={"move_index": completed_step},
            sqlite_source=connection,
            provenance_fingerprints=PROVENANCE,
            scientific_argv=SCIENTIFIC_ARGV,
            command_sha256=COMMAND_SHA256,
            total_steps=TOTAL_STEPS,
        )

    connection.close()
    assert not root.exists()


def test_open_sqlite_transaction_fails_closed_without_publication(tmp_path) -> None:
    connection = _database(tmp_path / "live.sqlite3")
    connection.execute("INSERT INTO graph_state VALUES ('uncommitted', 'value')")
    root = tmp_path / "checkpoints"

    with pytest.raises(GenerationCheckpointError, match="open transaction"):
        _save(root, connection)

    connection.rollback()
    connection.close()
    assert not (root / LATEST_FILENAME).exists()
    assert not list(root.glob("step-*"))


def test_checksum_tamper_is_rejected_before_load(tmp_path) -> None:
    connection = _database(tmp_path / "live.sqlite3")
    root = tmp_path / "checkpoints"
    validation = _save(root, connection)
    connection.close()
    state_path = validation.checkpoint_dir / "generation_state.pt"
    state_path.write_bytes(state_path.read_bytes() + b"tamper")

    with pytest.raises(GenerationCheckpointError, match="size mismatch"):
        validate_generation_checkpoint(root)


def test_provenance_mismatch_fails_closed(tmp_path) -> None:
    connection = _database(tmp_path / "live.sqlite3")
    root = tmp_path / "checkpoints"
    _save(root, connection)
    connection.close()
    wrong = {**PROVENANCE, "project_commit": "f" * 40}

    with pytest.raises(GenerationCheckpointError, match="provenance differs"):
        load_generation_checkpoint(root, expected_provenance=wrong)


def test_scientific_cli_parameter_drift_fails_closed(tmp_path) -> None:
    connection = _database(tmp_path / "live.sqlite3")
    root = tmp_path / "checkpoints"
    _save(root, connection)
    connection.close()
    changed_argv = (*SCIENTIFIC_ARGV[:-1], "--parent-limit=499")

    with pytest.raises(GenerationCheckpointError, match="scientific argv differs"):
        load_generation_checkpoint(
            root,
            expected_provenance=PROVENANCE,
            expected_scientific_argv=changed_argv,
            expected_command_sha256=scientific_command_sha256(changed_argv),
            expected_total_steps=TOTAL_STEPS,
        )


def test_total_steps_drift_fails_closed(tmp_path) -> None:
    connection = _database(tmp_path / "live.sqlite3")
    root = tmp_path / "checkpoints"
    _save(root, connection)
    connection.close()

    with pytest.raises(GenerationCheckpointError, match="total_steps differs"):
        load_generation_checkpoint(
            root,
            expected_provenance=PROVENANCE,
            expected_scientific_argv=SCIENTIFIC_ARGV,
            expected_command_sha256=COMMAND_SHA256,
            expected_total_steps=TOTAL_STEPS + 1,
        )


def test_incomplete_tmp_directory_is_ignored(tmp_path) -> None:
    connection = _database(tmp_path / "live.sqlite3")
    root = tmp_path / "checkpoints"
    validation = _save(root, connection)
    connection.close()
    incomplete = root / ".step-000000000018-crashed.tmp"
    incomplete.mkdir()
    (incomplete / "generation_state.pt").write_bytes(b"partial")

    listed = list_generation_checkpoints(root)
    loaded = load_generation_checkpoint(root, expected_provenance=PROVENANCE)

    assert listed == [validation.checkpoint_dir]
    assert loaded.completed_step == 17


def test_unsafe_latest_is_ignored_and_repaired_from_valid_physical_checkpoint(
    tmp_path,
) -> None:
    connection = _database(tmp_path / "live.sqlite3")
    root = tmp_path / "checkpoints"
    _save(root, connection)
    connection.close()
    (root / LATEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": "comrecgc_generation_checkpoint_latest_v1",
                "checkpoint_dir": "../outside",
                "completed_step": 17,
                "checkpoint_digest": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    validation = validate_generation_checkpoint(root)
    repaired = json.loads((root / LATEST_FILENAME).read_text(encoding="utf-8"))

    assert validation.completed_step == 17
    assert repaired["checkpoint_dir"] == "step-000000000017"
    audit = json.loads(
        (root / "checkpoint_recovery_audit.json").read_text(encoding="utf-8")
    )
    assert audit["latest_repaired"] is True


def test_latest_advances_only_to_fully_valid_published_checkpoint(tmp_path) -> None:
    connection = _database(tmp_path / "live.sqlite3")
    root = tmp_path / "checkpoints"
    first = _save(root, connection, step=17)
    second = _save(root, connection, step=18)
    connection.close()

    loaded = load_generation_checkpoint(root, expected_provenance=PROVENANCE)

    assert loaded.completed_step == 18
    assert list_generation_checkpoints(root) == [
        first.checkpoint_dir,
        second.checkpoint_dir,
    ]


def test_complete_newer_directory_repairs_stale_latest(tmp_path) -> None:
    connection = _database(tmp_path / "live.sqlite3")
    root = tmp_path / "checkpoints"
    first = _save(root, connection, step=17)
    second = _save(root, connection, step=18)
    connection.close()
    (root / LATEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": "comrecgc_generation_checkpoint_latest_v1",
                "checkpoint_dir": first.checkpoint_dir.name,
                "completed_step": first.completed_step,
                "checkpoint_digest": first.checkpoint_digest,
            }
        ),
        encoding="utf-8",
    )

    loaded = load_generation_checkpoint(root, expected_provenance=PROVENANCE)
    latest = json.loads((root / LATEST_FILENAME).read_text(encoding="utf-8"))

    assert loaded.completed_step == second.completed_step
    assert latest["checkpoint_dir"] == second.checkpoint_dir.name
    assert latest["checkpoint_digest"] == second.checkpoint_digest


def test_mirror_then_retention_keeps_latest_two_and_small_audit_history(tmp_path) -> None:
    connection = _database(tmp_path / "live.sqlite3")
    root = tmp_path / "fast" / "checkpoints"
    mirror = tmp_path / "persistent" / "checkpoints"
    for step in (1, 2, 3):
        validation = _save(root, connection, step=step)
        mirrored = mirror_generation_checkpoint(
            validation.checkpoint_dir, mirror, expected_provenance=PROVENANCE
        )
        assert mirrored.checkpoint_digest == validation.checkpoint_digest
    connection.close()

    removed = prune_mirrored_generation_checkpoints(
        root, mirror, keep_last=2, expected_provenance=PROVENANCE
    )

    assert [row["completed_step"] for row in removed] == [1]
    assert [path.name for path in list_generation_checkpoints(root)] == [
        "step-000000000002",
        "step-000000000003",
    ]
    assert [path.name for path in list_generation_checkpoints(mirror)] == [
        "step-000000000002",
        "step-000000000003",
    ]
    assert (root / "retention_history/step-000000000001.json").is_file()
    assert (mirror / "retention_history/step-000000000001.json").is_file()
    for step in (2, 3):
        marker = json.loads(
            (
                root
                / f"step-{step:012d}"
                / "_CHECKPOINT_MIRRORED.json"
            ).read_text(encoding="utf-8")
        )
        assert marker["checkpoint_mirrored"] is True


def test_retention_refuses_corrupt_mirror_without_deleting_local(tmp_path) -> None:
    connection = _database(tmp_path / "live.sqlite3")
    root = tmp_path / "fast" / "checkpoints"
    mirror = tmp_path / "persistent" / "checkpoints"
    for step in (1, 2, 3):
        validation = _save(root, connection, step=step)
        mirror_generation_checkpoint(validation.checkpoint_dir, mirror)
    connection.close()
    corrupt = mirror / "step-000000000001/generation_state.pt"
    corrupt.write_bytes(corrupt.read_bytes() + b"corrupt")

    with pytest.raises(GenerationCheckpointError):
        prune_mirrored_generation_checkpoints(root, mirror, keep_last=2)

    assert sorted(path.name for path in root.glob("step-*") if path.is_dir()) == [
        "step-000000000001",
        "step-000000000002",
        "step-000000000003",
    ]


def test_restore_refuses_destination_with_live_sqlite_sidecar(tmp_path) -> None:
    connection = _database(tmp_path / "live.sqlite3")
    root = tmp_path / "checkpoints"
    _save(root, connection)
    connection.close()
    destination = tmp_path / "restored.sqlite3"
    destination.write_bytes(b"old")
    Path(f"{destination}-wal").write_bytes(b"live")

    with pytest.raises(GenerationCheckpointError, match="live WAL/SHM"):
        restore_generation_checkpoint(
            root,
            destination_sqlite_path=destination,
            expected_provenance=PROVENANCE,
            restore_rng=False,
        )

    assert destination.read_bytes() == b"old"
