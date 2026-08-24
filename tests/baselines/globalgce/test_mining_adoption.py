from __future__ import annotations

import json
from pathlib import Path
import pickle
import sqlite3

import networkx as nx
import pytest

from src.baselines import globalgce_mining_adoption as adoption
from src.baselines import globalgce_resumable


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    source_root = tmp_path / "v5" / "train_candidates" / "attempt-0"
    support = (
        source_root
        / "native"
        / "globalgce_training_checkpoints"
        / "gspan"
        / "support_7_fixture"
    )
    support.mkdir(parents=True)
    database = support / "frequent_patterns.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE roots(root_index INTEGER PRIMARY KEY, root_label TEXT, "
            "complete INTEGER, pattern_count INTEGER)"
        )
        connection.execute(
            "CREATE TABLE patterns(root_index INTEGER, local_index INTEGER, "
            "support INTEGER, payload BLOB, PRIMARY KEY(root_index, local_index))"
        )
        connection.execute(
            "INSERT INTO metadata VALUES('input_fingerprint','fingerprint-v2')"
        )
        rows = []
        for index, support_value in enumerate((11, 10, 9)):
            graph = nx.Graph()
            graph.add_node(0, label=index + 1)
            graph.add_node(1, label=index + 2)
            graph.add_edge(0, 1, label=1)
            rows.append((0, index, support_value, pickle.dumps(graph)))
        connection.executemany("INSERT INTO patterns VALUES(?,?,?,?)", rows)
        connection.execute("INSERT INTO roots VALUES(0,'root',1,3)")
    checkpoint = support / "checkpoint.json"
    _json(
        checkpoint,
        {
            "schema_version": adoption.CHECKPOINT_SCHEMA_VERSION,
            "stage": "complete",
            "exact_top_k_pruning": False,
            "root_count": 1,
            "completed_root_count": 1,
            "frequent_subgraph_count": 3,
            "input_fingerprint": "fingerprint-v2",
            "sqlite_path": str(database),
        },
    )
    run_manifest = source_root / "run_manifest.json"
    _json(
        run_manifest,
        {
            "status": "RUNNING",
            "run_complete": False,
            "oracle_backend": "gnn",
            "classifier_family": "gine",
            "rf_oracle_used": False,
            "calibration_loaded": False,
            "test_loaded": False,
            "config": {
                "seed": 13,
                "top_k_native": 2,
                "min_freq": 7,
                "gspan_exact_top_k_pruning": False,
            },
        },
    )
    task_state = tmp_path / "control" / "state.json"
    _json(
        task_state,
        {
            "task_id": "bace_globalgce_train_candidates",
            "status": "FAILED",
            "output_root": str(source_root),
            "attempt": {"state": "FAILED", "exit_code": 1},
        },
    )
    official = tmp_path / "official"
    official.mkdir()
    native_csv = tmp_path / "BACE" / "train.csv"
    native_csv.parent.mkdir()
    native_csv.write_text("molecule_id,smiles,label\n", encoding="utf-8")
    source_manifest = tmp_path / "source" / "source_manifest.jsonl"
    source_manifest.parent.mkdir()
    source_manifest.write_text("{}\n", encoding="utf-8")
    gine = tmp_path / "gine"
    gine.mkdir()
    (gine / "model.pt").write_bytes(b"frozen-gine")
    monkeypatch.setattr(
        adoption,
        "_official_commit",
        lambda _root: adoption.EXPECTED_OFFICIAL_COMMIT,
    )
    monkeypatch.setattr(
        adoption,
        "_recompute_bace_traversal_fingerprint",
        lambda **_kwargs: {
            "fingerprint_schema": "globalgce_native_traversal_order_v2",
            "input_fingerprint": "fingerprint-v2",
            "graph_count": 2,
            "root_count": 1,
            "settings": {"min_support": 7, "top_k": 2},
            "source_train_index_sha256": "train-order",
        },
    )
    return {
        "source_run_manifest": run_manifest,
        "source_task_state": task_state,
        "source_checkpoint": checkpoint,
        "source_sqlite": database,
        "official_root": official,
        "native_train_csv": native_csv,
        "source_manifest": source_manifest,
        "gine_checkpoint": gine,
    }


def _build(
    fixture: dict[str, Path], tmp_path: Path
) -> tuple[Path, dict]:
    output = tmp_path / "adoption"
    identity = adoption.build_globalgce_gspan_adoption(
        **fixture,
        output_dir=output,
        expected_pattern_count=3,
        expected_root_count=1,
        min_freq=7,
        top_k=2,
        seed=13,
    )
    return output / "adoption_proof.json", identity


def test_exhaustive_v2_adoption_recomputes_stable_topk_and_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    proof, identity = _build(fixture, tmp_path)
    assert identity["status"] == "PASS"
    assert identity["pattern_count"] == 3
    assert identity["root_count"] == 1
    graphs, supports = adoption.load_adopted_globalgce_top_k(
        proof, validated_identity=identity
    )
    assert supports == [11, 10]
    assert [graph.nodes[0]["label"] for graph in graphs] == [1, 2]
    assert (proof.parent / "ADOPTION_PASS").is_file()
    assert (proof.parent / "PASS").is_file()


def test_adoption_rejects_nonempty_wal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    Path(str(fixture["source_sqlite"]) + "-wal").write_bytes(b"uncheckpointed")
    with pytest.raises(adoption.GlobalGCEMiningAdoptionError, match="WAL"):
        _build(fixture, tmp_path)


def test_adoption_rejects_parent_task_not_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    state = json.loads(fixture["source_task_state"].read_text())
    state["status"] = "RUNNING"
    state["attempt"]["state"] = "RUNNING"
    _json(fixture["source_task_state"], state)
    with pytest.raises(adoption.GlobalGCEMiningAdoptionError, match="not FAILED"):
        _build(fixture, tmp_path)


def test_adoption_validation_rejects_source_database_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    proof, _identity = _build(fixture, tmp_path)
    with sqlite3.connect(fixture["source_sqlite"]) as connection:
        connection.execute("UPDATE patterns SET support=12 WHERE local_index=0")
    with pytest.raises(adoption.GlobalGCEMiningAdoptionError, match="SQLite"):
        adoption.validate_globalgce_gspan_adoption_proof(proof)


def test_adoption_validation_rejects_selected_payload_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    proof, _identity = _build(fixture, tmp_path)
    (proof.parent / "selected_top20.pkl").write_bytes(b"tampered")
    with pytest.raises(adoption.GlobalGCEMiningAdoptionError, match="selected-top20"):
        adoption.validate_globalgce_gspan_adoption_proof(proof)


def test_resumable_consumer_uses_adopted_topk_once_and_restores_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graphs = [nx.path_graph(2), nx.path_graph(3)]
    identity = {"top_k": 2, "min_freq": 7, "status": "PASS"}
    monkeypatch.setattr(
        adoption,
        "validate_globalgce_gspan_adoption_proof",
        lambda _path: identity,
    )
    monkeypatch.setattr(
        adoption,
        "load_adopted_globalgce_top_k",
        lambda _path, validated_identity=None: (graphs, [9, 8]),
    )

    class FakeFSG:
        topk = 2

        def find_fs(self, _min_freq: int):
            raise AssertionError("legacy mining must not run")

    class FakeModel:
        fsg = FakeFSG()

        def get_fs_expanded_data(self, _loader):
            selected = self.fsg.find_fs(7)
            return selected, "train", "val", "test"

    model = FakeModel()
    original = model.fsg.find_fs
    expanded, observed = globalgce_resumable._get_fs_expanded_data_from_adoption(
        model=model,
        train_loader=object(),
        proof_path=tmp_path / "proof.json",
    )
    assert expanded == (graphs, "train", "val", "test")
    assert observed == identity
    assert model.fsg.find_fs.__func__ is original.__func__
