from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from src.baselines.comrecgc.checkpoint_audit import audit_generation_checkpoint
from src.baselines.comrecgc.live_graph_state import AuthoritativeGraphStore


def graph() -> SimpleNamespace:
    return SimpleNamespace(
        x=np.asarray([[1], [2]], dtype=np.int64),
        edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        num_nodes=2,
    )


def test_missing_checkpoint_is_not_resume_safe(tmp_path) -> None:
    (tmp_path / "progress.json").write_text(
        json.dumps({"current_step": 46690, "run_complete": False}), encoding="utf-8"
    )
    audit = audit_generation_checkpoint(tmp_path)
    assert audit["RESUME_SAFE"] is False
    assert "missing_checkpoint_manifest" in audit["reasons"]


def test_complete_checkpoint_referential_closure_is_resume_safe(tmp_path) -> None:
    graph_state = tmp_path / "graph_state"
    graph_state.mkdir()
    store_path = graph_state / "authoritative_graph_store.sqlite3"
    store = AuthoritativeGraphStore(store_path)
    store.put("head", [graph(), np.asarray([1.0]), np.asarray([2.0])])
    checksum = store.integrity_audit()["content_sha256"]
    store.close()
    (tmp_path / "progress.json").write_text(
        json.dumps({"current_step": 1000, "run_complete": False}), encoding="utf-8"
    )
    manifest = {
        "atomic_complete": True,
        "current_step": 1000,
        "rng_state": {"python": 1, "numpy": 2, "torch_cpu": 3, "torch_cuda": 4},
        "current_graph_hashes": ["head"],
        "transition_source_hashes": ["head"],
        "transition_destination_hashes": ["head"],
        "live_reference_hashes": ["head"],
        "resolvable_hashes": ["head"],
        "unresolved_lookups": 0,
        "backing_store_path": str(store_path),
        "backing_store_content_sha256": checksum,
    }
    (graph_state / "checkpoint_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert audit_generation_checkpoint(tmp_path)["RESUME_SAFE"] is True


def test_incomplete_transition_closure_rejects_resume(tmp_path) -> None:
    (tmp_path / "progress.json").write_text(
        json.dumps({"current_step": 1000}), encoding="utf-8"
    )
    (tmp_path / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "atomic_complete": True,
                "current_step": 1000,
                "rng_state": {"python": 1, "numpy": 2, "torch_cpu": 3, "torch_cuda": 4},
                "current_graph_hashes": ["head"],
                "transition_source_hashes": ["missing"],
                "transition_destination_hashes": [],
                "live_reference_hashes": [],
                "resolvable_hashes": ["head"],
                "unresolved_lookups": 0,
            }
        ),
        encoding="utf-8",
    )
    audit = audit_generation_checkpoint(tmp_path)
    assert audit["RESUME_SAFE"] is False
    assert audit["checks"]["referential_closure_resolvable"] is False
