from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.baselines.comrecgc.contracts import GenerationParameters, sha256_file, write_json
from src.baselines.comrecgc.freeze_recovery import validate_completed_generation_freeze
from src.baselines.comrecgc.graph_trace import stable_untyped_graph_sha256
from src.baselines.comrecgc.live_graph_state import AuthoritativeGraphStore


torch = pytest.importorskip("torch")


def _graph(values: list[int], edges: list[tuple[int, int]]) -> SimpleNamespace:
    return SimpleNamespace(
        x=np.asarray([[value] for value in values], dtype=np.int64),
        edge_index=np.asarray(edges, dtype=np.int64).T
        if edges
        else np.empty((2, 0), dtype=np.int64),
        num_nodes=len(values),
        comrecgc_parent_id="parent",
    )


def _source_root(tmp_path, *, invalid_destinations: int = 0):
    root = tmp_path / "generation"
    trace = root / "trace"
    chunks = trace / "selected_action_trace_chunks"
    graph_state = root / "graph_state"
    chunks.mkdir(parents=True)
    graph_state.mkdir(parents=True)
    source = _graph([1, 2], [(0, 1), (1, 0)])
    target = _graph([2], [])
    store = AuthoritativeGraphStore(
        graph_state / "authoritative_graph_store.sqlite3"
    )
    store.put("source", [source, np.asarray([1.0]), np.asarray([2.0])])
    store_audit = store.integrity_audit()
    store.close()
    payload = {
        "graph_map": {
            "target": [target, np.asarray([1.0]), np.asarray([2.0])]
        },
        "counterfactual_candidates": [{"graph_hash": "target", "frequency": 1}],
        "traversed_hashes": ["source", "target"],
    }
    torch.save(payload, root / "counterfactuals.pt")
    event = {
        "move_index": 49_999,
        "head_index": 0,
        "event": "selected_transition",
        "parent_id": "parent",
        "source_official_hash": "source",
        "target_official_hash": "target",
        "source_graph_sha256": stable_untyped_graph_sha256(source),
        "target_graph_sha256": stable_untyped_graph_sha256(target),
        "action": ["NR", 0, 0],
    }
    chunk = chunks / "part-000000.jsonl"
    chunk.write_text(json.dumps(event, sort_keys=True) + "\n", encoding="utf-8")
    write_json(
        trace / "selected_action_trace_manifest.json",
        {
            "schema_version": 1,
            "format": "chunked_jsonl",
            "row_count": 1,
            "chunks": [
                {
                    "index": 0,
                    "path": "selected_action_trace_chunks/part-000000.jsonl",
                    "row_count": 1,
                    "sha256": sha256_file(chunk),
                }
            ],
        },
    )
    write_json(
        root / "resolved_config.json",
        {
            "dataset": "aids",
            "mode": "full",
            "project_commit": "base",
            "parent_limit": 1,
            "parameters": GenerationParameters.for_mode("full").__dict__,
        },
    )
    write_json(
        root / "_RUN_FAILED.json",
        {
            "stage": "project_generation",
            "message": "Selected trace references a graph absent from the frozen payload.",
        },
    )
    write_json(
        root / "graph_state_audit.json",
        {
            "move_count": 50_000,
            "unresolved_lookups": 0,
            "unresolved_transition_source_count": 0,
            "invalid_transition_destination_count": invalid_destinations,
            "backing_store": store_audit,
        },
    )
    return root


def test_completed_walk_is_safe_for_freeze_without_rng_resume_state(tmp_path) -> None:
    root = _source_root(tmp_path)

    audit, payload = validate_completed_generation_freeze(
        source_generation_dir=root,
        dataset="aids",
        dataset_dir=tmp_path / "unused",
        source_csv=tmp_path / "unused.csv",
        expected_project_commit="base",
    )

    assert audit["random_walk_complete"] is True
    assert audit["RNG_state_present"] is False
    assert audit["rng_state_required_for_freeze_only"] is False
    assert audit["FREEZE_ONLY_RECOVERY_SAFE"] is True
    assert payload is not None
    assert "source" in payload["graph_map"]


def test_transition_integrity_failure_blocks_freeze_only(tmp_path) -> None:
    root = _source_root(tmp_path, invalid_destinations=1)

    audit, payload = validate_completed_generation_freeze(
        source_generation_dir=root,
        dataset="aids",
        dataset_dir=tmp_path / "unused",
        source_csv=tmp_path / "unused.csv",
        expected_project_commit="base",
    )

    assert audit["FREEZE_ONLY_RECOVERY_SAFE"] is False
    assert audit["checks"]["transition_destinations_valid"] is False
    assert payload is None
