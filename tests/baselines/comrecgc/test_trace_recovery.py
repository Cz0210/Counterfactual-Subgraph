from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from src.baselines.comrecgc.contracts import (
    UPSTREAM_COMMIT,
    GenerationParameters,
    sha256_file,
    write_json,
)
from src.baselines.comrecgc.graph_trace import stable_graph_sha256
from src.baselines.comrecgc.trace_recovery import recover_mutagenicity_trace_run


@dataclass
class Graph:
    x: torch.Tensor
    edge_index: torch.Tensor
    num_nodes: int
    comrecgc_parent_id: str = "parent-1"
    comrecgc_trace_node_ids: tuple[str, ...] = ("source:0", "source:1")


def _graph(atom: int = 0) -> Graph:
    return Graph(
        x=torch.tensor(
            [[1.0 - atom, float(atom)], [0.0, 1.0]], dtype=torch.float32
        ),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        num_nodes=2,
    )


def _prepare_source(tmp_path: Path) -> tuple[Path, Path, str]:
    source_root = tmp_path / "source"
    trace_root = source_root / "trace"
    chunks = trace_root / "selected_action_trace_chunks"
    chunks.mkdir(parents=True)
    source_graph = _graph()
    target_graph = _graph(atom=1)
    traced_payload = {
        "graph_map": {"source": [source_graph], "target": [target_graph]},
        "counterfactual_candidates": [
            {
                "graph_hash": "target",
                "frequency": 2,
                "importance_parts": [0.8000005, 1.0],
            }
        ],
        "traversed_hashes": ["target"],
    }
    reference_payload = {
        "graph_map": {"source": [source_graph], "target": [target_graph]},
        "counterfactual_candidates": [
            {"graph_hash": "target", "frequency": 2, "importance_parts": [0.8, 1.0]}
        ],
        "traversed_hashes": ["target"],
    }
    torch.save(traced_payload, source_root / "counterfactuals.pt")
    reference = tmp_path / "reference.pt"
    torch.save(reference_payload, reference)
    write_json(
        source_root / "_RUN_FAILED.json",
        {
            "stage": "project_generation",
            "error_class": "ValueError",
            "message": "Trace-on/off candidate topology, features, frequency, importance, or order differ.",
            "calibration_loaded": False,
            "test_loaded": False,
        },
    )
    write_json(
        source_root / "resolved_config.json",
        {
            "dataset": "mutagenicity",
            "mode": "smoke",
            "parent_limit": 64,
            "parameters": GenerationParameters.for_mode("smoke").__dict__,
            "upstream_commit": UPSTREAM_COMMIT,
            "calibration_loaded": False,
            "test_loaded": False,
            "generation_parent_ids": [f"parent-{index}" for index in range(64)],
        },
    )
    event = {
        "move_index": 1,
        "head_index": 0,
        "event": "selected_transition",
        "source_official_hash": "source",
        "target_official_hash": "target",
        "source_graph_sha256": stable_graph_sha256(source_graph),
        "target_graph_sha256": stable_graph_sha256(target_graph),
        "action_resolution": "missing",
        "action": None,
        "parent_id": "parent-1",
    }
    chunk = chunks / "part-000000.jsonl"
    chunk.write_text(json.dumps(event, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "format": "chunked_jsonl",
        "chunk_size": 512,
        "row_count": 1,
        "chunks": [
            {
                "index": 0,
                "path": "selected_action_trace_chunks/part-000000.jsonl",
                "row_count": 1,
                "bytes": chunk.stat().st_size,
                "sha256": sha256_file(chunk),
                "materialization": "atomic_write",
            }
        ],
    }
    write_json(trace_root / "selected_action_trace_manifest.json", manifest)
    write_json(
        trace_root / "_TRACE_COMPLETE.json",
        {
            "trace_complete": True,
            "selected_trace_manifest_sha256": sha256_file(
                trace_root / "selected_action_trace_manifest.json"
            ),
            "candidate_lineage_sha256": "old-incomplete-lineage",
        },
    )
    write_json(
        trace_root / "trace_summary.json",
        {
            "candidate_count": 1,
            "candidate_lineage_resolved_count": 0,
            "rng_calls_added": 0,
        },
    )
    return source_root, reference, sha256_file(reference)


def test_recover_existing_trace_without_algorithm_rerun(tmp_path: Path) -> None:
    source, reference, reference_sha = _prepare_source(tmp_path)
    source_inventory = {
        path.relative_to(source).as_posix(): sha256_file(path)
        for path in source.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "recovered"

    result = recover_mutagenicity_trace_run(
        source_failed_generation_dir=source,
        reference_counterfactuals_path=reference,
        output_dir=output,
        expected_reference_sha256=reference_sha,
        expected_candidate_count=1,
    )

    assert result["recovery_validation_passed"] is True
    assert result["algorithm_rerun"] is False
    assert result["candidate_order_unchanged"] is True
    assert result["inferred_action_count"] == 1
    assert result["counterfactuals_materialization_mode"] == "hardlink"
    assert os.stat(source / "counterfactuals.pt").st_ino == os.stat(
        output / "counterfactuals.pt"
    ).st_ino
    lineage = json.loads(
        (output / "trace/candidate_action_lineage.json").read_text(encoding="utf-8")
    )
    assert lineage[0]["action_lineage_resolved"] is True
    assert lineage[0]["actions"][0]["action"] == ["NLC", 0, 1]
    assert (output / "_RUN_COMPLETE.json").is_file()
    assert not (output / "_RUN_FAILED.json").exists()
    assert source_inventory == {
        path.relative_to(source).as_posix(): sha256_file(path)
        for path in source.rglob("*")
        if path.is_file()
    }


def test_recovery_rejects_non_parity_failure(tmp_path: Path) -> None:
    source, reference, reference_sha = _prepare_source(tmp_path)
    marker = json.loads((source / "_RUN_FAILED.json").read_text(encoding="utf-8"))
    marker["message"] = "CUDA out of memory"
    write_json(source / "_RUN_FAILED.json", marker)

    with pytest.raises(ValueError, match="non-parity algorithm failure"):
        recover_mutagenicity_trace_run(
            source_failed_generation_dir=source,
            reference_counterfactuals_path=reference,
            output_dir=tmp_path / "blocked",
            expected_reference_sha256=reference_sha,
            expected_candidate_count=1,
        )
