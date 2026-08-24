from __future__ import annotations

import json
from pathlib import Path

from src.baselines.comrecgc.contracts import sha256_file
from src.baselines.comrecgc.generation_divergence import (
    diagnose_generation_divergence,
    write_generation_divergence_report,
)


def _write_run(root: Path, *, action: list[object], candidate: str) -> None:
    trace = root / "_native_aux/trace"
    chunks = trace / "selected_action_trace_chunks"
    chunks.mkdir(parents=True)
    chunk = chunks / "part-000000.jsonl"
    rows = [
        {
            "event": "selected_transition",
            "head_index": 0,
            "move_index": 0,
            "parent_id": "BACE_0",
            "action": ["NLC", 1, 2],
            "source_graph_sha256": "a" * 64,
            "target_graph_sha256": "b" * 64,
            "source_official_hash": "process-specific-a",
            "target_official_hash": "process-specific-b",
        },
        {
            "event": "selected_transition",
            "head_index": 1,
            "move_index": 0,
            "parent_id": "BACE_1",
            "action": action,
            "source_graph_sha256": "c" * 64,
            "target_graph_sha256": candidate,
            "source_official_hash": "source",
            "target_official_hash": "target",
        },
    ]
    chunk.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (trace / "selected_action_trace_manifest.json").write_text(
        json.dumps(
            {
                "row_count": 2,
                "chunks": [
                    {
                        "path": "selected_action_trace_chunks/part-000000.jsonl",
                        "row_count": 2,
                        "sha256": sha256_file(chunk),
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    lineage = trace / "candidate_action_lineage_index.jsonl"
    lineage.write_text(
        json.dumps(
            {
                "candidate_index": 0,
                "parent_id": "BACE_1",
                "stable_graph_sha256": candidate,
                "action_count": 1,
                "lineage_root_status": "frozen_source_graph_exact",
                "lineage_storage": "selected_trace_predecessor_index",
                "official_graph_hash": "process-specific",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_reports_first_stable_selected_transition_and_candidate_difference(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy"
    optimized = tmp_path / "optimized"
    _write_run(legacy, action=["EA", 2, 31], candidate="d" * 64)
    _write_run(optimized, action=["EA", 2, 29], candidate="e" * 64)
    report = diagnose_generation_divergence(
        legacy_root=legacy,
        optimized_root=optimized,
    )
    assert report["status"] == "DIVERGENCE_IDENTIFIED"
    selected = report["first_stable_selected_transition_difference"]
    assert selected["row_index"] == 1
    assert selected["head_index"] == 1
    assert selected["move_index"] == 0
    assert selected["differences"]["action"] == {
        "legacy": ["EA", 2, 31],
        "optimized": ["EA", 2, 29],
    }
    assert report["first_candidate_sequence_difference"]["candidate_index"] == 0


def test_official_hash_only_difference_is_not_scientific_transition_difference(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy"
    optimized = tmp_path / "optimized"
    _write_run(legacy, action=["EA", 2, 31], candidate="d" * 64)
    _write_run(optimized, action=["EA", 2, 31], candidate="d" * 64)
    first = (
        optimized
        / "_native_aux/trace/selected_action_trace_chunks/part-000000.jsonl"
    )
    rows = [json.loads(line) for line in first.read_text().splitlines()]
    rows[0]["target_official_hash"] = "other-process-hash"
    first.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_path = (
        optimized / "_native_aux/trace/selected_action_trace_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["chunks"][0]["sha256"] = sha256_file(first)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    report = diagnose_generation_divergence(
        legacy_root=legacy,
        optimized_root=optimized,
    )
    assert report["first_any_selected_trace_difference"]["row_index"] == 0
    assert report["first_stable_selected_transition_difference"] is None
    assert report["first_candidate_sequence_difference"] is None


def test_writer_uses_fresh_root_and_terminal_marker(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    optimized = tmp_path / "optimized"
    _write_run(legacy, action=["EA", 2, 31], candidate="d" * 64)
    _write_run(optimized, action=["EA", 2, 29], candidate="e" * 64)
    output = tmp_path / "report"
    report = write_generation_divergence_report(
        legacy_root=legacy,
        optimized_root=optimized,
        output_dir=output,
    )
    assert report["status"] == "DIVERGENCE_IDENTIFIED"
    assert (output / "first_divergence.json").is_file()
    assert (output / "FIRST_DIVERGENCE_IDENTIFIED").is_file()
