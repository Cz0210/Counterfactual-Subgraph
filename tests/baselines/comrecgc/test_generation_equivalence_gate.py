from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.baselines.comrecgc.contracts import sha256_file
from src.baselines.comrecgc.equivalence import audit_generation_equivalence


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _run_root(tmp_path: Path, *, role: str, frequency: int = 2) -> Path:
    torch = pytest.importorskip("torch")
    root = tmp_path / role
    trace = root / "_native_aux/trace"
    chunks = trace / "selected_action_trace_chunks"
    chunks.mkdir(parents=True)
    graph = SimpleNamespace(
        x=[[1.0, 0.0], [0.0, 1.0]],
        edge_index=[[0, 1], [1, 0]],
        num_nodes=2,
    )
    payload = {
        "graph_map": {101: [graph, [0.25, 0.5], [2, 1]]},
        "graph_index_map": {101: 0},
        "counterfactual_candidates": [
            {
                "graph_hash": 101,
                "frequency": frequency,
                "importance_parts": [0.75, 1.0],
            }
        ],
        "traversed_hashes": [[101]],
        "input_graphs_covered": [1.0, 0.0],
    }
    torch.save(payload, root / "counterfactuals.pt")
    payload_sha256 = sha256_file(root / "counterfactuals.pt")
    chunk = chunks / "part-000000.jsonl"
    chunk.write_text('{"move_index":1,"target_official_hash":101}\n', encoding="utf-8")
    _write_json(
        trace / "selected_action_trace_manifest.json",
        {
            "row_count": 1,
            "chunks": [
                {
                    "path": "selected_action_trace_chunks/part-000000.jsonl",
                    "row_count": 1,
                }
            ],
        },
    )
    engine = (
        "legacy_sequential_rdkit_v1"
        if role == "legacy"
        else "ordered_bounded_rdkit_process_pool_v1"
    )
    manifest = {
        "run_complete": True,
        "dataset": "bace",
        "diagnostic_only": True,
        "paper_eligible": False,
        "diagnostic_equivalence_steps": 500,
        "equivalence_gate_role": role,
        "traversed_step_count": 500,
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "upstream_commit": "upstream",
        "generation_parent_ids_sha256": "parents",
        "oracle_checkpoint_hash": "oracle",
        "cf_mode": "strict_flip",
        "parent_limit": 360,
        "parameters": {"seed": 0, "steps": 500, "heads": 5},
        "distance_model": {"checkpoint_sha256": "distance"},
        "device_contract": {
            "algorithm_device": "cuda:0",
            "graph_identity_device": "cpu",
        },
        "dataset_audit": {"dataset_sha256": "dataset"},
        "internal_prediction_counts": {"0": 1, "1": 1},
        "bace_preprocessing": {"engine": engine},
        "counterfactuals_sha256": payload_sha256,
    }
    _write_json(root / "run_manifest.json", manifest)
    _write_json(
        root / "_RUN_COMPLETE.json",
        {"run_complete": True, "counterfactuals_sha256": payload_sha256},
    )
    _write_json(
        root / "DIAGNOSTIC_ONLY.json",
        {
            "diagnostic_only": True,
            "paper_eligible": False,
            "steps": 500,
            "role": role,
        },
    )
    return root


def test_equivalence_gate_writes_pass_only_for_matching_fresh_prefixes(
    tmp_path: Path,
) -> None:
    legacy = _run_root(tmp_path, role="legacy")
    optimized = _run_root(tmp_path, role="optimized")
    result = audit_generation_equivalence(
        legacy_root=legacy,
        optimized_root=optimized,
        output_dir=tmp_path / "audit",
        expected_steps=500,
    )
    assert result["status"] == "PASS"
    assert (tmp_path / "audit/PASS").is_file()
    assert not (tmp_path / "audit/FAIL.json").exists()


def test_equivalence_gate_fails_closed_on_candidate_frequency_drift(
    tmp_path: Path,
) -> None:
    legacy = _run_root(tmp_path, role="legacy")
    optimized = _run_root(tmp_path, role="optimized", frequency=3)
    with pytest.raises(RuntimeError, match="equivalence gate failed"):
        audit_generation_equivalence(
            legacy_root=legacy,
            optimized_root=optimized,
            output_dir=tmp_path / "audit",
            expected_steps=500,
        )
    assert not (tmp_path / "audit/PASS").exists()
    assert (tmp_path / "audit/FAIL.json").is_file()


def test_equivalence_gate_fails_closed_on_unbound_completion_marker(
    tmp_path: Path,
) -> None:
    legacy = _run_root(tmp_path, role="legacy")
    optimized = _run_root(tmp_path, role="optimized")
    _write_json(
        optimized / "_RUN_COMPLETE.json",
        {"run_complete": True, "counterfactuals_sha256": "0" * 64},
    )
    with pytest.raises(RuntimeError, match="equivalence gate failed"):
        audit_generation_equivalence(
            legacy_root=legacy,
            optimized_root=optimized,
            output_dir=tmp_path / "audit",
            expected_steps=500,
        )
    assert not (tmp_path / "audit/PASS").exists()
    assert (tmp_path / "audit/FAIL.json").is_file()


def test_equivalence_gate_fails_closed_on_device_contract_drift(
    tmp_path: Path,
) -> None:
    legacy = _run_root(tmp_path, role="legacy")
    optimized = _run_root(tmp_path, role="optimized")
    optimized_manifest_path = optimized / "run_manifest.json"
    optimized_manifest = json.loads(
        optimized_manifest_path.read_text(encoding="utf-8")
    )
    optimized_manifest["device_contract"]["graph_identity_device"] = "cuda:0"
    _write_json(optimized_manifest_path, optimized_manifest)

    with pytest.raises(RuntimeError, match="equivalence gate failed"):
        audit_generation_equivalence(
            legacy_root=legacy,
            optimized_root=optimized,
            output_dir=tmp_path / "audit",
            expected_steps=500,
        )

    failure = json.loads((tmp_path / "audit/FAIL.json").read_text(encoding="utf-8"))
    assert "device_contract" in failure["identity_mismatches"]
    assert not (tmp_path / "audit/PASS").exists()
