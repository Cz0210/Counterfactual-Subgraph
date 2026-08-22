from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from src.baselines.comrecgc.bace_preprocessing import PREPROCESS_ENGINE
from src.baselines.comrecgc.contracts import (
    UPSTREAM_COMMIT,
    GenerationParameters,
    sha256_file,
    stable_json_sha256,
)
from src.baselines.comrecgc.full_acceleration_gate import (
    FullAccelerationGateError,
    build_full_acceleration_gate,
    validate_full_acceleration_gate,
)
from src.baselines.comrecgc.equivalence import (
    EQUIVALENCE_SCHEMA,
    FLOAT_ABS_TOLERANCE,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _root(
    tmp_path: Path,
    *,
    budget: int,
    oracle_hash: str,
    distance_hash: str,
    parent_hash: str,
    dataset_audit: dict[str, object],
) -> Path:
    root = tmp_path / f"m{budget}"
    runtime = {
        "engine": PREPROCESS_ENGINE,
        "workers": 4,
        "max_inflight": 64,
        "source_cache_capacity": 1024,
        "candidate_cache_capacity": 8192,
        "scientific_order_preserved": True,
        "rng_calls_added": 0,
    }
    base = {
        "run_complete": True,
        "dataset": "bace",
        "diagnostic_only": True,
        "paper_eligible": False,
        "diagnostic_equivalence_steps": budget,
        "traversed_step_count": budget,
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "upstream_commit": UPSTREAM_COMMIT,
        "generation_parent_ids_sha256": parent_hash,
        "oracle_checkpoint_hash": oracle_hash,
        "gnn": {
            "checkpoint_sha256": oracle_hash,
            "oracle_checkpoint_hash": oracle_hash,
            "oracle_backend": "gnn",
            "classifier_family": "gine",
            "rf_oracle_used": False,
        },
        "cf_mode": "strict_flip",
        "parent_limit": 360,
        "parameters": asdict(
            replace(GenerationParameters.for_mode("full"), steps=budget)
        ),
        "scientific_argv": ["--batch-size=128"],
        "distance_model": {"checkpoint_sha256": distance_hash},
        "dataset_audit": dataset_audit,
    }
    trace_rows: dict[str, list[dict[str, object]]] = {}
    payload_hashes: dict[str, str] = {}
    for role in ("legacy", "optimized"):
        role_root = root / role
        counterfactuals = role_root / "counterfactuals.pt"
        counterfactuals.parent.mkdir(parents=True, exist_ok=True)
        counterfactuals.write_bytes(f"same-{budget}".encode("utf-8"))
        payload_hashes[role] = sha256_file(counterfactuals)
        _write(
            role_root / "_RUN_COMPLETE.json",
            {
                "run_complete": True,
                "counterfactuals_sha256": payload_hashes[role],
            },
        )
        _write(
            role_root / "DIAGNOSTIC_ONLY.json",
            {
                "diagnostic_only": True,
                "paper_eligible": False,
                "steps": budget,
                "role": role,
            },
        )
        chunk = (
            role_root
            / "_native_aux/trace/selected_action_trace_chunks/part-000000.jsonl"
        )
        chunk.parent.mkdir(parents=True, exist_ok=True)
        chunk.write_text('{"move_index":1}\n', encoding="utf-8")
        trace_rows[role] = [
            {
                "path": "selected_action_trace_chunks/part-000000.jsonl",
                "row_count": 1,
                "sha256": sha256_file(chunk),
            }
        ]
    _write(
        root / "legacy/run_manifest.json",
        {
            **base,
            "equivalence_gate_role": "legacy",
            "bace_preprocessing": {"engine": "legacy_sequential_rdkit_v1"},
            "counterfactuals_sha256": payload_hashes["legacy"],
        },
    )
    _write(
        root / "optimized/run_manifest.json",
        {
            **base,
            "equivalence_gate_role": "optimized",
            "bace_preprocessing": runtime,
            "counterfactuals_sha256": payload_hashes["optimized"],
        },
    )
    audit = {
        "schema_version": EQUIVALENCE_SCHEMA,
        "status": "PASS",
        "expected_steps": budget,
        "legacy_root": str((root / "legacy").resolve()),
        "optimized_root": str((root / "optimized").resolve()),
        "legacy_counterfactuals_sha256": payload_hashes["legacy"],
        "optimized_counterfactuals_sha256": payload_hashes["optimized"],
        "identity_mismatches": [],
        "payload": {
            "candidate_parity": {"trace_parity_passed": True},
            "graph_map_key_order_exact": True,
            "graph_index_map_exact": True,
            "traversed_hashes_exact": True,
            "input_graphs_covered_max_abs_difference": 0.0,
            "graph_identity_mismatch_count": 0,
            "graph_embedding_or_element_max_abs_difference": 0.0,
            "float_abs_tolerance": FLOAT_ABS_TOLERANCE,
            "failures": [],
        },
        "selected_trace_chunks_exact": True,
        "legacy_trace_chunks": trace_rows["legacy"],
        "optimized_trace_chunks": trace_rows["optimized"],
        "failures": [],
        "paper_eligible": False,
    }
    audit["summary_sha256"] = stable_json_sha256(audit)
    _write(
        root / "audit/equivalence_summary.json",
        audit,
    )
    (root / "audit/PASS").write_text("PASS\n", encoding="utf-8")
    (root / "PASS").write_text("PASS\n", encoding="utf-8")
    return root


def test_exact_500_1000_gate_binds_full_runtime_and_checkpoints(tmp_path: Path) -> None:
    gnn = tmp_path / "gine"
    gnn.mkdir()
    (gnn / "model.pt").write_bytes(b"frozen-gine")
    distance = tmp_path / "distance.pt"
    distance.write_bytes(b"distance")
    parent_hash = stable_json_sha256(["p0", "p1"])
    dataset_audit = {"dataset_sha256": "d" * 64, "rows": 360}
    roots = [
        _root(
            tmp_path,
            budget=budget,
            oracle_hash=sha256_file(gnn / "model.pt"),
            distance_hash=sha256_file(distance),
            parent_hash=parent_hash,
            dataset_audit=dataset_audit,
        )
        for budget in (500, 1000)
    ]
    gate = build_full_acceleration_gate(
        m500_root=roots[0], m1000_root=roots[1], output_dir=tmp_path / "gate"
    )
    gate_path = tmp_path / "gate/FULL_ACCELERATION_GATE.json"
    validated = validate_full_acceleration_gate(
        gate_path,
        expected_gate_sha256=sha256_file(gate_path),
        gnn_checkpoint=gnn,
        distance_checkpoint=distance,
        generation_parent_ids_sha256=parent_hash,
        dataset_audit=dataset_audit,
        parent_limit=360,
        parameters=GenerationParameters.for_mode("full"),
        preprocess_engine=PREPROCESS_ENGINE,
        batch_size=128,
        preprocess_workers=4,
        preprocess_max_inflight=64,
        source_cache_capacity=1024,
        candidate_cache_capacity=8192,
    )
    assert gate["budgets"] == [500, 1000]
    assert gate["full_generation_steps"] == 50_000
    assert validated["status"] == "PASS"


def test_gate_rejects_cross_budget_oracle_drift(tmp_path: Path) -> None:
    distance = "d" * 64
    parents = "c" * 64
    audit = {"rows": 360}
    m500 = _root(
        tmp_path,
        budget=500,
        oracle_hash="a" * 64,
        distance_hash=distance,
        parent_hash=parents,
        dataset_audit=audit,
    )
    m1000 = _root(
        tmp_path,
        budget=1000,
        oracle_hash="b" * 64,
        distance_hash=distance,
        parent_hash=parents,
        dataset_audit=audit,
    )
    with pytest.raises(FullAccelerationGateError, match="scientific contracts differ"):
        build_full_acceleration_gate(
            m500_root=m500, m1000_root=m1000, output_dir=tmp_path / "gate"
        )


def test_gate_rejects_legacy_optimized_batch_size_drift(tmp_path: Path) -> None:
    roots = [
        _root(
            tmp_path,
            budget=budget,
            oracle_hash="a" * 64,
            distance_hash="b" * 64,
            parent_hash="c" * 64,
            dataset_audit={"rows": 360},
        )
        for budget in (500, 1000)
    ]
    legacy_path = roots[0] / "legacy/run_manifest.json"
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    legacy["scientific_argv"] = ["--batch-size=64"]
    _write(legacy_path, legacy)
    with pytest.raises(FullAccelerationGateError, match="scientific contracts differ"):
        build_full_acceleration_gate(
            m500_root=roots[0], m1000_root=roots[1], output_dir=tmp_path / "gate"
        )


def test_full_validator_rejects_runtime_or_gate_hash_drift(tmp_path: Path) -> None:
    gnn = tmp_path / "gine"
    gnn.mkdir()
    (gnn / "model.pt").write_bytes(b"g")
    distance = tmp_path / "distance.pt"
    distance.write_bytes(b"d")
    parent_hash = "a" * 64
    audit = {"rows": 360}
    roots = [
        _root(
            tmp_path,
            budget=budget,
            oracle_hash=sha256_file(gnn / "model.pt"),
            distance_hash=sha256_file(distance),
            parent_hash=parent_hash,
            dataset_audit=audit,
        )
        for budget in (500, 1000)
    ]
    build_full_acceleration_gate(
        m500_root=roots[0], m1000_root=roots[1], output_dir=tmp_path / "gate"
    )
    gate_path = tmp_path / "gate/FULL_ACCELERATION_GATE.json"
    common = dict(
        gate_path=gate_path,
        expected_gate_sha256=sha256_file(gate_path),
        gnn_checkpoint=gnn,
        distance_checkpoint=distance,
        generation_parent_ids_sha256=parent_hash,
        dataset_audit=audit,
        parent_limit=360,
        parameters=GenerationParameters.for_mode("full"),
        preprocess_engine=PREPROCESS_ENGINE,
        batch_size=128,
        preprocess_workers=4,
        preprocess_max_inflight=64,
        source_cache_capacity=1024,
        candidate_cache_capacity=8192,
    )
    with pytest.raises(FullAccelerationGateError, match="runtime differs"):
        validate_full_acceleration_gate(**{**common, "preprocess_workers": 5})
    with pytest.raises(FullAccelerationGateError, match="SHA256 mismatch"):
        validate_full_acceleration_gate(**{**common, "expected_gate_sha256": "0" * 64})


def test_full_validator_rechecks_replay_roots_after_gate_publication(
    tmp_path: Path,
) -> None:
    gnn = tmp_path / "gine"
    gnn.mkdir()
    (gnn / "model.pt").write_bytes(b"g")
    distance = tmp_path / "distance.pt"
    distance.write_bytes(b"d")
    parent_hash = "a" * 64
    audit_contract = {"rows": 360}
    roots = [
        _root(
            tmp_path,
            budget=budget,
            oracle_hash=sha256_file(gnn / "model.pt"),
            distance_hash=sha256_file(distance),
            parent_hash=parent_hash,
            dataset_audit=audit_contract,
        )
        for budget in (500, 1000)
    ]
    build_full_acceleration_gate(
        m500_root=roots[0], m1000_root=roots[1], output_dir=tmp_path / "gate"
    )
    gate_path = tmp_path / "gate/FULL_ACCELERATION_GATE.json"
    audit_path = roots[1] / "audit/equivalence_summary.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["status"] = "FAIL"
    audit.pop("summary_sha256")
    audit["summary_sha256"] = stable_json_sha256(audit)
    _write(audit_path, audit)
    with pytest.raises(FullAccelerationGateError, match="audit is not clean"):
        validate_full_acceleration_gate(
            gate_path,
            expected_gate_sha256=sha256_file(gate_path),
            gnn_checkpoint=gnn,
            distance_checkpoint=distance,
            generation_parent_ids_sha256=parent_hash,
            dataset_audit=audit_contract,
            parent_limit=360,
            parameters=GenerationParameters.for_mode("full"),
            preprocess_engine=PREPROCESS_ENGINE,
            batch_size=128,
            preprocess_workers=4,
            preprocess_max_inflight=64,
            source_cache_capacity=1024,
            candidate_cache_capacity=8192,
        )
