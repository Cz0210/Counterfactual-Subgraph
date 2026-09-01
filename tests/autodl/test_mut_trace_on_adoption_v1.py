from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts.autodl import run_mut_fast_accurate_v2 as successor
from scripts.autodl import run_mut_trace_mode_equivalence as equivalence
from scripts.autodl import run_mut_trace_on_adoption_worker as worker
from src.baselines.comrecgc.contracts import sha256_file, stable_json_sha256
from src.utils import autodl_mut_trace_on_adoption_v1 as policy


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _audit_tree(commit: str, classification: str) -> dict[str, Any]:
    return {
        "status": "PASS",
        "commit": commit,
        "unknown_branches": [],
        "failed_scientific_assertions": [],
        "scientific_assertions": {"graph_closure_only": True},
        "branches": [{"classification": classification}],
    }


def _valid_trace_audit() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": policy.AUDIT_SCHEMA,
        "status": "PASS",
        "trace_is_observational": True,
        "trace_rng_mutation_found": False,
        "trace_algorithm_state_mutation_found": False,
        "trace_control_flow_mutation_found": False,
        "trace_candidate_selection_is_observational": True,
        "trace_operational_side_effects_found": True,
        "trace_post_walk_payload_serialization_mutation_found": True,
        "trace_post_walk_graph_closure_only": True,
        "static_audit_sufficient_for_adoption": False,
        "dynamic_500_step_equivalence_required": True,
        "full_trace_on_off_parity_claimed": False,
        "failures": [],
        "historical": _audit_tree(
            successor.SOURCE_PROJECT_COMMIT,
            "CHECKPOINT_SERIALIZATION_ONLY",
        ),
        "instrumentation": _audit_tree(
            successor.INSTRUMENTATION_PROJECT_COMMIT,
            "OBSERVATIONAL_WRITE_ONLY",
        ),
    }
    value["audit_sha256"] = stable_json_sha256(value)
    return value


def _valid_input_manifest(source: Path) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": "mut_trace_equivalence_input_manifest_v1",
        "source_algorithm_commit": successor.SOURCE_PROJECT_COMMIT,
        "execution_commit": successor.INSTRUMENTATION_PROJECT_COMMIT,
        "upstream_commit": successor.MUT_UPSTREAM_COMMIT,
        "formal_M_MAX": 50_000,
        "candidate_capacity": 100_000,
        "seed": 0,
        "parent_limit": 1448,
        "batch_size": 128,
        "device": "cuda:0",
        "pythonhashseed": "0",
        "algorithm_registry_identity": (
            "pinned_upstream_embedding_bytes_python_hash_seed0"
        ),
        "audit_graph_identity": "stable_untyped_graph_sha256",
        "dataset_sha256": successor.SOURCE_DATASET_SHA256,
        "split_source_cohort_sha256": successor.SOURCE_PARENT_ORDER_SHA256,
        "calibration_loaded": False,
        "test_loaded": False,
        "historical_artifact_root": str(source),
        "rf_oracle": {
            "loaded_by_generation_canary": False,
            "sha256": successor.MUT_RF_ORACLE_SHA256,
        },
        "input_files": {
            "gnn_checkpoint": {"sha256": successor.MUT_GNN_SHA256},
            "distance_checkpoint": {"sha256": successor.MUT_DISTANCE_SHA256},
        },
    }
    value["manifest_sha256"] = stable_json_sha256(value)
    return value


def _valid_equivalence(input_manifest: Path) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": "mut_trace_on_off_500_step_equivalence_v1",
        "status": "PASS",
        "trace_on_off_stepwise_exact": True,
        "first_semantic_divergence_step": None,
        "trace_on_checkpoint_reload_pass": True,
        "trace_off_checkpoint_reload_pass": True,
        "post_reload_trace_mode_equivalence_pass": True,
        "trace_on_trace_enabled": True,
        "trace_off_trace_enabled": False,
        "trace_only_files_excluded_from_scientific_digest": True,
        "step_action_trace_exact": True,
        "rng_state_exact": True,
        "classifier_probability_trace_exact": True,
        "step_semantic_fields_present": True,
        "step500_checkpoint_serialized_candidate_records_exact": True,
        "step500_checkpoint_candidate_universe_exact": True,
        "checkpoint_algorithm_scientific_state_exact": True,
        "checkpoint_rng_state_exact": True,
        "checkpoint_sqlite_logical_state_exact": True,
        "checkpoint_graph_registry_exact": True,
        "resolved_config_scientific_binding_exact": True,
        "post_walk_prefix_finalization_performed": False,
        "post_walk_candidate_semantics_bound_by_static_audit": True,
        "full_50k_trace_on_off_parity_claimed": False,
        "checkpoint_gates": {
            "trace_on_local_mirror": True,
            "trace_off_local_mirror": True,
            "trace_on_off_scientific_state": True,
        },
        "trace_on_checkpoint_state_audit": {"status": "PASS"},
        "trace_off_checkpoint_state_audit": {"status": "PASS"},
        "final_scientific_candidate_records_exact": True,
        "final_candidate_universe_exact": True,
        "arms_overlapped": False,
        "max_concurrent_arms": 1,
        "steps_compared": 500,
        "post_reload_steps_compared": 10,
        "calibration_loaded": False,
        "test_loaded": False,
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": sha256_file(input_manifest),
    }
    value["summary_sha256"] = stable_json_sha256(value)
    return value


def _write_candidate_pair_dbscan_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    source = tmp_path / "counterfactuals.pt"
    source.write_bytes(b"tiny synthetic source payload")
    monkeypatch.setattr(
        policy,
        "_torch_payload",
        lambda _path: {
            "graph_map": {"graph-0": object(), "graph-1": object(), "graph-2": object()},
            "counterfactual_candidates": [
                {"graph_hash": "graph-0", "importance_parts": [0.75]},
                {"graph_hash": "graph-1", "importance_parts": [0.25]},
                {"graph_hash": "graph-2", "importance_parts": [0.90]},
            ],
        },
    )
    graph_hashes = ["graph-0", "graph-2"]
    generation_indices = [0, 2]
    universe_sha = policy._stable_sha256(graph_hashes)

    pair_root = tmp_path / "pair"
    pair_root.mkdir()
    all_pairs = np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.int64)
    all_vectors = np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    pairs_path = pair_root / "pairs.npy"
    vectors_path = pair_root / "vectors.npy"
    np.save(pairs_path, all_pairs, allow_pickle=False)
    np.save(vectors_path, all_vectors, allow_pickle=False)

    chunks: list[dict[str, Any]] = []
    for index, (start, stop, row_start, row_stop) in enumerate(
        ((0, 1, 0, 2), (1, 2, 2, 3))
    ):
        chunk_pairs_path = pair_root / f"pairs-{index}.npy"
        chunk_vectors_path = pair_root / f"vectors-{index}.npy"
        np.save(chunk_pairs_path, all_pairs[row_start:row_stop], allow_pickle=False)
        np.save(chunk_vectors_path, all_vectors[row_start:row_stop], allow_pickle=False)
        identity = {
            "chunk_index": index,
            "candidate_start": start,
            "candidate_stop": stop,
            "candidate_graph_hashes_sha256": policy._stable_sha256(
                graph_hashes[start:stop]
            ),
            "generation_indices_sha256": policy._stable_sha256(
                generation_indices[start:stop]
            ),
        }
        chunks.append(
            {
                "chunk_index": index,
                "row_count": row_stop - row_start,
                "pairs_path": str(chunk_pairs_path),
                "pairs_sha256": sha256_file(chunk_pairs_path),
                "vectors_path": str(chunk_vectors_path),
                "vectors_sha256": sha256_file(chunk_vectors_path),
                "vectors_dtype": "float32",
                "vector_dim": 2,
                "first_pair": all_pairs[row_start].tolist(),
                "last_pair": all_pairs[row_stop - 1].tolist(),
                "scientific_identity": identity,
                "scientific_identity_sha256": policy._stable_sha256(identity),
            }
        )
    pair_science = {
        "candidate_count": 2,
        "candidate_graph_hashes_sha256": universe_sha,
        "generation_indices_sha256": policy._stable_sha256(generation_indices),
        "pair_order": "candidate_major_parent_minor",
    }
    pair_manifest = pair_root / "run_manifest.json"
    _write_json(
        pair_manifest,
        {
            "schema_version": "comrecgc_external_pair_store_v1",
            "run_complete": True,
            "candidate_major_parent_minor_order": True,
            "chunk_count": len(chunks),
            "chunks": chunks,
            "scientific_identity": pair_science,
            "scientific_identity_sha256": policy._stable_sha256(pair_science),
            "pairs_path": str(pairs_path),
            "pairs_sha256": sha256_file(pairs_path),
            "vectors_path": str(vectors_path),
            "vectors_sha256": sha256_file(vectors_path),
            "row_count": len(all_pairs),
            "vector_dim": 2,
            "vectors_dtype": "float32",
        },
    )

    dbscan_root = tmp_path / "dbscan"
    dbscan_root.mkdir()
    output_arrays = {
        "labels": np.asarray([0, 0, -1], dtype=np.int64),
        "core_mask": np.asarray([True, True, False], dtype=np.bool_),
        "neighbor_counts": np.asarray([2, 2, 1], dtype=np.int64),
    }
    dbscan_artifacts: dict[str, Any] = {}
    for label, array in output_arrays.items():
        path = dbscan_root / f"{label}.npy"
        np.save(path, array, allow_pickle=False)
        dbscan_artifacts[f"{label}_path"] = str(path)
        dbscan_artifacts[f"{label}_sha256"] = sha256_file(path)
    dbscan_manifest = dbscan_root / "run_manifest.json"
    _write_json(
        dbscan_manifest,
        {
            "schema_version": "comrecgc_external_memory_dbscan_v3",
            "run_complete": True,
            "approximation_used": False,
            "clustering_path": "sklearn_float64_exact_multi_component_v1",
            "scientific_identity": {
                "vectors_path": str(vectors_path),
                "vectors_sha256": sha256_file(vectors_path),
                "vectors_shape": [len(all_pairs), 2],
                "vectors_dtype": "float32",
                "distance_reference_dtype": "float64",
                "nearest_neighbors_algorithm": "brute",
                "nearest_neighbors_metric": "euclidean",
                "shortcut_contract": {
                    "reference_semantics": "SKLEARN_FLOAT64",
                    "comparison": "distance <= eps",
                    "failure_cap_used": False,
                },
                "contract": {"eps": 0.02, "min_samples": 3},
            },
            **dbscan_artifacts,
        },
    )
    return {
        "source_payload_path": source,
        "pair_manifest_path": pair_manifest,
        "dbscan_manifest_path": dbscan_manifest,
        "expected_candidate_universe_sha256": universe_sha,
        "expected_source_payload_sha256": sha256_file(source),
        "expected_candidate_count": 2,
        "candidate_capacity": 10,
        "first_chunk_vectors": Path(chunks[0]["vectors_path"]),
        "consolidated_vectors": vectors_path,
    }


def test_authorization_receipt_is_hash_closed_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "historical"
    source.mkdir()
    receipt = tmp_path / "control" / "authorization.json"
    written = policy.write_authorization_receipt(
        path=receipt,
        controller_id="mut-fast-test",
        source_root=source,
    )
    reopened, file_sha = policy.validate_authorization_receipt(
        receipt,
        expected_controller_id="mut-fast-test",
        expected_source_root=source,
    )
    assert reopened["authorization_sha256"] == written["authorization_sha256"]
    assert file_sha == written["receipt_file_sha256"]

    tampered = json.loads(receipt.read_text(encoding="utf-8"))
    tampered["allow_trace_on_50k_adoption"] = False
    _write_json(receipt, tampered)
    with pytest.raises(policy.MutTraceAuthorizationError, match="authorization"):
        policy.validate_authorization_receipt(
            receipt,
            expected_controller_id="mut-fast-test",
            expected_source_root=source,
        )

    physical = tmp_path / "physical.json"
    physical.write_text("{}\n", encoding="utf-8")
    linked = tmp_path / "authorization-link.json"
    linked.symlink_to(physical)
    with pytest.raises(policy.MutTraceAuthorizationError, match="non-symlink"):
        policy.validate_authorization_receipt(
            linked,
            expected_controller_id="mut-fast-test",
            expected_source_root=source,
        )


@pytest.mark.parametrize("changed_headroom", [63, 65])
def test_authorized_canary_headroom_is_exactly_64_gib(
    tmp_path: Path, changed_headroom: int
) -> None:
    source = tmp_path / "historical"
    source.mkdir()
    receipt = tmp_path / f"authorization-{changed_headroom}.json"
    value = policy.authorization_payload(
        controller_id="mut-fast-test",
        source_root=source,
    )
    assert policy.CANARY_REQUIRED_HEADROOM_GIB == 64
    assert policy.CANARY_REQUIRED_HEADROOM_BYTES == 64 * 1024**3
    value["canary_parent_headroom_gib"] = changed_headroom
    unhashed = {key: item for key, item in value.items() if key != "authorization_sha256"}
    value["authorization_sha256"] = policy._stable_sha256(unhashed)
    _write_json(receipt, value)
    with pytest.raises(
        policy.MutTraceAuthorizationError,
        match="canary_parent_headroom_gib",
    ):
        policy.validate_authorization_receipt(
            receipt,
            expected_controller_id="mut-fast-test",
            expected_source_root=source,
        )


def test_unknown_trace_branch_fails_static_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = tmp_path / "review"
    runtime = review / "src" / "baselines" / "comrecgc" / "runtime.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_text(
        "def mutate_science(trace_enabled, state, original, graph, action, trace_recorder):\n"
        "    result = _apply_neighbor_with_lineage(original, graph, action)\n"
        "    trace_recorder.record_enumerated(result)\n"
        "    if trace_enabled:\n"
        "        state.append('scientific mutation')\n"
        "    return result\n",
        encoding="utf-8",
    )
    (runtime.parent / "graph_trace.py").write_text(
        "class ActionTraceRecorder:\n"
        "    pass\n\n"
        "def wrap_move(original):\n"
        "    def wrapped(*args, **kwargs):\n"
        "        result = original(*args, **kwargs)\n"
        "        return result\n"
        "    return wrapped\n\n"
        "def write(payload):\n"
        "    payload.clear()\n"
        "    payload.update({})\n",
        encoding="utf-8",
    )
    (runtime.parent / "frozen_payload.py").write_text(
        "def build_frozen_payload_closure(payload):\n"
        "    frozen = dict(payload)\n"
        "    frozen['graph_map'] = {}\n"
        "    evidence = {\n"
        "        \"candidate_order_changed\": False,\n"
        "        \"candidate_payload_changed\": False,\n"
        "        \"scientific_parameters_changed\": False,\n"
        "    }\n"
        "    return frozen, evidence\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(policy, "_git_head", lambda _root: policy.SOURCE_PROJECT_COMMIT)
    monkeypatch.setattr(policy, "_git_science_tree_status", lambda _root: [])
    result = policy._tree_trace_inventory(
        review,
        expected_commit=policy.SOURCE_PROJECT_COMMIT,
    )
    assert result["status"] == "FAIL"
    assert len(result["unknown_branches"]) == 1
    assert result["unknown_branches"][0]["classification"] == "UNKNOWN"


def test_worker_reopens_graph_closure_only_static_audit_fail_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace_semantics_audit.json"
    value = _valid_trace_audit()
    _write_json(path, value)
    assert worker._validate_trace_code_audit(path)[
        "trace_post_walk_graph_closure_only"
    ] is True

    value["trace_post_walk_graph_closure_only"] = False
    value["audit_sha256"] = stable_json_sha256(
        {key: item for key, item in value.items() if key != "audit_sha256"}
    )
    _write_json(path, value)
    with pytest.raises(worker.MutTraceWorkerError, match="graph_closure_only"):
        worker._validate_trace_code_audit(path)


def test_worker_waits_for_64g_without_holding_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = iter(
        [
            {"headroom_bytes": 6 * 1024**3},
            {"headroom_bytes": 64 * 1024**3},
        ]
    )
    heartbeats: list[tuple[str, dict[str, Any]]] = []
    sleeps: list[int] = []
    monkeypatch.setattr(worker, "_cgroup_snapshot", lambda _root: next(snapshots))
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = worker._wait_for_64g_parent_headroom(
        Path("/unused"),
        heartbeat=lambda state, **extra: heartbeats.append((state, extra)),
    )

    assert result["headroom_bytes"] == 64 * 1024**3
    assert sleeps == [worker.HEADROOM_WAIT_SECONDS]
    assert heartbeats == [
        (
            "WAITING_FOR_64G_PARENT_HEADROOM",
            {
                "parent_headroom_bytes": 6 * 1024**3,
                "required_parent_headroom_bytes": (
                    policy.CANARY_REQUIRED_HEADROOM_BYTES
                ),
                "gpu_lock_held": False,
            },
        )
    ]


def test_worker_scopes_git_safe_directory_to_frozen_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = tmp_path / "vendor-comrecgc"
    upstream.mkdir()
    gnn = tmp_path / "gnn.pt"
    distance = tmp_path / "distance.pt"
    teacher = tmp_path / "teacher.pkl"
    for path in (gnn, distance, teacher):
        path.write_bytes(path.name.encode("utf-8"))

    commands: list[list[str]] = []

    def fake_check_output(command: list[str], **_kwargs: Any) -> str:
        commands.append(command)
        if command[-2:] == ["rev-parse", "HEAD"]:
            return successor.MUT_UPSTREAM_COMMIT + "\n"
        if command[-3:] == ["status", "--porcelain", "--untracked-files=no"]:
            return ""
        raise AssertionError(command)

    expected_hashes = {
        gnn.resolve(): worker.HISTORICAL_GNN_SHA256,
        distance.resolve(): worker.HISTORICAL_DISTANCE_SHA256,
        teacher.resolve(): worker.HISTORICAL_RF_ORACLE_SHA256,
    }
    monkeypatch.setattr(worker.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(worker, "sha256_file", lambda path: expected_hashes[path])

    result = worker._validate_frozen_replay_contract(
        {
            "replay": {
                "parent_limit": 1448,
                "batch_size": 128,
                "steps": 500,
                "seed": 0,
                "upstream_root": str(upstream),
                "gnn_checkpoint": str(gnn),
                "distance_checkpoint": str(distance),
            },
            "standardization": {"teacher_path": str(teacher)},
        }
    )

    safe_prefix = [
        "git",
        "-c",
        f"safe.directory={upstream.resolve()}",
        "-C",
        str(upstream.resolve()),
    ]
    assert commands == [
        [*safe_prefix, "rev-parse", "HEAD"],
        [*safe_prefix, "status", "--porcelain", "--untracked-files=no"],
    ]
    assert result["status"] == "PASS"


def test_worker_reopens_embedded_and_external_transitive_binding_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    common_root = tmp_path / "common"
    adoption_root = tmp_path / "adoption"
    source_root.mkdir()
    adoption_root.mkdir()
    source_payload = source_root / "counterfactuals.pt"
    source_payload.write_bytes(b"source")
    pair_manifest = tmp_path / "pair" / "run_manifest.json"
    dbscan_manifest = common_root / "external_memory/dbscan/run_manifest.json"
    _write_json(pair_manifest, {"status": "source-bound"})
    _write_json(dbscan_manifest, {"approximation_used": False})
    _write_json(
        common_root / "external_memory/pair_store_adoption/run_manifest.json",
        {"source_manifest_path": str(pair_manifest)},
    )
    expected = policy.EXPECTED_CANDIDATE_UNIVERSE_SHA256
    monkeypatch.setattr(worker, "SOURCE_PAYLOAD_SHA256", sha256_file(source_payload))
    receipt: dict[str, Any] = {
        "schema_version": worker.BINDING_SCHEMA,
        "status": "PASS",
        "binding_kind": worker.BINDING_KIND,
        "source_payload_path": str(source_payload),
        "source_payload_sha256": sha256_file(source_payload),
        "candidate_count": 50_620,
        "source_native_candidate_universe_sha": expected,
        "pair_store_source_candidate_universe_sha": expected,
        "pair_store_manifest_path": str(pair_manifest),
        "pair_store_manifest_sha256": sha256_file(pair_manifest),
        "dbscan_manifest_path": str(dbscan_manifest),
        "dbscan_manifest_sha256": sha256_file(dbscan_manifest),
        "dbscan_native_candidate_universe_sha": None,
        "dbscan_native_candidate_universe_field_present": False,
        "dbscan_transitively_bound_candidate_universe_sha": expected,
        "dbscan_approximation_used": False,
        "candidate_universe_binding_state": "PASS",
    }
    receipt["binding_sha256"] = worker._binding_self_sha256(receipt)
    monkeypatch.setattr(
        worker,
        "verify_mut_candidate_pair_dbscan_binding",
        lambda **_kwargs: dict(receipt),
    )
    external = adoption_root / "candidate_universe_binding.json"
    _write_json(external, receipt)
    adoption: dict[str, Any] = {
        "source_payload_sha256": sha256_file(source_payload),
        "candidate_universe_sha": expected,
        "source_native_candidate_universe_sha": expected,
        "pair_store_source_candidate_universe_sha": expected,
        "dbscan_native_candidate_universe_sha": None,
        "dbscan_transitively_bound_candidate_universe_sha": expected,
        "candidate_universe_binding_state": "PASS",
        "transitive_binding_kind": worker.BINDING_KIND,
        "dbscan_native_candidate_universe_field_present": False,
        "dbscan_universe_binding_via_pair_vectors": True,
        "pair_store_manifest_sha256": sha256_file(pair_manifest),
        "dbscan_manifest_sha256": sha256_file(dbscan_manifest),
        "candidate_pair_dbscan_binding_receipt": receipt,
        "candidate_pair_dbscan_binding_sha256": receipt["binding_sha256"],
        "candidate_pair_dbscan_binding_path": str(external),
        "candidate_pair_dbscan_binding_file_sha256": sha256_file(external),
    }
    adoption["binding_sha256"] = stable_json_sha256(adoption)

    reopened = worker._validate_transitive_candidate_universe_binding(
        spec={
            "historical_source_root": str(source_root),
            "completed_common_root": str(common_root),
        },
        adoption=adoption,
        adoption_root=adoption_root,
    )
    assert reopened == receipt

    receipt["dbscan_approximation_used"] = True
    receipt["binding_sha256"] = worker._binding_self_sha256(receipt)
    _write_json(external, receipt)
    adoption["candidate_pair_dbscan_binding_receipt"] = receipt
    adoption["candidate_pair_dbscan_binding_sha256"] = receipt["binding_sha256"]
    adoption["candidate_pair_dbscan_binding_file_sha256"] = sha256_file(external)
    adoption["binding_sha256"] = stable_json_sha256(
        {key: value for key, value in adoption.items() if key != "binding_sha256"}
    )
    with pytest.raises(worker.MutTraceWorkerError, match="approximation_used"):
        worker._validate_transitive_candidate_universe_binding(
            spec={
                "historical_source_root": str(source_root),
                "completed_common_root": str(common_root),
            },
            adoption=adoption,
            adoption_root=adoption_root,
        )


def test_step_comparator_excludes_only_trace_row_metadata() -> None:
    left = {
        "schema_version": "mut_trace_common_step_state_v1",
        "phase": "continuous",
        "trace_mode": "on",
        "step": 17,
        "rng_state_sha256": "a" * 64,
        "candidate_state": {"candidate_count": 4},
    }
    right = {
        **left,
        "schema_version": "ignored-serialization-label",
        "phase": "reload",
        "trace_mode": "off",
    }
    assert equivalence._row_science(left) == equivalence._row_science(right)

    # A scientific field must never be hidden merely because trace metadata is
    # allowed to differ.
    right["candidate_state"] = {"candidate_count": 5}
    assert equivalence._row_science(left) != equivalence._row_science(right)

    # Trace files are outside the common observer.  If a trace-only pathname is
    # accidentally injected into an observer row, it is not broadly ignored.
    right = {**left, "trace_output_path": "/different/trace.jsonl"}
    assert equivalence._row_science(left) != equivalence._row_science(right)


def test_step_comparator_identifies_first_semantic_divergence() -> None:
    left = {
        step: {
            "schema_version": "mut_trace_common_step_state_v1",
            "phase": "continuous",
            "trace_mode": "on",
            "step": step,
            "scientific_checkpoint_digest": str(step),
        }
        for step in range(1, 5)
    }
    right = {
        step: {**row, "trace_mode": "off"}
        for step, row in left.items()
    }
    right[3]["scientific_checkpoint_digest"] = "diverged"
    first = next(
        step
        for step in sorted(left)
        if equivalence._row_science(left[step])
        != equivalence._row_science(right[step])
    )
    assert first == 3


def test_source_pair_chunks_vectors_dbscan_transitive_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_candidate_pair_dbscan_chain(tmp_path, monkeypatch)
    result = policy.verify_mut_candidate_pair_dbscan_binding(
        **{
            key: value
            for key, value in inputs.items()
            if key not in {"first_chunk_vectors", "consolidated_vectors"}
        }
    )
    expected = inputs["expected_candidate_universe_sha256"]
    assert result["status"] == "PASS"
    assert result["source_native_candidate_universe_sha"] == expected
    assert result["pair_store_source_candidate_universe_sha"] == expected
    assert result["dbscan_native_candidate_universe_sha"] is None
    assert result["dbscan_native_candidate_universe_field_present"] is False
    assert result["dbscan_transitively_bound_candidate_universe_sha"] == expected
    assert result["dbscan_approximation_used"] is False
    assert (
        result["binding_kind"]
        == "transitive_generation_pair_store_vectors_dbscan_v1"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "source_universe",
        "pair_chunk_bytes",
        "consolidated_vector_bytes",
        "dbscan_vector_binding",
        "dbscan_native_candidate_sha",
        "dbscan_approximation",
    ],
)
def test_source_pair_chunks_vectors_dbscan_chain_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    inputs = _write_candidate_pair_dbscan_chain(tmp_path, monkeypatch)
    call = {
        key: value
        for key, value in inputs.items()
        if key not in {"first_chunk_vectors", "consolidated_vectors"}
    }
    if mutation == "source_universe":
        call["expected_candidate_universe_sha256"] = "f" * 64
    elif mutation == "pair_chunk_bytes":
        np.save(
            inputs["first_chunk_vectors"],
            np.asarray([[9.0, 9.0], [9.0, 9.0]], dtype=np.float32),
            allow_pickle=False,
        )
    elif mutation == "consolidated_vector_bytes":
        np.save(
            inputs["consolidated_vectors"],
            np.zeros((3, 2), dtype=np.float32),
            allow_pickle=False,
        )
    else:
        dbscan = json.loads(
            Path(inputs["dbscan_manifest_path"]).read_text(encoding="utf-8")
        )
        if mutation == "dbscan_vector_binding":
            dbscan["scientific_identity"]["vectors_sha256"] = "e" * 64
        elif mutation == "dbscan_native_candidate_sha":
            dbscan["scientific_identity"][
                "source_candidate_universe_sha256"
            ] = inputs["expected_candidate_universe_sha256"]
        else:
            dbscan["approximation_used"] = True
        _write_json(Path(inputs["dbscan_manifest_path"]), dbscan)
    with pytest.raises(policy.MutTraceAuthorizationError):
        policy.verify_mut_candidate_pair_dbscan_binding(**call)


def test_adoption_requires_both_authorization_and_static_audit(
    tmp_path: Path,
) -> None:
    inventory = tmp_path / "historical_inventory.json"
    _write_json(
        inventory,
        {
            "schema_version": "mut_historical_50k_inventory_v2",
            "status": "PASS",
            "trace_parity_passed": False,
        },
    )
    equivalence_gate = tmp_path / "equivalence.json"
    _write_json(equivalence_gate, {})
    source = tmp_path / "historical"
    source.mkdir()
    authorization = tmp_path / "authorization.json"
    policy.write_authorization_receipt(
        path=authorization,
        controller_id="mut-fast-test",
        source_root=source,
    )
    spec = {
        "controller_id": "mut-fast-test",
        "historical_source_root": str(source),
    }
    with pytest.raises(successor.MutFastError, match="requires both"):
        successor.publish_adoption(
            spec=spec,
            inventory_gate=inventory,
            equivalence_gate=equivalence_gate,
            output_dir=tmp_path / "adoption",
            authorization_receipt=authorization,
            trace_code_audit=None,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("closure", "Trace code audit"),
        ("checkpoint", "500-step equivalence"),
        ("gnn_config", "frozen input manifest"),
    ],
)
def test_adoption_fails_closed_on_static_checkpoint_and_config_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    source = tmp_path / "historical"
    source.mkdir()
    inventory = tmp_path / "historical_inventory.json"
    _write_json(
        inventory,
        {
            "schema_version": "mut_historical_50k_inventory_v2",
            "status": "PASS",
            "trace_parity_passed": False,
        },
    )
    authorization = tmp_path / "authorization.json"
    policy.write_authorization_receipt(
        path=authorization,
        controller_id="mut-fast-test",
        source_root=source,
    )
    audit = _valid_trace_audit()
    if mutation == "closure":
        audit["trace_post_walk_graph_closure_only"] = False
        audit["audit_sha256"] = stable_json_sha256(
            {key: value for key, value in audit.items() if key != "audit_sha256"}
        )
    audit_path = tmp_path / "trace_audit.json"
    _write_json(audit_path, audit)

    input_manifest = _valid_input_manifest(source)
    if mutation == "gnn_config":
        input_manifest["input_files"]["gnn_checkpoint"]["sha256"] = "f" * 64
        input_manifest["manifest_sha256"] = stable_json_sha256(
            {
                key: value
                for key, value in input_manifest.items()
                if key != "manifest_sha256"
            }
        )
    input_path = tmp_path / "equivalence_input_manifest.json"
    _write_json(input_path, input_manifest)
    gate = _valid_equivalence(input_path)
    if mutation == "checkpoint":
        gate["checkpoint_gates"]["trace_off_local_mirror"] = False
        gate["summary_sha256"] = stable_json_sha256(
            {key: value for key, value in gate.items() if key != "summary_sha256"}
        )
    gate_path = tmp_path / "trace_equivalence.json"
    _write_json(gate_path, gate)
    instrumentation = tmp_path / "instrumentation" / "equivalence.json"
    _write_json(instrumentation, {"status": "PASS"})
    monkeypatch.setattr(
        successor,
        "validate_instrumentation_equivalence_gate",
        lambda **_kwargs: {"status": "PASS"},
    )

    with pytest.raises(successor.MutFastError, match=message):
        successor.publish_adoption(
            spec={
                "controller_id": "mut-fast-test",
                "historical_source_root": str(source),
            },
            inventory_gate=inventory,
            equivalence_gate=gate_path,
            output_dir=tmp_path / "adoption",
            authorization_receipt=authorization,
            trace_code_audit=audit_path,
            instrumentation_equivalence_gate=instrumentation,
            canary_memory_receipt=tmp_path / "not-reached-memory.json",
        )


def test_adoption_rejects_candidate_universe_outside_authorized_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "historical"
    common = tmp_path / "common"
    pair = tmp_path / "pair" / "run_manifest.json"
    dbscan = common / "external_memory" / "dbscan" / "run_manifest.json"
    pair_adoption = (
        common / "external_memory" / "pair_store_adoption" / "run_manifest.json"
    )
    source_manifest = source / "run_manifest.json"
    source_payload = source / "counterfactuals.pt"
    source.mkdir()
    source_payload.write_bytes(b"historical payload")

    source_manifest_sha = "1" * 64
    pair_manifest_sha = "2" * 64
    dbscan_manifest_sha = "3" * 64
    vectors_path = str(tmp_path / "pair" / "vectors.npy")
    vectors_sha = "4" * 64
    _write_json(
        source_manifest,
        {
            "counterfactuals_sha256": successor.SOURCE_PAYLOAD_SHA256,
            "parameters": {"steps": 50_000, "candidate_capacity": 100_000},
        },
    )
    _write_json(
        pair,
        {
            "scientific_identity": {
                "counterfactuals_sha256": successor.SOURCE_PAYLOAD_SHA256,
                "dataset_fingerprint": successor.SOURCE_DATASET_SHA256,
                "parent_ids_sha256": successor.SOURCE_PARENT_ORDER_SHA256,
                "generation_manifest_sha256": source_manifest_sha,
                "candidate_graph_hashes_sha256": "f" * 64,
                "candidate_count": 50_620,
            },
            "vectors_path": vectors_path,
            "vectors_sha256": vectors_sha,
        },
    )
    _write_json(
        pair_adoption,
        {
            "source_manifest_path": str(pair),
            "source_manifest_sha256": pair_manifest_sha,
        },
    )
    _write_json(
        dbscan,
        {
            "run_complete": True,
            "approximation_used": False,
            "scientific_identity": {
                "vectors_path": vectors_path,
                "vectors_sha256": vectors_sha,
            },
        },
    )
    _write_json(
        common / "run_manifest.json",
        {
            "run_complete": True,
            "counterfactuals_sha256": successor.SOURCE_PAYLOAD_SHA256,
            "generation_manifest_sha256": source_manifest_sha,
            "common_recourse_count": 100,
            "external_memory_artifacts": {
                "engine": "external_memory_exact_v1",
                "pair_store_manifest": str(pair),
                "pair_store_manifest_sha256": pair_manifest_sha,
                "dbscan_manifest": str(dbscan),
                "dbscan_manifest_sha256": dbscan_manifest_sha,
            },
        },
    )

    inventory = tmp_path / "historical_inventory.json"
    _write_json(
        inventory,
        {
            "schema_version": "mut_historical_50k_inventory_v2",
            "status": "PASS",
            "trace_parity_passed": False,
        },
    )
    gate: dict[str, Any] = {
        "schema_version": "mut_trace_on_off_500_step_equivalence_v1",
        "status": "PASS",
        "trace_on_off_stepwise_exact": True,
        "first_semantic_divergence_step": None,
        "trace_on_checkpoint_reload_pass": True,
        "trace_off_checkpoint_reload_pass": True,
        "post_reload_trace_mode_equivalence_pass": True,
        "trace_on_trace_enabled": True,
        "trace_off_trace_enabled": False,
        "trace_only_files_excluded_from_scientific_digest": True,
        "step_action_trace_exact": True,
        "rng_state_exact": True,
        "classifier_probability_trace_exact": True,
        "step_semantic_fields_present": True,
        "step500_checkpoint_serialized_candidate_records_exact": True,
        "step500_checkpoint_candidate_universe_exact": True,
        "checkpoint_algorithm_scientific_state_exact": True,
        "checkpoint_rng_state_exact": True,
        "checkpoint_sqlite_logical_state_exact": True,
        "checkpoint_graph_registry_exact": True,
        "resolved_config_scientific_binding_exact": True,
        "post_walk_prefix_finalization_performed": False,
        "post_walk_candidate_semantics_bound_by_static_audit": True,
        "full_50k_trace_on_off_parity_claimed": False,
        "checkpoint_gates": {
            "trace_on_local_mirror": True,
            "trace_off_local_mirror": True,
            "trace_on_off_scientific_state": True,
        },
        "trace_on_checkpoint_state_audit": {"status": "PASS"},
        "trace_off_checkpoint_state_audit": {"status": "PASS"},
        "final_scientific_candidate_records_exact": True,
        "final_candidate_universe_exact": True,
        "arms_overlapped": False,
        "max_concurrent_arms": 1,
        "steps_compared": 500,
        "post_reload_steps_compared": 10,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    input_manifest: dict[str, Any] = {
        "schema_version": "mut_trace_equivalence_input_manifest_v1",
        "source_algorithm_commit": successor.SOURCE_PROJECT_COMMIT,
        "execution_commit": successor.INSTRUMENTATION_PROJECT_COMMIT,
        "upstream_commit": successor.MUT_UPSTREAM_COMMIT,
        "formal_M_MAX": 50_000,
        "candidate_capacity": 100_000,
        "seed": 0,
        "parent_limit": 1448,
        "batch_size": 128,
        "device": "cuda:0",
        "pythonhashseed": "0",
        "algorithm_registry_identity": (
            "pinned_upstream_embedding_bytes_python_hash_seed0"
        ),
        "audit_graph_identity": "stable_untyped_graph_sha256",
        "dataset_sha256": successor.SOURCE_DATASET_SHA256,
        "split_source_cohort_sha256": successor.SOURCE_PARENT_ORDER_SHA256,
        "calibration_loaded": False,
        "test_loaded": False,
        "historical_artifact_root": str(source),
        "rf_oracle": {
            "loaded_by_generation_canary": False,
            "sha256": successor.MUT_RF_ORACLE_SHA256,
        },
        "input_files": {
            "gnn_checkpoint": {"sha256": successor.MUT_GNN_SHA256},
            "distance_checkpoint": {"sha256": successor.MUT_DISTANCE_SHA256},
        },
    }
    input_manifest["manifest_sha256"] = stable_json_sha256(input_manifest)
    input_manifest_path = tmp_path / "equivalence_input_manifest.json"
    _write_json(input_manifest_path, input_manifest)
    gate["input_manifest"] = str(input_manifest_path)
    gate["input_manifest_sha256"] = "9" * 64
    gate["summary_sha256"] = stable_json_sha256(gate)
    equivalence_gate = tmp_path / "equivalence.json"
    _write_json(equivalence_gate, gate)

    authorization = tmp_path / "authorization.json"
    policy.write_authorization_receipt(
        path=authorization,
        controller_id="mut-fast-test",
        source_root=source,
    )
    audit: dict[str, Any] = {
        "schema_version": policy.AUDIT_SCHEMA,
        "status": "PASS",
        "trace_is_observational": True,
        "trace_rng_mutation_found": False,
        "trace_algorithm_state_mutation_found": False,
        "trace_control_flow_mutation_found": False,
        "trace_candidate_selection_is_observational": True,
        "trace_operational_side_effects_found": True,
        "trace_post_walk_payload_serialization_mutation_found": True,
        "trace_post_walk_graph_closure_only": True,
        "static_audit_sufficient_for_adoption": False,
        "dynamic_500_step_equivalence_required": True,
        "full_trace_on_off_parity_claimed": False,
        "historical": {
            "status": "PASS",
            "commit": successor.SOURCE_PROJECT_COMMIT,
            "unknown_branches": [],
            "failed_scientific_assertions": [],
            "scientific_assertions": {"graph_closure_only": True},
            "branches": [
                {
                    "classification": "CHECKPOINT_SERIALIZATION_ONLY",
                }
            ],
        },
        "instrumentation": {
            "status": "PASS",
            "commit": successor.INSTRUMENTATION_PROJECT_COMMIT,
            "unknown_branches": [],
            "failed_scientific_assertions": [],
            "scientific_assertions": {"graph_closure_only": True},
            "branches": [
                {
                    "classification": "OBSERVATIONAL_WRITE_ONLY",
                }
            ],
        },
    }
    audit["audit_sha256"] = stable_json_sha256(audit)
    audit_path = tmp_path / "audit.json"
    _write_json(audit_path, audit)

    instrumentation_gate = tmp_path / "instrumentation" / "equivalence.json"
    _write_json(instrumentation_gate, {"status": "PASS"})
    monkeypatch.setattr(
        successor,
        "validate_instrumentation_equivalence_gate",
        lambda **_kwargs: {
            "status": "PASS",
            "step_action_trace_exact": True,
            "rng_state_exact": True,
            "checkpoint_resume_exercised": True,
            "checkpoint_mirror_verified": True,
            "path": str(instrumentation_gate),
            "sha256": "8" * 64,
        },
    )
    phases = {
        phase: {
            "status": "PASS",
            "sample_count": 1,
            "peak_rss_bytes": policy.CANARY_RSS_STOP_BYTES,
            "minimum_parent_headroom_bytes": policy.CANARY_HEADROOM_STOP_BYTES,
        }
        for phase in worker.REQUIRED_TRACE_PHASES
    }
    memory: dict[str, Any] = {
        "schema_version": "mut_trace_mode_canary_memory_v1",
        "status": "PASS",
        "initial_parent_headroom_bytes": policy.CANARY_REQUIRED_HEADROOM_BYTES,
        "process_rss_peak_bytes": policy.CANARY_RSS_STOP_BYTES,
        "parent_headroom_min_bytes": policy.CANARY_HEADROOM_STOP_BYTES,
        "cgroup_failcnt_delta": 0,
        "cgroup_oom_delta": 0,
        "cgroup_oom_kill_delta": 0,
        "phases": phases,
        "protected_throughput_gate": {
            "status": "PASS",
            "missing_complete_five_minute_windows": [],
        },
    }
    memory["summary_sha256"] = stable_json_sha256(memory)
    memory_path = tmp_path / "canary_memory.json"
    _write_json(memory_path, memory)

    def fake_sha(path: str | Path) -> str:
        resolved = Path(path)
        if resolved == source_payload:
            return successor.SOURCE_PAYLOAD_SHA256
        if resolved == source_manifest:
            return source_manifest_sha
        if resolved == pair:
            return pair_manifest_sha
        if resolved == dbscan:
            return dbscan_manifest_sha
        return "9" * 64

    monkeypatch.setattr(successor, "sha256_file", fake_sha)
    with pytest.raises(successor.MutFastError, match="authorized_candidate_universe"):
        successor.publish_adoption(
            spec={
                "controller_id": "mut-fast-test",
                "historical_source_root": str(source),
                "completed_common_root": str(common),
                "proc_root": str(tmp_path / "proc"),
            },
            inventory_gate=inventory,
            equivalence_gate=equivalence_gate,
            output_dir=tmp_path / "adoption",
            authorization_receipt=authorization,
            trace_code_audit=audit_path,
            instrumentation_equivalence_gate=instrumentation_gate,
            canary_memory_receipt=memory_path,
        )


def test_protected_baseline_is_measured_without_wall_clock_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    read_count = [0]

    monkeypatch.setattr(policy.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(policy.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    def read_progress(task: dict[str, Any], *, proc_root: Path) -> dict[str, Any]:
        read_count[0] += 1
        return {
            "task_id": task["task_id"],
            "alive": True,
            "completed": False,
            "counter": clock[0],
        }

    monkeypatch.setattr(policy, "read_protected_progress", read_progress)
    manifest = {
        "tasks": [
            {
                "task_id": "protected",
                "pid": 1,
                "start_ticks": 2,
                "progress_path": "/progress.json",
                "counter_field": "step",
                "terminal_value": 1_000,
            }
        ]
    }
    result = policy.establish_protected_throughput_baseline(
        manifest,
        proc_root=Path("/proc"),
        baseline_seconds=300,
        poll_seconds=10,
    )
    assert result["status"] == "PASS"
    assert result["baseline_seconds"] == 300
    assert result["tasks"]["protected"]["units_per_second"] == 1.0
    assert read_count[0] >= 31


def test_protected_baseline_and_window_fail_closed_without_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(policy.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(policy.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(
        policy,
        "read_protected_progress",
        lambda task, *, proc_root: {
            "task_id": task["task_id"],
            "alive": True,
            "completed": False,
            "counter": 7.0,
        },
    )
    manifest = {
        "tasks": [
            {
                "task_id": "protected",
                "pid": 1,
                "start_ticks": 2,
                "progress_path": "/progress.json",
                "counter_field": "step",
                "terminal_value": 1_000,
            }
        ]
    }
    baseline = policy.establish_protected_throughput_baseline(
        manifest,
        proc_root=Path("/proc"),
        baseline_seconds=300,
        poll_seconds=10,
    )
    assert baseline["status"] == "FAIL"
    assert baseline["failures"] == ["no_positive_baseline:protected"]

    valid_baseline = {
        "status": "PASS",
        "tasks": {
            "protected": {
                "state": "ACTIVE",
                "units_per_second": 1.0,
            }
        },
    }
    epoch = [0.0]
    counter = [0.0]
    monkeypatch.setattr(policy.time, "time", lambda: epoch[0])
    monkeypatch.setattr(
        policy,
        "read_protected_progress",
        lambda task, *, proc_root: {
            "task_id": task["task_id"],
            "alive": True,
            "completed": False,
            "counter": counter[0],
            "sampled_at_unix": epoch[0],
        },
    )
    gate = policy.ProtectedThroughputGate(
        manifest,
        valid_baseline,
        proc_root=Path("/proc"),
    )
    assert gate.sample()["checked_window_count"] == 0
    before_window = gate.receipt()
    assert before_window["status"] == "FAIL"
    assert before_window["missing_complete_five_minute_windows"] == ["protected"]

    epoch[0] = 300.0
    counter[0] = 260.0
    sample = gate.sample()
    assert sample["status"] == "FAIL"
    assert sample["failures"] == ["protected_slowdown_gt_10_percent:protected"]
    receipt = gate.receipt()
    assert receipt["status"] == "FAIL"
    assert receipt["failures"] == ["protected_slowdown_gt_10_percent:protected"]
    assert receipt["failed_windows"][0]["slowdown_fraction"] > 0.10


@pytest.mark.parametrize("missing", ["phases", "protected_window"])
def test_memory_receipt_cannot_pass_with_incomplete_canary_evidence(
    tmp_path: Path, missing: str
) -> None:
    phases = {
        phase: {
            "sample_count": 1,
            "peak_rss_bytes": policy.CANARY_RSS_STOP_BYTES,
            "minimum_parent_headroom_bytes": policy.CANARY_HEADROOM_STOP_BYTES,
        }
        for phase in worker.REQUIRED_TRACE_PHASES
    }
    if missing == "phases":
        phases.pop("trace_off_reload")
    protected = {
        "status": "PASS",
        "missing_complete_five_minute_windows": [],
    }
    if missing == "protected_window":
        protected["missing_complete_five_minute_windows"] = ["taste_t11"]
    receipt: dict[str, Any] = {
        "schema_version": worker.MEMORY_SCHEMA,
        "status": "PASS",
        "arms_sequential": True,
        "max_concurrent_arms": 1,
        "phase_stats": phases,
        "protected_gate": protected,
        "failcnt_delta": 0,
        "oom_kill_delta": 0,
    }
    receipt["receipt_sha256"] = stable_json_sha256(receipt)
    path = tmp_path / f"memory-{missing}.json"
    _write_json(path, receipt)
    with pytest.raises(worker.MutTraceWorkerError, match="failed closed"):
        worker._validate_memory_receipt(path)


def test_worker_refuses_gnn_ablation_before_any_science(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUN_GNN_ABLATION", "1")
    with pytest.raises(worker.MutTraceWorkerError, match="must remain 0"):
        worker._run(SimpleNamespace())
