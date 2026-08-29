from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from src.baselines.comrecgc import external_memory_dbscan as external
from src.baselines.comrecgc.aids_pair_semantics import AIDS_PAIR_SEMANTICS_SCHEMA
from src.baselines.comrecgc.close_pair_view import (
    ALL_PAIRS_CLOSE_CERTIFICATE_SCHEMA,
    CLOSE_PAIR_VIEW_SCHEMA,
    FILTER_OPERATOR,
    NORMALIZED_DISTANCE_CONTRACT,
    PAIR_ORDER,
    PAIR_ORIENTATION,
    SCALE_CONTRACT,
)
from src.baselines.comrecgc.external_memory_recourse import PAIR_STORE_SCHEMA
from src.baselines.comrecgc.failed_selection_adoption import (
    CONTROL_NAMESPACE,
    CLOSE_TASK_ID,
    FINAL_TASK_ID,
    PRODUCTION_AUTHORITY,
    PRODUCTION_CONTROL_ROOT,
    READY_PREPARED_NAME,
    READY_NAME,
    RECEIPT_NAME,
    SOURCE_CONTROLLER_ID,
    FailedSelectionAdoptionError,
    FailedSelectionAuthority,
    TaskStateAuthority,
    _anchor_graph_summary,
    _create_or_validate_with_profile,
    _ready_marker_bytes,
    _stable_hash as _adoption_stable_hash,
)


sklearn = pytest.importorskip("sklearn")


def test_production_authority_is_exactly_the_c766_failed_route() -> None:
    assert PRODUCTION_CONTROL_ROOT == Path(
        "/autodl-fs/data/counterfactual-subgraph-runtime/control"
    )
    assert CONTROL_NAMESPACE == "four_methods_four_datasets_continuation"
    assert SOURCE_CONTROLLER_ID == (
        "four_methods_four_datasets_aids_comrecgc_exact_route_v5_snapshot_adopt_v1"
    )
    assert CLOSE_TASK_ID == "aids_comrecgc_theta_close_view_v1"
    assert FINAL_TASK_ID == (
        "aids_comrecgc_standardized_exact_route_v5_snapshot_adopt_v1"
    )
    assert PRODUCTION_AUTHORITY.controller_manifest_sha256 == (
        "7b2987bc2d223ebe3262cc15bc43bd1c0b030c6706a1c074959d154af5fd84d7"
    )
    assert PRODUCTION_AUTHORITY.close_gate_sha256 == (
        "042837003d8e07c41d10c283909f5dc545659d6a2ad99db25d8652509ac03e8b"
    )
    assert PRODUCTION_AUTHORITY.close_manifest_sha256 == (
        "d41792e65bc9989c9a2c0abb9ef4c552ed863c9e362d9bba72cf1cc6dd5d331a"
    )
    assert PRODUCTION_AUTHORITY.final_gate_sha256 == (
        "a7c46f485a18a42d5dce081528945cc859c5c53d4ad7a343c62ecb246089e65b"
    )
    assert PRODUCTION_AUTHORITY.shortcut_failure_sha256 == (
        "74bc3d73e99568b2cc05dfda3d62d39538acc1dae9ccd9fb4743346ad1e6cea5"
    )
    assert PRODUCTION_AUTHORITY.selection_manifest_sha256 == (
        "0c3e569d65fa299e2658321bf5cfd2961c0cab1d19cd49aa2799045d1cab6e8e"
    )
    assert PRODUCTION_AUTHORITY.checkpoint_sha256 == (
        "fa3ed4a566f1518876ebc58e3bbd0fc1e87d8d6b2a44a53576fba398f1fd0a3b"
    )
    assert PRODUCTION_AUTHORITY.failure_indices_sha256 == (
        "b56883c3c79d60e6cd582eb071278b78289b63a0b724ab55a407cd506d0502be"
    )
    assert PRODUCTION_AUTHORITY.anchor_indices_sha256 == (
        "0f70593d7d632fd1040ea5c5fbf128552afbb917185dd673832cd9214db028c4"
    )
    assert PRODUCTION_AUTHORITY.anchor_rows_sha256 == (
        "ff32eec327569527862cf18d1d9dbe5ac374a63486e757cb1e5587494a976012"
    )
    assert PRODUCTION_AUTHORITY.anchor_edges_sha256 == (
        "91aacf23b644ed89e11247b4e6c23ffe6b7c7cc994b3ab9bf0ef8e10e78f2f3a"
    )
    assert PRODUCTION_AUTHORITY.physical_pairs_sha256 == (
        "c83eba699f2b269bc92ab6b1be434c77a16d4f4113085150de6704b9f1a1df57"
    )
    assert PRODUCTION_AUTHORITY.physical_vectors_sha256 == (
        "68072364166c20364b8d079a08fd67f5008447db54f51b338f3f541eb54b39e5"
    )
    # The frozen shortcut failure starts from anchor position zero and records
    # a reached count of 114.  Canonical component order is by minimum anchor
    # position, not by descending component size.
    assert PRODUCTION_AUTHORITY.initial_component_sizes == (114, 149, 3)
    assert PRODUCTION_AUTHORITY.initial_component_sizes[0] == 114
    assert len(PRODUCTION_AUTHORITY.failed_tree_files) == 14
    assert PRODUCTION_AUTHORITY.close_state_authority.projection_sha256 == (
        "f2bcde0b4cf8b86082abb3bc9b7499c8a9459f1a1df92d8eada28996e332a780"
    )
    assert PRODUCTION_AUTHORITY.final_state_authority.projection_sha256 == (
        "b455b618d29ac807eecead64b3aa8f47bfdee67344dab9cfb566337d148c12ab"
    )
    for authority in (
        PRODUCTION_AUTHORITY.close_state_authority,
        PRODUCTION_AUTHORITY.final_state_authority,
    ):
        projection = authority.projection()
        assert set(projection) == {
            "created_at",
            "dataset",
            "instances",
            "reason",
            "schema_version",
            "stage",
            "state",
            "task_id",
            "updated_at",
        }
        assert projection["updated_at"] == "<MUTABLE>"
        main = projection["instances"]["main"]
        assert set(main) == {
            "adopted",
            "attempt",
            "child_pid",
            "command_sha256",
            "expected_output",
            "failure_class",
            "failure_reason",
            "gpu_colocation_gate",
            "gpu_colocation_gate_sha256",
            "gpu_index",
            "gpu_lock_mode",
            "gpu_memory_reservation_mb",
            "gpu_shared_workload_class",
            "gpu_uuid",
            "heartbeat_at",
            "instance_id",
            "launcher_identity",
            "launcher_pid",
            "log_path",
            "oom_retry_count",
            "required_absolute_output_files",
            "resume_from_checkpoint",
            "resume_source_output",
            "retry_kind",
            "run_id",
            "started_at",
            "state",
            "tmux_session",
            "transient_retry_count",
            "worker_identity",
            "worker_pid",
        }
        assert main["heartbeat_at"] == "<MUTABLE>"
        assert set(main["launcher_identity"]) == {
            "command_sha256",
            "pid",
            "start_ticks",
        }
        assert set(main["worker_identity"]) == {
            "command_sha256",
            "pid",
            "start_ticks",
        }


def test_anchor_component_order_uses_minimum_position_not_descending_size() -> None:
    edges = np.asarray(
        [
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (1, 6),
            (2, 4),
            (3, 5),
            (3, 6),
            (5, 6),
        ],
        dtype=np.intp,
    )

    sizes, labels, degrees, _neighborhoods = _anchor_graph_summary(7, edges)

    assert sizes == (3, 4)
    assert labels == (0, 1, 0, 1, 0, 1, 1)
    assert min(degrees) == 3


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _stable(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _npy(path: Path, value: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    return path


def _file_stat(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


@dataclass
class AuthorityFixture:
    profile: FailedSelectionAuthority
    output: Path
    proc: Path
    control: Path
    namespace: Path
    source_manifest: Path
    controller_root: Path
    close_gate: Path
    close_state: Path
    final_gate: Path
    final_state: Path
    close_root: Path
    final_root: Path
    vector_path: Path
    pair_contract: Path
    pair_semantics_bitmap: Path
    selection: Path


def _build_fixture(tmp_path: Path) -> AuthorityFixture:
    control = tmp_path / "control"
    output_parent = tmp_path / "outputs"
    proc = tmp_path / "proc"
    source_data = tmp_path / "source-data"
    for directory in (control, output_parent, proc, source_data):
        directory.mkdir(parents=True)

    namespace_name = "fixed_namespace"
    controller_id = "fixed_controller"
    close_task_id = "close"
    final_task_id = "final"
    namespace = control / namespace_name
    controller_root = namespace / controller_id
    source_manifest_path = namespace / "manifests" / f"{controller_id}.json"
    close_root = tmp_path / "science" / "close-attempt-0"
    pair_semantics_root = tmp_path / "science" / "pair-semantics-attempt-0"
    final_root = tmp_path / "science" / "final-attempt-0"
    dbscan_root = final_root / "common_recourse/external_memory/dbscan"
    close_root.mkdir(parents=True)
    pair_semantics_root.mkdir(parents=True)
    dbscan_root.mkdir(parents=True)
    (close_root / "PASS").write_bytes(b"PASS\n")

    vectors = np.zeros((6, 4), dtype=np.float32)
    vectors[3:, 0] = np.float32(0.2)
    vector_path = _npy(source_data / "recourse_vectors.npy", vectors)
    distances_path = _npy(
        source_data / "normalized_distances.npy",
        np.full(6, 0.05, dtype=np.float32),
    )
    pairs = np.asarray(
        [(parent, candidate) for candidate in range(3) for parent in range(2)],
        dtype=np.int64,
    )
    pairs_path = _npy(source_data / "pair_indices.npy", pairs)
    pair_manifest = _json(
        source_data / "run_manifest.json",
        {
            "schema_version": PAIR_STORE_SCHEMA,
            "run_complete": True,
            "candidate_major_parent_minor_order": True,
            "row_count": 6,
            "vector_dim": 4,
            "vectors_dtype": "float32",
            "pairs_path": str(pairs_path),
            "pairs_sha256": _sha(pairs_path),
            "vectors_path": str(vector_path),
            "vectors_sha256": _sha(vector_path),
        },
    )
    bitmap_path = _npy(close_root / "close_bitmap.npy", np.ones(6, dtype=np.bool_))
    pair_semantics_bitmap = _npy(
        pair_semantics_root
        / "distance_scan"
        / "close_pair_bitmap.greed.uint8.npy",
        np.ones(6, dtype=np.uint8),
    )
    pair_contract = _json(
        pair_semantics_root / "close_pair_contract.json",
        {
            "schema_version": AIDS_PAIR_SEMANTICS_SCHEMA,
            "status": "PASS",
            "physical_store_rows": 6,
            "logical_close_rows": 6,
            "all_pairs_close": True,
            "pair_order": PAIR_ORDER,
            "pair_orientation": PAIR_ORIENTATION,
            "pair_axis": ["parent_index", "candidate_index"],
            "pair_axis_all_rows_checked": True,
            "pair_axis_mismatch_count": 0,
            "close_bitmap": str(pair_semantics_bitmap),
            "close_bitmap_hash": _sha(pair_semantics_bitmap),
            "normalized_distances": str(distances_path),
            "normalized_distances_sha256": _sha(distances_path),
            "physical_vectors_sha256": _sha(vector_path),
            "theta": 0.1,
            "parent_count": 2,
            "candidate_count": 3,
            "filter_operator": FILTER_OPERATOR,
            "scale_contract": SCALE_CONTRACT,
            "normalized_distance_contract": NORMALIZED_DISTANCE_CONTRACT,
            "distance_checkpoint_hash": "a" * 64,
            "embedding_checkpoint_hash": "b" * 64,
            "source_pair_store_manifest": str(pair_manifest),
            "source_pair_store_manifest_sha256": _sha(pair_manifest),
            "source_mutated": False,
        },
    )
    certificate = _json(
        pair_semantics_root / "all_pairs_close_certificate.json",
        {
            "schema_version": ALL_PAIRS_CLOSE_CERTIFICATE_SCHEMA,
            "status": "PASS",
            "all_pairs_close_proven": True,
            "full_distance_scan_complete": True,
            "official_sample_comparison_pass": True,
            "normalization_audit_pass": True,
            "physical_store_rows": 6,
            "count_distance_le_theta": 6,
            "count_distance_gt_theta": 0,
            "count_distance_eq_theta": 0,
            "theta": 0.1,
            "filter_operator": FILTER_OPERATOR,
            "pair_orientation": PAIR_ORIENTATION,
            "pair_order": PAIR_ORDER,
            "physical_vectors_sha256": _sha(vector_path),
            "normalized_distances_sha256": _sha(distances_path),
            "distance_checkpoint_sha256": "a" * 64,
            "embedding_checkpoint_sha256": "b" * 64,
            "scale_contract": SCALE_CONTRACT,
            "normalized_distance_contract": NORMALIZED_DISTANCE_CONTRACT,
            "approximation_used": False,
        },
    )
    close_identity = {
        "contract": {
            "theta": 0.1,
            "parent_count": 2,
            "candidate_count": 3,
            "distance_checkpoint_sha256": "a" * 64,
            "embedding_checkpoint_sha256": "b" * 64,
            "scale_contract": SCALE_CONTRACT,
            "normalized_distance_contract": NORMALIZED_DISTANCE_CONTRACT,
            "filter_operator": FILTER_OPERATOR,
            "pair_orientation": PAIR_ORIENTATION,
            "chunk_order": PAIR_ORDER,
        },
        "schema_version": CLOSE_PAIR_VIEW_SCHEMA,
        "physical_vectors_path": str(vector_path),
        "physical_vectors_sha256": _sha(vector_path),
        "physical_vectors_stat_identity": _file_stat(vector_path),
        "physical_vectors_shape": [6, 4],
        "physical_vectors_dtype": "float32",
        "normalized_distances_path": str(distances_path),
        "normalized_distances_sha256": _sha(distances_path),
        "normalized_distances_stat_identity": _file_stat(distances_path),
        "normalized_distances_shape": [6],
        "normalized_distances_dtype": "float32",
        "pair_semantics_contract_path": str(pair_contract),
        "pair_semantics_contract_sha256": _sha(pair_contract),
        "pair_semantics_contract_stat_identity": _file_stat(pair_contract),
        "physical_store_rows": 6,
    }
    close_manifest = _json(
        close_root / "close_pair_contract.json",
        {
            "schema_version": CLOSE_PAIR_VIEW_SCHEMA,
            "status": "PASS",
            "run_complete": True,
            "eligible_for_dbscan": True,
            "blocking_reason": None,
            "scientific_identity": close_identity,
            "scientific_identity_sha256": _stable(close_identity),
            "physical_store_is_full_cartesian": True,
            "physical_store_rows": 6,
            "logical_close_rows": 6,
            "dbscan_input_count": 6,
            "all_pairs_close": True,
            "view_storage": "zero_copy_full_cartesian",
            "pairs_storage": "implicit_cartesian_v1",
            "large_vector_copy_materialized": False,
            "recourse_vectors_recomputed": False,
            "filter_operator": FILTER_OPERATOR,
            "pair_orientation": PAIR_ORIENTATION,
            "pair_axis": "col0=parent;col1=candidate",
            "chunk_order": PAIR_ORDER,
            "dbscan_input": "theta_close_recourse_only",
            "theta": 0.1,
            "recourse_vectors_path": str(vector_path),
            "recourse_vectors_sha256": _sha(vector_path),
            "pair_indices_path": None,
            "pair_indices_sha256": _sha(pairs_path),
            "physical_row_indices_path": None,
            "physical_row_indices_sha256": None,
            "recourse_vectors_copied_byte_exact_from_physical_rows": True,
            "recourse_vectors_zero_copy_indexed_from_physical_rows": False,
            "close_bitmap_path": str(bitmap_path),
            "close_bitmap_hash": _sha(bitmap_path),
            "all_pairs_close_certificate_path": str(certificate),
            "all_pairs_close_certificate_sha256": _sha(certificate),
            "approximation_used": False,
        },
    )

    contract = external.ExternalDBSCANContract(
        eps=0.02,
        min_samples=3,
        query_block_size=2,
        checkpoint_interval_blocks=1,
        max_rss_bytes=external._rss_bytes() + 512 * 1024**2,
        expected_sklearn_version=sklearn.__version__,
        shortcut_mode=external.ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
        shortcut_seed_count=3,
        shortcut_failure_cap=10,
        shortcut_query_block_size=2,
        exact_fallback_max_samples=0,
    )
    # This fixture represents the immutable c766 failure produced by the
    # pre-recovery engine.  The current engine correctly continues from this
    # disconnected anchor graph, so asking it to fail here would make the
    # adoption tests depend on reintroducing the historical bug.  Intercept
    # only the newly added recovery hand-off and materialize the exact legacy
    # failure/checkpoint contract with the production helpers instead.
    def legacy_disconnected_failure(**kwargs: object) -> None:
        root = Path(str(kwargs["root"]))
        state_path = Path(str(kwargs["state_path"]))
        identity = kwargs["identity"]
        active_contract = kwargs["contract"]
        anchor_indices = np.asarray(kwargs["anchor_indices"], dtype=np.intp)
        anchor_edges = np.asarray(kwargs["anchor_edges"], dtype=np.intp)
        anchor_rows = [
            [int(value) for value in row]
            for row in kwargs["anchor_rows"]
        ]
        selection_manifest = kwargs["selection_manifest"]
        assert isinstance(identity, dict)
        assert isinstance(active_contract, external.ExternalDBSCANContract)
        assert isinstance(selection_manifest, dict)

        reached = {0}
        frontier = [0]
        while frontier:
            current = frontier.pop()
            for neighbor in anchor_rows[current]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
        failure = external._shortcut_failure(
            root=root,
            identity=identity,
            reason="anchor_epsilon_graph_disconnected",
            num_samples=int(vectors.shape[0]),
            fallback_limit=int(active_contract.exact_fallback_max_samples),
            details={
                "anchor_count": int(len(anchor_indices)),
                "anchor_component_reached_count": int(len(reached)),
                "anchor_edge_count": int(len(anchor_edges)),
                "anchor_neighborhoods_sha256": hashlib.sha256(
                    json.dumps(anchor_rows, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            },
        )
        state = external._load_checkpoint(state_path)
        ledgers = external._load_progress_ledgers(
            state,
            identity=identity,
            num_samples=int(vectors.shape[0]),
        )
        selection_identity = selection_manifest["selection_identity"]
        selection_path = root / "adaptive_anchor_selection.json"
        extra = external._progress_checkpoint_extra(ledgers, identity=identity)
        extra.update(
            {
                "adaptive_selection_manifest_path": str(selection_path),
                "adaptive_selection_manifest_sha256": external._sha256_file(
                    selection_path
                ),
                "selected_anchor_indices_sha256": selection_identity[
                    "selected_anchor_indices_sha256"
                ],
                "shortcut_failure_path": failure["path"],
                "shortcut_failure_sha256": failure["sha256"],
                "shortcut_approximation_used": False,
            }
        )
        external._checkpoint(
            state_path,
            identity=identity,
            phase="shortcut_blocked",
            next_offset=0,
            peak_rss_bytes=int(kwargs["peak_rss_bytes"]),
            extra=extra,
        )
        raise external.ExternalMemoryDBSCANError(
            "EXACT_DBSCAN_COMPLEXITY_BLOCKED:"
            "reason=anchor_epsilon_graph_disconnected"
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            external,
            "_fit_adaptive_disconnected_component_recovery",
            legacy_disconnected_failure,
        )
        with pytest.raises(
            external.ExternalMemoryDBSCANError,
            match="anchor_epsilon_graph_disconnected",
        ):
            external.fit_external_memory_dbscan(
                vectors_path=vector_path,
                work_dir=dbscan_root,
                contract=contract,
            )
    selection = dbscan_root / "adaptive_anchor_selection.json"
    checkpoint = dbscan_root / "checkpoint.json"
    failure = dbscan_root / "shortcut_failure.json"
    failure_indices = dbscan_root / "adaptive_first_pass_failure_indices.npy"
    anchor_indices = dbscan_root / "shortcut_anchor_indices.npy"
    anchor_rows = dbscan_root / "adaptive_selected_anchor_rows.npy"
    anchor_edges = dbscan_root / "shortcut_anchor_edges.npy"

    tasks = [
        {"id": close_task_id, "expected_output": str(close_root)},
        {"id": final_task_id, "expected_output": str(final_root)},
    ]
    source_manifest = _json(
        source_manifest_path,
        {
            "schema_version": 1,
            "controller_id": controller_id,
            "paper_frozen": True,
            "tasks": tasks,
        },
    )
    controller_root.mkdir(parents=True)
    snapshot_payload = json.loads(source_manifest.read_text())
    snapshot_payload.update(
        {
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": _sha(source_manifest),
        }
    )
    _json(controller_root / "controller_manifest.json", snapshot_payload)

    def terminal(
        task_id: str,
        status: str,
        root: Path,
        *,
        worker_pid: int,
        child_pid: int,
    ) -> tuple[Path, Path, TaskStateAuthority]:
        run_id = f"{task_id}-main-0"
        run = {
            "instance_id": "main",
            "attempt": 0,
            "state": status,
            "run_id": run_id,
            "expected_output": str(root),
        }
        gate = _json(
            controller_root / "tasks" / task_id / "gate.json",
            {
                "schema_version": 1,
                "task_id": task_id,
                "status": status,
                "runs": [run],
            },
        )
        command_sha256 = ("d" if status == "PASS" else "e") * 64
        identity_sha256 = ("b" if status == "PASS" else "c") * 64
        failure_reason = None if status == "PASS" else "fixture semantic failure"
        failure_class = None if status == "PASS" else "SEMANTIC"
        instance = {
            **run,
            "adopted": False,
            "command_sha256": command_sha256,
            "child_pid": child_pid,
            "failure_class": failure_class,
            "launcher_pid": worker_pid,
            "launcher_identity": {
                "pid": worker_pid,
                "start_ticks": 77,
                "command_sha256": identity_sha256,
            },
            "worker_pid": worker_pid,
            "worker_identity": {
                "pid": worker_pid,
                "start_ticks": 77,
                "command_sha256": identity_sha256,
            },
            "gpu_index": None,
            "gpu_uuid": None,
            "gpu_lock_mode": "exclusive",
            "gpu_memory_reservation_mb": 0,
            "gpu_shared_workload_class": None,
            "gpu_colocation_gate": None,
            "gpu_colocation_gate_sha256": None,
            "log_path": "/fixture/controller.log",
            "required_absolute_output_files": [],
            "started_at": "2026-08-24T22:48:39+00:00",
            "tmux_session": None,
            "heartbeat_at": "2026-08-25T00:00:00+00:00",
            "oom_retry_count": 0,
            "transient_retry_count": 0,
            "retry_kind": None,
            "resume_from_checkpoint": None,
            "resume_source_output": None,
            "failure_reason": failure_reason,
        }
        reason = None if status == "PASS" else "main:SEMANTIC"
        state = _json(
            controller_root / "tasks" / task_id / "state.json",
            {
                "schema_version": 1,
                "dataset": "aids",
                "stage": "AM_COMRECGC_HELDOUT_EVAL",
                "task_id": task_id,
                "state": status,
                "reason": reason,
                "created_at": "2026-08-24T19:14:15+00:00",
                "updated_at": "2026-08-25T00:00:00+00:00",
                "instances": {"main": instance},
            },
        )
        authority_without_hash = TaskStateAuthority(
            schema_version=1,
            dataset="aids",
            stage="AM_COMRECGC_HELDOUT_EVAL",
            task_id=task_id,
            state=status,
            reason=reason,
            instance_id="main",
            attempt=0,
            run_id=run_id,
            expected_output=str(root),
            command_sha256=command_sha256,
            child_pid=child_pid,
            failure_class=failure_class,
            launcher_pid=worker_pid,
            launcher_start_ticks=77,
            launcher_command_sha256=identity_sha256,
            worker_pid=worker_pid,
            worker_start_ticks=77,
            worker_command_sha256=identity_sha256,
            failure_reason_length=(
                None if failure_reason is None else len(failure_reason)
            ),
            failure_reason_sha256=(
                None
                if failure_reason is None
                else hashlib.sha256(failure_reason.encode()).hexdigest()
            ),
            projection_sha256="0" * 64,
        )
        authority = replace(
            authority_without_hash,
            projection_sha256=_adoption_stable_hash(
                authority_without_hash.projection()
            ),
        )
        return gate, state, authority

    close_gate, close_state, close_state_authority = terminal(
        close_task_id,
        "PASS",
        close_root,
        worker_pid=12344,
        child_pid=22344,
    )
    final_gate, final_state, final_state_authority = terminal(
        final_task_id,
        "FAILED",
        final_root,
        worker_pid=12345,
        child_pid=22345,
    )
    for relative in (
        "FAILED.json",
        "common_recourse/external_memory/pair_store_adoption/run_manifest.json",
        "continuation_resume_contract.json",
        "generation_adoption_manifest.json",
        "stage_checkpoints/common_recourse.json",
        "stage_state.json",
        "upstream_checkout_audit.json",
    ):
        _json(final_root / relative, {"fixture": relative})
    failed_tree_files = tuple(
        (path.relative_to(final_root).as_posix(), _sha(path))
        for path in sorted(
            (value for value in final_root.rglob("*") if value.is_file()),
            key=str,
        )
    )
    profile = FailedSelectionAuthority(
        control_root=str(control),
        output_parent=str(output_parent),
        proc_root=str(proc),
        namespace=namespace_name,
        controller_id=controller_id,
        close_task_id=close_task_id,
        final_task_id=final_task_id,
        close_state_authority=close_state_authority,
        final_state_authority=final_state_authority,
        failed_tree_files=failed_tree_files,
        controller_manifest_sha256=_sha(source_manifest),
        close_gate_sha256=_sha(close_gate),
        close_manifest_sha256=_sha(close_manifest),
        final_gate_sha256=_sha(final_gate),
        checkpoint_sha256=_sha(checkpoint),
        shortcut_failure_sha256=_sha(failure),
        selection_manifest_sha256=_sha(selection),
        failure_indices_sha256=_sha(failure_indices),
        anchor_indices_sha256=_sha(anchor_indices),
        anchor_rows_sha256=_sha(anchor_rows),
        anchor_edges_sha256=_sha(anchor_edges),
        physical_pairs_sha256=_sha(pairs_path),
        physical_vectors_sha256=_sha(vector_path),
        physical_rows=6,
        vector_features=4,
        parent_count=2,
        candidate_count=3,
        failure_count=3,
        anchor_count=6,
        anchor_edge_count=6,
        seed_count=3,
        initial_component_sizes=(3, 3),
    )
    return AuthorityFixture(
        profile=profile,
        output=output_parent / "adoption",
        proc=proc,
        control=control,
        namespace=namespace,
        source_manifest=source_manifest,
        controller_root=controller_root,
        close_gate=close_gate,
        close_state=close_state,
        final_gate=final_gate,
        final_state=final_state,
        close_root=close_root,
        final_root=final_root,
        vector_path=vector_path,
        pair_contract=pair_contract,
        pair_semantics_bitmap=pair_semantics_bitmap,
        selection=selection,
    )


def _adopt(case: AuthorityFixture) -> dict:
    return _create_or_validate_with_profile(
        output_dir=case.output,
        profile=case.profile,
        proc_root=case.proc,
    )


def _rebind_pair_semantics_bitmap(
    case: AuthorityFixture,
    bitmap_reference: Path,
) -> None:
    pair_contract = json.loads(case.pair_contract.read_text(encoding="utf-8"))
    pair_contract["close_bitmap"] = str(bitmap_reference)
    pair_contract["close_bitmap_hash"] = _sha(bitmap_reference)
    _json(case.pair_contract, pair_contract)

    close_manifest_path = case.close_root / "close_pair_contract.json"
    close_manifest = json.loads(close_manifest_path.read_text(encoding="utf-8"))
    identity = close_manifest["scientific_identity"]
    identity["pair_semantics_contract_sha256"] = _sha(case.pair_contract)
    identity["pair_semantics_contract_stat_identity"] = _file_stat(
        case.pair_contract
    )
    close_manifest["scientific_identity_sha256"] = _stable(identity)
    _json(close_manifest_path, close_manifest)
    case.profile = replace(
        case.profile,
        close_manifest_sha256=_sha(close_manifest_path),
    )


def _rebind_materialized_bitmap(case: AuthorityFixture) -> None:
    bitmap_path = case.close_root / "close_bitmap.npy"
    close_manifest_path = case.close_root / "close_pair_contract.json"
    close_manifest = json.loads(close_manifest_path.read_text(encoding="utf-8"))
    close_manifest["close_bitmap_hash"] = _sha(bitmap_path)
    _json(close_manifest_path, close_manifest)
    case.profile = replace(
        case.profile,
        close_manifest_sha256=_sha(close_manifest_path),
    )


def _resign_shortcut_failure_reached_count(
    case: AuthorityFixture,
    reached_count: int,
) -> None:
    dbscan_root = case.final_root / "common_recourse/external_memory/dbscan"
    failure_path = dbscan_root / "shortcut_failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    failure["details"]["anchor_component_reached_count"] = reached_count
    _json(failure_path, failure)

    checkpoint_path = dbscan_root / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["shortcut_failure_sha256"] = _sha(failure_path)
    checkpoint.pop("checkpoint_payload_sha256")
    checkpoint["checkpoint_payload_sha256"] = external._stable_hash(checkpoint)
    _json(checkpoint_path, checkpoint)

    failed_tree_files = tuple(
        (path.relative_to(case.final_root).as_posix(), _sha(path))
        for path in sorted(
            (value for value in case.final_root.rglob("*") if value.is_file()),
            key=str,
        )
    )
    case.profile = replace(
        case.profile,
        shortcut_failure_sha256=_sha(failure_path),
        checkpoint_sha256=_sha(checkpoint_path),
        failed_tree_files=failed_tree_files,
    )


def _rewrite_signed_terminal(case: AuthorityFixture, payload: dict) -> None:
    receipt = case.output / RECEIPT_NAME
    receipt.chmod(0o644)
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    receipt.write_bytes(raw)
    prepared = case.output / READY_PREPARED_NAME
    prepared.chmod(0o644)
    prepared.write_bytes(_ready_marker_bytes(raw))
    prepared.chmod(0o444)
    receipt.chmod(0o444)


def _proc_entry(
    proc: Path,
    pid: int,
    *,
    start_ticks: int,
    state: str = "S",
    cmdline: bytes = b"unrelated\0process\0",
) -> Path:
    root = proc / str(pid)
    root.mkdir(parents=True, exist_ok=True)
    fields = ["0"] * 20
    # _proc_generation indexes Linux field 22 as fields[19] after the state.
    fields[18] = str(start_ticks)
    (root / "stat").write_text(f"{pid} (fixture) {state} " + " ".join(fields))
    (root / "cmdline").write_bytes(cmdline)
    return root


def test_failed_selection_adoption_is_recovery_only_and_idempotent(
    tmp_path: Path,
) -> None:
    case = _build_fixture(tmp_path)
    assert case.pair_semantics_bitmap.parent.parent == case.pair_contract.parent
    materialized_bitmap = case.close_root / "close_bitmap.npy"
    assert np.load(case.pair_semantics_bitmap, allow_pickle=False).dtype == np.uint8
    assert np.load(materialized_bitmap, allow_pickle=False).dtype == np.bool_
    assert _sha(case.pair_semantics_bitmap) != _sha(materialized_bitmap)
    first = _adopt(case)
    second = _adopt(case)
    assert first == second
    assert first["source_final_status"] == "FAILED"
    assert first["status"] == "RECOVERY_ONLY_READY"
    assert first["artifact_kind"] == (
        "aids_c766_failed_selection_recovery_evidence_v3"
    )
    assert first["failed_evidence_adopted_for_recovery_only"] is True
    assert first["ordinary_pass_dependency_eligible"] is False
    assert first["scientific_result_pass"] is False
    assert first["failed_selection"]["dbscan_partition_proven"] is False
    assert first["failed_selection"]["initial_component_sizes"] == [3, 3]
    shortcut_failure = json.loads(
        Path(first["failed_selection"]["failure_artifact"]).read_text(
            encoding="utf-8"
        )
    )
    assert shortcut_failure["details"]["anchor_component_reached_count"] == 3
    assert shortcut_failure["details"]["anchor_component_reached_count"] == (
        first["failed_selection"]["initial_component_sizes"][0]
    )
    assert first["failed_selection"]["unique_seed_component"] is True
    assert first["failed_selection"]["seed_component_ids"] == [0, 0, 0]
    assert first["failed_selection"]["seed_component_size"] == 3
    assert first["failed_selection"]["anchor_degree_including_self_min"] == 3
    assert len(
        first["failed_selection"][
            "initial_component_canonical_labels_sha256"
        ]
    ) == 64
    assert len(
        first["failed_selection"]["anchor_degrees_including_self_sha256"]
    ) == 64
    assert set(first["terminal_reopen_task_state_observations"]) == {
        "close",
        "final",
    }
    assert all(
        len(first["terminal_reopen_task_state_observations"][name]) == 2
        for name in ("close", "final")
    )
    marker_lines = (case.output / READY_NAME).read_text().splitlines()
    assert marker_lines[0] == (
        "AIDS_C766_FAILED_SELECTION_RECOVERY_EVIDENCE_READY_V3"
    )
    assert marker_lines[1].startswith("receipt_sha256=")
    assert len(marker_lines[1]) == len("receipt_sha256=") + 64
    assert marker_lines[1].split("=", 1)[1] == _sha(
        case.output / "failed_selection_adoption_receipt.json"
    )
    assert not (case.output / "PASS").exists()


def test_shortcut_failure_reached_count_binds_canonical_component_zero(
    tmp_path: Path,
) -> None:
    case = _build_fixture(tmp_path)
    assert case.profile.initial_component_sizes == (3, 3)
    _resign_shortcut_failure_reached_count(case, 4)

    with pytest.raises(
        FailedSelectionAdoptionError,
        match="disconnected shortcut failure changed",
    ):
        _adopt(case)
    assert not (case.output / RECEIPT_NAME).exists()
    assert not (case.output / READY_NAME).exists()


@pytest.mark.parametrize("dtype", [np.bool_, np.int16])
def test_pair_semantics_scan_bitmap_requires_uint8(
    tmp_path: Path,
    dtype: type[np.generic],
) -> None:
    case = _build_fixture(tmp_path)
    _npy(case.pair_semantics_bitmap, np.ones(6, dtype=dtype))
    _rebind_pair_semantics_bitmap(case, case.pair_semantics_bitmap)

    with pytest.raises(
        FailedSelectionAdoptionError,
        match="pair-semantics scan bitmap schema changed",
    ):
        _adopt(case)
    assert not (case.output / RECEIPT_NAME).exists()
    assert not (case.output / READY_NAME).exists()


def test_materialized_close_bitmap_requires_bool(tmp_path: Path) -> None:
    case = _build_fixture(tmp_path)
    bitmap_path = case.close_root / "close_bitmap.npy"
    _npy(bitmap_path, np.ones(6, dtype=np.uint8))
    _rebind_materialized_bitmap(case)

    with pytest.raises(
        FailedSelectionAdoptionError,
        match="close bitmap schema changed",
    ):
        _adopt(case)
    assert not (case.output / RECEIPT_NAME).exists()
    assert not (case.output / READY_NAME).exists()


def test_pair_semantics_scan_bitmap_rejects_non_binary_values(
    tmp_path: Path,
) -> None:
    case = _build_fixture(tmp_path)
    _npy(
        case.pair_semantics_bitmap,
        np.asarray([1, 1, 1, 2, 1, 1], dtype=np.uint8),
    )
    _rebind_pair_semantics_bitmap(case, case.pair_semantics_bitmap)

    with pytest.raises(
        FailedSelectionAdoptionError,
        match="scan bitmap contains non-binary values",
    ):
        _adopt(case)
    assert not (case.output / RECEIPT_NAME).exists()
    assert not (case.output / READY_NAME).exists()


def test_pair_semantics_scan_bitmap_requires_bounded_row_equivalence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_fixture(tmp_path)
    _npy(
        case.pair_semantics_bitmap,
        np.asarray([1, 1, 1, 1, 0, 1], dtype=np.uint8),
    )
    _rebind_pair_semantics_bitmap(case, case.pair_semantics_bitmap)
    from src.baselines.comrecgc import failed_selection_adoption as adoption

    monkeypatch.setattr(adoption, "PAIR_BITMAP_COMPARE_BLOCK_ROWS", 2)
    with pytest.raises(
        FailedSelectionAdoptionError,
        match="not row-wise equivalent",
    ):
        _adopt(case)
    assert not (case.output / RECEIPT_NAME).exists()
    assert not (case.output / READY_NAME).exists()


def test_equivalent_false_bitmap_rows_contradict_all_pairs_close(
    tmp_path: Path,
) -> None:
    case = _build_fixture(tmp_path)
    scan = np.asarray([1, 1, 1, 0, 1, 1], dtype=np.uint8)
    _npy(case.pair_semantics_bitmap, scan)
    _rebind_pair_semantics_bitmap(case, case.pair_semantics_bitmap)
    bitmap_path = case.close_root / "close_bitmap.npy"
    _npy(bitmap_path, scan.astype(np.bool_))
    _rebind_materialized_bitmap(case)

    with pytest.raises(
        FailedSelectionAdoptionError,
        match="contradicts the all-pairs-close authority",
    ):
        _adopt(case)
    assert not (case.output / RECEIPT_NAME).exists()
    assert not (case.output / READY_NAME).exists()


@pytest.mark.parametrize("artifact", ["scan", "materialized"])
def test_each_close_bitmap_is_bound_to_its_own_contract_hash(
    tmp_path: Path,
    artifact: str,
) -> None:
    case = _build_fixture(tmp_path)
    if artifact == "scan":
        _npy(
            case.pair_semantics_bitmap,
            np.asarray([1, 1, 1, 0, 1, 1], dtype=np.uint8),
        )
    else:
        _npy(
            case.close_root / "close_bitmap.npy",
            np.asarray([True, True, True, False, True, True], dtype=np.bool_),
        )

    with pytest.raises(FailedSelectionAdoptionError, match="SHA256 mismatch"):
        _adopt(case)
    assert not (case.output / RECEIPT_NAME).exists()
    assert not (case.output / READY_NAME).exists()


@pytest.mark.parametrize(
    ("variant", "message"),
    [
        ("sibling", "escaped its authority root"),
        ("file_symlink", "symlink component"),
        ("directory_alias", "symlink component"),
        ("dotdot", "not one physical path"),
    ],
)
def test_pair_semantics_bitmap_rejects_nonphysical_descendant_references(
    tmp_path: Path,
    variant: str,
    message: str,
) -> None:
    case = _build_fixture(tmp_path)
    contract_root = case.pair_contract.parent
    if variant == "sibling":
        reference = contract_root.parent / "sibling-close-bitmap.npy"
        shutil.copy2(case.pair_semantics_bitmap, reference)
    elif variant == "file_symlink":
        reference = contract_root / "close-bitmap-link.npy"
        reference.symlink_to(case.pair_semantics_bitmap)
    elif variant == "directory_alias":
        alias = contract_root / "distance-scan-alias"
        alias.symlink_to(case.pair_semantics_bitmap.parent, target_is_directory=True)
        reference = alias / case.pair_semantics_bitmap.name
    elif variant == "dotdot":
        reference = (
            case.pair_semantics_bitmap.parent
            / ".."
            / case.pair_semantics_bitmap.parent.name
            / case.pair_semantics_bitmap.name
        )
    else:  # pragma: no cover - parameter list is closed above.
        raise AssertionError(variant)
    _rebind_pair_semantics_bitmap(case, reference)

    with pytest.raises(FailedSelectionAdoptionError, match=message):
        _adopt(case)
    assert not case.output.exists()
    lock = case.output.parent / (
        f".{case.output.name}.failed-selection-adoption.lock"
    )
    assert not lock.exists()


def test_nested_pair_semantics_bitmap_replacement_is_rejected_on_reopen(
    tmp_path: Path,
) -> None:
    case = _build_fixture(tmp_path)
    receipt = _adopt(case)
    expected_sha256 = _sha(case.pair_semantics_bitmap)
    payload = case.pair_semantics_bitmap.read_bytes()
    original_inode = case.pair_semantics_bitmap.stat().st_ino
    case.pair_semantics_bitmap.unlink()
    case.pair_semantics_bitmap.write_bytes(payload)
    assert case.pair_semantics_bitmap.stat().st_ino != original_inode
    assert _sha(case.pair_semantics_bitmap) == expected_sha256

    with pytest.raises(
        FailedSelectionAdoptionError,
        match="terminal adoption evidence changed: source_artifacts",
    ):
        _adopt(case)
    assert receipt["status"] == "RECOVERY_ONLY_READY"
    assert not (case.output / READY_NAME).exists()


def test_source_manifest_requires_real_absent_taste_flag_even_if_rehashed(
    tmp_path: Path,
) -> None:
    case = _build_fixture(tmp_path)
    original = json.loads(case.source_manifest.read_text(encoding="utf-8"))
    assert "run_tastemolnet" not in original

    injected = {**original, "run_tastemolnet": 0}
    _json(case.source_manifest, injected)
    rebound_sha256 = _sha(case.source_manifest)
    snapshot = {
        **injected,
        "source_manifest": str(case.source_manifest),
        "source_manifest_sha256": rebound_sha256,
    }
    _json(case.controller_root / "controller_manifest.json", snapshot)
    case.profile = replace(
        case.profile,
        controller_manifest_sha256=rebound_sha256,
    )

    with pytest.raises(FailedSelectionAdoptionError, match="source controller"):
        _adopt(case)
    assert not (case.output / READY_NAME).exists()


def test_pid_reuse_is_exit_evidence_but_live_original_generation_blocks(
    tmp_path: Path,
) -> None:
    reused = _build_fixture(tmp_path / "reused")
    _proc_entry(reused.proc, 12345, start_ticks=88)
    receipt = _adopt(reused)
    assert receipt["process_exit"]["worker_observation"] == (
        "ORIGINAL_GENERATION_EXITED_PID_REUSED"
    )

    live = _build_fixture(tmp_path / "live")
    _proc_entry(live.proc, 12345, start_ticks=77)
    with pytest.raises(FailedSelectionAdoptionError, match="still alive"):
        _adopt(live)
    assert not (live.output / READY_NAME).exists()

    live_child = _build_fixture(tmp_path / "live-child")
    _proc_entry(live_child.proc, 22345, start_ticks=999)
    with pytest.raises(FailedSelectionAdoptionError, match="child PID is still live"):
        _adopt(live_child)
    assert not (live_child.output / READY_NAME).exists()


def test_terminal_reopen_allows_only_safe_dynamic_proc_observation_changes(
    tmp_path: Path,
) -> None:
    case = _build_fixture(tmp_path / "absent-to-reused")
    receipt = _adopt(case)
    assert receipt["process_exit"]["worker_observation"] == (
        "ORIGINAL_GENERATION_EXITED_PID_ABSENT"
    )
    assert receipt["process_exit"]["child_observation"] == (
        "RECORDED_CHILD_PID_ABSENT"
    )
    _proc_entry(case.proc, 12345, start_ticks=88)
    _proc_entry(case.proc, 22345, start_ticks=999, state="Z")
    assert _adopt(case) == receipt

    churn = _build_fixture(tmp_path / "zombie-to-absent")
    _proc_entry(churn.proc, 12345, start_ticks=77, state="Z")
    _proc_entry(churn.proc, 22345, start_ticks=999, state="Z")
    zombie_receipt = _adopt(churn)
    assert zombie_receipt["process_exit"]["worker_observation"] == (
        "ORIGINAL_GENERATION_EXITED_ZOMBIE"
    )
    shutil.rmtree(churn.proc / "12345")
    shutil.rmtree(churn.proc / "22345")
    assert _adopt(churn) == zombie_receipt


def test_writable_source_fd_blocks_without_signalling(tmp_path: Path) -> None:
    case = _build_fixture(tmp_path)
    process = case.proc / "77"
    (process / "fd").mkdir(parents=True)
    (process / "fdinfo").mkdir()
    (process / "fd" / "4").symlink_to(case.vector_path)
    (process / "fdinfo" / "4").write_text("flags:\t0100001\n")
    with pytest.raises(FailedSelectionAdoptionError, match="writable process"):
        _adopt(case)
    assert not (case.output / READY_NAME).exists()


def test_wrong_gate_status_and_output_closure_fail_closed(tmp_path: Path) -> None:
    status_case = _build_fixture(tmp_path / "status")
    payload = json.loads(status_case.final_gate.read_text())
    payload["status"] = "PASS"
    _json(status_case.final_gate, payload)
    status_case.profile = replace(
        status_case.profile,
        final_gate_sha256=_sha(status_case.final_gate),
    )
    with pytest.raises(FailedSelectionAdoptionError, match="exact FAILED"):
        _adopt(status_case)

    path_case = _build_fixture(tmp_path / "path")
    payload = json.loads(path_case.final_gate.read_text())
    payload["runs"][0]["expected_output"] = str(path_case.final_root / "other")
    _json(path_case.final_gate, payload)
    path_case.profile = replace(
        path_case.profile,
        final_gate_sha256=_sha(path_case.final_gate),
    )
    with pytest.raises(FailedSelectionAdoptionError, match="output/attempt closure"):
        _adopt(path_case)

    failure_case = _build_fixture(tmp_path / "failure-class")
    payload = json.loads(failure_case.final_state.read_text())
    payload["instances"]["main"]["failure_class"] = "EXECUTION"
    _json(failure_case.final_state, payload)
    with pytest.raises(FailedSelectionAdoptionError, match="projection changed"):
        _adopt(failure_case)


def test_copied_controller_tree_in_fake_namespace_is_not_authority(
    tmp_path: Path,
) -> None:
    case = _build_fixture(tmp_path)
    fake = case.control / "fake_namespace"
    shutil.copytree(case.namespace, fake)
    shutil.rmtree(case.namespace)
    with pytest.raises(FailedSelectionAdoptionError, match="fixed AutoDL namespace"):
        _adopt(case)


def test_selection_tamper_and_symlink_are_rejected(tmp_path: Path) -> None:
    tamper = _build_fixture(tmp_path / "tamper")
    tamper.selection.write_text(tamper.selection.read_text() + " ")
    with pytest.raises(FailedSelectionAdoptionError, match="SHA256 mismatch"):
        _adopt(tamper)

    linked = _build_fixture(tmp_path / "linked")
    backup = linked.selection.parent / "selection-backup.json"
    shutil.copy2(linked.selection, backup)
    linked.selection.unlink()
    linked.selection.symlink_to(backup)
    with pytest.raises(FailedSelectionAdoptionError, match="symlink"):
        _adopt(linked)

    injected = _build_fixture(tmp_path / "injected")
    _json(
        injected.final_root
        / "common_recourse/external_memory/dbscan/cluster_partition.json",
        {"status": "PASS", "dbscan_partition_proven": True},
    )
    with pytest.raises(FailedSelectionAdoptionError, match="inventory changed"):
        _adopt(injected)


def test_partial_output_and_replaced_lock_inode_never_reopen(tmp_path: Path) -> None:
    partial = _build_fixture(tmp_path / "partial")
    _adopt(partial)
    (partial.output / READY_NAME).unlink()
    with pytest.raises((FailedSelectionAdoptionError, FileNotFoundError)):
        _adopt(partial)

    replaced = _build_fixture(tmp_path / "replaced")
    _adopt(replaced)
    lock = replaced.output.parent / (
        f".{replaced.output.name}.failed-selection-adoption.lock"
    )
    lock.unlink()
    lock.write_bytes(b"replacement")
    with pytest.raises(FailedSelectionAdoptionError, match="lock identity"):
        _adopt(replaced)
    assert (replaced.output / READY_NAME).exists()


def test_stale_unowned_lock_recovers_and_parent_symlink_writes_nothing(
    tmp_path: Path,
) -> None:
    stale = _build_fixture(tmp_path / "stale")
    lock = stale.output.parent / f".{stale.output.name}.failed-selection-adoption.lock"
    lock.write_bytes(b"")
    assert _adopt(stale)["status"] == "RECOVERY_ONLY_READY"

    escaped = _build_fixture(tmp_path / "escaped")
    real = escaped.output.parent / "real"
    real.mkdir()
    alias = escaped.output.parent / "alias"
    alias.symlink_to(real, target_is_directory=True)
    bad_output = alias / "adoption"
    with pytest.raises(FailedSelectionAdoptionError, match="direct fresh child"):
        _create_or_validate_with_profile(
            output_dir=bad_output,
            profile=escaped.profile,
            proc_root=escaped.proc,
        )
    assert not (alias / ".adoption.failed-selection-adoption.lock").exists()

    linked_lock = _build_fixture(tmp_path / "linked-lock")
    target = linked_lock.output.parent / "lock-target"
    target.write_bytes(b"")
    lock = linked_lock.output.parent / (
        f".{linked_lock.output.name}.failed-selection-adoption.lock"
    )
    lock.symlink_to(target)
    with pytest.raises(FailedSelectionAdoptionError, match="lock is a symlink"):
        _adopt(linked_lock)
    assert not linked_lock.output.exists()


def test_output_must_be_direct_disjoint_fresh_child_before_any_write(
    tmp_path: Path,
) -> None:
    nested = _build_fixture(tmp_path / "nested")
    nested.profile = replace(
        nested.profile,
        output_parent=str(nested.vector_path.parent),
    )
    hostile = nested.vector_path.parent / "inside-pair-store"
    with pytest.raises(FailedSelectionAdoptionError, match="overlaps source"):
        _create_or_validate_with_profile(
            output_dir=hostile,
            profile=nested.profile,
            proc_root=nested.proc,
        )
    assert not hostile.exists()
    assert not (
        hostile.parent / f".{hostile.name}.failed-selection-adoption.lock"
    ).exists()

    indirect = _build_fixture(tmp_path / "indirect")
    child_parent = indirect.output.parent / "nested-parent"
    child_parent.mkdir()
    with pytest.raises(FailedSelectionAdoptionError, match="direct fresh child"):
        _create_or_validate_with_profile(
            output_dir=child_parent / "child",
            profile=indirect.profile,
            proc_root=indirect.proc,
        )
    assert not (child_parent / "child").exists()


@pytest.mark.parametrize("replace_on_inspection", [1, 2])
def test_named_lock_or_output_replacement_during_two_inspections_is_caught(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replace_on_inspection: int,
) -> None:
    case = _build_fixture(tmp_path)
    from src.baselines.comrecgc import failed_selection_adoption as adoption

    original = adoption._inspect_authority
    calls = 0

    def replacing_inspection(**kwargs):
        nonlocal calls
        calls += 1
        result = original(**kwargs)
        if calls == replace_on_inspection:
            lock = case.output.parent / (
                f".{case.output.name}.failed-selection-adoption.lock"
            )
            lock.unlink()
            lock.write_bytes(b"hostile replacement")
        return result

    monkeypatch.setattr(adoption, "_inspect_authority", replacing_inspection)
    with pytest.raises(FailedSelectionAdoptionError, match="lock inode changed"):
        _adopt(case)
    assert calls >= replace_on_inspection
    assert not (case.output / READY_NAME).exists()


@pytest.mark.parametrize("replace_on_inspection", [1, 2])
def test_output_replacement_during_two_authority_inspections_is_caught(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replace_on_inspection: int,
) -> None:
    case = _build_fixture(tmp_path)
    from src.baselines.comrecgc import failed_selection_adoption as adoption

    original = adoption._inspect_authority
    calls = 0
    displaced = case.output.parent / "displaced-output"

    def replacing_inspection(**kwargs):
        nonlocal calls
        calls += 1
        result = original(**kwargs)
        if calls == replace_on_inspection:
            case.output.rename(displaced)
            case.output.mkdir()
        return result

    monkeypatch.setattr(adoption, "_inspect_authority", replacing_inspection)
    with pytest.raises(FailedSelectionAdoptionError, match="output.*inode changed"):
        _adopt(case)
    assert calls >= replace_on_inspection
    assert not (case.output / READY_NAME).exists()
    assert not (displaced / READY_NAME).exists()


def test_state_projection_allows_heartbeat_churn_but_rejects_pid_forgery(
    tmp_path: Path,
) -> None:
    mutable = _build_fixture(tmp_path / "mutable")
    payload = json.loads(mutable.final_state.read_text())
    payload["updated_at"] = "2026-08-25T00:01:00+00:00"
    payload["instances"]["main"]["heartbeat_at"] = (
        "2026-08-25T00:01:00+00:00"
    )
    _json(mutable.final_state, payload)
    receipt = _adopt(mutable)
    observed = receipt["task_state_observations"]["final"]
    assert all(
        row["projection_sha256"]
        == mutable.profile.final_state_authority.projection_sha256
        for row in observed
    )

    forged = _build_fixture(tmp_path / "forged")
    _proc_entry(forged.proc, 12345, start_ticks=77)
    payload = json.loads(forged.final_state.read_text())
    instance = payload["instances"]["main"]
    instance["worker_pid"] = 54321
    instance["worker_identity"] = {
        "pid": 54321,
        "start_ticks": 999,
        "command_sha256": "f" * 64,
    }
    _json(forged.final_state, payload)
    with pytest.raises(FailedSelectionAdoptionError, match="projection changed"):
        _adopt(forged)
    assert not (forged.output / READY_NAME).exists()


def test_state_bytes_may_change_between_projection_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_fixture(tmp_path)
    from src.baselines.comrecgc import failed_selection_adoption as adoption

    original = adoption._validate_failed_selection
    changed = False

    def change_heartbeat(**kwargs):
        nonlocal changed
        result = original(**kwargs)
        if not changed:
            payload = json.loads(case.final_state.read_text())
            payload["updated_at"] = "2026-08-25T00:02:00+00:00"
            payload["instances"]["main"]["heartbeat_at"] = (
                "2026-08-25T00:02:00+00:00"
            )
            _json(case.final_state, payload)
            changed = True
        return result

    monkeypatch.setattr(adoption, "_validate_failed_selection", change_heartbeat)
    receipt = _adopt(case)
    observations = receipt["task_state_observations"]["final"]
    assert observations[0]["observed_sha256"] != observations[1]["observed_sha256"]
    assert observations[0]["projection_sha256"] == observations[1][
        "projection_sha256"
    ]


@pytest.mark.parametrize(
    ("mutator", "pattern"),
    [
        (lambda payload: payload.pop("dataset"), "schema changed"),
        (
            lambda payload: payload["instances"]["main"].pop("launcher_identity"),
            "task-state",
        ),
        (
            lambda payload: payload["instances"]["main"].update(
                {"command_sha256": "f" * 64}
            ),
            "projection changed",
        ),
        (
            lambda payload: payload["instances"]["main"].update(
                {"attempt": False}
            ),
            "strict integer",
        ),
    ],
)
def test_state_projection_missing_or_tampered_fields_fail_closed(
    tmp_path: Path,
    mutator,
    pattern: str,
) -> None:
    case = _build_fixture(tmp_path)
    payload = json.loads(case.final_state.read_text())
    mutator(payload)
    _json(case.final_state, payload)
    with pytest.raises(FailedSelectionAdoptionError, match=pattern):
        _adopt(case)
    assert not (case.output / READY_NAME).exists()


def test_namespace_rename_plus_byte_copy_during_discovery_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_fixture(tmp_path)
    from src.baselines.comrecgc import failed_selection_adoption as adoption

    original = adoption._derive_authority_paths
    displaced = case.control / "displaced-namespace"
    replaced = False

    def replacing_paths(profile):
        nonlocal replaced
        paths = original(profile)
        if not replaced and case.output.exists():
            case.namespace.rename(displaced)
            shutil.copytree(displaced, case.namespace)
            replaced = True
        return paths

    monkeypatch.setattr(adoption, "_derive_authority_paths", replacing_paths)
    with pytest.raises(FailedSelectionAdoptionError, match="namespace inode changed"):
        _adopt(case)
    assert replaced is True
    assert not (case.output / READY_NAME).exists()


def test_system_exit_during_full_preterminal_reopen_never_publishes_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_fixture(tmp_path)
    from src.baselines.comrecgc import failed_selection_adoption as adoption

    original = adoption._validate_with_profile

    def crash(**kwargs):
        if kwargs["receipt_name"] == adoption.PRETERMINAL_NAME:
            raise SystemExit(99)
        return original(**kwargs)

    monkeypatch.setattr(adoption, "_validate_with_profile", crash)
    with pytest.raises(SystemExit, match="99"):
        _adopt(case)
    assert not (case.output / READY_NAME).exists()
    assert not (case.output / "PASS").exists()


def test_terminal_lock_replacement_revokes_only_receipt_bound_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_fixture(tmp_path)
    _adopt(case)
    from src.baselines.comrecgc import failed_selection_adoption as adoption

    original = adoption._inspect_authority
    lock = case.output.parent / f".{case.output.name}.failed-selection-adoption.lock"
    replaced = False

    def replace_lock(**kwargs):
        nonlocal replaced
        result = original(**kwargs)
        if not replaced:
            lock.unlink()
            lock.write_bytes(b"hostile replacement lock")
            replaced = True
        return result

    monkeypatch.setattr(adoption, "_inspect_authority", replace_lock)
    with pytest.raises(FailedSelectionAdoptionError, match="lock inode changed"):
        _adopt(case)
    assert replaced is True
    assert not (case.output / READY_NAME).exists()
    assert lock.read_bytes() == b"hostile replacement lock"


def test_same_lock_inode_metadata_drift_revokes_receipt_bound_ready(
    tmp_path: Path,
) -> None:
    case = _build_fixture(tmp_path)
    _adopt(case)
    lock = case.output.parent / f".{case.output.name}.failed-selection-adoption.lock"
    inode = lock.stat().st_ino
    lock.write_bytes(b"same inode, changed metadata")
    assert lock.stat().st_ino == inode
    with pytest.raises(FailedSelectionAdoptionError, match="lock identity changed"):
        _adopt(case)
    assert not (case.output / READY_NAME).exists()


def test_terminal_output_rename_revokes_displaced_marker_not_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_fixture(tmp_path)
    _adopt(case)
    from src.baselines.comrecgc import failed_selection_adoption as adoption

    original = adoption._inspect_authority
    displaced = case.output.parent / "displaced-terminal-output"
    replacement_marker = b"replacement ready must survive\n"
    replaced = False

    def replace_output(**kwargs):
        nonlocal replaced
        result = original(**kwargs)
        if not replaced:
            case.output.rename(displaced)
            case.output.mkdir()
            (case.output / READY_NAME).write_bytes(replacement_marker)
            replaced = True
        return result

    monkeypatch.setattr(adoption, "_inspect_authority", replace_output)
    with pytest.raises(FailedSelectionAdoptionError, match="output.*inode changed"):
        _adopt(case)
    assert not (displaced / READY_NAME).exists()
    assert (case.output / READY_NAME).read_bytes() == replacement_marker


def test_terminal_reopen_rehashes_sources_and_binds_output_inode(
    tmp_path: Path,
) -> None:
    changed = _build_fixture(tmp_path / "changed")
    _adopt(changed)
    changed.pair_contract.write_text(changed.pair_contract.read_text() + " ")
    with pytest.raises(FailedSelectionAdoptionError, match="SHA256 mismatch"):
        _adopt(changed)
    assert not (changed.output / READY_NAME).exists()

    moved = _build_fixture(tmp_path / "moved")
    _adopt(moved)
    old = moved.output.parent / "old-output"
    moved.output.rename(old)
    moved.output.mkdir()
    for source in old.iterdir():
        shutil.copy2(source, moved.output / source.name)
    with pytest.raises(FailedSelectionAdoptionError, match="output inode"):
        _adopt(moved)
    assert (moved.output / READY_NAME).exists()
    assert (old / READY_NAME).exists()


def test_replaced_ready_inode_is_never_deleted_on_terminal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_fixture(tmp_path)
    _adopt(case)
    from src.baselines.comrecgc import failed_selection_adoption as adoption

    original = adoption._validate_with_profile
    replacement = b"hostile replacement marker\n"

    def replace_marker(**kwargs):
        marker = case.output / READY_NAME
        marker.unlink()
        marker.write_bytes(replacement)
        raise FailedSelectionAdoptionError("forced terminal failure")

    monkeypatch.setattr(adoption, "_validate_with_profile", replace_marker)
    with pytest.raises(FailedSelectionAdoptionError, match="cannot safely revoke"):
        _adopt(case)
    assert (case.output / READY_NAME).read_bytes() == replacement
    monkeypatch.setattr(adoption, "_validate_with_profile", original)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.update({"forged_scientific_pass": True}),
        lambda payload: payload["instances"]["main"].update(
            {"forged_scientific_route": "attacker"}
        ),
        lambda payload: payload["instances"]["main"]["worker_identity"].update(
            {"forged_generation": 99}
        ),
        lambda payload: payload["instances"]["main"][
            "launcher_identity"
        ].update({"forged_generation": 99}),
    ],
)
def test_task_state_exact_key_sets_reject_unknown_fields(
    tmp_path: Path,
    mutator,
) -> None:
    case = _build_fixture(tmp_path)
    payload = json.loads(case.close_state.read_text())
    mutator(payload)
    _json(case.close_state, payload)
    with pytest.raises(FailedSelectionAdoptionError, match="schema changed"):
        _adopt(case)
    assert not (case.output / READY_NAME).exists()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("updated_at",), None),
        (("updated_at",), {"forged": True}),
        (("updated_at",), "not-a-timestamp"),
        (("instances", "main", "heartbeat_at"), None),
        (("instances", "main", "heartbeat_at"), {"forged": True}),
        (("instances", "main", "heartbeat_at"), "2026-08-25T00:00:00"),
    ],
)
def test_only_two_well_typed_utc_state_values_may_drift(
    tmp_path: Path,
    path: tuple[str, ...],
    value,
) -> None:
    case = _build_fixture(tmp_path)
    payload = json.loads(case.final_state.read_text())
    target = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    _json(case.final_state, payload)
    with pytest.raises(FailedSelectionAdoptionError, match="UTC timestamps"):
        _adopt(case)
    assert not (case.output / READY_NAME).exists()


def test_created_at_is_static_scientific_projection_authority(tmp_path: Path) -> None:
    case = _build_fixture(tmp_path)
    payload = json.loads(case.final_state.read_text())
    payload["created_at"] = "2026-08-24T19:14:16+00:00"
    _json(case.final_state, payload)
    with pytest.raises(FailedSelectionAdoptionError, match="projection changed"):
        _adopt(case)
    assert not (case.output / READY_NAME).exists()


@pytest.mark.parametrize(
    "mutator",
    [
        lambda receipt: receipt.update({"PASS": True}),
        lambda receipt: receipt["failed_selection"].update(
            {"forged_partition_pass": True}
        ),
        lambda receipt: receipt["lock"]["identity"].update(
            {"forged_inode": 1}
        ),
        lambda receipt: receipt["task_state_observations"]["close"][0].update(
            {"forged_observation": True}
        ),
    ],
)
def test_resigned_terminal_receipt_rejects_every_extra_nested_key(
    tmp_path: Path,
    mutator,
) -> None:
    case = _build_fixture(tmp_path)
    _adopt(case)
    receipt = json.loads((case.output / RECEIPT_NAME).read_text())
    mutator(receipt)
    _rewrite_signed_terminal(case, receipt)
    with pytest.raises(FailedSelectionAdoptionError, match="schema changed"):
        _adopt(case)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda process: process["expected_worker_identity"].update(
            {"pid": 54321}
        ),
        lambda process: process.update({"recorded_child_pid": 54322}),
        lambda process: process.update({"old_science_worker_exited": False}),
        lambda process: process.update({"signals_sent": ["SIGTERM"]}),
        lambda process: process.update(
            {"worker_observation": "ORIGINAL_GENERATION_STILL_LIVE"}
        ),
        lambda process: process.update(
            {"child_observation": "RECORDED_CHILD_PID_REUSED"}
        ),
        lambda process: process.update(
            {"child_observation": "NO_RECORDED_CHILD_PID"}
        ),
    ],
)
def test_rebound_terminal_receipt_rejects_process_exit_tampering(
    tmp_path: Path,
    mutator,
) -> None:
    case = _build_fixture(tmp_path)
    _adopt(case)
    receipt = json.loads((case.output / RECEIPT_NAME).read_text())
    mutator(receipt["process_exit"])
    _rewrite_signed_terminal(case, receipt)
    with pytest.raises(FailedSelectionAdoptionError):
        _adopt(case)
    assert not (case.output / READY_NAME).exists()


@pytest.mark.parametrize(
    "authority_kind",
    [
        "controller_manifest",
        "close_gate",
        "final_gate",
        "close_state",
        "final_state",
        "proc_worker_generation",
        "source_artifact",
        "source_directory",
        "failed_tree",
    ],
)
def test_post_ready_full_reopen_revokes_ready_for_drift_after_second_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority_kind: str,
) -> None:
    case = _build_fixture(tmp_path)
    from src.baselines.comrecgc import failed_selection_adoption as adoption

    original = adoption._validate_with_profile
    calls = 0
    ready_seen = False

    def drift_after_second_scan() -> None:
        if authority_kind == "controller_manifest":
            case.source_manifest.write_text(case.source_manifest.read_text() + " ")
        elif authority_kind == "close_gate":
            case.close_gate.write_text(case.close_gate.read_text() + " ")
        elif authority_kind == "final_gate":
            case.final_gate.write_text(case.final_gate.read_text() + " ")
        elif authority_kind in {"close_state", "final_state"}:
            path = (
                case.close_state
                if authority_kind == "close_state"
                else case.final_state
            )
            payload = json.loads(path.read_text())
            payload["dataset"] = "hostile-drift"
            _json(path, payload)
        elif authority_kind == "proc_worker_generation":
            _proc_entry(case.proc, 12345, start_ticks=77)
        elif authority_kind == "source_artifact":
            case.pair_contract.write_text(case.pair_contract.read_text() + " ")
        elif authority_kind == "source_directory":
            source = case.pair_contract.parent
            displaced = source.parent / f"{source.name}-displaced"
            source.rename(displaced)
            shutil.copytree(displaced, source)
        elif authority_kind == "failed_tree":
            _json(
                case.final_root
                / "common_recourse/external_memory/dbscan/cluster_partition.json",
                {"status": "PASS", "dbscan_partition_proven": True},
            )
        else:  # pragma: no cover - parameter list is closed above.
            raise AssertionError(authority_kind)

    def injecting_validation(**kwargs):
        nonlocal calls, ready_seen
        calls += 1
        if kwargs["require_ready"]:
            ready_seen = (case.output / READY_NAME).exists()
            drift_after_second_scan()
        return original(**kwargs)

    monkeypatch.setattr(adoption, "_validate_with_profile", injecting_validation)
    with pytest.raises(FailedSelectionAdoptionError):
        _adopt(case)
    assert calls == 2
    assert ready_seen is True
    assert not (case.output / READY_NAME).exists()
    assert (case.output / READY_PREPARED_NAME).exists()


def test_post_ready_full_reopen_rehashes_every_source_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.baselines.comrecgc import failed_selection_adoption as adoption

    probe = _build_fixture(tmp_path / "probe")
    probe_evidence = adoption._inspect_authority(
        profile=probe.profile,
        proc_root=probe.proc,
    )
    artifact_count = len(probe_evidence["source_artifacts"])
    assert artifact_count > 0

    for index in range(artifact_count):
        case = _build_fixture(tmp_path / f"artifact-{index}")
        evidence = adoption._inspect_authority(
            profile=case.profile,
            proc_root=case.proc,
        )
        target = Path(evidence["source_artifacts"][index]["path"])
        original = adoption._validate_with_profile
        mutated = False

        def mutate_one_artifact(**kwargs):
            nonlocal mutated
            if kwargs["require_ready"] and not mutated:
                assert (case.output / READY_NAME).exists()
                target.write_bytes(target.read_bytes() + b"\npost-ready-drift")
                mutated = True
            return original(**kwargs)

        with monkeypatch.context() as patcher:
            patcher.setattr(
                adoption,
                "_validate_with_profile",
                mutate_one_artifact,
            )
            with pytest.raises(FailedSelectionAdoptionError):
                _adopt(case)
        assert mutated is True, index
        assert not (case.output / READY_NAME).exists(), index


def test_post_ready_full_reopen_rebinds_every_source_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.baselines.comrecgc import failed_selection_adoption as adoption

    probe = _build_fixture(tmp_path / "probe")
    probe_evidence = adoption._inspect_authority(
        profile=probe.profile,
        proc_root=probe.proc,
    )
    directory_count = len(probe_evidence["source_directory_authority"])
    assert directory_count > 0

    for index in range(directory_count):
        case = _build_fixture(tmp_path / f"directory-{index}")
        evidence = adoption._inspect_authority(
            profile=case.profile,
            proc_root=case.proc,
        )
        source = Path(evidence["source_directory_authority"][index]["path"])
        displaced = source.parent / f".{source.name}.post-ready-displaced"
        original = adoption._validate_with_profile
        mutated = False

        def mutate_one_directory(**kwargs):
            nonlocal mutated
            if kwargs["require_ready"] and not mutated:
                assert (case.output / READY_NAME).exists()
                source.rename(displaced)
                shutil.copytree(displaced, source)
                mutated = True
            return original(**kwargs)

        with monkeypatch.context() as patcher:
            patcher.setattr(
                adoption,
                "_validate_with_profile",
                mutate_one_directory,
            )
            with pytest.raises(FailedSelectionAdoptionError):
                _adopt(case)
        assert mutated is True, index
        assert not (case.output / READY_NAME).exists(), index


def test_extra_failed_tree_file_injected_after_second_hash_walk_blocks_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_fixture(tmp_path)
    from src.baselines.comrecgc import failed_selection_adoption as adoption

    original = adoption._validate_failed_tree_inventory
    calls = 0

    def inject_after_second_walk(**kwargs):
        nonlocal calls
        calls += 1
        result = original(**kwargs)
        if calls == 2:
            _json(
                case.final_root
                / "common_recourse/external_memory/dbscan/cluster_partition.json",
                {"status": "PASS", "dbscan_partition_proven": True},
            )
        return result

    monkeypatch.setattr(
        adoption,
        "_validate_failed_tree_inventory",
        inject_after_second_walk,
    )
    with pytest.raises(FailedSelectionAdoptionError, match="inventory changed"):
        _adopt(case)
    assert calls == 2
    assert not (case.output / READY_NAME).exists()


def test_pass_injected_after_second_full_inspection_blocks_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_fixture(tmp_path)
    from src.baselines.comrecgc import failed_selection_adoption as adoption

    original = adoption._inspect_authority
    calls = 0

    def inject_after_second_inspection(**kwargs):
        nonlocal calls
        calls += 1
        result = original(**kwargs)
        if calls == 2:
            (case.final_root / "PASS").write_bytes(b"PASS\n")
        return result

    monkeypatch.setattr(adoption, "_inspect_authority", inject_after_second_inspection)
    with pytest.raises(FailedSelectionAdoptionError, match="inventory changed"):
        _adopt(case)
    assert calls == 2
    assert not (case.output / READY_NAME).exists()


def test_failed_root_rename_byte_copy_after_second_inspection_blocks_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_fixture(tmp_path)
    from src.baselines.comrecgc import failed_selection_adoption as adoption

    original = adoption._inspect_authority
    calls = 0
    displaced = case.final_root.parent / "displaced-final-attempt-0"

    def replace_failed_root(**kwargs):
        nonlocal calls
        calls += 1
        result = original(**kwargs)
        if calls == 2:
            case.final_root.rename(displaced)
            shutil.copytree(displaced, case.final_root)
        return result

    monkeypatch.setattr(adoption, "_inspect_authority", replace_failed_root)
    with pytest.raises(FailedSelectionAdoptionError, match="root inode changed"):
        _adopt(case)
    assert calls == 2
    assert not (case.output / READY_NAME).exists()


def test_output_parent_rename_recreate_before_lock_open_cannot_redirect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_fixture(tmp_path)
    from src.baselines.comrecgc import failed_selection_adoption as adoption

    original = adoption._discover_source_locations
    displaced = case.output.parent.parent / "displaced-output-parent"
    replacement = case.output.parent
    replaced = False

    def replace_parent(profile):
        nonlocal replaced
        result = original(profile)
        if not replaced:
            replacement.rename(displaced)
            replacement.mkdir()
            replaced = True
        return result

    monkeypatch.setattr(adoption, "_discover_source_locations", replace_parent)
    with pytest.raises(FailedSelectionAdoptionError, match="output parent inode"):
        _adopt(case)
    assert replaced is True
    assert not case.output.exists()
    assert not (
        replacement / f".{case.output.name}.failed-selection-adoption.lock"
    ).exists()
    assert not (displaced / READY_NAME).exists()


def test_paired_slurm_is_static_autodl_only_cli_parity() -> None:
    root = Path(__file__).resolve().parents[3]
    wrapper = root / "scripts/slurm/adopt_aids_c766_failed_selection.sh"
    text = wrapper.read_text()
    assert "#SBATCH --partition=A800" in text
    assert "#SBATCH --gres=gpu:a800:1" in text
    assert "#SBATCH --output=logs/%j.out" in text
    assert "#SBATCH --error=logs/%j.err" in text
    assert "conda activate smiles_pip118" in text
    assert "cd /share/home/u20526/czx/counterfactual-subgraph" in text
    assert "export PYTHONPATH=$PWD" in text
    assert "--config configs/hpc.yaml" in text
    assert (
        "outputs/autodl/recovery_evidence/aids_c766_failed_selection_v1/"
        "FRESH_CHILD_REQUIRED"
    ) in text
    assert text.index("exit 78") < text.index(
        "python scripts/autodl/adopt_aids_c766_failed_selection.py"
    )
